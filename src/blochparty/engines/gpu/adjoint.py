"""The GPU engines' hand-written adjoint (``_GPUSimulate``), shared by both backends: the forward stores the
state before every ``checkpoint_every`` segments; the backward replays each interval (``bloch_replay``) and
sweeps it in reverse (``bloch_adjoint``), per-segment gradients summed in order (``seg_reduce``), so the result
is the same for any interval and every run."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from ... import exchange
from ...phantom import Phantom
from ...result import SimResult
from ...sequence import Sequence
from ..prepare import prepare
from .driver import ADJOINT_COILS, _forward


def default_checkpoint_every(n_segments: int) -> int:
    """About sqrt(segments), a multiple of 64 (the adjoint's per-spin gradients are flushed every 64
    segments; with intervals of whole flushes the result does not depend on the interval)."""
    return 64 * max(1, math.ceil(math.sqrt(max(n_segments, 1)) / 64))


class _Opts(NamedTuple):
    is_rf: list
    adc: list
    spoil: list
    every: int
    mode: int
    mma: bool | None
    lineshape: str
    cutoff: float | None
    memory: float
    backend: object

N_SPIN_GRADS = 20  # rows of the adjoint's per-spin gradients: pos (3), db0, pd, r1, r2, b1, conc (7), bound (5)


class _GPUSimulate(torch.autograd.Function):
    """The forward kernels with checkpoints; backward: bloch_adjoint, one dispatch per interval."""

    @staticmethod
    def forward(ctx, opts: _Opts, pos, pd, t1, t2, db0, coils_ri, dt, rf, grad, adc_phase, m0, b1, b0_offset, conc,
                bound, rf_freq):
        ctx.set_materialize_grads(False)
        d = lambda t: None if t is None else t.detach()
        ph = Phantom(d(pos), d(pd), d(t1), d(t2), d(db0), b1=d(b1),
                     bound=None if bound is None else exchange.BoundPool(*d(bound).unbind(-1), lineshape=opts.lineshape,
                                                                         cutoff=opts.cutoff, memory=opts.memory))
        sq = Sequence(d(dt), d(rf), d(grad), d(adc_phase), opts.is_rf, opts.adc, opts.spoil, {}, d(b0_offset), d(rf_freq))
        coils = torch.complex(coils_ri[..., 0], coils_ri[..., 1]).detach()
        from ...fields import SpinFields

        res, cx = _forward(ph, sq, coils, d(m0), opts.every, 256, None, opts.mma, 0, None, True,
                           SpinFields(d(pos), d(conc), opts.mode), keep=True, backend=opts.backend)
        ctx.opts, ctx.cx = opts, cx
        ctx.like = [None if t is None else (t.dtype, t.device) for t in
                    (pos, pd, t1, t2, db0, coils_ri, dt, rf, grad, adc_phase, m0, b1, b0_offset, conc, bound, rf_freq)]
        sig = torch.view_as_real(res.signal.T.contiguous())  # (n_adc, C, 2)
        m = res.m if res.m_bound is None else torch.cat([res.m, res.m_bound[:, None]], 1)
        ctx.save_for_backward(t1, t2, adc_phase, sig, bound, b1, conc)
        return sig, m

    @staticmethod
    @once_differentiable
    def backward(ctx, g_sig, g_m):
        opts, cx = ctx.opts, ctx.cx
        t1, t2, adc_phase, sig, bound, b1, conc = ctx.saved_tensors
        # pos, pd, t1, t2, db0, coils, dt, rf, grad, adc_phase, m0, b1, b0_offset, conc, bound, rf_freq
        need = list(ctx.needs_input_grad[1:])
        if g_sig is None:
            need[5] = need[9] = False
        if (g_sig is None and g_m is None) or not any(need):
            return (None,) * 17
        dev, f32 = opts.backend.device, torch.float32
        n, nc, S, every = t1.shape[0], cx["n_coils"], len(opts.is_rf), opts.every
        adc_idx = torch.tensor(opts.adc).nonzero()[:, 0]
        h = torch.zeros(max(cx["n_adc"], 1), nc, 2, dtype=f32)
        if g_sig is not None and cx["n_adc"]:  # dL/d(raw coil sums): dL/dsignal rotated by exp(+i psi)
            psi = adc_phase.detach().double().cpu()[adc_idx][:, None]
            gr, gi = g_sig[..., 0].double().cpu(), g_sig[..., 1].double().cpu()
            h = torch.stack([gr * torch.cos(psi) - gi * torch.sin(psi), gr * torch.sin(psi) + gi * torch.cos(psi)], -1).float()
        lam = torch.zeros(4, n, dtype=f32)
        if g_m is not None:
            lam[: g_m.shape[1]] = g_m.detach().float().cpu().T
        need_seg = 2 if any(need[i] for i in (6, 8, 12)) else (1 if need[7] or need[15] else 0)
        need_sens = need[5]
        lam = lam.to(dev).contiguous()
        GS = torch.zeros(N_SPIN_GRADS, n, device=dev, dtype=f32)
        GC = torch.zeros(2 * max(nc, 16) if need_sens else 1, n, device=dev, dtype=f32)
        threads, sgs = cx["threads"], cx["threads"] // 32
        R = torch.zeros(every // 4 * 4 * n, device=dev, dtype=f32)  # RBLOCK = 4, rrows <= 4
        GSEG = torch.zeros(sgs * every * 8 if need_seg else 1, device=dev, dtype=f32)
        # coil gradients in registers (padded with zero coils) only when they are wanted: the arrays cost occupancy
        ncr = next((c for c in ADJOINT_COILS[1:] if c >= nc), 0) if need_sens else 0
        sens = cx["sens"]
        if ncr > nc:
            sens = torch.cat([sens, torch.zeros(2 * (ncr - nc), n, device=dev, dtype=f32)]).contiguous()
            h = torch.cat([h, torch.zeros(h.shape[0], ncr - nc, 2, dtype=f32, device=h.device)], 1)
        SEG = torch.zeros(S * 8 if need_seg else 1, device=dev, dtype=f32)
        lib = opts.backend.lib()
        h = h.to(dev).contiguous()
        kernel = getattr(lib, f"bloch_adjoint{cx['suffix']}_c{ncr}")
        replay = getattr(lib, "bloch_replay" + cx["suffix"])
        n_int = len(cx["ckpts"])
        u32 = lambda rows: torch.from_numpy(np.array(rows, dtype=np.uint32).view(np.int32)).to(dev)
        prms = u32([[n, nc, c * every, min(S, (c + 1) * every), cx["mt"], cx["mode"], need_seg, need_sens,
                     g_sig is not None, every] for c in range(n_int)])
        reds = u32([[sgs, every, c * every, min(S, (c + 1) * every) - c * every] for c in range(n_int)])
        for c in reversed(range(n_int)):
            s0, s1 = c * every, min(S, (c + 1) * every)
            prm = prms[c]
            replay(cx["ckpts"][c], cx["spins"], cx["segs"], cx["flags"], cx["pool"], cx["frf"], cx["line"], cx["log_f"],
                   cx["conc"], R, prm, threads=threads, group_size=256)
            kernel(cx["ckpts"][c], cx["spins"], sens, cx["segs"], cx["flags"], cx["rows"], h, cx["pool"], cx["frf"],
                   cx["line"], cx["log_f"], cx["conc"], R, lam, GS, GC, GSEG, prm, threads=threads, group_size=256)
            if need_seg:
                lib.seg_reduce(GSEG, SEG, reds[c], threads=8 * (s1 - s0))
        opts.backend.synchronize()
        GS, lam = GS.cpu().double(), lam.cpu().double()
        out = [None] * 16
        if need[0]:
            out[0] = GS[0:3].T.contiguous()
        if need[1]:
            out[1] = GS[4]
        if need[2]:
            out[2] = -GS[5] / t1.detach().double().cpu() ** 2  # r1 = 1 / t1
        if need[3]:
            out[3] = -GS[6] / t2.detach().double().cpu() ** 2
        if need[4]:
            out[4] = GS[3]
        if need_sens:
            gc = GC.cpu().double()[: 2 * nc]
            out[5] = torch.stack([gc[0::2].T, gc[1::2].T], -1)
        if need_seg:
            seg = SEG.cpu().double().view(S, 8)
            out[6], out[7], out[8], out[12] = seg[:, 0], seg[:, 1:3], seg[:, 3:6], seg[:, 6]
            out[15] = seg[:, 7]
        if need[9]:  # signal = raw exp(-i psi): dsignal/dpsi = -i signal
            g = torch.zeros(S, dtype=torch.float64)
            sg, gs = sig.double(), g_sig.double().cpu()
            g[adc_idx] = (gs[..., 0] * sg[..., 1] - gs[..., 1] * sg[..., 0]).sum(-1)
            out[9] = g
        if need[10]:
            out[10] = lam[: 4 if bound is not None else 3].T.contiguous()
        if need[11]:
            out[11] = GS[7]
        if need[13]:
            out[13] = GS[8:8 + conc.shape[1]].T.contiguous()
        if need[14]:  # f, k, t1b (r1b = 1 / t1b), t2b, dw
            t1b = bound.detach().double().cpu()[:, 2]
            out[14] = torch.stack([GS[15], GS[16], -GS[17] / t1b**2, GS[18], GS[19]], -1)
        return (None, *[None if g is None or not need[i] else g.to(dtype=ctx.like[i][0], device=ctx.like[i][1]) for i, g in enumerate(out)])


def _simulate_grad(phantom: Phantom, seq: Sequence, coils, m_init, scanner, maps, mma, every, backend) -> SimResult:
    """simulate_metal with gradients: the inputs as tensors of an autograd Function (_GPUSimulate). What it
    differentiates: capabilities.SUPPORTED (checked by the caller)."""
    n, pd = len(phantom), phantom.pd
    prep = prepare(phantom, seq, coils, m_init, scanner, maps)
    fields = prep.fields
    db0, b1 = phantom.db0, phantom.b1
    if maps is not None:  # static spins: the maps sampled at the true positions, differentiably
        from ...maps import sample

        v = sample(maps.channels(pd.dtype), torch.as_tensor(maps.origin, dtype=pd.dtype),
                   torch.as_tensor(maps.spacing, dtype=pd.dtype), phantom.pos)
        db0 = db0 + v[:, 0]
        b1 = v[:, 1] if b1 is None else b1 * v[:, 1]
        if maps.coils is not None:
            coils = torch.complex(v[:, 2::2], v[:, 3::2])
    if coils is None:
        coils = torch.ones(n, 1, dtype=torch.complex64 if pd.dtype == torch.float32 else torch.complex128)
    bound, rf_freq, lineshape, cutoff, memory = None, None, "superlorentzian", None, 300.0
    if phantom.bound is not None:
        pool = prep.bound
        bound = torch.stack([pool.f, pool.k, pool.t1, pool.t2, pool.dw], -1)
        rf_freq = exchange.rf_frequency(seq)
        lineshape, memory = pool.lineshape, float(pool.memory)
        cutoff = None if pool.cutoff is None else float(pool.cutoff)
    mx, my, mz, mb = prep.m0  # mb: the bound pool's Mz (default f PD), None without one
    m0 = torch.stack([mx, my, mz] + ([] if mb is None else [mb]), -1)
    every = every or default_checkpoint_every(len(seq))
    if every % 64:
        raise ValueError(f"checkpoint_every must be a multiple of 64, got {every}")
    opts = _Opts(list(seq.is_rf), list(seq.adc), list(seq.spoil), every, fields.mode, mma, lineshape, cutoff, memory,
                 backend)
    sig, m = _GPUSimulate.apply(opts, fields.pos, pd, phantom.t1, phantom.t2, db0, torch.view_as_real(coils),
                                  seq.dt, seq.rf, seq.grad, seq.adc_phase, m0, b1, seq.b0_offset, fields.conc, bound,
                                  rf_freq)
    signal = torch.view_as_complex(sig.contiguous()).T
    if bound is not None:
        return SimResult(signal, m[:, :3], m[:, 3])
    return SimResult(signal, m)
