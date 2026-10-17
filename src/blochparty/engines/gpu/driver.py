"""The GPU engines' Python driver, shared by the Metal and CUDA backends: phantom, coils, maps and sequence
into float32 buffers, spins sorted by cost, dispatch in chunks of segments, NUFFT readouts and reused pulses
(``_forward``). The backend (``engines.gpu`` docstring) supplies the device, the kernels and synchronisation;
the kernels of both take the same buffers in the same order."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch
from torch import Tensor

from ... import exchange
from ...fields import spin_fields
from ...phantom import Phantom
from ...result import SimResult
from ...sequence import Sequence
from ..prepare import prepare

# limits and choices of the kernels, the same on both backends
MAX_COILS = 32  # the direct (simd_sum) reduction; the matrix-product one has no limit
MAX_POOLS = 5  # free exchanging pools (the shader's MAX_POOLS; one kernel per count)
ADJOINT_COILS = (0, 1, 2, 4, 8, 16)  # adjoint kernels with the coils in registers (padded up); 0: any, from memory
MMA_MIN_COILS = 2


def _field_model_buffer(scanner) -> Tensor:
    """The shader's fm buffer (nl_displacement, conc_at): nonlinearity flag, L, R0, f0, z asymmetry, then
    A and B (3, L, L) with R0 N_nm folded in and zero for m > n (fields.GradientNonlinearity)."""
    head = [0.0, 1.0, 1.0, 1.0, float(scanner.z_asymmetry)]
    if scanner.field_strength is not None and scanner.concomitant is not None:
        head[3] = float(scanner.larmor)
    coef = torch.zeros(0, dtype=torch.float64)
    nl = scanner.nonlinearity
    if nl is not None:
        size = nl.n_max + 1
        norm = torch.zeros(size, size, dtype=torch.float64)
        for nn in range(size):
            for m in range(nn + 1):
                norm[nn, m] = 1.0 if m == 0 else math.sqrt((2 * nn + 1) * math.factorial(nn - m) / (2 * math.factorial(nn + m)))
        scale = nl.r0 * norm
        a, b = (t.detach().double().cpu() * scale for t in (nl.alpha, nl.beta))
        head[:3] = [1.0, float(size), float(nl.r0)]
        coef = torch.cat([a.reshape(-1), b.reshape(-1)])
    return torch.cat([torch.tensor(head, dtype=torch.float64), coef]).float().contiguous()


def _cost_order(phantom: Phantom, seq: Sequence) -> Tensor | None:
    """Static spins in order of their free-pool kernel cost: an RF segment takes sub-steps in proportion to
    the spin's off-resonance there, and the lanes of a simdgroup wait for its slowest spin (a CEST
    spectrum's M0 offset at -1500 ppm is ~80 sub-steps per 100 us sample, the other offsets ~2)."""
    rf = torch.tensor(seq.is_rf)
    if not bool(rf.any()):
        return None
    g = seq.grad.detach().double().cpu()[rf].abs().amax(0)
    off = 0.0 if seq.b0_offset is None else float(seq.b0_offset.detach().double()[rf].abs().max())
    cost = phantom.db0.detach().double().cpu().abs() + phantom.pos.detach().double().cpu().abs() @ g + off
    return torch.argsort(cost, stable=True)

NUFFT_MIN_SAMPLES = 16  # readouts shorter than this keep the per-sample path

NU_HEAD = 14  # = the shader's NU_HEAD: mg w terms beta n_adc wmax tau[8]

NUFFT_WMAX = 32  # widest threadgroup window of grid cells for the NUFFT path


def _readout_runs(seq: Sequence) -> list[tuple[int, int]]:
    """(first segment, samples) of every run of >= NUFFT_MIN_SAMPLES consecutive ADC segments without RF,
    with one dt, gradient and b0_offset to 1e-9 (a spoil only on the last), of the most common (samples, dt).
    A builder readout's first sample sits half a dwell in, so its run starts at the second sample."""
    adc, rf, sp = (np.asarray(f, dtype=bool) for f in (seq.adc, seq.is_rf, seq.spoil))
    dt = seq.dt.detach().double().cpu().numpy()
    g = seq.grad.detach().double().cpu().numpy()
    b0 = np.zeros_like(dt) if seq.b0_offset is None else seq.b0_offset.detach().double().cpu().numpy()
    ok = adc & ~rf
    same = np.zeros(len(dt), dtype=bool)
    # equal to the last bits: segment edges come from absolute times (dt 9.999999999999999e-06 vs 1e-05)
    tol_g = 1e-9 * max(float(np.abs(g).max()), 1.0)
    same[1:] = (ok[1:] & ok[:-1] & ~sp[:-1] & (np.abs(dt[1:] - dt[:-1]) <= 1e-9 * dt[1:])
                & (np.abs(g[1:] - g[:-1]) <= tol_g).all(1) & (np.abs(b0[1:] - b0[:-1]) <= 1e-9))
    starts = np.flatnonzero(ok & ~same)
    ends = np.append(starts[1:], len(dt))
    runs = []
    for a, e in zip(starts, ends):
        k = a + 1
        while k < e and same[k]:
            k += 1
        if k - a >= NUFFT_MIN_SAMPLES:
            runs.append((int(a), int(k - a)))
    if not runs:
        return []
    kinds = {}
    for a, k in runs:
        kinds.setdefault((k, round(float(dt[a]) * 1e12)), []).append((a, k))
    return max(kinds.values(), key=len)

PULSE_MEMORY = 2 << 30  # bytes of per-spin pulse maps at most (pulses="reuse")


def _reused_pulses(seq: Sequence, flags: Tensor, frf: Tensor | None, tol: float = 1e-6):
    """[(first segment, length, [(segment, phase), ...])] of every pulse that recurs: a run of >= 2
    consecutive RF segments without ADC (a spoil only on its last) with the dt, gradient, b0_offset, RF
    frequency and flags of an earlier run and its RF times e^{i phase} (to ``tol`` of the peak; phase by
    least squares). Most segments saved first. ``flags``: the kernel's (bound-only bits applied)."""
    f = flags[: len(seq)].numpy()
    S = len(seq)
    live = ((f & 1) != 0) & ((f & 2) == 0)
    runs, a = [], None
    for s in range(S):
        if live[s]:
            a = s if a is None else a
            if f[s] & 4:
                runs.append((a, s + 1 - a))
                a = None
        elif a is not None:
            runs.append((a, s - a))
            a = None
    if a is not None:
        runs.append((a, S - a))
    dt = seq.dt.detach().double().cpu().numpy()
    rf = seq.rf.detach().double().cpu().numpy()
    z = rf[:, 0] + 1j * rf[:, 1]
    g = seq.grad.detach().double().cpu().numpy()
    b0 = np.zeros(S) if seq.b0_offset is None else seq.b0_offset.detach().double().cpu().numpy()
    fr = np.zeros(S) if frf is None else frf[:S].detach().cpu().double().numpy()
    close = lambda x, y: np.abs(x - y).max() <= tol * max(np.abs(x).max(), np.abs(y).max(), 1e-300)
    groups, bucket = [], {}
    for a, k in runs:
        if k < 2:
            continue
        i, key = slice(a, a + k), (k, round(float(dt[a:a + k].sum()) * 1e9))
        for gi in bucket.get(key, ()):
            r = groups[gi][0]
            j = slice(r, r + k)
            if not (np.array_equal(f[i], f[j]) and close(dt[i], dt[j]) and close(g[i], g[j]) and close(b0[i], b0[j])
                    and close(fr[i], fr[j])):
                continue
            c = np.vdot(z[j], z[i])
            ph = float(np.angle(c)) if c != 0 else 0.0
            if close(z[i], z[j] * np.exp(1j * ph)):
                groups[gi][2].append((a, ph))
                break
        else:
            bucket.setdefault(key, []).append(len(groups))
            groups.append((a, k, [(a, 0.0)]))
    groups = [p for p in groups if len(p[2]) > 1]
    return sorted(groups, key=lambda p: -(len(p[2]) - 1) * p[1])


def _es_hat(kp: np.ndarray, ng: int, w: int, beta: float) -> np.ndarray:
    """Fourier transform of the exponential-of-semicircle kernel exp(beta (sqrt(1 - (2t/w)^2) - 1)), t in grid
    cells, at modes kp of an ng-cell grid (Barnett, Magland, af Klinteberg 2019); trapezoid quadrature."""
    t = np.linspace(-w / 2, w / 2, 400 * w + 1)
    z = 2 * t / w
    v = np.where(np.abs(z) < 1, np.exp(beta * (np.sqrt(np.clip(1 - z * z, 0, None)) - 1)), 0.0)
    return np.trapezoid(v[None, :] * np.cos(2 * np.pi * np.outer(kp, t) / ng), t, axis=1)


def _take(phantom: Phantom, idx: Tensor) -> Phantom:
    """The (static) spins ``idx`` of a phantom with free pools."""
    n, pd = len(phantom), phantom.pd
    pick = lambda t: None if t is None else t[idx]
    pools = tuple(type(q)(*(getattr(q.expand(n, pd), a)[idx] for a in ("f", "k", "t1", "t2", "dw"))) for q in phantom.pools or ())
    bound = None if phantom.bound is None else phantom.bound.expand(n, pd)[idx]
    return dataclasses.replace(phantom, pos=phantom.pos[idx], pd=pd[idx], t1=phantom.t1[idx], t2=phantom.t2[idx],
                               db0=phantom.db0[idx], b1=pick(phantom.b1), bound=bound, pools=pools)


def _forward(phantom: Phantom, seq: Sequence, coils, m_init, chunk, group, scanner, mma, seed, maps,
             ordered: bool = False, fields_in=None, keep: bool = False, nufft=None, reuse: bool = False, *, backend):
    """The forward kernels. ``fields_in``: the field model's inputs (``fields.SpinFields``) in place of
    spin_fields(scanner, pos). ``keep``: also return what the adjoint needs, with the state before every
    ``chunk`` segments (one dispatch each). ``nufft``: (width, terms) of NUFFT readouts; ``reuse``: reused
    pulse maps (simulate_metal). ``backend``: the backend (``metal.METAL``, ``cuda.CUDA``)."""
    runs = []
    for opt, on in (("readouts='nufft'", nufft is not None), ("pulses='reuse'", reuse)):
        if not on:
            continue
        if keep:
            raise ValueError(f"{opt} is forward only")
        for what, present in (("diffusion", phantom.diffusion is not None), ("motion", phantom.motion is not None),
                              ("free pools", bool(getattr(phantom, "pools", None))),
                              ("the generalized bound pool", phantom.bound is not None and phantom.bound.model == "generalized"),
                              ("concomitant fields", bool(spin_fields(scanner, phantom.pos).mode))):
            if present:
                raise ValueError(f"{opt} is for static spins without {what}")
        mma = True
    if nufft is not None:
        runs = _readout_runs(seq)
        mma = True
        if runs and not ordered:  # sorted along the readout's phase: a threadgroup covers a few grid cells
            a = runs[0][0]
            off = 0.0 if seq.b0_offset is None else float(seq.b0_offset[a])
            pos_enc = spin_fields(scanner, phantom.pos).pos
            turns = float(seq.dt[a]) * (pos_enc.detach().double().cpu() @ seq.grad[a].detach().double().cpu()
                                        + phantom.db0.detach().double().cpu() + off)
            order = torch.argsort(turns - torch.floor(turns), stable=True)
            inv = torch.argsort(order)
            r = _forward(_take(phantom, order), seq, None if coils is None else coils[order],
                         None if m_init is None else m_init[order], chunk, group, scanner, mma, seed, maps, True,
                         nufft=nufft, reuse=reuse, backend=backend)
            return SimResult(r.signal, r.m[inv], None if r.m_bound is None else r.m_bound[inv])
    pools = tuple(getattr(phantom, "pools", None) or ())
    if pools:  # as the core (free pools with the generalized bound pool: capabilities.REFUSED)
        if len(pools) > MAX_POOLS:
            raise NotImplementedError(f"at most {MAX_POOLS} free pools on the GPU, got {len(pools)}")
        # moving spins keep their order: a walk's random numbers belong to the spin's index
        moving = phantom.diffusion is not None or phantom.motion is not None
        order = None if ordered or moving else _cost_order(phantom, seq)
        if order is not None:  # run the spins sorted by cost, return them in the caller's order
            inv = torch.argsort(order)
            r = _forward(_take(phantom, order), seq, None if coils is None else coils[order],
                         None if m_init is None else m_init[order], chunk, group, scanner, mma, seed, maps, True,
                         backend=backend)
            return SimResult(r.signal, r.m[inv], None if r.m_bound is None else r.m_bound[inv], r.m_pools[inv])
    dev, f32 = backend.device, torch.float32
    n = len(phantom)
    prep = prepare(phantom, seq, coils, m_init, scanner, maps)  # the checks and what every engine derives alike
    b1 = phantom.b1 if getattr(phantom, "b1", None) is not None else torch.ones(n)
    # encoding positions (nonlinearity), concomitant coefficients
    fields = fields_in if fields_in is not None else prep.fields
    moving = prep.moving
    db0 = phantom.db0
    mapped, map_bufs = 0, [torch.zeros(1, device=dev, dtype=f32), torch.zeros(9, device=dev, dtype=f32)]
    if maps is not None:  # scanner-frame fields: sampled once for static spins, per segment for moving ones
        from ...maps import sample

        grid = maps.channels(torch.float64)
        origin = torch.as_tensor(maps.origin, dtype=torch.float64)
        spacing = torch.as_tensor(maps.spacing, dtype=torch.float64)
        if not moving:  # at the true positions (the maps are scanner-frame fields)
            v = sample(grid, origin, spacing, phantom.pos.detach().double().cpu())
            db0 = phantom.db0 + v[:, 0].to(phantom.db0.dtype)
            b1 = b1.to(v.dtype) * v[:, 1]
            if maps.coils is not None:
                coils = torch.complex(v[:, 2::2], v[:, 3::2])
        else:
            mapped = 2 if maps.coils is not None else 1
            if maps.coils is not None:  # placeholders: the kernel samples the maps at every ADC sample
                coils = torch.zeros(n, maps.coils.shape[0], dtype=torch.complex64)
                mma = False
            nz, ny, nx = maps.shape()
            mg = torch.cat([origin, spacing, torch.tensor([nx, ny, nz], dtype=torch.float64)])
            map_bufs = [grid.reshape(-1).to(dev, f32).contiguous(), mg.to(dev, f32).contiguous()]
    spins = torch.stack([fields.pos[:, 0], fields.pos[:, 1], fields.pos[:, 2], phantom.pd, 1 / phantom.t1,
                         1 / phantom.t2, db0, b1.to(phantom.pd.dtype)]).to(dev, f32).contiguous()
    if coils is None:
        coils = torch.ones(n, 1, dtype=torch.complex64)
    n_coils = coils.shape[1]
    mma = n_coils >= MMA_MIN_COILS if mma is None else mma
    if not mma and n_coils > MAX_COILS:
        raise ValueError(f"at most {MAX_COILS} coils with mma=False")
    if backend.max_coils is not None and n_coils > backend.max_coils:
        raise ValueError(f"at most {backend.max_coils} coils on {backend.name}")
    if group != 256:
        raise ValueError("the Metal kernel's sample buffer is sized for threadgroups of 256")
    threads = math.ceil(n / group) * group
    if mma:  # (2, 8T, threads): real and imaginary planes, zero-padded to whole coil tiles and the grid
        n_tiles = math.ceil(n_coils / 8)
        sens = torch.zeros(2, 8 * n_tiles, threads, dtype=f32)
        sens[0, :n_coils, :n] = coils.real.T.float()
        sens[1, :n_coils, :n] = coils.imag.T.float()
        sens = sens.to(dev)
    else:
        sens = torch.stack([coils.real, coils.imag], -1).reshape(n, 2 * n_coils).T.to(dev, f32).contiguous()
    b0 = getattr(seq, "b0_offset", None)
    b0 = torch.zeros_like(seq.dt) if b0 is None else b0
    psi = torch.remainder(seq.adc_phase.double(), 2 * math.pi)  # RF spoiling makes phases large: wrap before float32
    segs = torch.stack([seq.dt.double(), seq.rf[:, 0].double(), seq.rf[:, 1].double(), seq.grad[:, 0].double(),
                        seq.grad[:, 1].double(), seq.grad[:, 2].double(), psi, b0.double()], 1).to(dev, f32).contiguous()
    adc = torch.tensor(seq.adc, dtype=torch.int32)
    flags = (torch.tensor(seq.is_rf, dtype=torch.int32) | (adc << 1) |
             (torch.tensor(seq.spoil, dtype=torch.int32) << 2))
    bound_only = getattr(seq, "bound_only", None)
    if bound_only is not None and any(bound_only):  # RF only the bound pool feels (Sequence.bound_only)
        bo = torch.tensor(bound_only, dtype=torch.int32)
        flags = flags | (bo << 4) if phantom.bound is not None else flags & ~bo  # no bound pool: free segments
    flags_cpu = flags
    flags = flags.to(dev)
    rows = (torch.cumsum(adc, 0) - 1).clamp_min(0).to(torch.int32).to(dev)
    n_adc = int(adc.sum())
    # the state as float32 hi + lo (L), split exactly from float64: the kernels carry lo (two_sum) across dispatches
    mx, my, mz, mb = prep.m0
    M64 = torch.stack([mx, my, mz]).detach().double().cpu()
    mb64 = torch.zeros(n, dtype=torch.float64)
    pool = prep.bound
    if pool is not None:  # magnetization transfer: the bound pool's Mz, parameters, RF frequency, lineshape
        mb64 = mb.detach().double().cpu()
        mt_bufs = [mb64.float().to(dev).contiguous(),
                   torch.stack([pool.f, pool.k, 1 / pool.t1, pool.t2, pool.dw]).detach().to(dev, f32).contiguous()]
        gen_bufs = None
        if pool.model == "generalized":  # modes instead of the saturation lineshape
            from ...generalized_bloch import modes as green_modes

            pm, cm = green_modes(pool.lineshape, pool.memory)
            table = torch.tensor([[q.real, q.imag, c.real, c.imag] for q, c in zip(pm, cm)], dtype=f32)
            t2b_min = float(pool.t2.detach().min())
            dts = seq.dt.detach().double().cpu()
            nsub = torch.tensor([max(1, math.ceil(float(d) / (0.5 * t2b_min))) if r else 1
                                 for d, r in zip(dts, seq.is_rf)], dtype=torch.int32)
            gen_bufs = [torch.zeros(2 * len(pm), n, device=dev, dtype=f32), table.to(dev), nsub.to(dev), len(pm)]
            dummy = torch.zeros(1, device=dev, dtype=f32)
            mt_bufs += [dummy, dummy, dummy, 2]
        else:
            frf = exchange.rf_frequency(seq.to(dtype=torch.float64, device="cpu")).detach()
            if pool.cutoff is None and pool.lineshape != "lorentzian":  # kind 3: the modes' sum of Lorentzians
                from ...generalized_bloch import modes as green_modes

                pm, cm = green_modes(pool.lineshape, pool.memory)
                log_f = torch.tensor([[q.real, q.imag, c.real, c.imag] for q, c in zip(pm, cm)], dtype=torch.float64)
                line = torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0, float(len(pm))], dtype=f32)
            else:
                lx, log_f = exchange._sl_table(torch.float64, torch.device("cpu"))
                kind = 1.0 if pool.lineshape == "lorentzian" else 2.0
                cut = 0.0 if pool.cutoff is None else float(pool.cutoff)
                line = torch.tensor([kind, cut, float(lx[0]), float(lx[1] - lx[0]), exchange._SL_XMIN,
                                     float(len(lx))], dtype=f32)
            mt_bufs += [frf.to(dev, f32).contiguous(), line.to(dev), log_f.reshape(-1).to(dev, f32).contiguous(), 1]
    else:
        dummy = torch.zeros(1, device=dev, dtype=f32)
        mt_bufs = [dummy, dummy, dummy, dummy, dummy, 0]
        gen_bufs = None
    M = M64.float()
    lo_buf = torch.cat([M64 - M.double(), (mb64 - mb64.float().double())[None]]).float().to(dev).contiguous()
    M = M.to(dev).contiguous()
    if fields.mode:
        conc = torch.zeros(7, n, dtype=torch.float64)
        conc[: fields.conc.shape[1]] = fields.conc.detach().double().cpu().T
        conc_buf = [conc.to(dev, f32).contiguous(), int(fields.mode)]
    else:
        conc_buf = [torch.zeros(1, device=dev, dtype=f32), 0]
    sig = torch.zeros(max(n_adc, 1) * n_coils * 2, device=dev, dtype=f32)
    if gen_bufs is None:
        dz = torch.zeros(1, device=dev, dtype=f32)
        gen_bufs = [dz, dz, torch.zeros(1, device=dev, dtype=torch.int32), 0]
    # moving spins under the field model move in true coordinates; the kernel encodes them per segment
    fmove = prep.field_model is not None
    fm_buf = torch.zeros(1, device=dev, dtype=f32)
    if fmove:
        fm_buf = _field_model_buffer(scanner).to(dev)
    r_start = (phantom.pos if fmove else fields.pos).detach().double().cpu()
    if phantom.diffusion is not None:  # walkers start at their positions (encoding ones without fmove); L L^T = D
        D = phantom.diffusion.detach().double().cpu()
        L = torch.diag_embed(torch.sqrt(D).unsqueeze(-1).expand(n, 3)) if D.dim() == 1 else torch.linalg.cholesky(D)
        chol = torch.stack([L[:, 0, 0], L[:, 1, 0], L[:, 1, 1], L[:, 2, 0], L[:, 2, 1], L[:, 2, 2]])
        start = r_start
        dzi = torch.zeros(1, device=dev, dtype=torch.int32)
        geom_bufs = [dzi, torch.zeros(1, device=dev, dtype=f32), 0]
        if phantom.restriction is not None:  # walkers start inside their compartments, as the core
            from ...restriction import KINDS, n_sub, start_offsets, tau_max

            if D.dim() != 1:
                raise ValueError("restricted diffusion needs an isotropic diffusivity (N,)")
            res = phantom.restriction.expand(n, start)
            centre = start
            start = centre + start_offsets(res, int(seed), np.arange(n, dtype=np.uint64), centre)
            tmax = tau_max(res, D)
            wsub = torch.tensor([n_sub(float(d), tmax) for d in seq.dt.detach().double().cpu()], dtype=torch.int32)
            ax = res.axis if res.axis is not None else torch.zeros(n, 3, dtype=torch.float64)
            geom = torch.cat([res.size[None], ax.T, centre.T]).to(dev, f32).contiguous()
            geom_bufs = [wsub.to(dev), geom, KINDS.index(res.kind) + 1]
        walk_bufs = [start.T.to(dev, f32).contiguous(), chol.to(dev, f32).contiguous(),
                     int(seed) & 0xFFFFFFFF, (int(seed) >> 32) & 0xFFFFFFFF, *geom_bufs]
    else:  # motion alone: the tissue-frame position stays r0
        dz = torch.zeros(1, device=dev, dtype=f32)
        walk_bufs = [r_start.T.to(dev, f32).contiguous() if phantom.motion is not None else dz, dz, 0, 0,
                     torch.zeros(1, device=dev, dtype=torch.int32), dz, 0]
    move_bufs = [torch.zeros(1, device=dev, dtype=f32), torch.zeros(1, device=dev, dtype=f32)]
    if phantom.motion is not None:
        mo = phantom.motion
        seg = mo.per_segment(seq.dt.detach().double().cpu())
        S_ = len(seq)
        w = seg["w"] if seg["w"] is not None else torch.zeros(S_, dtype=torch.float64)
        A = seg["A"] if seg["A"] is not None else torch.eye(3, dtype=torch.float64).expand(S_, 3, 3)
        b = seg["b"] if seg["b"] is not None else torch.zeros(S_, 3, dtype=torch.float64)
        mseg = torch.cat([seg["t"][:, None], w[:, None], A.reshape(S_, 9), b], 1)
        zero3 = torch.zeros(n, 3, dtype=torch.float64)
        vel = mo.velocity.detach().double().cpu() if mo.velocity is not None else zero3
        dsp = mo.displacement.detach().double().cpu() if mo.displacement is not None else zero3
        move_bufs = [mseg.to(dev, f32).contiguous(), torch.cat([vel.T, dsp.T]).to(dev, f32).contiguous()]
    dz = torch.zeros(1, device=dev, dtype=f32)
    bm_bufs = [dz, dz]
    if pools:  # state (2 nc, N): hi, lo of water, pools (x y z) and the bound pool's Mz (bm_load); split exactly
        pd64 = phantom.pd.detach().double().cpu()
        nc = 3 * (MAX_POOLS + 1) + 1
        init = torch.zeros(nc, n, dtype=torch.float64)
        if m_init is None:
            init[2] = pd64
        else:
            init[:3] = m_init[:, :3].detach().double().cpu().T
        params = torch.zeros(5 * MAX_POOLS, n, dtype=torch.float64)
        for k, q in enumerate(pools):
            q = q.expand(n, pd64)
            init[5 + 3 * k] = q.f.detach() * pd64
            params[5 * k: 5 * k + 5] = torch.stack([q.f, q.k, 1 / q.t1, 1 / q.t2, q.dw]).detach()
        if pool is not None:
            init[nc - 1] = mb.detach().double().cpu()
        hi = init.float()
        bm_bufs = [torch.cat([hi, (init - hi.double()).float()]).to(dev).contiguous(), params.to(dev, f32).contiguous()]
    suffix = "_gen" if gen_bufs[3] else ("_full" if (mt_bufs[5] or conc_buf[1]) else "")
    if pools:
        suffix = f"_bm{len(pools)}"
    if phantom.diffusion is not None or phantom.motion is not None:
        suffix += "_diff"
    lib, S = backend.lib(), len(seq)
    # buffers 7-19 (MB pool frf line log_f | conc | U modes nsub | P chol | wsub geom) and the scalars
    MB, pool_b, frf_b, line_b, logf_b, mt_flag = mt_bufs
    conc_b, mode_flag = conc_buf
    U_b, modes_b, nsub_b, nj_v = gen_bufs
    P_b, chol_b, seed_lo, seed_hi, wsub_b, geom_b, kind_v = walk_bufs
    bufs = (MB, pool_b, frf_b, line_b, logf_b, conc_b, U_b, modes_b, nsub_b, P_b, chol_b, wsub_b, geom_b, *move_bufs,
            *map_bufs)

    # NUFFT readouts: run starts, one merged free segment per run appended after the sequence (the jump),
    # the grid rows after the samples in sig
    run_of = torch.full((S + len(runs),), -1, dtype=torch.int32)
    nu = torch.zeros(NU_HEAD + 3 * len(runs) if runs else 1, dtype=torch.float64)
    ng = 0
    if runs:
        width, terms = nufft
        if not 1 <= terms <= 8:
            raise ValueError("nufft_terms must be 1..8")
        n_r, dt_r = runs[0][1], float(seq.dt[runs[0][0]])
        ng = 2 * n_r + (2 * n_r) % 2
        nodes = 0.5 * (dt_r + n_r * dt_r) + 0.5 * (n_r * dt_r - dt_r) * np.cos(np.pi * (2 * np.arange(terms) + 1) / (2 * terms))
        nu[:6] = torch.tensor([ng, width, terms, 2.30 * width, n_adc, NUFFT_WMAX], dtype=torch.float64)
        nu[6:6 + terms] = torch.from_numpy(nodes)
        merged = []
        for r, (a, k) in enumerate(runs):
            run_of[a] = r
            nu[NU_HEAD + 3 * r: NU_HEAD + 3 * r + 3] = torch.tensor([k, S + r, k // 2 + 1], dtype=torch.float64)
            row = segs[a].clone()
            row[0], row[1], row[2], row[6] = k * float(seq.dt[a]), 0.0, 0.0, 0.0
            merged.append(row)
        segs = torch.cat([segs, torch.stack(merged)]).contiguous()
        flags = torch.cat([flags, torch.zeros(len(runs), dtype=torch.int32, device=dev)]).contiguous()
        rows = torch.cat([rows, torch.zeros(len(runs), dtype=torch.int32, device=dev)]).contiguous()
        if pool is not None:
            MB, pool_b, frf_b, line_b, logf_b, mt_flag = mt_bufs
            frf_b = torch.cat([frf_b, torch.zeros(len(runs), device=dev, dtype=f32)]).contiguous()
            mt_bufs = [MB, pool_b, frf_b, line_b, logf_b, mt_flag]
            bufs = (MB, pool_b, frf_b, *bufs[3:])
        sig = torch.zeros((n_adc + len(runs) * terms * ng) * n_coils * 2, device=dev, dtype=f32)
    # reused pulses: maps per spin from each pulse's first occurrence; occurrences marked -2 - id in
    # run_of with their phase in the psi column; lengths after the NUFFT data in nu
    pulses, pbase = (_reused_pulses(seq, flags_cpu, frf_b if pool is not None else None) if reuse else []), 0
    if pulses:
        R = 4 if pool is not None else 3
        K = R * (R + 1)
        pulses = pulses[: max(0, PULSE_MEMORY // (4 * K * threads))]
    if pulses:
        pbase = len(nu)
        nu = torch.cat([nu, torch.tensor([k for _, k, _ in pulses], dtype=torch.float64)])
        idx, phase = [], []
        for pid, (a, k, occ) in enumerate(pulses):
            for s_, ph in occ:
                run_of[s_] = -2 - pid
                idx.append(s_), phase.append(ph % (2 * math.pi))
        segs[torch.tensor(idx, device=dev), 6] = torch.tensor(phase, dtype=f32, device=dev)
        pinfo = torch.tensor([[a, k] for a, k, _ in pulses], dtype=torch.int32).reshape(-1).to(dev)
        U_b = torch.zeros(len(pulses) * K * threads, device=dev, dtype=f32)
        prm_p = torch.tensor([n, threads, len(pulses), int(pool is not None)], dtype=torch.int32).to(dev)
        getattr(lib, "pulse_maps" + suffix)(U_b, spins, segs, flags, mt_bufs[1], mt_bufs[2], mt_bufs[3], mt_bufs[4],
                                            pinfo, prm_p, threads=len(pulses) * threads, group_size=group)
        bufs = (*bufs[:6], U_b, *bufs[7:])
    run_of, nu_b = run_of.to(dev), nu.to(dev, f32).contiguous()
    spans = runs + [(s_, k) for _, k, occ in pulses for s_, _ in occ]
    bounds, s0 = [], 0  # dispatches never split a run or a reused pulse
    while s0 < S:
        bounds.append(s0)
        e = min(S, s0 + chunk)
        for a, k in spans:
            if a < e < a + k:
                e = a if a > s0 else a + k
                break
        s0 = e
    starts = bounds
    ends = bounds[1:] + [S]
    tiles = (n_tiles, threads) if mma else (0, 0)
    prms = torch.from_numpy(np.array([[n, n_coils, s0, s1, *tiles, mt_flag, mode_flag, nj_v, seed_lo, seed_hi,
                                       kind_v, int(phantom.motion is not None), int(phantom.diffusion is not None), mapped,
                                       int(fmove), int(bool(runs)), int(bool(pulses)), pbase] for s0, s1 in zip(starts, ends)],
                                     dtype=np.uint32).view(np.int32)).to(dev)

    ckpts = []
    mb_row = mt_bufs[0][None] if pool is not None else torch.zeros(1, n, device=dev, dtype=f32)
    for c, s0 in enumerate(starts):  # one upload of every dispatch's parameters (prms[c]: a view)
        if keep:  # the state before this dispatch: m (3), mb, their low parts (4)
            ckpts.append(torch.cat([M, mb_row, lo_buf]).clone())
        if mma:
            getattr(lib, "bloch_chunk_mma" + suffix)(M, spins, sens, segs, flags, rows, sig, *bufs, prms[c],
                                                     *bm_bufs, fm_buf, lo_buf, run_of, nu_b, threads=threads,
                                                     group_size=group)
        else:
            getattr(lib, "bloch_chunk" + suffix)(M, spins, sens, segs, flags, rows, sig, *bufs, prms[c],
                                                 *bm_bufs, fm_buf, lo_buf, threads=threads, group_size=group)
    backend.synchronize()
    full = sig.cpu()
    sig = full[: n_adc * n_coils * 2].view(n_adc, n_coils, 2)
    signal = torch.complex(sig[..., 0], sig[..., 1]).T.contiguous()
    if runs:  # the grids: FFT, deconvolve the kernel, recombine the decay terms, demodulate, add to the samples
        width, terms = nufft
        grid = full[n_adc * n_coils * 2:].view(len(runs), terms, ng, n_coils, 2).double()
        F = torch.fft.fft(torch.complex(grid[..., 0], grid[..., 1]), dim=2)
        n_r, dt_r = runs[0][1], float(seq.dt[runs[0][0]])
        kp = np.arange(-(n_r // 2), n_r - n_r // 2)
        hat = torch.from_numpy(_es_hat(kp, ng, width, 2.30 * width))
        Fk = F[:, :, torch.from_numpy(kp % ng), :] / hat[None, None, :, None]  # (runs, terms, n, C)
        tt = (np.arange(n_r) + 1) * dt_r
        nodes = nu[6:6 + terms].numpy()
        lag = np.ones((terms, n_r))
        for a_ in range(terms):
            for b_ in range(terms):
                if a_ != b_:
                    lag[a_] *= (tt - nodes[b_]) / (nodes[a_] - nodes[b_])
        vals = torch.einsum("mi,rmic->ric", torch.from_numpy(lag).to(Fk.dtype), Fk)  # (runs, n, C)
        seg_idx = torch.tensor([a + i for a, k in runs for i in range(k)])
        psi_r = torch.remainder(seq.adc_phase.detach().double().cpu()[seg_idx], 2 * math.pi)
        vals = vals.reshape(-1, n_coils) * torch.exp(-1j * psi_r)[:, None]
        adc_row = (torch.cumsum(torch.tensor(seq.adc, dtype=torch.int64), 0) - 1)[seg_idx]
        signal = signal.to(torch.complex128)
        signal[:, adc_row] += vals.T
        signal = signal.to(torch.complex64)
    if pools:
        st = bm_bufs[0].cpu().double()
        st = (st[:len(st) // 2] + st[len(st) // 2:]).float()
        m_pools = torch.stack([st[3 + 3 * k: 6 + 3 * k].T for k in range(len(pools))], 1).contiguous()
        return SimResult(signal, st[:3].T.contiguous(), st[-1].clone() if pool is not None else None, m_pools)
    lo = lo_buf.cpu().double()
    m = (M.cpu().double() + lo[:3]).float().T.contiguous()
    res = SimResult(signal, m, (mt_bufs[0].cpu().double() + lo[3]).float() if pool is not None else None)
    if not keep:
        return res
    sens_d = torch.stack([coils.real, coils.imag], -1).reshape(n, 2 * n_coils).T.to(dev, f32).contiguous()
    return res, dict(ckpts=ckpts, spins=spins, sens=sens_d, segs=segs, flags=flags, rows=rows, pool=mt_bufs[1],
                     frf=mt_bufs[2], line=mt_bufs[3], log_f=mt_bufs[4], mt=mt_flag, conc=conc_buf[0], mode=conc_buf[1],
                     n_adc=n_adc, n_coils=n_coils, threads=threads, suffix=suffix, sig=sig)
