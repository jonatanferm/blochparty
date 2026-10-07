"""Fused CPU kernel: ``simulate_fused`` is ``blochparty.simulate``, fast, gradients included.

The pure-PyTorch simulator runs ~30 tensor ops over all spins per segment, so it is
memory-bandwidth bound. Spins are independent, so the C++ kernel (``kernel.h``) instead
takes a block of spins, keeps its magnetization in L1 and steps it through every
segment, with blocks spread over threads. The math is that of ``bloch.py``: in float64
the two agree to ~1e-13. In float32 the kernel is closer to the float64 result than
the PyTorch simulator in float32 is on short sequences (see ``tests/test_fused.py``);
its rounding grows with the number of segments, so on long ones (2D 96²) it is ~3x
further off (~1e-5) than the PyTorch simulator's closed-form free runs.

Magnetization transfer (``Phantom.bound``, ``exchange.py``) runs in the kernel too: a fourth
component per spin (the bound pool's Mz; m and checkpoints are then (N, 4)), the exact
exchange step in place of longitudinal relaxation, saturated during RF; about 1.45x the
cost on a 3D GRE. The adjoint covers it, the pool parameters and the RF frequency included.

Gradients come from a hand-written adjoint in the same kernel (reverse sweep over
checkpointed segment intervals, see ``kernel.h``), wrapped in an autograd Function: they
reach every tensor ``simulate`` differentiates (``Phantom.b1`` included) and whatever built
those tensors, and match autograd through ``simulate`` to ~1e-14 in float64
(``tests/test_fused_grad.py``, ``tests/test_phantom_b1.py``).
First-order only. CPU, float32 / float64.

The extension is compiled on first use with ``torch.utils.cpp_extension.load`` (needs
a C++20 compiler, as PyTorch's headers do, and ``ninja``) and cached by PyTorch
(``TORCH_EXTENSIONS_DIR``, default ``~/.cache/torch_extensions``). Extra compiler flags
can be passed in ``BLOCHPARTY_FUSED_CFLAGS``. ``kernel.h`` itself is C++17.
"""

from __future__ import annotations

import functools
import hashlib
import math
import os
import platform
from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from .. import exchange
from ..bloch import SimResult
from ..fields import Scanner, spin_fields
from ..phantom import Phantom
from ..sequence import Sequence

_HERE = Path(__file__).parent
BLOCK_SIZES = (16, 32, 64, 128, 256)
_RF, _ADC, _SPOIL = 1, 2, 4  # segment flags, as in kernel.h


@functools.cache
def _ext():
    from torch.utils.cpp_extension import load

    # PyTorch only hashes the .cpp for its rebuild check; the define makes kernel.h count too
    digest = hashlib.sha1((_HERE / "kernel.h").read_bytes() + (_HERE / "bm.h").read_bytes()).hexdigest()[:12]
    flags = ["-O3", "-fopenmp-simd", f"-DBLOCHPARTY_KERNEL_HASH={digest}"]
    if platform.machine() in ("x86_64", "AMD64"):
        flags.append("-march=native")  # AVX2 / AVX-512 instead of the SSE2 baseline
    flags += os.environ.get("BLOCHPARTY_FUSED_CFLAGS", "").split()
    return load(name="blochparty_fused", sources=[str(_HERE / "bindings.cpp")],
                extra_include_paths=[str(_HERE)], extra_cflags=flags, verbose=False)


def available() -> bool:
    """Whether the extension builds (or is already built) on this machine."""
    try:
        _ext()
        return True
    except Exception:
        return False


def default_block(n_spins: int, n_threads: int, dtype=torch.float32) -> int:
    """Largest block that still fits L1 (1 KB per lane array in float32, 0.5 KB in float64,
    best on an Apple M4) while leaving >= 4 blocks per thread for load balancing."""
    block = 256 if dtype == torch.float32 else 64
    while block > BLOCK_SIZES[0] and -(-n_spins // block) < 4 * n_threads:
        block //= 2
    return block


def default_checkpoint_every(n_segments: int) -> int:
    """About sqrt(segments): checkpoint memory and replay buffer balance."""
    return max(1, math.isqrt(max(n_segments - 1, 0)) + 1)


class _Opts(NamedTuple):
    n_threads: int
    block: int
    every: int
    conc_mode: int
    line: tuple = (0, 0.0, 0.0, 1.0, 0.0)  # bound pool: lineshape kind, cutoff, table x0, h, xmin
    t2b_min: float = 0.0  # generalized Bloch: the sub-step scale


class _FusedSimulate(torch.autograd.Function):
    """Forward stores m every ``every`` segments; backward is the kernel's reverse sweep."""

    @staticmethod
    def forward(ctx, opts: _Opts, pos, pd, t1, t2, db0, sens, dt, rf, grad, adc_phase, m0, b1, b0_offset, conc,
                bound, rf_freq, flags, log_f, modes):
        ctx.set_materialize_grads(False)
        ctx.opts = opts
        m = m0.detach().clone()
        sig, ckpt = _ext().simulate(pos, pd, t1, t2, db0, sens, dt, rf, grad, adc_phase, flags, m,
                                    opts.n_threads, opts.block, opts.every, b1, b0_offset, conc, opts.conc_mode,
                                    bound, log_f, rf_freq, *opts.line, modes, opts.t2b_min)
        ctx.save_for_backward(pos, pd, t1, t2, db0, sens, dt, rf, grad, adc_phase, flags, ckpt, sig, b1, b0_offset,
                              conc, bound, rf_freq, log_f, modes)
        return sig, m

    @staticmethod
    @once_differentiable
    def backward(ctx, g_sig, g_m):
        opts = ctx.opts
        (pos, pd, t1, t2, db0, sens, dt, rf, grad, adc_phase, flags, ckpt, sig, b1, b0_offset, conc, bound, rf_freq,
         log_f, modes) = ctx.saved_tensors
        # pos, pd, t1, t2, db0, sens, dt, rf, grad, adc_phase, m0, b1, b0_offset, conc, bound, rf_freq
        need = list(ctx.needs_input_grad[1:17])
        if g_sig is None:  # sensitivities and receiver phase only reach the signal
            need[5] = need[9] = False
        if (g_sig is None and g_m is None) or not any(need):
            return (None,) * 20
        prep = lambda g: None if g is None else g.to(pd.dtype).contiguous()
        # kernel order: dt, rf, grad, adc_phase, pos, pd, t1, t2, db0, sens, m, b1, b0_offset, conc, bound, rf_freq
        needs = [*need[6:10], *need[:6], *need[10:]]
        out = _ext().adjoint(pos, pd, t1, t2, db0, sens, dt, rf, grad, adc_phase, flags, ckpt, opts.every, sig,
                             prep(g_sig), prep(g_m), needs, opts.n_threads, opts.block, b1, b0_offset, conc,
                             opts.conc_mode, bound, log_f, rf_freq, *opts.line, modes, opts.t2b_min)
        g_dt, g_rf, g_grad, g_psi, g_pos, g_pd, g_t1, g_t2, g_db0, g_sens, g_m0, g_b1, g_off, g_conc, g_bd, g_frf = out
        return (None, g_pos, g_pd, g_t1, g_t2, g_db0, g_sens, g_dt, g_rf, g_grad, g_psi, g_m0, g_b1, g_off, g_conc,
                g_bd, g_frf, None, None, None)


class _BMOpts(NamedTuple):
    n_threads: int
    every: int
    conc_mode: int
    line: tuple


class _FusedBM(torch.autograd.Function):
    """Free exchanging pools (bm.h): forward with checkpoints, backward the exact adjoint."""

    @staticmethod
    def forward(ctx, opts: _BMOpts, pos, pd, t1, t2, db0, sens, dt, rf, grad, adc_phase, m0, b1, b0_offset, conc,
                pools, bound, rf_freq, flags, log_f):
        ctx.set_materialize_grads(False)
        ctx.opts = opts
        m = m0.detach().clone()
        sig, ckpt = _ext().bm_simulate(pos, pd, t1, t2, db0, sens, dt, rf, grad, adc_phase, flags, m, opts.n_threads,
                                       opts.every, b1, b0_offset, conc, opts.conc_mode, pools, bound, log_f, rf_freq,
                                       *opts.line)
        ctx.save_for_backward(pos, pd, t1, t2, db0, sens, dt, rf, grad, adc_phase, flags, ckpt, sig, b1, b0_offset,
                              conc, pools, bound, rf_freq, log_f)
        return sig, m

    @staticmethod
    @once_differentiable
    def backward(ctx, g_sig, g_m):
        opts = ctx.opts
        (pos, pd, t1, t2, db0, sens, dt, rf, grad, adc_phase, flags, ckpt, sig, b1, b0_offset, conc, pools, bound,
         rf_freq, log_f) = ctx.saved_tensors
        need = list(ctx.needs_input_grad[1:18])  # pos ... rf_freq
        if g_sig is None:
            need[5] = need[9] = False
        if (g_sig is None and g_m is None) or not any(need):
            return (None,) * 20
        prep = lambda g: None if g is None else g.to(pd.dtype).contiguous()
        needs = [*need[6:10], *need[:6], *need[10:]]
        out = _ext().bm_adjoint(pos, pd, t1, t2, db0, sens, dt, rf, grad, adc_phase, flags, ckpt, opts.every, sig,
                                prep(g_sig), prep(g_m), needs, opts.n_threads, b1, b0_offset, conc, opts.conc_mode,
                                pools, bound, log_f, rf_freq, *opts.line)
        (g_dt, g_rf, g_grad, g_psi, g_pos, g_pd, g_t1, g_t2, g_db0, g_sens, g_m0, g_b1, g_off, g_conc, g_pools, g_bd,
         g_frf) = out
        return (None, g_pos, g_pd, g_t1, g_t2, g_db0, g_sens, g_dt, g_rf, g_grad, g_psi, g_m0, g_b1, g_off, g_conc,
                g_pools, g_bd, g_frf, None, None)


def _simulate_pools(phantom, seq, coils, *, m_init, n_threads, checkpoint_every, scanner) -> SimResult:
    """simulate_fused with free exchanging pools: the exact engine of bm.h (bloch._simulate_exchange)."""
    pd = phantom.pd
    dtype = pd.dtype
    if phantom.bound is not None and phantom.bound.model == "generalized":
        raise NotImplementedError("free pools with the generalized Bloch bound pool are not implemented")
    n_threads = n_threads or os.cpu_count() or 1
    real = lambda t: t.to(dtype=dtype, device="cpu").contiguous()
    n = len(phantom)
    pools = [q.expand(n, pd) for q in phantom.pools]
    pool = None if phantom.bound is None else phantom.bound.expand(n, pd)
    if coils is None:
        sens = torch.zeros(n, 1, 2, dtype=dtype)
        sens[..., 0] = 1.0
    else:
        sens = torch.stack([real(coils.real), real(coils.imag)], dim=-1)
    # state: water, pools (x, y, z), bound Mzb
    m0 = [torch.zeros_like(pd), torch.zeros_like(pd), pd] if m_init is None else list(real(m_init)[:, :3].unbind(-1))
    for q in pools:
        m0 += [torch.zeros_like(pd), torch.zeros_like(pd), q.f * pd]
    if pool is not None:
        m0.append(real(m_init)[:, 3] if m_init is not None and m_init.shape[-1] == 4 else pool.f * pd)
    m0 = torch.stack(m0, -1).to(dtype).contiguous()
    flag = lambda bits, value: torch.tensor(bits, dtype=torch.uint8).reshape(len(seq)) * value
    flags = flag(seq.is_rf, _RF) | flag(seq.adc, _ADC) | flag(seq.spoil, _SPOIL)
    fields = spin_fields(scanner, phantom.pos)
    line, log_f, bound, rf_freq = (0, 0.0, 0.0, 1.0, 0.0), None, None, None
    if pool is not None:
        kind = 1 if pool.lineshape == "lorentzian" else 2
        lx, log_f = exchange._sl_table(dtype, torch.device("cpu"))
        line = (kind, float(pool.cutoff), float(lx[0]), float(lx[1] - lx[0]), exchange._SL_XMIN)
        log_f = log_f.contiguous()
        bound = torch.stack([real(pool.f), real(pool.k), real(pool.t1), real(pool.t2), real(pool.dw)], -1)
        rf_freq = real(exchange.rf_frequency(seq.to(dtype=dtype, device="cpu")))
    pools_t = torch.stack([torch.stack([real(q.f), real(q.k), real(q.t1), real(q.t2), real(q.dw)], -1) for q in pools],
                          1).contiguous()
    inputs = (real(fields.pos), real(pd), real(phantom.t1), real(phantom.t2), real(phantom.db0), sens.contiguous(),
              real(seq.dt), real(seq.rf), real(seq.grad), real(seq.adc_phase), m0,
              None if phantom.b1 is None else real(phantom.b1),
              None if seq.b0_offset is None else real(seq.b0_offset),
              None if fields.conc is None else real(fields.conc), pools_t, bound, rf_freq)
    if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in inputs):
        every = checkpoint_every or default_checkpoint_every(len(seq))
        sig, m = _FusedBM.apply(_BMOpts(n_threads, every, fields.mode, line), *inputs, flags, log_f)
    else:
        args = [None if t is None else t.detach() for t in inputs]
        m = args[10].clone()
        b1, b0_offset, conc, pools_d, bound_d, rf_freq_d = args[11:]
        sig, _ = _ext().bm_simulate(*args[:10], flags, m, n_threads, 0, b1, b0_offset, conc, fields.mode, pools_d,
                                    bound_d, log_f, rf_freq_d, *line)
    np_ = len(pools)
    m_pools = torch.stack([m[:, 3 * (i + 1):3 * (i + 2)] for i in range(np_)], 1)
    return SimResult(torch.view_as_complex(sig).T, m[:, :3], m[:, 3 * (np_ + 1)] if pool is not None else None,
                     m_pools)


def simulate_fused(
    phantom: Phantom,
    seq: Sequence,
    coils: Tensor | None = None,
    *,
    m_init: Tensor | None = None,
    n_threads: int | None = None,
    block: int | None = None,
    checkpoint_every: int | None = None,
    scanner: Scanner | None = None,
) -> SimResult:
    """Run ``seq`` on ``phantom``: same arguments and result as ``blochparty.simulate``,
    gradients included (``scanner``: the field model, as in ``simulate``).

    n_threads: worker threads (default: all logical CPUs).
    block: spins per block, one of ``BLOCK_SIZES`` (default: ``default_block``).
    checkpoint_every: when gradients are needed, the forward pass stores the
        magnetization of every spin before every ``checkpoint_every`` segments, and the
        backward pass replays one such interval at a time (default:
        ``default_checkpoint_every``). Gradients do not depend on it.
    """
    if getattr(phantom, "motion", None) is not None:
        raise NotImplementedError("motion (Phantom.motion) is in blochparty.simulate and the Metal kernel")
    if getattr(phantom, "diffusion", None) is not None:
        raise NotImplementedError("diffusion (Phantom.diffusion) is in blochparty.simulate and the Metal kernel")
    if getattr(phantom, "pools", None):
        return _simulate_pools(phantom, seq, coils, m_init=m_init, n_threads=n_threads,
                               checkpoint_every=checkpoint_every, scanner=scanner)
    pd = phantom.pd
    dtype = pd.dtype
    if dtype not in (torch.float32, torch.float64):
        raise TypeError(f"simulate_fused supports float32 and float64, got {dtype}")
    if pd.device.type != "cpu":
        raise ValueError("simulate_fused runs on the CPU")
    if checkpoint_every is not None and checkpoint_every < 1:
        raise ValueError(f"checkpoint_every must be positive, got {checkpoint_every}")
    n_threads = n_threads or os.cpu_count() or 1
    real = lambda t: t.to(dtype=dtype, device="cpu").contiguous()
    n = len(phantom)
    if coils is None:
        sens = torch.zeros(n, 1, 2, dtype=dtype)
        sens[..., 0] = 1.0
    else:
        sens = torch.stack([real(coils.real), real(coils.imag)], dim=-1)
    pool = None if phantom.bound is None else phantom.bound.expand(n, pd)
    if m_init is None:
        m0 = torch.stack([torch.zeros_like(pd), torch.zeros_like(pd), pd], dim=-1)
    else:
        m0 = real(m_init)
    if pool is not None and m0.shape[-1] == 3:  # bound pool at equilibrium, f PD
        m0 = torch.cat([m0, (pool.f * pd)[:, None]], dim=-1)
    gen = pool is not None and pool.model == "generalized"
    modes = None
    if gen:  # generalized Bloch: the mode table and the modes' state (at rest), after mzb
        from ..generalized_bloch import modes as green_modes

        pm, cm = green_modes(pool.lineshape, pool.memory)
        modes = torch.tensor([[p.real, p.imag, c.real, c.imag] for p, c in zip(pm, cm)], dtype=dtype)
        m0 = torch.cat([m0, m0.new_zeros(n, 2 * len(pm))], dim=-1)
    elif pool is None and m0.shape[-1] == 4:
        raise ValueError("m_init has a bound-pool column but the phantom has no bound pool")
    flag = lambda bits, value: torch.tensor(bits, dtype=torch.uint8).reshape(len(seq)) * value
    flags = flag(seq.is_rf, _RF) | flag(seq.adc, _ADC) | flag(seq.spoil, _SPOIL)
    fields = spin_fields(scanner, phantom.pos)
    inputs = (real(fields.pos), real(pd), real(phantom.t1), real(phantom.t2), real(phantom.db0), sens.contiguous(),
              real(seq.dt), real(seq.rf), real(seq.grad), real(seq.adc_phase), real(m0),
              None if phantom.b1 is None else real(phantom.b1),
              None if seq.b0_offset is None else real(seq.b0_offset),
              None if fields.conc is None else real(fields.conc))
    line, log_f, bound_in = (0, 0.0, 0.0, 1.0, 0.0), None, (None, None)
    t2b_min = float(pool.t2.detach().min()) if gen else 0.0
    if gen:  # no saturation lineshape or RF frequency: the modes carry the pool's response
        bound_in = (torch.stack([real(pool.f), real(pool.k), real(pool.t1), real(pool.t2), real(pool.dw)], -1), None)
    elif pool is not None:
        kind = 1 if pool.lineshape == "lorentzian" else 2
        lx, log_f = exchange._sl_table(dtype, torch.device("cpu"))
        line = (kind, float(pool.cutoff), float(lx[0]), float(lx[1] - lx[0]), exchange._SL_XMIN)
        log_f = log_f.contiguous()
        bound_in = (torch.stack([real(pool.f), real(pool.k), real(pool.t1), real(pool.t2), real(pool.dw)], -1),
                    real(exchange.rf_frequency(seq.to(dtype=dtype, device="cpu"))))
    block = block or default_block(n, n_threads, dtype)
    if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (*inputs, *bound_in)):
        every = checkpoint_every or default_checkpoint_every(len(seq))
        sig, m = _FusedSimulate.apply(_Opts(n_threads, block, every, fields.mode, line, t2b_min), *inputs, *bound_in,
                                      flags, log_f, modes)
    else:
        *spin_seq, m, b1, b0_offset, conc = (None if t is None else t.detach() for t in inputs)
        bound, rf_freq = (None if t is None else t.detach() for t in bound_in)
        m = m.clone()
        sig, _ = _ext().simulate(*spin_seq, flags, m, n_threads, block, 0, b1, b0_offset, conc, fields.mode,
                                 bound, log_f, rf_freq, *line, modes, t2b_min)
    assert sig.shape[0] == seq.n_adc
    if pool is not None:
        return SimResult(torch.view_as_complex(sig).T, m[:, :3], m[:, 3])
    return SimResult(torch.view_as_complex(sig).T, m)


__all__ = ["BLOCK_SIZES", "available", "default_block", "default_checkpoint_every", "simulate_fused"]
