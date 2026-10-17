"""Bloch simulation on the Apple GPU (Metal), forward and adjoint.

The same physics as the core (``bloch.py``) and the retired fused C++ kernel (``fused/kernel.h``,
removed 2026-10-09 and kept in the git history; the comments below cite it where code was ported
from it): per segment exact
free precession and relaxation (relaxation in the ``m += (m - m0) * expm1(-dt/T)`` form, which
float32 needs), Strang-split Rodrigues rotations on RF segments, B1+ scaling and per-segment
uniform B0 offsets; ADC samples at segment ends, demodulated by exp(-i psi); ideal spoils. RF
segments add their increments (rotation as m += s1 w x m + s2 w x (w x m), relaxation, exchange)
by an exact two-sum into a per-spin low part: float32 otherwise rounds all of m every segment, and
long RF trains (CW saturation, fine-raster shaped pulses) drifted from the core by 1e-2 after 1e5
segments.

One GPU thread per spin steps through the segments with its magnetization in registers. The
coil sum at ADC samples has two implementations. With two or more coils (``bloch_chunk_mma``),
each simdgroup buffers 8 samples of its 32 spins and does the sum as a complex matrix product
(coils x spins) . (spins x samples) on 8x8 simdgroup matrices, demodulating afterwards; the cost
barely grows with the coil count (48^3 GRE: 0.41 s for 1 coil, 0.42 s for 8, 1.6 s for 64). With
one coil (``bloch_chunk``) each sample is summed with simd_sum. Either way the per-simdgroup sums
are reduced across the threadgroup and added with one atomic per threadgroup. Segments are
dispatched in chunks so no single command buffer runs for long. Metal has no double precision:
everything is float32.

Magnetization transfer (``Phantom.bound``): the bound pool's Mz is a fourth state per thread and
longitudinal relaxation the exact exchange step of ``exchange.py``, saturated during RF (about
+15 % on a 3D GRE).

Free exchanging pools (``Phantom.pools``, up to ``MAX_POOLS``, with or without the rate-model bound
pool): the Bloch-McConnell system of ``bloch._simulate_exchange``, one kernel per pool count
(``_bm1`` ...). Each segment applies exp(A dt) to the state by a Taylor series on sub-steps, the
common precession of free segments factored out exactly, increments added by two-sum; a float32
matrix exponential per segment was 1e-2 off over a CEST saturation train. A segment costs sub-steps
in proportion to |A| dt, so spins are sorted by that cost before the dispatch (a simdgroup waits
for its slowest lane) and returned in the caller's order.

The scanner field model (``fields.Scanner``: gradient nonlinearity through the encoding
positions, concomitant fields at leading order or exact) enters the off-resonance as in
``fused/kernel.h``.

Gradients: a hand-written adjoint (``engines.gpu.adjoint._GPUSimulate``), the reverse sweep of fused/kernel.h on the GPU.
The forward stores the state (hi and lo parts) before every ``checkpoint_every`` segments; per interval
in reverse order, ``bloch_replay`` reruns the forward keeping m before every 4th segment, and ``bloch_adjoint``
recomputes each block of 4 into threadgroup memory and sweeps it backward. Per-spin gradients stay with
their thread; per-segment ones are summed per simdgroup and then in order (``seg_reduce``), so the result
is the same for any interval and every run. Not for moving spins, free pools or the generalized bound
pool. What runs where: ``docs/ENGINES.md``.

This module is the Metal backend of the GPU driver (``engines.gpu``): the library build from ``../shaders`` and
``*.metal.in``, the backend object and ``simulate_metal``; ``blochparty.metal`` is this module.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ....bloch import SimResult
from ....phantom import Phantom
from ....sequence import Sequence
from .. import _simulate, source
from ..adjoint import N_SPIN_GRADS, default_checkpoint_every  # noqa: F401 (public names of blochparty.metal)
from ..driver import (ADJOINT_COILS, MAX_COILS, MAX_POOLS, MMA_MIN_COILS, NU_HEAD, NUFFT_MIN_SAMPLES,  # noqa: F401
                      NUFFT_WMAX, PULSE_MEMORY)

# the device code (../shaders, shared with CUDA) and the Metal kernels, templated by @SUFFIX@, @V@, @DIFF@, @NP@, @NC@
_SOURCE = source("shaders/common.metal")
_ADJOINT_COMMON = source("shaders/adjoint_common.metal")
_KERNELS = source("metal/kernels.metal.in")
_PULSE_KERNEL = source("metal/pulse.metal.in")
_ADJOINT_KERNEL = source("metal/adjoint.metal.in")

_LIB = None


def _library_source() -> str:
    """The Metal library's text: the shared device code and every kernel variant (what ``_lib`` compiles)."""
    # each kernel four times (advance<V>): plain, with the rate-model bound pool and/or the field
    # model, with the generalized Bloch modes, and with free exchanging pools; extra per-thread
    # state costs the plain kernels ~10-30 % in register pressure otherwise
    variants = [(suffix + dsuf, v, diff, "0") for suffix, v in (("", "0"), ("_full", "1"), ("_gen", "2"))
                for dsuf, diff in (("", "false"), ("_diff", "true"))]
    # free pools, one kernel per pool count
    variants += [(f"_bm{k}{dsuf}", "3", diff, str(k)) for k in range(1, MAX_POOLS + 1)
                 for dsuf, diff in (("", "false"), ("_diff", "true"))]
    kernels = "".join(_KERNELS.replace("@SUFFIX@", suffix).replace("@V@", v).replace("@DIFF@", diff)
                      .replace("@NP@", np_) for suffix, v, diff, np_ in variants)
    assert f"#define MAX_POOLS {MAX_POOLS} " in _SOURCE
    replay, adjoint = _ADJOINT_KERNEL.split("// NC: coils held in registers")
    adjoint = "// NC: coils held in registers" + adjoint
    for suffix, v in (("", "0"), ("_full", "1")):
        kernels += replay.replace("@SUFFIX@", suffix).replace("@V@", v)
        kernels += "".join(adjoint.replace("@SUFFIX@", suffix).replace("@V@", v).replace("@NC@", str(c)) for c in ADJOINT_COILS)
    for suffix, v in (("", "0"), ("_full", "1")):
        kernels += _PULSE_KERNEL.replace("@SUFFIX@", suffix).replace("@V@", v)
    return _SOURCE + _ADJOINT_COMMON + kernels


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(_library_source())
    return _LIB


def available() -> bool:
    try:
        return torch.backends.mps.is_available() and _lib() is not None
    except Exception:
        return False


class _Metal:
    """The driver's backend: where buffers live, the compiled kernels (called as ``kernel(*buffers, threads=,
    group_size=)``) and how to wait for them. ``blochparty.cuda`` supplies another with the same kernels."""

    name = "metal"
    max_coils = None  # the matrix-product reduction has no limit

    @property
    def device(self) -> torch.device:
        return torch.device("mps")

    def lib(self):
        return _lib()

    def synchronize(self) -> None:
        torch.mps.synchronize()

METAL = _Metal()


def simulate_metal(phantom: Phantom, seq: Sequence, coils: Tensor | None = None, *, m_init: Tensor | None = None,
                   chunk: int = 32768, group: int = 256, scanner=None, mma: bool | None = None,
                   seed: int = 0, maps=None, checkpoint_every: int | None = None, readouts: str = "direct",
                   nufft_width: int = 6, nufft_terms: int = 3, pulses: str = "step") -> SimResult:
    """Forward simulation on the Apple GPU; same arguments and result as ``blochparty.simulate``
    (results come back on the CPU, float32). ``chunk`` segments per dispatch; ``group`` threads
    per threadgroup (must be 256). ``mma`` picks the coil reduction: matrix products on simdgroup
    matrices (default from ``MMA_MIN_COILS`` coils) or one simd_sum per coil and sample.

    Gradients (when grad mode is on and an input requires grad) come from a hand-written adjoint on the GPU
    (``bloch_adjoint``): to the sequence (dt, rf, grad, adc_phase, b0_offset), the phantom (pos, pd, t1, t2,
    db0, b1, the rate-model bound pool), coils, m_init and, through the encoding positions and concomitant
    coefficients, the scanner field model. The forward stores the state before every ``checkpoint_every``
    segments (a multiple of 64; default about sqrt(segments)); the backward replays one interval at a time.
    Not with diffusion, motion, free pools or the generalized bound pool (raises).

    ``readouts="nufft"`` (forward only; gradients keep the per-sample path): every readout of at least
    16 samples with one dt, gradient and B0 offset is evaluated as a type-1 NUFFT (docs/PERFORMANCE.md
    2.1): the spins are sorted along the readout's phase, each spreads its ``nufft_terms`` decay terms
    onto a 2x oversampled grid with an exponential-of-semicircle kernel of ``nufft_width`` cells, and
    jumps over the readout in one step; an FFT on the CPU gives the samples. For static spins without
    concomitant fields, diffusion, motion, free pools or the generalized bound pool (raises otherwise).
    Width 6 and 3 terms: ~1e-5 of the signal (float32's level).

    ``pulses="reuse"`` (forward only): a pulse (two or more consecutive RF segments) that recurs, up to a
    global RF phase, is stepped once per spin into its affine map (``pulse_maps``, from its first
    occurrence), and every occurrence applies that map turned about z by its phase (relaxation and
    exchange commute with z-rotations, so this is the same map, not an approximation). Costs
    ``PULSE_MEMORY`` bytes at most of maps (12 or 20 floats per spin and pulse, the most-used pulses
    first); for static spins without concomitant fields, diffusion, motion, free pools or the
    generalized bound pool (raises otherwise)."""
    return _simulate(METAL, phantom, seq, coils, m_init, chunk, group, scanner, mma, seed, maps, checkpoint_every,
                     readouts, nufft_width, nufft_terms, pulses)
