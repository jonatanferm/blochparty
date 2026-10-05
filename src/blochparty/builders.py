"""Ready-made sequences.

``gre_cartesian`` / ``gre_2d`` / ``gre_3d`` default to the idealised sequence (hard pulse,
instantaneous gradients, ideal spoiling); ``spoiling``, ``slice_thickness`` and ``ramps``
switch on the realistic versions, built from the ``System``-limited blocks of
``sequence.py``. ``se_2d``, ``tse_2d``, ``bssfp_2d`` and ``epi_2d`` are always built that way.

What each returns in ``meta``: ``kspace_shape`` when the ADC samples reshape directly to
the Cartesian grid (GRE, SE, bSSFP: ``bp.reconstruct`` works). TSE (line ordering) and EPI
(alternating readouts) instead give ``matrix`` and ``kspace_index``, the flat grid index of
every ADC sample; ``cartesian_kspace`` / ``cartesian_recon`` handle every builder.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from typing import NamedTuple

import torch
from torch import Tensor

from .recon import ifftnc, kspace_grid
from .sequence import (_T_TOL, ADC, RFPulse, Sequence, SequenceBuilder, System, Trapezoid, _ceil_to, _vector,
                       block_duration, block_pulse, sinc_pulse, trapezoid)

SPOILING = ("ideal", "gradient", "rf")


def gre_cartesian(
    shape: tuple[int, ...],
    fov: float = 0.22,
    flip_deg=15.0,
    tr=10e-3,
    te: float | None = None,
    dwell: float = 10e-6,
    rf_duration: float | None = None,
    rf_phase: float = 0.0,
    n_dummy: int = 0,
    dtype=torch.float64,
    *,
    spoiling: str = "ideal",
    slice_thickness: float | None = None,
    slice_offset: float = 0.0,
    ramps: bool = False,
    system: System | None = None,
    spoiler_cycles: float = 4.0,
    rf_spoil_increment: float = 117.0,
    time_bw_product: float = 4.0,
    apodization: float = 0.5,
) -> Sequence:
    """Cartesian spoiled gradient echo, 2D ``(ny, nx)`` or 3D ``(nz, ny, nx)``.

    Each TR reads one line along x. Lines are ordered with ky fastest (then kz), so the
    ADC samples reshape directly to ``shape``. Along an axis of length n, index i
    samples k = (i - n/2) / fov; the echo (k = 0) is at index n/2 on every axis.
    ``flip_deg`` and ``tr`` may be tensors that require grad.

    By default: a non-selective hard pulse of ``rf_duration`` (20 us), ideal (instantaneous)
    gradients and ideal spoiling at the end of every TR. The realistic options:

    * ``spoiling``: "ideal" (zero Mxy at the end of each TR), "gradient" (a spoiler lobe on x
      bringing the net x moment per TR to ``spoiler_cycles`` cycles across a voxel, with
      phase-encode rewinders), or "rf" (the same plus quadratic RF spoiling: pulse j has
      phase ``rf_phase`` + increment * j (j + 1) / 2 and the receiver follows it).
    * ``slice_thickness`` [m]: windowed-sinc excitation of ``rf_duration`` (default 2 ms) with
      slice-select gradient on z, centred at ``slice_offset``, rephased in the prephaser
      (a slab in 3D). Default None: non-selective.
    * ``ramps``: trapezoidal gradients within ``system``'s limits (default ``System()``);
      otherwise rectangles. The ADC stays on the readout flat top.

    With any of these, pulses respect the RF dead / ringdown times and dummy TRs repeat
    the full TR (gradients included, no ADC). TE runs from the pulse centre to the k = 0
    sample; with ``te`` None the prephaser lasts 1 ms (rectangles) or is as short as possible.
    """
    if spoiling not in SPOILING:
        raise ValueError(f"spoiling must be one of {SPOILING}, not {spoiling!r}")
    if spoiling != "ideal" or slice_thickness is not None or ramps:
        return _gre_blocks(shape, fov, flip_deg, tr, te, dwell, rf_duration, rf_phase, n_dummy, dtype,
                           spoiling, slice_thickness, slice_offset, ramps, system or System(), spoiler_cycles,
                           rf_spoil_increment, time_bw_product, apodization)
    rf_duration = 20e-6 if rf_duration is None else rf_duration
    *pe_shape, nx = shape
    dk = 1.0 / fov
    t_read = nx * dwell
    t_echo = (nx / 2 + 0.5) * dwell  # from readout start to the k = 0 sample
    t_pre = 1e-3 if te is None else te - rf_duration / 2 - t_echo
    if t_pre <= 0:
        raise ValueError(f"TE too short: prephaser would last {t_pre * 1e3:.3f} ms")
    t_fill = tr - rf_duration - t_pre - t_read
    if float(t_fill) <= 0:
        raise ValueError(f"TR too short by {-float(t_fill) * 1e3:.3f} ms")

    b = SequenceBuilder(dtype=dtype)
    for _ in range(n_dummy):
        b.rf(rf_duration, flip_deg=flip_deg, phase=rf_phase)
        b.delay(tr - rf_duration).spoil()
    for idx in itertools.product(*(range(m) for m in pe_shape)):
        k_pe = [(i - m // 2) * dk for i, m in zip(idx, pe_shape)]  # (ky,) or (kz, ky)
        ky, kz = k_pe[-1], (k_pe[0] if len(k_pe) == 2 else 0.0)
        b.rf(rf_duration, flip_deg=flip_deg, phase=rf_phase)
        b.delay(t_pre, grad=(-(nx / 2 + 0.5) * dk / t_pre, ky / t_pre, kz / t_pre))
        b.readout(nx, dwell, grad=(dk / dwell, 0.0, 0.0))
        b.delay(t_fill).spoil()
    te_actual = rf_duration / 2 + t_pre + t_echo
    return b.build(kspace_shape=tuple(shape), fov=fov, te=te_actual, tr=tr, dwell=dwell)


def gre_2d(n: int = 64, fov: float = 0.22, **kwargs) -> Sequence:
    """2D n x n spoiled GRE; see ``gre_cartesian`` for the options."""
    return gre_cartesian((n, n), fov, **kwargs)


def gre_3d(n: int = 32, fov: float = 0.22, **kwargs) -> Sequence:
    """3D n x n x n spoiled GRE (phase encoding on y and z); see ``gre_cartesian``."""
    return gre_cartesian((n, n, n), fov, **kwargs)


def readout_bandwidth_per_pixel(seq: Sequence) -> float:
    """Hz per pixel along the readout (1 / readout duration)."""
    shape = seq.meta["kspace_shape"] if "kspace_shape" in seq.meta else seq.meta["matrix"]
    return 1.0 / (shape[-1] * seq.meta["dwell"])


def cartesian_kspace(signal: Tensor, seq: Sequence) -> Tensor:
    """(C, n_adc) signal -> (C, *matrix) Cartesian k-space for any builder here: places
    every sample at ``meta["kspace_index"]`` when present (TSE, EPI), else reshapes."""
    if "kspace_index" not in seq.meta:
        return kspace_grid(signal, seq)
    shape = seq.meta["matrix"]
    out = signal.new_zeros(signal.shape[0], math.prod(shape))
    out[:, seq.meta["kspace_index"]] = signal
    return out.reshape(signal.shape[0], *shape)


def cartesian_recon(signal: Tensor, seq: Sequence) -> Tensor:
    """(C, n_adc) signal -> (C, *matrix) coil images (centred iFFT of ``cartesian_kspace``)."""
    k = cartesian_kspace(signal, seq)
    return ifftnc(k, k.ndim - 1)


# --- realistic building blocks -------------------------------------------------------


class _Excitation(NamedTuple):
    grads: tuple  # slice-select gradient, if any
    rf: RFPulse  # phase 0; rotate with _with_phase
    rephase: Tensor  # (3,) area undoing the slice-select moment after the pulse centre
    duration: float  # of the block
    pre: float  # block start -> pulse centre
    post: float  # pulse centre -> block end


def _excitation(flip_deg, rf_duration, slice_thickness, slice_offset, time_bw_product, apodization,
                sysm: System, dtype) -> _Excitation:
    """Hard pulse, or windowed sinc with slice select when ``slice_thickness`` is given."""
    if slice_thickness is None:
        rf = block_pulse(flip_deg, 20e-6 if rf_duration is None else rf_duration, system=sysm, dtype=dtype)
        grads, reph = (), torch.zeros(3, dtype=dtype)
    else:
        rf, gz, reph = sinc_pulse(flip_deg, 2e-3 if rf_duration is None else rf_duration,
                                  slice_thickness=slice_thickness, time_bw_product=time_bw_product,
                                  apodization=apodization, slice_offset=slice_offset, system=sysm, dtype=dtype)
        grads = (gz,)
    dur = block_duration(*grads, rf=rf, system=sysm)
    centre = rf.delay + rf.center
    return _Excitation(grads, rf, reph, dur, centre, dur - centre)


def _with_phase(rf: RFPulse, phase) -> RFPulse:
    """``rf`` rotated by ``phase`` [rad] (a float or tensor)."""
    ph = torch.as_tensor(phase, dtype=rf.signal.dtype)
    c, s = torch.cos(ph), torch.sin(ph)
    x, y = rf.signal.unbind(-1)
    return replace(rf, signal=torch.stack([c * x - s * y, s * x + c * y], dim=-1))


def _scaled(rf: RFPulse, factor: float) -> RFPulse:
    return replace(rf, signal=rf.signal * factor)


class _Readout(NamedTuple):
    grad: Trapezoid
    adc_delay: float
    t_k0: float  # block start -> the k = 0 sample
    area_k0: float  # x moment from block start to the k = 0 sample
    area: float  # total x moment of the readout lobe
    duration: float


def _readout(n: int, dwell: float, fov: float, ramps: bool, sysm: System, dtype, centred: bool) -> _Readout:
    """Readout lobe on x with the ADC on its flat top; sample i sits at kx = (i - n/2) / fov.

    Not centred: samples at (i + 1/2) dwell after the flat top starts (the Pulseq ADC
    convention, as in ``SequenceBuilder.readout``). Centred: half a dwell earlier, so the
    k = 0 sample is the middle of the lobe and the lobe splits into equal halves around
    the echo (spin echoes, balanced SSFP)."""
    sysm.check_raster(dwell, sysm.adc_raster, "dwell")
    dk = 1.0 / fov
    amp, flat = dk / dwell, n * dwell
    if ramps:
        g = trapezoid("x", amplitude=amp, flat_time=flat, system=sysm, dtype=dtype)
    else:
        g = Trapezoid(_vector(amp, "x", dtype), 0.0, flat, 0.0)
    r = g.rise
    g_delay = max(0.0, 0.5 * dwell - r) if centred else 0.0  # the first sample must be inside the block
    g = replace(g, delay=g_delay)
    if centred:
        adc_delay, t_k0, area_k0 = g_delay + r - 0.5 * dwell, g_delay + r + 0.5 * flat, amp * (0.5 * r + 0.5 * flat)
    else:
        adc_delay, t_k0, area_k0 = r, r + (n / 2 + 0.5) * dwell, amp * 0.5 * r + (n / 2 + 0.5) * dk
    return _Readout(g, adc_delay, t_k0, area_k0, amp * (r + flat), g.duration)


def _lobe(area, duration: float | None, ramps: bool, sysm: System, dtype) -> Trapezoid:
    """Gradient lobe with (3,) ``area``: a trapezoid lasting ``duration`` (None: as short as
    possible) or, without ramps, a rectangle of that duration (None: the shortest
    trapezoid's)."""
    if ramps:
        return trapezoid(area=area, duration=duration, system=sysm, dtype=dtype)
    a = _vector(area, None, dtype)
    if duration is None:
        duration = _min_duration(float(a.detach().abs().max()), sysm)
    return Trapezoid(a / duration if duration > 0 else a * 0.0, 0.0, duration, 0.0)


def _min_duration(area: float, sysm: System) -> float:
    """Duration of the shortest trapezoid with this area on one axis."""
    return trapezoid("x", area=area, system=sysm).duration


def _rf_spoil_phase(j: int, increment_deg: float) -> float:
    """Quadratic RF-spoiling phase of pulse j [rad]: increment * j (j + 1) / 2, wrapped in degrees."""
    return math.radians((increment_deg * j * (j + 1) / 2) % 360.0)


def _pad(b: SequenceBuilder, t: float) -> None:
    if t > _T_TOL:
        b.delay(t)


def _gre_blocks(shape, fov, flip_deg, tr, te, dwell, rf_duration, rf_phase, n_dummy, dtype, spoiling,
                slice_thickness, slice_offset, ramps, sysm, spoiler_cycles, rf_spoil_increment,
                time_bw_product, apodization) -> Sequence:
    """``gre_cartesian`` with realistic options."""
    *pe_shape, nx = shape
    dk = 1.0 / fov
    exc = _excitation(flip_deg, rf_duration, slice_thickness, slice_offset, time_bw_product, apodization, sysm, dtype)
    ro = _readout(nx, dwell, fov, ramps, sysm, dtype, centred=False)
    lines = []
    for idx in itertools.product(*(range(m) for m in pe_shape)):
        k_pe = [(i - m // 2) * dk for i, m in zip(idx, pe_shape)]
        lines.append((k_pe[-1], k_pe[0] if len(k_pe) == 2 else 0.0))  # (ky, kz)
    k_max = max(max(abs(ky), abs(kz)) for ky, kz in lines)
    pre_x, reph_z = -ro.area_k0, exc.rephase[2]

    if te is None:
        t_pre = _min_duration(max(abs(pre_x), k_max + abs(float(reph_z))), sysm) if ramps else 1e-3
    else:
        t_pre = te - exc.post - ro.t_k0
        if t_pre <= 0:
            raise ValueError(f"TE too short: prephaser would last {t_pre * 1e3:.3f} ms")
    spoil_x, t_spoil = 0.0, 0.0
    if spoiling != "ideal":  # net x moment per TR: spoiler_cycles across a voxel
        spoil_x = spoiler_cycles * nx / fov - (pre_x + ro.area)
        t_spoil = _min_duration(max(abs(spoil_x), k_max), sysm)
    t_fill = tr - (exc.duration + t_pre + ro.duration + t_spoil)
    if float(t_fill) < -_T_TOL:
        raise ValueError(f"TR too short by {-float(t_fill) * 1e3:.3f} ms")

    b = SequenceBuilder(dtype=dtype, system=sysm)
    rf_phases = []

    def one_tr(j, ky, kz, acquire):
        psi = _rf_spoil_phase(j, rf_spoil_increment) if spoiling == "rf" else 0.0
        rf_phases.append(psi)
        b.block(*exc.grads, rf=_with_phase(exc.rf, rf_phase + psi))
        b.block(_lobe((pre_x, ky, kz + reph_z), t_pre, ramps, sysm, dtype))
        b.block(ro.grad, adc=ADC(nx, dwell, ro.adc_delay, psi) if acquire else None, duration=ro.duration)
        if spoiling != "ideal":
            b.block(_lobe((spoil_x, -ky, -kz), t_spoil, ramps, sysm, dtype))
        if isinstance(t_fill, Tensor) or t_fill > _T_TOL:
            b.delay(t_fill)
        if spoiling == "ideal":
            b.spoil()

    for j in range(n_dummy):
        one_tr(j, 0.0, 0.0, acquire=False)
    for j, (ky, kz) in enumerate(lines):
        one_tr(n_dummy + j, ky, kz, acquire=True)
    return b.build(kspace_shape=tuple(shape), fov=fov, te=exc.post + t_pre + ro.t_k0, tr=tr, dwell=dwell,
                   spoiling=spoiling, slice_thickness=slice_thickness, slice_offset=slice_offset, ramps=ramps,
                   rf_phases=rf_phases, system=sysm)


# --- spin echo / turbo spin echo -----------------------------------------------------------


def se_2d(n: int = 64, fov: float = 0.22, te: float = 20e-3, tr: float = 0.5, **kwargs) -> Sequence:
    """2D spin echo: one line per TR (90 deg along x, 180 deg along y, crushers around it,
    readout centred at ``te``). ``meta["kspace_shape"]`` is set, so ``bp.reconstruct``
    works. Options as for ``tse_2d``."""
    seq = tse_2d(n, fov, etl=1, esp=te, tr=tr, refocus_flip_deg=kwargs.pop("refocus_flip_deg", 180.0), **kwargs)
    seq.meta["kspace_shape"] = (n, n)
    return seq


def tse_2d(
    n: int = 64,
    fov: float = 0.22,
    etl: int = 8,
    esp: float | None = None,
    tr: float = 3.0,
    excite_flip_deg=90.0,
    refocus_flip_deg=180.0,
    dwell: float = 10e-6,
    rf_duration: float | None = None,
    ordering: str = "linear",
    n_dummy: int = 0,
    dtype=torch.float64,
    *,
    slice_thickness: float | None = None,
    slice_offset: float = 0.0,
    crusher_cycles: float = 2.0,
    spoiling: str = "ideal",
    spoiler_cycles: float = 4.0,
    ramps: bool = True,
    system: System | None = None,
    time_bw_product: float = 4.0,
    apodization: float = 0.5,
) -> Sequence:
    """2D turbo spin echo (CPMG): per shot a ``excite_flip_deg`` pulse along x, then ``etl``
    refocusing pulses along y every ``esp`` (None: the shortest possible), echoes at
    j * esp after the excitation centre, one phase-encode line each (rewound after it).

    ``refocus_flip_deg``: a scalar or ``etl`` values (e.g. a variable-flip train); may be
    tensors that require grad. ``ordering``: "linear" (echo e covers the e-th band of
    k-space; effective TE at the echo holding ky = 0) or "centric" (lines sorted by |ky|,
    so the first echo holds the centre). Needs n % etl == 0; n / etl shots, each lasting
    ``tr``, after ``n_dummy`` dummy shots. Every refocusing pulse sits between two crusher
    lobes on z of ``crusher_cycles`` across a voxel (``slice_thickness``, or fov / n).
    The read prephaser comes before the first refocusing pulse, so each echo interval is
    symmetric (the CPMG condition). ``spoiling``: "ideal" or "gradient" (a z spoiler of
    ``spoiler_cycles``) at the end of every shot.

    Returns ``meta["matrix"] = (n, n)`` and ``meta["kspace_index"]`` (grid index of every
    ADC sample; use ``cartesian_kspace`` / ``cartesian_recon``), ``te`` (effective),
    ``echo_times`` and ``line_echo`` (echo index of every ky line).
    """
    if n % etl:
        raise ValueError(f"n = {n} is not a multiple of etl = {etl}")
    if spoiling not in ("ideal", "gradient"):
        raise ValueError(f"spoiling must be 'ideal' or 'gradient', not {spoiling!r}")
    sysm = system or System()
    dk, n_shots = 1.0 / fov, n // etl
    flips = refocus_flip_deg
    if not isinstance(flips, Tensor) or flips.ndim == 0:
        flips = [flips] * etl if not isinstance(flips, (list, tuple)) else list(flips)
    if len(flips) != etl:
        raise ValueError(f"{len(flips)} refocusing flips for etl = {etl}")
    order = list(range(n))
    if ordering == "centric":
        order = sorted(order, key=lambda i: (abs(i - n // 2), i < n // 2))
    elif ordering != "linear":
        raise ValueError(f"ordering must be 'linear' or 'centric', not {ordering!r}")
    lines = [[order[e * n_shots + s] for e in range(etl)] for s in range(n_shots)]  # [shot][echo] -> ky index

    args = (rf_duration, slice_thickness, slice_offset, time_bw_product, apodization, sysm, dtype)
    exc = _excitation(excite_flip_deg, *args)
    refs = [_excitation(f, *args) for f in flips]
    ref = refs[0]  # all refocusing blocks share the timing; only the flip differs
    ro = _readout(n, dwell, fov, ramps, sysm, dtype, centred=True)
    crush = crusher_cycles / (slice_thickness if slice_thickness is not None else fov / n)
    t_p = _min_duration(max(ro.area_k0, abs(crush + float(exc.rephase[2]))), sysm)
    t_c = _min_duration(max(n // 2 * dk, crush), sysm)
    h0 = exc.post + t_p + ref.pre  # excitation centre -> first refocusing centre
    h1 = ref.post + t_c + ro.t_k0  # refocusing centre -> echo
    h2 = ro.duration - ro.t_k0 + t_c + ref.pre  # echo -> next refocusing centre
    esp_min = _ceil_to(2 * max(h0, h1, h2), 2 * sysm.grad_raster)
    if esp is None:
        esp = esp_min
    elif esp < 2 * max(h0, h1, h2) - _T_TOL:
        raise ValueError(f"echo spacing too short: minimum {2 * max(h0, h1, h2) * 1e3:.3f} ms")

    b = SequenceBuilder(dtype=dtype, system=sysm)
    kspace_index = []
    for shot in range(n_dummy + n_shots):
        acquire = shot >= n_dummy
        t_start = b.time
        b.block(*exc.grads, rf=exc.rf)
        b.block(_lobe((ro.area_k0, 0.0, crush + exc.rephase[2]), t_p, ramps, sysm, dtype))
        _pad(b, esp / 2 - h0)
        for e, r in enumerate(refs):
            line = lines[shot - n_dummy][e] if acquire else n // 2
            ky = (line - n // 2) * dk
            b.block(*r.grads, rf=_with_phase(r.rf, math.pi / 2))
            b.block(_lobe((0.0, ky, crush), t_c, ramps, sysm, dtype))
            _pad(b, esp / 2 - h1)
            b.block(ro.grad, adc=ADC(n, dwell, ro.adc_delay) if acquire else None, duration=ro.duration)
            _pad(b, esp / 2 - h2)
            b.block(_lobe((0.0, -ky, crush), t_c, ramps, sysm, dtype))
            if acquire:
                kspace_index += range(line * n, line * n + n)
        if spoiling == "gradient":
            b.block(_lobe((0.0, 0.0, spoiler_cycles * crush / crusher_cycles), None, ramps, sysm, dtype))
        t_fill = tr - (b.time - t_start)
        if t_fill < -_T_TOL:
            raise ValueError(f"TR too short by {-t_fill * 1e3:.3f} ms")
        _pad(b, t_fill)
        if spoiling == "ideal":
            b.spoil()
    line_echo = [0] * n
    for s in range(n_shots):
        for e in range(etl):
            line_echo[lines[s][e]] = e
    return b.build(matrix=(n, n), kspace_index=torch.tensor(kspace_index), fov=fov, dwell=dwell, tr=tr, esp=esp,
                   etl=etl, te=(line_echo[n // 2] + 1) * esp, echo_times=[(e + 1) * esp for e in range(etl)],
                   line_echo=line_echo, ordering=ordering, slice_thickness=slice_thickness, ramps=ramps,
                   system=sysm)


# --- balanced SSFP ----------------------------------------------------------------------------


def bssfp_2d(
    n: int = 64,
    fov: float = 0.22,
    flip_deg=40.0,
    tr: float | None = None,
    dwell: float = 10e-6,
    rf_duration: float | None = None,
    phase_cycle_deg: float = 180.0,
    prep: bool = True,
    n_dummy: int = 0,
    dtype=torch.float64,
    *,
    slice_thickness: float | None = None,
    slice_offset: float = 0.0,
    ramps: bool = True,
    system: System | None = None,
    time_bw_product: float = 2.0,
    apodization: float = 0.5,
) -> Sequence:
    """2D balanced SSFP: every TR has zero net gradient moment on every axis, the echo
    (k = 0 sample) sits at TE = TR / 2, and pulse j has phase j * ``phase_cycle_deg`` (the
    receiver follows it). ``prep``: an alpha/2 pulse TR/2 before the first pulse, with the
    phase the train would have one pulse earlier (for 180 deg cycling: the sign-alternated
    -alpha/2 catalyst). ``n_dummy`` TRs without ADC follow it. ``tr`` None: the shortest.
    ``meta["kspace_shape"]`` is set (one line per TR, ky ascending)."""
    sysm = system or System()
    dk = 1.0 / fov
    exc = _excitation(flip_deg, rf_duration, slice_thickness, slice_offset, time_bw_product, apodization,
                      sysm, dtype)
    ro = _readout(n, dwell, fov, ramps, sysm, dtype, centred=True)
    reph_z = exc.rephase[2]  # the pulse is symmetric: equal slice-select moments before / after its centre
    t_p = _min_duration(max(ro.area_k0, n // 2 * dk, abs(float(reph_z))), sysm)
    h1 = exc.post + t_p + ro.t_k0  # pulse centre -> echo
    h2 = ro.duration - ro.t_k0 + t_p + exc.pre  # echo -> next pulse centre
    # between the prep and the first pulse: rephase the one and prephase the other
    prep_lobe = _lobe((0.0, 0.0, 2 * reph_z), None, ramps, sysm, dtype) if slice_thickness is not None else None
    h0 = exc.post + (prep_lobe.duration if prep_lobe else 0.0) + exc.pre if prep else 0.0
    half = max(h0, h1, h2)
    if tr is None:
        tr = _ceil_to(2 * half, 2 * sysm.grad_raster)
    elif tr < 2 * half - _T_TOL:
        raise ValueError(f"TR too short: minimum {2 * half * 1e3:.3f} ms")
    phase = lambda j: math.radians((j * phase_cycle_deg) % 360.0)

    b = SequenceBuilder(dtype=dtype, system=sysm)
    if prep:
        b.block(*exc.grads, rf=_with_phase(_scaled(exc.rf, 0.5), phase(-1)))
        if prep_lobe is not None:
            b.block(prep_lobe)
        _pad(b, tr / 2 - h0)
    rf_phases = []
    for j in range(n_dummy + n):
        acquire = j >= n_dummy
        ky = (j - n_dummy - n // 2) * dk if acquire else 0.0
        psi = phase(j)
        rf_phases.append(psi)
        b.block(*exc.grads, rf=_with_phase(exc.rf, psi))
        b.block(_lobe((-ro.area_k0, ky, reph_z), t_p, ramps, sysm, dtype))
        _pad(b, tr / 2 - h1)
        b.block(ro.grad, adc=ADC(n, dwell, ro.adc_delay, psi) if acquire else None, duration=ro.duration)
        _pad(b, tr / 2 - h2)
        b.block(_lobe((-ro.area_k0, -ky, reph_z), t_p, ramps, sysm, dtype))
    return b.build(kspace_shape=(n, n), fov=fov, te=tr / 2, tr=tr, dwell=dwell, phase_cycle_deg=phase_cycle_deg,
                   rf_phases=rf_phases, slice_thickness=slice_thickness, ramps=ramps, system=sysm)


# --- EPI ----------------------------------------------------------------------------------------


def epi_2d(
    n: int = 64,
    fov: float = 0.22,
    flip_deg=90.0,
    te: float | None = None,
    dwell: float = 4e-6,
    rf_duration: float | None = None,
    dtype=torch.float64,
    *,
    slice_thickness: float | None = None,
    slice_offset: float = 0.0,
    ramps: bool = True,
    system: System | None = None,
    time_bw_product: float = 4.0,
    apodization: float = 0.5,
) -> Sequence:
    """Single-shot blipped 2D EPI: n readouts of alternating sign along x, ky stepped by
    a y blip between them. With ``ramps`` the readouts are trapezoids sampled on their
    flat tops, each blip is centred on the junction of the fall and rise ramps (a gap is
    added only if it does not fit), and the prephaser includes the ramp moment; without,
    rectangles and one-raster blips. ``te`` (k = 0, from the pulse centre; None: the
    shortest) is reached by a delay after the prephaser.

    Line j is ky index j; even lines run kx forwards, odd lines backwards. Returns
    ``meta["matrix"]``, ``meta["kspace_index"]`` (use ``cartesian_kspace`` /
    ``cartesian_recon``), ``esp`` (line period) and ``bandwidth_per_pixel_pe`` =
    1 / (n esp): a uniform off-resonance f shifts the image by f / that pixels along +y.
    """
    sysm = system or System()
    dk = 1.0 / fov
    exc = _excitation(flip_deg, rf_duration, slice_thickness, slice_offset, time_bw_product, apodization,
                      sysm, dtype)
    ro = _readout(n, dwell, fov, ramps, sysm, dtype, centred=False)
    r, flat = ro.grad.rise, ro.grad.flat
    if ramps:
        blip = trapezoid("y", area=dk, system=sysm, dtype=dtype)
    else:
        blip = Trapezoid(_vector(dk / sysm.grad_raster, "y", dtype), 0.0, sysm.grad_raster, 0.0)
    gap = _ceil_to(blip.duration - 2 * r, sysm.grad_raster) if blip.duration > 2 * r + _T_TOL else 0.0
    esp = 2 * r + flat + gap
    grads, adcs = [], []
    for j in range(n):
        grads.append(replace(ro.grad, amplitude=ro.grad.amplitude * (-1) ** j, delay=j * esp))
        adcs.append(ADC(n, dwell, j * esp + r))
        if j < n - 1:  # centred between the fall of line j and the rise of line j + 1
            centre = 0.5 * ((j * esp + r + flat) + ((j + 1) * esp + r))
            grads.append(replace(blip, delay=centre - 0.5 * blip.duration))
    pre = _lobe((-ro.area_k0, -n // 2 * dk, exc.rephase[2]), None, ramps, sysm, dtype)
    j0 = n // 2
    i0 = n // 2 if j0 % 2 == 0 else n // 2 - 1  # the k = 0 sample of the ky = 0 line
    t_k0 = j0 * esp + r + (i0 + 0.5) * dwell
    te_min = exc.post + pre.duration + t_k0
    if te is None:
        te = te_min
    elif te < te_min - _T_TOL:
        raise ValueError(f"TE too short: minimum {te_min * 1e3:.3f} ms")

    b = SequenceBuilder(dtype=dtype, system=sysm)
    b.block(*exc.grads, rf=exc.rf)
    b.block(pre)
    _pad(b, te - te_min)
    b.block(*grads, adc=adcs)
    index = [j * n + (i if j % 2 == 0 else n - 1 - i) for j in range(n) for i in range(n)]
    return b.build(matrix=(n, n), kspace_index=torch.tensor(index), fov=fov, te=te, dwell=dwell, esp=esp,
                   bandwidth_per_pixel_pe=1.0 / (n * esp), slice_thickness=slice_thickness, ramps=ramps,
                   system=sysm)
