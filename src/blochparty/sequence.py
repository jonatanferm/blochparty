"""Sequence representation: a flat list of piecewise-constant segments.

Each segment holds RF and gradients constant for ``dt`` seconds. Continuous
quantities (``dt``, ``rf``, ``grad``, ``adc_phase``) are tensors so they can
carry gradients; structural flags (``is_rf``, ``adc``, ``spoil``) are plain
booleans that decide control flow and are never differentiated.

Units follow Pulseq: RF in Hz (gamma_bar * B1), gradients in Hz/m
(gamma_bar * G), so gradient areas are k-space positions in cycles/m.

Realistic building blocks (``System`` limits, ``Trapezoid`` / ``GradientWaveform``
gradients, ``RFPulse`` hard and windowed-sinc pulses, ``ADC``) are combined into
segments by ``SequenceBuilder.block``, which works like a Pulseq block: the block is
split at every event boundary (gradient corners, gradient-raster points on sloped
pieces, RF raster steps, ADC samples). Gradients are piecewise linear, so each
segment's midpoint value is its exact average and gradient moments (k-space) are
exact at every segment end; spins see ramps as a staircase on the gradient raster.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from torch import Tensor

from .constants import PULSEQ_GAMMA_BAR

_T_TOL = 1e-11  # [s] event times closer than this are the same breakpoint
_AXES = {"x": 0, "y": 1, "z": 2}


@dataclass
class Sequence:
    dt: Tensor  # (S,) segment durations [s]
    rf: Tensor  # (S, 2) RF field (x, y) components [Hz]
    grad: Tensor  # (S, 3) gradient [Hz/m]
    adc_phase: Tensor  # (S,) receiver phase [rad], used where adc is True
    is_rf: list[bool]
    adc: list[bool]  # sample the signal at the end of the segment
    spoil: list[bool]  # ideal crusher: zero transverse magnetization at segment end
    meta: dict = field(default_factory=dict)
    # (S,) spatially uniform field [Hz] added to every spin's off-resonance (B0 eddy
    # currents, drift, ...); None means zeros
    b0_offset: Tensor | None = None
    # (S,) RF carrier frequency [Hz] per segment (a pulse turning as exp(-i 2 pi f t) is
    # resonant with spins at offset f), used for the bound-pool saturation (``exchange.py``);
    # None: derived from the phase of consecutive RF segments (``exchange.rf_frequency``)
    rf_freq: Tensor | None = None
    # RF segments only the bound pool feels (None: none): the water precesses and relaxes as in a free
    # segment, the bound pool is saturated by the segment's RF at ``rf_freq`` (a multi-slice sequence's
    # far slices, ``builders.tse_2d(bound_only_beyond=...)``); not a scanner sequence
    bound_only: list[bool] | None = None

    def __len__(self) -> int:
        return self.dt.shape[0]

    @property
    def n_adc(self) -> int:
        return sum(self.adc)

    @property
    def duration(self) -> Tensor:
        return self.dt.sum()

    def subdivide_rf(self, k: int) -> Sequence:
        """Split every RF segment into ``k`` equal parts. Relaxation during RF is
        handled by second-order splitting, so this shrinks that error by ~k**2."""
        from .exchange import rf_frequency  # the carrier is fixed before the pieces lose the phase steps

        reps = torch.tensor([k if r else 1 for r in self.is_rf], device=self.dt.device)
        rep = lambda t: t.repeat_interleave(reps, dim=0)

        def tail(flags):  # end-of-segment flags stay on the last piece
            out = []
            for f, r in zip(flags, self.is_rf):
                out += ([False] * (k - 1) + [f]) if r else [f]
            return out

        return replace(
            self,
            meta=_remap_spans(self.meta, reps.cpu().numpy()),
            dt=rep(self.dt / reps.to(self.dt.dtype)),
            rf=rep(self.rf),
            grad=rep(self.grad),
            adc_phase=rep(self.adc_phase),
            b0_offset=None if self.b0_offset is None else rep(self.b0_offset),
            rf_freq=rep(rf_frequency(self)),
            is_rf=[r for r in self.is_rf for _ in range(k if r else 1)],
            adc=tail(self.adc),
            spoil=tail(self.spoil),
            bound_only=None if self.bound_only is None else
            [b for b, r in zip(self.bound_only, self.is_rf) for _ in range(k if r else 1)],
        )

    def merge_free(self) -> Sequence:
        """Each stretch of consecutive segments without RF or ADC (a spoil may end one) as one
        segment with the stretch's duration, gradient moment and ``b0_offset`` integral; RF and
        ADC segments are kept bit for bit. For static spins the phase and relaxation at the end of
        such a stretch depend on nothing else, so the result is the same; not for concomitant
        fields (quadratic in g), diffusion or motion, which see the gradient's time course. The
        builders cut ramps and crushers on the gradient raster for Pulseq and the design checks;
        an engine does not need that (realism TSE 128: 36 335 -> 19 726 segments, docs/PERFORMANCE.md
        2.4). Span-based meta (``blocks``, ``block_segments``, ``segment_block``) is dropped:
        a merged sequence is for simulation, not export."""
        is_rf, adc, spoil = (np.asarray(f, dtype=bool) for f in (self.is_rf, self.adc, self.spoil))
        free = ~is_rf & ~adc
        n = len(free)
        start = np.ones(n, dtype=bool)
        start[1:] = ~free[1:] | ~free[:-1] | spoil[:-1]
        gid = np.cumsum(start) - 1
        first = np.flatnonzero(start)
        last = np.append(first[1:] - 1, n - 1)
        merged = free[first] & (last > first)  # groups that change
        if not merged.any():
            return self
        dev = self.dt.device
        g = torch.as_tensor(gid, device=dev)
        take = torch.as_tensor(first, device=dev)
        rows = torch.as_tensor(np.flatnonzero(merged), device=dev)
        total = lambda x: torch.zeros((len(first), *x.shape[1:]), dtype=x.dtype, device=dev).index_add(0, g, x)
        dt_sum = total(self.dt)
        dt = self.dt[take].index_put((rows,), dt_sum[rows])
        grad = self.grad[take].index_put((rows,), total(self.grad * self.dt[:, None])[rows] / dt_sum[rows, None])
        b0 = None
        if self.b0_offset is not None:
            b0 = self.b0_offset[take].index_put((rows,), total(self.b0_offset * self.dt)[rows] / dt_sum[rows])
        rf_freq = None if self.rf_freq is None else self.rf_freq[take].index_put((rows,), torch.zeros_like(rows, dtype=self.rf_freq.dtype))
        meta = {k: v for k, v in self.meta.items() if k not in ("blocks", "block_segments", "segment_block")}
        return Sequence(dt, self.rf[take], grad, self.adc_phase[take], is_rf[first].tolist(), adc[first].tolist(),
                        spoil[last].tolist(), meta | {"merged_free": True}, b0, rf_freq,
                        None if self.bound_only is None else np.asarray(self.bound_only, dtype=bool)[first].tolist())

    def to(self, dtype=None, device=None) -> Sequence:
        conv = lambda t: t.to(dtype=dtype, device=device)
        return replace(
            self,
            dt=conv(self.dt),
            rf=conv(self.rf),
            grad=conv(self.grad),
            adc_phase=conv(self.adc_phase),
            b0_offset=None if self.b0_offset is None else conv(self.b0_offset),
            rf_freq=None if self.rf_freq is None else conv(self.rf_freq),
        )


def _remap_spans(meta: dict, reps: np.ndarray) -> dict:
    """``meta`` with segment spans (builder ``blocks``, imported ``block_segments``) moved to the
    segments of a sequence in which segment i became ``reps[i]`` segments."""
    if "blocks" not in meta and "block_segments" not in meta and "segment_block" not in meta:
        return meta
    edge = np.concatenate([[0], np.cumsum(reps)])  # new index of every old boundary
    out = dict(meta)
    if "blocks" in meta:
        out["blocks"] = [{**b, "span": (int(edge[b["span"][0]]), int(edge[b["span"][1]]))} for b in meta["blocks"]]
    if "block_segments" in meta:
        out["block_segments"] = [(int(edge[a]), int(edge[b])) for a, b in meta["block_segments"]]
    if "segment_block" in meta:
        out["segment_block"] = np.repeat(np.asarray(meta["segment_block"]), reps)
    return out


def kspace_trajectory(seq: Sequence) -> Tensor:
    """(n_adc, 3) k-space [1/m] at every ADC sample (segment ends): the gradient moment since the
    last excitation, negated at each refocusing pulse. Pulses are runs of RF segments centred at
    their midpoints; excitations flip less than 90.01 deg about their own frequency
    (``pulseq.flip_angle``), as in pypulseq's ``calculate_kspace``, which it equals on written
    files (tests/test_pulseq_write.py). NaN before the first excitation."""
    from .pulseq import flip_angle

    dt = seq.dt.detach().cpu().double().numpy()
    tb = np.concatenate([[0.0], np.cumsum(dt)])
    moment = np.concatenate([np.zeros((1, 3)), np.cumsum(seq.grad.detach().cpu().double().numpy() * dt[:, None], 0)])
    at = lambda t: np.stack([np.interp(t, tb, moment[:, j]) for j in range(3)], -1)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], np.asarray(seq.is_rf, np.int8), [0]])))
    starts, ends = edges[::2], edges[1::2]
    rf = seq.rf.detach().cpu().double().numpy()
    b1 = rf[:, 0] + 1j * rf[:, 1]
    centres = 0.5 * (tb[starts] + tb[ends])
    k_after = np.full((len(starts), 3), np.nan)  # k at each pulse centre, just after the pulse
    k, prev = None, None
    for n, (i, j) in enumerate(zip(starts, ends)):
        if flip_angle(b1[i:j], dt[i:j]) < 90.01:
            k = np.zeros(3)
        elif k is not None:
            k = -(k + at(centres[n]) - at(prev))
        prev = centres[n]
        if k is not None:
            k_after[n] = k
    t = tb[1:][np.asarray(seq.adc, bool)]
    last = np.searchsorted(centres, t, side="right") - 1
    out = np.full((len(t), 3), np.nan)
    ok = last >= 0
    out[ok] = k_after[last[ok]] + at(t[ok]) - at(centres[last[ok]])
    return torch.from_numpy(out)


# --- hardware limits and events --------------------------------------------------------


@dataclass(frozen=True)
class System:
    """Hardware limits and timing rasters, like pypulseq's ``Opts`` but in blochparty units.
    Defaults are typical 3 T values: 40 mT/m, 170 T/m/s, Siemens-like rasters and RF dead
    times. Limits apply to each gradient axis separately."""

    max_grad: float = 40e-3 * PULSEQ_GAMMA_BAR  # [Hz/m]
    max_slew: float = 170.0 * PULSEQ_GAMMA_BAR  # [Hz/m/s]
    grad_raster: float = 10e-6  # [s]
    rf_raster: float = 1e-6  # [s]
    adc_raster: float = 100e-9  # [s] dwell times must be multiples of this
    rf_dead_time: float = 100e-6  # [s] from block start to the earliest RF sample
    rf_ringdown_time: float = 30e-6  # [s] after the last RF sample, before the block may end

    def ramp_time(self, amplitude) -> float:
        """Shortest ramp (on the gradient raster) from 0 to ``amplitude`` [Hz/m]."""
        return _ceil_to(abs(float(amplitude)) / self.max_slew, self.grad_raster)

    def check_raster(self, t: float, raster: float, what: str) -> None:
        if abs(t / raster - round(t / raster)) > 1e-6:
            raise ValueError(f"{what} = {t:.9g} s is not a multiple of the raster {raster:.9g} s")


def _ceil_to(t: float, raster: float) -> float:
    return math.ceil(t / raster - 1e-9) * raster


def _vector(value, axis, dtype) -> Tensor:
    """(3,) tensor from a scalar on ``axis`` ("x", "y", "z") or from a 3-vector (axis None)."""
    if axis is None:
        if isinstance(value, Tensor):
            return value.to(dtype)
        return torch.stack([torch.as_tensor(v, dtype=dtype) for v in value])
    unit = torch.zeros(3, dtype=dtype)
    unit[_AXES[axis]] = 1.0
    return torch.as_tensor(value, dtype=dtype) * unit


@dataclass
class Trapezoid:
    """Trapezoidal gradient ``amplitude`` (3,) [Hz/m] on any combination of axes: ramp up
    over ``rise``, constant for ``flat``, ramp down over ``fall`` [s], after ``delay``.
    Make one with ``trapezoid()``, which applies the system limits; a trapezoid with zero
    ramps is a rectangle (the ideal, instantaneous gradients of the simple builders)."""

    amplitude: Tensor
    rise: float
    flat: float
    fall: float
    delay: float = 0.0

    @property
    def duration(self) -> float:
        return self.delay + self.rise + self.flat + self.fall

    @property
    def area(self) -> Tensor:
        return self.amplitude * (self.flat + 0.5 * (self.rise + self.fall))

    def corners(self) -> tuple[np.ndarray, Tensor]:
        """(times from the block start, (m, 3) values) of the piecewise-linear waveform."""
        t = self.delay + np.cumsum([0.0, self.rise, self.flat, self.fall])
        shape = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=self.amplitude.dtype)
        return t, shape[:, None] * self.amplitude[None, :]


@dataclass
class GradientWaveform:
    """Piecewise-linear gradient through ``values`` (m, 3) [Hz/m] at ``times`` (m,) [s]
    from the block start; zero outside. Make one with ``gradient_waveform()``."""

    times: np.ndarray
    values: Tensor

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    @property
    def area(self) -> Tensor:
        dt = torch.as_tensor(np.diff(self.times), dtype=self.values.dtype)
        return (0.5 * (self.values[1:] + self.values[:-1]) * dt[:, None]).sum(0)

    def corners(self) -> tuple[np.ndarray, Tensor]:
        return self.times, self.values


@dataclass
class RFPulse:
    """RF ``signal`` (k, 2) [Hz], piecewise constant on steps of ``raster`` [s], starting
    ``delay`` after the block start. ``center`` is the time from the pulse start to its
    effective centre (the reference for TE and for slice rephasing)."""

    signal: Tensor
    raster: float
    delay: float
    center: float

    @property
    def duration(self) -> float:
        return self.signal.shape[0] * self.raster


@dataclass
class ADC:
    """``n`` samples at ``delay + (i + 1/2) * dwell`` after the block start (Pulseq
    convention); ``phase`` [rad] is the receiver phase (scalar or (n,))."""

    n: int
    dwell: float
    delay: float = 0.0
    phase: float | Tensor = 0.0

    @property
    def times(self) -> np.ndarray:
        return self.delay + (np.arange(self.n) + 0.5) * self.dwell

    @property
    def duration(self) -> float:
        return self.delay + self.n * self.dwell


def trapezoid(axis: str | None = None, *, area=None, amplitude=None, flat_area=None, flat_time=None,
              duration=None, rise_time=None, delay: float = 0.0, system: System | None = None,
              dtype=torch.float64) -> Trapezoid:
    """Trapezoid on ``axis`` ("x", "y", "z"), or along a direction when the given area /
    amplitude is a 3-vector (``axis=None``). Ramps are the shortest the slew rate allows,
    on the gradient raster; with several axes the largest component sets the timing. Give

    * ``area`` alone: the shortest trapezoid (a triangle if it fits),
    * ``area`` and ``duration``: the lowest amplitude lasting exactly ``duration``,
    * ``flat_area`` and ``flat_time``: e.g. a readout (the ramps add their own area),
    * ``amplitude`` and ``flat_time`` (or ``duration``).

    ``rise_time`` overrides the ramp length. Areas and amplitudes may be tensors that
    require grad (timings are fixed from their values). Raises if the result exceeds the
    system's gradient or slew limits."""
    sysm = system or System()
    tensor = lambda v: _vector(v, axis, dtype)
    if amplitude is not None:
        if flat_time is None and duration is None:
            raise ValueError("amplitude needs flat_time or duration")
        amp = tensor(amplitude)
        ramp = rise_time if rise_time is not None else sysm.ramp_time(amp.abs().max())
        flat = flat_time if flat_time is not None else duration - 2 * ramp
    elif flat_area is not None:
        if flat_time is None:
            raise ValueError("flat_area needs flat_time")
        amp = tensor(flat_area) / flat_time
        ramp = rise_time if rise_time is not None else sysm.ramp_time(amp.abs().max())
        flat = flat_time
    elif area is not None:
        a_vec = tensor(area)
        a = float(a_vec.abs().max())
        if duration is not None:
            if rise_time is not None:
                ramp = rise_time
            else:
                disc = duration**2 - 4 * a / sysm.max_slew
                if disc < 0:
                    raise ValueError(f"area {a:.6g} /m does not fit in {duration * 1e3:.4g} ms at the slew limit")
                ramp = _ceil_to(0.5 * (duration - math.sqrt(disc)), sysm.grad_raster) if a > 0 else 0.0
            flat = duration - 2 * ramp
        elif a <= sysm.max_grad**2 / sysm.max_slew:  # triangle
            ramp = rise_time if rise_time is not None else _ceil_to(math.sqrt(a / sysm.max_slew), sysm.grad_raster)
            flat = 0.0
        else:
            ramp = rise_time if rise_time is not None else sysm.ramp_time(sysm.max_grad)
            flat = _ceil_to(a / sysm.max_grad - ramp, sysm.grad_raster)
        if flat < -_T_TOL:
            raise ValueError(f"ramps of {ramp * 1e6:.4g} us do not fit in {duration * 1e3:.4g} ms")
        flat = max(flat, 0.0)
        amp = a_vec / (flat + ramp) if flat + ramp > 0 else a_vec * 0.0
    else:
        raise ValueError("give area, flat_area or amplitude")
    if flat < -_T_TOL:
        raise ValueError(f"negative flat time {flat:.3g} s")
    g = Trapezoid(amp, float(ramp), max(float(flat), 0.0), float(ramp), float(delay))
    _check_limits(g, sysm)
    return g


def _check_limits(g, sysm: System) -> None:
    """Raise if any axis of a gradient event exceeds the system's amplitude or slew limit."""
    t, v = g.corners()
    v = v.detach()
    peak = float(v.abs().max()) if v.numel() else 0.0
    if peak > sysm.max_grad * (1 + 1e-9):
        raise ValueError(f"gradient {peak:.6g} Hz/m exceeds max_grad {sysm.max_grad:.6g} Hz/m "
                         f"by {100 * (peak / sysm.max_grad - 1):.3g} %")
    dt = torch.as_tensor(np.diff(t), dtype=v.dtype)
    dv = (v[1:] - v[:-1]).abs().max(dim=1).values
    jump = (dt <= _T_TOL) & (dv > 0)
    if bool(jump.any()):
        raise ValueError("gradient jumps instantaneously (infinite slew rate)")
    slew = float((dv / dt.clamp_min(_T_TOL)).max()) if dt.numel() else 0.0
    if slew > sysm.max_slew * (1 + 1e-9):
        raise ValueError(f"slew rate {slew:.6g} Hz/m/s exceeds max_slew {sysm.max_slew:.6g} Hz/m/s "
                         f"by {100 * (slew / sysm.max_slew - 1):.3g} %")


def gradient_waveform(samples, raster: float | None = None, *, delay: float = 0.0,
                      system: System | None = None, dtype=torch.float64) -> GradientWaveform:
    """Arbitrary gradient: ``samples`` (k, 3) [Hz/m] are the values at ``delay + j * raster``
    (default: the gradient raster), joined linearly, so each raster interval carries the
    exact average (g_j + g_{j+1}) / 2. Start and end at zero unless the neighbouring
    blocks continue the waveform. Raises if a sample exceeds ``max_grad`` or a step
    exceeds ``max_slew``."""
    sysm = system or System()
    raster = raster or sysm.grad_raster
    vals = torch.as_tensor(samples, dtype=dtype) if not isinstance(samples, Tensor) else samples.to(dtype)
    if vals.ndim != 2 or vals.shape[1] != 3:
        raise ValueError(f"samples must be (k, 3), got {tuple(vals.shape)}")
    g = GradientWaveform(delay + raster * np.arange(vals.shape[0]), vals)
    _check_limits(g, sysm)
    return g


MAX_RF_STEP = 20e-6  # [s] hard pulses are cut into segments no longer than this


def _rf_steps(duration) -> int:
    """Segments for a hard pulse: relaxation is split around each segment's rotation, so one
    segment for a long large-flip pulse misses the decay while M is transverse (~T_RF / 2T2 for a
    180: 0.5 % at 912 us and T2 = 75 ms, ``validation/real/asslaender_ir.py``). The count depends
    on the duration only, so outputs stay smooth in a differentiable flip angle."""
    return max(1, math.ceil(float(torch.as_tensor(duration).detach()) / MAX_RF_STEP - 1e-6))


def block_pulse(flip_deg, duration: float, *, phase=0.0, system: System | None = None,
                dtype=torch.float64, n_steps: int | None = None) -> RFPulse:
    """Hard (rectangular) pulse after the system's RF dead time; flip and phase [rad] may be
    tensors. ``n_steps`` segments (default: <= ``MAX_RF_STEP`` long each)."""
    sysm = system or System()
    n = _rf_steps(duration) if n_steps is None else n_steps
    amp = torch.as_tensor(flip_deg, dtype=dtype) / 360.0 / duration
    ph = torch.as_tensor(phase, dtype=dtype)
    signal = torch.stack([amp * torch.cos(ph), amp * torch.sin(ph)])[None].expand(n, 2)
    return RFPulse(signal, float(duration) / n, sysm.rf_dead_time, 0.5 * duration)


def sinc_pulse(flip_deg, duration: float, *, slice_thickness: float, time_bw_product: float = 4.0,
               apodization: float = 0.5, slice_offset: float = 0.0, phase=0.0, system: System | None = None,
               dtype=torch.float64) -> tuple[RFPulse, Trapezoid, Tensor]:
    """Windowed-sinc slice-selective pulse: (rf, slice-select gradient, rephase area (3,)).

    The envelope is ((1 - a) + a cos(2 pi t / T)) sinc(BW t) on |t| < T/2 with
    BW = ``time_bw_product`` / T, sampled on the RF raster and scaled so the flip angle at
    the slice centre is exactly ``flip_deg``. The slice-select gradient on z has amplitude
    BW / ``slice_thickness`` [Hz/m]; the pulse sits in the middle of its flat top, after the
    RF dead time. ``slice_offset`` [m] is a frequency offset f = G z0: spins at z0 precess as
    exp(-i 2 pi f t), so the pulse turns the same way, exp(-i 2 pi f (t - t_centre)), with
    zero phase at its centre. The rephase area undoes the gradient moment from the pulse
    centre to the end of the fall ramp (use it for a separate or merged rephasing lobe;
    refocusing pulses need none).
    """
    sysm = system or System()
    sysm.check_raster(duration, sysm.rf_raster, "RF duration")
    n = round(duration / sysm.rf_raster)
    t = (np.arange(n) + 0.5) * sysm.rf_raster - 0.5 * duration  # from the pulse centre
    bw = time_bw_product / duration
    env = ((1 - apodization) + apodization * np.cos(2 * np.pi * t / duration)) * np.sinc(bw * t)
    env = torch.as_tensor(env / (env.sum() * sysm.rf_raster), dtype=dtype)  # unit area: 1 cycle
    amp = torch.as_tensor(flip_deg, dtype=dtype) / 360.0 * env
    g_amp = bw / slice_thickness
    ph = torch.as_tensor(phase, dtype=dtype) - torch.as_tensor(2 * np.pi * g_amp * slice_offset * t, dtype=dtype)
    signal = torch.stack([amp * torch.cos(ph), amp * torch.sin(ph)], dim=-1)

    ramp = sysm.ramp_time(g_amp)
    flat = _ceil_to(duration, sysm.grad_raster)
    margin = 0.5 * (flat - duration)  # pulse centred on the flat top
    g_delay = max(0.0, sysm.rf_dead_time - ramp - margin)
    gz = trapezoid("z", amplitude=g_amp, flat_time=flat, rise_time=ramp, delay=g_delay, system=sysm, dtype=dtype)
    rf = RFPulse(signal, sysm.rf_raster, g_delay + ramp + margin, 0.5 * duration)
    rephase = -gz.amplitude * (0.5 * flat + 0.5 * ramp)  # moment from the centre to the end
    return rf, gz, rephase


def block_duration(*grads, rf: RFPulse | None = None, adc: ADC | list[ADC] | None = None,
                   system: System | None = None) -> float:
    """How long ``SequenceBuilder.block`` makes a block of these events: until the last
    gradient or ADC sample ends, or the RF pulse plus the ringdown time."""
    adcs = [] if adc is None else ([adc] if isinstance(adc, ADC) else list(adc))
    ends = [0.0] + [g.duration for g in grads] + [a.duration for a in adcs]
    if rf is not None:
        ends.append(rf.delay + rf.duration + (system or System()).rf_ringdown_time)
    return max(ends)


def _interp(mid: np.ndarray, t: np.ndarray, v: Tensor) -> Tensor:
    """Piecewise-linear (t, v) at the points ``mid`` (none of which is a corner); zero outside."""
    j = np.searchsorted(t, mid, side="right") - 1
    valid = (j >= 0) & (j < len(t) - 1)
    j = np.clip(j, 0, len(t) - 2)
    w = (mid - t[j]) / np.where(valid, t[j + 1] - t[j], 1.0)
    w_t = torch.as_tensor(np.where(valid, w, 0.0), dtype=v.dtype, device=v.device)[:, None]
    on = torch.as_tensor(valid, dtype=v.dtype, device=v.device)[:, None]
    jt = torch.as_tensor(j, device=v.device)
    return (v[jt] * (1 - w_t) + v[jt + 1] * w_t) * on


def _subdivide(t: np.ndarray, sloped: np.ndarray, step: float) -> np.ndarray:
    """Extra points cutting each sloped interval of ``t`` into equal pieces of at most ``step``."""
    out = [np.zeros(0)]
    for a, b in zip(t[:-1][sloped], t[1:][sloped]):
        n = math.ceil((b - a) / step - 1e-9)
        if n > 1:
            out.append(a + (b - a) * np.arange(1, n) / n)
    return np.concatenate(out)


class SequenceBuilder:
    """Appends segments block by block; every numeric argument may be a tensor
    that requires grad (e.g. a flip angle or TR being optimized)."""

    def __init__(self, dtype=torch.float64, device=None, system: System | None = None):
        self.dtype, self.device = dtype, device
        self.system = system or System()  # limits for trapezoid / gradient_waveform / block
        self.time = 0.0  # [s] duration so far
        self._dt, self._rf, self._grad, self._phase = [], [], [], []
        self._is_rf, self._adc, self._spoil = [], [], []
        self._blocks = []  # provenance: one record per call, with its segment span (Sequence.meta["blocks"])

    def _record(self, start: int, kind: str, **events):
        """Remember a call's events and the segments it produced (``pulseq.write_pulseq`` replays them)."""
        self._blocks.append({"kind": kind, "span": (start, len(self._is_rf)), **events})

    def _t(self, x) -> Tensor:
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)

    def _grad3(self, grad, n: int) -> Tensor:
        """Gradient as (n, 3): accepts None, a (3,) / (n, 3) tensor, or a 3-sequence
        whose entries may themselves be (differentiable) scalars."""
        if grad is None:
            g = torch.zeros(3, dtype=self.dtype, device=self.device)
        elif isinstance(grad, Tensor):
            g = grad.to(dtype=self.dtype, device=self.device)
        else:
            g = torch.stack([self._t(x) for x in grad])
        return g if g.ndim == 2 else g.expand(n, 3)

    def _append(self, dt, rf, grad, phase, *, is_rf, adc):
        n = dt.shape[0]
        self._dt.append(dt)
        self._rf.append(rf)
        self._grad.append(grad)
        self._phase.append(phase)
        self._is_rf += is_rf if isinstance(is_rf, list) else [is_rf] * n
        self._adc += adc if isinstance(adc, list) else [adc] * n
        self._spoil += [False] * n
        self.time += float(dt.detach().sum())

    def rf(self, duration, *, flip_deg=None, amplitude=None, phase=0.0, grad=None, n_steps=None):
        """Rectangular (hard) pulse; give either ``flip_deg`` or ``amplitude`` [Hz], in ``n_steps``
        segments (default: <= ``MAX_RF_STEP`` long each). Relaxation is split around
        each segment's whole rotation, so one segment for a large flip misses the T2 decay while M
        is transverse (~T_RF / (2 T2) for a 180: 0.5 % at 912 us and T2 = 75 ms,
        ``validation/real/asslaender_ir.py``)."""
        dur = self._t(duration)
        amp = self._t(flip_deg) / 360.0 / dur if flip_deg is not None else self._t(amplitude)
        if n_steps is None:
            n_steps = _rf_steps(dur)
        ph = self._t(phase)
        rf = torch.stack([amp * torch.cos(ph), amp * torch.sin(ph)]).expand(n_steps, 2)
        dt = (dur / n_steps).expand(n_steps)
        start = len(self._is_rf)
        self._append(dt, rf, self._grad3(grad, n_steps), torch.zeros_like(dt), is_rf=True, adc=False)
        self._record(start, "rf", duration=duration, flip_deg=flip_deg, amplitude=amplitude, phase=phase, grad=grad,
                     n_steps=n_steps)
        return self

    def rf_waveform(self, samples, dt, grad=None):
        """Arbitrary RF: ``samples`` is (k, 2) in Hz, one per raster step of ``dt``."""
        rf = self._t(samples)
        n = rf.shape[0]
        dts = self._t(dt).expand(n)
        start = len(self._is_rf)
        self._append(dts, rf, self._grad3(grad, n), torch.zeros_like(dts), is_rf=True, adc=False)
        self._record(start, "rf_waveform", samples=samples, dt=dt, grad=grad)
        return self

    def delay(self, duration, grad=None):
        dt = self._t(duration).reshape(1)
        start = len(self._is_rf)
        self._append(dt, torch.zeros(1, 2, dtype=self.dtype, device=self.device),
                     self._grad3(grad, 1), torch.zeros_like(dt), is_rf=False, adc=False)
        self._record(start, "delay", duration=duration, grad=grad)
        return self

    def readout(self, n: int, dwell, grad=None, phase=0.0):
        """``n`` ADC samples at (i + 0.5) * dwell after the block start; the block
        lasts n * dwell, matching the Pulseq ADC convention."""
        dw = self._t(dwell)
        dt = torch.cat([(0.5 * dw).reshape(1), dw.expand(n - 1), (0.5 * dw).reshape(1)])
        rf = torch.zeros(n + 1, 2, dtype=self.dtype, device=self.device)
        ph = self._t(phase).expand(n + 1)
        start = len(self._is_rf)
        self._append(dt, rf, self._grad3(grad, n + 1), ph, is_rf=False, adc=[True] * n + [False])
        self._record(start, "readout", n=n, dwell=dwell, grad=grad, phase=phase)
        return self

    def block(self, *grads, rf: RFPulse | None = None, adc: ADC | list[ADC] | None = None,
              duration: float | None = None):
        """Events played together, like a Pulseq block: any number of gradients
        (``Trapezoid`` / ``GradientWaveform``, summed where they overlap), at most one RF
        pulse and any number of ADC events. The block lasts until its last event ends (RF
        plus the ringdown time), or ``duration``. It is split at every event boundary;
        each segment carries the exact average gradient, so moments are exact at every
        segment end, and sloped pieces are cut on the gradient raster."""
        sysm = self.system
        adcs = [] if adc is None else ([adc] if isinstance(adc, ADC) else list(adc))
        if rf is not None and rf.delay < sysm.rf_dead_time - _T_TOL:
            raise ValueError(f"RF starts {rf.delay * 1e6:.4g} us into the block, before the dead time "
                             f"({sysm.rf_dead_time * 1e6:.4g} us)")
        end = block_duration(*grads, rf=rf, adc=adcs, system=sysm)
        if duration is not None:
            if duration < end - _T_TOL:
                raise ValueError(f"block events last {end:.9g} s, longer than duration {duration:.9g} s")
            end = float(duration)
        if end <= _T_TOL:
            return self

        points, shapes = [np.array([0.0, end])], []
        for g in grads:
            t, v = g.corners()
            shapes.append((t, v.to(dtype=self.dtype, device=self.device)))
            vd = v.detach()
            sloped = (vd[1:] != vd[:-1]).any(dim=1).numpy() & (np.diff(t) > _T_TOL)
            points += [t, _subdivide(t, sloped, sysm.grad_raster)]
        if rf is not None:
            points.append(rf.delay + rf.raster * np.arange(rf.signal.shape[0] + 1))
        for a in adcs:
            points.append(a.times)
        t = np.sort(np.concatenate(points))
        t = t[np.concatenate([[True], np.diff(t) > _T_TOL])]
        if t[-1] > end + _T_TOL or t[0] < -_T_TOL:
            raise ValueError(f"events extend outside the block [0, {end:.9g}] s")
        t = np.append(t[t < end - _T_TOL], end)
        mid, n = 0.5 * (t[1:] + t[:-1]), len(t) - 1

        g = torch.zeros(n, 3, dtype=self.dtype, device=self.device)
        for tc, v in shapes:  # every corner is a breakpoint: midpoint value = exact average
            g = g + _interp(mid, tc, v)
        rf_seg = torch.zeros(n, 2, dtype=self.dtype, device=self.device)
        is_rf = np.zeros(n, dtype=bool)
        if rf is not None:
            is_rf = (mid > rf.delay) & (mid < rf.delay + rf.duration)
            k = np.clip(np.floor((mid - rf.delay) / rf.raster).astype(int), 0, rf.signal.shape[0] - 1)
            on = torch.as_tensor(is_rf, dtype=self.dtype, device=self.device)[:, None]
            signal = rf.signal.to(dtype=self.dtype, device=self.device)
            rf_seg = signal[torch.as_tensor(k, device=self.device)] * on
        is_adc = np.zeros(n, dtype=bool)
        phase = torch.zeros(n, dtype=self.dtype, device=self.device)
        for a in adcs:
            ta = a.times
            seg = np.searchsorted(t, ta - _T_TOL) - 1  # each sample ends a segment
            if np.any(seg < 0) or np.any(np.abs(t[seg + 1] - ta) > _T_TOL) or np.any(is_adc[seg]):
                raise ValueError("ADC samples must lie inside the block, after its start, without overlap")
            is_adc[seg] = True
            ph = torch.as_tensor(a.phase, dtype=self.dtype, device=self.device).expand(a.n)
            phase = phase.index_put((torch.as_tensor(seg, device=self.device),), ph)
        start = len(self._is_rf)
        self._append(self._t(np.diff(t)), rf_seg, g, phase, is_rf=is_rf.tolist(), adc=is_adc.tolist())
        self._record(start, "block", grads=list(grads), rf=rf, adc=adcs, duration=duration)
        return self

    def trapezoid(self, axis: str | None = None, **kwargs):
        """Append a trapezoid as its own block; arguments as for ``trapezoid()`` (the
        builder's system is used)."""
        return self.block(trapezoid(axis, system=self.system, dtype=self.dtype, **kwargs))

    def gradient_waveform(self, samples, raster: float | None = None):
        """Append an arbitrary gradient as its own block; see ``gradient_waveform()``."""
        return self.block(gradient_waveform(samples, raster, system=self.system, dtype=self.dtype))

    def sinc_pulse(self, flip_deg, duration: float, *, slice_thickness: float, rephase: bool = True, **kwargs):
        """Append a slice-selective windowed-sinc excitation (pulse + slice-select gradient)
        and, with ``rephase``, its shortest rephasing lobe; options as for ``sinc_pulse()``."""
        rf, gz, reph = sinc_pulse(flip_deg, duration, slice_thickness=slice_thickness, system=self.system,
                                  dtype=self.dtype, **kwargs)
        self.block(gz, rf=rf)
        return self.block(trapezoid(area=reph, system=self.system, dtype=self.dtype)) if rephase else self

    def spoil(self):
        """Ideal spoiler: transverse magnetization is zeroed at the end of the last segment."""
        self._spoil[-1] = True
        return self

    def build(self, **meta) -> Sequence:
        """The segments. ``meta["blocks"]`` records every call's events and segment span (block
        provenance, which ``pulseq.write_pulseq`` writes from); other ``meta`` entries are kept."""
        meta = {"blocks": [dict(b) for b in self._blocks], "system": self.system, **meta}
        return Sequence(
            dt=torch.cat(self._dt),
            rf=torch.cat(self._rf),
            grad=torch.cat(self._grad),
            adc_phase=torch.cat(self._phase),
            is_rf=list(self._is_rf),
            adc=list(self._adc),
            spoil=list(self._spoil),
            meta=meta,
        )
