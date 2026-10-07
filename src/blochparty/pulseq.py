"""Pulseq import: ``.seq`` file or ``pypulseq.Sequence`` -> ``Sequence``.

Every block is split at all of its event boundaries: RF raster steps (or the
points of a time-shaped pulse), the RF centre, gradient corner points, ADC sample
times and the block end. Each ADC sample therefore lands exactly on a segment end.
Gradients are piecewise linear in Pulseq and every corner is a breakpoint, so the
midpoint value of a segment is its exact average gradient: gradient moments, and so
k-space, are exact at every segment end. Ramps (and other sloped pieces) are further
cut at ``ramp_raster`` so that spins see a staircase close to the true ramp.

Conventions (Pulseq -> blochparty), checked in ``tests/test_pulseq.py``:

* Units already match: RF in Hz, gradients in Hz/m, time in s.
* Pulseq fixes the sign of a frequency offset physically: a pulse with frequency
  offset f excites the spins whose Larmor frequency is higher by f (slice position
  = f / G, as in ``pypulseq``'s ``calculate_kspace``). Those spins precess as
  exp(-i 2 pi f t) in blochparty, so the pulse must turn the same way. With the
  Pulseq RF phase defined as phase(t) = phase_offset + 2 pi f t (from the start of
  the shape, as in the Pulseq specification), the field is the complex conjugate
  of the Pulseq waveform: ``rf_x + i rf_y = conj(signal * exp(i phase(t)))``.
  So a real pulse (phase 0) still tips +z to +y (Mxy = +i), while Pulseq phase phi
  gives Mxy = i exp(-i phi) sin(alpha).
* ADC: the receiver phase is psi = -(phase_offset + 2 pi f t_i + modulation_i), with
  t_i = (i + 1/2) dwell from the ADC start. RF and ADC phases written with the same
  value cancel (RF spoiling) and an ADC frequency offset follows the spins at +f.
* ``convention="mrzero"`` uses the Pulseq phases as written instead (field
  ``signal * exp(i phase(t))``, receiver phase +phase): the reading of MRzero, whose
  PDG simulation then equals blochparty's up to the constant factor -i (its signal
  carries an extra +pi/2 receiver phase and it rotates the RF the other way).
  The two readings are mirror images of each other: conj(signal) on the spatially
  inverted phantom with negated off-resonance. Frequency offsets then excite -f.

Label, trigger and soft-delay extensions are ignored (soft delays at their default
duration). Anything else (unknown extensions such as rotations, RF shims) raises.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .sequence import MAX_RF_STEP, Sequence

_T_TOL = 1e-11  # [s] breakpoints closer than this are merged (Pulseq rasters are >= 1e-8 s)
_KNOWN_EXTENSIONS = {"LABELSET", "LABELINC", "TRIGGERS", "DELAYS"}
_KNOWN_BLOCK_FIELDS = {"block_duration", "rf", "gx", "gy", "gz", "adc", "label", "soft_delay",
                       "trigger", "delay"}


def _load(seq_or_path):
    try:
        import pypulseq as pp
    except ImportError as e:  # pragma: no cover
        raise ImportError("read_pulseq needs pypulseq: pip install 'blochparty[pulseq]'") from e
    if isinstance(seq_or_path, (str, Path)):
        seq = pp.Sequence()
        seq.read(str(seq_or_path))
        return seq
    return seq_or_path


def _subdivide(t: np.ndarray, sloped: np.ndarray, step: float) -> np.ndarray:
    """Points ``t`` plus extra points cutting each sloped interval into pieces <= step."""
    out = [t]
    for a, b in zip(t[:-1][sloped], t[1:][sloped]):
        n = math.ceil((b - a) / step - 1e-9)
        if n > 1:
            out.append(a + (b - a) * np.arange(1, n) / n)
    return np.concatenate(out)


def _grad_shape(g, raster: float) -> tuple[np.ndarray, np.ndarray]:
    """(times, values) of a gradient event as a piecewise-linear waveform from the block start."""
    if g.type == "trap":
        t = np.cumsum([0.0, g.rise_time, g.flat_time, g.fall_time])
        v = g.amplitude * np.array([0.0, 1.0, 1.0, 0.0])
    elif g.time_id in (0, -1):  # samples at raster-cell centres, plus first/last at the edges
        t = np.concatenate([[0.0], g.tt, [g.tt[-1] + 0.5 * raster]])
        v = np.concatenate([[g.first], g.waveform, [g.last]])
    else:  # extended trapezoid / explicit time shape
        t, v = np.asarray(g.tt, dtype=float), np.asarray(g.waveform, dtype=float)
    keep = np.concatenate([[True], np.diff(t) > _T_TOL])  # drop zero-length pieces
    return g.delay + t[keep], v[keep]


def _check_extensions(seq) -> None:
    unknown = set(getattr(seq, "extension_string_idx", [])) - _KNOWN_EXTENSIONS
    if unknown:
        raise NotImplementedError(f"unsupported Pulseq extension(s): {sorted(unknown)}")


def read_pulseq(seq_or_path, *, dtype=torch.float64, device=None, ramp_raster: float | None = None,
                rf_raster: float | None = None, max_rf_step: float | None = None, convention: str = "physical",
                **meta) -> Sequence:
    """Convert a Pulseq sequence (``.seq`` path or ``pypulseq.Sequence``) to a ``Sequence``.

    ramp_raster: longest piece a sloped gradient section is cut into (default: the
        gradient raster time). Moments are exact regardless; this sets how finely
        spins see the ramp shape.
    rf_raster: same for time-shaped RF pulses that vary or carry a frequency offset
        (default: the RF raster time). Pulses on the default RF raster are always
        stepped sample by sample.
    max_rf_step: also cut constant RF pieces (block pulses without a frequency offset)
        into pieces <= this [s] (default ``sequence.MAX_RF_STEP``, 20 us). Relaxation is
        split around each piece's rotation, so one piece misses the decay while M is
        transverse (a 1 s on-resonance block pulse: errors of 0.6 in Mz; a 1 ms 180:
        ~T_RF / 2T2).
    convention: "physical" (default) or "mrzero"; see the module docstring.
    meta: extra entries for ``Sequence.meta`` (e.g. ``kspace_shape``).
    """
    if convention not in ("physical", "mrzero"):
        raise ValueError(f"convention must be 'physical' or 'mrzero', not {convention!r}")
    sign = -1.0 if convention == "physical" else 1.0  # phase sign; -1 conjugates
    seq = _load(seq_or_path)
    _check_extensions(seq)
    sysd = seq.system
    grad_raster, rf_raster_def = sysd.grad_raster_time, sysd.rf_raster_time
    ramp_raster = ramp_raster or grad_raster
    rf_raster = rf_raster or rf_raster_def
    f0_mhz = sysd.gamma * sysd.B0 * 1e-6  # system frequency [MHz], weights the ppm offsets

    dts, grads, rfs, rf_flags, adc_flags, adc_phases = [], [], [], [], [], []
    adc_counts, block_segments = [], []
    n_seg = 0
    for idx in seq.block_events:
        block = seq.get_block(idx)
        extra = {k for k, v in vars(block).items() if v is not None} - _KNOWN_BLOCK_FIELDS
        if extra:
            raise NotImplementedError(f"block {idx}: unsupported event(s) {sorted(extra)}")
        dur = float(seq.block_durations[idx])
        if dur <= 0:
            block_segments.append((n_seg, n_seg))
            continue
        points = [np.array([0.0, dur])]

        shapes = []
        for ch in ("gx", "gy", "gz"):
            g = getattr(block, ch)
            if g is None:
                shapes.append(None)
                continue
            t, v = _grad_shape(g, grad_raster)
            shapes.append((t, v))
            points.append(_subdivide(t, np.diff(v) != 0, ramp_raster))

        rf = block.rf
        if rf is not None:
            f_rf = rf.freq_offset + rf.freq_ppm * f0_mhz
            ph_rf = rf.phase_offset + rf.phase_ppm * f0_mhz
            sig = np.asarray(rf.signal, dtype=complex)
            t_rf = np.asarray(rf.t, dtype=float)
            on_raster = np.allclose(t_rf, (np.arange(len(t_rf)) + 0.5) * rf_raster_def, atol=_T_TOL, rtol=0)
            if on_raster:  # one constant step per raster cell
                edges = np.arange(len(t_rf) + 1) * rf_raster_def
            else:  # piecewise linear between (t, signal) points
                sloped = (np.diff(sig) != 0) | (f_rf != 0)
                edges = np.sort(_subdivide(t_rf, sloped, rf_raster))
                edges = _subdivide(edges, np.ones(len(edges) - 1, dtype=bool), max_rf_step or MAX_RF_STEP)
            points += [rf.delay + edges, np.array([rf.delay + rf.center])]
            rf_lo, rf_hi = rf.delay + edges.min(), rf.delay + edges.max()

        adc = block.adc
        if adc is not None:
            t_adc = adc.delay + (np.arange(adc.num_samples) + 0.5) * adc.dwell
            points.append(t_adc)

        t = np.sort(np.concatenate(points))
        t = t[np.concatenate([[True], np.diff(t) > _T_TOL])]
        if t[-1] > dur + _T_TOL:
            raise ValueError(f"block {idx}: events last {t[-1]:.9g} s, longer than the block ({dur:.9g} s)")
        t = t[t < dur - _T_TOL]
        t = np.append(t, dur)
        if t[0] > _T_TOL:  # the first point is the block start
            t = np.concatenate([[0.0], t])
        mid, n = 0.5 * (t[1:] + t[:-1]), len(t) - 1

        g = np.zeros((n, 3))
        for j, shape in enumerate(shapes):
            if shape is not None:  # every corner is a breakpoint: midpoint value = exact average
                g[:, j] = np.interp(mid, *shape, left=0.0, right=0.0)

        b1 = np.zeros(n, dtype=complex)
        is_rf = np.zeros(n, dtype=bool)
        if rf is not None:
            is_rf = (mid > rf_lo) & (mid < rf_hi)
            tr = mid[is_rf] - rf.delay  # time from the start of the RF shape
            if on_raster:
                k = np.clip(np.floor(tr / rf_raster_def).astype(int), 0, len(sig) - 1)
                ts, vals = t_rf[k], sig[k]  # sample value and its time (for the phase ramp)
            else:
                ts = tr
                vals = np.interp(tr, t_rf, sig.real) + 1j * np.interp(tr, t_rf, sig.imag)
            b1[is_rf] = vals * np.exp(1j * (ph_rf + 2 * math.pi * f_rf * ts))
        if sign < 0:
            b1 = b1.conj()

        is_adc = np.zeros(n, dtype=bool)
        psi = np.zeros(n)
        if adc is not None:
            ends = np.searchsorted(t, t_adc - _T_TOL)  # index of the breakpoint at each sample
            assert np.all(np.abs(t[ends] - t_adc) <= _T_TOL)
            seg = ends - 1
            f_adc = adc.freq_offset + adc.freq_ppm * f0_mhz
            ph = adc.phase_offset + adc.phase_ppm * f0_mhz + 2 * math.pi * f_adc * (t_adc - adc.delay)
            mod = getattr(adc, "phase_modulation", None)
            if mod is not None and len(mod):
                ph = ph + np.asarray(mod, dtype=float)
            is_adc[seg] = True
            psi[seg] = sign * ph
            adc_counts.append(adc.num_samples)

        dts.append(np.diff(t))
        grads.append(g)
        rfs.append(b1)
        rf_flags.append(is_rf)
        adc_flags.append(is_adc)
        adc_phases.append(psi)
        block_segments.append((n_seg, n_seg + n))
        n_seg += n

    tensor = lambda x: torch.as_tensor(np.concatenate(x), dtype=dtype, device=device)
    rf_all = np.concatenate(rfs)
    out = Sequence(
        dt=tensor(dts),
        rf=torch.as_tensor(np.stack([rf_all.real, rf_all.imag], axis=-1), dtype=dtype, device=device),
        grad=tensor(grads),
        adc_phase=tensor(adc_phases),
        is_rf=np.concatenate(rf_flags).tolist(),
        adc=np.concatenate(adc_flags).tolist(),
        spoil=[False] * n_seg,
        meta={"adc_per_readout": adc_counts, "block_segments": block_segments,
              "definitions": dict(getattr(seq, "definitions", {})), "convention": convention},
    )
    try:  # k-space trajectory at the ADC samples, from pypulseq itself
        k_adc = seq.calculate_kspace()[0]
        out.meta["kspace_trajectory"] = torch.as_tensor(np.asarray(k_adc).T, dtype=dtype)
    except Exception:  # pragma: no cover - optional, pypulseq versions differ
        pass
    out.meta.update(meta)
    return out
