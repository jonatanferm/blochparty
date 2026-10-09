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
        # pypulseq's default (remove_duplicates) rounds the event libraries before merging them: amplitudes
        # to six significant digits, timings to 1 us. Files keep the numbers they carry.
        seq.read(str(seq_or_path), remove_duplicates=False)
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
              "segment_block": np.repeat(np.asarray(list(seq.block_events), dtype=np.int64),
                                         [b - a for a, b in block_segments]),
              "definitions": dict(getattr(seq, "definitions", {})), "convention": convention},
    )
    if convention == "physical":  # what write_pulseq re-emits (it writes the physical reading)
        out.meta["pulseq_source"] = seq
        out.meta["pulseq_fingerprint"] = _fingerprint(out)
    try:  # k-space trajectory at the ADC samples, from pypulseq itself
        k_adc = seq.calculate_kspace()[0]
        out.meta["kspace_trajectory"] = torch.as_tensor(np.asarray(k_adc).T, dtype=dtype)
    except Exception:  # pragma: no cover - optional, pypulseq versions differ
        pass
    out.meta.update(meta)
    return out


# --- export ---------------------------------------------------------------------------------
#
# write_pulseq writes Pulseq 1.5 text itself, with every number at full double precision:
# pypulseq's own writer rounds amplitudes to six significant digits ("{:12g}"), so its files
# do not reproduce a sequence better than ~5e-7. The files are read back by pypulseq (and
# read_pulseq) in the tests, which is the independent check of the format.


def _events_from_record(rec: dict, sysm) -> dict:
    """A builder ``block`` record as Pulseq events: {"duration", "rf", "grads": {axis: event}, "adc": [...]}.
    RF: conj of the field (the physical reading of ``read_pulseq``), on the RF raster; gradients: a
    trapezoid where one trapezoid makes up the axis, else the piecewise-linear sum as an extended
    trapezoid (corners on the gradient raster)."""
    from .sequence import Trapezoid, block_duration

    grads = rec["grads"]
    rf = rec["rf"]
    adcs = rec["adc"]
    end = block_duration(*grads, rf=rf, adc=adcs, system=sysm) if rec["duration"] is None else float(rec["duration"])
    out = {"duration": end, "rf": None, "grads": {}, "adc": [_adc_event(a) for a in adcs]}
    if rf is not None:
        sig = rf.signal.detach().to(torch.float64).cpu().numpy()
        steps = rf.raster / sysm.rf_raster
        if abs(steps - round(steps)) > 1e-6 or round(steps) < 1:
            raise ValueError(f"RF steps of {rf.raster:.9g} s are not a multiple of the RF raster {sysm.rf_raster:.9g} s")
        sig = np.repeat(sig[:, 0] - 1j * sig[:, 1], round(steps))  # Pulseq signal = conj(field)
        out["rf"] = {"signal": sig, "delay": float(rf.delay), "center": float(rf.center)}
    for axis in range(3):
        parts = [g for g in grads if bool((g.corners()[1][:, axis].detach() != 0).any())]
        if not parts:
            continue
        if len(parts) == 1 and isinstance(parts[0], Trapezoid):
            g = parts[0]
            out["grads"][axis] = {"type": "trap", "amplitude": float(g.amplitude[axis]), "rise": g.rise, "flat": g.flat,
                                  "fall": g.fall, "delay": g.delay}
            continue
        times = np.unique(np.concatenate([g.corners()[0] for g in parts]))
        times = times[np.concatenate([[True], np.diff(times) > _T_TOL])]
        vals = sum(np.interp(times, g.corners()[0], g.corners()[1][:, axis].detach().double().cpu().numpy(),
                             left=0.0, right=0.0) for g in parts)
        out["grads"][axis] = {"type": "ext", "times": times, "values": vals}
    return out


def _split_adcs(ev: dict, sysm) -> list[dict]:
    """Pulseq blocks carry one ADC: a block with several is cut between them, on the gradient raster
    (gradients continue across the cuts as extended trapezoids)."""
    adcs = sorted(ev["adc"], key=lambda a: a["delay"])
    if len(adcs) <= 1:
        return [ev]
    if ev["rf"] is not None:
        raise NotImplementedError("a block with RF and several ADC events cannot be written")
    cuts = []
    for a, b in zip(adcs[:-1], adcs[1:]):
        lo, hi = a["delay"] + a["num"] * a["dwell"], b["delay"]  # end of a's last sample cell, start of b's first
        c = math.ceil(lo / sysm.grad_raster - 1e-9) * sysm.grad_raster
        if c > hi + _T_TOL:
            raise ValueError(f"no gradient-raster point between ADC events at {lo:.9g} s and {hi:.9g} s")
        cuts.append(c)
    edges = [0.0] + cuts + [ev["duration"]]
    shapes = {}
    for axis, g in ev["grads"].items():
        if g["type"] == "trap":
            t = g["delay"] + np.cumsum([0.0, g["rise"], g["flat"], g["fall"]])
            shapes[axis] = (t, g["amplitude"] * np.array([0.0, 1.0, 1.0, 0.0]))
        else:
            shapes[axis] = (g["times"], g["values"])
    out = []
    for k, (a0, a1) in enumerate(zip(edges[:-1], edges[1:])):
        piece = {"duration": a1 - a0, "rf": None, "grads": {}, "adc": [{**adcs[k], "delay": adcs[k]["delay"] - a0}]}
        for axis, (t, v) in shapes.items():
            inside = (t > a0 + _T_TOL) & (t < a1 - _T_TOL)
            tt = np.concatenate([[a0], t[inside], [a1]])
            vv = np.interp(tt, t, v, left=0.0, right=0.0)
            if np.any(vv != 0):
                piece["grads"][axis] = {"type": "ext", "times": tt - a0, "values": vv}
        out.append(piece)
    return out


def _adc_event(a) -> dict:
    """A builder ``ADC`` as Pulseq fields. read_pulseq's physical reading gives the receiver phase
    psi = -(Pulseq phase offset), so the offset is -psi."""
    ph = (a.phase.detach() if isinstance(a.phase, torch.Tensor) else torch.tensor(a.phase, dtype=torch.float64))
    ph = ph.double().reshape(-1)  # not torch.as_tensor(float): that is float32
    if not bool((ph == ph[0]).all()):
        raise NotImplementedError("an ADC event with a varying receiver phase")
    return {"num": int(a.n), "dwell": float(a.dwell), "delay": float(a.delay), "freq": 0.0, "phase": -float(ph[0]),
            "freq_ppm": 0.0, "phase_ppm": 0.0}


def _scaled(x: float, unit: float) -> float:
    """A number v to write for the value x [s] in a field the reader multiplies by ``unit`` (us: 1e-6),
    chosen so that v * unit == x exactly: x / unit itself does not always round-trip (60 * 1e-6 is
    not 6e-05), so nearby and shorter decimals are tried; x / unit if none does."""
    v = x / unit
    cands = [v, round(v, 6), round(v, 9), round(v, 12), np.nextafter(v, np.inf), np.nextafter(v, -np.inf)]
    for c in cands:
        if float(c) * unit == x:
            return float(c)
    return float(v)


def _rf_use(signal: np.ndarray, raster: float) -> str:
    """'e' or 'r', as pypulseq guesses for files without the field: excitation below 90.01 deg."""
    flip = abs(np.sum(signal) * raster) * 360.0
    return "e" if flip < 90.01 else "r"


class _Writer:
    """Pulseq 1.5 text: events and shapes deduplicated into libraries, numbers at full precision."""

    def __init__(self, sysm):
        self.sys = sysm
        self.blocks, self.rf, self.grad, self.adc, self.shapes = [], {}, {}, {}, {}

    def _shape(self, values) -> int:
        key = tuple(float(v) for v in values)
        if key not in self.shapes:
            self.shapes[key] = len(self.shapes) + 1
        return self.shapes[key]

    @staticmethod
    def _lib(lib: dict, key) -> int:
        if key not in lib:
            lib[key] = len(lib) + 1
        return lib[key]

    def _on(self, t: float, raster: float, what: str) -> None:
        if abs(t / raster - round(t / raster)) > 1e-6:
            raise ValueError(f"{what} = {t:.12g} s is not on the raster {raster:.9g} s")

    def add(self, ev: dict) -> None:
        sysm = self.sys
        self._on(ev["duration"], sysm.grad_raster, "block duration")
        rf_id = 0
        if ev["rf"] is not None:
            sig = ev["rf"]["signal"]
            amp = float(np.abs(sig).max())
            if amp == 0:
                raise ValueError("an RF event with a zero signal")
            mag = np.abs(sig) / amp
            pha = np.mod(np.angle(sig), 2 * math.pi) / (2 * math.pi)
            if "times" in ev["rf"]:
                t_id = self._shape([_scaled(float(x), sysm.rf_raster) for x in ev["rf"]["times"]])
            else:
                t_id = 0
            use = ev["rf"].get("use") or _rf_use(sig, sysm.rf_raster)
            key = (amp, self._shape(mag), self._shape(pha), t_id, _scaled(ev["rf"]["center"], 1e-6),
                   _scaled(ev["rf"]["delay"], 1e-6),
                   ev["rf"].get("freq_ppm", 0.0), ev["rf"].get("phase_ppm", 0.0), ev["rf"].get("freq", 0.0),
                   ev["rf"].get("phase", 0.0), use)
            rf_id = self._lib(self.rf, key)
        g_ids = [0, 0, 0]
        for axis, g in ev["grads"].items():
            if g["type"] == "trap":
                for name in ("rise", "flat", "fall", "delay"):
                    self._on(g[name], sysm.grad_raster, f"trapezoid {name}")
                key = ("trap", g["amplitude"], *(_scaled(g[n], 1e-6) for n in ("rise", "flat", "fall", "delay")))
                g_ids[axis] = self._lib(self.grad, key)
            elif g["type"] == "ext":
                t, v = np.asarray(g["times"], dtype=float), np.asarray(g["values"], dtype=float)
                for x in t:
                    self._on(float(x), sysm.grad_raster, "gradient corner")
                delay = float(t[0])
                amp = float(np.abs(v).max())
                shape = v / amp if amp > 0 else v
                key = ("ext", amp, float(v[0]), float(v[-1]), self._shape(shape),
                       self._shape([_scaled(float(x) - delay, sysm.grad_raster) for x in t]), _scaled(delay, 1e-6))
                g_ids[axis] = self._lib(self.grad, key)
            else:  # arbitrary on the default raster, from an imported file: waveform at cell centres
                v = np.asarray(g["waveform"], dtype=float)
                amp = float(np.abs(v).max())
                key = ("arb", amp, float(g["first"]), float(g["last"]), self._shape(v / amp if amp > 0 else v), 0,
                       _scaled(float(g["delay"]), 1e-6))
                g_ids[axis] = self._lib(self.grad, key)
        adc_id = 0
        if ev["adc"]:
            (a,) = ev["adc"]
            self._on(a["dwell"], sysm.adc_raster, "ADC dwell")
            key = (a["num"], _scaled(a["dwell"], 1e-9), _scaled(a["delay"], 1e-6), a["freq_ppm"], a["phase_ppm"], a["freq"],
                   a["phase"], 0)
            adc_id = self._lib(self.adc, key)
        dur = round(ev["duration"] / sysm.grad_raster)
        self.blocks.append((dur, rf_id, *g_ids, adc_id))

    def text(self, name: str | None) -> str:
        sysm = self.sys
        f = lambda x: repr(float(x))
        out = ["# Pulseq sequence file", "# Created by blochparty (full precision)", "", "[VERSION]", "major 1", "minor 5",
               "revision 0", "", "[DEFINITIONS]", f"AdcRasterTime {f(sysm.adc_raster)}",
               f"BlockDurationRaster {f(sysm.grad_raster)}", f"GradientRasterTime {f(sysm.grad_raster)}",
               f"RadiofrequencyRasterTime {f(sysm.rf_raster)}"]
        if name:
            out.append(f"Name {name}")
        out += ["", "# Format of blocks:", "# NUM DUR RF  GX  GY  GZ  ADC  EXT", "[BLOCKS]"]
        out += [f"{i + 1} {b[0]} {b[1]} {b[2]} {b[3]} {b[4]} {b[5]} 0" for i, b in enumerate(self.blocks)]
        if self.rf:
            out += ["", "# Format of RF events:",
                    "# id ampl. mag_id phase_id time_shape_id center delay freqPPM phasePPM freq phase use", "[RF]"]
            out += [f"{i} {f(k[0])} {k[1]} {k[2]} {k[3]} {f(k[4])} {f(k[5])} {f(k[6])} {f(k[7])} {f(k[8])} {f(k[9])} {k[10]}"
                    for k, i in self.rf.items()]
        # [GRADIENTS] and [TRAP] share one id space (the block table does not say which section)
        arb = [(k, i) for k, i in self.grad.items() if k[0] != "trap"]
        trap = [(k, i) for k, i in self.grad.items() if k[0] == "trap"]
        if arb:
            out += ["", "# Format of arbitrary gradients:", "# id amplitude first last amp_shape_id time_shape_id delay",
                    "[GRADIENTS]"]
            out += [f"{i} {f(k[1])} {f(k[2])} {f(k[3])} {k[4]} {k[5]} {f(k[6])}" for k, i in arb]
        if trap:
            out += ["", "# Format of trapezoid gradients:", "# id amplitude rise flat fall delay", "[TRAP]"]
            out += [f"{i} {f(k[1])} {f(k[2])} {f(k[3])} {f(k[4])} {f(k[5])}" for k, i in trap]
        if self.adc:
            out += ["", "# Format of ADC events:", "# id num dwell delay freqPPM phasePPM freq phase phase_id", "[ADC]"]
            out += [f"{i} {k[0]} {f(k[1])} {f(k[2])} {f(k[3])} {f(k[4])} {f(k[5])} {f(k[6])} {k[7]}" for k, i in self.adc.items()]
        if self.shapes:
            out += ["", "# Sequence Shapes", "[SHAPES]"]
            for vals, i in self.shapes.items():
                out += ["", f"shape_id {i}", f"num_samples {len(vals)}"] + [f(v) for v in vals]
        # the signature (Pulseq spec): md5 of everything above it; the newline before [SIGNATURE] belongs to the
        # signature. Readers also need a line after the last shape (pypulseq's shape loop runs on at end of file)
        body = "\n".join(out) + "\n"
        import hashlib

        return body + "\n[SIGNATURE]\n# md5 of this file up to the newline before [SIGNATURE]\nType md5\n" \
            f"Hash {hashlib.md5(body.encode()).hexdigest()}\n"


def _replay(blocks: list[dict], sysm, dtype) -> Sequence:
    """The sequence the provenance records describe (a fresh builder running the same calls)."""
    from .sequence import SequenceBuilder

    b = SequenceBuilder(dtype=dtype, system=sysm)
    for rec in blocks:
        kind = rec["kind"]
        if kind == "block":
            b.block(*rec["grads"], rf=rec["rf"], adc=rec["adc"], duration=rec["duration"])
        elif kind == "delay":
            b.delay(rec["duration"], grad=rec["grad"])
        elif kind == "rf":
            b.rf(rec["duration"], flip_deg=rec["flip_deg"], amplitude=rec["amplitude"], phase=rec["phase"],
                 grad=rec["grad"], n_steps=rec["n_steps"])
        elif kind == "rf_waveform":
            b.rf_waveform(rec["samples"], rec["dt"], grad=rec["grad"])
        elif kind == "readout":
            b.readout(rec["n"], rec["dwell"], grad=rec["grad"], phase=rec["phase"])
        else:
            raise ValueError(f"unknown block record {kind!r}")
    return b.build()


def write_pulseq(seq: Sequence, path, *, system=None, spoil: str = "raise", name: str | None = None) -> None:
    """Write ``seq`` as a Pulseq 1.5 file, losslessly: ``read_pulseq`` of the file gives the same
    segments (up to the extra breakpoint it puts at each RF centre) and so the same simulation.

    The sequence must carry block provenance (``meta["blocks"]``, from ``SequenceBuilder``), which
    is replayed and must reproduce ``seq`` exactly: a sequence changed afterwards (by
    ``girf.apply_girf``, say: write the nominal one) raises rather than being written from stale
    events. Each builder block becomes one Pulseq block (a block with several ADC events is cut
    between them, its gradients continuing across the cuts). Not writable, and raising: ideal
    spoilers (``spoil="drop"`` writes the sequence without them), the dead-time-free ``rf`` /
    ``rf_waveform`` / ``readout`` calls and ``delay`` with a gradient (instantaneous gradient steps
    and RF without dead time are not hardware), timings off the rasters of ``system`` (default: the
    builder's ``System``, ``meta["system"]``). ``b0_offset`` and ``rf_freq`` model the scanner, not the sequence, and
    are not written. Pulse use is written as excitation below 90.01 deg and refocusing above (as
    pypulseq guesses); only ``calculate_kspace`` reads it.

    Sequences read with ``read_pulseq`` are written back from their source events
    (``meta["pulseq_source"]``), if their segments are unchanged since the import."""
    from .sequence import System

    if spoil not in ("raise", "drop"):
        raise ValueError(f"spoil must be 'raise' or 'drop', not {spoil!r}")
    if any(seq.spoil) and spoil == "raise":
        raise ValueError("ideal spoilers (Sequence.spoil) cannot be written to Pulseq; use crusher gradients, or "
                         "spoil='drop' to write the sequence without them")
    if "pulseq_source" in seq.meta:
        _write_imported(seq, path, name)
        return
    blocks = seq.meta.get("blocks")
    if not blocks:
        raise NotImplementedError("write_pulseq needs block provenance (a SequenceBuilder sequence or read_pulseq)")
    sysm = system or seq.meta.get("system") or System()
    replayed = _replay(blocks, sysm, seq.dt.dtype)
    same = (len(replayed) == len(seq) and replayed.is_rf == list(seq.is_rf) and replayed.adc == list(seq.adc)
            and all(torch.equal(getattr(replayed, k).detach(), getattr(seq, k).detach())
                    for k in ("dt", "rf", "grad", "adc_phase")))
    if not same:
        raise ValueError("the sequence differs from what its block records build (changed after building?); "
                         "write the sequence as built")
    w = _Writer(sysm)
    for rec in blocks:
        kind = rec["kind"]
        if kind == "delay":
            if rec["grad"] is not None and bool((torch.as_tensor(rec["grad"]) != 0).any()):
                raise NotImplementedError("a delay with a gradient is an instantaneous step: use a block with a trapezoid")
            w.add({"duration": float(rec["duration"]), "rf": None, "grads": {}, "adc": []})
        elif kind == "block":
            for ev in _split_adcs(_events_from_record(rec, sysm), sysm):
                w.add(ev)
        else:
            raise NotImplementedError(f"SequenceBuilder.{kind} makes ideal segments (no RF dead time, instantaneous "
                                      "gradients); build with SequenceBuilder.block to write Pulseq")
    Path(path).write_text(w.text(name))


def _fingerprint(seq: Sequence) -> str:
    import hashlib

    h = hashlib.sha1()
    for t in (seq.dt, seq.rf, seq.grad, seq.adc_phase):
        h.update(t.detach().to(torch.float64).cpu().numpy().tobytes())
    h.update(np.packbits(np.array([seq.is_rf, seq.adc, seq.spoil], dtype=bool)).tobytes())
    return h.hexdigest()


def _write_imported(seq: Sequence, path, name: str | None) -> None:
    """Re-emit the blocks of the source ``pypulseq.Sequence`` at full precision (labels and other
    extensions, which ``read_pulseq`` ignores, are dropped)."""
    from .sequence import System

    if seq.meta.get("pulseq_fingerprint") != _fingerprint(seq):
        raise ValueError("the sequence was changed after read_pulseq; write the sequence as imported")
    src = seq.meta["pulseq_source"]
    sd = src.system
    sysm = System(grad_raster=sd.grad_raster_time, rf_raster=sd.rf_raster_time, adc_raster=sd.adc_raster_time)
    if abs(src.block_duration_raster - sd.grad_raster_time) > 1e-15:
        raise NotImplementedError("a block duration raster different from the gradient raster")
    w = _Writer(sysm)
    for idx in src.block_events:
        block = src.get_block(idx)
        ev = {"duration": float(src.block_durations[idx]), "rf": None, "grads": {}, "adc": []}
        rf = block.rf
        if rf is not None:
            t = np.asarray(rf.t, dtype=float)
            on_raster = np.allclose(t, (np.arange(len(t)) + 0.5) * sd.rf_raster_time, atol=_T_TOL, rtol=0)
            center = getattr(rf, "center", None)
            if center is None:
                from pypulseq import calc_rf_center

                center = calc_rf_center(rf)[0]
            ev["rf"] = {"signal": np.asarray(rf.signal, dtype=complex), "delay": float(rf.delay), "center": float(center),
                        "freq": float(rf.freq_offset), "phase": float(rf.phase_offset),
                        "freq_ppm": float(getattr(rf, "freq_ppm", 0.0)), "phase_ppm": float(getattr(rf, "phase_ppm", 0.0)),
                        "use": (getattr(rf, "use", "") or "u")[0]}
            if not on_raster:
                ev["rf"]["times"] = t
        for axis, ch in enumerate(("gx", "gy", "gz")):
            g = getattr(block, ch)
            if g is None:
                continue
            if g.type == "trap":
                ev["grads"][axis] = {"type": "trap", "amplitude": float(g.amplitude), "rise": float(g.rise_time),
                                     "flat": float(g.flat_time), "fall": float(g.fall_time), "delay": float(g.delay)}
            elif g.time_id in (0, -1):
                ev["grads"][axis] = {"type": "arb", "waveform": np.asarray(g.waveform, dtype=float),
                                     "first": float(g.first), "last": float(g.last), "delay": float(g.delay)}
            else:
                ev["grads"][axis] = {"type": "ext", "times": float(g.delay) + np.asarray(g.tt, dtype=float),
                                     "values": np.asarray(g.waveform, dtype=float)}
        a = block.adc
        if a is not None:
            mod = getattr(a, "phase_modulation", None)
            if mod is not None and len(mod) and np.any(np.asarray(mod) != 0):
                raise NotImplementedError("ADC phase modulation")
            ev["adc"] = [{"num": int(a.num_samples), "dwell": float(a.dwell), "delay": float(a.delay),
                          "freq": float(a.freq_offset), "phase": float(a.phase_offset),
                          "freq_ppm": float(getattr(a, "freq_ppm", 0.0)), "phase_ppm": float(getattr(a, "phase_ppm", 0.0))}]
        w.add(ev)
    Path(path).write_text(w.text(name))
