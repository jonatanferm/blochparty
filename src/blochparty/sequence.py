"""Sequence representation: a flat list of piecewise-constant segments.

Each segment holds RF and gradients constant for ``dt`` seconds. Continuous
quantities (``dt``, ``rf``, ``grad``, ``adc_phase``) are tensors so they can
carry gradients; structural flags (``is_rf``, ``adc``, ``spoil``) are plain
booleans that decide control flow and are never differentiated.

Units follow Pulseq: RF in Hz (gamma_bar * B1), gradients in Hz/m
(gamma_bar * G), so gradient areas are k-space positions in cycles/m.

The building blocks and ``SequenceBuilder`` are in ``sequences/builder.py``; ready-made sequences in
``sequences/library.py``; Pulseq import and export in ``sequences/pulseq/``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from torch import Tensor

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
            bound_only=None
            if self.bound_only is None
            else [b for b, r in zip(self.bound_only, self.is_rf) for _ in range(k if r else 1)],
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
        rf_freq = (
            None
            if self.rf_freq is None
            else self.rf_freq[take].index_put((rows,), torch.zeros_like(rows, dtype=self.rf_freq.dtype))
        )
        meta = {k: v for k, v in self.meta.items() if k not in ("blocks", "block_segments", "segment_block")}
        return Sequence(
            dt,
            self.rf[take],
            grad,
            self.adc_phase[take],
            is_rf[first].tolist(),
            adc[first].tolist(),
            spoil[last].tolist(),
            meta | {"merged_free": True},
            b0,
            rf_freq,
            None if self.bound_only is None else np.asarray(self.bound_only, dtype=bool)[first].tolist(),
        )

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


def flip_angle(signal: np.ndarray, dt) -> float:
    """Flip angle [deg] of an RF pulse (complex samples [Hz] over steps ``dt`` [s]) about its own
    frequency: |sum s exp(-i 2 pi f t) dt| * 360, with f the median phase step between samples (a
    slice-selective pulse off the slice centre carries f = G z0, which makes |sum s dt| cancel; the
    median ignores the pi steps at the zero crossings of a sinc)."""
    s = np.asarray(signal, dtype=complex)
    dt = np.broadcast_to(np.asarray(dt, dtype=float), s.shape)
    t = np.cumsum(dt) - 0.5 * dt
    on = np.abs(s) > 1e-6 * np.abs(s).max() if s.size else np.zeros(0, bool)
    steps = np.angle(s[1:] * np.conj(s[:-1]))[on[1:] & on[:-1]]
    f = float(np.median(steps / (2 * math.pi * 0.5 * (dt[1:] + dt[:-1])[on[1:] & on[:-1]]))) if steps.size else 0.0
    return float(abs(np.sum(s * np.exp(-2j * math.pi * f * t) * dt)) * 360.0)


def kspace_trajectory(seq: Sequence) -> Tensor:
    """(n_adc, 3) k-space [1/m] at every ADC sample (segment ends): the gradient moment since the
    last excitation, negated at each refocusing pulse. Pulses are runs of RF segments centred at
    their midpoints; excitations flip less than 90.01 deg about their own frequency
    (``flip_angle``), as in pypulseq's ``calculate_kspace``, which it equals on written
    files (tests/sequences/test_pulseq_write.py). NaN before the first excitation."""
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
