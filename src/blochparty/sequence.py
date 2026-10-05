"""Sequence representation: a flat list of piecewise-constant segments.

Each segment holds RF and gradients constant for ``dt`` seconds. Continuous
quantities (``dt``, ``rf``, ``grad``, ``adc_phase``) are tensors so they can
carry gradients; structural flags (``is_rf``, ``adc``, ``spoil``) are plain
booleans that decide control flow and are never differentiated.

Units follow Pulseq: RF in Hz (gamma_bar * B1), gradients in Hz/m
(gamma_bar * G), so gradient areas are k-space positions in cycles/m.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from torch import Tensor


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
        reps = torch.tensor([k if r else 1 for r in self.is_rf], device=self.dt.device)
        rep = lambda t: t.repeat_interleave(reps, dim=0)

        def tail(flags):  # end-of-segment flags stay on the last piece
            out = []
            for f, r in zip(flags, self.is_rf):
                out += ([False] * (k - 1) + [f]) if r else [f]
            return out

        return replace(
            self,
            dt=rep(self.dt / reps.to(self.dt.dtype)),
            rf=rep(self.rf),
            grad=rep(self.grad),
            adc_phase=rep(self.adc_phase),
            is_rf=[r for r in self.is_rf for _ in range(k if r else 1)],
            adc=tail(self.adc),
            spoil=tail(self.spoil),
        )

    def to(self, dtype=None, device=None) -> Sequence:
        conv = lambda t: t.to(dtype=dtype, device=device)
        return replace(
            self,
            dt=conv(self.dt),
            rf=conv(self.rf),
            grad=conv(self.grad),
            adc_phase=conv(self.adc_phase),
        )


class SequenceBuilder:
    """Appends segments block by block; every numeric argument may be a tensor
    that requires grad (e.g. a flip angle or TR being optimized)."""

    def __init__(self, dtype=torch.float64, device=None):
        self.dtype, self.device = dtype, device
        self._dt, self._rf, self._grad, self._phase = [], [], [], []
        self._is_rf, self._adc, self._spoil = [], [], []

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
        self._is_rf += [is_rf] * n
        self._adc += adc if isinstance(adc, list) else [adc] * n
        self._spoil += [False] * n

    def rf(self, duration, *, flip_deg=None, amplitude=None, phase=0.0, grad=None, n_steps=1):
        """Rectangular (hard) pulse; give either ``flip_deg`` or ``amplitude`` [Hz]."""
        dur = self._t(duration)
        amp = self._t(flip_deg) / 360.0 / dur if flip_deg is not None else self._t(amplitude)
        ph = self._t(phase)
        rf = torch.stack([amp * torch.cos(ph), amp * torch.sin(ph)]).expand(n_steps, 2)
        dt = (dur / n_steps).expand(n_steps)
        self._append(dt, rf, self._grad3(grad, n_steps), torch.zeros_like(dt), is_rf=True, adc=False)
        return self

    def rf_waveform(self, samples, dt, grad=None):
        """Arbitrary RF: ``samples`` is (k, 2) in Hz, one per raster step of ``dt``."""
        rf = self._t(samples)
        n = rf.shape[0]
        dts = self._t(dt).expand(n)
        self._append(dts, rf, self._grad3(grad, n), torch.zeros_like(dts), is_rf=True, adc=False)
        return self

    def delay(self, duration, grad=None):
        dt = self._t(duration).reshape(1)
        self._append(dt, torch.zeros(1, 2, dtype=self.dtype, device=self.device),
                     self._grad3(grad, 1), torch.zeros_like(dt), is_rf=False, adc=False)
        return self

    def readout(self, n: int, dwell, grad=None, phase=0.0):
        """``n`` ADC samples at (i + 0.5) * dwell after the block start; the block
        lasts n * dwell, matching the Pulseq ADC convention."""
        dw = self._t(dwell)
        dt = torch.cat([(0.5 * dw).reshape(1), dw.expand(n - 1), (0.5 * dw).reshape(1)])
        rf = torch.zeros(n + 1, 2, dtype=self.dtype, device=self.device)
        ph = self._t(phase).expand(n + 1)
        self._append(dt, rf, self._grad3(grad, n + 1), ph, is_rf=False, adc=[True] * n + [False])
        return self

    def spoil(self):
        """Ideal spoiler: transverse magnetization is zeroed at the end of the last segment."""
        self._spoil[-1] = True
        return self

    def build(self, **meta) -> Sequence:
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
