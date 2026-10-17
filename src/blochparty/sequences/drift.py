"""Slow and physiological field changes as sequence transforms (REALISM.md, R1).

A spatially uniform field that changes with time is a per-segment ``Sequence.b0_offset``;
a field linear in position, b(t).r, is a gradient, so it is added to ``Sequence.grad``.
Both are exact: every segment gets the exact time average of the field over the segment,
so the phase at every segment end is exact. (With a ``GradientNonlinearity`` the added
linear field is encoded through the coils' nonlinear positions, and with concomitant
fields on it adds concomitant terms; both are small for these fields.)

* ``linear_drift``: f(t) = offset + rate t (and optionally a linear field changing at a
  constant rate), e.g. gradient-heating drift over a scan;
* ``exponential_drift``: f(t) = amplitude (1 - exp(-t / tau)), the settling of passive shims;
* ``respiration``: f(t) = amplitude sin(2 pi t / period + phase) with an optional linear
  term gradient sin(...), the breathing-induced field in the head.

Parameters may be tensors that require grad. t is the time from the sequence start.
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import Tensor

from ..sequence import Sequence


def _times(seq: Sequence) -> tuple[Tensor, Tensor]:
    """Segment start times and durations [s]."""
    return torch.cumsum(seq.dt, 0) - seq.dt, seq.dt


def add_field(seq: Sequence, uniform: Tensor | None = None, linear: Tensor | None = None) -> Sequence:
    """Add per-segment averages of a uniform field (S,) [Hz] to ``b0_offset`` and of a linear
    field (S, 3) [Hz/m] to the gradients."""
    out = seq
    if uniform is not None:
        out = replace(out, b0_offset=uniform if seq.b0_offset is None else seq.b0_offset + uniform)
    if linear is not None:
        out = replace(out, grad=seq.grad + linear)
    return out


def _vec(v, like: Tensor) -> Tensor:
    return torch.as_tensor(v, dtype=like.dtype, device=like.device)


def linear_drift(seq: Sequence, offset=0.0, rate=0.0, gradient_rate=None) -> Sequence:
    """f(t) = offset + rate t [Hz, Hz/s]; ``gradient_rate`` (3,) [Hz/m/s] adds the linear field
    gradient_rate t . r."""
    t0, dt = _times(seq)
    mid = t0 + 0.5 * dt  # the average of a linear function is its midpoint value
    uniform = _vec(offset, mid) + _vec(rate, mid) * mid
    linear = None if gradient_rate is None else mid[:, None] * _vec(gradient_rate, mid)
    return add_field(seq, uniform, linear)


def exponential_drift(seq: Sequence, amplitude, time_constant, gradient_amplitude=None) -> Sequence:
    """f(t) = amplitude (1 - exp(-t / tau)) [Hz]; ``gradient_amplitude`` (3,) [Hz/m] adds the
    linear field with the same time course."""
    t0, dt = _times(seq)
    tau = _vec(time_constant, t0)
    # average of exp(-t / tau) over [t0, t0 + dt]: exp(-t0 / tau) (1 - exp(-dt / tau)) tau / dt
    decay = torch.exp(-t0 / tau) * (-torch.expm1(-dt / tau)) * tau / dt
    shape = 1 - decay
    uniform = _vec(amplitude, t0) * shape
    linear = None if gradient_amplitude is None else shape[:, None] * _vec(gradient_amplitude, t0)
    return add_field(seq, uniform, linear)


def respiration(seq: Sequence, amplitude, period, gradient=None, phase=0.0) -> Sequence:
    """f(t) = amplitude sin(2 pi t / period + phase) [Hz], plus the linear field
    gradient sin(...) . r for ``gradient`` (3,) [Hz/m]."""
    t0, dt = _times(seq)
    w = 2 * math.pi / _vec(period, t0)
    a = w * t0 + _vec(phase, t0)
    h = 0.5 * w * dt
    # average of sin over [a, a + 2h]: sin(a + h) sin(h) / h (no cancellation for short segments)
    shape = torch.sin(a + h) * torch.where(h > 1e-8, torch.sin(h) / h, 1 - h * h / 6)
    uniform = _vec(amplitude, t0) * shape
    linear = None if gradient is None else shape[:, None] * _vec(gradient, t0)
    return add_field(seq, uniform, linear)


__all__ = ["add_field", "exponential_drift", "linear_drift", "respiration"]
