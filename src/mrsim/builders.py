"""Ready-made sequences."""

from __future__ import annotations

import torch

from .sequence import Sequence, SequenceBuilder


def gre_2d(
    n: int = 64,
    fov: float = 0.22,
    flip_deg=15.0,
    tr=10e-3,
    te: float | None = None,
    dwell: float = 10e-6,
    rf_duration: float = 20e-6,
    rf_phase: float = 0.0,
    n_dummy: int = 0,
    dtype=torch.float64,
) -> Sequence:
    """2D Cartesian spoiled gradient echo with a non-selective hard pulse, ideal
    gradients and ideal spoiling at the end of every TR.

    Line j samples ky = (j - n/2) / fov; sample i of a line samples
    kx = (i - n/2) / fov, so the echo (k = 0) is sample n/2 of line n/2.
    ``flip_deg`` and ``tr`` may be tensors that require grad.
    """
    dk = 1.0 / fov
    t_read = n * dwell
    t_echo = (n / 2 + 0.5) * dwell  # from readout start to the k = 0 sample
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
    for j in range(n):
        b.rf(rf_duration, flip_deg=flip_deg, phase=rf_phase)
        b.delay(t_pre, grad=(-(n / 2 + 0.5) * dk / t_pre, (j - n // 2) * dk / t_pre, 0.0))
        b.readout(n, dwell, grad=(dk / dwell, 0.0, 0.0))
        b.delay(t_fill).spoil()
    te_actual = rf_duration / 2 + t_pre + t_echo
    return b.build(kspace_shape=(n, n), fov=fov, te=te_actual, tr=tr, dwell=dwell)


def readout_bandwidth_per_pixel(seq: Sequence) -> float:
    """Hz per pixel along the readout (1 / readout duration)."""
    return 1.0 / (seq.meta["kspace_shape"][1] * seq.meta["dwell"])

