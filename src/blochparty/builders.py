"""Ready-made sequences."""

from __future__ import annotations

import itertools

import torch

from .sequence import Sequence, SequenceBuilder


def gre_cartesian(
    shape: tuple[int, ...],
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
    """Cartesian spoiled gradient echo, 2D ``(ny, nx)`` or 3D ``(nz, ny, nx)``, with a
    non-selective hard pulse, ideal gradients and ideal spoiling at the end of every TR.

    Each TR reads one line along x. Lines are ordered with ky fastest (then kz), so the
    ADC samples reshape directly to ``shape``. Along an axis of length n, index i
    samples k = (i - n/2) / fov; the echo (k = 0) is at index n/2 on every axis.
    ``flip_deg`` and ``tr`` may be tensors that require grad.
    """
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
    return 1.0 / (seq.meta["kspace_shape"][-1] * seq.meta["dwell"])
