"""Receive coil sensitivities and thermal noise."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def birdcage_sensitivities(
    pos: Tensor, n_coils: int = 8, radius: float = 0.18, normalize: bool = False
) -> Tensor:
    """(N, C) complex sensitivities of ``n_coils`` elements evenly spaced on a circle.

    A smooth analytic model, not an EM solution: magnitude falls off as
    radius / distance and the phase winds around each element. With
    ``normalize`` the root-sum-of-squares over coils is 1 everywhere.
    """
    theta = 2 * math.pi * torch.arange(n_coils, dtype=pos.dtype, device=pos.device) / n_coils
    dx = pos[:, :1] - radius * torch.cos(theta)
    dy = pos[:, 1:2] - radius * torch.sin(theta)
    dist = torch.sqrt(dx**2 + dy**2).clamp_min(0.05 * radius)
    sens = torch.polar(radius / dist, torch.atan2(dy, dx) - theta)
    if normalize:
        sens = sens / torch.sqrt((sens.abs() ** 2).sum(dim=1, keepdim=True))
    return sens


def add_noise(signal: Tensor, sigma: float, cov: Tensor | None = None, generator=None) -> Tensor:
    """Complex Gaussian thermal noise on (C, ...) k-space.

    ``sigma`` is the per-channel standard deviation (of each of real and imag
    parts); ``cov`` an optional (C, C) channel correlation matrix scaled by sigma**2.
    """
    c = signal.shape[0]
    real_dtype = signal.real.dtype
    shape = (2, *signal.shape)
    w = torch.randn(shape, dtype=real_dtype, device=signal.device, generator=generator)
    noise = torch.complex(w[0], w[1]) * sigma
    if cov is not None:
        chol = torch.linalg.cholesky(cov.to(signal.dtype))
        noise = (chol @ noise.reshape(c, -1)).reshape(signal.shape)
    return signal + noise
