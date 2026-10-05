"""Basic reconstruction for Cartesian data."""

from __future__ import annotations

import torch
from torch import Tensor

from .sequence import Sequence


def kspace_grid(signal: Tensor, seq: Sequence) -> Tensor:
    """(C, n_adc) signal -> (C, ny, nx) k-space with k = 0 at index (ny/2, nx/2)."""
    return signal.reshape(signal.shape[0], *seq.meta["kspace_shape"])


def ifft2c(k: Tensor) -> Tensor:
    """Centred inverse 2D DFT over the last two axes. With the default (backward)
    normalisation, a spin of magnetization m at a voxel centre gives pixel value m."""
    dims = (-2, -1)
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k, dim=dims)), dim=dims)


def rss(images: Tensor, dim: int = 0) -> Tensor:
    return torch.sqrt((images.abs() ** 2).sum(dim=dim))


def coil_combine(images: Tensor, sens: Tensor) -> Tensor:
    """Sensitivity-weighted (Roemer / SENSE R=1) combination of (C, ...) coil images."""
    num = (sens.conj() * images).sum(dim=0)
    den = (sens.abs() ** 2).sum(dim=0)
    return torch.where(den > 0, num / den.clamp_min(1e-30), torch.zeros_like(num))
