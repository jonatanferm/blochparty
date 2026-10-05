"""Basic reconstruction for Cartesian data."""

from __future__ import annotations

import torch
from torch import Tensor

from .sequence import Sequence


def kspace_grid(signal: Tensor, seq: Sequence) -> Tensor:
    """(C, n_adc) signal -> (C, *kspace_shape) k-space with k = 0 at index n/2 on each axis."""
    return signal.reshape(signal.shape[0], *seq.meta["kspace_shape"])


def ifftnc(k: Tensor, ndim: int) -> Tensor:
    """Centred inverse DFT over the last ``ndim`` axes. With the default (backward)
    normalisation, a spin of magnetization m at a voxel centre gives voxel value m."""
    dims = tuple(range(-ndim, 0))
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(k, dim=dims), dim=dims), dim=dims)


def ifft2c(k: Tensor) -> Tensor:
    return ifftnc(k, 2)


def ifft3c(k: Tensor) -> Tensor:
    return ifftnc(k, 3)


def reconstruct(signal: Tensor, seq: Sequence) -> Tensor:
    """(C, n_adc) Cartesian signal -> (C, *kspace_shape) coil images."""
    return ifftnc(kspace_grid(signal, seq), len(seq.meta["kspace_shape"]))


def rss(images: Tensor, dim: int = 0) -> Tensor:
    return torch.sqrt((images.abs() ** 2).sum(dim=dim))


def coil_combine(images: Tensor, sens: Tensor) -> Tensor:
    """Sensitivity-weighted (Roemer / SENSE R=1) combination of (C, ...) coil images."""
    num = (sens.conj() * images).sum(dim=0)
    den = (sens.abs() ** 2).sum(dim=0)
    return torch.where(den > 0, num / den.clamp_min(1e-30), torch.zeros_like(num))
