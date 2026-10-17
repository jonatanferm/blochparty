"""Reconstruction for Cartesian data: k-space gridding, centred FFTs, RSS and coil combination; and for
undersampled data, line masks (uniform, variable-density, random) and CG-SENSE with given coil maps."""

from __future__ import annotations

import numpy as np
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


# ---- orthonormal transforms, line masks and SENSE (undersampled acquisitions)


def fft2c(x: Tensor) -> Tensor:
    """Centred orthonormal 2D FFT over the last two axes (``ifft2c_ortho`` is its exact inverse and adjoint)."""
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ifft2c_ortho(k: Tensor) -> Tensor:
    """Centred orthonormal inverse 2D FFT over the last two axes: the inverse and adjoint of ``fft2c`` (``ifft2c``
    keeps the FFT's 1/N on the inverse)."""
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def uniform_mask(n: int, lines: int, centre: int = 4) -> Tensor:
    """(n,) bool over the phase-encode lines: ``lines`` of them, the central ``centre`` always, the rest evenly spaced."""
    keep = torch.zeros(n, dtype=torch.bool)
    keep[n // 2 - centre // 2 : n // 2 - centre // 2 + centre] = True
    rest = max(lines - centre, 0)
    if rest:
        cand = torch.nonzero(~keep).squeeze(1)
        pick = torch.linspace(0, len(cand) - 1, rest).round().long()
        keep[cand[pick]] = True
    return keep


def variable_density_mask(n: int, lines: int, centre: int = 4, seed: int = 0) -> Tensor:
    """(n,) bool: the central ``centre`` lines and the rest drawn without replacement from a Gaussian density of
    width n/5 over the grid (a zero-filled image then shows incoherent aliasing instead of folded copies of the
    head); deterministic in ``seed``."""
    keep = uniform_mask(n, centre, centre)
    k = torch.arange(n, dtype=torch.float32) - n / 2
    w = torch.exp(-(k**2) / (2 * (n / 5) ** 2)) * (~keep)
    rest = max(lines - centre, 0)
    if rest:  # (torch.multinomial refuses zero samples)
        g = torch.Generator().manual_seed(seed)
        keep[torch.multinomial(w, rest, replacement=False, generator=g)] = True
    return keep


def random_mask(n: int, accel: int, centre: float, rng: np.random.Generator) -> np.ndarray:
    """(n,) bool: n // accel lines, the central fraction ``centre`` always, the rest uniformly at random (``rng``)."""
    keep = np.zeros(n, bool)
    c = int(round(centre * n))
    keep[n // 2 - c // 2 : n // 2 - c // 2 + c] = True
    rest = np.flatnonzero(~keep)
    m = max(n // accel - c, 0)
    keep[rng.choice(rest, size=m, replace=False)] = True
    return keep


def cg_sense(k: Tensor, maps: Tensor, mask: Tensor, iters: int = 20, lam: float = 1e-3) -> Tensor:
    """Least-squares SENSE by conjugate gradient: x = argmin ||M F S x - k||^2 + lam ||x||^2 for (C, ky, kx) k-space
    whose phase-encode lines ``mask`` (ky,) are sampled, (C, n, n) coil maps; returns the (n, n) complex image."""
    m = mask[None, :, None]

    def A(x):  # image -> masked multi-coil k-space
        return fft2c(maps * x[None]) * m

    def AH(y):  # the adjoint of A (orthonormal transforms on both sides)
        return (maps.conj() * ifft2c_ortho(y * m)).sum(0)

    b = AH(k)
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = (r.conj() * r).sum().real
    for _ in range(iters):
        Ap = AH(A(p)) + lam * p
        alpha = rs / (p.conj() * Ap).sum().real
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r.conj() * r).sum().real
        if rs_new < 1e-12:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x
