"""Receive coil sensitivities and thermal noise."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


def birdcage_sensitivities(
    pos: Tensor,
    n_coils: int = 8,
    radius: float = 0.18,
    normalize: bool = False,
    radius_y: float | None = None,
    element_radius: float | None = None,
) -> Tensor:
    """(N, C) complex sensitivities of ``n_coils`` elements evenly spaced on a circle (an ellipse with semi-axes
    ``radius`` along x and ``radius_y`` along y when the latter is given: a head array hugging a head).

    A smooth analytic model, not an EM solution: magnitude falls off as
    radius / distance and the phase winds around each element. With
    ``normalize`` the root-sum-of-squares over coils is 1 everywhere. With ``element_radius`` (a, the loops' radius)
    the magnitude is a loop's on-axis field, a^2 / (a^2 + d^2)^(3/2) (times radius for scale), which is steeper than
    1 / d within a few a of the element: a head array's elements sit 1-3 cm from the scalp (2026-10-13).
    """
    theta = 2 * math.pi * torch.arange(n_coils, dtype=pos.dtype, device=pos.device) / n_coils
    dx = pos[:, :1] - radius * torch.cos(theta)
    dy = pos[:, 1:2] - (radius if radius_y is None else radius_y) * torch.sin(theta)
    dist = torch.sqrt(dx**2 + dy**2).clamp_min(0.05 * radius)
    if element_radius is None:
        mag = radius / dist
    else:
        a = element_radius
        mag = (
            radius * a**2 / (a**2 + dist**2) ** 1.5 * (a**2 + radius**2) ** 1.5 / a**2
        )  # = 1 at the centre of a circle
    sens = torch.polar(mag, torch.atan2(dy, dx) - theta)
    if normalize:
        sens = sens / torch.sqrt((sens.abs() ** 2).sum(dim=1, keepdim=True))
    return sens


def birdcage_b1(pos: Tensor, centre_brightening=0.0, radius: float = 0.1) -> Tensor:
    """(N,) relative transmit (B1+) scale of a birdcage driven in circular polarisation,
    for ``Phantom.b1``; the coil axis is z.

    An ideal birdcage gives a uniform field (``centre_brightening=0``, all ones). At high
    field the wavelength in tissue approaches the body size and the field brightens
    in the centre; this models it with the circularly polarised standing wave in a long
    dielectric cylinder, B1+ = J0(k rho) (rho = distance from the axis), normalised to 1 on
    the axis and with k chosen so that the field at ``radius`` is 1 / (1 + centre_brightening)
    of the centre's (e.g. ~0.3-0.5 for a head at 7 T, as a rough guide). ``centre_brightening``
    may be a tensor; requires k radius < 2.405 (the first zero of J0).
    """
    cb = torch.as_tensor(centre_brightening, dtype=pos.dtype)
    target = 1.0 / (1.0 + float(cb.detach()))
    if not 0 < target <= 1:
        raise ValueError(f"centre_brightening must be >= 0, got {float(cb.detach())}")
    # k radius: J0(x) = target on [0, 2.405) by bisection, then one Newton step through
    # which gradients reach centre_brightening (dJ0/dx = -J1)
    lo, hi = 0.0, 2.404825557695773
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _j0(mid) > target else (lo, mid)
    x = torch.as_tensor(0.5 * (lo + hi), dtype=pos.dtype)
    if cb.requires_grad and target < 1:
        x = x + (torch.special.bessel_j0(x) - 1.0 / (1.0 + cb)) / torch.special.bessel_j1(x)
    rho = torch.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    return _bessel_j0(x / radius * rho)


def _j0(x: float) -> float:
    return float(torch.special.bessel_j0(torch.tensor(x, dtype=torch.float64)))


def _bessel_j0(x: Tensor) -> Tensor:
    """J0 with a gradient (torch.special.bessel_j0 has none): dJ0/dx = -J1."""
    j0 = torch.special.bessel_j0(x.detach())
    if not x.requires_grad:
        return j0
    return j0 - torch.special.bessel_j1(x.detach()) * (x - x.detach())


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


# ---- the receive array as the realism preset places and normalises it


def coil_offset(anatomy, zs, cy: float, radius: float, gap: float) -> float:
    """Anterior offset [m] of the receive ring's centre from the field-of-view centre ``cy`` when the head
    rests on the coil's posterior side: the ring's posterior-most point ``gap`` behind the head's most
    posterior point (from 30 mm below slice 0 up: the occiput, not the neck). The real images are
    brighter at the back than at the front (white matter at the front 0.89 of the back's, the outer
    ring 0.71: validation/realism/orientation.py's display), as a head lying on the posterior elements
    gives; a ring centred on the head gives 1.00 and 0.82."""
    head = (sum(v for k, v in anatomy.fractions.items() if k != "background") > 0.5).numpy()
    z = anatomy.origin[2] + np.arange(anatomy.shape[0]) * anatomy.voxel[0]
    rows = head[z >= zs[0] - 30e-3]
    back = anatomy.origin[1] + np.nonzero(rows.any((0, 2)))[0][0] * anatomy.voxel[1]
    return back - cy - gap + radius


def normalised_sensitivities(
    sens: torch.Tensor, pos: torch.Tensor, smooth_mm: float, bin_mm: float = 4.0
) -> torch.Tensor:
    """The receive array as the scanner shows it after its intensity normalisation (Siemens "Prescan Normalize" and
    its kin divide the image by a smooth estimate of the array's sensitivity): ``sens`` (N, C) divided by the
    root-sum-of-squares sensitivity smoothed over ``smooth_mm`` (Gaussian sigma) in the slice plane. The estimate is
    built from the spins themselves (binned at ``bin_mm``, normalised convolution over the head, bilinear back), so
    where the sensitivity rises faster than the estimate can follow (the last centimetre or two under the elements)
    the normalised image keeps the excess, as the real scans do (journal 2026-10-13: the outer 2 cm 1.3-2.7 x brighter
    than a flat receive gives while the interior matches one). ``pos`` in the turned frame (x anterior-posterior, y
    left-right)."""
    from scipy import ndimage

    rss = sens.abs().pow(2).sum(1).sqrt().cpu().numpy()
    xy = pos[:, :2].cpu().numpy() / (bin_mm * 1e-3)
    lo = np.floor(xy.min(0)) - 2
    ij = np.floor(xy - lo).astype(int)
    shape = tuple(ij.max(0) + 3)
    num = np.zeros(shape)
    den = np.zeros(shape)
    np.add.at(num, (ij[:, 0], ij[:, 1]), rss)
    np.add.at(den, (ij[:, 0], ij[:, 1]), 1.0)
    sigma = smooth_mm / bin_mm
    smooth = ndimage.gaussian_filter(num, sigma) / np.maximum(ndimage.gaussian_filter(den, sigma), 1e-9)
    est = ndimage.map_coordinates(smooth, (xy - lo).T, order=1, mode="nearest")
    return sens / torch.as_tensor(est, dtype=sens.real.dtype, device=sens.device)[:, None]
