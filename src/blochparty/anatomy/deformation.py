"""Smooth random deformations of an anatomy (cubic B-spline displacement fields), their Jacobian, and
resampling an anatomy through one."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .grid import Anatomy, _renormalize


def _bspline_weights(n, h, spacing):
    """(n, m) uniform cubic B-spline weights at n grid points h apart, of control points
    ``spacing`` apart that cover the grid with one to spare beyond each end."""
    t = torch.arange(n, dtype=torch.float64) * (h / spacing)
    i = t.floor().long()
    u = t - i
    w = torch.stack(((1 - u) ** 3, 3 * u**3 - 6 * u**2 + 4, -3 * u**3 + 3 * u**2 + 3 * u + 1, u**3), -1) / 6
    return torch.zeros(n, int(i[-1]) + 4, dtype=torch.float64).scatter_(1, i[:, None] + torch.arange(4), w)


def random_deformation(shape, voxel, generator: torch.Generator, *, magnitude=3e-3, spacing=40e-3):
    """Smooth random displacement (3, z, y, x) [m] on a (z, y, x) grid, components along z, y, x:
    a cubic B-spline of control-point displacements drawn independently from N(0, magnitude^2),
    ``spacing`` [m] apart. Returns (displacement, control points (3, cz, cy, cx))."""
    if not magnitude >= 0 or not spacing > 0:
        raise ValueError("deformation needs magnitude >= 0 and spacing > 0 [m]")
    weights = [_bspline_weights(n, h, spacing) for n, h in zip(shape, voxel)]
    control = torch.randn((3, *(w.shape[1] for w in weights)), generator=generator, dtype=torch.float64) * magnitude
    return torch.einsum("kabc,za,yb,xc->kzyx", control, *weights), control


def deformation_jacobian(displacement: Tensor, voxel) -> Tensor:
    """det(I + grad d) on the grid (central differences, one-sided at the faces): the local
    volume ratio of x -> x + d(x). Positive everywhere means the map does not fold."""
    g = [[torch.gradient(displacement[k], spacing=[h], dim=l)[0] for l, h in enumerate(voxel)] for k in range(3)]
    for k in range(3):
        g[k][k] = g[k][k] + 1
    return (
        g[0][0] * (g[1][1] * g[2][2] - g[1][2] * g[2][1])
        - g[0][1] * (g[1][0] * g[2][2] - g[1][2] * g[2][0])
        + g[0][2] * (g[1][0] * g[2][1] - g[1][1] * g[2][0])
    )


def deform(anatomy: Anatomy, displacement: Tensor, order: int = 1) -> Anatomy:
    """Pull the anatomy back through x -> x + d(x): every volume's new value at x is its old value
    at x + d(x) (``order`` 1: trilinear; outside the grid, air and ``chi_background``). Fractions stay a convex
    combination of fractions, so they still sum to at most one; volumes change by the Jacobian. ``order`` 3: cubic
    B-spline (scipy.ndimage.map_coordinates), which keeps more of the fine structure (at a random sub-voxel
    offset, trilinear keeps ~0.76 of the amplitude at 0.29 cycles per voxel, the cubic spline ~0.95); its
    overshoot is clamped and the fractions renormalised."""
    shape = anatomy.shape
    if tuple(displacement.shape) != (3, *shape):
        raise ValueError("displacement must be (3, z, y, x) on the anatomy grid [m]")
    if order not in (1, 3):
        raise ValueError("order must be 1 (trilinear) or 3 (cubic B-spline)")
    ref = next(iter(anatomy.fractions.values()))
    if order == 3:
        from scipy import ndimage

        coords = np.stack(
            [
                np.arange(n, dtype=np.float32).reshape([-1 if i == k else 1 for i in range(3)])
                + (displacement[k] / h).numpy().astype(np.float32)
                for k, (n, h) in enumerate(zip(shape, anatomy.voxel))
            ]
        )

        def warp3(volume, fill=0.0):
            out = ndimage.map_coordinates(
                (volume - fill).numpy().astype(np.float32), coords, order=3, mode="constant", cval=0.0, prefilter=True
            )
            return torch.from_numpy(out).to(volume.dtype) + fill

        fractions = _renormalize({k: warp3(v).clamp(0, 1) for k, v in anatomy.fractions.items()})
        chi = None if anatomy.chi is None else warp3(anatomy.chi, anatomy.chi_background)
        return Anatomy(
            fractions,
            anatomy.voxel,
            anatomy.origin,
            chi,
            {k: warp3(v) for k, v in anatomy.maps.items()},
            anatomy.chi_background,
            anatomy.map_field_T,
        )
    axes = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) for n in shape), indexing="ij")
    # grid_sample, align_corners=True: index i of n sits at -1 + 2 i / (n - 1); its last axis is (x, y, z)
    grid = torch.stack(
        [
            2 * (a + displacement[k] / h) / max(n - 1, 1) - 1
            for k, (a, h, n) in reversed(list(enumerate(zip(axes, anatomy.voxel, shape))))
        ],
        -1,
    )
    grid = grid[None].to(ref.dtype)
    del axes

    def warp(volume, fill=0.0):
        out = torch.nn.functional.grid_sample(
            (volume - fill)[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        return out[0, 0] + fill

    fractions = _renormalize({k: warp(v).clamp(0, 1) for k, v in anatomy.fractions.items()})
    chi = None if anatomy.chi is None else warp(anatomy.chi, anatomy.chi_background)
    return Anatomy(
        fractions,
        anatomy.voxel,
        anatomy.origin,
        chi,
        {k: warp(v) for k, v in anatomy.maps.items()},
        anatomy.chi_background,
        anatomy.map_field_T,
    )
