"""Phantoms: collections of isochromats (spins) with tissue properties."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass
class Phantom:
    pos: Tensor  # (N, 3) position [m]
    pd: Tensor  # (N,) proton density = equilibrium magnetization (arbitrary units)
    t1: Tensor  # (N,) [s]
    t2: Tensor  # (N,) [s]
    db0: Tensor  # (N,) off-resonance [Hz]

    def __len__(self) -> int:
        return self.pd.shape[0]

    def to(self, dtype=None, device=None) -> Phantom:
        conv = lambda t: t.to(dtype=dtype, device=device)
        return replace(self, pos=conv(self.pos), pd=conv(self.pd), t1=conv(self.t1),
                       t2=conv(self.t2), db0=conv(self.db0))


def grid_positions(shape: int | tuple[int, ...], fov: float, dtype=torch.float64) -> Tensor:
    """(*shape, 3) voxel-centre positions of a Cartesian grid with isotropic ``fov``.

    ``shape`` is ``n`` (an n x n 2D grid at z = 0), ``(ny, nx)`` or ``(nz, ny, nx)``: the
    last axis is x. Index i along an axis of length n maps to (i - n/2) * fov / n,
    matching the Cartesian k-space layout.
    """
    shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
    axes = [(torch.arange(n, dtype=dtype) - n // 2) * (fov / n) for n in shape]
    xyz = list(reversed(torch.meshgrid(*axes, indexing="ij")))
    if len(xyz) == 2:
        xyz.append(torch.zeros_like(xyz[0]))
    return torch.stack(xyz, dim=-1)


def phantom_from_maps(pd: Tensor, t1: Tensor, t2: Tensor, fov: float, db0: Tensor | None = None) -> Phantom:
    """One spin per voxel centre of 2D or 3D maps; voxels with PD == 0 are dropped."""
    mask = pd > 0
    pos = grid_positions(pd.shape, fov, dtype=pd.dtype)[mask]
    db0 = torch.zeros_like(pd) if db0 is None else db0
    return Phantom(pos=pos, pd=pd[mask], t1=t1[mask], t2=t2[mask], db0=db0[mask])


# Approximate tissue properties at 3 T: (PD, T1 [s], T2 [s]).
TISSUES = {
    "fat": (0.9, 0.38, 0.10),
    "wm": (0.69, 0.85, 0.07),
    "gm": (0.8, 1.4, 0.10),
    "csf": (1.0, 4.0, 2.0),
    "lesion_t2": (0.9, 1.6, 0.25),
    "lesion_t1": (0.8, 0.4, 0.08),
}
LABELS = ["background", *TISSUES]

# Modified Shepp-Logan geometry (Toft): x0, y0, semi-axes a, b, rotation [deg],
# in coordinates normalised to [-1, 1]. Each ellipse paints a tissue; later ones win.
_SHEPP_LOGAN_2D = [
    (0.0, 0.0, 0.69, 0.92, 0.0, "fat"),
    (0.0, -0.0184, 0.6624, 0.874, 0.0, "wm"),
    (0.22, 0.0, 0.11, 0.31, -18.0, "csf"),
    (-0.22, 0.0, 0.16, 0.41, 18.0, "csf"),
    (0.0, 0.35, 0.21, 0.25, 0.0, "gm"),
    (0.0, 0.1, 0.046, 0.046, 0.0, "lesion_t2"),
    (0.0, -0.1, 0.046, 0.046, 0.0, "lesion_t2"),
    (-0.08, -0.605, 0.046, 0.023, 0.0, "lesion_t1"),
    (0.0, -0.606, 0.023, 0.023, 0.0, "lesion_t1"),
    (0.06, -0.605, 0.023, 0.046, 0.0, "lesion_t1"),
]

# 3D version (ellipsoids, after Kak & Slaney's 3D phantom): x0, y0, z0, semi-axes
# a, b, c, rotation about z [deg]. The original's small out-of-plane tilts are omitted.
_SHEPP_LOGAN_3D = [
    (0.0, 0.0, 0.0, 0.69, 0.92, 0.81, 0.0, "fat"),
    (0.0, -0.0184, 0.0, 0.6624, 0.874, 0.78, 0.0, "wm"),
    (0.22, 0.0, 0.0, 0.11, 0.31, 0.22, -18.0, "csf"),
    (-0.22, 0.0, 0.0, 0.16, 0.41, 0.28, 18.0, "csf"),
    (0.0, 0.35, -0.15, 0.21, 0.25, 0.41, 0.0, "gm"),
    (0.0, 0.1, 0.25, 0.046, 0.046, 0.05, 0.0, "lesion_t2"),
    (0.0, -0.1, 0.25, 0.046, 0.046, 0.05, 0.0, "lesion_t2"),
    (-0.08, -0.605, 0.0, 0.046, 0.023, 0.05, 0.0, "lesion_t1"),
    (0.0, -0.606, 0.0, 0.023, 0.023, 0.02, 0.0, "lesion_t1"),
    (0.06, -0.605, 0.0, 0.023, 0.046, 0.02, 0.0, "lesion_t1"),
]


def shepp_logan_maps(n: int, ndim: int = 2, dtype=torch.float64) -> dict[str, Tensor]:
    """Label, PD, T1 and T2 maps of a tissue-labelled Shepp-Logan phantom, shape
    (n, n) or (n, n, n) ordered (z, y, x). Background has PD = 0 and T1 = T2 = 1 s
    (never simulated)."""
    x, y, z = grid_positions((n,) * ndim, 2.0, dtype=dtype).unbind(-1)
    if ndim == 2:
        shapes = [(x0, y0, 0.0, a, b, math.inf, deg, t) for x0, y0, a, b, deg, t in _SHEPP_LOGAN_2D]
    else:
        shapes = _SHEPP_LOGAN_3D
    label = torch.zeros(x.shape, dtype=torch.long)
    for x0, y0, z0, a, b, c, deg, tissue in shapes:
        th = math.radians(deg)
        xr = (x - x0) * math.cos(th) + (y - y0) * math.sin(th)
        yr = -(x - x0) * math.sin(th) + (y - y0) * math.cos(th)
        label[(xr / a) ** 2 + (yr / b) ** 2 + ((z - z0) / c) ** 2 <= 1.0] = LABELS.index(tissue)
    table = torch.tensor([(0.0, 1.0, 1.0), *TISSUES.values()], dtype=dtype)
    props = table[label]
    return {"label": label, "pd": props[..., 0], "t1": props[..., 1], "t2": props[..., 2]}


def shepp_logan(n: int = 64, fov: float = 0.22, ndim: int = 2, dtype=torch.float64) -> Phantom:
    maps = shepp_logan_maps(n, ndim=ndim, dtype=dtype)
    return phantom_from_maps(maps["pd"], maps["t1"], maps["t2"], fov)
