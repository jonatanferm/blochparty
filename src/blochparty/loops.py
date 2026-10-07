"""Physically based coils: arrays of circular loops in the quasi-static limit.

Each loop (centre, unit normal, radius) carries a unit current, discretised into ``n_seg``
straight pieces. Biot-Savart gives its field B and vector potential A at any points (the
quasi-static limit: wavelength >> coil and object, i.e. low field; at 3 T and above use
EM-simulated maps through ``maps.FieldMaps`` instead).

* Receive sensitivity by reciprocity: B1- = (Bx - i By) / 2 [T/A], the weight blochparty's signal
  (coil x Mxy, Mxy precessing as exp(-i 2 pi f t), demodulated by exp(+i w0 t)) needs: Faraday's
  law on a lab-frame precessing dipole gives 2 i w0 B1- Mxy (``tests/test_loops.py``). (Until
  2026-10-07 this was conj(B1-), which mirrored every coil's phase pattern; magnitudes unchanged.)
* Transmit: B1+ = (Bx + i By) / 2 of the driven combination; |B1+| relative to a reference
  point is the transmit scale (``Phantom.b1`` / ``FieldMaps.b1``).
* Noise: the sample-loss resistance matrix R_ij = w^2 int sigma A_i . A_j dV [Ohm] (Roemer et
  al., MRM 1990; quasi-static induced E = -i w A, without the conservative part from surface
  charges), plus each coil's own resistance on the diagonal. Thermal noise covariance is
  proportional to R (4 k T bandwidth R), which ``receive.thermal_noise`` takes as ``cov``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

MU0 = 4e-7 * math.pi


@dataclass
class Loop:
    centre: tuple[float, float, float]
    normal: tuple[float, float, float]
    radius: float
    n_seg: int = 128

    def segments(self, dtype=torch.float64) -> tuple[Tensor, Tensor]:
        """(n_seg, 3) midpoints and (n_seg, 3) vectors dl of the discretised loop (right-handed
        about the normal)."""
        n = torch.as_tensor(self.normal, dtype=dtype)
        n = n / n.norm()
        ref = torch.zeros(3, dtype=dtype)
        ref[int(n.abs().argmin())] = 1.0
        e1 = torch.linalg.cross(n, ref)
        e1 = e1 / e1.norm()
        e2 = torch.linalg.cross(n, e1)
        phi = torch.linspace(0, 2 * math.pi, self.n_seg + 1, dtype=dtype)
        pts = torch.as_tensor(self.centre, dtype=dtype) + self.radius * (
            torch.cos(phi)[:, None] * e1 + torch.sin(phi)[:, None] * e2)
        return 0.5 * (pts[1:] + pts[:-1]), pts[1:] - pts[:-1]


def fields(loops: list[Loop], pts: Tensor, chunk: int = 4096) -> tuple[Tensor, Tensor]:
    """B [T/A] and A [T m/A] of every loop at points (P, 3): (C, P, 3) each (midpoint rule on
    the segments: accurate a few segment lengths away from the wire)."""
    Bs, As = [], []
    for lp in loops:
        mid, dl = lp.segments(pts.dtype)
        b, a = [], []
        for s in range(0, pts.shape[0], chunk):
            r = pts[s:s + chunk, None, :] - mid[None]  # (p, n_seg, 3)
            d = r.norm(dim=-1, keepdim=True)
            b.append(MU0 / (4 * math.pi) * (torch.linalg.cross(dl[None].expand_as(r), r) / d**3).sum(1))
            a.append(MU0 / (4 * math.pi) * (dl[None] / d).sum(1))
        Bs.append(torch.cat(b))
        As.append(torch.cat(a))
    return torch.stack(Bs), torch.stack(As)


def receive_sensitivities(B: Tensor) -> Tensor:
    """(P, C) complex receive sensitivities B1- = (Bx - i By) / 2 from real fields (C, P, 3)."""
    return (0.5 * torch.complex(B[..., 0], -B[..., 1])).T


def b1_plus(B: Tensor, drive: Tensor | None = None) -> Tensor:
    """(P,) complex B1+ of the coils driven with complex weights ``drive`` (C,) (default: all 1)."""
    b1p = 0.5 * torch.complex(B[..., 0], B[..., 1])  # (C, P)
    w = torch.ones(B.shape[0], dtype=b1p.dtype) if drive is None else drive.to(b1p.dtype)
    return (w[:, None] * b1p).sum(0)


def resistance(A: Tensor, conductivity: Tensor, volume: Tensor, freq_hz: float, coil_r: float = 0.0) -> Tensor:
    """(C, C) sample-loss resistance R_ij = w^2 sum sigma A_i . A_j dV over the points (with
    conductivity [S/m] and volume [m^3] per point), plus ``coil_r`` [Ohm] on the diagonal."""
    w = 2 * math.pi * freq_hz
    weighted = A * (conductivity * volume)[None, :, None]
    R = w**2 * torch.einsum("ipk,jpk->ij", weighted, A)
    return R + coil_r * torch.eye(A.shape[0], dtype=A.dtype)


def field_maps(loops: list[Loop], origin, spacing, shape, reference=(0.0, 0.0, 0.0), drive: Tensor | None = None,
               scale: float | None = None):
    """``maps.FieldMaps`` of a loop array on a grid (shape (nz, ny, nx); origin and spacing in x y
    z): receive sensitivities B1- (scaled by ``scale``, default 1 / max |B1-| over the grid)
    and |B1+| of the driven array relative to ``reference``."""
    from .maps import FieldMaps

    nz, ny, nx = shape
    o = torch.as_tensor(origin, dtype=torch.float64)
    h = torch.as_tensor(spacing, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(*(o[i] + h[i] * torch.arange(m, dtype=torch.float64) for i, m in ((2, nz), (1, ny), (0, nx))),
                                indexing="ij")
    pts = torch.stack([xx, yy, zz], -1).reshape(-1, 3)
    B, _ = fields(loops, torch.cat([pts, torch.as_tensor([reference], dtype=torch.float64)]))
    sens = receive_sensitivities(B[:, :-1])  # (P, C)
    scale = 1 / float(sens.abs().max()) if scale is None else scale
    bp = b1_plus(B, drive).abs()
    return FieldMaps(origin, spacing, b1=(bp[:-1] / bp[-1]).reshape(nz, ny, nx),
                     coils=(scale * sens).T.reshape(len(loops), nz, ny, nx))
