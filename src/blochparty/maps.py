"""Scanner-frame field maps, sampled where each spin is.

``FieldMaps(origin, spacing, b0=, b1=, coils=)``: regular 3D grids (arrays ordered (z, y, x),
as everywhere in blochparty; ``origin`` the position [m] of voxel (0, 0, 0), ``spacing`` (3,)
in x, y, z order) of the scanner's own B0 offset [Hz], relative B1+ and complex receive
sensitivities (C, z, y, x). They stay with the scanner while the tissue moves (``motion.py``):
every segment samples them trilinearly at each spin's effective position, so a moving spin
sees the coil, transmit field and shim it is actually in. The phantom's ``db0`` (e.g. from
susceptibility) and ``b1`` stay attached to the tissue and add to / multiply the maps.
Outside the grid, values are those of the nearest edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class FieldMaps:
    origin: Tensor | tuple  # (3,) [m], x y z of voxel (0, 0, 0)
    spacing: Tensor | tuple  # (3,) [m], x y z
    b0: Tensor | None = None  # (nz, ny, nx) [Hz]
    b1: Tensor | None = None  # (nz, ny, nx) relative transmit scale
    coils: Tensor | None = None  # (C, nz, ny, nx) complex receive sensitivities

    def shape(self) -> tuple[int, int, int]:
        for m in (self.b0, self.b1, self.coils):
            if m is not None:
                return tuple(m.shape[-3:])
        raise ValueError("FieldMaps has no maps")

    def channels(self, dtype=torch.float64) -> Tensor:
        """(K, nz, ny, nx) real channels in the kernel's order: b0, b1, then (Re, Im) per coil."""
        ch = []
        shape = self.shape()
        ch.append(self.b0.to(dtype) if self.b0 is not None else torch.zeros(shape, dtype=dtype))
        ch.append(self.b1.to(dtype) if self.b1 is not None else torch.ones(shape, dtype=dtype))
        if self.coils is not None:
            for c in self.coils:
                ch += [c.real.to(dtype), c.imag.to(dtype)]
        return torch.stack(ch)


def sample(grid: Tensor, origin: Tensor, spacing: Tensor, pos: Tensor) -> Tensor:
    """Trilinear interpolation of ``grid`` (K, nz, ny, nx) at positions (..., 3) -> (..., K),
    clamped to the grid's edges."""
    k, nz, ny, nx = grid.shape
    f = (pos - origin) / spacing  # continuous indices, x y z
    hi = torch.tensor([nx - 1, ny - 1, nz - 1], dtype=pos.dtype, device=pos.device)
    f = torch.minimum(torch.clamp(f, min=0), hi)
    i0 = torch.minimum(torch.floor(f.detach()), torch.clamp(hi - 1, min=0)).long()
    w = f - i0.to(f.dtype)
    i1 = torch.minimum(i0 + 1, hi.long())
    out = 0
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                ix = i1[..., 0] if dx else i0[..., 0]
                iy = i1[..., 1] if dy else i0[..., 1]
                iz = i1[..., 2] if dz else i0[..., 2]
                wt = (w[..., 0] if dx else 1 - w[..., 0]) * (w[..., 1] if dy else 1 - w[..., 1]) * \
                     (w[..., 2] if dz else 1 - w[..., 2])
                out = out + wt[..., None] * grid[:, iz, iy, ix].movedim(0, -1)
    return out
