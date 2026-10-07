"""Full-wave coil fields into blochparty: B1+ maps, receive sensitivities and noise covariance.

Reads the map files of mariepy (a Python port of MARIE 3.0, github.com/pulserver/mariepy; not a
dependency: only its documented ``.npz`` format is read): per channel ``plus = Bx + j By`` and
``minus = Bx - j By`` [T per unit drive] at voxel centres with the time dependence exp(+j w t),
zero outside the body's ``mask``, on a grid whose voxel (i, j, k) sits at origin + h (i, j, k)
(x, y, z). With B0 along +z, protons precess clockwise seen from +z (blochparty's
Mxy ~ exp(-i 2 pi f t)), so, from the lab-frame decomposition of the phasors (checked by the
Faraday test of ``tests/test_loops.py`` in the quasi-static limit):

* transmit: the co-rotating component, B1+ = conj(minus) / 2 for a drive sum_c w_c channel_c
  (conj(sum_c w_c minus_c) / 2), |B1+| relative to a reference gives ``FieldMaps.b1``;
* receive: the weight on Mxy is conj(plus) / 2 per channel (``FieldMaps.coils``);
* noise: the sample-loss resistance R_ij = int sigma E_i . conj(E_j) dV, written beside the maps
  by ``validation/em/run_brainweb.py``; thermal noise covariance is proportional to it
  (``receive.thermal_noise(cov=...)``).

For real (quasi-static) fields these reduce to ``loops.b1_plus`` and ``loops.receive_sensitivities``.
``FieldMaps.b1`` is a real scale: the transmit phase is dropped (exact for one transmit channel up
to an image phase; parallel transmit needs complex B1+ in the engines).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .maps import FieldMaps


@dataclass
class EMMaps:
    plus: Tensor  # (C, nx, ny, nz) complex, exp(+j w t) phasors [T per unit drive], mariepy's (i, j, k) = (x, y, z)
    minus: Tensor
    mask: Tensor  # (nx, ny, nz) bool
    origin: tuple[float, float, float]  # x y z [m] of voxel (0, 0, 0)
    resolution: float  # [m]
    frequency_hz: float
    channels: list[str]
    metadata: dict


def read_mariepy_maps(path) -> EMMaps:
    with np.load(Path(path), allow_pickle=False) as f:
        meta = json.loads(str(f["metadata"]))
        plus, minus, mask = (torch.from_numpy(np.asarray(f[k])) for k in ("plus", "minus", "mask"))
    if plus.ndim != 4 or plus.shape != minus.shape or tuple(mask.shape) != tuple(plus.shape[1:]):
        raise ValueError("unexpected map shapes")
    return EMMaps(plus.to(torch.complex128), minus.to(torch.complex128), mask.bool(), tuple(meta["origin"]),
                  float(meta["resolution"]), float(meta["frequency_hz"]), list(meta["channels"]), meta)


def read_noise_resistance(path) -> Tensor:
    """(C, C) sample-loss resistance [Ohm] written by validation/em/run_brainweb.py."""
    with np.load(Path(path), allow_pickle=False) as f:
        return torch.from_numpy(np.asarray(f["resistance"])).to(torch.complex128)


def transmit_b1(em: EMMaps, drive: Tensor | None = None) -> Tensor:
    """(nx, ny, nz) complex B1+ [T] of the drive sum_c w_c channel_c (default all 1):
    conj(sum_c w_c minus_c) / 2."""
    w = torch.ones(em.minus.shape[0], dtype=torch.complex128) if drive is None else torch.as_tensor(drive).to(torch.complex128)
    return torch.conj(torch.einsum("c,c...->...", w, em.minus)) / 2


def receive_weights(em: EMMaps) -> Tensor:
    """(C, nx, ny, nz) receive sensitivities conj(plus) / 2 [T per unit current]."""
    return torch.conj(em.plus) / 2


def quadrature_drive(em: EMMaps) -> Tensor:
    """The two-channel drive (1, +-j) whose B1+ is larger over the mask: a birdcage's co-rotating
    quadrature mode (the other one mostly counter-rotates and barely excites)."""
    if em.plus.shape[0] != 2:
        raise ValueError("quadrature needs exactly two channels")
    drives = [torch.tensor([1, s], dtype=torch.complex128) for s in (1j, -1j)]
    return max(drives, key=lambda d: float(transmit_b1(em, d).abs()[em.mask].mean()))


def _fill(volume: Tensor, mask: Tensor) -> Tensor:
    """Copy each outside voxel's nearest inside value, so trilinear sampling at the body's edge
    is not pulled towards the zeros the solver stores outside it."""
    from scipy import ndimage

    _, idx = ndimage.distance_transform_edt(~mask.numpy(), return_indices=True)
    return volume[(..., *[torch.from_numpy(i) for i in idx])]


def field_maps(transmit: EMMaps | None = None, receive: EMMaps | None = None, *, drive: Tensor | None = None,
               reference: str | tuple = "median", coil_scale: float | None = None) -> FieldMaps:
    """``FieldMaps`` (arrays (z, y, x)) from a transmit and / or a receive map set on one grid.

    ``b1`` = |B1+| / reference, with ``reference`` "median" (the median |B1+| over the body, the
    usual nominal-flip calibration) or a position (x, y, z) [m] (nearest voxel). ``drive`` for the
    transmit set: default ``quadrature_drive`` for two channels, else all ones. ``coils`` =
    receive weights scaled by ``coil_scale`` (default 1 / max |weight| over the body)."""
    sets = [m for m in (transmit, receive) if m is not None]
    if not sets:
        raise ValueError("give a transmit or a receive map set")
    ref = sets[0]
    for m in sets[1:]:
        if m.mask.shape != ref.mask.shape or m.origin != ref.origin or m.resolution != ref.resolution:
            raise ValueError("transmit and receive maps must share a grid")
    b1 = coils = None
    if transmit is not None:
        if drive is None:
            drive = quadrature_drive(transmit) if transmit.plus.shape[0] == 2 else None
        mag = transmit_b1(transmit, drive).abs()
        if reference == "median":
            r = float(mag[transmit.mask].median())
        else:
            i = [int(round((p - o) / transmit.resolution)) for p, o in zip(reference, transmit.origin)]
            r = float(mag[i[0], i[1], i[2]])
        b1 = _fill(mag / r, transmit.mask).permute(2, 1, 0).contiguous()
    if receive is not None:
        w = receive_weights(receive)
        s = 1 / float(w.abs()[:, receive.mask].max()) if coil_scale is None else coil_scale
        coils = torch.stack([_fill(c * s, receive.mask).permute(2, 1, 0) for c in w]).contiguous()
    h = ref.resolution
    return FieldMaps(ref.origin, (h, h, h), b1=b1, coils=coils)


__all__ = ["EMMaps", "field_maps", "quadrature_drive", "read_mariepy_maps", "read_noise_resistance",
           "receive_weights", "transmit_b1"]
