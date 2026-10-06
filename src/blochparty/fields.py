"""Scanner field model: what the spins see beyond the ideal B0 + G.r (REALISM.md, R1).

``Scanner`` is an opt-in argument of ``simulate`` and ``simulate_fused``; ``None`` keeps
the ideal model. Everything here is a differentiable function of its tensors, so the
hardware parameters (field strength, harmonic coefficients) get gradients through both
simulators.

Concomitant (Maxwell) fields. A gradient coil cannot make a pure Bz = G.r: Maxwell's
equations (div B = curl B = 0) force transverse components. For the cylindrical coil
set, with z-coil asymmetry a (a = 1/2 is the symmetric coil),

    Bx = gx z - a gz x,   By = gy z - (1 - a) gz y,   Bz = f0 + g.r        (all in Hz)

and spins precess about |B|. To leading order in 1/f0 (f0 = gamma_bar B0), |B| - f0 =
g.r + (Bx^2 + By^2) / (2 f0); for a = 1/2 that is Bernstein's

    f_c = [(gx^2 + gy^2) z^2 + gz^2 (x^2 + y^2) / 4 - gx gz x z - gy gz y z] / (2 f0).

f_c is a quadratic form in g with per-spin coefficients (``concomitant_coefficients``),
so a free run's phase needs only the cumulative moments int g_i g_j dt, exact for
piecewise-constant segments: the closed-form free runs stay closed form. Ramps cut on
the gradient raster are staircases of their segment averages, whose int g^2 is low by
(step)^2 dt / 12 per segment. The "exact" mode evaluates |B| itself (for checks and
very low field). The concomitant terms use the true spin positions and the linear
coil model, also when a gradient nonlinearity is set.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

from .constants import GAMMA_BAR

CONCOMITANT_MODES = ("leading", "exact")


@dataclass
class Scanner:
    """Hardware field model.

    field_strength: B0 [T] (float or tensor; differentiable). Concomitant fields are
        modelled when it is given.
    concomitant: "leading" (quadratic form, the default), "exact" (|B| of the full vector
        field) or None (off).
    z_asymmetry: share a of the z coil's transverse field along x (1 - a along y);
        1/2 for the usual symmetric cylindrical coil.
    """

    field_strength: float | Tensor | None = None
    concomitant: str | None = "leading"
    z_asymmetry: float | Tensor = 0.5

    def __post_init__(self):
        if self.concomitant is not None and self.concomitant not in CONCOMITANT_MODES:
            raise ValueError(f"concomitant must be one of {CONCOMITANT_MODES} or None, got {self.concomitant!r}")

    @property
    def larmor(self) -> float | Tensor:
        """f0 = gamma_bar B0 [Hz]."""
        return GAMMA_BAR * self.field_strength


class SpinFields(NamedTuple):
    """Per-spin inputs of the simulators' field model.

    pos: (N, 3) encoding positions (the true ones without a gradient nonlinearity).
    conc: None; (N, 6) quadratic-form coefficients (mode "leading"); or (N, 7) transverse
        coefficients and f0 (mode "exact"); see ``concomitant_coefficients`` and
        ``transverse_coefficients``.
    mode: 0 none, 1 leading order, 2 exact (as in the fused kernel).
    """

    pos: Tensor
    conc: Tensor | None
    mode: int


def transverse_coefficients(pos: Tensor, z_asymmetry=0.5) -> Tensor:
    """(N, 6) rows (ax, ay) with transverse field (Bx, By) = (ax.g, ay.g) [Hz] at ``pos``."""
    x, y, z = pos.unbind(-1)
    zero = torch.zeros_like(x)
    a = torch.as_tensor(z_asymmetry, dtype=pos.dtype, device=pos.device)
    return torch.stack([z, zero, -a * x, zero, z, -(1 - a) * y], dim=-1)


def concomitant_coefficients(pos: Tensor, f0, z_asymmetry=0.5) -> Tensor:
    """(N, 6) coefficients c of the leading-order concomitant field
    f_c = c . (gx^2, gy^2, gz^2, gx gy, gx gz, gy gz) [Hz], with g in Hz/m and f0 in Hz."""
    t = transverse_coefficients(pos, z_asymmetry)
    ax, ay = t[:, :3], t[:, 3:]
    sq = ax * ax + ay * ay
    cross = lambda i, j: 2 * (ax[:, i] * ax[:, j] + ay[:, i] * ay[:, j])
    c = torch.stack([sq[:, 0], sq[:, 1], sq[:, 2], cross(0, 1), cross(0, 2), cross(1, 2)], dim=-1)
    return c / (2 * f0)


def gradient_monomials(g: Tensor) -> Tensor:
    """(..., 6) products (gx^2, gy^2, gz^2, gx gy, gx gz, gy gz) of gradients (..., 3)."""
    gx, gy, gz = g.unbind(-1)
    return torch.stack([gx * gx, gy * gy, gz * gz, gx * gy, gx * gz, gy * gz], dim=-1)


def exact_excess(lin: Tensor, tx: Tensor, ty: Tensor, f0) -> Tensor:
    """|B| - f0 - lin for Bz = f0 + lin and transverse (tx, ty), without cancellation:
    t^2 / (sqrt(u^2 + t^2) + u) with u = f0 + lin."""
    t2 = tx * tx + ty * ty
    u = f0 + lin
    return t2 / (torch.sqrt(u * u + t2) + u)


def spin_fields(scanner: Scanner | None, pos: Tensor) -> SpinFields:
    """The per-spin field-model inputs for ``scanner`` at true positions ``pos``."""
    if scanner is None:
        return SpinFields(pos, None, 0)
    enc = pos
    if scanner.concomitant is None or scanner.field_strength is None:
        return SpinFields(enc, None, 0)
    f0 = torch.as_tensor(scanner.larmor, dtype=pos.dtype, device=pos.device)
    if scanner.concomitant == "leading":
        return SpinFields(enc, concomitant_coefficients(pos, f0, scanner.z_asymmetry), 1)
    t = transverse_coefficients(pos, scanner.z_asymmetry)
    return SpinFields(enc, torch.cat([t, f0.expand(pos.shape[0], 1)], dim=-1), 2)


__all__ = ["CONCOMITANT_MODES", "Scanner", "SpinFields", "concomitant_coefficients", "exact_excess",
           "gradient_monomials", "spin_fields", "transverse_coefficients"]
