"""Scanner field model: what the spins see beyond the ideal B0 + G.r (REALISM.md, R1).

``Scanner`` is an opt-in argument of ``simulate`` and ``simulate_metal``; ``None`` keeps
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
piecewise-constant segments: the closed-form free runs stay closed form. Moving spins (motion,
diffusion) re-evaluate the coefficients and the encoding positions every segment, at their
true positions. Ramps cut on
the gradient raster are staircases of their segment averages, whose int g^2 is low by
(step)^2 dt / 12 per segment. The "exact" mode evaluates |B| itself (for checks and
very low field). The concomitant terms use the true spin positions and the linear
coil model, also when a gradient nonlinearity is set.

Gradient nonlinearity. Each coil's Bz per unit gradient is r_a + d_a(r), with the
deviation d_a a solid-harmonic expansion (``GradientNonlinearity``, gradunwarp's
convention for Siemens coefficient files)

    d_a(r) = R0 sum_{n,m} (r / R0)^n N_nm P_n^m(cos theta) (A_nm cos(m phi) + B_nm sin(m phi)),

N_nm = sqrt((2n + 1) (n - m)! / (2 (n + m)!)) for m > 0 and 1 for m = 0, P_n^m without the
Condon-Shortley phase. The encoding positions f(r) = r + d(r) replace r in G.r, so the
simulators need nothing new: gradients reach the coefficients through the position
gradient. The basis is evaluated in Cartesian form (recurrences in x, y, z), so it is
smooth everywhere and exactly zero where the coefficients are.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor

from .constants import GAMMA_BAR

CONCOMITANT_MODES = ("leading", "exact")


def solid_harmonics(pos: Tensor, n_max: int, r0: float) -> tuple[Tensor, Tensor]:
    """(N, n_max + 1, n_max + 1) cosine and sine terms R0 (r / R0)^n N_nm P_n^m(cos theta)
    (cos(m phi), sin(m phi)) at ``pos`` (zero for m > n): the basis of ``GradientNonlinearity``.

    Recurrences: (x + i y)^m / R0^m = C_m + i S_m, and r^n P_n^m(cos theta) / rho^m = Q_n^m(z, r^2)
    with Q_m^m = (2m - 1)!!, Q_{m+1}^m = (2m + 1) z Q_m^m and
    (n - m) Q_n^m = (2n - 1) z Q_{n-1}^m - (n + m - 1) r^2 Q_{n-2}^m (scaled by R0)."""
    x, y, z = (pos / r0).unbind(-1)
    r2 = x * x + y * y + z * z
    cos, sin = [torch.ones_like(x)], [torch.zeros_like(x)]
    for m in range(n_max):
        cos.append(x * cos[m] - y * sin[m])
        sin.append(x * sin[m] + y * cos[m])
    zero = torch.zeros_like(x)
    out_c = [[zero] * (n_max + 1) for _ in range(n_max + 1)]
    out_s = [[zero] * (n_max + 1) for _ in range(n_max + 1)]
    for m in range(n_max + 1):
        q = [None] * (n_max + 1)
        q[m] = torch.full_like(x, float(math.prod(range(1, 2 * m, 2))))
        if m + 1 <= n_max:
            q[m + 1] = (2 * m + 1) * z * q[m]
        for n in range(m + 2, n_max + 1):
            q[n] = ((2 * n - 1) * z * q[n - 1] - (n + m - 1) * r2 * q[n - 2]) / (n - m)
        for n in range(m, n_max + 1):
            norm = 1.0 if m == 0 else math.sqrt((2 * n + 1) * math.factorial(n - m) / (2 * math.factorial(n + m)))
            out_c[n][m] = r0 * norm * q[n] * cos[m]
            out_s[n][m] = r0 * norm * q[n] * sin[m]
    stack = lambda t: torch.stack([torch.stack(row, -1) for row in t], -2)
    return stack(out_c), stack(out_s)


@dataclass
class GradientNonlinearity:
    """Deviation of each gradient coil's field from linearity: solid-harmonic coefficients
    per axis (gradunwarp's Siemens convention, see the module docstring).

    alpha, beta: (3, L, L) coefficients A_nm, B_nm of the x, y and z coils (n, m < L; entries
        with m > n are ignored). Tensors that require grad get gradients.
    r0: reference radius [m].
    """

    alpha: Tensor
    beta: Tensor
    r0: float

    @property
    def n_max(self) -> int:
        return self.alpha.shape[-1] - 1

    def displacement(self, pos: Tensor) -> Tensor:
        """(N, 3) d(r): encoding position minus true position [m]."""
        c, s = solid_harmonics(pos, self.n_max, self.r0)
        mask = torch.ones(self.n_max + 1, self.n_max + 1, dtype=torch.bool).tril()
        a = (self.alpha * mask).to(pos.dtype)
        b = (self.beta * mask).to(pos.dtype)
        return torch.einsum("pnm,anm->pa", c, a) + torch.einsum("pnm,anm->pa", s, b)

    def encoding_positions(self, pos: Tensor) -> Tensor:
        """(N, 3) f(r) = r + d(r): where the gradients encode a spin at ``pos``."""
        return pos + self.displacement(pos)

    @classmethod
    def from_file(cls, path, r0: float | None = None, dtype=torch.float64) -> GradientNonlinearity:
        """Read a gradunwarp-style coefficient file. Two layouts are recognised:

        * Siemens ``.grad`` / ``.coef`` lines ``<no> A( n, m) <value> <axis>`` (or ``B``), with the
          reference radius on a line ``<r0> m = R0`` (or ``R0 = <r0> [m|mm]``);
        * gradunwarp's variable lines ``alpha_x n m <value>`` / ``beta_z n m <value>`` (read
          literally: gradunwarp's legacy ``.coef`` reader permutes the axes, this one does not).

        ``r0`` [m] overrides (or supplies) the reference radius."""
        entries, radius = [], None
        siemens = re.compile(
            r"^\s*\d+\s+([AB])\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s+([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s+([xyz])\s*$",
            re.IGNORECASE,
        )
        named = re.compile(r"^\s*(alpha|beta)_([xyz])\s+(\d+)\s+(\d+)\s+([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*;?\s*$")
        for line in Path(path).read_text().splitlines():
            if line.lstrip().startswith("#"):
                continue
            if m := siemens.match(line):
                kind, n, mm, value, axis = m.groups()
                entries.append(
                    ("alpha" if kind.upper() == "A" else "beta", axis.lower(), int(n), int(mm), float(value))
                )
            elif m := named.match(line):
                kind, axis, n, mm, value = m.groups()
                entries.append((kind, axis, int(n), int(mm), float(value)))
            elif m := re.search(r"([\d.]+(?:[eE][-+]?\d+)?)\s*m\s*=\s*R0", line):
                radius = float(m.group(1))
            elif m := re.search(r"R0\s*=\s*([\d.]+(?:[eE][-+]?\d+)?)\s*(mm|m)?\b", line):
                radius = float(m.group(1)) * (1e-3 if m.group(2) == "mm" else 1.0)
        radius = r0 if r0 is not None else radius
        if radius is None:
            raise ValueError(f"{path}: no reference radius (R0) in the file; pass r0")
        if not entries:
            raise ValueError(f"{path}: no coefficients found")
        size = max(max(n, m) for _, _, n, m, _ in entries) + 1
        coef = {"alpha": torch.zeros(3, size, size, dtype=dtype), "beta": torch.zeros(3, size, size, dtype=dtype)}
        for kind, axis, n, m, value in entries:
            if m > n:
                raise ValueError(f"{path}: coefficient ({n}, {m}) has m > n")
            coef[kind]["xyz".index(axis), n, m] = value
        return cls(coef["alpha"], coef["beta"], radius)


@dataclass
class Scanner:
    """Hardware field model.

    field_strength: B0 [T] (float or tensor; differentiable). Concomitant fields are
        modelled when it is given.
    concomitant: "leading" (quadratic form, the default), "exact" (|B| of the full vector
        field) or None (off).
    z_asymmetry: share a of the z coil's transverse field along x (1 - a along y);
        1/2 for the usual symmetric cylindrical coil.
    nonlinearity: gradient-coil deviation from linearity (``GradientNonlinearity``), or None.
    """

    field_strength: float | Tensor | None = None
    concomitant: str | None = "leading"
    z_asymmetry: float | Tensor = 0.5
    nonlinearity: GradientNonlinearity | None = None

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
    mode: 0 none, 1 leading order, 2 exact (as in the Metal kernel).
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
    enc = pos if scanner.nonlinearity is None else scanner.nonlinearity.encoding_positions(pos)
    if scanner.concomitant is None or scanner.field_strength is None:
        return SpinFields(enc, None, 0)
    f0 = torch.as_tensor(scanner.larmor, dtype=pos.dtype, device=pos.device)
    if scanner.concomitant == "leading":
        return SpinFields(enc, concomitant_coefficients(pos, f0, scanner.z_asymmetry), 1)
    t = transverse_coefficients(pos, scanner.z_asymmetry)
    return SpinFields(enc, torch.cat([t, f0.expand(pos.shape[0], 1)], dim=-1), 2)


__all__ = [
    "CONCOMITANT_MODES",
    "GradientNonlinearity",
    "Scanner",
    "SpinFields",
    "concomitant_coefficients",
    "exact_excess",
    "gradient_monomials",
    "solid_harmonics",
    "spin_fields",
    "transverse_coefficients",
]
