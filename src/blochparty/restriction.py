"""Restricted diffusion: every spin walks inside its own impermeable compartment.

``Restriction(kind, size, axis)``: kind "planes" (a slab of width ``size`` between two planes
with normal ``axis``), "cylinder" (radius ``size``, axis ``axis``) or "sphere" (radius
``size``); ``size`` (N,) or scalar [m], ``axis`` (N, 3) or (3,), unit (planes, cylinder). The
compartment is centred at the spin's position; the walker starts uniformly inside it (from
``rng.uniforms``, so every engine starts it at the same place).

Walls make the free Brownian segment of ``bloch._walk`` inexact, so a segment is cut into
sub-steps of at most ``tau_max = min(size^2 / (200 D))`` (rms step <= size / 10 per axis);
each sub-step moves by sqrt(2 D h) n1, reflects off the walls (exactly for planes,
specularly at the crossing point for cylinders and spheres) and contributes its midpoint
plus the free bridge term sqrt(2 D h / 12) n2 to the segment's effective position (their
average). Errors are first order in the sub-step near walls; the reference checks are the
long-diffusion-time and finite-time short-pulse results (tests/physics/test_restriction.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .rng import normals, uniforms

KINDS = ("planes", "cylinder", "sphere")


@dataclass
class Restriction:
    kind: str
    size: Tensor | float  # slab width or radius [m]
    axis: Tensor | tuple | None = None  # unit normal (planes) or axis (cylinder)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.kind != "sphere" and self.axis is None:
            raise ValueError(f"a {self.kind} restriction needs an axis")

    def expand(self, n: int, like: Tensor) -> Restriction:
        size = torch.as_tensor(self.size, dtype=like.dtype, device=like.device).expand(n)
        axis = None
        if self.axis is not None:
            axis = torch.as_tensor(self.axis, dtype=like.dtype, device=like.device).expand(n, 3)
            axis = axis / axis.norm(dim=-1, keepdim=True)
        return Restriction(self.kind, size, axis)

    def to(self, dtype=None, device=None) -> Restriction:
        conv = lambda v: v.to(dtype=dtype, device=device) if isinstance(v, Tensor) else v
        return Restriction(self.kind, conv(self.size), conv(self.axis))


def tau_max(res: Restriction, diffusivity: Tensor) -> float:
    """Longest sub-step [s]: rms step <= size / 10 per axis for every spin."""
    return float((res.size.detach() ** 2 / (200 * diffusivity.detach())).min())


def n_sub(dt: float, tmax: float) -> int:
    return max(1, math.ceil(dt / tmax))


def reflect(a: Tensor, b: Tensor, res: Restriction) -> Tensor:
    """End offset of a step from offset a (inside) to b, reflected off the walls: exactly for
    planes (mirror at +-L/2); for cylinders and spheres specularly, i.e. the rest of the step
    after the wall crossing is mirrored about the tangent plane there (up to 8 bounces)."""
    if res.kind == "planes":
        n = res.axis
        s = (b * n).sum(-1)
        half = 0.5 * res.size
        t = s
        for _ in range(8):  # mirror at +-L/2 until inside
            t = torch.where(t > half, 2 * half - t, torch.where(t < -half, -2 * half - t, t))
        return b + (t - s)[:, None] * n
    R = res.size[:, None]
    if res.kind == "cylinder":  # only the perpendicular components see the wall
        ax = res.axis
        along = (b * ax).sum(-1, keepdim=True) * ax
        a = a - (a * ax).sum(-1, keepdim=True) * ax
        b = b - along
    else:
        along = torch.zeros_like(b)
    for _ in range(8):
        out = (b * b).sum(-1, keepdim=True) > R * R
        if not bool(out.any()):
            break
        v = b - a
        # |a + t v| = R for t in (0, 1]: t = (-a.v + sqrt((a.v)^2 + |v|^2 (R^2 - |a|^2))) / |v|^2
        vv = (v * v).sum(-1, keepdim=True).clamp_min(1e-300)
        av = (a * v).sum(-1, keepdim=True)
        disc = (av * av + vv * (R * R - (a * a).sum(-1, keepdim=True))).clamp_min(0)
        t = ((-av + torch.sqrt(disc)) / vv).clamp(0, 1)
        h = a + t * v
        nrm = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-300)
        rest = (1 - t) * v
        rest = rest - 2 * (rest * nrm).sum(-1, keepdim=True) * nrm
        a = torch.where(out, h, a)
        b = torch.where(out, h + rest, b)
    return along + b


def start_offsets(res: Restriction, seed: int, idx, like: Tensor) -> Tensor:
    """(N, 3) uniform start offsets inside each compartment (rng.uniforms)."""
    u = torch.as_tensor(uniforms(seed, idx), dtype=like.dtype, device=like.device)
    if res.kind == "planes":
        return ((u[:, 0] - 0.5) * res.size)[:, None] * res.axis
    if res.kind == "cylinder":
        a = res.axis
        # an orthonormal pair perpendicular to a: e1 from the least-aligned coordinate axis
        ref = torch.zeros_like(a)
        ref.scatter_(1, a.abs().argmin(-1, keepdim=True), 1.0)
        e1 = torch.linalg.cross(a, ref)
        e1 = e1 / e1.norm(dim=-1, keepdim=True)
        e2 = torch.linalg.cross(a, e1)
        r = res.size * torch.sqrt(u[:, 0])
        th = 2 * math.pi * u[:, 1]
        return r[:, None] * (torch.cos(th)[:, None] * e1 + torch.sin(th)[:, None] * e2)
    # sphere: radius R u^(1/3), direction from (cos th sin ph...) with cos ph uniform
    r = res.size * u[:, 0] ** (1 / 3)
    cz = 2 * u[:, 1] - 1
    sz = torch.sqrt(torch.clamp(1 - cz * cz, min=0))
    th = 2 * math.pi * u[:, 2]
    return r[:, None] * torch.stack([sz * torch.cos(th), sz * torch.sin(th), cz], -1)


def walk_segment(r: Tensor, centre: Tensor, sqrt_d: Tensor, res: Restriction, dt: Tensor, segment: int, nsub: int,
                 seed: int, idx) -> tuple[Tensor, Tensor]:
    """Effective position over a segment and the end position of restricted walkers."""
    h = dt / nsub
    acc = torch.zeros_like(r)
    for k in range(nsub):
        z = torch.as_tensor(normals(seed, idx, segment, k + 1), dtype=r.dtype, device=r.device)
        step = torch.sqrt(2 * h) * sqrt_d[:, None] * z[:, :3]
        rb = centre + reflect(r - centre, r + step - centre, res)
        acc = acc + 0.5 * (r + rb) + torch.sqrt(2 * h / 12) * sqrt_d[:, None] * z[:, 3:6]
        r = rb
    return acc / nsub, r
