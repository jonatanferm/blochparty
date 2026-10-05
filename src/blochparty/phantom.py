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
    b1: Tensor | None = None  # (N,) relative transmit (B1+) scale of every RF pulse; None = 1

    def __len__(self) -> int:
        return self.pd.shape[0]

    def to(self, dtype=None, device=None) -> Phantom:
        conv = lambda t: None if t is None else t.to(dtype=dtype, device=device)
        return replace(self, pos=conv(self.pos), pd=conv(self.pd), t1=conv(self.t1),
                       t2=conv(self.t2), db0=conv(self.db0), b1=conv(self.b1))


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


GAMMA_BAR = 42.577478518e6  # 1H gyromagnetic ratio / 2 pi [Hz/T]

# Six-peak fat spectrum (Ren et al., J Lipid Res 2008; ISMRM fat-water toolbox): chemical shift
# relative to water [ppm] and relative amplitude. Fat resonates below water: negative shifts.
FAT_PEAKS = ((-3.80, 0.087), (-3.40, 0.693), (-2.60, 0.128), (-1.94, 0.004), (-0.39, 0.039), (0.60, 0.048))


def _subvoxel_counts(spins_per_voxel, ndim: int) -> tuple[int, ...]:
    """Spins per voxel side along each map axis (z, y, x order)."""
    if isinstance(spins_per_voxel, int):
        side = round(spins_per_voxel ** (1 / ndim))
        if side**ndim != spins_per_voxel:
            raise ValueError(f"spins_per_voxel={spins_per_voxel} is not a perfect {ndim}-th power; "
                             f"give spins per side along each axis as a tuple instead")
        return (side,) * ndim
    counts = tuple(int(k) for k in spins_per_voxel)
    if len(counts) != ndim or min(counts) < 1:
        raise ValueError(f"spins_per_voxel needs {ndim} positive counts (map axis order), got {spins_per_voxel}")
    return counts


def phantom_from_maps(
    pd: Tensor,
    t1: Tensor,
    t2: Tensor,
    fov: float,
    db0: Tensor | None = None,
    *,
    spins_per_voxel: int | tuple[int, ...] = 1,
    jitter: str | None = None,
    seed: int = 0,
    t2prime: Tensor | None = None,
    t2prime_sampling: str = "quantile",
    b1: Tensor | None = None,
    fat_fraction: Tensor | None = None,
    field_strength: float = 3.0,
    fat_t1=None,
    fat_t2=None,
    fat_peaks=FAT_PEAKS,
) -> Phantom:
    """Spins from 2D or 3D maps (ordered (z, y, x), see ``grid_positions``); voxels with
    PD == 0 are dropped. By default one spin per voxel centre.

    spins_per_voxel: sub-voxel isochromats, each carrying PD / K of its voxel. An int K
        must be a perfect square (2D) or cube (3D); a tuple gives spins per voxel side
        along each map axis, e.g. ``(1, 8)`` for 8 spins along x in 2D. Spins sit on a
        regular sub-grid (cell centres), or, with ``jitter="stratified"``, at one uniform
        random point per sub-cell (seeded by ``seed``). Rule of thumb (see
        ``validation/RESULTS.md``): spins per voxel side must exceed the cycles of
        dephasing across a voxel (gradient moment x voxel size) that still carry signal;
        with fewer, those configurations alias back as spurious echoes.
    t2prime: T2' map [s] (inf = none). Every spin of a voxel gets a static frequency
        offset drawn from a Lorentzian of half-width 1 / (2 pi T2'), so the voxel's FID
        decays with 1 / T2* = 1 / T2 + 1 / T2' while a spin echo refocuses it. With
        ``t2prime_sampling="quantile"`` (default) a voxel's K spins get the K Cauchy
        quantiles at (j + 1/2) / K, shuffled over the voxel's positions (seeded);
        ``"random"`` draws them. Needs many spins per voxel: with K quantiles the FID
        follows exp(-t / T2') to ~0.5 / sqrt(K) (of the t = 0 signal) over the first
        few T2', ~2 / sqrt(K) out to K T2' / 4 (``tests/test_phantom_t2prime.py``). One
        spin gets offset 0.
    b1: relative transmit (B1+) map; every RF pulse is scaled by it (see ``Phantom.b1``).
    fat_fraction: fat signal fraction map in [0, 1]. Water spins keep PD (1 - f); every
        spin of a voxel with f > 0 also gets one fat spin per peak of ``fat_peaks``
        ((shift [ppm], relative amplitude) pairs, amplitudes normalised to sum 1) with
        PD f a_p, T1 / T2 ``fat_t1`` / ``fat_t2`` (scalars or maps, default the 3 T
        ``TISSUES["fat"]`` values) and off-resonance shift x 1e-6 x GAMMA_BAR x
        ``field_strength`` [T] on top of the voxel's: fat precesses slower than water.
        Fat spins come after all water spins.

    Every map may require grad; the phantom's tensors are differentiable functions of
    them (positions and the random draws are not).
    """
    mask = pd > 0
    pos = grid_positions(pd.shape, fov, dtype=pd.dtype)[mask]
    db0 = torch.zeros_like(pd) if db0 is None else db0
    take = lambda m: None if m is None else m[mask]
    pd_v, t1_v, t2_v, db0_v, b1_v = pd[mask], t1[mask], t2[mask], db0[mask], take(b1)
    counts = _subvoxel_counts(spins_per_voxel, pd.ndim)
    k = math.prod(counts)
    if jitter not in (None, "stratified"):
        raise ValueError(f"jitter must be None or 'stratified', got {jitter!r}")
    if t2prime_sampling not in ("quantile", "random"):
        raise ValueError(f"t2prime_sampling must be 'quantile' or 'random', got {t2prime_sampling!r}")
    if k == 1 and jitter is None and t2prime is None and fat_fraction is None:  # spins at voxel centres
        return Phantom(pos=pos, pd=pd_v, t1=t1_v, t2=t2_v, db0=db0_v, b1=b1_v)

    gen = torch.Generator().manual_seed(seed)
    n_vox = pos.shape[0]
    if k > 1 or jitter is not None:
        # offset of each spin from its voxel centre, as a fraction of the voxel, per axis (x, y, z)
        frac = torch.zeros(n_vox, k, 3, dtype=pd.dtype)
        cells = torch.cartesian_prod(*(torch.arange(c) for c in counts)).reshape(k, len(counts))
        u = (torch.rand(n_vox, k, len(counts), generator=gen, dtype=pd.dtype) if jitter == "stratified"
             else torch.full((n_vox, k, len(counts)), 0.5, dtype=pd.dtype))
        sides = torch.tensor(counts, dtype=pd.dtype)
        frac[..., : len(counts)] = ((cells + u) / sides - 0.5).flip(-1)  # map axes (z, y, x) -> (x, y, z)
        voxel = torch.tensor([fov / n for n in reversed(pd.shape)] + [0.0] * (3 - pd.ndim), dtype=pd.dtype)
        pos = (pos[:, None, :] + frac * voxel).reshape(-1, 3)
    rep = lambda v: None if v is None else v.repeat_interleave(k)
    pd_s, t1_s, t2_s, db0_s, b1_s = rep(pd_v / k if k > 1 else pd_v), rep(t1_v), rep(t2_v), rep(db0_v), rep(b1_v)

    if t2prime is not None:
        if t2prime_sampling == "quantile":
            q = (torch.arange(k, dtype=pd.dtype) + 0.5) / k
            q = q[torch.argsort(torch.rand(n_vox, k, generator=gen), dim=1)]  # shuffled per voxel
        else:
            q = torch.rand(n_vox, k, generator=gen, dtype=pd.dtype)
        cauchy = torch.tan(math.pi * (q - 0.5))  # standard Cauchy quantiles
        half_width = 1.0 / (2 * math.pi * t2prime[mask])  # Lorentzian HWHM [Hz]
        db0_s = db0_s + (half_width[:, None] * cauchy).reshape(-1)

    if fat_fraction is None:
        return Phantom(pos=pos, pd=pd_s, t1=t1_s, t2=t2_s, db0=db0_s, b1=b1_s)

    ff = rep(fat_fraction[mask])
    fat = ff > 0
    peaks = torch.tensor(fat_peaks, dtype=pd.dtype).reshape(-1, 2)
    shift_hz, amp = peaks[:, 0] * (1e-6 * GAMMA_BAR * field_strength), peaks[:, 1] / peaks[:, 1].sum()
    n_peaks, n_fat = peaks.shape[0], int(fat.sum())

    def fat_param(v, default):  # scalar or map -> per fat spin
        v = torch.as_tensor(default if v is None else v, dtype=pd.dtype)
        return v.expand(n_fat) if v.ndim == 0 else rep(v[mask])[fat]

    tile = lambda v: v.repeat(n_peaks)
    fat_pd = ((pd_s * ff)[fat][None, :] * amp[:, None]).reshape(-1)
    return Phantom(
        pos=torch.cat([pos, pos[fat].repeat(n_peaks, 1)]),
        pd=torch.cat([pd_s * (1 - ff), fat_pd]),
        t1=torch.cat([t1_s, tile(fat_param(fat_t1, TISSUES["fat"][1]))]),
        t2=torch.cat([t2_s, tile(fat_param(fat_t2, TISSUES["fat"][2]))]),
        db0=torch.cat([db0_s, (db0_s[fat][None, :] + shift_hz[:, None]).reshape(-1)]),
        b1=None if b1_s is None else torch.cat([b1_s, tile(b1_s[fat])]),
    )


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


def shepp_logan(n: int = 64, fov: float = 0.22, ndim: int = 2, dtype=torch.float64, *,
                chemical_shift: bool = False, **kwargs) -> Phantom:
    """Shepp-Logan phantom; ``kwargs`` go to ``phantom_from_maps`` (sub-voxel spins, T2',
    B1+, ...). ``chemical_shift`` makes the "fat" tissue fat (fat fraction 1, multi-peak
    spectrum) instead of a water-like spin with fat's T1 / T2."""
    maps = shepp_logan_maps(n, ndim=ndim, dtype=dtype)
    if chemical_shift:
        kwargs["fat_fraction"] = (maps["label"] == LABELS.index("fat")).to(dtype)
    return phantom_from_maps(maps["pd"], maps["t1"], maps["t2"], fov, **kwargs)


# --- B0 field maps ------------------------------------------------------------------------
# Off-resonance [Hz] at given positions (N, 3) [m], with B0 along z; add the result to a
# phantom's db0 (``replace(ph, db0=ph.db0 + b0_shim(ph.pos, z=20.0))``). Evaluated at the
# spin positions, so sub-voxel spins sample the field inside their voxel.


def b0_shim(pos: Tensor, *, offset=0.0, x=0.0, y=0.0, z=0.0, z2=0.0, zx=0.0, zy=0.0, x2y2=0.0, xy=0.0) -> Tensor:
    """Smooth field of first- and second-order shim terms (solid harmonics, so every term
    solves Laplace's equation, as a real field must): ``offset`` [Hz]; ``x``, ``y``,
    ``z`` [Hz/m]; ``z2`` * (z^2 - (x^2 + y^2) / 2), ``zx`` * z x, ``zy`` * z y,
    ``x2y2`` * (x^2 - y^2), ``xy`` * x y [Hz/m^2]. Coefficients may be tensors."""
    px, py, pz = pos.unbind(-1)
    return (offset + x * px + y * py + z * pz + z2 * (pz**2 - 0.5 * (px**2 + py**2)) + zx * pz * px
            + zy * pz * py + x2y2 * (px**2 - py**2) + xy * px * py)


def b0_sphere(pos: Tensor, centre, radius: float, dchi: float, field_strength: float = 3.0) -> Tensor:
    """Field of a sphere whose susceptibility differs from its surroundings by ``dchi``
    (SI, e.g. ~9.4e-6 for air in tissue), in a field of ``field_strength`` [T] along z.
    Closed form, Lorentz-sphere corrected (the field a spin feels): zero inside, a dipole
    outside, gamma_bar B0 dchi / 3 (a / r)^3 (3 cos^2(theta) - 1)."""
    d = pos - torch.as_tensor(centre, dtype=pos.dtype)
    r2 = (d**2).sum(-1)
    cos2 = d[..., 2] ** 2 / r2.clamp_min(1e-300)
    out = GAMMA_BAR * field_strength * dchi / 3 * (radius**2 / r2.clamp_min(1e-300)) ** 1.5 * (3 * cos2 - 1)
    return torch.where(r2 > radius**2, out, torch.zeros_like(out))


def b0_cylinder(pos: Tensor, centre, radius: float, dchi: float, axis=(1.0, 0.0, 0.0),
                field_strength: float = 3.0) -> Tensor:
    """Field of an infinite cylinder (through ``centre``, along ``axis``) whose susceptibility
    differs by ``dchi`` (SI), B0 along z, theta the angle between axis and B0. Closed form,
    Lorentz corrected: inside gamma_bar B0 dchi / 6 (3 cos^2(theta) - 1); outside
    gamma_bar B0 dchi / 2 sin^2(theta) (a / rho)^2 cos(2 phi), rho the distance from the
    axis and phi the angle around it from B0's projection."""
    n = torch.as_tensor(axis, dtype=pos.dtype)
    n = n / torch.linalg.vector_norm(n)
    d = pos - torch.as_tensor(centre, dtype=pos.dtype)
    perp = d - (d @ n)[..., None] * n  # component perpendicular to the axis
    rho2 = (perp**2).sum(-1)
    cos_t2 = n[2] ** 2
    scale = GAMMA_BAR * field_strength * dchi
    inside = scale / 6 * (3 * cos_t2 - 1)
    # sin^2(theta) cos(2 phi) = 2 perp_z^2 / rho^2 - sin^2(theta)
    outside = scale / 2 * radius**2 / rho2.clamp_min(1e-300) * (2 * perp[..., 2] ** 2 / rho2.clamp_min(1e-300)
                                                               - (1 - cos_t2))
    return torch.where(rho2 > radius**2, outside, inside.expand_as(outside))
