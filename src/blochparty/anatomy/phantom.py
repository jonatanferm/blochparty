"""An anatomy into spins: tissue properties, field maps (B0 from susceptibility, shimmed), macroscopic R2',
texture, and ``to_phantom``."""

from __future__ import annotations

import math
import warnings

import torch
from torch import Tensor

from ..constants import GAMMA_BAR
from ..exchange import BoundPool
from ..maps import FieldMaps, sample
from ..phantom import FAT_PEAKS, Phantom, _subvoxel_counts
from ..susceptibility import shim, susceptibility_to_b0
from .grid import AIR_CHI, Anatomy, _triple
from .tables import DEFAULT_TISSUES


def _properties(anatomy, table, field_T):
    if field_T not in table:
        raise ValueError(f"no tissue table for {field_T} T; provide explicit values (no field interpolation)")
    props = table[field_T]
    missing = [k for k, v in anatomy.fractions.items() if k not in props and v.any()]
    if missing:
        raise ValueError(f"missing tissue properties at {field_T} T: {missing}")
    return props


def anatomy_field_maps(
    anatomy, table=DEFAULT_TISSUES, field_T=3.0, *, shim_order=2, brain_mask=None, b0_dir=(0, 0, 1), padding=2.0
):
    """Return (tissue susceptibility FieldMaps, scanner correction FieldMaps).

    Sample the FIRST into moving Phantom.db0, and pass the SECOND as scanner
    maps to simulate(..., maps=...). For stationary anatomy their sum is the
    shimmed map. Recompute the first for rotations relative to B0. Full-source
    computation precedes any spin-region crop. Empty/planar masks cannot fit
    all 3D harmonics: choose a lower order or use shim_order=None explicitly.
    """
    ref = next(iter(anatomy.fractions.values()))
    chi = anatomy.chi
    if chi is None:
        props = _properties(anatomy, table, field_T)
        chi = torch.full_like(ref, AIR_CHI)
        for name, frac in anatomy.fractions.items():
            if name in props:
                chi = chi + frac * (props[name].chi.mean - AIR_CHI)
    background = anatomy.chi_background if anatomy.chi is not None else AIR_CHI
    df = susceptibility_to_b0(chi, anatomy.voxel, field_T, b0_dir=b0_dir, padding=padding, background=background)
    correction = torch.zeros_like(df)
    if shim_order is not None:
        if brain_mask is None:
            brain_mask = (
                anatomy.maps["brain_mask"] > 0.5
                if "brain_mask" in anatomy.maps
                else sum(
                    (anatomy.fractions[k] for k in ("wm", "gm", "csf") if k in anatomy.fractions), torch.zeros_like(ref)
                )
                > 0.5
            )
        residual, _ = shim(df, brain_mask, shim_order, voxel=anatomy.voxel, origin=anatomy.origin)
        correction = residual - df
    args = (anatomy.origin, tuple(reversed(anatomy.voxel)))
    return FieldMaps(*args, b0=df), FieldMaps(*args, b0=correction)


# The R2* measurement a T2* value or map came from (voxel (dz, dy, dx) [m], echo times [s]):
# the macroscopic part of its decay is estimated at these, not at the simulation grid.
# Default: the protocol of the 3 T WM/GM table values (Shin et al. 2025: 1 x 1 mm in plane, 2 mm
# slices, assumed axial; 43 echoes TE 3 + 1.47 k ms to 66.18 ms). Other table T2* entries are
# priors without a protocol of their own and use the same.
DEFAULT_R2STAR_ACQUISITION = {"voxel": (2e-3, 1e-3, 1e-3), "te": tuple(3e-3 + 1.47e-3 * k for k in range(43))}
# AHEAD (MP2RAGEME, Caan et al. 2019: second-inversion echoes 3 / 11.5 / 19 / 27.5 ms). Voxel 0.64 mm:
# the commonly cited AHEAD resolution; the papers say "submillimetre" (unverified here).
AHEAD_R2STAR_ACQUISITION = {"voxel": (0.64e-3, 0.64e-3, 0.64e-3), "te": (3e-3, 11.5e-3, 19e-3, 27.5e-3)}


def estimate_macroscopic_r2prime(df: Tensor, voxel, acquisition=None) -> Tensor:
    """The part of a measured R2* that the voxel-scale field ``df`` [Hz] on a (z, y, x) grid of
    spacing ``voxel`` [m] causes by itself [1/s]: each axis's linear gradient g [Hz/m] dephases a
    box voxel of the acquisition's size d as |sinc(g d t)| (independent axes multiply), and a
    log-linear magnitude fit over the acquisition's echo times turns that into a rate, as a
    multi-echo R2* fit would (Fernandez-Seara & Wehrli, MRM 2000; Yablonskiy, MRM 1998). Gradients
    are central differences on the grid; |sinc| is floored at 1e-3. Fields resolved finer than the
    grid, and curvature within the voxel, are not included. Where this alone exceeds the measured
    R2* - R2 (a field too strong for the stated acquisition), ``to_phantom`` clips R2' at 0."""
    acq = DEFAULT_R2STAR_ACQUISITION if acquisition is None else acquisition
    te = torch.as_tensor(acq["te"], dtype=df.dtype)
    if te.ndim != 1 or len(te) < 2 or (te <= 0).any():
        raise ValueError("acquisition needs two or more positive echo times")
    d = _triple(acq["voxel"], "acquisition voxel", True)
    grads = [
        torch.gradient(df, spacing=float(h), dim=ax)[0] if df.shape[ax] > 1 else torch.zeros_like(df)
        for ax, h in enumerate(voxel)
    ]  # a single-voxel axis has no resolved gradient
    tc = te - te.mean()
    num = torch.zeros_like(df)
    for k in range(len(te)):  # sum_k tc_k ln F(t_k), one echo at a time (memory)
        lnf = sum(torch.sinc(g * di * te[k]).abs().clamp_min(1e-3).log() for g, di in zip(grads, d))
        num = num + tc[k] * lnf
    return (-num / tc.square().sum()).clamp_min(0)


_VARIABLE = ("t1", "t2", "pd")


def gaussian_random_field(anatomy: Anatomy, length: float, generator: torch.Generator) -> tuple[Tensor, tuple, tuple]:
    """A zero-mean, unit-variance Gaussian random field over the anatomy's extent with covariance
    exp(-r^2 / (2 length^2)) [length in m], on a coarse (z, y, x) grid of spacing length / 4 padded
    by 3 lengths (so periodic copies do not correlate across the head): (grid, origin (x, y, z),
    spacing (x, y, z)) for ``maps.sample``. White noise is filtered by the square root of the
    spectral density exp(-2 pi^2 length^2 |k|^2) and divided by the exact discrete standard
    deviation sqrt(mean S); trilinear sampling between nodes lowers the SD by < 1 %."""
    if not math.isfinite(length) or length <= 0:
        raise ValueError("variability length must be positive [m]")
    h, pad = length / 4, 3 * length
    lo = [o - v / 2 - pad for o, v in zip(anatomy.origin, reversed(anatomy.voxel))]  # (x, y, z)
    n = [math.ceil((v * m + 2 * pad) / h) + 1 for v, m in zip(reversed(anatomy.voxel), reversed(anatomy.shape))]
    shape = tuple(reversed(n))  # (z, y, x)
    k = [torch.fft.fftfreq(m, d=h, dtype=torch.float64) for m in shape]
    k2 = k[0][:, None, None] ** 2 + k[1][None, :, None] ** 2 + k[2][None, None, :] ** 2
    spectrum = torch.exp(-2 * math.pi**2 * length**2 * k2)
    white = torch.randn(shape, generator=generator, dtype=torch.float64)
    field = torch.fft.ifftn(spectrum.sqrt() * torch.fft.fftn(white, norm="ortho"), norm="ortho").real
    return field / spectrum.mean().sqrt(), tuple(lo), (h, h, h)


def _variability_factors(anatomy, names, variability, region=None):
    """{(tissue, property): (nz, ny, nx) mean-1 log-normal factor} with relative SD per property.
    The fields span the whole anatomy (one realisation for every crop); ``region`` (three unit-step
    slices, z/y/x) evaluates them at its voxels only, with the same values as cropping afterwards."""
    import zlib

    unknown = set(variability) - {*_VARIABLE, "length", "seed"}
    if unknown or "length" not in variability:
        raise ValueError(
            f"variability takes {_VARIABLE} relative SDs, 'length' [m] and optional 'seed'; got {sorted(variability)}"
        )
    seed = int(variability.get("seed", 0))
    ref = next(iter(anatomy.fractions.values()))
    bounds = (
        [s.indices(n)[:2] for s, n in zip(region, anatomy.shape)]
        if region is not None
        else [(0, n) for n in anatomy.shape]
    )
    nz, ny, nx = (hi - lo for lo, hi in bounds)
    idx = torch.cartesian_prod(*(torch.arange(lo, hi) for lo, hi in bounds)).to(torch.float64)
    centres = torch.tensor(anatomy.origin, dtype=torch.float64) + idx.flip(-1) * torch.tensor(anatomy.voxel[::-1])
    out = {}
    for name in names:
        for prop in _VARIABLE:
            sd = float(variability.get(prop, 0.0))
            if sd < 0 or not math.isfinite(sd):
                raise ValueError("variability SDs must be nonnegative")
            if sd == 0:
                continue
            gen = torch.Generator().manual_seed((seed * 1_000_003 + zlib.crc32(f"{name}/{prop}".encode())) % 2**63)
            grid, origin, spacing = gaussian_random_field(anatomy, float(variability["length"]), gen)
            g = sample(
                grid[None],
                torch.tensor(origin, dtype=torch.float64),
                torch.tensor(spacing, dtype=torch.float64),
                centres,
            )[:, 0].reshape(nz, ny, nx)
            sigma2 = math.log1p(sd**2)  # log-normal with mean 1 and SD sd
            out[name, prop] = torch.exp(math.sqrt(sigma2) * g - sigma2 / 2).to(ref.dtype)
    return out


def to_phantom(
    anatomy: Anatomy,
    table=DEFAULT_TISSUES,
    field_T=3.0,
    *,
    fov_region=None,
    spins_per_voxel=8,
    jitter="stratified",
    seed=0,
    sampling="per_tissue",
    mt=False,
    diffusion=False,
    chi_to_b0=True,
    shim_order=2,
    t2star="auto",
    macroscopic_r2prime=None,
    field_maps=None,
    variability=None,
    r2star_acquisition=None,
) -> Phantom:
    """Assemble independent tissue spins, never average their relaxation constants.

    ``per_tissue`` gives K spins to each nonzero tissue/voxel, PD=fraction*PD/K
    (exact conservation). ``categorical`` draws K tissue identities per voxel;
    missing mass is air. It is unbiased but only conserves each tissue's PD in
    expectation. ``systematic`` assigns them from the evenly spaced (j + U) / K, one U per
    voxel (systematic resampling), dealt to the subcells at random: each tissue gets
    floor(K f) or ceil(K f) spins, unbiased, with a voxel error O(1/K) instead of O(1/sqrt K)
    (docs/PERFORMANCE.md 2.6). ``proportional`` gives each nonzero tissue max(1, round(K f))
    spins at random subcells and PD = fraction*PD/n: each voxel's composition exact like
    ``per_tissue`` (no salt-and-pepper from composition draws) at about K spins per voxel.
    K is a perfect cube or (kz,ky,kx). ``jitter="inplane"`` jitters x and y only and puts z at the
    subcell centres: through a 2D slice's steep profile, random z positions make each pixel's slice
    integral noisy (a visible grain), a z quadrature does not. Jitter samples each subcell;
    None uses subcell centres. Seeds change positions (and categorical draws),
    not material properties. Sampled B0 necessarily changes with positions.
    PD is per anatomy voxel, matching phantom_from_maps; multiply by physical
    voxel volume when comparing absolute signals across anatomy resolutions.

    ``t2star='auto'`` (default): with chi fields, ``'separate'``; without, R2' = R2* - R2.
    ``'separate'``: a measured R2* (map R2star or table T2*) contains R2, microscopic R2'
    and the dephasing from the voxel-scale field this simulation computes itself; the last is
    estimated by ``estimate_macroscopic_r2prime`` from the field (tissue + shim) at the voxel size and
    echo times of the R2* measurement (``r2star_acquisition``, default
    ``DEFAULT_R2STAR_ACQUISITION``; ``AHEAD_R2STAR_ACQUISITION`` for ``load_ahead``) and
    subtracted: R2' = max(R2* - R2 - R2'_macro, 0) goes onto the spins, and the simulated field
    supplies the rest. ``'table'`` uses R2*=map R2star or tissue table, subtracts R2, and
    REQUIRES ``macroscopic_r2prime`` [1/s] when chi is enabled. This is a supplied
    calibration, not a universal linewidth inferred from field gradients.
    Alternatively maps['intrinsic_r2prime'] supplies already separated R2'.
    ``'off'`` disables broadening. Uses the same stratified Cauchy quantiles as
    phantom_from_maps; K=1 has only the median (zero), so finite R2' requires K>1.
    Quantile assignments use a separate fixed RNG; spatial seed does not alter
    their distribution. Fat classes use phantom.FAT_PEAKS (Ren et al. 2008).

    Optional R1/R2 maps override tissue rates voxelwise. M0 gives total voxel
    density, distributed among signal-producing tissues in proportion to their
    memberships (not multiplied by the air fraction a second time). MT converts
    total bound fraction p to BoundPool.f=p/(1-p), assumes bound T1=free T1,
    and uses model='rate'. Missing requested MT/diffusion parameters raise.
    ``field_maps`` can reuse a precomputed (tissue, scanner-shim) pair. This
    stationary combined db0 is NOT suitable for motion; use anatomy_field_maps
    and keep only susceptibility attached to the moving tissue instead.

    fov_region is three voxel slices: B0 is computed on the full anatomy first.
    Apple GPU full 1-mm head (categorical K=1, 5,328,321 spins): 32-sample FID
    after 50 RF segments, 0.686 s / 7.77 million spins/s; phantom 149 MB, process
    peak RSS 3.64 GB, MPS driver allocation 1.08 GB. This benchmark excludes B0,
    T2', MT and diffusion; it is not full k-space imaging speed. Reproduction
    and memory-accounting limits: validation/anatomy/RESULTS.md.
    Maps ``t1_scale``, ``t2_scale``, ``pd_scale`` (if the anatomy holds them) multiply every non-fat tissue's T1, T2 and
    PD in each voxel: a residual fitted to a subject's own scans (validation/anatomy/prepare_ixi.py); fat is left out
    (its apparent T2 depends on the sequence through J coupling).
    ``variability`` (default None: none) = {"t1": sd, "t2": sd, "pd": sd, "length": L [m],
    "seed": 0}: each tissue's T1, T2 and PD are multiplied by its own mean-1 log-normal field
    of relative SD ``sd`` and Gaussian correlation exp(-r^2 / 2L^2) (``gaussian_random_field``),
    evaluated at voxel centres on the full anatomy before ``fov_region``. Its seed is separate
    from ``seed`` (positions), so jitter does not change the texture. It models within-tissue
    texture, not the population SDs of the table (``Distribution.sd``).
    """
    if sampling not in ("per_tissue", "categorical", "systematic", "proportional") or jitter not in (
        None,
        "stratified",
        "inplane",
    ):
        raise ValueError("invalid sampling or jitter mode")
    if t2star not in ("auto", "off", "table", "separate"):
        raise ValueError("t2star must be auto, separate, off or table")
    if t2star == "auto":
        t2star = "separate" if chi_to_b0 else "table"
    if t2star == "separate" and not chi_to_b0:
        raise ValueError("t2star='separate' needs chi_to_b0 (the field it separates)")
    if anatomy.map_field_T is not None and field_T != anatomy.map_field_T:
        raise ValueError(f"relaxation maps were acquired at {anatomy.map_field_T} T; no automatic field scaling")
    counts = _subvoxel_counts(spins_per_voxel, 3)
    k = math.prod(counts)
    props = _properties(anatomy, table, field_T)
    maps_pair = None
    if chi_to_b0:
        maps_pair = (
            field_maps if field_maps is not None else anatomy_field_maps(anatomy, table, field_T, shim_order=shim_order)
        )
    if t2star == "table" and chi_to_b0 and macroscopic_r2prime is None and "intrinsic_r2prime" not in anatomy.maps:
        raise ValueError("observed T2* plus chi requires macroscopic_r2prime calibration")
    if macroscopic_r2prime is not None:
        macro = torch.as_tensor(macroscopic_r2prime, dtype=next(iter(anatomy.fractions.values())).dtype)
        if macro.ndim and macro.shape != anatomy.shape or not torch.isfinite(macro).all() or (macro < 0).any():
            raise ValueError("macroscopic_r2prime must be a nonnegative scalar or full anatomy map")
        if fov_region is not None and macro.ndim:
            macro = macro[fov_region]
    else:
        macro = 0.0
    if fov_region is not None:
        if len(fov_region) != 3 or any(
            not isinstance(s, slice) or s.indices(n)[2] != 1 for s, n in zip(fov_region, anatomy.shape)
        ):
            raise ValueError("fov_region needs three unit-step slices in z/y/x order")
        bounds = [s.indices(n)[:2] for s, n in zip(fov_region, anatomy.shape)]
    factors = {}
    if variability:  # one realisation on the full anatomy, evaluated where the spins are
        live = [n for n, p in props.items() if n in anatomy.fractions and p.pd.mean > 0]
        factors = _variability_factors(anatomy, live, variability, region=fov_region)
    macro_est = None
    if t2star == "separate" and "intrinsic_r2prime" not in anatomy.maps and macroscopic_r2prime is None:
        if any(tuple(fm.b0.shape) != anatomy.shape for fm in maps_pair):
            raise ValueError("t2star='separate' needs field maps on the anatomy grid (anatomy_field_maps)")
        df = sum(fm.b0.to(torch.float64) for fm in maps_pair)
        if fov_region is None:
            macro_est = estimate_macroscopic_r2prime(df, anatomy.voxel, r2star_acquisition)
        else:  # the region and a 1-voxel margin: the central differences inside it are unchanged
            pad = tuple(slice(max(lo - 1, 0), min(hi + 1, n)) for (lo, hi), n in zip(bounds, anatomy.shape))
            est = estimate_macroscopic_r2prime(df[pad], anatomy.voxel, r2star_acquisition)
            macro_est = est[tuple(slice(lo - p.start, hi - p.start) for (lo, hi), p in zip(bounds, pad))]
    if fov_region is not None:
        anatomy = anatomy.crop(fov_region)
    ref = next(iter(anatomy.fractions.values()))
    gen = torch.Generator().manual_seed(seed)
    qgen = torch.Generator().manual_seed(0)
    cells = torch.cartesian_prod(*(torch.arange(c) for c in counts)).reshape(k, 3)
    names = sorted(name for name, p in props.items() if name in anatomy.fractions and p.pd.mean > 0)
    signal_fraction = None
    if "M0" in anatomy.maps:
        density = anatomy.maps["M0"]
        signal_fraction = sum((anatomy.fractions[name] for name in names), torch.zeros_like(ref))
        if (density < 0).any() or ((density > 0) & (signal_fraction == 0)).any():
            raise ValueError("M0 must be nonnegative and supported by signal-producing tissue")
    category = None
    if sampling in ("categorical", "systematic"):
        # Sequential thresholds avoid a (voxels,K,tissues) temporary.
        u = torch.rand((*anatomy.shape, k), dtype=ref.dtype, generator=gen)
        if sampling == "systematic":  # (j + U) / K, one U per voxel, dealt to the subcells at random
            u = (torch.arange(k, dtype=ref.dtype) + u[..., :1]) / k
            u = torch.gather(u, -1, torch.argsort(torch.rand(u.shape, generator=gen), dim=-1))
        category = torch.full(u.shape, -1, dtype=torch.long)
        cumulative = torch.zeros_like(ref)
        for i, name in enumerate(names):
            previous = cumulative
            cumulative = cumulative + anatomy.fractions[name]
            category[(u >= previous[..., None]) & (u < cumulative[..., None])] = i
    parts = {key: [] for key in ("pos", "pd", "t1", "t2", "db0", "f", "rate", "t2b", "diff")}
    active_broadening = False
    for ti, name in enumerate(names):
        p = props[name]
        fraction = anatomy.fractions[name]
        mask = fraction > 0 if category is None else (category == ti).any(-1)
        if "M0" in anatomy.maps:
            mask = mask & (anatomy.maps["M0"] > 0)
        if not mask.any():
            continue
        idx = mask.nonzero()
        n = len(idx)
        u = (
            torch.rand((n, k, 3), dtype=ref.dtype, generator=gen)
            if jitter
            else torch.full((n, k, 3), 0.5, dtype=ref.dtype)
        )
        if jitter == "inplane":  # z at the subcell centres: a quadrature of the slice profile, no z sampling noise
            u[..., 0] = 0.5
        offsets = (cells + u) / torch.tensor(counts) - 0.5
        origin = torch.tensor(anatomy.origin, dtype=ref.dtype)
        spacing = torch.tensor(anatomy.voxel, dtype=ref.dtype)
        pos = origin + (idx[:, None, :] + offsets).flip(-1) * spacing.flip(-1)
        select = torch.ones((n, k), dtype=torch.bool) if category is None else category[mask] == ti
        if sampling == "proportional":  # n_t = max(1, round(k f)) of the k subcells, at random
            n_t = (fraction[mask] * k).round().clamp(1, k)
            rank = torch.argsort(torch.argsort(torch.rand((n, k), generator=gen), dim=1), dim=1)
            select = rank < n_t[:, None]
        expand = lambda value: (value[:, None].expand(-1, k) if value.ndim else value.expand(n, k))[select]
        parameter = lambda key, default: anatomy.maps[key][mask] if key in anatomy.maps else ref.new_tensor(default)
        density = parameter("M0", p.pd.mean)
        if signal_fraction is not None:
            density = density / signal_fraction[mask]
        r1, r2 = parameter("R1", 1 / p.t1.mean), parameter("R2", 1 / p.t2.mean)
        r2_base = r2  # R2' = R2* - R2 uses the untextured R2: texture must not create negative R2'
        if (name, "t1") in factors:  # T1 x factor
            r1 = r1 / factors[name, "t1"][mask]
        if (name, "t2") in factors:
            r2 = r2 / factors[name, "t2"][mask]
        if (name, "pd") in factors:
            density = density * factors[name, "pd"][mask]
        if not (p.fat_fraction and p.fat_fraction.mean > 0.5):  # per-voxel residual multipliers (maps), not on fat
            if "t1_scale" in anatomy.maps:
                r1 = r1 / anatomy.maps["t1_scale"][mask]
            if "t2_scale" in anatomy.maps:
                r2 = r2 / anatomy.maps["t2_scale"][mask]
            if "pd_scale" in anatomy.maps:
                density = density * anatomy.maps["pd_scale"][mask]
        if (density < 0).any() or (r1 <= 0).any() or (r2 <= 0).any():
            raise ValueError("M0 must be nonnegative; R1 and R2 positive")
        pd = expand(density) / k
        if sampling == "proportional":
            pd = expand(density * fraction[mask] / n_t)
        elif category is None:
            pd = pd * expand(fraction[mask])
        pos = pos[select]
        db0 = torch.zeros_like(pd)
        if maps_pair is not None:
            for fm in maps_pair:
                db0 += sample(fm.b0[None], ref.new_tensor(fm.origin), ref.new_tensor(fm.spacing), pos)[:, 0]
        if t2star != "off":
            rate = None
            if "intrinsic_r2prime" in anatomy.maps:
                rate = anatomy.maps["intrinsic_r2prime"][mask]
            elif t2star in ("table", "separate"):
                rstar = parameter("R2star", 1 / p.t2star.mean if p.t2star else 1 / p.t2.mean)
                if macro_est is not None:  # an estimate: clip, a mismatch is not an inconsistent calibration
                    rate = (rstar - r2_base - macro_est[mask].to(rstar.dtype)).clamp_min(0)
                else:
                    rate = rstar - r2_base - (macro[mask] if isinstance(macro, Tensor) and macro.ndim else macro)
            if rate is not None:
                if (rate < -1e-6).any():
                    raise ValueError("R2* - R2 - macroscopic R2' is negative; incompatible calibration")
                active_broadening |= bool((rate > 0).any())
                q = (torch.arange(k, dtype=ref.dtype) + 0.5) / k
                q = q[torch.argsort(torch.rand(n, k, generator=qgen), dim=1)]
                db0 += expand(rate.clamp_min(0)) * torch.tan(math.pi * (q - 0.5))[select] / (2 * math.pi)
        f = p.bound_fraction.mean if p.bound_fraction else 0.0
        if mt and (p.bound_fraction is None or (f > 0 and (p.exchange_rate is None or p.bound_t2 is None))):
            raise ValueError(f"missing MT properties for {name}")
        if diffusion and p.diffusivity is None:
            raise ValueError(f"missing diffusivity for {name}")
        # Chemical species share spatial/relaxation/MT parameters within this class (the myelin water,
        # if any, has its own T2).
        ff = p.fat_fraction.mean if p.fat_fraction else 0.0
        mwf = p.myelin_water_fraction.mean if p.myelin_water_fraction else 0.0
        species = [((1 - ff) * (1 - mwf), 0.0, None)] if ff < 1 else []
        if ff < 1 and mwf > 0:
            species.append(((1 - ff) * mwf, 0.0, p.myelin_water_t2.mean))
        if ff > 0:
            total_amp = sum(amp for _, amp in FAT_PEAKS)
            species += [(ff * amp / total_amp, shift * 1e-6 * GAMMA_BAR * field_T, None) for shift, amp in FAT_PEAKS]
        for weight, shift, t2_own in species:
            parts["pos"].append(pos)
            parts["pd"].append(pd * weight)
            parts["t1"].append(1 / expand(r1))
            parts["t2"].append(1 / expand(r2) if t2_own is None else torch.full_like(pd, t2_own))
            parts["db0"].append(db0 + shift)
            if mt:
                parts["f"].append(torch.full_like(pd, f / (1 - f)))
                parts["rate"].append(torch.full_like(pd, p.exchange_rate.mean if f else 0.0))
                parts["t2b"].append(torch.full_like(pd, p.bound_t2.mean if f else 12e-6))
            if diffusion:
                parts["diff"].append(torch.full_like(pd, p.diffusivity.mean))
    if active_broadening and k == 1:
        warnings.warn(
            "one spin per tissue/voxel cannot represent T2' broadening; use more spins", UserWarning, stacklevel=2
        )
    cat = lambda key: (
        torch.cat(parts[key]) if parts[key] else torch.empty((0, 3) if key == "pos" else (0,), dtype=ref.dtype)
    )
    ph = Phantom(*(cat(key) for key in ("pos", "pd", "t1", "t2", "db0")))
    if mt:
        ph.bound = BoundPool(cat("f"), cat("rate"), ph.t1, cat("t2b"), model="rate")
    if diffusion:
        ph.diffusion = cat("diff")
    return ph
