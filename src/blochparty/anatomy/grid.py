"""The anatomy on a physical grid: tissue fractions and properties (``Distribution``, ``TissueProperties``,
``Anatomy``), conservative resampling, and the susceptibilities of air and water."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

AIR_CHI = 0.36e-6  # Jordan et al., JMRI 2013, doi:10.1002/jmri.23762, Theory.
WATER_CHI = -9.05e-6  # Same source, 37 C; absolute SI, not ppm.


@dataclass(frozen=True)
class Distribution:
    """Mean and SD with provenance. SD=None means unreported, not zero variance.

    SDs are metadata, not independent per-spin draws. M4 adds spatial variability.
    ``assumed`` also identifies animal/other-tissue proxies and field extrapolation.
    """

    mean: float
    sd: float | None
    citation: str
    assumed: bool = False

    def __post_init__(self):
        if (
            not math.isfinite(self.mean)
            or (self.sd is not None and (not math.isfinite(self.sd) or self.sd < 0))
            or not self.citation
        ):
            raise ValueError("a distribution needs a finite mean, nonnegative SD (or None), and citation")


@dataclass(frozen=True)
class TissueProperties:
    pd: Distribution
    t1: Distribution
    t2: Distribution
    chi: Distribution  # absolute SI
    t2star: Distribution | None = None  # observed, not intrinsically separated
    bound_fraction: Distribution | None = None  # fraction of TOTAL pool, not M0b/M0f
    exchange_rate: Distribution | None = None  # bound -> free, 1/s
    bound_t2: Distribution | None = None
    diffusivity: Distribution | None = None  # isotropic m^2/s
    fat_fraction: Distribution | None = None
    # free water with a short T2 (myelin water: water between the myelin lamellae, T2 15-40 ms), as a
    # fraction of the free water; the rest has ``t2``. None: one water compartment.
    myelin_water_fraction: Distribution | None = None
    myelin_water_t2: Distribution | None = None

    def __post_init__(self):
        if self.pd.mean < 0 or self.t1.mean <= 0 or self.t2.mean <= 0:
            raise ValueError("PD must be nonnegative and relaxation times positive")
        if self.t2star is not None and not 0 < self.t2star.mean <= self.t2.mean:
            raise ValueError("T2* must be positive and no greater than T2")
        for value in (self.bound_fraction, self.fat_fraction):
            if value is not None and not 0 <= value.mean <= 1:
                raise ValueError("pool fractions must lie in [0,1]")
        if self.bound_fraction is not None and self.bound_fraction.mean == 1:
            raise ValueError("a bound pool requires a nonzero free pool")
        if self.diffusivity is not None and self.diffusivity.mean < 0:
            raise ValueError("diffusivity must be nonnegative")
        if self.exchange_rate is not None and self.exchange_rate.mean < 0:
            raise ValueError("exchange rate must be nonnegative")
        if self.bound_t2 is not None and self.bound_t2.mean <= 0:
            raise ValueError("bound T2 must be positive")
        if self.myelin_water_fraction is not None:
            if not 0 <= self.myelin_water_fraction.mean < 1:
                raise ValueError("myelin water fraction must lie in [0, 1)")
            if self.myelin_water_fraction.mean > 0 and (self.myelin_water_t2 is None or self.myelin_water_t2.mean <= 0):
                raise ValueError("a myelin water fraction needs a positive myelin_water_t2")


def _triple(values, name, positive=False):
    values = tuple(float(v) for v in values)
    if len(values) != 3 or any(not math.isfinite(v) or (positive and v <= 0) for v in values):
        raise ValueError(f"{name} needs three finite" + (" positive" if positive else "") + " values")
    return values


def _renormalize(fractions: dict) -> dict:
    """Divide every class by max(total, 1): resampling by cumulative-sum differences leaves
    rounding of ~(axis length) x eps per class, which can push a sum of ones past 1 + 16 eps
    in float64. Volumes change by that rounding only."""
    total = sum(fractions.values())
    scale = torch.where(total > 1, total, torch.ones_like(total))
    return {k: v / scale for k, v in fractions.items()}


def _resample(grid, old, new, fill=0.0):
    """Exact separable cell-overlap integrals, no dense interpolation matrix."""
    out = grid
    for axis, (a, b) in enumerate(zip(old, new)):
        if a == b:
            continue
        n = out.shape[axis]
        m = math.ceil(n * a / b - 1e-10)
        # Accumulate in float64 even for float32 membership volumes.
        v = out.movedim(axis, 0).double()
        edges = torch.arange(m + 1, dtype=torch.float64) * (b / a)
        clipped = edges.clamp(max=n)
        idx = clipped.floor().long().clamp(max=n - 1)
        integral = torch.cat([torch.zeros_like(v[:1]), v.cumsum(0)], 0)
        shape = (-1,) + (1,) * (v.ndim - 1)
        values = integral[idx] + (clipped - idx).reshape(shape) * v[idx]
        values += (edges - clipped).reshape(shape) * fill
        out = ((values[1:] - values[:-1]) * (a / b)).to(grid.dtype).movedim(0, axis)
    return out


@dataclass
class Anatomy:
    """CPU fractional volumes; missing mass is non-signalling air.

    Fractions may sum to less than one, but not exceed one. Background is an
    optional explicit air class. M0 is total water density per voxel volume;
    R1/R2/R2star and intrinsic_r2prime are rates in 1/s. Map names not
    consumed by assembly are retained by spatial utilities. Origin is a voxel
    centre, not a corner; axis directions must already be positive x/y/z.
    """

    fractions: dict[str, Tensor]
    voxel: tuple[float, float, float]
    origin: tuple[float, float, float]
    chi: Tensor | None = None
    maps: dict[str, Tensor] = field(default_factory=dict)
    chi_background: float = AIR_CHI  # same SI reference as chi
    map_field_T: float | None = None  # acquisition field of measured relaxation maps

    def __post_init__(self):
        self.voxel = _triple(self.voxel, "voxel", True)
        self.origin = _triple(self.origin, "origin")
        if not math.isfinite(self.chi_background):
            raise ValueError("chi_background must be finite SI susceptibility")
        if self.map_field_T is not None and (not math.isfinite(self.map_field_T) or self.map_field_T <= 0):
            raise ValueError("map_field_T must be positive or None")
        if not self.fractions:
            raise ValueError("anatomy requires at least one tissue")
        ref = next(iter(self.fractions.values()))
        if (
            ref.ndim != 3
            or min(ref.shape) == 0
            or ref.dtype not in (torch.float32, torch.float64)
            or ref.device.type != "cpu"
        ):
            raise ValueError("anatomy requires nonempty CPU float32/float64 (z,y,x) tensors")
        total = torch.zeros_like(ref)
        for name, value in self.fractions.items():
            self._check_map(value)
            if (value < 0).any() or (value > 1).any():
                raise ValueError(f"invalid fractions for {name}")
            total += value
        if (total > 1 + 16 * torch.finfo(ref.dtype).eps).any():
            raise ValueError("tissue fractions sum to more than one")
        for value in ([self.chi] if self.chi is not None else []) + list(self.maps.values()):
            self._check_map(value)

    def _check_map(self, value):
        ref = next(iter(self.fractions.values()))
        if (
            value.shape != ref.shape
            or value.dtype != ref.dtype
            or value.device != ref.device
            or not torch.isfinite(value).all()
        ):
            raise ValueError("all anatomy maps must have matching shape/dtype/device and finite values")

    @property
    def shape(self):
        return tuple(next(iter(self.fractions.values())).shape)

    def crop(self, region):
        """Crop by three unit-step slices in z/y/x order, updating physical origin.

        Cropping discards susceptibility sources. Prefer to_phantom(fov_region=)
        to restrict spins while evaluating B0 from the complete source anatomy.
        """
        if len(region) != 3 or any(not isinstance(s, slice) for s in region):
            raise ValueError("region needs three slices in z/y/x order")
        bounds = [s.indices(n) for s, n in zip(region, self.shape)]
        if any(step != 1 or hi <= lo for lo, hi, step in bounds):
            raise ValueError("crop must be nonempty with unit steps")
        region = tuple(slice(lo, hi) for lo, hi, _ in bounds)
        origin = tuple(o + lo * h for o, (lo, _, _), h in zip(self.origin, reversed(bounds), reversed(self.voxel)))
        take = lambda v: v[region]
        return Anatomy(
            {k: take(v) for k, v in self.fractions.items()},
            self.voxel,
            origin,
            None if self.chi is None else take(self.chi),
            {k: take(v) for k, v in self.maps.items()},
            self.chi_background,
            self.map_field_T,
        )

    def slab(self, z0, thickness):
        """Physical z interval [z0,z0+thickness], snapped outward to cell boundaries."""
        if not math.isfinite(z0) or not math.isfinite(thickness) or thickness <= 0:
            raise ValueError("slab needs finite z0 and positive thickness in m")
        edge = self.origin[2] - self.voxel[0] / 2
        lo = max(0, math.floor((z0 - edge) / self.voxel[0] + 1e-10))
        hi = min(self.shape[0], math.ceil((z0 + thickness - edge) / self.voxel[0] - 1e-10))
        if hi <= lo:
            raise ValueError("slab does not intersect anatomy")
        return self.crop((slice(lo, hi), slice(None), slice(None)))

    def resample(self, voxel):
        """Volume-weighted cell averages; preserves sum(fraction)*voxel_volume.

        Lower outer boundary is fixed; upper edge expands by <1 new voxel if
        needed. Exterior has fractions/maps zero and chi=chi_background. This is not
        nearest-neighbour labelling or recovery of unknown sub-cell anatomy.
        With M0 present, relaxation rates are M0-weighted averages, so empty air
        does not dilute rates. Coarsening rates remains a compartment approximation:
        an average rate cannot preserve a general multi-exponential time course.
        """
        voxel = _triple(voxel, "voxel", True)
        origin = tuple(o + (b - a) / 2 for o, a, b in zip(self.origin, reversed(self.voxel), reversed(voxel)))
        maps = {k: _resample(v, self.voxel, voxel) for k, v in self.maps.items()}
        if "M0" in maps:
            density = maps["M0"]
            for name in ("R1", "R2", "R2star", "intrinsic_r2prime"):
                if name in maps:
                    weighted = _resample(self.maps[name] * self.maps["M0"], self.voxel, voxel)
                    maps[name] = torch.where(
                        density > 0, weighted / density.clamp_min(torch.finfo(density.dtype).tiny), 0.0
                    )
        return Anatomy(
            _renormalize({k: _resample(v, self.voxel, voxel).clamp(0, 1) for k, v in self.fractions.items()}),
            voxel,
            origin,
            None if self.chi is None else _resample(self.chi, self.voxel, voxel, self.chi_background),
            maps,
            self.chi_background,
            self.map_field_T,
        )
