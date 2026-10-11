"""Fuzzy anatomy on a physical grid, conservative resampling, and sub-voxel spins.

Arrays/voxel sizes: (z,y,x); positions/origin: (x,y,z) metres. Assembly runs on
CPU; transfer the resulting Phantom to an engine afterwards. Fractions describe
piecewise-constant *densities*, not the location of a boundary inside a cell.
Use a finer anatomy grid to resolve a sharp boundary independently of reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import warnings

import numpy as np
import torch
from torch import Tensor

from .constants import GAMMA_BAR
from .exchange import BoundPool
from .maps import FieldMaps, sample
from .phantom import FAT_PEAKS, Phantom, _subvoxel_counts
from .susceptibility import shim, susceptibility_to_b0

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
        if not math.isfinite(self.mean) or (self.sd is not None and
                (not math.isfinite(self.sd) or self.sd < 0)) or not self.citation:
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


# Every fallback explicitly says it is an assumption. These are simulation priors,
# not a normative human atlas. No values in phantom.TISSUES are changed.
_BW = "BrainWeb MRISIM tissue_mr_parameters.txt (1996), named tissue row; https://brainweb.bic.mni.mcgill.ca/brainweb/tissue_mr_parameters.txt"
_ST = "Stanisz et al. 2005, Table 1, doi:10.1002/mrm.20605; 37 C excised animal tissue (human blood)"
_CA = "Caan et al. 2019, Table 1 MP2RAGEME column, doi:10.1002/hbm.24490; human 7 T"
_RO = ("Rooney et al. 2007, doi:10.1002/mrm.21122: CSF T1 ~4.4 s, nearly field-independent (value as quoted by "
       "Spijkerman et al. 2018, ref. 13)")
_SP = ("Spijkerman et al. 2018, doi:10.1007/s10334-017-0659-3: 3 T ventricular CSF T2 median 2.03 s "
       "(range 1.73-2.16 s; peripheral 1.79 s), T2-prepared, 7 volunteers")
_SH = ("Shin et al. 2025, Table 3 average across 4 subjects (mean +- between-subject SD), doi:10.1002/mrm.30624: "
       "human 3 T (Prisma), FreeSurfer GM/WM, 1 x 1 x 2 mm, TE 3-66.18 ms (43 echoes), mono-exponential, no "
       "macroscopic-field correction; the abstract swaps the WM/GM labels")
_DB = "de Bazelaire et al. 2004, doi:10.1148/radiol.2303021331: subcutaneous fat at 3 T, abdomen"
_IT = ("IT'IS Foundation tissue database, relaxation times, SAT (subcutaneous fat) 1.5 T: mean of 2 studies "
       "(T1 288-343 ms, T2 58-165 ms); itis.swiss, checked 2026-10-07")


def _assumed(value, reason):
    return Distribution(value, None, "Assumption: " + reason, True)


def _tables():
    # BrainWeb's legacy simulator values are useful for reproducibility only.
    legacy = dict(csf=(1., 2.569, .329, .058), gm=(.86, .833, .083, .069),
                  wm=(.77, .500, .070, .061), fat=(1., .350, .070, .058),
                  muscle=(1., .900, .047, .030))
    legacy |= {"skin": legacy["muscle"], "connective": legacy["muscle"],
               "dura": legacy["muscle"], "marrow": legacy["fat"],
               "vessels": legacy["gm"], "skull": (0., 1., .001, .001),
               "background": (0., 1., 1., 1.)}
    bw = {}
    for name, values in legacy.items():
        assumed = name not in ("csf", "gm", "wm", "fat", "muscle")
        params = [Distribution(v, None, _BW + f"; {name}" +
                   ("; proxy/placeholder for new tissue, not a measured value" if assumed else ""), assumed)
                  for v in values]
        chi = AIR_CHI if name == "background" else WATER_CHI
        if name == "skull":
            chi -= 2e-6  # M1's cited scale; explicit approximate contrast, not UTE calibration.
        bw[name] = TissueProperties(params[0], params[1], params[2],
            _assumed(chi, "water-like soft tissue; bone -2 ppm relative to water; air +0.36 ppm; see susceptibility.py"),
            params[3], fat_fraction=_assumed(float(name in ("fat", "marrow")), "pure lipid class or no lipid"))
    # A published simulator constant is not a measured human property. Flag
    # every legacy fallback, especially when carrying 1.5 T priors to 3 T.
    default = {}
    for strength in (1.5, 3.):
        default[strength] = {}
        for name, p in bw.items():
            fallback = {key: _assumed(getattr(p, key).mean,
                        getattr(p, key).citation + f"; legacy 1.5 T simulator prior used at {strength} T")
                        for key in ("pd", "t1", "t2", "t2star")}
            default[strength][name] = replace(p, **fallback)
    # 1.5 T / 3 T means and specimen SDs, seconds. In-vitro proxies are flagged.
    rows = {"wm": ((.884,.050,.072,.004), (1.084,.045,.069,.003)),
            "gm": ((1.124,.050,.095,.008), (1.820,.114,.099,.007)),
            "muscle": ((1.008,.020,.044,.006), (1.412,.013,.050,.004)),
            "vessels": ((1.441,.120,.290,.030), (1.932,.085,.275,.050))}
    for strength, index in ((1.5, 0), (3., 1)):
        for name, measurements in rows.items():
            t1, s1, t2, s2 = measurements[index]
            # T2* at 1.5 T is a legacy model prior; at 3 T WM/GM are whole-brain measurements
            # (they replace occipital Wansapura 1999 values, GM 41.6 < WM 48.4 ms, which made cortex
            # dark at long TE; Peters et al., ISMRM 2006 #926, also find GM 59.7 > WM 54.6 ms).
            star = min(bw[name].t2star.mean, t2)
            star_dist = _assumed(star, _BW + "; T2* proxy, clipped to T2; no measured SD")
            if strength == 3 and name in ("wm", "gm"):
                star_dist = Distribution(.0515, .0022, _SH) if name == "wm" else Distribution(.0559, .0018, _SH)
            default[strength][name] = replace(default[strength][name],
                t1=Distribution(t1,s1,_ST + f"; {name}, {strength} T",True),
                t2=Distribution(t2,s2,_ST + f"; {name}, {strength} T",True), t2star=star_dist)
        for name in ("skin", "connective", "dura"):
            proxy = default[strength]["muscle"]
            default[strength][name] = replace(proxy,
                t1=_assumed(proxy.t1.mean, _ST + f"; muscle used for {name}"),
                t2=_assumed(proxy.t2.mean, _ST + f"; muscle used for {name}"))
        # CSF and subcutaneous fat: measured values replace the legacy 1.5 T simulator priors,
        # whose CSF (T1 2.57 s, T2 0.33 s) was the largest error in T2-weighted contrast.
        csf, fat = default[strength]["csf"], default[strength]["fat"]
        csf_t2 = Distribution(2.03, None, _SP, strength != 3.)
        default[strength]["csf"] = replace(csf, t1=Distribution(4.4, None, _RO),
            t2=csf_t2 if strength == 3. else replace(csf_t2, citation="Assumption: " + _SP + "; 3 T value used at 1.5 T"),
            t2star=_assumed(2.03, "CSF T2* = T2: no intrinsic broadening; pulsation and macroscopic fields not included"))
        if strength == 3.:
            default[strength]["fat"] = replace(fat, t1=Distribution(.382, .013, _DB), t2=Distribution(.068, .004, _DB))
        else:
            default[strength]["fat"] = replace(fat, t1=Distribution(.3124, .029, _IT), t2=Distribution(.1174, .0564, _IT),
                t2star=_assumed(.058, _BW + "; fat T2* legacy prior"))
        for name, values in _mt_and_diffusion(strength).items():
            default[strength][name] = replace(default[strength][name], **values)
    # Only verified 7 T brain entries: the remaining classes require user-supplied
    # values. T2 is an explicit 3 T proxy; the T1/T2* means AND SDs are measured.
    default[7.] = {}
    for name, t1, s1, star, ss in (("wm",1.19,.02,.0267,.0001), ("gm",2.,.08,.0324,.0007)):
        p = replace(default[3.][name], **_mt_and_diffusion(7.)[name])
        default[7.][name] = replace(p, t1=Distribution(t1,s1,_CA), t2star=Distribution(star,ss,_CA),
            t2=_assumed(p.t2.mean, _ST + "; 3 T T2 carried to 7 T pending calibrated value"))
    return default, {1.5: bw}


# Two-pool MT and isotropic diffusion (ROADMAP M5). MT in BoundPool's convention: bound_fraction
# p = M0b / (M0b + M0f), exchange_rate = bound -> free [1/s] (exchange.py), super-Lorentzian T2b.
_YA16 = ("Yarnykh 2016, doi:10.1002/mrm.25811: 3 T single-point MPF = M0b/(M0b+M0f), whole-brain WM and GM of "
         "8 healthy subjects, mean +- between-subject SD")
_YA12 = ("Yarnykh 2012, doi:10.1002/mrm.23224, Eq. 2 and 7: 3 T multi-point pulsed qMT, whole-brain histogram "
         "medians of 2 controls and 2 MS patients; R = k(1-f)/f is the bound-to-free rate; super-Lorentzian T2B")
_SI = ("Sinclair et al. 2010, doi:10.1002/mrm.22562: in vivo qMT of human calf muscle at 3 T, soleus, 10 "
       "subjects, mean +- SD; f of the total pool, super-Lorentzian T2b")
_HE = ("Helenius et al. 2002, AJNR 23:194 (www.ajnr.org/content/23/2/194): ADC at b = 0/1000 s/mm^2, 80 adults "
       "at 1.5 T, mean +- SD; diffusivity does not depend on B0")
_HO = ("Holz, Heil and Sacco 2000, doi:10.1039/b005319h: water self-diffusion, Speedy-Angell fit "
       "D0 (T/215.05 K - 1)^2.063, D0 = 1.635e-8 m^2/s, at 37 C (2.299e-9 at 25 C); in vivo CSF adds pulsation")
_LI = ("Li et al. 2014, doi:10.1002/nbm.3159, Table 2: DTI mean diffusivity of 8 thigh muscles at 3 T; mean and "
       "SD of the 8 muscle means")
_FU = ("Funck et al. 2018, doi:10.1002/mrm.26919: whole blood at 37 C, flow-compensated, suggested D_b = 1.54 +- "
       "0.12 um^2/ms (falls with haematocrit)")
_WE = ("Weidlich et al., ISMRM 2018 #2511: gluteal subcutaneous fat at 3 T, lipid ADC 1.83e-5 mm^2/s at 2 mm "
       "slices, rising with slice thickness (motion): an upper bound")
_CH = ("Chen et al. 2010, doi:10.1002/mrm.22208: water exchanges magnetization with protein and membrane "
       "protons but not with triglyceride (fat) protons")
_SM = "Smith et al., ISMRM 2005 #417: CSF MTR 0.01 +- 0.02 at 3 T (frontal WM 0.48 +- 0.01)"


def _mt_and_diffusion(strength):
    """MT and diffusion fields per tissue class; MT measured at 3 T is flagged at other fields
    (Stanisz et al. 2005 find qMT parameters at 1.5 T and 3 T similar). Diffusivity does not
    depend on B0. Classes without a semi-solid pool get bound_fraction 0 only."""
    def mt(value, sd, citation, assumed=False):
        if strength != 3.:
            citation, assumed = "Assumption: " + citation + f"; 3 T value used at {strength} T", True
        return Distribution(value, sd, citation, assumed)
    rate = mt(19., None, _YA12 + ": R = 19.0 /s")
    t2b = mt(9.7e-6, None, _YA12 + ": T2B = 9.7 us")
    out = {"wm": dict(bound_fraction=mt(.1277, .0041, _YA16 + ": WM 12.77 +- 0.41 %"), exchange_rate=rate,
                      bound_t2=t2b, diffusivity=Distribution(.70e-9, .03e-9, _HE + ": white matter")),
           "gm": dict(bound_fraction=mt(.0673, .0015, _YA16 + ": GM 6.73 +- 0.15 %"), exchange_rate=rate,
                      bound_t2=t2b, diffusivity=Distribution(.89e-9, .04e-9, _HE + ": cortical grey matter")),
           "csf": dict(bound_fraction=mt(0., None, _SM + ": no semi-solid pool"),
                       diffusivity=Distribution(3.04e-9, None, _HO)),
           "muscle": dict(bound_fraction=mt(.084, .006, _SI), bound_t2=mt(5.9e-6, .2e-6, _SI),
                          exchange_rate=_assumed(19., "no converged value in human muscle (Sinclair et al. 2010, "
                                                 "doi:10.1002/mrm.22562); the brain's R from " + _YA12),
                          diffusivity=Distribution(1.78e-9, .08e-9, _LI)),
           "fat": dict(bound_fraction=Distribution(0., None, _CH), diffusivity=Distribution(1.83e-11, None, _WE)),
           "vessels": dict(bound_fraction=_assumed(0., "blood's MT is small next to tissue's, the premise of "
                           "MT-suppressed TOF angiography (Pike et al. 1992, doi:10.1002/mrm.1910250217)"),
                           diffusivity=Distribution(1.54e-9, .12e-9, _FU))}
    for name in ("skin", "connective", "dura"):
        out[name] = {key: _assumed(d.mean, d.citation.removeprefix("Assumption: ") + f"; muscle used for {name}")
                     for key, d in out["muscle"].items()}
    out["marrow"] = {key: _assumed(d.mean, d.citation + "; fat used for marrow, a lipid class here")
                     for key, d in out["fat"].items()}
    return out


DEFAULT_TISSUES, BRAINWEB_TISSUES = _tables()


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


def _resample(grid, old, new, fill=0.):
    """Exact separable cell-overlap integrals, no dense interpolation matrix."""
    out = grid
    for axis, (a, b) in enumerate(zip(old, new)):
        if a == b:
            continue
        n = out.shape[axis]
        m = math.ceil(n*a/b - 1e-10)
        # Accumulate in float64 even for float32 membership volumes.
        v = out.movedim(axis, 0).double()
        edges = torch.arange(m+1, dtype=torch.float64) * (b/a)
        clipped = edges.clamp(max=n)
        idx = clipped.floor().long().clamp(max=n-1)
        integral = torch.cat([torch.zeros_like(v[:1]), v.cumsum(0)], 0)
        shape = (-1,) + (1,)*(v.ndim-1)
        values = integral[idx] + (clipped-idx).reshape(shape)*v[idx]
        values += (edges-clipped).reshape(shape)*fill
        out = ((values[1:]-values[:-1])*(a/b)).to(grid.dtype).movedim(0, axis)
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
        if ref.ndim != 3 or min(ref.shape) == 0 or ref.dtype not in (torch.float32,torch.float64) or ref.device.type != "cpu":
            raise ValueError("anatomy requires nonempty CPU float32/float64 (z,y,x) tensors")
        total = torch.zeros_like(ref)
        for name, value in self.fractions.items():
            self._check_map(value)
            if (value < 0).any() or (value > 1).any():
                raise ValueError(f"invalid fractions for {name}")
            total += value
        if (total > 1 + 16*torch.finfo(ref.dtype).eps).any():
            raise ValueError("tissue fractions sum to more than one")
        for value in ([self.chi] if self.chi is not None else []) + list(self.maps.values()):
            self._check_map(value)

    def _check_map(self, value):
        ref = next(iter(self.fractions.values()))
        if value.shape != ref.shape or value.dtype != ref.dtype or value.device != ref.device or not torch.isfinite(value).all():
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
        bounds = [s.indices(n) for s,n in zip(region,self.shape)]
        if any(step != 1 or hi <= lo for lo,hi,step in bounds):
            raise ValueError("crop must be nonempty with unit steps")
        region = tuple(slice(lo,hi) for lo,hi,_ in bounds)
        origin = tuple(o + lo*h for o,(lo,_,_),h in zip(self.origin,reversed(bounds),reversed(self.voxel)))
        take = lambda v: v[region]
        return Anatomy({k:take(v) for k,v in self.fractions.items()}, self.voxel, origin,
                       None if self.chi is None else take(self.chi), {k:take(v) for k,v in self.maps.items()},
                       self.chi_background, self.map_field_T)

    def slab(self, z0, thickness):
        """Physical z interval [z0,z0+thickness], snapped outward to cell boundaries."""
        if not math.isfinite(z0) or not math.isfinite(thickness) or thickness <= 0:
            raise ValueError("slab needs finite z0 and positive thickness in m")
        edge = self.origin[2] - self.voxel[0]/2
        lo = max(0, math.floor((z0-edge)/self.voxel[0] + 1e-10))
        hi = min(self.shape[0], math.ceil((z0+thickness-edge)/self.voxel[0] - 1e-10))
        if hi <= lo:
            raise ValueError("slab does not intersect anatomy")
        return self.crop((slice(lo,hi),slice(None),slice(None)))

    def resample(self, voxel):
        """Volume-weighted cell averages; preserves sum(fraction)*voxel_volume.

        Lower outer boundary is fixed; upper edge expands by <1 new voxel if
        needed. Exterior has fractions/maps zero and chi=chi_background. This is not
        nearest-neighbour labelling or recovery of unknown sub-cell anatomy.
        With M0 present, relaxation rates are M0-weighted averages, so empty air
        does not dilute rates. Coarsening rates remains a compartment approximation:
        an average rate cannot preserve a general multi-exponential time course.
        """
        voxel = _triple(voxel,"voxel",True)
        origin = tuple(o+(b-a)/2 for o,a,b in zip(self.origin,reversed(self.voxel),reversed(voxel)))
        maps = {k:_resample(v,self.voxel,voxel) for k,v in self.maps.items()}
        if "M0" in maps:
            density = maps["M0"]
            for name in ("R1", "R2", "R2star", "intrinsic_r2prime"):
                if name in maps:
                    weighted = _resample(self.maps[name]*self.maps["M0"],self.voxel,voxel)
                    maps[name] = torch.where(density > 0,weighted/density.clamp_min(torch.finfo(density.dtype).tiny),0.)
        return Anatomy(_renormalize({k:_resample(v,self.voxel,voxel).clamp(0,1) for k,v in self.fractions.items()}),
            voxel,origin,None if self.chi is None else _resample(self.chi,self.voxel,voxel,self.chi_background),
            maps,
            self.chi_background,self.map_field_T)


def load_brainweb(subject: int, root, *, voxel=None, dtype=torch.float32) -> Anatomy:
    """Load the 20-subject fuzzy models from checksum-pinned raw-short gzip files.

    Run validation/anatomy/fetch_brainweb.py, or obtain the public raw-short files
    manually and provide a subjectNN.json manifest (see validation/anatomy/README.md).
    Subject IDs are 4,5,6,18,20,38,41..54, not consecutive 1..20. Native arrays
    are (362,434,362) at 0.5 mm, z/y/x, positive MINC axes; origin is read from
    the captured header in the manifest. The similarly named simulated T1 image
    has a different 1-mm grid. No flipping/transposing of these raw arrays is needed.

    Raw shorts are 0..4095 memberships (captured little-endian, despite the
    legacy format page claiming big-endian). Quantization can perturb their sum, so all
    12 classes (including background) are jointly normalized where their sum
    exceeds one, before resampling. Deficits remain air, not amplified tissue.
    ``voxel`` downsamples each class as it loads, limiting memory. Returned
    memberships are normalized at native resolution first, preserving volumes.
    Loading never downloads data or silently repairs a checksum mismatch.
    """
    if subject not in (4,5,6,18,20,38,*range(41,55)):
        raise ValueError("invalid BrainWeb 20-model subject ID")
    if dtype not in (torch.float32,torch.float64):
        raise ValueError("dtype must be float32 or float64")
    root = Path(root)
    manifest = json.loads((root/f"subject{subject:02d}.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("subject") != subject:
        raise ValueError("invalid BrainWeb manifest")
    names = ("background","csf","gm","wm","fat","muscle","skin","skull","vessels","connective","dura","marrow")
    shape = tuple(manifest["shape"])
    old = tuple(manifest["voxel"])
    new = old if voxel is None else _triple(voxel,"voxel",True)
    # uint16 maps + one uint32 total, instead of twelve native float64 volumes.
    raw_maps = {}
    storage = np.dtype(manifest["dtype"])
    if storage.str not in ("<u2", ">u2", "|u1") or manifest["scale"] <= 0:
        raise ValueError("unsupported raw encoding")
    total = np.zeros(shape,dtype=np.uint32)
    for name in names:
        entry = manifest["files"][name]
        path = root/entry["path"]
        if path.resolve().parent != root.resolve():
            raise ValueError("manifest paths must name files directly inside root")
        raw = gzip.open(path,"rb").read() if path.suffix == ".gz" else path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"checksum mismatch: {path}")
        if len(raw) != math.prod(shape)*storage.itemsize:
            raise ValueError(f"wrong byte count: {path}")
        array = np.frombuffer(raw,dtype=storage).reshape(shape)
        if array.max() > manifest["scale"]:
            raise ValueError("raw values exceed scale; check byte order")
        raw_maps[name] = array
        total += array
    total = np.maximum(total,manifest["scale"])
    fractions = {}
    for name in names:
        # Divide in float64, then convert only after conservative resampling.
        value = torch.from_numpy(raw_maps.pop(name).astype(np.float64) / total)
        fractions[name] = _resample(value,old,new).to(dtype).clamp(0,1)
    origin = tuple(o+(b-a)/2 for o,a,b in zip(manifest["origin"],reversed(old),reversed(new)))
    return Anatomy(_renormalize(fractions),new,origin)


# Label IDs published with the QSM Challenge 2.0 head model; independently
# listed in qsm-forward/label.json, commit 92149383b0f43f47d9a351ceb4f525c0c4ee8299.
QSM_LABELS = {1: "caudate", 2: "globus_pallidus", 3: "putamen", 4: "red_nucleus",
              5: "dentate_nucleus", 6: "substantia_nigra", 7: "thalamus", 8: "wm",
              9: "gm", 10: "csf", 11: "vessels", 12: "fat", 13: "skull",
              14: "background", 15: "muscle", 16: "calcification"}


def _file_sha256(path):
    """Hash decompressed bytes so a gzip header change does not change identity."""
    digest = hashlib.sha256()
    with (gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")) as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_qsm2(root, resolution="1mm", *, dtype=torch.float32) -> Anatomy:
    """Load locally supplied QSM Challenge 2.0 maps using root/qsm2.json pins.

    Optional dependency: pip install 'blochparty[anatomy]'. The full-head source
    directory (chimodel, maps, masks) must be obtained under its access terms;
    validation/anatomy/prepare_qsm2.py records a reviewed local manifest. This
    function never downloads or authenticates. Both chi models can be selected
    by preparing distinct manifests/roots; no filename guessing at load time.

    NIfTI arrays are reoriented to positive RAS axes without interpolation, then
    transposed to z/y/x. Axis-aligned permutations/flips are supported; oblique
    or sheared grids raise rather than silently drop their affine. All maps must
    share a grid after reorientation. Spatial units are declared in the manifest
    (and checked against the header if set). Chi is converted from ppm to SI,
    retaining its original reference and explicit surrounding chi_background.

    ``resolution`` is 'native', '0.64mm', '1mm', or a (dz,dy,dx) tuple in m;
    non-native grids use conservative cell averages, not k-space truncation.
    A manifest with kind='challenge' loads the public challenge ground truth
    (chi and segmentation only): background susceptibility is zero, no measured
    relaxation/density maps are invented, and its brain mask is derived from
    labels 1..11 and 16. It is distinct from the full-head source package.

    Sixteen tissue memberships are retained; original background label 0 joins
    air label 14. M0, R1, R2star and brain_mask are retained as maps. R1/R2star
    belong to the manifest's acquisition field (normally 7 T), not automatically
    to the requested simulation field. There is no measured R2 map in this
    release: full Bloch assembly needs an explicit tissue table for T2, including
    the deep-grey labels. Observed R2star must still follow to_phantom's
    macroscopic/intrinsic separation policy; it is never silently used as R2.
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("load_qsm2 requires the anatomy extra: pip install 'blochparty[anatomy]'") from exc
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be float32 or float64")
    root = Path(root).resolve()
    manifest = json.loads((root/"qsm2.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("chi_unit") != "ppm":
        raise ValueError("unsupported QSM manifest schema or chi units")
    kind = manifest.get("kind", "full-head")
    if kind not in ("full-head", "challenge"):
        raise ValueError("unknown QSM package kind")
    unit = manifest.get("spatial_unit")
    units = {"mm": .001, "meter": 1., "micron": 1e-6}
    if unit not in units:
        raise ValueError("manifest needs an explicit NIfTI spatial_unit")
    scale = units[unit]
    background = float(manifest["chi_background_ppm"])*1e-6
    map_field = float(manifest["map_field_T"]) if kind == "full-head" else None
    arrays = {}
    affine = shape = None
    keys = ("chi", "M0", "R1", "R2star", "segmentation", "brain_mask") if kind == "full-head" else ("chi", "segmentation")
    for key in keys:
        entry = manifest["files"][key]
        path = (root/entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("QSM manifest paths must stay inside root")
        if _file_sha256(path) != entry["sha256"]:
            raise ValueError(f"checksum mismatch: {path}")
        img = nib.load(path)
        header_unit = img.header.get_xyzt_units()[0]
        if header_unit not in ("unknown", unit):
            raise ValueError(f"NIfTI/manifest spatial unit mismatch: {key}")
        img = nib.as_closest_canonical(img)
        mat = img.affine.copy()
        if len(img.shape) != 3 or min(img.shape) < 1 or not np.isfinite(mat).all():
            raise ValueError("QSM maps must have a finite affine and nonempty 3D grid")
        linear = mat[:3, :3]
        if (np.diag(linear) <= 0).any() or not np.allclose(linear, np.diag(np.diag(linear)), rtol=0, atol=1e-6):
            raise ValueError("oblique/sheared QSM grids need explicit common-grid resampling first")
        if affine is None:
            affine, shape = mat, img.shape
        elif img.shape != shape or not np.allclose(mat, affine, rtol=0, atol=1e-5):
            raise ValueError(f"QSM maps do not share a physical grid: {key}")
        data = img.get_fdata(dtype=np.float64 if dtype == torch.float64 else np.float32)
        arrays[key] = torch.from_numpy(np.ascontiguousarray(data.transpose(2,1,0)))
        if not torch.isfinite(arrays[key]).all():
            raise ValueError(f"non-finite QSM map: {key}")
    labels = arrays.pop("segmentation")
    if (labels != labels.round()).any() or (labels < 0).any() or (labels > 16).any():
        raise ValueError("unknown QSM segmentation labels; expected integers 0..16")
    if kind == "full-head":
        if (arrays["M0"] < 0).any() or (arrays["brain_mask"] < 0).any() or (arrays["brain_mask"] > 1).any():
            raise ValueError("QSM M0 must be nonnegative and brain mask in [0,1]")
        for key in ("R1", "R2star"):
            if (arrays[key] < 0).any() or (arrays[key][arrays["M0"] > 0] <= 0).any():
                raise ValueError(f"QSM {key} must be positive where M0 is nonzero")
    else:
        if background != 0:
            raise ValueError("challenge brain-only chi requires zero exterior susceptibility")
        arrays["brain_mask"] = (((labels >= 1) & (labels <= 11)) | (labels == 16)).to(dtype)
    fractions = {name: (labels == label).to(dtype) for label,name in QSM_LABELS.items()
                 if (labels == label).any()}
    fractions["background"] = ((labels == 0) | (labels == 14)).to(dtype)
    chi = arrays.pop("chi")*1e-6
    voxel = tuple((np.diag(affine)[:3]*scale)[::-1])
    origin = tuple(affine[:3,3]*scale)
    anatomy = Anatomy(fractions,voxel,origin,chi,arrays,background,map_field)
    if isinstance(resolution, str):
        options = {"native": None, "0.64mm": (.00064,)*3, "1mm": (.001,)*3}
        if resolution not in options:
            raise ValueError("resolution must be native, 0.64mm, 1mm, or a voxel tuple")
        resolution = options[resolution]
    return anatomy if resolution is None else anatomy.resample(resolution)


# Units of the AHEAD v3 templates: the publisher record does not state them. ppm and 1/s
# follow from the values (WM R1 ~0.8 /s and R2* ~ 35 /s at 7 T match Caan et al. 2019, same
# MP2RAGEME method; pallidal QSM ~ +0.1 is ppm, not ppb); checked in validation/anatomy_ahead.py.
AHEAD_UNITS = {"chi": 1e-6, "R1": 1.0, "R2star": 1.0, "relative_PD": 1.0}


def load_ahead(root, resolution="1mm", *, dtype=torch.float32) -> Anatomy:
    """Load the AHEAD v3 7 T population-median brain templates (Bazin et al.; figshare
    doi:10.21942/uva.19646364.v3, CC BY 4.0) fetched by validation/anatomy/fetch_ahead.py into
    ``root`` (pins: root/verified_source.json, decompressed SHA-256 checked here).

    A brain-only template: one tissue class ``"brain"`` (where R1 > 0 and PD > 0; QSM can be
    zero inside tissue, so it does not define the mask), maps ``M0`` (relative PD, not absolute
    density), ``R1``, ``R2star`` [1/s] and ``brain_mask``, and ``chi`` (QSM, converted ppm -> SI,
    relative to its own reference, so ``chi_background = 0``: no air or bone around it).
    ``map_field_T = 7``: simulating at another field raises (no field scaling). There is no
    R2 map: ``to_phantom`` needs a table entry ``"brain"`` at 7 T giving T2 (and the T2* policy
    as for other measured R2star maps). Stored RPS; reoriented to RAS without interpolation and
    transposed to (z, y, x). ``resolution``: 'native' (0.5 mm), '1mm' (default, conservative
    cell averages) or a (dz, dy, dx) tuple [m]. Optional dependency: nibabel (``[anatomy]``).
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("load_ahead requires the anatomy extra: pip install 'blochparty[anatomy]'") from exc
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be float32 or float64")
    root = Path(root).resolve()
    manifest = json.loads((root/"verified_source.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("doi") != "10.21942/uva.19646364.v3":
        raise ValueError("unexpected AHEAD manifest")
    arrays, affine = {}, None
    for key in ("chi", "R1", "R2star", "relative_PD"):
        entry = manifest["files"][key]
        path = (root/entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("AHEAD manifest paths must stay inside root")
        if _file_sha256(path) != entry["decompressed_sha256"]:
            raise ValueError(f"checksum mismatch: {path}")
        img = nib.as_closest_canonical(nib.load(path))
        mat = img.affine
        if img.header.get_xyzt_units()[0] != "mm" or not np.allclose(mat[:3, :3], np.diag(np.diag(mat[:3, :3])), atol=1e-6) \
                or (np.diag(mat)[:3] <= 0).any():
            raise ValueError(f"unexpected AHEAD grid: {key}")
        if affine is None:
            affine = mat
        elif not np.allclose(mat, affine, atol=1e-6):
            raise ValueError(f"AHEAD maps do not share a grid: {key}")
        data = img.get_fdata(dtype=np.float64 if dtype == torch.float64 else np.float32)
        arrays[key] = torch.from_numpy(np.ascontiguousarray(data.transpose(2, 1, 0))) * AHEAD_UNITS[key]
    pd, r1, r2s = arrays["relative_PD"], arrays["R1"], arrays["R2star"]
    mask = ((r1 > 0) & (pd > 0)).to(dtype)
    if (pd < 0).any() or (r1 < 0).any() or (r2s < 0).any():
        raise ValueError("AHEAD PD and rates must be nonnegative")
    voxel = tuple((np.diag(affine)[:3]*1e-3)[::-1])
    origin = tuple(affine[:3, 3]*1e-3)
    maps = {"M0": pd*mask, "R1": torch.where(mask > 0, r1, 0.), "R2star": torch.where(mask > 0, r2s, 0.), "brain_mask": mask}
    anatomy = Anatomy({"brain": mask}, voxel, origin, arrays["chi"]*mask, maps, 0.0, 7.0)
    if isinstance(resolution, str):
        options = {"native": None, "1mm": (.001,)*3}
        if resolution not in options:
            raise ValueError("resolution must be native, 1mm, or a voxel tuple")
        resolution = options[resolution]
    return anatomy if resolution is None else anatomy.resample(resolution)


def _properties(anatomy, table, field_T):
    if field_T not in table:
        raise ValueError(f"no tissue table for {field_T} T; provide explicit values (no field interpolation)")
    props = table[field_T]
    missing = [k for k,v in anatomy.fractions.items() if k not in props and v.any()]
    if missing:
        raise ValueError(f"missing tissue properties at {field_T} T: {missing}")
    return props


def anatomy_field_maps(anatomy, table=DEFAULT_TISSUES, field_T=3., *, shim_order=2,
                       brain_mask=None, b0_dir=(0,0,1), padding=2.):
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
        props = _properties(anatomy,table,field_T)
        chi = torch.full_like(ref,AIR_CHI)
        for name,frac in anatomy.fractions.items():
            if name in props:
                chi = chi + frac*(props[name].chi.mean-AIR_CHI)
    background = anatomy.chi_background if anatomy.chi is not None else AIR_CHI
    df = susceptibility_to_b0(chi,anatomy.voxel,field_T,b0_dir=b0_dir,padding=padding,background=background)
    correction = torch.zeros_like(df)
    if shim_order is not None:
        if brain_mask is None:
            brain_mask = (anatomy.maps["brain_mask"] > .5 if "brain_mask" in anatomy.maps else
                sum((anatomy.fractions[k] for k in ("wm","gm","csf") if k in anatomy.fractions),
                    torch.zeros_like(ref)) > .5)
        residual,_ = shim(df,brain_mask,shim_order,voxel=anatomy.voxel,origin=anatomy.origin)
        correction = residual-df
    args = (anatomy.origin,tuple(reversed(anatomy.voxel)))
    return FieldMaps(*args,b0=df),FieldMaps(*args,b0=correction)


# The R2* measurement a T2* value or map came from (voxel (dz, dy, dx) [m], echo times [s]):
# the macroscopic part of its decay is estimated at these, not at the simulation grid.
# Default: the protocol of the 3 T WM/GM table values (Shin et al. 2025: 1 x 1 mm in plane, 2 mm
# slices, assumed axial; 43 echoes TE 3 + 1.47 k ms to 66.18 ms). Other table T2* entries are
# priors without a protocol of their own and use the same.
DEFAULT_R2STAR_ACQUISITION = {"voxel": (2e-3, 1e-3, 1e-3), "te": tuple(3e-3 + 1.47e-3 * k for k in range(43))}
# AHEAD (MP2RAGEME, Caan et al. 2019: second-inversion echoes 3 / 11.5 / 19 / 27.5 ms). Voxel 0.64 mm:
# the commonly cited AHEAD resolution; the papers say "submillimetre" (unverified here).
AHEAD_R2STAR_ACQUISITION = {"voxel": (.64e-3, .64e-3, .64e-3), "te": (3e-3, 11.5e-3, 19e-3, 27.5e-3)}


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
    grads = [torch.gradient(df, spacing=float(h), dim=ax)[0] if df.shape[ax] > 1 else torch.zeros_like(df)
             for ax, h in enumerate(voxel)]  # a single-voxel axis has no resolved gradient
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
        raise ValueError(f"variability takes {_VARIABLE} relative SDs, 'length' [m] and optional 'seed'; got {sorted(variability)}")
    seed = int(variability.get("seed", 0))
    ref = next(iter(anatomy.fractions.values()))
    bounds = [s.indices(n)[:2] for s, n in zip(region, anatomy.shape)] if region is not None else \
        [(0, n) for n in anatomy.shape]
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
            g = sample(grid[None], torch.tensor(origin, dtype=torch.float64), torch.tensor(spacing, dtype=torch.float64),
                       centres)[:, 0].reshape(nz, ny, nx)
            sigma2 = math.log1p(sd**2)  # log-normal with mean 1 and SD sd
            out[name, prop] = torch.exp(math.sqrt(sigma2) * g - sigma2 / 2).to(ref.dtype)
    return out


def to_phantom(anatomy: Anatomy, table=DEFAULT_TISSUES, field_T=3., *, fov_region=None,
               spins_per_voxel=8, jitter="stratified", seed=0, sampling="per_tissue",
               mt=False, diffusion=False, chi_to_b0=True, shim_order=2,
               t2star="auto", macroscopic_r2prime=None, field_maps=None,
               variability=None, r2star_acquisition=None) -> Phantom:
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
    ``variability`` (default None: none) = {"t1": sd, "t2": sd, "pd": sd, "length": L [m],
    "seed": 0}: each tissue's T1, T2 and PD are multiplied by its own mean-1 log-normal field
    of relative SD ``sd`` and Gaussian correlation exp(-r^2 / 2L^2) (``gaussian_random_field``),
    evaluated at voxel centres on the full anatomy before ``fov_region``. Its seed is separate
    from ``seed`` (positions), so jitter does not change the texture. It models within-tissue
    texture, not the population SDs of the table (``Distribution.sd``).
    """
    if sampling not in ("per_tissue","categorical","systematic","proportional") or jitter not in (None,"stratified","inplane"):
        raise ValueError("invalid sampling or jitter mode")
    if t2star not in ("auto","off","table","separate"):
        raise ValueError("t2star must be auto, separate, off or table")
    if t2star == "auto":
        t2star = "separate" if chi_to_b0 else "table"
    if t2star == "separate" and not chi_to_b0:
        raise ValueError("t2star='separate' needs chi_to_b0 (the field it separates)")
    if anatomy.map_field_T is not None and field_T != anatomy.map_field_T:
        raise ValueError(f"relaxation maps were acquired at {anatomy.map_field_T} T; no automatic field scaling")
    counts = _subvoxel_counts(spins_per_voxel,3)
    k = math.prod(counts)
    props = _properties(anatomy,table,field_T)
    maps_pair = None
    if chi_to_b0:
        maps_pair = field_maps if field_maps is not None else anatomy_field_maps(anatomy,table,field_T,shim_order=shim_order)
    if t2star == "table" and chi_to_b0 and macroscopic_r2prime is None and "intrinsic_r2prime" not in anatomy.maps:
        raise ValueError("observed T2* plus chi requires macroscopic_r2prime calibration")
    if macroscopic_r2prime is not None:
        macro = torch.as_tensor(macroscopic_r2prime,dtype=next(iter(anatomy.fractions.values())).dtype)
        if macro.ndim and macro.shape != anatomy.shape or not torch.isfinite(macro).all() or (macro < 0).any():
            raise ValueError("macroscopic_r2prime must be a nonnegative scalar or full anatomy map")
        if fov_region is not None and macro.ndim:
            macro = macro[fov_region]
    else:
        macro = 0.
    if fov_region is not None:
        if len(fov_region) != 3 or any(not isinstance(s, slice) or s.indices(n)[2] != 1
                                       for s, n in zip(fov_region, anatomy.shape)):
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
    cells = torch.cartesian_prod(*(torch.arange(c) for c in counts)).reshape(k,3)
    names = sorted(name for name,p in props.items() if name in anatomy.fractions and p.pd.mean > 0)
    signal_fraction = None
    if "M0" in anatomy.maps:
        density = anatomy.maps["M0"]
        signal_fraction = sum((anatomy.fractions[name] for name in names),torch.zeros_like(ref))
        if (density < 0).any() or ((density > 0) & (signal_fraction == 0)).any():
            raise ValueError("M0 must be nonnegative and supported by signal-producing tissue")
    category = None
    if sampling in ("categorical", "systematic"):
        # Sequential thresholds avoid a (voxels,K,tissues) temporary.
        u = torch.rand((*anatomy.shape,k),dtype=ref.dtype,generator=gen)
        if sampling == "systematic":  # (j + U) / K, one U per voxel, dealt to the subcells at random
            u = (torch.arange(k,dtype=ref.dtype)+u[...,:1])/k
            u = torch.gather(u,-1,torch.argsort(torch.rand(u.shape,generator=gen),dim=-1))
        category = torch.full(u.shape,-1,dtype=torch.long)
        cumulative = torch.zeros_like(ref)
        for i,name in enumerate(names):
            previous = cumulative
            cumulative = cumulative+anatomy.fractions[name]
            category[(u >= previous[...,None]) & (u < cumulative[...,None])] = i
    parts = {key:[] for key in ("pos","pd","t1","t2","db0","f","rate","t2b","diff")}
    active_broadening = False
    for ti,name in enumerate(names):
        p = props[name]
        fraction = anatomy.fractions[name]
        mask = fraction > 0 if category is None else (category == ti).any(-1)
        if "M0" in anatomy.maps:
            mask = mask & (anatomy.maps["M0"] > 0)
        if not mask.any():
            continue
        idx = mask.nonzero()
        n = len(idx)
        u = torch.rand((n,k,3),dtype=ref.dtype,generator=gen) if jitter else torch.full((n,k,3),.5,dtype=ref.dtype)
        if jitter == "inplane":  # z at the subcell centres: a quadrature of the slice profile, no z sampling noise
            u[...,0] = .5
        offsets = (cells+u)/torch.tensor(counts)-.5
        origin = torch.tensor(anatomy.origin,dtype=ref.dtype)
        spacing = torch.tensor(anatomy.voxel,dtype=ref.dtype)
        pos = origin+(idx[:,None,:]+offsets).flip(-1)*spacing.flip(-1)
        select = torch.ones((n,k),dtype=torch.bool) if category is None else category[mask] == ti
        if sampling == "proportional":  # n_t = max(1, round(k f)) of the k subcells, at random
            n_t = (fraction[mask]*k).round().clamp(1, k)
            rank = torch.argsort(torch.argsort(torch.rand((n,k),generator=gen),dim=1),dim=1)
            select = rank < n_t[:,None]
        expand = lambda value: (value[:,None].expand(-1,k) if value.ndim else value.expand(n,k))[select]
        parameter = lambda key,default: anatomy.maps[key][mask] if key in anatomy.maps else ref.new_tensor(default)
        density = parameter("M0",p.pd.mean)
        if signal_fraction is not None:
            density = density/signal_fraction[mask]
        r1, r2 = parameter("R1",1/p.t1.mean),parameter("R2",1/p.t2.mean)
        r2_base = r2  # R2' = R2* - R2 uses the untextured R2: texture must not create negative R2'
        if (name, "t1") in factors:  # T1 x factor
            r1 = r1 / factors[name, "t1"][mask]
        if (name, "t2") in factors:
            r2 = r2 / factors[name, "t2"][mask]
        if (name, "pd") in factors:
            density = density * factors[name, "pd"][mask]
        if (density < 0).any() or (r1 <= 0).any() or (r2 <= 0).any():
            raise ValueError("M0 must be nonnegative; R1 and R2 positive")
        pd = expand(density) / k
        if sampling == "proportional":
            pd = expand(density*fraction[mask]/n_t)
        elif category is None:
            pd = pd*expand(fraction[mask])
        pos = pos[select]
        db0 = torch.zeros_like(pd)
        if maps_pair is not None:
            for fm in maps_pair:
                db0 += sample(fm.b0[None],ref.new_tensor(fm.origin),ref.new_tensor(fm.spacing),pos)[:,0]
        if t2star != "off":
            rate = None
            if "intrinsic_r2prime" in anatomy.maps:
                rate = anatomy.maps["intrinsic_r2prime"][mask]
            elif t2star in ("table", "separate"):
                rstar = parameter("R2star",1/p.t2star.mean if p.t2star else 1/p.t2.mean)
                if macro_est is not None:  # an estimate: clip, a mismatch is not an inconsistent calibration
                    rate = (rstar-r2_base-macro_est[mask].to(rstar.dtype)).clamp_min(0)
                else:
                    rate = rstar-r2_base-(macro[mask] if isinstance(macro,Tensor) and macro.ndim else macro)
            if rate is not None:
                if (rate < -1e-6).any():
                    raise ValueError("R2* - R2 - macroscopic R2' is negative; incompatible calibration")
                active_broadening |= bool((rate > 0).any())
                q = (torch.arange(k,dtype=ref.dtype)+.5)/k
                q = q[torch.argsort(torch.rand(n,k,generator=qgen),dim=1)]
                db0 += expand(rate.clamp_min(0))*torch.tan(math.pi*(q-.5))[select]/(2*math.pi)
        f = p.bound_fraction.mean if p.bound_fraction else 0.
        if mt and (p.bound_fraction is None or (f > 0 and (p.exchange_rate is None or p.bound_t2 is None))):
            raise ValueError(f"missing MT properties for {name}")
        if diffusion and p.diffusivity is None:
            raise ValueError(f"missing diffusivity for {name}")
        # Chemical species share spatial/relaxation/MT parameters within this class (the myelin water,
        # if any, has its own T2).
        ff = p.fat_fraction.mean if p.fat_fraction else 0.
        mwf = p.myelin_water_fraction.mean if p.myelin_water_fraction else 0.
        species = [((1-ff)*(1-mwf),0.,None)] if ff < 1 else []
        if ff < 1 and mwf > 0:
            species.append(((1-ff)*mwf,0.,p.myelin_water_t2.mean))
        if ff > 0:
            total_amp = sum(amp for _,amp in FAT_PEAKS)
            species += [(ff*amp/total_amp,shift*1e-6*GAMMA_BAR*field_T,None) for shift,amp in FAT_PEAKS]
        for weight,shift,t2_own in species:
            parts["pos"].append(pos)
            parts["pd"].append(pd*weight)
            parts["t1"].append(1/expand(r1))
            parts["t2"].append(1/expand(r2) if t2_own is None else torch.full_like(pd,t2_own))
            parts["db0"].append(db0+shift)
            if mt:
                parts["f"].append(torch.full_like(pd,f/(1-f)))
                parts["rate"].append(torch.full_like(pd,p.exchange_rate.mean if f else 0.))
                parts["t2b"].append(torch.full_like(pd,p.bound_t2.mean if f else 12e-6))
            if diffusion:
                parts["diff"].append(torch.full_like(pd,p.diffusivity.mean))
    if active_broadening and k == 1:
        warnings.warn("one spin per tissue/voxel cannot represent T2' broadening; use more spins",UserWarning,stacklevel=2)
    cat = lambda key: torch.cat(parts[key]) if parts[key] else torch.empty((0,3) if key == "pos" else (0,),dtype=ref.dtype)
    ph = Phantom(*(cat(key) for key in ("pos","pd","t1","t2","db0")))
    if mt:
        ph.bound = BoundPool(cat("f"),cat("rate"),ph.t1,cat("t2b"),model="rate")
    if diffusion:
        ph.diffusion = cat("diff")
    return ph


@dataclass
class DigitalPhantom:
    """An anatomy ready to simulate: fractional tissue volumes (``Anatomy``), ONE field strength's
    tissue table, the ``to_phantom`` options that complete it (texture, spin seed, ...) and
    ``meta``, the record of how it was made (``sample_head`` puts every draw there; the dataset
    writer stores it). ``DigitalPhantom(a, DEFAULT_TISSUES[3.], 3.).to_phantom(**kw)`` is
    ``to_phantom(a, DEFAULT_TISSUES, 3., **kw)``, bit for bit; keyword arguments override
    ``options``.
    """
    anatomy: Anatomy
    tissues: dict[str, TissueProperties]
    field_T: float = 3.
    options: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not math.isfinite(self.field_T) or self.field_T <= 0:
            raise ValueError("field_T must be positive [T]")
        if not all(isinstance(p, TissueProperties) for p in self.tissues.values()):
            raise ValueError("tissues maps class names to TissueProperties at field_T")

    @property
    def table(self) -> dict:
        return {self.field_T: self.tissues}

    def to_phantom(self, **overrides) -> Phantom:
        return to_phantom(self.anatomy, self.table, self.field_T, **(self.options | overrides))

    def field_maps(self, **kwargs):
        """``anatomy_field_maps`` of this anatomy and table (reuse via ``to_phantom(field_maps=...)``)."""
        return anatomy_field_maps(self.anatomy, self.table, self.field_T, **kwargs)


def _bspline_weights(n, h, spacing):
    """(n, m) uniform cubic B-spline weights at n grid points h apart, of control points
    ``spacing`` apart that cover the grid with one to spare beyond each end."""
    t = torch.arange(n, dtype=torch.float64) * (h / spacing)
    i = t.floor().long()
    u = t - i
    w = torch.stack(((1 - u) ** 3, 3 * u ** 3 - 6 * u ** 2 + 4, -3 * u ** 3 + 3 * u ** 2 + 3 * u + 1, u ** 3), -1) / 6
    return torch.zeros(n, int(i[-1]) + 4, dtype=torch.float64).scatter_(1, i[:, None] + torch.arange(4), w)


def random_deformation(shape, voxel, generator: torch.Generator, *, magnitude=3e-3, spacing=40e-3):
    """Smooth random displacement (3, z, y, x) [m] on a (z, y, x) grid, components along z, y, x:
    a cubic B-spline of control-point displacements drawn independently from N(0, magnitude^2),
    ``spacing`` [m] apart. Returns (displacement, control points (3, cz, cy, cx))."""
    if not magnitude >= 0 or not spacing > 0:
        raise ValueError("deformation needs magnitude >= 0 and spacing > 0 [m]")
    weights = [_bspline_weights(n, h, spacing) for n, h in zip(shape, voxel)]
    control = torch.randn((3, *(w.shape[1] for w in weights)), generator=generator, dtype=torch.float64) * magnitude
    return torch.einsum("kabc,za,yb,xc->kzyx", control, *weights), control


def deformation_jacobian(displacement: Tensor, voxel) -> Tensor:
    """det(I + grad d) on the grid (central differences, one-sided at the faces): the local
    volume ratio of x -> x + d(x). Positive everywhere means the map does not fold."""
    g = [[torch.gradient(displacement[k], spacing=[h], dim=l)[0] for l, h in enumerate(voxel)] for k in range(3)]
    for k in range(3):
        g[k][k] = g[k][k] + 1
    return (g[0][0] * (g[1][1] * g[2][2] - g[1][2] * g[2][1]) - g[0][1] * (g[1][0] * g[2][2] - g[1][2] * g[2][0])
            + g[0][2] * (g[1][0] * g[2][1] - g[1][1] * g[2][0]))


def deform(anatomy: Anatomy, displacement: Tensor) -> Anatomy:
    """Pull the anatomy back through x -> x + d(x): every volume's new value at x is its old value
    at x + d(x) (trilinear; outside the grid, air and ``chi_background``). Fractions stay a convex
    combination of fractions, so they still sum to at most one; volumes change by the Jacobian."""
    shape = anatomy.shape
    if tuple(displacement.shape) != (3, *shape):
        raise ValueError("displacement must be (3, z, y, x) on the anatomy grid [m]")
    ref = next(iter(anatomy.fractions.values()))
    axes = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) for n in shape), indexing="ij")
    # grid_sample, align_corners=True: index i of n sits at -1 + 2 i / (n - 1); its last axis is (x, y, z)
    grid = torch.stack([2 * (a + displacement[k] / h) / max(n - 1, 1) - 1
                        for k, (a, h, n) in reversed(list(enumerate(zip(axes, anatomy.voxel, shape))))], -1)
    grid = grid[None].to(ref.dtype)
    del axes

    def warp(volume, fill=0.):
        out = torch.nn.functional.grid_sample((volume - fill)[None, None], grid, mode="bilinear",
                                              padding_mode="zeros", align_corners=True)
        return out[0, 0] + fill

    fractions = _renormalize({k: warp(v).clamp(0, 1) for k, v in anatomy.fractions.items()})
    chi = None if anatomy.chi is None else warp(anatomy.chi, anatomy.chi_background)
    return Anatomy(fractions, anatomy.voxel, anatomy.origin, chi, {k: warp(v) for k, v in anatomy.maps.items()},
                   anatomy.chi_background, anatomy.map_field_T)


def draw_tissues(tissues: dict, generator: torch.Generator) -> tuple[dict, dict]:
    """One subject's tissue table from a table of population Distributions: every property with a
    reported SD is drawn once per class, log-normal with the table's mean and SD (normal for a
    nonpositive mean such as susceptibility), independently; properties without an SD keep their
    mean. Fractions are kept below 0.999 and T2* at or below T2. Returns (table, draws)."""
    out, draws = {}, {}
    for name in sorted(tissues):
        p, values = tissues[name], {}
        for f in fields(p):
            d = getattr(p, f.name)
            if d is None or not d.sd:
                continue
            z = float(torch.randn((), generator=generator, dtype=torch.float64))
            if d.mean > 0:
                s2 = math.log1p((d.sd / d.mean) ** 2)
                values[f.name] = d.mean * math.exp(math.sqrt(s2) * z - s2 / 2)
            else:
                values[f.name] = d.mean + d.sd * z
        for key in ("bound_fraction", "fat_fraction"):
            if key in values:
                values[key] = min(values[key], .999)
        t2 = values.get("t2", p.t2.mean)
        if p.t2star is not None and values.get("t2star", p.t2star.mean) > t2:
            values["t2star"] = t2
        draws[name] = values
        out[name] = replace(p, **{k: Distribution(v, None, "Drawn by draw_tissues from: " + getattr(p, k).citation,
                                                  getattr(p, k).assumed) for k, v in values.items()})
    return out, draws


# Scalp layers at the vertex, 30 healthy adults, 22 MHz ultrasound: epidermis + dermis 1.61 +- 0.23 mm,
# subcutaneous tissue 4.14 +- 1.10 mm, whole scalp 5.76 +- 1.14 mm (Li et al. 2024, Skin Res Technol
# 30:e13863, doi:10.1111/srt.13863, Table 5).
SCALP_LAYERS = (1.61e-3, 4.14e-3)


def layer_scalp(anatomy: Anatomy, dermis: float = SCALP_LAYERS[0], subcutaneous: float = SCALP_LAYERS[1]) -> Anatomy:
    """BrainWeb lumps the scalp's skin and its subcutaneous (fatty) layer into one class, "Muscle/Skin"
    (``skin`` here), which T2-weighted scans show as a dark skin line over a bright fat layer. Split it
    by depth below the head's surface: the outer ``dermis`` [m] stays skin, the next ``subcutaneous``
    [m] becomes adipose tissue (``fat``), deeper skin stays skin. Depth: the Euclidean distance of a
    voxel centre below the boundary of the head (1 - background > 0.5), each voxel weighted by its
    overlap with the layer (one voxel size across). Defaults: ``SCALP_LAYERS``."""
    from scipy import ndimage

    fr = anatomy.fractions
    if not {"skin", "fat", "background"} <= set(fr):
        raise ValueError("layer_scalp needs the background, skin and fat classes (BrainWeb)")
    inside = ((1 - fr["background"]) > .5).numpy()
    h = float(np.mean(anatomy.voxel))
    # distance to the nearest outside voxel centre, less half a voxel: the boundary lies between them
    depth = ndimage.distance_transform_edt(inside, sampling=anatomy.voxel) - h / 2
    lo, hi = dermis, dermis + subcutaneous
    w = np.clip((np.minimum(depth + h / 2, hi) - np.maximum(depth - h / 2, lo)) / h, 0, 1) * inside
    move = fr["skin"] * torch.from_numpy(w).to(fr["skin"].dtype)
    return replace(anatomy, fractions=fr | {"skin": fr["skin"] - move, "fat": fr["fat"] + move})


# Deep grey nuclei in BrainWeb's models, which lie in MNI (ICBM) space and label them grey matter:
# ellipsoids (centre |x|, y, z; semi-axes) [mm], drawn around the nuclei in the models' grey-matter
# maps (subjects 4, 38, 44 and 54 by eye) and centred near their usual MNI centroids. They hold no
# cortex: the insula, cingulate and amygdala lie outside them. The caudate is a head and a body.
DEEP_GREY = {"caudate": (((14., 14., 9.), (6., 13., 12.)), ((17., -6., 21.), (5., 14., 8.))),
             "putamen": (((25., 1., 1.), (7., 20., 13.)),),
             "pallidum": (((19., -4., -1.), (5., 9., 6.)),),
             "thalamus": (((11., -19., 6.), (9., 15., 10.)),)}


def label_deep_grey(anatomy: Anatomy) -> Anatomy:
    """Split the deep grey nuclei (caudate, putamen, pallidum, thalamus: ``DEEP_GREY``) out of the
    ``gm`` class: at 3 T their iron gives them T2 near white matter's (57-76 ms against cortex's
    ~95), so T2-weighted scans show them dark where a single grey-matter class makes them bright.
    Grey-matter membership inside a nucleus's region moves to that nucleus (the nearest region's
    where they overlap); white matter and CSF stay. Needs MNI coordinates, i.e. a BrainWeb model
    before ``deform``; much of the pallidum is white matter there (bright on T1) and stays so."""
    fr = anatomy.fractions
    if "gm" not in fr or set(DEEP_GREY) & set(fr):
        raise ValueError("label_deep_grey needs a gm class and no deep grey classes yet (BrainWeb)")
    x, y, z = ((o + np.arange(n) * h) * 1e3 for o, n, h in zip(anatomy.origin, anatomy.shape[::-1], anatomy.voxel[::-1]))
    score = np.full((len(DEEP_GREY),) + tuple(anatomy.shape), -np.inf, dtype=np.float32)
    for i, regions in enumerate(DEEP_GREY.values()):
        for (cx, cy, cz), (ax, ay, az) in regions:
            r2 = ((((z - cz) / az) ** 2)[:, None, None] + (((y - cy) / ay) ** 2)[None, :, None]
                  + (((np.abs(x) - cx) / ax) ** 2)[None, None, :])
            np.maximum(score[i], 1 - r2, out=score[i])
    best, inside = torch.from_numpy(score.argmax(0)), torch.from_numpy(score.max(0) > 0)
    gm, zero = fr["gm"], torch.zeros_like(fr["gm"])
    out = {k: torch.where(inside, zero, gm) if k == "gm" else v for k, v in fr.items()}
    for i, name in enumerate(DEEP_GREY):
        out[name] = torch.where(inside & (best == i), gm, zero)
    return replace(anatomy, fractions=out)


# Lateral-ventricle volume [mL] by age: Yamada et al. 2023 (Eur Radiol 33:7099, doi:10.1007/s00330-023-09632-x,
# Table 3; 133 healthy Japanese adults, AI segmentation of 3D T1): < 40 y 17.3 +- 7.2 (n 47), 40-59 y
# 19.4 +- 7.5 (n 49), >= 60 y 33.4 +- 13.4 (n 37); placed at 30, 50 and 70 y, linear between, flat beyond.
BRAINWEB_AGE = 30.  # BrainWeb's 20 models are young adults, taken as 30
VENTRICLE_AGE = ((30., 17.3, 7.2), (50., 19.4, 7.5), (70., 33.4, 13.4))


def ventricle_factor(age: float) -> float:
    """How much larger the lateral ventricles are at ``age`` [years] than at ``BRAINWEB_AGE``: the
    ratio of ``VENTRICLE_AGE``'s means (1 at 30, 1.12 at 50, 1.93 at 70 and beyond). BrainWeb's
    young adults carry the between-subject spread."""
    a, m, _ = (np.array(c) for c in zip(*VENTRICLE_AGE))
    return float(np.interp(age, a, m) / np.interp(BRAINWEB_AGE, a, m))


# Lateral ventricles in BrainWeb's MNI space: CSF inside |x| 1.5-32, y -48..38, z -2..36 mm, less the
# quadrigeminal/ambient cisterns (y < -22, |x| < 9) and the insular cisterns (z < 8, |x| > 24), in
# components that reach 12 mm inside the brain's hull (checked by eye on subjects 4, 38, 44, 54;
# 13-29 mL each at MNI size, ~9-19 mL resized: Yamada 2023 young adults 17.3 +- 7.2 mL).
def ventricle_mask(anatomy: Anatomy) -> np.ndarray:
    """The lateral ventricles (bool, z/y/x) of a BrainWeb model in MNI space (before ``deform``)."""
    from scipy import ndimage

    fr = anatomy.fractions
    x, y, z = ((o + np.arange(n) * h) * 1e3 for o, n, h in zip(anatomy.origin, anatomy.shape[::-1], anatomy.voxel[::-1]))
    ax, Y, Z = np.abs(x)[None, None, :], y[None, :, None], z[:, None, None]
    region = (ax >= 1.5) & (ax <= 32) & (Y >= -48) & (Y <= 38) & (Z >= -2) & (Z <= 36)
    region &= ~((Y < -22) & (ax < 9)) & ~((Z < 8) & (ax > 24))
    csf = (fr["csf"].numpy() > .5) & region
    brain = sum(v for k, v in fr.items() if k in ("gm", "wm", *DEEP_GREY)).numpy() > .5
    hull = ndimage.binary_fill_holes(ndimage.binary_closing(brain, iterations=round(4e-3 / anatomy.voxel[1])))
    deep = ndimage.distance_transform_edt(hull, sampling=anatomy.voxel) > 12e-3
    lab, _ = ndimage.label(csf)
    keep = np.unique(lab[csf & deep])
    keep = keep[keep > 0]
    sizes = ndimage.sum(np.ones_like(lab), lab, keep) * np.prod(anatomy.voxel)
    return np.isin(lab, keep[sizes > 0.2e-6])


def ventricle_displacement(anatomy: Anatomy, mask: np.ndarray, offset: float, decay: float = 10e-3) -> Tensor:
    """(3, z, y, x) [m] pull-back displacement (new(x) = old(x + displacement(x))) that moves the walls
    of each lateral ventricle (``mask`` split at x = 0) out by ``offset`` [m]: points within
    ``offset`` of a ventricle take its nearest interior voxel (CSF), points beyond take what lay
    nearer it by offset exp(-(d - offset) / ``decay``); the inside is unchanged. The tissue around
    is compressed, not pushed away (the pull-back keeps no mass): enlargement as periventricular
    loss. The two sides' maps add, so the septum between them stays."""
    from scipy import ndimage

    h = np.array(anatomy.voxel).reshape(3, 1, 1, 1)
    grid = np.indices(anatomy.shape) * h
    x = anatomy.origin[0] + np.arange(anatomy.shape[2]) * anatomy.voxel[2]
    out = np.zeros((3, *anatomy.shape))
    for side in (x < 0, x >= 0):
        m = mask & side[None, None, :]
        if not m.any():
            continue
        core = ndimage.binary_erosion(m)
        core = core if core.any() else m
        d = ndimage.distance_transform_edt(~m, sampling=anatomy.voxel)
        _, idx = ndimage.distance_transform_edt(~core, sampling=anatomy.voxel, return_indices=True)
        w = np.where(d <= 0, 0., np.where(d <= offset, 1., offset / np.maximum(d, 1e-9) * np.exp(-(d - offset) / decay)))
        w[d > offset + 5 * decay] = 0.
        out += (idx * h - grid) * w
    return torch.from_numpy(np.stack([ndimage.gaussian_filter(c, 1.) for c in out]))


def ventricle_offset(anatomy: Anatomy, factor: float, mask: np.ndarray | None = None) -> tuple[float, float]:
    """The ``ventricle_displacement`` wall offset [m] that scales the lateral ventricles' CSF by
    ``factor`` (bisection, to 2 %), and their volume before [m^3]."""
    mask = ventricle_mask(anatomy) if mask is None else mask
    csf = anatomy.fractions["csf"] * torch.from_numpy(mask).to(anatomy.fractions["csf"].dtype)
    v0 = float(csf.sum())
    if factor <= 1:
        return 0., v0 * float(np.prod(anatomy.voxel))
    ref = anatomy.fractions["csf"]
    region = torch.from_numpy(ndimage_dilate(mask, 20e-3, anatomy.voxel)).to(ref.dtype)
    vol = lambda a: float((deform(Anatomy({"c": ref * region}, anatomy.voxel, anatomy.origin),
                                  ventricle_displacement(anatomy, mask, a).to(ref.dtype)).fractions["c"]).sum())
    base = vol(0.)
    target = base + (factor - 1) * v0
    lo, hi = 0., 1e-3
    while vol(hi) < target and hi < 16e-3:
        lo, hi = hi, 2 * hi
    for _ in range(12):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if vol(mid) < target else (lo, mid)
        if hi - lo < .02 * hi:
            break
    return (lo + hi) / 2, v0 * float(np.prod(anatomy.voxel))


def ndimage_dilate(mask: np.ndarray, radius: float, voxel) -> np.ndarray:
    from scipy import ndimage

    return ndimage.distance_transform_edt(~mask, sampling=voxel) <= radius


# Sulcal width, the mean span of the intrasulcal CSF along the normals of the sulcal mesh (BrainVISA on
# FreeSurfer surfaces, 1.5 T T1, five major sulci per hemisphere), in the PATH study: 1.27 +- 0.17 mm at
# 47.2 y (n 403) and 1.49 +- 0.20 mm at 67.0 y (n 390) (Jin et al. 2018, Front Aging Neurosci 10:339,
# doi:10.3389/fnagi.2018.00339, Results and Table 1); linear between and beyond. Kochunov et al. 2005
# (Hum Brain Mapp 26:210; 90 adults aged 20-82, the same definition) report 0.71 mm per decade, six
# times steeper.
SULCAL_WIDTH_AGE = ((47.2, 1.27e-3, 0.17e-3), (67.04, 1.49e-3, 0.20e-3))


def sulcal_width(age: float) -> tuple[float, float]:
    """Mean and between-subject SD of sulcal width [m] at ``age`` [years] (``SULCAL_WIDTH_AGE``)."""
    (a0, w0, s0), (a1, w1, s1) = SULCAL_WIDTH_AGE
    t = (float(age) - a0) / (a1 - a0)
    return w0 + t * (w1 - w0), max(s0 + t * (s1 - s0), 0.)


def _deep_grey_r2(origin, shape, voxel):
    """Each ``DEEP_GREY`` ellipsoid's normalised squared radius on a grid (origin x, y, z; shape and voxel z, y, x; MNI)."""
    x, y, z = ((o + np.arange(n) * h) * 1e3 for o, n, h in zip(origin, shape[::-1], voxel[::-1]))
    for regions in DEEP_GREY.values():
        for (cx, cy, cz), (ax, ay, az) in regions:
            yield ((((z - cz) / az) ** 2)[:, None, None] + (((y - cy) / ay) ** 2)[None, :, None]
                   + (((np.abs(x) - cx) / ax) ** 2)[None, None, :])


def open_sulci(anatomy: Anatomy, width: float, separation: float = 3e-3, opposed: float = 0.5) -> Anatomy:
    """Open closed sulci to ``width`` [m] of CSF. BrainWeb's fuzzy models (1.5 T scans at 1 mm) give
    narrow sulci to grey matter: two cortical banks touch with no CSF between them, and T2-weighted
    slices show speckle where real ones show sulcal lines. A closed sulcus is grey matter reached from
    two white-matter banks: the lambda-medial axis of the region outside white matter (Chazal and
    Lieutier 2005), i.e. neighbouring voxels whose nearest white-matter points lie >= ``separation``
    apart, in directions facing each other (cosine below -``opposed``: within 60 deg of opposite, so
    a shallow groove in the white surface is not a sulcus). Bumps of the white surface smaller than
    that make no axis (ridges of the distance to white matter do: an earlier version opened those and
    made specks). The cerebellum's fissures get the same width (no measurement there). Kept: grey
    matter > 0.2, outside the deep grey regions (``DEEP_GREY`` scaled by 1.3), in sheets touching the
    CSF around the brain. A sheet voxel's distance to the midplane is half the difference of its
    distances to the two banks, its neighbours' adds their distance to the sheet; CSF fills width / 2
    either side, voxel-weighted, never below what is there, taken from grey matter. Needs MNI
    coordinates (a BrainWeb model before ``deform``) and a fine grid: ``sample_head`` opens at the
    native 0.5 mm and then resamples."""
    from scipy import ndimage

    if width <= 0:
        return anatomy
    fr = anatomy.fractions
    gm_all, wm_all, csf_all = (fr[k].numpy() for k in ("gm", "wm", "csf"))
    brain = (gm_all + wm_all) > .5
    if not brain.any():
        return anatomy
    vox = np.array(anatomy.voxel, dtype=np.float64)
    pad = [int(np.ceil(4e-3 / v)) for v in vox]
    box = []
    for k in range(3):
        idx = np.nonzero(brain.any(tuple(j for j in range(3) if j != k)))[0]
        box.append(slice(max(int(idx[0]) - pad[k], 0), int(idx[-1]) + pad[k] + 1))
    box = tuple(box)
    gm, wm, csf = (a[box].astype(np.float32) for a in (gm_all, wm_all, csf_all))
    white = wm > .5
    F = ndimage.distance_transform_edt(~white, sampling=vox, return_distances=False, return_indices=True).astype(np.int32)
    # the strongest shock at each voxel: its partner across it (axis, side) and the separation
    best = np.zeros(gm.shape, np.float32)
    part = np.full(gm.shape, -1, np.int8)
    grid = np.indices(gm.shape, dtype=np.int32)
    for ax in range(3):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[ax], b[ax] = slice(0, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        sep = np.sqrt(sum(((F[k][a] - F[k][b]) * vox[k]) ** 2 for k in range(3))).astype(np.float32)
        dot = sum((F[k][a] - grid[k][a]).astype(np.float32) * (F[k][b] - grid[k][b]).astype(np.float32)
                  * np.float32(vox[k] ** 2) for k in range(3))
        norm = np.sqrt(sum(((F[k][a] - grid[k][a]) * vox[k]) ** 2 for k in range(3))
                       * sum(((F[k][b] - grid[k][b]) * vox[k]) ** 2 for k in range(3))).astype(np.float32)
        hit = np.where((sep >= separation) & (dot < -opposed * norm), sep, 0).astype(np.float32)
        for sl, code in ((a, 2 * ax), (b, 2 * ax + 1)):
            upd = hit > best[sl]
            best[sl] = np.where(upd, hit, best[sl])
            part[sl] = np.where(upd, code, part[sl])
    del grid
    origin = tuple(o + sl.start * h for o, sl, h in zip(anatomy.origin, box[::-1], anatomy.voxel[::-1]))
    deep = np.zeros(gm.shape, bool)
    for r2 in _deep_grey_r2(origin, gm.shape, anatomy.voxel):
        deep |= r2 < 1.3 ** 2
    sheet = (best > 0) & (gm > .2) & ~deep
    # the CSF around the brain: CSF > 0.5 connected to the space outside the (closed, filled) brain
    c = csf > .5
    outside = ~ndimage.binary_fill_holes(ndimage.binary_closing(brain[box], iterations=max(pad[0] * 3 // 4, 1)))
    lab, _ = ndimage.label(c | outside)
    outer = np.isin(lab, np.unique(lab[outside])) & c
    lab, _ = ndimage.label(sheet, structure=np.ones((3, 3, 3)))
    touch = np.unique(lab[ndimage.binary_dilation(outer, iterations=2) & sheet])
    sheet = np.isin(lab, touch[touch > 0])
    del lab, outer, outside, c
    # each voxel near a sheet: half the difference of its distances to the two banks, its own (its
    # nearest white point) and the other (whichever of the nearest sheet voxel's two banks is farther)
    h = float(np.mean(vox))
    dist, I = ndimage.distance_transform_edt(~sheet, sampling=vox, return_indices=True)
    near = dist <= width / 2 + h
    del dist
    p = np.stack(np.nonzero(near))
    s_ = I[:, near]
    del I
    code = part[s_[0], s_[1], s_[2]].astype(np.int64)
    partner = s_ + np.where(np.arange(3)[:, None] == code // 2, np.where(code % 2 == 0, 1, -1), 0)
    own, a_, b_ = (F[:, i[0], i[1], i[2]] * vox[:, None] for i in (p, s_, partner))
    other = np.where(np.linalg.norm(a_ - own, axis=0) > np.linalg.norm(b_ - own, axis=0), a_, b_)
    pos = p * vox[:, None]
    r = np.abs(np.linalg.norm(pos - other, axis=0) - np.linalg.norm(pos - own, axis=0)) / 2
    del F, part, best
    target = np.zeros(gm.shape)
    target[near] = np.clip((width / 2 - r) / h + .5, 0, 1)
    move = np.clip(np.minimum(target - csf, gm), 0, None) * ~deep
    full = torch.zeros_like(fr["gm"])
    full[box] = torch.from_numpy(move).to(full.dtype)
    return replace(anatomy, fractions=fr | {"gm": fr["gm"] - full, "csf": fr["csf"] + full})


# Head breadth (euryon-euryon) and length (glabella-opisthocranion) [m]: mean, SD and their correlation,
# from the ANSUR II public data (2012 Anthropometric Survey of U.S. Army Personnel, Gordon et al. 2015,
# NATICK/TR-15/007; ANSUR_II_MALE_Public.csv n = 4082, ANSUR_II_FEMALE_Public.csv n = 1986, via
# openlab.psu.edu/ansur2, computed 2026-10-10).
ANSUR_HEAD = {"male": dict(breadth=(0.1543, 0.0055), length=(0.1995, 0.0070), corr=0.18),
              "female": dict(breadth=(0.1478, 0.0052), length=(0.1898, 0.0074), corr=0.22)}


def _head_mask(anatomy: Anatomy) -> np.ndarray:
    """Voxels more than half tissue (any class but background; outside a resampled grid is empty)."""
    return (sum(v for k, v in anatomy.fractions.items() if k != "background") > .5).numpy()


def head_dimensions(anatomy: Anatomy, depth: float = 0.1) -> tuple[float, float]:
    """(breadth, length) [m] of a head: the largest left-right (x) and anterior-posterior (y) extents of
    1 - background > 0.5 within ``depth`` below the vertex, above the brow and the nose, where a
    caliper measures them (euryon-euryon, glabella-opisthocranion)."""
    head = _head_mask(anatomy)
    zs = np.nonzero(head.any((1, 2)))[0]
    top = int(zs.max())
    sl = head[max(top - round(depth / anatomy.voxel[0]), 0): top + 1]
    ys, xs = np.nonzero(sl.any((0, 2)))[0], np.nonzero(sl.any((0, 1)))[0]
    return (xs.max() - xs.min() + 1) * anatomy.voxel[2], (ys.max() - ys.min() + 1) * anatomy.voxel[1]


_INTRACRANIAL = ("gm", "wm", "csf", "dura", "vessels", *DEEP_GREY)


def intracranial_mask(anatomy: Anatomy) -> np.ndarray:
    """The skull's contents: brain, CSF, dura and vessels (more than half a voxel), closed by 2 voxels
    and filled."""
    from scipy import ndimage

    inside = sum(v for k, v in anatomy.fractions.items() if k in _INTRACRANIAL).numpy() > .5
    return ndimage.binary_fill_holes(ndimage.binary_closing(inside, iterations=2))


def _extents(mask: np.ndarray, voxel, top: int, depth: float) -> tuple[float, float]:
    sl = mask[max(top - round(depth / voxel[0]), 0): top + 1]
    ys, xs = np.nonzero(sl.any((0, 2)))[0], np.nonzero(sl.any((0, 1)))[0]
    return (xs.max() - xs.min() + 1) * voxel[2], (ys.max() - ys.min() + 1) * voxel[1]


def anthropometric_scaling(anatomy: Anatomy, generator: np.random.Generator, shell: bool = False):
    """Draw a sex (1:1) and a head breadth and length from ``ANSUR_HEAD`` (bivariate normal) and return
    the scale factors (z, y, x) that give ``anatomy``'s head that size (z: their geometric mean),
    the centre of the head's bounding box [m, (z, y, x) from the grid origin] and the draw. BrainWeb's
    models live in MNI space, scaled to the template: subject 04's head is 174 x 207 mm (breadth x
    length), its brain 1580 cm^3; an average man's 154 x 200 mm gives ~1250 cm^3. ``shell``: the
    factors scale the skull's contents only (``intracranial_mask``), the skull and scalp keeping the
    model's thickness around them (``_shell_translation``): head = factor x contents + shell at the
    caliper level; the same draw."""
    sex = "male" if generator.random() < .5 else "female"
    a = ANSUR_HEAD[sex]
    (mb, sb), (ml, sl), r = a["breadth"], a["length"], a["corr"]
    z0, z1 = generator.standard_normal(2)
    breadth, length = mb + sb * z0, ml + sl * (r * z0 + math.sqrt(1 - r * r) * z1)
    b0, l0 = head_dimensions(anatomy)
    sx, sy = float(breadth / b0), float(length / l0)
    head = _head_mask(anatomy)
    if shell:  # the contents' extents at the caliper level; the rest of the head's is shell
        top = int(np.nonzero(head.any((1, 2)))[0].max())
        bi, li = _extents(intracranial_mask(anatomy), anatomy.voxel, top, 0.1)
        sx, sy = float((breadth - (b0 - bi)) / bi), float((length - (l0 - li)) / li)
    centre = tuple(float((np.nonzero(head.any(tuple(j for j in range(3) if j != k)))[0][[0, -1]].mean()) * h)
                   for k, h in enumerate(anatomy.voxel))
    draw = {"sex": sex, "breadth": float(breadth), "length": float(length), "source_breadth": float(b0),
            "source_length": float(l0)}
    if shell:
        draw |= {"source_contents_breadth": float(bi), "source_contents_length": float(li)}
    return (math.sqrt(sx * sy), sy, sx), centre, draw


def _shell_translation(anatomy: Anatomy, factors, centre, smooth: float = 2e-3, reach=(15e-3, 30e-3)) -> Tensor:
    """(3, z, y, x) [m]: where resizing by ``factors`` about ``centre`` moves each point when the skull's
    contents scale and everything outside them moves with its nearest contents voxel (translated,
    not scaled, so the skull and scalp keep their thickness): (S - I)(p - c), p the point itself
    inside, the nearest ``intracranial_mask`` voxel outside, there Gaussian-smoothed by ``smooth`` [m]
    over outside points only (normalised convolution: the inside stays exact). Beyond ``reach`` [m]
    outside the contents (the face and neck, where the nearest point jumps) it blends linearly into
    plain scaling, (S - I)(x - c)."""
    from scipy import ndimage

    inside = intracranial_mask(anatomy)
    dist, idx = ndimage.distance_transform_edt(~inside, sampling=anatomy.voxel, return_indices=True)
    w = np.clip((reach[1] - dist) / (reach[1] - reach[0]), 0, 1)
    sigma = [smooth / h for h in anatomy.voxel]
    norm = ndimage.gaussian_filter((~inside).astype(float), sigma, mode="nearest")
    out = []
    for k, (f, c, h, n) in enumerate(zip(factors, centre, anatomy.voxel, anatomy.shape)):
        x = (np.arange(n) * h - c).reshape([-1 if j == k else 1 for j in range(3)])
        t = w * (f - 1) * (idx[k] * h - c) + (1 - w) * (f - 1) * x
        smoothed = ndimage.gaussian_filter(np.where(inside, 0., t), sigma, mode="nearest") / np.maximum(norm, 1e-6)
        out.append(np.where(inside, (f - 1) * np.broadcast_to(x, inside.shape), smoothed))
    return torch.from_numpy(np.stack(out))


def _sample_field(field: Tensor, pos: Tensor, voxel) -> Tensor:
    """Trilinear values of a (3, z, y, x) field at positions (3, z, y, x) [m from the grid origin],
    clamped to the grid."""
    shape = field.shape[1:]
    grid = torch.stack([2 * pos[k] / h / max(n - 1, 1) - 1 for k, (h, n) in reversed(list(enumerate(zip(voxel, shape))))], -1)
    out = torch.nn.functional.grid_sample(field[None].to(grid.dtype), grid[None], mode="bilinear", padding_mode="border",
                                          align_corners=True)
    return out[0]


BRAINWEB_SUBJECTS = (4, 5, 6, 18, 20, 38, *range(41, 55))
DEFAULT_TEXTURE = {"t1": .05, "t2": .05, "pd": .02, "length": .01}  # validation/anatomy/README.md, M4


def sample_head(seed: int, *, subject=None, field_T=3., voxel=(1e-3,) * 3, root=None, table=DEFAULT_TISSUES,
                magnitude=3e-3, spacing=40e-3, texture=DEFAULT_TEXTURE, dtype=torch.float32,
                scalp: str = "brainweb", size: str = "brainweb", yaw_sd: float = 0.,
                age: float | tuple[float, float] | None = None, deep_grey: bool = False,
                ventricle_age: float | tuple[float, float] | None = None,
                pitch: tuple[float, float] | None = None) -> DigitalPhantom:
    """A new head for every seed (ROADMAP M5, P1): a BrainWeb subject (``subject``, or one of the
    20 drawn by the seed), deformed by ``random_deformation`` (``magnitude``, ``spacing``; halved
    until the Jacobian exceeds 0.2 everywhere), one subject's tissue table (``draw_tissues``),
    within-tissue texture (``texture``, ``to_phantom(variability=...)``) and its own spin seed.
    Susceptibility -> B0, shim and the R2* policy are ``to_phantom``'s defaults. The result is
    deterministic in the seed, and ``meta`` records every draw (subject, control points,
    properties, seeds). ``root`` is the BrainWeb data directory (default: the environment
    variable BLOCHPARTY_BRAINWEB; validation/anatomy/fetch_brainweb.py fetches it). ``scalp``:
    "brainweb" keeps BrainWeb's skin class, "layered" splits it into skin and subcutaneous fat
    (``layer_scalp``) after the deformation. ``size``: "brainweb" keeps the model's (MNI-template)
    size, "anthropometric" scales the head to a breadth and length drawn from ANSUR II
    (``anthropometric_scaling``), composed with the deformation into one resampling;
    "anthropometric_shell" reaches the same size by scaling the skull's contents and keeping the
    skull's and scalp's thickness. ``yaw_sd``
    [deg]: the head turned about the scanner's z axis through its centre by N(0, yaw_sd^2) (0: as
    the model lies; BrainWeb is aligned to MNI space, scanned heads are not). ``age`` [years] (a
    number, or (low, high) to draw uniformly): the closed sulci opened (``open_sulci``, on the native
    0.5 mm grid before resampling) to a width drawn from ``sulcal_width`` (mean and between-subject SD
    for that age, its own stream); None leaves them as the model has them. ``deep_grey``: the
    caudate, putamen, pallidum and thalamus as their own classes (``label_deep_grey``), which
    ``table`` must hold (``tissuedb.tissue_table``); they are drawn from their own random stream, so
    the other classes' draws do not change. ``ventricle_age`` [years] (a number, (low, high) to
    draw uniformly, or "age" for the head's ``age``): the lateral ventricles enlarged by ``ventricle_factor`` (Yamada 2023) through
    ``ventricle_displacement``, in MNI space before the deformation. ``pitch`` (mean, SD) [deg]:
    the head tipped about the left-right axis through its centre, nose down positive, by a normal
    draw from its own stream (applied before the yaw); None: as the model lies (MNI, AC-PC level)."""
    if scalp not in ("brainweb", "layered"):
        raise ValueError(f"scalp must be 'brainweb' or 'layered', not {scalp!r}")
    if size not in ("brainweb", "anthropometric", "anthropometric_shell"):
        raise ValueError(f"size must be 'brainweb', 'anthropometric' or 'anthropometric_shell', not {size!r}")
    root = root or os.environ.get("BLOCHPARTY_BRAINWEB")
    if root is None:
        raise ValueError("pass root=, the BrainWeb data directory (validation/anatomy/fetch_brainweb.py)")
    if field_T not in table:
        raise ValueError(f"no tissue table for {field_T} T")
    if isinstance(ventricle_age, str) and (ventricle_age != "age" or age is None):
        raise ValueError('ventricle_age must be years, (low, high), or "age" with age given')
    if deep_grey and (missing := [k for k in DEEP_GREY if k not in table[field_T]]):
        raise ValueError(f"deep_grey needs tissue properties for {missing} (tissuedb.tissue_table)")
    streams = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(int(seed)).spawn(5)]
    if subject is None:
        subject = BRAINWEB_SUBJECTS[int(np.random.default_rng(streams[0]).integers(len(BRAINWEB_SUBJECTS)))]
    age_draw = None
    if age is not None:  # sulci opened on the native 0.5 mm grid, then resampled
        years = float(age) if np.isscalar(age) else float(np.random.default_rng(
            np.random.SeedSequence(int(seed)).spawn(7)[6]).uniform(*age))
        mean, sd = sulcal_width(years)
        width = max(mean + sd * float(np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(11)[10]).standard_normal()), 0.)
        native = load_brainweb(subject, root, dtype=dtype)
        native = Anatomy({k: native.fractions[k] for k in ("gm", "wm", "csf")}, native.voxel, native.origin)
        move = native.fractions["gm"] - open_sulci(native, width).fractions["gm"]  # grey matter -> CSF
        age_draw = {"years": years, "sulcal_width": width, "sulcal_csf": float(move.sum()) * float(np.prod(native.voxel))}
        anatomy = load_brainweb(subject, root, voxel=voxel, dtype=dtype)
        move = torch.minimum(_resample(move.double(), native.voxel, anatomy.voxel).to(dtype), anatomy.fractions["gm"]).clamp(min=0)
        del native
        anatomy = replace(anatomy, fractions=anatomy.fractions | {"gm": anatomy.fractions["gm"] - move,
                                                                  "csf": anatomy.fractions["csf"] + move})
    else:
        anatomy = load_brainweb(subject, root, voxel=voxel, dtype=dtype)
    if deep_grey:
        anatomy = label_deep_grey(anatomy)  # in MNI space, before the deformation
    vent_draw = None
    if ventricle_age is not None:
        vrng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(9)[8])
        if isinstance(ventricle_age, str):  # "age": the head's age (validated above)
            years = age_draw["years"]
        else:
            years = float(ventricle_age) if np.isscalar(ventricle_age) else float(vrng.uniform(*ventricle_age))
        vent_draw = {"years": years, "factor": ventricle_factor(years)}
    displacement, control = random_deformation(anatomy.shape, anatomy.voxel, torch.Generator().manual_seed(streams[1]),
                                               magnitude=magnitude, spacing=spacing)
    size_draw, pose = None, None
    total = lambda d: d
    if size != "brainweb" or yaw_sd or pitch is not None:
        # pull back x -> c + S^-1 R^-1 (x + d(x) - c): deformed, then turned by the yaw and resized about c
        rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(6)[5])
        if size != "brainweb":
            factors, centre, size_draw = anthropometric_scaling(anatomy, rng, shell=size == "anthropometric_shell")
            size_draw["factors"] = list(factors)
        else:
            factors = (1., 1., 1.)
            head = _head_mask(anatomy)
            centre = tuple(float(np.nonzero(head.any(tuple(j for j in range(3) if j != k)))[0][[0, -1]].mean() * h)
                           for k, h in enumerate(anatomy.voxel))
        yaw = math.radians(yaw_sd) * float(rng.standard_normal()) if yaw_sd else 0.
        pose = {"yaw_deg": math.degrees(yaw), "centre": list(centre), "factors": list(factors)}
        tilt = 0.
        if pitch is not None:
            prng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(10)[9])
            tilt = math.radians(pitch[0] + pitch[1] * float(prng.standard_normal()))
            pose["pitch_deg"] = math.degrees(tilt)
        dt = displacement.dtype
        s = torch.tensor(factors, dtype=dt).view(3, 1, 1, 1)
        grid = torch.stack(torch.meshgrid(*(torch.arange(n, dtype=dt) * h for n, h in zip(anatomy.shape, anatomy.voxel)),
                                          indexing="ij"))
        c = torch.tensor(centre, dtype=dt).view(3, 1, 1, 1)
        cos, sin = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(tilt), math.sin(tilt)

        shell = _shell_translation(anatomy, factors, centre).to(dt) if size == "anthropometric_shell" else None

        def total(d):
            v = grid + d - c
            v = torch.stack([v[0], cos * v[1] + sin * v[2], -sin * v[1] + cos * v[2]])  # yaw^-1 in the (y, x) plane
            if tilt:  # pitch^-1 in the (z, y) plane (nose down: z' = z cos - y sin, y' = y cos + z sin)
                v = torch.stack([cp * v[0] + sp * v[1], -sp * v[0] + cp * v[1], v[2]])
            u = v / s
            if shell is not None:  # invert u -> u + T(c + u) by fixed-point steps (|dT/du| <= 1 - s < 1)
                for _ in range(6):
                    u = v - _sample_field(shell, c + u, anatomy.voxel).to(dt)
            return c + u - grid
    if vent_draw is not None and vent_draw["factor"] > 1:  # MNI space, before the deformation and resizing
        mask = ventricle_mask(anatomy)
        vent_draw["offset"], vent_draw["source_volume"] = ventricle_offset(anatomy, vent_draw["factor"], mask)
        anatomy = deform(anatomy, ventricle_displacement(anatomy, mask, vent_draw["offset"]).to(displacement.dtype))
    scale = 1.
    while (jac := deformation_jacobian(total(displacement * scale), anatomy.voxel)).min() <= .2:
        scale /= 2
    tissues, draws = draw_tissues({k: v for k, v in table[field_T].items() if k not in DEEP_GREY},
                                  torch.Generator().manual_seed(streams[2]))
    if deep_grey:
        stream = int(np.random.SeedSequence(int(seed)).spawn(8)[7].generate_state(1)[0])
        more, more_draws = draw_tissues({k: table[field_T][k] for k in DEEP_GREY}, torch.Generator().manual_seed(stream))
        tissues, draws = tissues | more, draws | more_draws
    options = {"seed": streams[4]}
    if texture:
        options["variability"] = dict(texture) | {"seed": streams[3]}
    meta = {"sampler": "anatomy.sample_head", "seed": int(seed), "subject": subject, "field_T": field_T, "scalp": scalp,
            "voxel": list(anatomy.voxel), "streams": streams,
            "deformation": {"magnitude": magnitude, "spacing": spacing, "scale": scale,
                            "jacobian": [float(jac.min()), float(jac.max())], "control": control.tolist()},
            "draws": draws, "options": options}
    if size_draw is not None:
        meta["size"] = size_draw
    if pose is not None:
        meta["pose"] = pose
    if deep_grey:
        meta["deep_grey"] = list(DEEP_GREY)
    if vent_draw is not None:
        meta["ventricles"] = vent_draw
    anatomy = deform(anatomy, total(displacement * scale))
    if scalp == "layered":  # on the final head: the measured layer thicknesses, not resized ones
        anatomy = layer_scalp(anatomy)
    if age_draw is not None:
        meta["age"] = age_draw
    return DigitalPhantom(anatomy, tissues, field_T, options, meta)
