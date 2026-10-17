"""The head sampler: ``DigitalPhantom``, drawn tissues, anthropometric scaling, and
``sample_head``."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..phantom import Phantom
from .brainweb import load_brainweb
from .deformation import deform, deformation_jacobian, random_deformation
from .edits import (
    DEEP_GREY,
    GENERATED,
    SCALP_LAYERS,
    SCALP_LAYERS_SD,
    add_dermis,
    add_falx,
    add_lesions,
    gather_cortical_csf,
    label_deep_grey,
    layer_scalp,
    open_sulci,
    sulcal_widening,
    sulcal_width,
    ventricle_displacement,
    ventricle_factor,
    ventricle_mask,
    ventricle_offset,
)
from .grid import Anatomy, Distribution, TissueProperties, _resample
from .ixi import ixi_subjects, load_ixi
from .phantom import anatomy_field_maps, to_phantom
from .tables import DEFAULT_TISSUES


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
                age: float | tuple[float, float] | None = None, deep_grey: bool = False, falx: bool = False,
                ventricle_age: float | tuple[float, float] | None = None,
                pitch: tuple[float, float] | None = None, sulci: str = "conserved",
                source: str = "brainweb", interpolation: int = 1, residual: bool = False,
                dermis: bool = False, ixi_prepared=None, lesions: dict | None = None) -> DigitalPhantom:
    """A new head for every seed (ROADMAP M5, P1): a BrainWeb subject (``subject``, or one of the
    20 drawn by the seed), deformed by ``random_deformation`` (``magnitude``, ``spacing``; halved
    until the Jacobian exceeds 0.2 everywhere), one subject's tissue table (``draw_tissues``),
    within-tissue texture (``texture``, ``to_phantom(variability=...)``) and its own spin seed.
    Susceptibility -> B0, shim and the R2* policy are ``to_phantom``'s defaults. The result is
    deterministic in the seed, and ``meta`` records every draw (subject, control points,
    properties, seeds). ``root`` is the BrainWeb data directory (default: the environment
    variable BLOCHPARTY_BRAINWEB; validation/anatomy/fetch_brainweb.py fetches it). ``scalp``:
    "brainweb" keeps BrainWeb's skin class, "layered" splits it into skin and subcutaneous fat
    ("layered_drawn": this head's layer thicknesses drawn with ``SCALP_LAYERS_SD``, its own stream)
    (``layer_scalp``) after the deformation. ``size``: "brainweb" keeps the model's (MNI-template)
    size, "anthropometric" scales the head to a breadth and length drawn from ANSUR II
    (``anthropometric_scaling``), composed with the deformation into one resampling;
    "anthropometric_shell" reaches the same size by scaling the skull's contents and keeping the
    skull's and scalp's thickness. ``yaw_sd``
    [deg]: the head turned about the scanner's z axis through its centre by N(0, yaw_sd^2) (0: as
    the model lies; BrainWeb is aligned to MNI space, scanned heads are not). ``age`` [years] (a
    number, or (low, high) to draw uniformly): the closed sulci opened (``open_sulci``, on the native
    0.5 mm grid before resampling); None leaves them as the model has them. ``sulci``: "conserved"
    gathers the cortex's diffuse CSF (``gather_cortical_csf``) into the sheets at the width that keeps
    the CSF volume, widened by ``sulcal_widening`` for the age; "measured" opens them to a width drawn
    from ``sulcal_width`` (Jin 2018's major sulci; their own stream), which on fastMRI 3 T T2w adds
    more bright CSF than the real images show. ``deep_grey``: the
    caudate, putamen, pallidum and thalamus as their own classes (``label_deep_grey``), which
    ``table`` must hold (``tissuedb.tissue_table``); they are drawn from their own random stream, so
    the other classes' draws do not change. ``falx``: the falx cerebri as dura in the interhemispheric
    fissure (``add_falx``, in MNI space before the deformation). ``interpolation``: the order of the final pull-back
    (``deform``: 1 trilinear, 3 cubic B-spline, which keeps more fine structure). ``residual`` (``source="ixi"``): the
    subject's fitted T2 / PD multipliers (``load_ixi``), carried through the pose. ``ventricle_age`` [years] (a number, (low, high) to
    draw uniformly, or "age" for the head's ``age``): the lateral ventricles enlarged by ``ventricle_factor`` (Yamada 2023) through
    ``ventricle_displacement``, in MNI space before the deformation. ``pitch`` (mean, SD) [deg]:
    the head tipped about the left-right axis through its centre, nose down positive, by a normal
    draw from its own stream (applied before the yaw); None: as the model lies (MNI, AC-PC level). ``lesions``: ``add_lesions`` keywords (count, size_mm, region, ...) for white-matter lesions placed in MNI
    space before the deformation, from their own stream (the record of each in ``meta["lesions"]``). ``dermis``: the
    skin's dermis as its own class (``add_dermis``, on the final head; with "layered_drawn", this head's dermis
    thickness), which ``table`` must hold. Classes the generator adds (``GENERATED``) draw from their own stream, so
    adding them to a table leaves the other classes' draws unchanged."""
    if source not in ("brainweb", "ixi"):
        raise ValueError(f"source must be 'brainweb' or 'ixi', not {source!r}")
    if source == "ixi" and (age is not None or ventricle_age is not None):
        raise ValueError("IXI heads have their own sulci, ventricles and age: no age= or ventricle_age=")
    if size == "native" and source != "ixi":
        raise ValueError("size='native' needs source='ixi' (a subject with its own size)")
    if scalp not in ("brainweb", "layered", "layered_drawn"):
        raise ValueError(f"scalp must be 'brainweb', 'layered' or 'layered_drawn', not {scalp!r}")
    if size not in ("brainweb", "anthropometric", "anthropometric_shell", "native"):
        raise ValueError(f"size must be 'brainweb', 'anthropometric', 'anthropometric_shell' or 'native', not {size!r}")
    root = root or os.environ.get("BLOCHPARTY_BRAINWEB")
    if root is None:
        raise ValueError("pass root=, the BrainWeb data directory (validation/anatomy/fetch_brainweb.py)")
    if field_T not in table:
        raise ValueError(f"no tissue table for {field_T} T")
    if sulci not in ("conserved", "measured"):
        raise ValueError(f"sulci must be 'conserved' or 'measured', not {sulci!r}")
    if isinstance(ventricle_age, str) and (ventricle_age != "age" or age is None):
        raise ValueError('ventricle_age must be years, (low, high), or "age" with age given')
    if deep_grey and (missing := [k for k in DEEP_GREY if k not in table[field_T]]):
        raise ValueError(f"deep_grey needs tissue properties for {missing} (tissuedb.tissue_table)")
    streams = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(int(seed)).spawn(5)]
    ixi_meta = None
    if source == "ixi":
        ixi_root = Path(root) / "ixi"
        if subject is None:
            pool = ixi_subjects(ixi_root, ixi_prepared)
            subject = pool[int(np.random.default_rng(streams[0]).integers(len(pool)))]
        subject = int(str(subject).removeprefix("IXI"))
    elif subject is None:
        subject = BRAINWEB_SUBJECTS[int(np.random.default_rng(streams[0]).integers(len(BRAINWEB_SUBJECTS)))]
    age_draw = None
    if source == "ixi":
        anatomy, ixi_meta = load_ixi(subject, ixi_root, voxel=voxel, dtype=dtype, residual=residual, prepared=ixi_prepared)
    elif age is not None:  # sulci opened on the native 0.5 mm grid, then resampled
        years = float(age) if np.isscalar(age) else float(np.random.default_rng(
            np.random.SeedSequence(int(seed)).spawn(7)[6]).uniform(*age))
        native = load_brainweb(subject, root, dtype=dtype)
        native = Anatomy({k: native.fractions[k] for k in ("gm", "wm", "csf")}, native.voxel, native.origin)
        cell = float(np.prod(native.voxel))
        if sulci == "measured":
            mean, sd = sulcal_width(years)
            width = max(mean + sd * float(np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(11)[10]).standard_normal()), 0.)
            opened = open_sulci(native, width)
            age_draw = {"years": years, "sulci": sulci, "sulcal_width": width}
        else:  # conserved: the cortex's diffuse CSF gathered into the sheets, plus the widening with age
            gathered = gather_cortical_csf(native)
            volume = float(native.fractions["csf"].sum() - gathered.fractions["csf"].sum()) * cell
            opened = open_sulci(gathered, volume=max(volume, 0.), widen=sulcal_widening(years))
            age_draw = {"years": years, "sulci": sulci, "gathered_csf": volume, "widen": sulcal_widening(years)}
        move = native.fractions["gm"] - opened.fractions["gm"]  # grey matter -> CSF (signed)
        del opened
        age_draw["sulcal_csf"] = float(move.sum()) * cell
        anatomy = load_brainweb(subject, root, voxel=voxel, dtype=dtype)
        move = _resample(move.double(), native.voxel, anatomy.voxel).to(dtype)
        move = torch.maximum(torch.minimum(move, anatomy.fractions["gm"]), -anatomy.fractions["csf"])
        del native
        anatomy = replace(anatomy, fractions=anatomy.fractions | {"gm": anatomy.fractions["gm"] - move,
                                                                  "csf": anatomy.fractions["csf"] + move})
    else:
        anatomy = load_brainweb(subject, root, voxel=voxel, dtype=dtype)
    if deep_grey:
        anatomy = label_deep_grey(anatomy)  # in MNI space, before the deformation
    if falx:
        anatomy = add_falx(anatomy)  # in MNI space (x = 0 is the midline), before the deformation
    lesion_records = None
    if lesions:  # white-matter lesions by region (add_lesions), in MNI space before the deformation, their own stream
        lrng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(16)[15])
        anatomy, lesion_records = add_lesions(anatomy, lrng, **lesions)
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
        if size == "native":  # the subject's own size: the scale of its registration (MNI -> its world)
            lin = np.array(ixi_meta["affine_mm"])[:3, :3]
            factors = tuple(float(np.linalg.norm(lin[:, a])) for a in (2, 1, 0))  # (z, y, x)
            head = _head_mask(anatomy)
            centre = tuple(float(np.nonzero(head.any(tuple(j for j in range(3) if j != k)))[0][[0, -1]].mean() * h)
                           for k, h in enumerate(anatomy.voxel))
            size_draw = {"native": True, "factors": list(factors)}
        elif size != "brainweb":
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
    tissues, draws = draw_tissues({k: v for k, v in table[field_T].items() if k not in DEEP_GREY and k not in GENERATED},
                                  torch.Generator().manual_seed(streams[2]))
    generated = {k: table[field_T][k] for k in GENERATED if k in table[field_T]}
    if generated:
        stream = int(np.random.SeedSequence(int(seed)).spawn(14)[13].generate_state(1)[0])
        more, more_draws = draw_tissues(generated, torch.Generator().manual_seed(stream))
        tissues, draws = tissues | more, draws | more_draws
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
    if lesion_records is not None:
        meta["lesions"] = {"spec": dict(lesions), "placed": lesion_records}
    field = total(displacement * scale)
    # where MNI's lateral-ventricle bodies (x 0, y -15, z +20 mm) landed: the output voxel whose source point is
    # nearest (a prior for the slice-stack landmark; the models are MNI-registered, the pose moves them)
    src = torch.stack(torch.meshgrid(*(torch.arange(n, dtype=field.dtype) * h for n, h in zip(anatomy.shape, anatomy.voxel)),
                                     indexing="ij")) + field
    mni = torch.tensor([20e-3 - anatomy.origin[2], -15e-3 - anatomy.origin[1], 0. - anatomy.origin[0]], dtype=field.dtype)
    k = int(((src - mni.view(3, 1, 1, 1)) ** 2).sum(0).argmin())
    meta["ventricle_landmark"] = [float(anatomy.origin[2] + (k // (anatomy.shape[1] * anatomy.shape[2])) * anatomy.voxel[0])]
    del src
    anatomy = deform(anatomy, field, order=interpolation)
    if scalp == "layered":  # on the final head: the measured layer thicknesses, not resized ones
        anatomy = layer_scalp(anatomy)
    elif scalp == "layered_drawn":  # this head's thicknesses: normal draws with the measured between-subject SDs
        z = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(12)[11]).standard_normal(2)
        layers = tuple(max(m + sd * float(v), 0.25 * m) for m, sd, v in zip(SCALP_LAYERS, SCALP_LAYERS_SD, z))
        meta["scalp_layers"] = list(layers)
        anatomy = layer_scalp(anatomy, *layers)
    if dermis:
        anatomy = add_dermis(anatomy, meta["scalp_layers"][0] if "scalp_layers" in meta else SCALP_LAYERS[0])
        meta["dermis"] = True
    if age_draw is not None:
        meta["age"] = age_draw
    if ixi_meta is not None:
        meta["source"] = "ixi"
        meta["ixi"] = {k: ixi_meta[k] for k in ("ixi_id", "age", "sex", "lncc", "volumes_ml")}
    return DigitalPhantom(anatomy, tissues, field_T, options, meta)
