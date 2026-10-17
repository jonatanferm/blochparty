"""Edits of an anatomy the head generator applies: scalp layers, dermis, lesions, deep grey labels, the falx,
ventricles by age, sulci."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from torch import Tensor

from .deformation import deform
from .grid import Anatomy

# Scalp layers at the vertex, 30 healthy adults, 22 MHz ultrasound: epidermis + dermis 1.61 +- 0.23 mm,
# subcutaneous tissue 4.14 +- 1.10 mm, whole scalp 5.76 +- 1.14 mm (Li et al. 2024, Skin Res Technol
# 30:e13863, doi:10.1111/srt.13863, Table 5).
SCALP_LAYERS = (1.61e-3, 4.14e-3)
SCALP_LAYERS_SD = (0.23e-3, 1.10e-3)  # between the 30 subjects (same table)


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


GENERATED = ("dermis", "eye", "lesion")  # classes the head generator adds (add_dermis; prepare_ixi's orbit eye class; add_lesions)


def add_dermis(anatomy: Anatomy, thickness: float = SCALP_LAYERS[0]) -> Anatomy:
    """The skin's dermis as its own class (``dermis``): T2-weighted 0.57 mm fastMRI images show a thin dark line at the
    surface of the scalp, the dermis's collagen-bound water (T2 ~10 ms: ``DEFAULT_TISSUES``), which a single bright
    soft-tissue class lacks. The outer ``thickness`` [m] below the head's boundary (1 - background > 0.5; default the
    scalp's epidermis + dermis, ``SCALP_LAYERS``) takes, in each voxel, its overlap with that layer of every tissue there
    (they keep their proportions, so every voxel's total is conserved); voxels outside the boundary that hold tissue
    are surface, all of it dermis. Depth as in ``layer_scalp``."""
    from scipy import ndimage

    fr = anatomy.fractions
    if "background" not in fr or "dermis" in fr:
        raise ValueError("add_dermis needs a background class and no dermis yet")
    tissue = 1 - fr["background"]
    inside = (tissue > .5).numpy()
    h = float(np.mean(anatomy.voxel))
    depth = ndimage.distance_transform_edt(inside, sampling=anatomy.voxel) - h / 2
    w = np.clip((np.minimum(depth + h / 2, thickness) - np.maximum(depth + h / 2 - h, 0)) / h, 0, 1)
    w = torch.from_numpy(np.where(inside, w, 1.)).to(tissue.dtype)
    out = {k: (v if k == "background" else v * (1 - w)) for k, v in fr.items()}
    out["dermis"] = tissue * w
    return replace(anatomy, fractions=out)


LESION_REGIONS = ("any", "periventricular", "deep", "juxtacortical")


def add_lesions(anatomy: Anatomy, rng, *, count: int = 3, size_mm=(3., 10.), region: str = "any",
                edge_mm: float = 1.0, spacing_mm: float = 10., z_mm=None, z_tolerance_mm: float = 3.) -> tuple[Anatomy, list[dict]]:
    """White-matter lesions as their own class (``lesion``; ROADMAP M7: "a brain with this kind of lesion there"): ``count``
    ellipsoidal blobs with soft edges, each replacing the white matter it covers (every voxel's total is conserved; grey
    matter, CSF and the rest keep their fractions), centred on white matter of ``region``: "periventricular" (within
    10 mm of the lateral ventricles), "deep" (10 mm or more from them and 5 mm or more from the cortex),
    "juxtacortical" (within 3 mm of grey matter) or "any". Sizes: the major semi-axis is drawn log-uniform between
    ``size_mm`` (full lengths, mm), the two others 0.5-1 of it, the orientation uniform; the edge is a logistic ramp of
    ``edge_mm`` (partial volume); centres at least ``spacing_mm`` apart; ``rng`` a numpy Generator. Needs MNI
    coordinates for the ventricle region (a model before ``deform``). Returns the anatomy and one record per lesion
    (centre [MNI mm], semi-axes [mm], rotation, volume [mL] actually placed). ``z_mm``: centres only within
    ``z_tolerance_mm`` of these MNI z levels (lesions at the levels an experiment images). The shapes are synthetic
    (ellipsoids, no mass effect, no rim); the tissue values of ``lesion`` are the measured ones (``DEFAULT_TISSUES``)."""
    from scipy import ndimage

    if region not in LESION_REGIONS:
        raise ValueError(f"region must be one of {LESION_REGIONS}, not {region!r}")
    fr = anatomy.fractions
    if "wm" not in fr:
        raise ValueError("add_lesions needs a wm class")
    voxel = np.asarray(anatomy.voxel, float)
    wm = fr["wm"].numpy()
    candidates = wm > .5
    if region != "any":
        gm = (fr["gm"].numpy() > .5) if "gm" in fr else np.zeros_like(candidates)
        d_gm = ndimage.distance_transform_edt(~gm, sampling=voxel)
        if region == "juxtacortical":
            candidates &= d_gm <= 3e-3
        else:
            vent = ventricle_mask(anatomy)
            d_vent = ndimage.distance_transform_edt(~vent, sampling=voxel)
            candidates &= (d_vent <= 10e-3) if region == "periventricular" else ((d_vent >= 10e-3) & (d_gm >= 5e-3))
    if z_mm is not None:
        zc = (anatomy.origin[2] + np.arange(anatomy.shape[0]) * voxel[0]) * 1e3
        near = np.zeros(anatomy.shape[0], bool)
        for z in np.atleast_1d(z_mm):
            near |= np.abs(zc - float(z)) <= z_tolerance_mm
        candidates &= near[:, None, None]
    idx = np.flatnonzero(candidates)
    if len(idx) == 0:
        raise ValueError(f"no white matter for region {region!r}")
    grid = [np.arange(n) * h for n, h in zip(anatomy.shape, voxel)]  # z, y, x [m] from the array origin
    centres = []
    for _ in range(count):
        for _try in range(200):
            k = int(rng.choice(idx))
            c = np.array(np.unravel_index(k, candidates.shape)) * voxel
            if all(np.linalg.norm(c - o) >= spacing_mm * 1e-3 for o in centres):
                centres.append(c)
                break
        else:
            break  # no room for more at this spacing
    lesion = np.zeros_like(wm)
    records = []
    for c in centres:
        a = np.exp(rng.uniform(np.log(size_mm[0]), np.log(size_mm[1]))) * 1e-3 / 2
        axes = np.array([a, a * rng.uniform(.5, 1.), a * rng.uniform(.5, 1.)])
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        w_, x_, y_, z_ = q  # a uniform random rotation from a unit quaternion
        rot = np.array([[1 - 2 * (y_ ** 2 + z_ ** 2), 2 * (x_ * y_ - z_ * w_), 2 * (x_ * z_ + y_ * w_)],
                        [2 * (x_ * y_ + z_ * w_), 1 - 2 * (x_ ** 2 + z_ ** 2), 2 * (y_ * z_ - x_ * w_)],
                        [2 * (x_ * z_ - y_ * w_), 2 * (y_ * z_ + x_ * w_), 1 - 2 * (x_ ** 2 + y_ ** 2)]])
        reach = axes.max() + 3 * edge_mm * 1e-3
        lo = np.maximum(np.floor((c - reach) / voxel).astype(int), 0)
        hi = np.minimum(np.ceil((c + reach) / voxel).astype(int) + 1, np.array(candidates.shape))
        sub = tuple(slice(l, h) for l, h in zip(lo, hi))
        zz, yy, xx = np.meshgrid(*(g[s_] - cc for g, s_, cc in zip(grid, sub, c)), indexing="ij")
        local = rot.T @ np.stack([zz.ravel(), yy.ravel(), xx.ravel()])
        rho = np.sqrt(((local / axes[:, None]) ** 2).sum(0)).reshape(zz.shape)
        # signed distance from the surface along the radius, in m, then a logistic edge
        dist = (1 - rho) * axes.min()
        w = 1 / (1 + np.exp(-dist / (edge_mm * 1e-3 / 4)))
        w = np.maximum(w, lesion[sub])
        lesion[sub] = w
        records.append({"centre_mm": ((c + np.array(anatomy.origin[::-1])) * 1e3)[::-1].round(1).tolist(),
                        "semi_axes_mm": (axes * 1e3).round(2).tolist(), "rotation": rot.round(4).tolist()})
    taken = wm * lesion
    out = {k: (v - torch.from_numpy(taken).to(v.dtype) if k == "wm" else v) for k, v in fr.items()}
    out["lesion"] = torch.from_numpy(taken).to(fr["wm"].dtype) + (out["lesion"] if "lesion" in out else 0)
    placed = replace(anatomy, fractions=out)
    for r in records:  # the volume each lesion took from white matter, at the placed (soft) fractions
        c = (np.array(r["centre_mm"][::-1]) * 1e-3 - np.array(anatomy.origin[::-1]))
        reach = max(r["semi_axes_mm"]) * 1e-3 + 3 * edge_mm * 1e-3
        lo = np.maximum(np.floor((c - reach) / voxel).astype(int), 0)
        hi = np.minimum(np.ceil((c + reach) / voxel).astype(int) + 1, np.array(candidates.shape))
        sub = tuple(slice(l, h) for l, h in zip(lo, hi))
        r["volume_ml"] = round(float(taken[sub].sum() * np.prod(voxel) * 1e6), 3)
    return placed, records


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


# The falx cerebri, the dural sheet in the interhemispheric fissure: 0.91 +- 0.05 (SE) mm thick (Walsh et al. 2021,
# Acta Biomater 134:388, doi:10.1016/j.actbio.2021.07.043); its free edge above the corpus callosum by 0.6-14 mm with
# the falx's type and position (front / middle / back, 52 cadavers: type I 14 / 12 / 2 mm, II 6 / 7 / 1, III 2 / 2 / 0.6;
# Kayalioglu et al. 2004, Neurosciences 9:257, doi:10.17712/1658-3183.1287): 6 mm, type II.
FALX_THICKNESS = 0.91e-3
FALX_ABOVE_CALLOSUM = 6e-3


def add_falx(anatomy: Anatomy, thickness: float = FALX_THICKNESS, above_callosum: float = FALX_ABOVE_CALLOSUM,
             half_width: float = 8e-3) -> Anatomy:
    """The falx cerebri as dura in the interhemispheric fissure. BrainWeb's and IXI-segmented heads have none (dura
    0 at the midline), so their fissure is solid CSF: simulated T2w, its peak 4.7 x white matter against fastMRI's
    3.9. On an MNI-frame anatomy the subject's midsagittal plane x = a + b y + c z is fitted to the CSF-weighted
    centres of the rows (z, y) whose CSF within ``half_width`` of x = 0 lies between grey matter on both sides (and
    refitted within half that of the first plane); then per row a sheet ``thickness`` thick at the CSF-weighted centre
    within 2.5 mm of the plane turns CSF into dura (no more than the CSF present: CSF and dura together are
    conserved), where grey matter (cortex) lies on both sides within 6 mm, the row holds at least a sheet's width of
    CSF there, and no corpus callosum (white matter bridging the plane: at it and 3 mm to either side) lies above, or
    within ``above_callosum`` below: so not in the third or lateral ventricles, nor between the cingulate gyri under
    the free edge."""
    fr = anatomy.fractions
    if not {"csf", "gm", "wm", "dura"} <= set(fr):
        raise ValueError("add_falx needs the csf, gm, wm and dura classes (BrainWeb, IXI)")
    vz, vy, vx = anatomy.voxel
    nz, ny, nx = anatomy.shape
    x = anatomy.origin[0] + np.arange(nx) * vx
    yy, zz = np.meshgrid(anatomy.origin[1] + np.arange(ny) * vy, anatomy.origin[2] + np.arange(nz) * vz)  # (z, y)
    csf_all, gm_all, wm_all = (fr[k].numpy() for k in ("csf", "gm", "wm"))

    def rows(centre, width):
        """per row: CSF amount and centre within ``width`` of ``centre`` (z, y), cortex on both sides"""
        off = x[None, None, :] - centre[..., None]
        win = np.abs(off) <= width
        csf = csf_all * win
        amount = csf.sum(-1)
        cx = (csf * off).sum(-1) / np.maximum(amount, 1e-9)
        d = off - cx[..., None]
        cortex = (((gm_all * ((d < -vx / 2) & (d >= -6e-3))).max(-1) >= .3)
                  & ((gm_all * ((d > vx / 2) & (d <= 6e-3))).max(-1) >= .3))
        return amount, centre + cx, cortex

    plane = np.zeros((nz, ny))
    for width in (half_width, half_width / 2):
        amount, cx, cortex = rows(plane, width)
        fit = cortex & (amount * vx >= thickness)
        if fit.sum() >= 3:
            A = np.stack([np.ones(fit.sum()), yy[fit], zz[fit]], 1)
            coef = np.linalg.lstsq(A * amount[fit, None], cx[fit] * amount[fit], rcond=None)[0]
            plane = coef[0] + coef[1] * yy + coef[2] * zz
    amount, cx, cortex = rows(plane, 2.5e-3)

    def wm_at(shift):  # white matter at the voxel nearest the plane + shift, per row
        i = np.clip(np.round((plane + shift - x[0]) / vx).astype(int), 0, nx - 1)
        return np.take_along_axis(wm_all, i[..., None], -1)[..., 0]
    callosum = (wm_at(0.) > .5) & (wm_at(-3e-3) > .5) & (wm_at(3e-3) > .5)  # (z, y)
    above = np.flip(np.cumsum(np.flip(callosum, 0), 0), 0) - callosum > 0  # callosum somewhere higher
    below = np.zeros_like(callosum)
    for k in range(1, int(round(above_callosum / vz)) + 1):
        below[k:] |= callosum[:-k]
    ok = cortex & ~above & ~below & (amount * vx >= thickness)
    lo, hi = cx - thickness / 2, cx + thickness / 2
    cover = np.clip(np.minimum(hi[..., None], x + vx / 2) - np.maximum(lo[..., None], x - vx / 2), 0, None) / vx
    move = torch.from_numpy(np.minimum(cover * ok[..., None], csf_all).astype(np.float32)).to(fr["csf"].dtype)
    return replace(anatomy, fractions=fr | {"csf": fr["csf"] - move, "dura": fr["dura"] + move})


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


def sulcal_widening(age: float) -> float:
    """Sulcal widening [m] beyond ``BRAINWEB_AGE`` at ``age`` [years]: the slope of ``SULCAL_WIDTH_AGE``
    (0.011 mm per year), none below."""
    (a0, w0, _), (a1, w1, _) = SULCAL_WIDTH_AGE
    return max(float(age) - BRAINWEB_AGE, 0.) * (w1 - w0) / (a1 - a0)


def gather_cortical_csf(anatomy: Anatomy) -> Anatomy:
    """Grey matter and CSF sharing a voxel outside the deep grey regions (``DEEP_GREY`` scaled by 1.3)
    go to the larger share. BrainWeb's fuzzy models spread sulcal CSF thinly through the cortex (3.9 %
    of subject 04's grey-dominant voxels, 38 mL at 0.5 mm), which lifts T2-weighted grey matter by
    ~9 %; real cortex keeps its CSF in sulci and the subarachnoid space. Other classes are untouched;
    needs MNI coordinates (a BrainWeb model before ``deform``)."""
    fr = anatomy.fractions
    gm, csf = fr["gm"], fr["csf"]
    deep = np.zeros(tuple(gm.shape), bool)
    for r2 in _deep_grey_r2(anatomy.origin, tuple(gm.shape), anatomy.voxel):
        deep |= r2 < 1.3 ** 2
    mix = (gm > 0) & (csf > 0) & ~torch.from_numpy(deep)
    to_csf = mix & (csf >= gm)
    both, zero = gm + csf, torch.zeros_like(gm)
    return replace(anatomy, fractions=fr | {"gm": torch.where(mix, torch.where(to_csf, zero, both), gm),
                                            "csf": torch.where(mix, torch.where(to_csf, both, zero), csf)})


def _deep_grey_r2(origin, shape, voxel):
    """Each ``DEEP_GREY`` ellipsoid's normalised squared radius on a grid (origin x, y, z; shape and voxel z, y, x; MNI)."""
    x, y, z = ((o + np.arange(n) * h) * 1e3 for o, n, h in zip(origin, shape[::-1], voxel[::-1]))
    for regions in DEEP_GREY.values():
        for (cx, cy, cz), (ax, ay, az) in regions:
            yield ((((z - cz) / az) ** 2)[:, None, None] + (((y - cy) / ay) ** 2)[None, :, None]
                   + (((np.abs(x) - cx) / ax) ** 2)[None, None, :])


def open_sulci(anatomy: Anatomy, width: float | None = None, separation: float = 3e-3, opposed: float = 0.5, *,
               volume: float | None = None, widen: float = 0.) -> Anatomy:
    """Open closed sulci to ``width`` [m] of CSF, or to the width that adds ``volume`` [m^3] of CSF
    (solved by bisection), plus ``widen`` [m]. BrainWeb's fuzzy models (1.5 T scans at 1 mm) give
    narrow sulci to grey matter: two cortical banks touch with no CSF between them, and T2-weighted
    slices show speckle where real ones show sulcal lines. A closed sulcus is grey matter reached from
    two white-matter banks: the lambda-medial axis of the region outside white matter (Chazal and
    Lieutier 2005), i.e. neighbouring voxels whose nearest white-matter points lie >= ``separation``
    apart, in directions facing each other (cosine below -``opposed``: within 60 deg of opposite, so
    a shallow groove in the white surface is not a sulcus). Bumps of the white surface smaller than
    that make no axis (ridges of the distance to white matter do: an earlier version opened those and
    made specks). The cerebellum's fissures get the same width (no measurement there). Kept: grey
    matter > 0.2, outside the deep grey regions (``DEEP_GREY`` scaled by 1.3), in sheets touching the
    CSF around the brain. Each voxel near a sheet takes its distance to the midplane as half the
    difference of its distances to the two banks; CSF fills width / 2 either side, voxel-weighted,
    never below what is there, taken from grey matter. Needs MNI coordinates (a BrainWeb model before
    ``deform``) and a fine grid: ``sample_head`` opens at the native 0.5 mm and then resamples."""
    from scipy import ndimage

    if (width is None) == (volume is None):
        raise ValueError("give width or volume")
    if (width if width is not None else volume) <= 0 and widen <= 0:
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
    reach = (width if width is not None else 4e-3) + widen
    dist, I = ndimage.distance_transform_edt(~sheet, sampling=vox, return_indices=True)
    near = dist <= reach / 2 + h
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
    gm_n, csf_n, keep = gm[near], csf[near], ~deep[near]
    cell = float(np.prod(vox))

    def fill(w):
        return np.clip(np.minimum(np.clip((w / 2 - r) / h + .5, 0, 1) - csf_n, gm_n), 0, None) * keep

    if width is None:  # the width whose fill adds `volume`
        lo, hi = 0., 4e-3
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if fill(mid).sum() * cell < volume else (lo, mid)
        width = (lo + hi) / 2
    move = np.zeros(gm.shape)
    move[near] = fill(width + widen)
    full = torch.zeros_like(fr["gm"])
    full[box] = torch.from_numpy(move).to(full.dtype)
    return replace(anatomy, fractions=fr | {"gm": fr["gm"] - full, "csf": fr["csf"] + full})
