"""The realism preset's heads: one per seed (``head_for``), cleaned (detached tissue, the vessels' flow void), the
slice stack placed on its ventricles (``stack``), the field of view centred on it (``centre``), its B1+ from a
full-wave body-coil map (``b1_plus``), turned so anterior-posterior is the readout axis (``turn``)."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

import blochparty as bp
from ..anatomy import Distribution
from .protocols import SLICES


PROTOCOL_SPACING = 5e-3  # contiguous 5 mm slices


MONRO_BELOW_BODIES = 10e-3  # the foramen of Monro / basal-ganglia level below the ventricle bodies


def stack(anatomy, mode: str = "landmarks", offset: float = MONRO_BELOW_BODIES, prior_z: float | None = None) -> np.ndarray:
    """z [m] of slices 0..15. "landmarks": slice 0 at the largest lateral-ventricle cross-section, the
    brain top at slice 13.5. "protocol": slice 0 ``MONRO_BELOW_BODIES`` below that (the real slice 0
    shows frontal horns, caudate heads, thalami and the third ventricle), then ``PROTOCOL_SPACING``
    apart whatever the head's size; fitted to the real stacks' head-area profile (area / slice-0 area
    1.0 to slice 7, 0.53 at 12, 0.23 at 15; the fit alone trades offset against spacing)."""
    csf = anatomy.fractions["csf"].numpy()
    brain = sum(v for k, v in anatomy.fractions.items() if k in ("wm", "gm", *bp.anatomy.DEEP_GREY)).numpy()
    z = anatomy.origin[2] + np.arange(anatomy.shape[0]) * anatomy.voxel[0]
    inside = ndimage.binary_erosion((brain + csf) > 0.5, iterations=round(15e-3 / anatomy.voxel[1]))
    deep = (csf * inside).reshape(len(z), -1).sum(1)
    if prior_z is not None:  # within 12 mm of where MNI's ventricle bodies landed (segmented heads such as IXI's
        # keep wide fissure and sulcal CSF inside the eroded brain near the vertex: it took one head's landmark 25 mm up)
        deep = np.where(np.abs(z - prior_z) <= 12e-3, deep, 0)
    top = z[np.nonzero(brain.reshape(len(z), -1).sum(1) * np.prod(anatomy.voxel[1:]) > 50e-6)[0].max()]
    z0 = z[np.argmax(deep)]
    if mode == "protocol":
        return z0 - offset + np.arange(SLICES) * PROTOCOL_SPACING
    return z0 + np.arange(SLICES) * (top - z0) / 13.5


def flow_void(head) -> None:
    """BrainWeb's vessel class holds the dural sinuses and large vessels; their blood flows at
    ~10-30 cm/s, and through a 5 mm slice a TSE echo at 113 ms keeps < 3 % of static blood's signal
    from 10 cm/s up (the engine's moving spins, tests/physics/test_motion.py::test_tse_flow_void). Their
    spins are left out (PD 0 after the draw, so no other draw moves); susceptibility stays."""
    p = head.tissues["vessels"]
    head.tissues["vessels"] = replace(p, pd=Distribution(0., None, "TSE flow void (simulate_heads.flow_void): " + p.pd.citation, True))


def drop_detached(head) -> None:
    """Tissue not attached to the head becomes air: BrainWeb subject 38 has a 0.4 mL blob of "skin"
    20 mm in front of the forehead (a marker in its source scan), subject 42 hundreds of faint
    "skin" pieces (fractions 0.05-0.5) around the neck and head, and several have voxel specks; in
    the simulated slices they showed as dashes and lines outside the head. Everything more than 3
    voxels from the head mask's largest component (tissue > 0.5) is dropped, whatever its fraction,
    and so are thin protrusions on it (more than 2 voxels outside its opening by ~3 mm)."""
    an = head.anatomy
    fr = an.fractions
    total = sum(v for k, v in fr.items() if k != "background").numpy()
    lab, n = ndimage.label(total > .5)
    if n == 0:
        return
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    head_mask = lab == np.argmax(sizes)
    # and thin protrusions on its surface (subject 18 has a marker blob attached to the right side):
    # what an opening by ~3 mm removes, beyond 2 voxels of the opened head
    opened = ndimage.binary_opening(head_mask, structure=np.ones((3, 3, 3)), iterations=2)
    keep = ndimage.binary_dilation(opened, iterations=2) & ndimage.binary_dilation(head_mask, iterations=3)
    drop = torch.from_numpy(~keep & (total > 0))
    if not bool(drop.any()):
        return
    moved = sum(torch.where(drop, v, torch.zeros_like(v)) for k, v in fr.items() if k != "background")
    out = {k: torch.where(drop, torch.zeros_like(v), v) for k, v in fr.items() if k != "background"}
    if "background" in fr:
        out["background"] = torch.where(drop, (fr["background"] + moved).clamp(max=1), fr["background"])
    head.anatomy = replace(an, fractions=out)


def trace_sulci(head, smooth=0.75e-3, ratio=0.8, curvature=0.0006, min_voxels=30, width=1e-3, fill=0.7) -> float:
    """Experimental: open BrainWeb's closed sulci where its own faint CSF trace runs (its fuzzy CSF is
    0.02-0.15 along their midlines, LESSONS). Hessian of the CSF fraction smoothed by ``smooth`` [m]
    inside grey matter; sheet-like where the most negative eigenvalue (< -``curvature``) dominates the
    other two by 1 / ``ratio``; thinned by non-maximum suppression along its eigenvector; connected
    sheets of >= ``min_voxels``; widened to ``width`` [m] and filled to ``fill`` CSF from grey matter.
    Returns the CSF added [m^3]."""
    an = head.anatomy
    fr = an.fractions
    h = float(np.mean(an.voxel))
    csf, gm = fr["csf"].numpy().astype(np.float32), fr["gm"].numpy().astype(np.float32)
    cs = ndimage.gaussian_filter(csf, smooth / h)
    g = np.gradient(cs)
    region = (gm > .5) & ~ndimage.binary_dilation(csf > .4, iterations=1)
    idx = np.nonzero(region)
    H = np.empty((len(idx[0]), 3, 3), np.float32)
    for i in range(3):
        gi = np.gradient(g[i])
        for j in range(i, 3):
            H[:, i, j] = H[:, j, i] = gi[j][idx]
    del g, gi
    ev, evec = np.linalg.eigh(H)
    del H
    l1 = ev[:, 0]
    sheet = (l1 < -curvature) & (np.abs(ev[:, 1]) < ratio * np.abs(l1)) & (np.abs(ev[:, 2]) < ratio * np.abs(l1))
    z, y, x = (a[sheet] for a in idx)
    nz, ny, nx = (evec[sheet, k, 0] for k in range(3))
    c0 = cs[z, y, x]
    up = ndimage.map_coordinates(cs, [z + nz, y + ny, x + nx], order=1)
    down = ndimage.map_coordinates(cs, [z - nz, y - ny, x - nx], order=1)
    keep = (c0 >= up) & (c0 >= down)
    thin = np.zeros(cs.shape, bool)
    thin[z[keep], y[keep], x[keep]] = True
    lab, _ = ndimage.label(thin, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    thin = np.isin(lab, np.nonzero(sizes >= min_voxels)[0])
    band = ndimage.distance_transform_edt(~thin, sampling=an.voxel) <= width / 2
    move = torch.from_numpy(np.minimum(gm, fill) * band).to(fr["gm"].dtype)
    head.anatomy = replace(an, fractions=fr | {"gm": fr["gm"] - move, "csf": fr["csf"] + move})
    return float(move.sum()) * float(np.prod(an.voxel))


def thin_convexity_csf(head, thickness: float) -> None:
    """A calibration to the fastMRI scans, not measured anatomy: CSF within ``thickness`` [m] of the
    calvarium (skull and marrow) and outside the brain's hull (closed by 3 mm: the subarachnoid layer, not the
    sulci) becomes inner table. Real profiles aligned at the bone (frontal: 0.16 at the bone, then
    0.28-0.44) show a thinner, dimmer layer under the calvarium than BrainWeb's (0.29, then up to
    0.64), and 0.57 mm reconstructions a thin line at most; 1-2 mm brackets them."""
    an = head.anatomy
    fr = an.fractions
    brain = sum(v for k, v in fr.items() if k in ("gm", "wm", *bp.anatomy.DEEP_GREY)).numpy() > .5
    hull = ndimage.binary_fill_holes(ndimage.binary_closing(brain, iterations=round(3e-3 / an.voxel[1])))
    bone = (fr["skull"] + fr["marrow"]).numpy() > .5  # the calvarium, not the falx and tentorium
    d = ndimage.distance_transform_edt(~bone, sampling=an.voxel)
    w = torch.from_numpy(np.clip((thickness + an.voxel[1] / 2 - d) / an.voxel[1], 0, 1) * ~hull).to(fr["csf"].dtype)
    move = fr["csf"] * w
    head.anatomy = replace(an, fractions=fr | {"csf": fr["csf"] - move, "skull": fr["skull"] + move})


def centre(anatomy, zs) -> tuple[float, float]:
    """(x, y) [m] of the head's bounding-box centre over the slice stack's z range: where an
    operator puts the field of view on the localiser (and the head sits in the coil).
    ``anatomy.origin`` is (x, y, z), ``voxel`` (z, y, x)."""
    head = (sum(v for k, v in anatomy.fractions.items() if k != "background") > 0.5).numpy()
    z = anatomy.origin[2] + np.arange(anatomy.shape[0]) * anatomy.voxel[0]
    rows = head[(z >= zs[0] - 2.5e-3) & (z <= zs[-1] + 2.5e-3)]
    ys, xs = np.nonzero(rows.any((0, 2)))[0], np.nonzero(rows.any((0, 1)))[0]
    return (anatomy.origin[0] + (xs[0] + xs[-1]) / 2 * anatomy.voxel[2],
            anatomy.origin[1] + (ys[0] + ys[-1]) / 2 * anatomy.voxel[1])


_B1 = {}


def b1_plus(head, pos: torch.Tensor, root) -> torch.Tensor:
    """Relative B1+ at spin positions [m, anatomy frame] of a generated head: the full-wave 3 T body
    birdcage map of BrainWeb subject 04 (validation/em/run_brainweb.py; mean 1 over its brain, 0.86-1.19
    p5-p95), sampled where each spin came from before ``sample_head`` resized and turned the head
    (BrainWeb's heads share MNI space). Ignores the small random deformation and the weaker dielectric
    effect of smaller heads. ``root``: the anatomy data (BrainWeb subject 04 and ``em/body.npz``)."""
    from blochparty import em
    from blochparty.maps import sample

    key = str(Path(root).resolve())
    if key not in _B1:
        fm = em.field_maps(transmit=em.read_mariepy_maps(Path(root) / "em" / "body.npz"))
        ref = bp.load_brainweb(4, root, voxel=(5e-3,) * 3, dtype=torch.float64)
        brain = (ref.fractions["gm"] + ref.fractions["wm"]) > 0.5
        p = torch.tensor(ref.origin, dtype=torch.float64) + brain.nonzero().double().flip(-1) * torch.tensor(ref.voxel[::-1], dtype=torch.float64)
        grid = fm.b1[None].double(), torch.tensor(fm.origin, dtype=torch.float64), torch.tensor(fm.spacing, dtype=torch.float64)
        _B1[key] = grid, float(sample(*grid, p)[:, 0].mean())
    pose = head.meta.get("pose")
    q = pos.double()
    if pose is not None:  # back to the source head: x -> c + S^-1 R^-1 (x - c), (z, y, x) grid-relative
        an = head.anatomy
        o = torch.tensor(an.origin, dtype=torch.float64)  # (x, y, z)
        c = torch.tensor(pose["centre"][::-1], dtype=torch.float64) + o  # (x, y, z)
        s = torch.tensor(pose["factors"][::-1], dtype=torch.float64)
        yaw = math.radians(pose["yaw_deg"])
        v = q - c
        v = torch.stack([math.cos(yaw) * v[:, 0] - math.sin(yaw) * v[:, 1], math.sin(yaw) * v[:, 0] + math.cos(yaw) * v[:, 1],
                         v[:, 2]], -1)
        tilt = math.radians(pose.get("pitch_deg", 0.))
        if tilt:  # pitch^-1 in the (y, z) plane, as sample_head's pull-back
            v = torch.stack([v[:, 0], -math.sin(tilt) * v[:, 2] + math.cos(tilt) * v[:, 1],
                             math.cos(tilt) * v[:, 2] + math.sin(tilt) * v[:, 1]], -1)
        q = c + v / s
    grid, norm = _B1[key]
    return sample(*grid, q)[:, 0] / norm


def turn(ph: bp.Phantom) -> bp.Phantom:
    """(x, y, z) -> (y, -x, z): anterior-posterior becomes the readout axis."""
    pos = torch.stack([ph.pos[:, 1], -ph.pos[:, 0], ph.pos[:, 2]], -1)
    ph.pos = pos
    return ph


def head_for(spec: dict, seed: int, table, root) -> bp.DigitalPhantom:
    """The realism preset's anatomy (simulate_heads --preset realistic --source ixi --residual) for this seed;
    ``root``: the anatomy data (``validation/anatomy/data``)."""
    a = spec["anatomy"]
    prepared = None if a.get("prepared") is None else Path(a["prepared"]).expanduser()
    subject = a.get("subject")
    if subject is None and a.get("subjects"):  # a pool of subjects (e.g. a held-out set for evaluation), drawn by the seed
        pool = list(a["subjects"])
        subject = int(pool[int(np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(17)[16]).integers(len(pool)))])
    head = bp.sample_head(seed, subject=subject, root=root, voxel=(1e-3,) * 3, table=table, scalp="brainweb",
                          ixi_prepared=prepared, size="native", yaw_sd=3.8, age=None, deep_grey=True,
                          magnitude=a.get("deform_mm", 3.0) * 1e-3, ventricle_age=None, pitch=(-10., 5.), source=a["source"],
                          falx=True, interpolation=3, residual=True, dermis=False, lesions=a.get("lesions"), texture={})
    drop_detached(head)
    flow_void(head)
    return head
