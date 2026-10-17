"""Simulated counterparts of the fastMRI 3 T T2-weighted brain scans (realism score, ROADMAP M5).

Each seed is a ``sample_head`` head imaged with the real protocol it is paired with (TR, TE and coil
count of one real volume, in turn): 2D TSE, ETL 16 with linear ordering (effective TE = 9 echo
spacings), 5 mm slices, 220 mm field of view, 128 x 128, RF on a 10 us raster. The slice stack is
placed by the same landmarks as the real stacks: slice 0 where the lateral ventricles are largest
(CSF more than 15 mm inside the brain) and the brain top at slice 13.5, so the spacing follows the
head (5.1 mm for BrainWeb subject 04, against 5 mm slices in the real data).

The real scans read out anterior-posterior; the phantom is turned by -90 deg about z, (x, y) ->
(y, -x), so the simulated readout (x) runs anterior-posterior too and chemical shift and artefacts
point the same way. ``to_real_orientation`` turns the image to the way fastMRI's reconstruction shows
it: posterior up (validation/realism/orientation.py; until 2026-10-12 the simulated heads were shown
anterior up against posterior-up real ones), columns as before (left-right is not established; the
score flips it at random).

Noise-free k-space (coils, 128, 128; k = 0 at index 64) is saved per slice; noise is added later,
calibrated to the real images' noise floor, so the noise model can change without simulating again.

Quality (``--quality``), measured on one 64 x 64 slice against 27 spins per tissue and voxel
(whose own seed-to-seed RMS difference is 0.0073 of the normalised image):

* ``full``: 8 spins per voxel drawing their tissue (``sampling="categorical"``), a +-3 mm slab,
  RF on a 100 us raster: 1.8 M spins, RMS 0.014;
* ``fast``: 4 such spins per voxel: 0.9 M spins, RMS 0.022;
* ``legacy``: what the first runs used, 8 spins per tissue and voxel, +-4 mm, 10 us: 9.4 M spins,
  RMS 0.013, about 14 times the GPU time of ``full``.

The RF raster costs nothing in accuracy (100 us against 10 us: RMS 0.001) and the +-3 mm slab
nothing against +-4 mm; ``--spins`` overrides the spin count.

Variants (for ablations, same seeds = same heads): ``--multislice`` plays the other 15 slices'
pulses in the real interleaved order (odd slices, then even; ``tse_2d(slice_order=...)``), so the
imaged slice gets their incidental magnetization transfer and cross-talk; ``--mt`` gives the spins
their bound pool (``to_phantom(mt=True)``); ``--table tissuedb`` draws the tissues from
``tissuedb.tissue_table(3.0)`` instead of ``anatomy.DEFAULT_TISSUES``.

    uv run python validation/realism/simulate_heads.py --seeds 0-19 --out ~/data/realism_sim
"""
from __future__ import annotations

import argparse
import gc
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

import blochparty as bp
import realism as R  # noqa: E402
from blochparty.anatomy import Distribution
from blochparty.sequence import System

ROOT = Path(__file__).resolve().parents[2] / "validation" / "anatomy" / "data"
FOV, THICKNESS, SLICES = 0.22, 5e-3, 16
ACQUIRED_LINES = 396  # phase-encode lines of the real AXT2 scans (encodedSpace y in all 21 headers)
# the real AXT2 echo trains (validation/realism/echo_train.py): their k-space energy steps every 19 lines (ky -161,
# -142, ..., -28, -9.5 | +9.5): 396 lines over 21 echoes in linear order, ky = 0 in echo 11 = TE / the header's echo
# spacing (113 / 10.26 ms and 105 / 9.56 ms)
REAL_ETL = 21


def real_protocols(manifest: Path) -> list[dict]:
    m = json.loads(manifest.read_text())
    return [{"file": k, "TR": v["TR_ms"] * 1e-3, "TE": v["TE_ms"] * 1e-3, "coils": v["kspace_shape"][1],
             "refocus_deg": v.get("flip_deg") or 180.,
             "esp": None if v.get("echo_spacing_ms") is None else v["echo_spacing_ms"] * 1e-3}
            for k, v in sorted(m.items()) if v["contrast"] == "AXT2" and (v.get("field_T") or 0) > 2]


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


def coil_offset(anatomy, zs, cy: float, radius: float, gap: float) -> float:
    """Anterior offset [m] of the receive ring's centre from the field-of-view centre ``cy`` when the head
    rests on the coil's posterior side: the ring's posterior-most point ``gap`` behind the head's most
    posterior point (from 30 mm below slice 0 up: the occiput, not the neck). The real images are
    brighter at the back than at the front (white matter at the front 0.89 of the back's, the outer
    ring 0.71: validation/realism/orientation.py's display), as a head lying on the posterior elements
    gives; a ring centred on the head gives 1.00 and 0.82."""
    head = (sum(v for k, v in anatomy.fractions.items() if k != "background") > 0.5).numpy()
    z = anatomy.origin[2] + np.arange(anatomy.shape[0]) * anatomy.voxel[0]
    rows = head[z >= zs[0] - 30e-3]
    back = anatomy.origin[1] + np.nonzero(rows.any((0, 2)))[0][0] * anatomy.voxel[1]
    return back - cy - gap + radius


B1_MAP = Path(__file__).resolve().parents[1] / "anatomy" / "data" / "em" / "body.npz"
_B1 = {}


def normalised_sensitivities(sens: torch.Tensor, pos: torch.Tensor, smooth_mm: float, bin_mm: float = 4.0) -> torch.Tensor:
    """The receive array as the scanner shows it after its intensity normalisation (Siemens "Prescan Normalize" and
    its kin divide the image by a smooth estimate of the array's sensitivity): ``sens`` (N, C) divided by the
    root-sum-of-squares sensitivity smoothed over ``smooth_mm`` (Gaussian sigma) in the slice plane. The estimate is
    built from the spins themselves (binned at ``bin_mm``, normalised convolution over the head, bilinear back), so
    where the sensitivity rises faster than the estimate can follow (the last centimetre or two under the elements)
    the normalised image keeps the excess, as the real scans do (journal 2026-10-13: the outer 2 cm 1.3-2.7 x brighter
    than a flat receive gives while the interior matches one). ``pos`` in the turned frame (x anterior-posterior, y
    left-right)."""
    from scipy import ndimage
    rss = sens.abs().pow(2).sum(1).sqrt().cpu().numpy()
    xy = pos[:, :2].cpu().numpy() / (bin_mm * 1e-3)
    lo = np.floor(xy.min(0)) - 2
    ij = np.floor(xy - lo).astype(int)
    shape = tuple(ij.max(0) + 3)
    num = np.zeros(shape); den = np.zeros(shape)
    np.add.at(num, (ij[:, 0], ij[:, 1]), rss); np.add.at(den, (ij[:, 0], ij[:, 1]), 1.0)
    sigma = smooth_mm / bin_mm
    smooth = ndimage.gaussian_filter(num, sigma) / np.maximum(ndimage.gaussian_filter(den, sigma), 1e-9)
    est = ndimage.map_coordinates(smooth, (xy - lo).T, order=1, mode="nearest")
    return sens / torch.as_tensor(est, dtype=sens.real.dtype, device=sens.device)[:, None]


def b1_plus(head, pos: torch.Tensor) -> torch.Tensor:
    """Relative B1+ at spin positions [m, anatomy frame] of a generated head: the full-wave 3 T body
    birdcage map of BrainWeb subject 04 (validation/em/run_brainweb.py; mean 1 over its brain, 0.86-1.19
    p5-p95), sampled where each spin came from before ``sample_head`` resized and turned the head
    (BrainWeb's heads share MNI space). Ignores the small random deformation and the weaker dielectric
    effect of smaller heads."""
    from blochparty import em
    from blochparty.maps import sample

    if "fm" not in _B1:
        fm = em.field_maps(transmit=em.read_mariepy_maps(B1_MAP))
        ref = bp.load_brainweb(4, ROOT, voxel=(5e-3,) * 3, dtype=torch.float64)
        brain = (ref.fractions["gm"] + ref.fractions["wm"]) > 0.5
        p = torch.tensor(ref.origin, dtype=torch.float64) + brain.nonzero().double().flip(-1) * torch.tensor(ref.voxel[::-1], dtype=torch.float64)
        grid = fm.b1[None].double(), torch.tensor(fm.origin, dtype=torch.float64), torch.tensor(fm.spacing, dtype=torch.float64)
        _B1["fm"], _B1["norm"] = grid, float(sample(*grid, p)[:, 0].mean())
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
    return sample(*_B1["fm"], q)[:, 0] / _B1["norm"]


def turn(ph: bp.Phantom) -> bp.Phantom:
    """(x, y, z) -> (y, -x, z): anterior-posterior becomes the readout axis."""
    pos = torch.stack([ph.pos[:, 1], -ph.pos[:, 0], ph.pos[:, 2]], -1)
    ph.pos = pos
    return ph


def to_real_orientation(img: np.ndarray) -> np.ndarray:
    """Simulated (y', x') image -> fastMRI display: rows posterior to anterior (fastMRI's stored images
    show the head posterior up: validation/realism/orientation.py), columns from the patient's right
    (+x) to left (left-right is not established against the real data)."""
    return img.T


def interleaved(count=SLICES) -> list[int]:
    """Siemens' default interleaved order: slices 1, 3, 5, ... then 2, 4, ... (0-based 0, 2, ..., 1, 3, ...)."""
    return list(range(0, count, 2)) + list(range(1, count, 2))


QUALITY = {  # spins per voxel (z, y, x), sampling, slab half-width [voxels], RF raster [s], jitter
    "full": ((2, 2, 2), "categorical", 3, 1e-4, "stratified"),
    "fast": ((1, 2, 2), "categorical", 3, 1e-4, "stratified"),
    "legacy": ((2, 2, 2), "per_tissue", 4, 1e-5, "stratified"),
    # each voxel's composition exact, z a quadrature through the slice profile: no visible grain
    "real": ((4, 2, 2), "proportional", 3, 1e-4, "inplane"),
    # 4x4x4 in-plane-jittered: the random positions' grain (Immerkaer sigma in flat brain, of p99) falls as
    # 1 / sqrt(spins): 0.026 at 4x2x2, 0.017 at 4x3x3, 0.0136 here; real scans 0.014 with their noise
    "real_fine": ((4, 4, 4), "proportional", 3, 1e-4, "inplane"),
}


def sequence(n: int, tr: float, te: float, slice_order=None, rf_raster=1e-4, spacing=None, readout_os=1,
             refocus_deg=180., ordering="linear", etl=16, esp=None) -> bp.Sequence:
    """The paired protocol. With ``slice_order``: the neighbours as their real pulses, the rest on
    the bound pool only (0.71 % against every pulse exact, ``tse_2d``). ``readout_os``: readout
    oversampling (the real scans: 2). ``refocus_deg``: the refocusing flip angle (the headers'
    ``flipAngle_deg``: 180 in 11 of the 21 3 T scans, 140-179 in the rest). ``ordering``: "linear" (ky = 0 in
    echo 9 of 16), "linear_full" (the n lines take the echoes they have in the real scans' 396-line linear
    acquisition, ``ACQUIRED_LINES``: echoes 6-11, 25 shots) or "reverse_centric" (ky = 0 in echo 16; echo
    spacing te / 16). ``etl`` / ``esp`` (default 16 / None: te / 9, te / 16 reverse-centric): the echo train, e.g.
    the real scans' (``REAL_ETL`` at the header's echo spacing: 19 shots of "linear_full")."""
    multi = {} if slice_order is None else dict(bound_only_beyond=1.5 * spacing)
    if readout_os != 1:
        multi["readout_oversampling"] = readout_os
    if esp is None:
        esp = round(te / (16 if ordering == "reverse_centric" else 9), 5)
    full = dict(acquired_lines=ACQUIRED_LINES) if ordering == "linear_full" else {}
    return bp.tse_2d(n, FOV, etl=etl, esp=esp, ordering="linear" if full else ordering, tr=tr, n_dummy=1, slice_thickness=THICKNESS,
                     refocus_flip_deg=refocus_deg, spoiling="gradient", ramps=True, system=System(rf_raster=rf_raster), slice_order=slice_order,
                     **multi, **full)


def kspace_on_fov(signal, seq) -> np.ndarray:
    """(C, n, n) k-space on the field of view: an oversampled readout is cropped to the central n
    pixels after the inverse FFT, as the real reconstruction crops the scanner's 2x readout."""
    k = bp.cartesian_kspace(signal, seq).numpy()
    n, m = k.shape[-2], k.shape[-1]
    if m != n:  # centred, orthonormal transforms as realism.centred_ifft / centred_fft
        img = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(k, axes=-1), axis=-1, norm="ortho"), axes=-1)
        img = img[..., (m - n) // 2:(m - n) // 2 + n]
        k = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(img, axes=-1), axis=-1, norm="ortho"), axes=-1)
    return k


def simulate(ph, seq, coils):
    if bp.engines.available("gpu"):
        out = bp.simulate(ph.to(torch.float32), seq.to(torch.float32), coils.to(torch.complex64), engine="gpu",
                          readouts="nufft", pulses="reuse")  # PERFORMANCE 2.1 and 2.3 (exact)
        return out.signal.to(torch.complex128), bp.engines.resolve("gpu")
    return bp.simulate(ph, seq, coils).signal, "core"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--slices", default="0-12")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--subject", type=int, default=None)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--manifest", type=Path, default=R.FASTMRI / "manifest.json", help="the real scans' manifest (realism.FASTMRI)")
    ap.add_argument("--out", type=Path, default=Path.home() / "data" / "realism_sim")
    ap.add_argument("--multislice", action="store_true", help="play the other 15 slices' pulses")
    ap.add_argument("--mt", action="store_true", help="bound pool on (to_phantom(mt=True))")
    ap.add_argument("--no-mt", action="store_true", help="ablation: bound pool off even under --preset realistic")
    ap.add_argument("--table", choices=("default", "tissuedb"), default="default")
    ap.add_argument("--scalp", choices=("brainweb", "layered", "layered_drawn"), default=None,
                    help="layered: BrainWeb's skin class split into skin and subcutaneous fat (anatomy.layer_scalp); "
                         "brainweb (default): the anatomy's own classes")
    ap.add_argument("--size", choices=("brainweb", "anthropometric", "anthropometric_shell"), default="brainweb",
                    help="anthropometric: heads scaled to ANSUR II breadth and length (BrainWeb is MNI-sized); "
                         "_shell: by scaling the skull's contents, the skull and scalp keeping their thickness")
    ap.add_argument("--centre", action="store_true", help="field of view centred on the head (as an operator would)")
    ap.add_argument("--centre-offset-mm", type=float, default=None,
                    help="with --centre: the field of view's centre this far anterior of the head's bounding-box centre "
                         "(the real operators leave the head 2.8 px = 4.8 mm further posterior than a centred box, r27 "
                         "vs fastMRI, gap.py features: the preset's default)")
    ap.add_argument("--stack", choices=("landmarks", "protocol"), default="landmarks", help="slice positions (stack())")
    ap.add_argument("--yaw-sd", type=float, default=0.0, help="head yaw SD [deg] (sample_head(yaw_sd=))")
    ap.add_argument("--pitch", default=None, help="mean,sd [deg], nose down positive (sample_head(pitch=))")
    ap.add_argument("--age", default=None, help="years, or lo,hi drawn uniformly per head: closed sulci opened to that age's width (sample_head(age=), Jin 2018)")
    ap.add_argument("--readout-os", type=int, default=1, help="readout oversampling (the real scans: 2)")
    ap.add_argument("--interp-order", type=int, choices=(1, 3), default=1,
                    help="order of the head's final pull-back (sample_head(interpolation=)): 3 = cubic B-spline")
    ap.add_argument("--residual", action="store_true",
                    help="IXI heads carry their fitted T2 / PD multipliers (sample_head(residual=True)), which replace "
                         "the assumed within-tissue texture (no texture with it)")
    ap.add_argument("--lesions", default=None,
                    help='JSON for sample_head(lesions=...), e.g. \'{"count": 3, "size_mm": [3, 10], "region": "periventricular"}\'')
    ap.add_argument("--dermis", action="store_true",
                    help="the skin's dermis as its own class (sample_head(dermis=True), anatomy.add_dermis: the outer "
                         "1.61 mm, T2 ~12 ms)")
    ap.add_argument("--no-texture", action="store_true",
                    help="no within-tissue texture (sample_head(texture={}); default: anatomy.DEFAULT_TEXTURE)")
    ap.add_argument("--falx", action="store_true", help="the falx cerebri as dura in the interhemispheric fissure "
                    "(sample_head(falx=True), anatomy.add_falx)")
    ap.add_argument("--deep-grey", action="store_true",
                    help="caudate, putamen, pallidum and thalamus as their own tissues (sample_head(deep_grey=); --table tissuedb)")
    ap.add_argument("--deform-mm", type=float, default=3.0, help="random deformation magnitude [mm] (sample_head(magnitude=))")
    ap.add_argument("--flow-void", action="store_true", help="BrainWeb's vessels (sinuses, large arteries) as TSE flow voids (flow_void())")
    ap.add_argument("--override", action="append", default=[],
                    help="experiment: class.property=value on the tissue table (e.g. gm.t2=0.083), marked assumed")
    ap.add_argument("--coil-radius", type=float, default=None,
                    help="receive ring radius [m] (birdcage_sensitivities; default 0.18, the preset 0.14); head coils' "
                         "elements sit at ~0.12-0.14 m")
    ap.add_argument("--coil-radius-lr", type=float, default=None,
                    help="the receive ring's left-right semi-axis [m] (an ellipse; --coil-radius is anterior-posterior)")
    ap.add_argument("--receive-draw", action="store_true",
                    help="draw the receive per head: ring semi-axes x U(1, 1.25), gap behind the occiput U(5, 30) mm, "
                         "normalisation length U(20, 60) mm (the real scans' spread of coils, positioning and normalisation)")
    ap.add_argument("--receive-scale", default="1.1,1.45",
                    help="with --receive-draw: the ring's semi-axes scale drawn uniformly in lo,hi (1.0,1.25 matched the "
                         "first 34 scans' scalp level 4.4 over WM; the 65-scan set reads 3.87 +- 1.02: 2026-10-15)")
    ap.add_argument("--normalise-range", default="20,60",
                    help="with --receive-draw: the normalisation length [mm] drawn uniformly in lo,hi (a longer length keeps "
                         "more of the array's shading, as the real scans' broad shading suggests: journal 2026-10-16)")
    ap.add_argument("--coil-element-radius", type=float, default=None,
                    help="the receive elements' loop radius [m]: a loop's on-axis falloff instead of 1 / distance")
    ap.add_argument("--receive-normalise-mm", type=float, default=None,
                    help="the scanner's intensity normalisation: receive sensitivities divided by their RSS smoothed over "
                         "this sigma [mm] in the slice plane (normalised_sensitivities); flat inside, the excess under "
                         "the elements kept")
    ap.add_argument("--coil-gap-mm", type=float, default=None,
                    help="the head rests on the coil's posterior side: the receive ring's back this far behind the "
                         "head's back (coil_offset); default: the ring centred on the head")
    ap.add_argument("--train", choices=("fitted", "header"), default="fitted",
                    help="echo train: 'fitted' ETL 16, echo spacing TE / 9 (TE / 16 reverse-centric); 'header' the real "
                         "scans' (REAL_ETL echoes at the header's echo spacing, ky = 0 in echo 11; needs --ordering "
                         "linear_full)")
    ap.add_argument("--ordering", choices=("linear", "linear_full", "reverse_centric"), default="linear",
                    help="TSE phase-encode ordering (sequence()); the real k-space is symmetric in ky")
    ap.add_argument("--refocus", choices=("180", "protocol"), default="180",
                    help="protocol: each paired scan's refocusing angle from its header")
    ap.add_argument("--ventricle-age", default=None,
                    help="years, lo,hi drawn uniformly per head, or 'age' (the head's --age): lateral ventricles enlarged for that age (Yamada 2023)")
    ap.add_argument("--trace-sulci", action="store_true", help="experimental: open sulci along BrainWeb's faint CSF trace")
    ap.add_argument("--sas-mm", type=float, default=0.0,
                    help="calibration: the outer mm of convexity CSF become inner table (thin_convexity_csf)")
    ap.add_argument("--b1", choices=("uniform", "birdcage"), default=None,
                    help="birdcage: B1+ of the full-wave 3 T body coil map (b1_plus)")
    ap.add_argument("--preset", choices=("realistic",), default=None,
                    help="realistic: --multislice --mt --table tissuedb --scalp layered_drawn --size anthropometric_shell "
                         "--centre --stack protocol --yaw-sd 3.8 --quality real --readout-os 2 --b1 birdcage --deep-grey "
                         "--flow-void --refocus protocol --ventricle-age 20,80 --sas-mm 1.5 --pitch -10,5 "
                         "--coil-radius 0.14 --coil-gap-mm 25 --falx --interp-order 3; with --source ixi: --size native, no ventricle age, no "
                         "rim calibration, the subject's own scalp (--scalp brainweb), slice 0 12.5 mm below the ventricle bodies")
    ap.add_argument("--stack-offset-mm", type=float, default=None,
                    help="protocol stack: slice 0 this far below the ventricle bodies' largest cross-section "
                         f"(default {MONRO_BELOW_BODIES * 1e3:g} mm)")
    ap.add_argument("--prepared", type=Path, default=None,
                    help="with --source ixi: the directory of prepared heads (default <root>/ixi/prepared); runs queued on one "
                         "machine each name their own instead of swapping one directory (LESSONS 2026-10-13)")
    ap.add_argument("--source", choices=("brainweb", "ixi"), default="brainweb",
                    help="ixi: heads segmented from IXI 3 T subjects (validation/anatomy/prepare_ixi.py); with the "
                         "preset: their own size, sulci and ventricles, no rim calibration")
    ap.add_argument("--quality", choices=tuple(QUALITY), default="full")
    ap.add_argument("--serial", action="store_true",
                    help="build each slice's phantom after the last one is simulated, not during it: one 10-15 M-spin "
                         "phantom and its coil maps in memory instead of two (the 16 GB mini swapped 6.7 GB in r24)")
    ap.add_argument("--spins", default=None, help="override: spins per voxel along z,y,x")
    args = ap.parse_args()
    args.receive_scale = tuple(float(v) for v in args.receive_scale.split(","))
    args.normalise_range = tuple(float(v) for v in args.normalise_range.split(","))
    if args.train == "header" and args.ordering != "linear_full":
        ap.error("--train header needs --ordering linear_full (the real scans' 396-line linear order)")
    if args.preset == "realistic":
        args.multislice = args.mt = args.centre = True
        args.mt = not args.no_mt
        args.centre_offset_mm = 4.8 if args.centre_offset_mm is None else args.centre_offset_mm
        args.table, args.size, args.stack, args.yaw_sd = "tissuedb", "anthropometric_shell", "protocol", 3.8
        # BrainWeb lumps scalp fat into "skin": a drawn layer; IXI heads keep their own scalp fat (prepare_ixi)
        args.scalp = args.scalp or ("brainweb" if args.source == "ixi" else "layered_drawn")
        args.quality = "real" if args.quality == "full" else args.quality
        args.readout_os = 2
        args.b1 = args.b1 or "birdcage"
        args.deep_grey = args.flow_void = args.falx = True
        args.interp_order = 3
        args.refocus = "protocol"
        args.ventricle_age = args.ventricle_age or "20,80"  # fastMRI ships no ages: adults, uniform (an assumption)
        args.sas_mm = args.sas_mm or 1.5  # a calibration (thin_convexity_csf)
        args.pitch = args.pitch or "-10,5"  # mean fitted to the real stacks' head-area profile; SD assumed
        # the head rests on the coil's posterior elements (coil_offset): a 14 cm ring 25 mm behind the head's back
        # matches the real images' front-to-back fall (white matter 0.89, outer ring 0.71; IXI012: 0.83, 0.70)
        args.coil_radius = args.coil_radius or 0.14
        args.coil_gap_mm = 25.0 if args.coil_gap_mm is None else args.coil_gap_mm
        if args.source == "ixi":  # real anatomy: no sulcal, ventricle, rim or scalp edits; the subject's own size
            args.size, args.ventricle_age, args.age = "native", None, None
            args.sas_mm = 0.0
            # slice 0 12.5 mm below the ventricle bodies: on proxy images of r22's 16 heads (each class's spin-echo
            # signal, 5 mm slabs, 1.72 mm pixels) the lateral-ventricle cross-section over slices 0-3 is 211 / 240 /
            # 250 / 223 px against the real 181 / 220 / 272 / 218 (10 mm: falls to 138 at slice 3; 15 mm: peaks a
            # slice late) and the head-area profile stays within 0.02 of the real one
            if args.stack_offset_mm is None:
                args.stack_offset_mm = 12.5

    args.scalp = args.scalp or "brainweb"
    args.coil_radius = args.coil_radius or 0.18
    args.b1 = args.b1 or "uniform"

    def span(s):  # "a-b" or "a-b:step"
        rng, _, step = s.partition(":")
        return range(int(rng.split("-")[0]), int(rng.split("-")[-1]) + 1, int(step or 1))

    spins, sampling, half, rf_raster, jitter = QUALITY[args.quality]
    if args.spins:
        spins = tuple(int(c) for c in args.spins.split(","))
    table = bp.DEFAULT_TISSUES
    if args.table == "tissuedb":
        from blochparty import tissuedb

        table = {3.0: tissuedb.tissue_table(3.0)}
    for item in args.override:
        key, value = item.split("=")
        cls, prop = key.split(".")
        table = {f: dict(t) for f, t in table.items()}
        old = getattr(table[3.0][cls], prop)
        table[3.0][cls] = replace(table[3.0][cls], **{prop: Distribution(float(value), None if old is None else old.sd,
                                                                          f"experiment override (--override {item})", True)})
    protocols = real_protocols(args.manifest)
    age = None if args.age is None else (tuple(float(a) for a in args.age.split(",")) if "," in args.age else float(args.age))
    pitch = None if args.pitch is None else tuple(float(a) for a in args.pitch.split(","))
    vage = None if args.ventricle_age is None else ("age" if args.ventricle_age == "age" else
                                                    tuple(float(a) for a in args.ventricle_age.split(","))
                                                    if "," in args.ventricle_age else float(args.ventricle_age))
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in span(args.seeds):
        dest = args.out / f"sim_{seed:03d}.npz"
        if dest.exists():
            continue
        t0 = time.perf_counter()
        proto = protocols[seed % len(protocols)]
        head = bp.sample_head(seed, subject=args.subject, root=args.root, voxel=(1e-3,) * 3, table=table, scalp=args.scalp, ixi_prepared=args.prepared,
                              size=args.size, yaw_sd=args.yaw_sd, age=age, deep_grey=args.deep_grey, magnitude=args.deform_mm * 1e-3,
                              ventricle_age=vage, pitch=pitch, source=args.source, falx=args.falx,
                              interpolation=args.interp_order, residual=args.residual, dermis=args.dermis,
                              lesions=json.loads(args.lesions) if args.lesions else None,
                              **({"texture": {}} if args.no_texture or args.residual else {}))
        drop_detached(head)
        if args.flow_void:
            flow_void(head)
        if args.sas_mm:
            thin_convexity_csf(head, args.sas_mm * 1e-3)
        traced = trace_sulci(head) if args.trace_sulci else None
        fields = head.field_maps()
        zs = stack(head.anatomy, args.stack, MONRO_BELOW_BODIES if args.stack_offset_mm is None else args.stack_offset_mm * 1e-3,
                   head.meta["ventricle_landmark"][0] if args.source == "ixi" else None)
        cx, cy = centre(head.anatomy, zs) if args.centre else (0.0, 0.0)
        cy += (args.centre_offset_mm or 0.0) * 1e-3
        # the receive as it varies from scan to scan (coil, positioning, the site's normalisation): per head, the ring's
        # semi-axes scaled, its gap behind the occiput and the normalisation length drawn (2026-10-13: across 34 real
        # scans the scalp's bright decile over white matter spans 2.1-6.0, SD 0.8, one run of ours 4.5 +- 0.4)
        draw = {}
        coil_radius, coil_radius_lr, gap_mm, norm_mm = args.coil_radius, args.coil_radius_lr, args.coil_gap_mm, args.receive_normalise_mm
        if args.receive_draw:
            rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(15)[14])
            lo, hi = args.receive_scale
            nlo, nhi = args.normalise_range
            draw = {"radius_scale": float(rng.uniform(lo, hi)), "gap_mm": float(rng.uniform(5., 30.)),
                    "normalise_mm": float(rng.uniform(nlo, nhi))}
            coil_radius = coil_radius * draw["radius_scale"]
            coil_radius_lr = None if coil_radius_lr is None else coil_radius_lr * draw["radius_scale"]
            gap_mm = draw["gap_mm"]
            norm_mm = draw["normalise_mm"] if norm_mm else norm_mm
        ring = 0.0 if gap_mm is None else coil_offset(head.anatomy, zs, cy, coil_radius, gap_mm * 1e-3)
        kspace, timing = {}, []

        def prepare(i):
            """Slice i's phantom, coils and merged sequence (CPU), built while the GPU runs slice i - 1."""
            order = [float(zs[j] - zs[i]) for j in interleaved()] if args.multislice else None
            seq = sequence(args.n, proto["TR"], proto["TE"], order, rf_raster, float(zs[1] - zs[0]), args.readout_os,
                           proto["refocus_deg"] if args.refocus == "protocol" else 180., args.ordering,
                           **(dict(etl=REAL_ETL, esp=proto["esp"]) if args.train == "header" else {}))
            k = round((zs[i] - head.anatomy.origin[2]) / head.anatomy.voxel[0])
            region = (slice(max(k - half, 0), k + half + 1), slice(None), slice(None))
            ph = head.to_phantom(fov_region=region, field_maps=fields, spins_per_voxel=spins, sampling=sampling, jitter=jitter,
                                 mt=args.mt)
            if args.b1 == "birdcage":
                ph.b1 = b1_plus(head, ph.pos).to(ph.pos.dtype)
            ph.pos = ph.pos - torch.tensor([cx, cy, zs[i]], dtype=ph.pos.dtype)
            ph = turn(ph)
            ring_pos = ph.pos - torch.tensor([ring, 0.0, 0.0], dtype=ph.pos.dtype) if ring else ph.pos  # turned: x is AP
            sens = bp.birdcage_sensitivities(ring_pos, proto["coils"], radius=coil_radius, radius_y=coil_radius_lr,
                                             element_radius=args.coil_element_radius)
            if norm_mm:
                sens = normalised_sensitivities(sens, ph.pos, norm_mm)
            return seq, ph, sens

        slices = list(span(args.slices))
        with ThreadPoolExecutor(max_workers=1) as pool:  # PERFORMANCE 2.7: CPU and GPU overlap (not with --serial)
            pending = None if args.serial else pool.submit(prepare, slices[0])
            for n, i in enumerate(slices):
                t = time.perf_counter()
                seq, ph, coils = prepare(i) if args.serial else pending.result()
                if n + 1 < len(slices) and not args.serial:
                    pending = pool.submit(prepare, slices[n + 1])
                signal, engine = simulate(ph, seq.merge_free(), coils)  # exact for static spins (PERFORMANCE 2.4)
                kspace[f"slice{i:02d}"] = kspace_on_fov(signal, seq).astype(np.complex64)
                timing.append(time.perf_counter() - t)
                print(f"seed {seed} slice {i}: {len(ph):,} spins, {timing[-1]:.1f} s ({engine})", flush=True)
                del seq, ph, coils, signal
        meta = {"seed": seed, "subject": head.meta["subject"], "protocol": proto, "z_mm": (zs * 1e3).round(2).tolist(),
                "multislice": args.multislice, "mt": args.mt, "table": args.table, "scalp": args.scalp,
                "size": head.meta.get("size", "brainweb"), "centre": [cx, cy], "pose": head.meta.get("pose"),
                "age": head.meta.get("age"), "source": args.source, "prepared": None if args.prepared is None else str(args.prepared), "ixi": head.meta.get("ixi"), "deep_grey": args.deep_grey, "deform_mm": args.deform_mm, "flow_void": args.flow_void, "sas_mm": args.sas_mm, "refocus": args.refocus, "falx": args.falx, "interp_order": args.interp_order, "texture": not (args.no_texture or args.residual), "residual": args.residual, "dermis": args.dermis, "lesions": head.meta.get("lesions"), "coil_radius": coil_radius, "coil_radius_lr": coil_radius_lr, "coil_element_radius": args.coil_element_radius, "coil_gap_mm": gap_mm, "coil_offset_mm": round(ring * 1e3, 1), "receive_normalise_mm": norm_mm, "receive_draw": draw or None, "ordering": args.ordering, "train": args.train,
                "override": args.override, "traced_sulci_ml": None if traced is None else round(traced * 1e6, 2),
                "ventricles": head.meta.get("ventricles"),
                "stack": args.stack, "stack_offset_mm": args.stack_offset_mm,
                "readout_os": args.readout_os, "b1": args.b1, "quality": args.quality,
                "spins": list(spins), "sampling": sampling, "jitter": jitter, "slab_voxels": 2 * half + 1, "rf_raster": rf_raster,
                "jacobian": head.meta["deformation"]["jacobian"], "n": args.n, "readouts": "nufft", "pulses": "reuse",
                "seconds": round(time.perf_counter() - t0)}
        np.savez(dest, meta=json.dumps(meta), **kspace)
        print(json.dumps(meta), flush=True)
        del head, fields, kspace
        gc.collect()  # one process over many heads grew to 15 GB on a 16 GB machine
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
