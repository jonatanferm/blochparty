"""Simulated counterparts of the fastMRI 3 T T2-weighted brain scans (realism score, ROADMAP M5).

Each seed is a ``sample_head`` head imaged with the real protocol it is paired with (TR, TE and coil
count of one real volume, in turn): 2D TSE, ETL 16 with linear ordering (effective TE = 9 echo
spacings), 5 mm slices, 220 mm field of view, 128 x 128, RF on a 10 us raster. The slice stack is
placed by the same landmarks as the real stacks: slice 0 where the lateral ventricles are largest
(CSF more than 15 mm inside the brain) and the brain top at slice 13.5, so the spacing follows the
head (5.1 mm for BrainWeb subject 04, against 5 mm slices in the real data).

The real scans read out anterior-posterior; the phantom is turned by -90 deg about z, (x, y) ->
(y, -x), so the simulated readout (x) runs anterior-posterior too and chemical shift and artefacts
point the same way. ``to_real_orientation`` turns the image back (anterior up, patient right on
the image left, as fastMRI shows it).

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
from blochparty.anatomy import Distribution
from blochparty.sequence import System

ROOT = Path(__file__).resolve().parents[2] / "validation" / "anatomy" / "data"
FOV, THICKNESS, SLICES = 0.22, 5e-3, 16


def real_protocols(manifest: Path) -> list[dict]:
    m = json.loads(manifest.read_text())
    return [{"file": k, "TR": v["TR_ms"] * 1e-3, "TE": v["TE_ms"] * 1e-3, "coils": v["kspace_shape"][1],
             "refocus_deg": v.get("flip_deg") or 180.}
            for k, v in sorted(m.items()) if v["contrast"] == "AXT2" and (v.get("field_T") or 0) > 2]


PROTOCOL_SPACING = 5e-3  # contiguous 5 mm slices
MONRO_BELOW_BODIES = 10e-3  # the foramen of Monro / basal-ganglia level below the ventricle bodies


def stack(anatomy, mode: str = "landmarks") -> np.ndarray:
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
    top = z[np.nonzero(brain.reshape(len(z), -1).sum(1) * np.prod(anatomy.voxel[1:]) > 50e-6)[0].max()]
    z0 = z[np.argmax(deep)]
    if mode == "protocol":
        return z0 - MONRO_BELOW_BODIES + np.arange(SLICES) * PROTOCOL_SPACING
    return z0 + np.arange(SLICES) * (top - z0) / 13.5


def flow_void(head) -> None:
    """BrainWeb's vessel class holds the dural sinuses and large vessels; their blood flows at
    ~10-30 cm/s, and through a 5 mm slice a TSE echo at 113 ms keeps < 3 % of static blood's signal
    from 10 cm/s up (the engine's moving spins, tests/test_motion.py::test_tse_flow_void). Their
    spins are left out (PD 0 after the draw, so no other draw moves); susceptibility stays."""
    p = head.tissues["vessels"]
    head.tissues["vessels"] = replace(p, pd=Distribution(0., None, "TSE flow void (simulate_heads.flow_void): " + p.pd.citation, True))


def drop_detached(head) -> None:
    """Tissue not attached to the head (BrainWeb subject 38 has a 0.4 mL blob of "skin" 20 mm in front
    of the forehead, a marker in its source scan, and a few voxel specks) becomes air: every
    connected component of the head mask (tissue > 0.5) but the largest, grown by a voxel to take
    its partial voxels."""
    an = head.anatomy
    fr = an.fractions
    tissue = sum(v for k, v in fr.items() if k != "background").numpy() > .5
    lab, n = ndimage.label(tissue)
    if n < 2:
        return
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    drop = torch.from_numpy(ndimage.binary_dilation((lab > 0) & (lab != np.argmax(sizes))) & ~(lab == np.argmax(sizes)))
    out = {k: torch.where(drop, torch.zeros_like(v), v) for k, v in fr.items() if k != "background"}
    if "background" in fr:
        moved = sum(torch.where(drop, v, torch.zeros_like(v)) for k, v in fr.items() if k != "background")
        out["background"] = torch.where(drop, (fr["background"] + moved).clamp(max=1), fr["background"])
    head.anatomy = replace(an, fractions=out)


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


B1_MAP = Path(__file__).resolve().parents[1] / "anatomy" / "data" / "em" / "body.npz"
_B1 = {}


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
    """Simulated (y', x') image -> fastMRI display: rows anterior to posterior, columns patient right
    to left."""
    return img.T[::-1, :]


def interleaved(count=SLICES) -> list[int]:
    """Siemens' default interleaved order: slices 1, 3, 5, ... then 2, 4, ... (0-based 0, 2, ..., 1, 3, ...)."""
    return list(range(0, count, 2)) + list(range(1, count, 2))


QUALITY = {  # spins per voxel (z, y, x), sampling, slab half-width [voxels], RF raster [s], jitter
    "full": ((2, 2, 2), "categorical", 3, 1e-4, "stratified"),
    "fast": ((1, 2, 2), "categorical", 3, 1e-4, "stratified"),
    "legacy": ((2, 2, 2), "per_tissue", 4, 1e-5, "stratified"),
    # each voxel's composition exact, z a quadrature through the slice profile: no visible grain
    "real": ((4, 2, 2), "proportional", 3, 1e-4, "inplane"),
}


def sequence(n: int, tr: float, te: float, slice_order=None, rf_raster=1e-4, spacing=None, readout_os=1,
             refocus_deg=180.) -> bp.Sequence:
    """The paired protocol. With ``slice_order``: the neighbours as their real pulses, the rest on
    the bound pool only (0.71 % against every pulse exact, ``tse_2d``). ``readout_os``: readout
    oversampling (the real scans: 2). ``refocus_deg``: the refocusing flip angle (the headers'
    ``flipAngle_deg``: 180 in 11 of the 21 3 T scans, 140-179 in the rest)."""
    multi = {} if slice_order is None else dict(bound_only_beyond=1.5 * spacing)
    if readout_os != 1:
        multi["readout_oversampling"] = readout_os
    return bp.tse_2d(n, FOV, etl=16, esp=round(te / 9, 5), tr=tr, n_dummy=1, slice_thickness=THICKNESS,
                     refocus_flip_deg=refocus_deg, spoiling="gradient", ramps=True, system=System(rf_raster=rf_raster), slice_order=slice_order,
                     **multi)


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
    from blochparty import metal

    if metal.available():
        out = metal.simulate_metal(ph.to(torch.float32), seq.to(torch.float32), coils.to(torch.complex64),
                                   readouts="nufft", pulses="reuse")  # PERFORMANCE 2.1 and 2.3 (exact)
        return out.signal.to(torch.complex128), "metal"
    return bp.simulate(ph, seq, coils).signal, "core"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--slices", default="0-12")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--subject", type=int, default=None)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--manifest", type=Path, default=Path.home() / "data" / "fastmri_brain" / "manifest.json")
    ap.add_argument("--out", type=Path, default=Path.home() / "data" / "realism_sim")
    ap.add_argument("--multislice", action="store_true", help="play the other 15 slices' pulses")
    ap.add_argument("--mt", action="store_true", help="bound pool on (to_phantom(mt=True))")
    ap.add_argument("--table", choices=("default", "tissuedb"), default="default")
    ap.add_argument("--scalp", choices=("brainweb", "layered"), default="brainweb",
                    help="layered: BrainWeb's skin class split into skin and subcutaneous fat (anatomy.layer_scalp)")
    ap.add_argument("--size", choices=("brainweb", "anthropometric", "anthropometric_shell"), default="brainweb",
                    help="anthropometric: heads scaled to ANSUR II breadth and length (BrainWeb is MNI-sized); "
                         "_shell: by scaling the skull's contents, the skull and scalp keeping their thickness")
    ap.add_argument("--centre", action="store_true", help="field of view centred on the head (as an operator would)")
    ap.add_argument("--stack", choices=("landmarks", "protocol"), default="landmarks", help="slice positions (stack())")
    ap.add_argument("--yaw-sd", type=float, default=0.0, help="head yaw SD [deg] (sample_head(yaw_sd=))")
    ap.add_argument("--pitch", default=None, help="mean,sd [deg], nose down positive (sample_head(pitch=))")
    ap.add_argument("--age", default=None, help="years, or lo,hi drawn uniformly per head (sample_head(age=): sulci)")
    ap.add_argument("--readout-os", type=int, default=1, help="readout oversampling (the real scans: 2)")
    ap.add_argument("--deep-grey", action="store_true",
                    help="caudate, putamen, pallidum and thalamus as their own tissues (sample_head(deep_grey=); --table tissuedb)")
    ap.add_argument("--deform-mm", type=float, default=3.0, help="random deformation magnitude [mm] (sample_head(magnitude=))")
    ap.add_argument("--flow-void", action="store_true", help="BrainWeb's vessels (sinuses, large arteries) as TSE flow voids (flow_void())")
    ap.add_argument("--coil-radius", type=float, default=0.18,
                    help="receive ring radius [m] (birdcage_sensitivities); head coils' elements sit at ~0.12-0.13 m")
    ap.add_argument("--refocus", choices=("180", "protocol"), default="180",
                    help="protocol: each paired scan's refocusing angle from its header")
    ap.add_argument("--ventricle-age", default=None,
                    help="years, or lo,hi drawn uniformly per head: lateral ventricles enlarged for that age (Yamada 2023)")
    ap.add_argument("--sas-mm", type=float, default=0.0,
                    help="calibration: the outer mm of convexity CSF become inner table (thin_convexity_csf)")
    ap.add_argument("--b1", choices=("uniform", "birdcage"), default="uniform",
                    help="birdcage: B1+ of the full-wave 3 T body coil map (b1_plus)")
    ap.add_argument("--preset", choices=("realistic",), default=None,
                    help="realistic: --multislice --mt --table tissuedb --scalp layered --size anthropometric_shell "
                         "--centre --stack protocol --yaw-sd 3.8 --quality real --readout-os 2 --b1 birdcage --deep-grey "
                         "--flow-void --refocus protocol --ventricle-age 20,80 --sas-mm 2 --pitch -10,5 (no --age: "
                         "open_sulci's 1 mm ridges "
                         "make CSF specks that look fake)")
    ap.add_argument("--quality", choices=tuple(QUALITY), default="full")
    ap.add_argument("--spins", default=None, help="override: spins per voxel along z,y,x")
    args = ap.parse_args()
    if args.preset == "realistic":
        args.multislice = args.mt = args.centre = True
        args.table, args.scalp, args.size, args.stack, args.yaw_sd = "tissuedb", "layered", "anthropometric_shell", "protocol", 3.8
        args.quality = "real"
        args.readout_os = 2
        args.b1 = "birdcage"
        args.deep_grey = args.flow_void = True
        args.refocus = "protocol"
        args.ventricle_age = args.ventricle_age or "20,80"  # fastMRI ships no ages: adults, uniform (an assumption)
        args.sas_mm = args.sas_mm or 2.0  # a calibration (thin_convexity_csf)
        args.pitch = args.pitch or "-10,5"  # mean fitted to the real stacks' head-area profile; SD assumed

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
    protocols = real_protocols(args.manifest)
    age = None if args.age is None else (tuple(float(a) for a in args.age.split(",")) if "," in args.age else float(args.age))
    pitch = None if args.pitch is None else tuple(float(a) for a in args.pitch.split(","))
    vage = None if args.ventricle_age is None else (tuple(float(a) for a in args.ventricle_age.split(","))
                                                     if "," in args.ventricle_age else float(args.ventricle_age))
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in span(args.seeds):
        dest = args.out / f"sim_{seed:03d}.npz"
        if dest.exists():
            continue
        t0 = time.perf_counter()
        proto = protocols[seed % len(protocols)]
        head = bp.sample_head(seed, subject=args.subject, root=args.root, voxel=(1e-3,) * 3, table=table, scalp=args.scalp,
                              size=args.size, yaw_sd=args.yaw_sd, age=age, deep_grey=args.deep_grey, magnitude=args.deform_mm * 1e-3,
                              ventricle_age=vage, pitch=pitch)
        drop_detached(head)
        if args.flow_void:
            flow_void(head)
        if args.sas_mm:
            thin_convexity_csf(head, args.sas_mm * 1e-3)
        fields = head.field_maps()
        zs = stack(head.anatomy, args.stack)
        cx, cy = centre(head.anatomy, zs) if args.centre else (0.0, 0.0)
        kspace, timing = {}, []

        def prepare(i):
            """Slice i's phantom, coils and merged sequence (CPU), built while the GPU runs slice i - 1."""
            order = [float(zs[j] - zs[i]) for j in interleaved()] if args.multislice else None
            seq = sequence(args.n, proto["TR"], proto["TE"], order, rf_raster, float(zs[1] - zs[0]), args.readout_os,
                           proto["refocus_deg"] if args.refocus == "protocol" else 180.)
            k = round((zs[i] - head.anatomy.origin[2]) / head.anatomy.voxel[0])
            region = (slice(max(k - half, 0), k + half + 1), slice(None), slice(None))
            ph = head.to_phantom(fov_region=region, field_maps=fields, spins_per_voxel=spins, sampling=sampling, jitter=jitter,
                                 mt=args.mt)
            if args.b1 == "birdcage":
                ph.b1 = b1_plus(head, ph.pos).to(ph.pos.dtype)
            ph.pos = ph.pos - torch.tensor([cx, cy, zs[i]], dtype=ph.pos.dtype)
            ph = turn(ph)
            return seq, ph, bp.birdcage_sensitivities(ph.pos, proto["coils"], radius=args.coil_radius)

        slices = list(span(args.slices))
        with ThreadPoolExecutor(max_workers=1) as pool:  # PERFORMANCE 2.7: CPU and GPU overlap
            pending = pool.submit(prepare, slices[0])
            for n, i in enumerate(slices):
                t = time.perf_counter()
                seq, ph, coils = pending.result()
                if n + 1 < len(slices):
                    pending = pool.submit(prepare, slices[n + 1])
                signal, engine = simulate(ph, seq.merge_free(), coils)  # exact for static spins (PERFORMANCE 2.4)
                kspace[f"slice{i:02d}"] = kspace_on_fov(signal, seq).astype(np.complex64)
                timing.append(time.perf_counter() - t)
                print(f"seed {seed} slice {i}: {len(ph):,} spins, {timing[-1]:.1f} s ({engine})", flush=True)
        meta = {"seed": seed, "subject": head.meta["subject"], "protocol": proto, "z_mm": (zs * 1e3).round(2).tolist(),
                "multislice": args.multislice, "mt": args.mt, "table": args.table, "scalp": args.scalp,
                "size": head.meta.get("size", "brainweb"), "centre": [cx, cy], "pose": head.meta.get("pose"),
                "age": head.meta.get("age"), "deep_grey": args.deep_grey, "deform_mm": args.deform_mm, "flow_void": args.flow_void, "sas_mm": args.sas_mm, "refocus": args.refocus, "coil_radius": args.coil_radius,
                "ventricles": head.meta.get("ventricles"),
                "stack": args.stack,
                "readout_os": args.readout_os, "b1": args.b1, "quality": args.quality,
                "spins": list(spins), "sampling": sampling, "jitter": jitter, "slab_voxels": 2 * half + 1, "rf_raster": rf_raster,
                "jacobian": head.meta["deformation"]["jacobian"], "n": args.n, "readouts": "nufft", "pulses": "reuse",
                "seconds": round(time.perf_counter() - t0)}
        np.savez(dest, meta=json.dumps(meta), **kspace)
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
