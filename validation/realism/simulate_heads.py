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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import time
from pathlib import Path

import numpy as np
import torch

import blochparty as bp
import realism as R  # noqa: E402
from blochparty.anatomy import Distribution

ROOT = Path(__file__).resolve().parents[2] / "validation" / "anatomy" / "data"
# the realism preset's building blocks live in blochparty.generate (imported here for this script's command line
# and for the scripts that use them through this module)
from blochparty.coils import coil_offset, normalised_sensitivities  # noqa: E402,F401
from blochparty.generate.heads import (MONRO_BELOW_BODIES, PROTOCOL_SPACING, b1_plus, centre, drop_detached,  # noqa: E402,F401
                                       flow_void, stack, thin_convexity_csf, trace_sulci, turn)
from blochparty.generate.protocols import (ACQUIRED_LINES, FOV, QUALITY, REAL_ETL, SLICES, THICKNESS,  # noqa: E402,F401
                                           interleaved, real_protocols, sequence)
from blochparty.generate.sample import kspace_on_fov, simulate, to_real_orientation  # noqa: E402,F401


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
                ph.b1 = b1_plus(head, ph.pos, ROOT).to(ph.pos.dtype)
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
