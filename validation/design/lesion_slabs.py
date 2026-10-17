"""Slab phantoms for the lesion-classification design experiment (ROADMAP M7, agentic design): from each prepared IXI
head, slabs at a few MNI levels rendered twice through the lesion stream, with lesions and without, as ready-to-simulate
``Phantom`` objects at 2 mm anatomy voxels (~80k spins for a 5 mm slab over 220 mm), with the label, the lesion pixel
count and the lesion mask on the acquisition grid.

    uv run python validation/design/lesion_slabs.py --prepared validation/anatomy/data/ixi/prepared_v17 --out ~/data/slabs

Writes ``<out>/slab_<subject>_<z>_<les|nol|les2|...>.pt`` (a dict: phantom fields, label, lesion_px, lesion_mask (n, n), meta)
and ``<out>/index.json`` (subject, level, lesions, train / test split by subject: the last ``--test-subjects`` subjects).
Incremental: existing slabs are kept and only missing ones built, so ``--draws 3`` on a finished folder adds two more
lesion draws (``les2``, ``les3``: other lesions on the same anatomy, more positives for the same heads) per level.
The phantoms carry the tissue table's 3 T values, the subject's residual maps and the bound pool (MT) and nothing else of
the realism preset (no deformation, pose, receive or noise: the experiment compares sequences, not realism).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from blochparty import tissuedb
from blochparty.anatomy import DigitalPhantom, add_lesions, load_ixi

HERE = Path(__file__).resolve().parent
IXI = HERE.parents[1] / "validation" / "anatomy" / "data" / "ixi"
FOV, THICKNESS = 0.22, 5e-3


def lesion_mask(anatomy, z_m, n, fov=FOV):
    """The lesion fraction averaged over the slab, on the n x n acquisition grid in the slab image's frame (rows along
    +y, columns along +x: ``kspace_grid`` + ``ifft2c`` of a phantom that is not turned), from the anatomy's voxel grid."""
    from scipy import ndimage
    fr = anatomy.fractions.get("lesion")
    if fr is None:
        return np.zeros((n, n), np.float32)
    z = anatomy.origin[2] + np.arange(anatomy.shape[0]) * anatomy.voxel[0]
    k = np.nonzero(np.abs(z - z_m) <= THICKNESS / 2 + 1e-9)[0]
    slab = fr.numpy()[k].mean(0)
    px = fov / n
    u = (np.arange(n) - n / 2 + .5) * px
    rows, cols = np.meshgrid(u, u, indexing="ij")
    # the head's box centre in the plane (as simulate_heads.centre), no operator offset here
    head = (sum(v for kk, v in anatomy.fractions.items() if kk != "background") > .5).numpy()
    ys, xs = np.nonzero(head[k].any(0))
    cy = anatomy.origin[1] + (ys.min() + ys.max()) / 2 * anatomy.voxel[1]
    cx = anatomy.origin[0] + (xs.min() + xs.max()) / 2 * anatomy.voxel[2]
    iy = (cy + rows - anatomy.origin[1]) / anatomy.voxel[1]
    ix = (cx + cols - anatomy.origin[0]) / anatomy.voxel[2]  # the slab image's own frame (rows +y, columns +x: no turn)
    return ndimage.map_coordinates(slab, [iy.ravel(), ix.ravel()], order=1, mode="constant", cval=0.).reshape(n, n).astype(np.float32), (cx, cy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, default=IXI / "prepared_v17")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--levels-mm", default="10,20,30", help="MNI z of the slab centres")
    ap.add_argument("--n", type=int, default=64, help="acquisition matrix (for the lesion mask)")
    ap.add_argument("--voxel-mm", type=float, default=2.0, help="anatomy voxel the heads are resampled to")
    ap.add_argument("--spins", default="1,2,2", help="spins per anatomy voxel (z, y, x): 4 in plane keep the grain below the lesions")
    ap.add_argument("--lesions", default="1,2", help="count range per imaged level (with); size 10-20 mm (3-6 pixels at 3.4 mm), white matter within 2 mm of the level")
    ap.add_argument("--test-subjects", type=int, default=12)
    ap.add_argument("--subjects", default=None, help="comma-separated ids (default: all prepared)")
    ap.add_argument("--lesion-t2", type=float, default=0.2, help="the task's lesion T2 [s] at 3 T: a conspicuous T2-hyperintense "
                    "lesion (chronic plaques / gliosis read near CSF on T2w); the table's 0.111 (West 2014's mixed MS lesions) "
                    "is only ~20 % above the surrounding tissue after partial volume and the residual maps")
    ap.add_argument("--lesion-pd", type=float, default=0.85)
    ap.add_argument("--draws", type=int, default=1, help="lesion draws per head (kinds les, les2, ...): more positives on the same anatomy")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.json"
    index = {r["file"]: r for r in json.loads(index_path.read_text())} if index_path.exists() else {}
    index = {f: r for f, r in index.items() if (args.out / f).exists()}
    ids = sorted(int(p.stem[3:]) for p in args.prepared.glob("IXI*.npz"))
    if args.subjects:
        ids = [int(s) for s in args.subjects.split(",")]
    test = set(ids[-args.test_subjects:]) if args.test_subjects > 0 else set()
    from dataclasses import replace

    from blochparty.anatomy import Distribution
    table = tissuedb.tissue_table(3.0)
    les = table["lesion"]
    table["lesion"] = replace(les, t2=Distribution(args.lesion_t2, None, f"design task: T2 {args.lesion_t2 * 1e3:.0f} ms assumed (lesion_slabs --lesion-t2)", True),
                              t2star=Distribution(args.lesion_t2 * .8, None, "design task: 0.8 x T2", True),
                              pd=Distribution(args.lesion_pd, None, "design task: PD assumed (lesion_slabs --lesion-pd)", True))
    levels = [float(v) * 1e-3 for v in args.levels_mm.split(",")]
    spins = tuple(int(v) for v in args.spins.split(","))
    lo, hi = (int(v) for v in args.lesions.split(","))
    kinds = ["les", "nol"] + [f"les{d}" for d in range(2, args.draws + 1)]
    for sid in ids:
        if all((args.out / f"slab_{sid:03d}_{round(z * 1e3):02d}_{kind}.pt").exists() for z in levels for kind in kinds):
            continue
        an, meta = load_ixi(sid, IXI, prepared=args.prepared, voxel=(args.voxel_mm * 1e-3,) * 3, residual=True)
        for kind in kinds:
            draw = 1 if kind == "les" else int(kind[3:]) if kind.startswith("les") else 0
            rng = np.random.default_rng(np.random.SeedSequence(sid).spawn(16)[15] if draw <= 1 else np.random.SeedSequence([sid, draw]))
            an_k, recs = an, []
            if draw:  # lesions at every imaged level, so each slab of this head holds some
                for z in levels:
                    count = int(rng.integers(lo, hi + 1))
                    an_k, more = add_lesions(an_k, rng, count=count, size_mm=(10., 20.), region="any", z_mm=[z * 1e3], z_tolerance_mm=2.)
                    recs += more
            head = DigitalPhantom(an_k, table, 3.0)
            for z in levels:
                dest = args.out / f"slab_{sid:03d}_{round(z * 1e3):02d}_{kind}.pt"
                if dest.exists():
                    continue
                k = round((z - an.origin[2]) / an.voxel[0])
                half = max(round(THICKNESS / 2 / an.voxel[0]), 1)
                ph = head.to_phantom(fov_region=(slice(k - half, k + half + 1), slice(None), slice(None)), spins_per_voxel=spins,
                                     sampling="proportional", jitter="inplane", mt=True)
                mask, centre = lesion_mask(an_k, z, args.n) if draw else (np.zeros((args.n, args.n), np.float32), None)
                if centre is None:
                    _, centre = lesion_mask(an, z, args.n) if "lesion" in an.fractions else (None, None)
                # centre the field of view on the head's box and put the slab at z = 0
                head_mask = (sum(v for kk, v in an.fractions.items() if kk != "background") > .5).numpy()
                ys, xs = np.nonzero(head_mask[k - half:k + half + 1].any(0))
                cx = an.origin[0] + (xs.min() + xs.max()) / 2 * an.voxel[2]
                cy = an.origin[1] + (ys.min() + ys.max()) / 2 * an.voxel[1]
                ph.pos = ph.pos - torch.tensor([cx, cy, z], dtype=ph.pos.dtype)
                ph = ph.to(torch.float32)
                lesion_px = float((mask > .5).sum())
                label = int(bool(draw) and lesion_px >= 4)
                torch.save({"phantom": ph, "label": label, "lesion_px": lesion_px, "lesion_mask": torch.from_numpy(mask),
                            "meta": {"subject": sid, "z_mm": z * 1e3, "kind": kind, "lesions": recs, "spins": len(ph), "voxel_mm": args.voxel_mm,
                                     "lesion_t2": args.lesion_t2, "lesion_pd": args.lesion_pd}}, dest)
                index[dest.name] = {"file": dest.name, "subject": sid, "z_mm": z * 1e3, "kind": kind, "label": label,
                                    "lesion_px": lesion_px, "split": "test" if sid in test else "train", "spins": len(ph)}
                print(f"{dest.name}: {len(ph):,} spins, lesion px {lesion_px:.0f}", flush=True)
        index_path.write_text(json.dumps(sorted(index.values(), key=lambda r: r["file"]), indent=1))
    print(f"{len(index)} slabs, {sum(i['label'] for i in index.values())} with a lesion in the slab; "
          f"{sum(i['split'] == 'test' for i in index.values())} test")


if __name__ == "__main__":
    main()
