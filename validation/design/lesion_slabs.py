"""Slab phantoms for the lesion-classification design experiment (ROADMAP M7, agentic design): from each prepared IXI
head, slabs at a few MNI levels rendered twice through the lesion stream, with lesions and without, as ready-to-simulate
``Phantom`` objects at 2 mm anatomy voxels (~80k spins for a 5 mm slab over 220 mm), with the label, the lesion pixel
count and the lesion mask on the acquisition grid.

    uv run python validation/design/lesion_slabs.py --prepared validation/anatomy/data/ixi/prepared_v17 --out ~/data/slabs

Writes ``<out>/slab_<subject>_<z>_<les|nol>.pt`` (a dict: phantom fields, label, lesion_px, lesion_mask (n, n), meta) and
``<out>/index.json`` (subject, level, lesions, train / test split by subject: the last ``--test-subjects`` subjects).
The phantoms carry the tissue table's 3 T values, the subject's residual maps and the bound pool (MT) and nothing else of
the realism preset (no deformation, pose, receive or noise: the experiment compares sequences, not realism).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import blochparty as bp
from blochparty import tissuedb
from blochparty.anatomy import DigitalPhantom, add_lesions, load_ixi

HERE = Path(__file__).resolve().parent
IXI = HERE.parents[1] / "validation" / "anatomy" / "data" / "ixi"
FOV, THICKNESS = 0.22, 5e-3


def lesion_mask(anatomy, z_m, n, fov=FOV):
    """The lesion fraction averaged over the slab, on the n x n acquisition grid (rows posterior -> anterior,
    columns the patient's left -> right, as generate.truth_on_grid), from the anatomy's own voxel grid."""
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
    ix = (cx - cols - anatomy.origin[0]) / anatomy.voxel[2]
    return ndimage.map_coordinates(slab, [iy.ravel(), ix.ravel()], order=1, mode="constant", cval=0.).reshape(n, n).astype(np.float32), (cx, cy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, default=IXI / "prepared_v17")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--levels-mm", default="10,20,30", help="MNI z of the slab centres")
    ap.add_argument("--n", type=int, default=64, help="acquisition matrix (for the lesion mask)")
    ap.add_argument("--voxel-mm", type=float, default=2.0, help="anatomy voxel the heads are resampled to")
    ap.add_argument("--spins", default="1,1,1")
    ap.add_argument("--lesions", default="1,2", help="count range per imaged level (with); size 6-15 mm, white matter within 2 mm of the level")
    ap.add_argument("--test-subjects", type=int, default=12)
    ap.add_argument("--subjects", default=None, help="comma-separated ids (default: all prepared)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ids = sorted(int(p.stem[3:]) for p in args.prepared.glob("IXI*.npz"))
    if args.subjects:
        ids = [int(s) for s in args.subjects.split(",")]
    test = set(ids[-args.test_subjects:]) if args.test_subjects > 0 else set()
    table = tissuedb.tissue_table(3.0)
    levels = [float(v) * 1e-3 for v in args.levels_mm.split(",")]
    spins = tuple(int(v) for v in args.spins.split(","))
    lo, hi = (int(v) for v in args.lesions.split(","))
    index = []
    for sid in ids:
        an, meta = load_ixi(sid, IXI, prepared=args.prepared, voxel=(args.voxel_mm * 1e-3,) * 3, residual=True)
        for kind in ("les", "nol"):
            rng = np.random.default_rng(np.random.SeedSequence(sid).spawn(16)[15])
            an_k, recs = an, []
            if kind == "les":  # lesions at every imaged level, so each slab of this head holds some
                for z in levels:
                    count = int(rng.integers(lo, hi + 1))
                    an_k, more = add_lesions(an_k, rng, count=count, size_mm=(6., 15.), region="any", z_mm=[z * 1e3], z_tolerance_mm=2.)
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
                mask, centre = lesion_mask(an_k, z, args.n) if kind == "les" else (np.zeros((args.n, args.n), np.float32), None)
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
                torch.save({"phantom": ph, "label": int(kind == "les" and lesion_px >= 2), "lesion_px": lesion_px, "lesion_mask": torch.from_numpy(mask),
                            "meta": {"subject": sid, "z_mm": z * 1e3, "kind": kind, "lesions": recs, "spins": len(ph), "voxel_mm": args.voxel_mm}}, dest)
                index.append({"file": dest.name, "subject": sid, "z_mm": z * 1e3, "kind": kind, "label": int(kind == "les" and lesion_px >= 2),
                              "lesion_px": lesion_px, "split": "test" if sid in test else "train", "spins": len(ph)})
                print(f"{dest.name}: {len(ph):,} spins, lesion px {lesion_px:.0f}", flush=True)
        (args.out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(index)} slabs, {sum(i['label'] for i in index)} with a lesion in the slab; "
          f"{sum(i['split'] == 'test' for i in index)} test")


if __name__ == "__main__":
    main()
