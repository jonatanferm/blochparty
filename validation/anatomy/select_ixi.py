"""Choose IXI subjects whose heads follow the real benchmark population (ROADMAP M5b item 2).

fastMRI's patients differ from IXI's volunteers in head size (simulated area 0.95 x real on r27) and in how much CSF
their slices show (the CNN's verdict tracks the CSF share; age matching alone did not help, journal 2026-10-13). This
script measures two cheap proxies on each subject's raw T2 (no registration): the head area at the level of the
lateral ventricles' largest section, and the share of CSF-bright voxels (> 2.5 x the white-matter mode) inside the
brain over the levels the realism stack covers (that level and 30 mm up); then picks ``--count`` subjects by
importance sampling so the chosen set's proxy distribution matches the targets (``--area`` and ``--csf``: real mean,
SD; defaults from r27's real slices, gap.py features: area 7129 +- 921 px at 1.72 mm = 211 +- 27 cm^2, CSF share
0.152 +- 0.068), with the ages spread as a tie-break. Writes ``<out>`` (JSON: ids, proxies, weights).

    uv run python validation/anatomy/select_ixi.py --count 64 --out validation/anatomy/data/ixi/selection.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def proxies(t2_path):
    import nibabel as nib
    from scipy import ndimage

    img = nib.load(t2_path)
    v = np.asarray(img.dataobj, dtype=np.float32)
    aff = img.affine
    # RAS: axis of z (superior) and its voxel size, in-plane voxel area
    zax = int(np.argmax(np.abs(aff[2, :3])))
    vox = np.sqrt((aff[:3, :3] ** 2).sum(0))
    dz = vox[zax]
    inplane = np.prod([vox[a] for a in range(3) if a != zax])
    v = np.moveaxis(v, zax, 0)
    if aff[2, zax] < 0:
        v = v[::-1]
    head = v > 0.1 * np.quantile(v, 0.99)
    head = np.stack([ndimage.binary_fill_holes(ndimage.binary_opening(h, iterations=2)) for h in head])
    # the brain: the largest bright-ish region well inside the head (erode 12 mm)
    inner = ndimage.binary_erosion(head, iterations=int(12 / vox.min()))
    vals = v[inner]
    hist, edges = np.histogram(vals[(vals > 0.1 * vals.max()) & (vals < 0.6 * vals.max())], 60)
    wm = edges[np.argmax(hist)] + (edges[1] - edges[0]) / 2
    csf = inner & (v > 2.5 * wm)
    # the level of the lateral ventricles: the slice with the most CSF within 25 mm of the head's centre column
    per = csf.sum((1, 2)).astype(float)
    zc = int(np.argmax(ndimage.uniform_filter1d(per, int(10 / dz) + 1)))
    k = max(1, int(round(30 / dz)))
    return {
        "area_cm2": float(head[zc].sum() * inplane / 100),
        "csf_share": float(csf[zc : zc + k].sum() / max(inner[zc : zc + k].sum(), 1)),
        "level": zc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=HERE / "data" / "ixi")
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--area", default="211,27", help="target head area mean,SD [cm^2]")
    ap.add_argument("--csf", default="0.152,0.068", help="target CSF-bright share mean,SD")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--calibration",
        type=Path,
        default=HERE / "ixi_calibration_r27.json",
        help="JSON {subjects: {id: {area_cm2, csf_share}}} measured on simulated slices of prepared subjects "
        "(gap.py features): a linear fit maps the raw-T2 proxies onto those units before the targets apply",
    )
    args = ap.parse_args()
    cache = args.root / "proxies.json"
    prox = json.loads(cache.read_text()) if cache.exists() else {}
    demo = json.loads((args.root / "demographics.json").read_text())
    for p in sorted((args.root / "T2").glob("IXI*-HH-*-T2.nii.gz")):
        sid = str(int(p.name[3:6]))
        if sid in prox or sid not in demo:
            continue
        try:
            prox[sid] = proxies(p)
        except Exception as e:  # a broken file: skip, say so
            print(f"{p.name}: {e}", flush=True)
            continue
        print(
            f"IXI{int(sid):03d} age {demo[sid]['age']:.0f}: area {prox[sid]['area_cm2']:.0f} cm2, csf {prox[sid]['csf_share']:.3f}",
            flush=True,
        )
        cache.write_text(json.dumps(prox, indent=1))
    ids = sorted(prox, key=int)
    a = np.array([prox[s]["area_cm2"] for s in ids])
    c = np.array([prox[s]["csf_share"] for s in ids])
    ta, tc = (float(x) for x in args.area.split(",")), (float(x) for x in args.csf.split(","))
    ta, tc = tuple(ta), tuple(tc)
    if (
        args.calibration and args.calibration.exists()
    ):  # proxies -> simulated-slice units, fitted on the prepared subjects
        cal = json.loads(args.calibration.read_text())["subjects"]
        both = [s for s in ids if s in cal]
        for name, x, key in (("area", a, "area_cm2"), ("csf", c, "csf_share")):
            px = np.array([prox[s][key] for s in both])
            y = np.array([cal[s][key] for s in both])
            slope, icpt = np.polyfit(px, y, 1)
            r = np.corrcoef(px, y)[0, 1]
            print(f"calibration {name}: realism = {slope:.3f} x proxy + {icpt:.3f} (r {r:.2f}, n {len(both)})")
            if name == "area":
                a = slope * a + icpt
            else:
                c = slope * c + icpt
    print(
        f"{len(ids)} subjects: area {a.mean():.0f} +- {a.std():.0f} (target {ta[0]} +- {ta[1]}), csf {c.mean():.3f} +- {c.std():.3f} "
        f"(target {tc[0]} +- {tc[1]})"
    )

    # importance weights: target density over the pool's own density (Gaussian fits), on both proxies
    def dens(x, mu, sd):
        return np.exp(-0.5 * ((x - mu) / sd) ** 2) / sd

    w = dens(a, *ta) / dens(a, a.mean(), a.std() + 1e-9) * dens(c, *tc) / dens(c, c.mean(), c.std() + 1e-9)
    w /= w.sum()
    rng = np.random.default_rng(args.seed)
    chosen = list(rng.choice(ids, size=min(args.count, len(ids)), replace=False, p=w))
    pick = np.array([ids.index(s) for s in chosen])
    ca, cc = a[pick], c[pick]  # in the calibrated units
    ages = np.array([demo[s]["age"] for s in chosen])
    print(
        f"chosen {len(chosen)}: area {ca.mean():.0f} +- {ca.std():.0f}, csf {cc.mean():.3f} +- {cc.std():.3f}, "
        f"ages {ages.min():.0f}-{ages.max():.0f} (median {np.median(ages):.0f})"
    )
    print("ids:", ",".join(sorted(chosen, key=int)))
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "targets": {"area_cm2": ta, "csf_share": tc},
                    "ids": sorted(chosen, key=int),
                    "proxies": {s: prox[s] for s in chosen},
                    "weights": dict(zip(ids, w.round(5).tolist())),
                },
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
