"""Seed-paired comparison of lesion_design.py arms across runs, with intervals from resampling test subjects:

    uv run python validation/design/paired_report.py --slabs ~/data/slabs ~/data/design10_split/*/results.json

Every arm's nets are matched by seed (``runs[i]["seed"]``; arms G and L of one job share initialisation, order and
noise), so "arm minus G" is a mean of per-seed differences. The interval resamples the test split's subjects with all
their slabs (slabs of one head are not independent) and recomputes each arm's mean-of-nets AUC on the resample, from
the test scores the runs keep. Where a learned design's ``_tetr`` and ``_perline`` parts (split_design.py) ran too,
it also prints the split: the TE/TR part, the per-line part, the full design and their interaction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from blochparty.design import auc


def collect(files):
    runs = {}
    for f in files:
        for arm, v in json.loads(Path(f).read_text()).items():
            if isinstance(v, dict) and "runs" in v:
                for x in v["runs"]:
                    runs.setdefault(arm, {})[x["seed"]] = x
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument(
        "--slabs", type=Path, required=True, help="the slab folder (its index.json gives labels and subjects)"
    )
    ap.add_argument("--min-lesion-px", type=float, default=4.0)
    ap.add_argument("--baseline", default="G_gre_rss")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    index = json.loads((args.slabs / "index.json").read_text())
    test = [r for r in index if not (0 < r["lesion_px"] < args.min_lesion_px) and r["split"] == "test"]
    y = np.array([r["label"] for r in test])
    subj = np.array([r["subject"] for r in test])
    runs = collect(args.results)
    seeds = sorted(runs[args.baseline])
    arms = [a for a in runs if sorted(runs[a]) == seeds]
    M = {a: np.stack([np.array(runs[a][s]["test_scores"]) for s in seeds]) for a in arms}
    A = {a: np.array([runs[a][s]["auc"] for s in seeds]) for a in arms}
    print(f"{len(seeds)} seeds {seeds}; {len(test)} test slabs, {int(y.sum())} with lesions, {len(set(subj))} subjects")
    rng = np.random.default_rng(0)
    people = np.array(sorted(set(subj)))
    rows = {p: np.nonzero(subj == p)[0] for p in people}
    boot = []
    for _ in range(args.boot):
        idx = np.concatenate([rows[p] for p in rng.choice(people, len(people))])
        if y[idx].min() != y[idx].max():
            boot.append({a: np.mean([auc(y[idx], m[idx]) for m in M[a]]) for a in arms})

    def effect(name, plus, minus):
        d = sum(A[a] for a in plus) - sum(A[a] for a in minus)
        b = np.array([sum(x[a] for a in plus) - sum(x[a] for a in minus) for x in boot])
        lo, hi = np.percentile(b, [2.5, 97.5])
        print(
            f"  {name:34s} {d.mean():+.3f}  positive in {int((d > 0).sum())}/{len(d)} seeds  subjects 95 % [{lo:+.3f}, {hi:+.3f}]"
        )

    print("test AUC, mean over the seeds' nets:")
    for a in sorted(arms, key=lambda a: A[a].mean()):
        acc = np.mean([runs[a][s]["accuracy"] for s in seeds])
        tr = np.mean([runs[a][s]["train_auc"] for s in seeds])
        print(
            f"  {a:34s} {A[a].mean():.3f} (sd {A[a].std(ddof=1):.3f}, {A[a].min():.3f}-{A[a].max():.3f})  accuracy {acc:.3f}  train {tr:.3f}"
        )
    print(f"minus {args.baseline}, paired by seed:")
    for a in arms:
        if a != args.baseline:
            effect(a, [a], [args.baseline])
    g = args.baseline
    for full in [a for a in arms if f"{a}_tetr" in A and f"{a}_perline" in A]:
        t, p = f"{full}_tetr", f"{full}_perline"
        print(f"split of {full}:")
        effect("TE/TR part (T - G)", [t], [g])
        effect("per-line part (P - G)", [p], [g])
        effect("full design (L - G)", [full], [g])
        effect("interaction (L - T - P + G)", [full, g], [t, p])


if __name__ == "__main__":
    main()
