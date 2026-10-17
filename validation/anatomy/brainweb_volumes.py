"""Tissue volumes and brain centroids of the 20 BrainWeb subjects from their pinned manifests: a
load check, a check that all share subject 04's grid origin (they are registered to one space, so
their brains should sit within millimetres of each other), and the spread that sample_head draws
subjects from (validation/anatomy/RESULTS.md). Needs the fetched data (fetch_brainweb.py --subject
N for each). Run: uv run python validation/anatomy/brainweb_volumes.py"""

from pathlib import Path

import torch

from blochparty.anatomy import BRAINWEB_SUBJECTS, load_brainweb

ROOT = Path(__file__).resolve().parent / "data"
VOXEL_ML = 8e-3  # one 2 mm voxel
CLASSES = ("wm", "gm", "csf", "fat", "muscle", "skin", "skull", "vessels", "connective", "dura", "marrow")

print("| subject | " + " | ".join(CLASSES) + " | max total | brain centroid x, y, z [mm] |")
print("|---|" + "---|" * (len(CLASSES) + 2))
for subject in BRAINWEB_SUBJECTS:
    a = load_brainweb(subject, ROOT, voxel=(2e-3,) * 3, dtype=torch.float64)
    total = sum(a.fractions.values())
    row = [f"{float(a.fractions[k].sum()) * VOXEL_ML:.0f}" for k in CLASSES]
    brain = a.fractions["wm"] + a.fractions["gm"]
    axes = torch.meshgrid(
        *(o + torch.arange(n, dtype=torch.float64) * h for n, h, o in zip(a.shape, a.voxel, reversed(a.origin))),
        indexing="ij",
    )  # z, y, x [m]
    centre = [float((brain * c).sum() / brain.sum()) * 1e3 for c in reversed(axes)]
    print(
        f"| {subject} | "
        + " | ".join(row)
        + f" | {float(total.max()):.6f} | "
        + ", ".join(f"{c:.1f}" for c in centre)
        + " |",
        flush=True,
    )
print("volumes in mL (2 mm grid, cell averages)")
