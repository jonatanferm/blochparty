"""Realistic anatomy end to end: a BrainWeb head at 3 T through blochparty.

The full head's susceptibility field (chi -> B0, air around the head) is computed once and
shimmed to second order over the brain; a 2 mm axial slab becomes partial-volume sub-voxel
spins with within-tissue texture, and three contrasts are simulated on the GPU (the PyTorch
core without one, slowly): T1-weighted spoiled GRE, T2-weighted TSE, and a long-TE GRE.

    uv run python validation/anatomy/fetch_brainweb.py            # once, ~1.4 GB, not in git
    uv run python examples/anatomy_brain.py [--z 60] [--out outputs/anatomy_brain.png]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import blochparty as bp

ROOT = Path(__file__).resolve().parents[1] / "validation" / "anatomy" / "data"


def run(phantom: bp.Phantom, seq: bp.Sequence) -> np.ndarray:
    if bp.engines.available("gpu"):
        s = bp.simulate(phantom.to(torch.float32), seq.to(torch.float32), engine="gpu").signal.to(torch.complex128)
        return bp.cartesian_recon(s, seq)[0].abs().numpy()
    return bp.cartesian_recon(bp.simulate(phantom, seq).signal, seq)[0].abs().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--subject", type=int, default=4)
    ap.add_argument("--z", type=int, default=60, help="first slice of the 2 mm slab (1 mm grid)")
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--out", type=Path, default=Path("outputs/anatomy_brain.png"))
    args = ap.parse_args()
    t0 = time.perf_counter()
    head = bp.load_brainweb(args.subject, args.root, voxel=(1e-3,) * 3)
    fields = bp.anatomy_field_maps(head, field_T=3.0)  # tissue susceptibility field, scanner shim
    ph = bp.to_phantom(head, field_T=3.0, fov_region=(slice(args.z, args.z + 2), slice(None), slice(None)),
                       spins_per_voxel=(2, 2, 2), field_maps=fields,
                       variability={"t1": 0.05, "t2": 0.05, "pd": 0.02, "length": 0.01})
    ph.pos = ph.pos - ph.pos.mean(0)  # centre the slab in the field of view
    print(f"{len(ph):,} spins, B0 {float(ph.db0.min()):.0f} to {float(ph.db0.max()):.0f} Hz "
          f"({time.perf_counter() - t0:.1f} s)")
    fov = 0.24
    protocols = {
        "T1w spoiled GRE (TR 20, TE 4.6 ms, 30 deg)": bp.gre_2d(args.n, fov, flip_deg=30.0, tr=20e-3, te=4.6e-3,
                                                              n_dummy=60),
        "T2w TSE (TR 4 s, ETL 8, ESP 12 ms)": bp.tse_2d(args.n, fov, etl=8, esp=12e-3, tr=4.0, n_dummy=1),
        "GRE TE 25 ms (TR 40 ms, 20 deg)": bp.gre_2d(args.n, fov, flip_deg=20.0, tr=40e-3, te=25e-3, n_dummy=40),
    }
    images = {}
    for name, seq in protocols.items():
        t = time.perf_counter()
        images[name] = run(ph, seq)
        print(f"{name}: {time.perf_counter() - t:.1f} s")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, len(images), figsize=(5 * len(images), 5.4))
    for a, (name, img) in zip(ax, images.items()):
        a.imshow(img[::-1], cmap="gray", vmax=np.percentile(img, 99.5))
        a.set_title(name, fontsize=10)
        a.axis("off")
    fig.suptitle(f"BrainWeb subject {args.subject:02d}, 3 T, slab z = {args.z}-{args.z + 1} mm: chi -> B0 + 2nd-order "
                 "shim, partial volume, tissue texture", fontsize=10)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=80)
    print(f"wrote {args.out} ({time.perf_counter() - t0:.1f} s total)")


if __name__ == "__main__":
    main()
