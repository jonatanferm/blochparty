"""Ten sampled heads, three contrasts at 3 T, as ISMRMRD raw data (ROADMAP M5).

Every seed is a new head (``sample_head``): a BrainWeb subject, smoothly deformed, with its own
tissue draws and texture. Each is imaged with scanner-playable sequences (ramps, windowed-sinc
slice selection, RF or gradient spoiling; RF on a 10 us raster, which changes the signal by
~7e-5 relative to 1 us) through an 8-channel birdcage receive array with thermal noise:

* T1w: 2D spoiled GRE, TR 250 ms, TE 2.5 ms, 70 deg;
* T2w: 2D TSE, TR 4 s, ETL 16, ESP 10 ms (TE 90 ms);
* T2*w: 2D spoiled GRE, TR 600 ms, TE 20 ms, 20 deg.

Each acquisition is written with ``io.write_dataset`` (signal, trajectory, the sequence and its
.seq file, the head's draws, seeds, engine, versions); the RSS images go to one figure.

    uv run python validation/anatomy/fetch_brainweb.py --subject N   # each of the 20, ~1.4 GB each
    uv run python examples/sampled_heads.py [--heads 10] [--out outputs/sampled_heads] [--replot]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import blochparty as bp
from blochparty.io import write_dataset
from blochparty.sequence import System

ROOT = Path(__file__).resolve().parents[1] / "validation" / "anatomy" / "data"
SYSTEM = System(rf_raster=1e-5)


def protocols(n: int, fov: float, thickness: float) -> dict:
    common = dict(ramps=True, slice_thickness=thickness, system=SYSTEM)
    # GRE echo times 5 us past round numbers: the echo is the centre of sample n/2, half a dwell
    # off the 10 us gradient raster, and only on-raster blocks can be written to Pulseq
    return {
        "T1w GRE (TR 250, TE 2.5 ms, 70 deg)": bp.gre_2d(n, fov, flip_deg=70.0, tr=0.25, te=2.505e-3, n_dummy=8,
                                                       spoiling="rf", **common),
        "T2w TSE (TR 4 s, ETL 16, TE 90 ms)": bp.tse_2d(n, fov, etl=16, esp=10e-3, tr=4.0, n_dummy=1,
                                                       spoiling="gradient", **common),
        "T2*w GRE (TR 600, TE 20 ms, 20 deg)": bp.gre_2d(n, fov, flip_deg=20.0, tr=0.6, te=20.005e-3, n_dummy=4,
                                                        spoiling="rf", **common),
    }


def simulate(ph: bp.Phantom, seq: bp.Sequence, coils: torch.Tensor) -> tuple[torch.Tensor, str]:
    try:
        from blochparty import metal

        if metal.available():
            out = metal.simulate_metal(ph.to(torch.float32), seq.to(torch.float32), coils.to(torch.complex64))
            return out.signal.to(torch.complex128), "metal"
    except ImportError:
        pass
    return bp.simulate(ph, seq, coils).signal, "core"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--heads", type=int, default=10)
    ap.add_argument("--first", type=int, default=0, help="first seed")
    ap.add_argument("--subject", type=int, default=None, help="fix the BrainWeb subject (default: drawn)")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--slab", type=int, default=4, help="spins within +-slab mm of the 5 mm slice")
    ap.add_argument("--snr", type=float, default=40.0, help="image SNR of bright tissue")
    ap.add_argument("--out", type=Path, default=Path("outputs/sampled_heads"))
    ap.add_argument("--figure", type=Path, default=Path("validation/anatomy/sampled_heads.png"))
    ap.add_argument("--replot", action="store_true", help="only redraw the figure from the datasets in --out")
    args = ap.parse_args()
    if args.replot:
        return replot(args.out, args.figure, {tag(n): n for n in protocols(args.n, 0.24, 5e-3)})
    args.out.mkdir(parents=True, exist_ok=True)
    fov, thickness = 0.24, 5e-3
    seqs = protocols(args.n, fov, thickness)
    images, log = {}, []
    for seed in range(args.first, args.first + args.heads):
        t0 = time.perf_counter()
        head = bp.sample_head(seed, subject=args.subject, root=args.root, voxel=(1e-3,) * 3)
        k = round(-head.anatomy.origin[2] / head.anatomy.voxel[0])  # the slice is at z = 0
        ph = head.to_phantom(fov_region=(slice(k - args.slab, k + args.slab + 1), slice(None), slice(None)),
                             spins_per_voxel=(2, 2, 2))
        coils = bp.birdcage_sensitivities(ph.pos, 8)
        entry = {"seed": seed, "subject": head.meta["subject"], "spins": len(ph),
                 "jacobian": head.meta["deformation"]["jacobian"], "setup_s": time.perf_counter() - t0}
        print(f"head {seed}: subject {head.meta['subject']}, {len(ph):,} spins ({entry['setup_s']:.0f} s)", flush=True)
        for name, seq in seqs.items():
            t = time.perf_counter()
            clean, engine = simulate(ph, seq, coils)
            level = float(np.percentile(bp.rss(bp.cartesian_recon(clean, seq)).numpy(), 99))
            sigma = level * args.n / args.snr  # image noise sigma / n for the centred iFFT
            noise_seed = 1000 * seed + len(images)
            signal = bp.add_noise(clean, sigma, generator=torch.Generator().manual_seed(noise_seed))
            write_dataset(args.out / f"head{seed:02d}_{tag(name)}.h5", signal, seq, phantom=head, seed=seed,
                          engine=engine, coils={"model": "birdcage", "n": 8},
                          noise={"sigma": sigma, "seed": noise_seed})
            images[seed, name] = bp.rss(bp.cartesian_recon(signal, seq)).numpy()
            entry[tag(name) + "_s"] = time.perf_counter() - t
            print(f"  {name}: {entry[tag(name) + '_s']:.0f} s ({engine})", flush=True)
        log.append(entry)
    (args.out / "log.json").write_text(json.dumps(log, indent=1))
    plot(images, list(seqs), {e["seed"]: e["subject"] for e in log}, args.figure)
    print(f"wrote {args.figure} and {len(images)} datasets in {args.out}")


def tag(name: str) -> str:
    return name.split()[0].replace("*", "s")


def replot(out: Path, figure: Path, names_by_tag: dict):
    """The figure again from the written datasets (read back with ``io.read_dataset``)."""
    from blochparty.io import read_dataset

    images, subjects = {}, {}
    for path in sorted(out.glob("head*_*.h5")):
        d = read_dataset(path)
        seed, name = d["provenance"]["seed"], names_by_tag[path.stem.split("_")[1]]
        images[seed, name] = bp.rss(bp.cartesian_recon(d["signal"].to(torch.complex128), d["sequence"])).numpy()
        subjects[seed] = d["phantom"]["meta"]["subject"]
    plot(images, [n for n in names_by_tag.values() if any(n == m for _, m in images)], subjects, figure)
    print(f"wrote {figure} from {len(images)} datasets in {out}")


def plot(images: dict, names: list, subjects: dict, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = sorted({s for s, _ in images})
    fig, ax = plt.subplots(len(names), len(seeds), figsize=(1.9 * len(seeds), 2.1 * len(names)), squeeze=False)
    for j, name in enumerate(names):
        for i, seed in enumerate(seeds):
            img = images[seed, name]
            ax[j, i].imshow(img[::-1], cmap="gray", vmax=np.percentile(img, 99.5))
            ax[j, i].set_xticks([]), ax[j, i].set_yticks([])
            if j == 0:
                ax[j, i].set_title(f"seed {seed} (subject {subjects[seed]})", fontsize=7)
        ax[j, 0].set_ylabel(name.replace(" (", "\n("), fontsize=7)
    fig.suptitle("sample_head: ten heads at 3 T (deformation, tissue draws, texture, chi -> B0 + shim, "
                 "8 coils, noise)", fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=90)

if __name__ == "__main__":
    main()
