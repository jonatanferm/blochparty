"""Benchmark 3: blochparty against a radial single-shot IR-bSSFP of the NIST T2 plane
(Scholand et al. 2023; data and protocol in ``scholand.py``).

Steps (cached in data/scholand/cache; the NUFFT stages run in ``scholand_recon.py`` in their
own processes, because finufft and torch cannot share one):

1. gold standards (IR-SE T1, SE T2 per sphere, refitted), the B1 map, delays / coil maps / noise;
2. a temporal basis from simulated signals (union of non-selective and slice-selective
   dictionaries), and the low-rank reconstruction of the measured data: per-sphere curves;
3. the same acquisition from a digital phantom with known curves through the same
   reconstruction: the bias of the method itself;
4. per-sphere (T1, T2) fits with each model variant, on the measured and the digital data,
   against the gold standards (and NIST's values for the short-T2 spheres, which the SE gold
   standard, first echo 15 ms, cannot measure).

    uv run python validation/real/fetch_scholand.py
    uv run python validation/real/scholand_irbssfp.py [--K 12]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import nist  # noqa: E402
import scholand as s  # noqa: E402

CACHE = s.CACHE


def recon_stage(*args):
    subprocess.run([sys.executable, str(HERE / "scholand_recon.py"), *map(str, args)], check=True)


def cached(name: str, make):
    path = CACHE / name
    if not path.exists():
        make(path)
    return np.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=12)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    gold = cached("gold.npz", lambda p: recon_stage("gold"))
    b1 = cached("b1.npz", lambda p: recon_stage("b1"))
    cached("prepared.npz", lambda p: recon_stage("prepare"))
    cached("basis.npz", lambda p: np.savez(p, basis=s.basis()))
    real = cached(f"real_K{args.K}.npz", lambda p: recon_stage("recon", CACHE / "basis.npz", args.K, p))

    t1g, t2g = gold["t1"][:, 0], gold["t2"][:, 0]
    t2n = np.array(nist.T2_MNCL2) * 1e-3
    b1s = b1["sphere_fit"]
    print(f"B1 (smooth fit to the background, rms {b1['fit_rms']:.3f}):", np.round(b1s, 3))
    print(f"real data: CG residual {float(real['cg_residual']):.4f}, data residual {float(real['data_residual']):.3f}")

    # digital phantom: spheres at the gold-standard T1 and (1-10) gold / (11-14) NIST T2,
    # background water; amplitudes from the measured curves
    t2_true = np.where(np.arange(14) < 10, t2g, t2n)

    def make_truth(p):
        sph = s.signals(t1g, t2_true, b1s)
        bgc = s.signals([3.0], [2.0], [1.0], slice_select=False)
        cur = np.concatenate([sph, bgc])
        m = real["curves"]
        amps = np.array([np.vdot(c[20:], mm[20:]) / np.vdot(c[20:], c[20:]) for c, mm in zip(cur, m)])
        np.savez(p, curves=cur, amps=amps)

    truth = cached("digital_truth.npz", make_truth)
    cached("digital_data.npz", lambda p: recon_stage("digital", CACHE / "digital_truth.npz", p))
    dig = cached(f"digital_K{args.K}.npz",
                 lambda p: recon_stage("recon", CACHE / "basis.npz", args.K, p, CACHE / "digital_data.npz"))

    W = {"0.1-1 s": slice(20, 205), "1-4.9 s": slice(205, 1000)}
    print("\nreconstruction error on the digital phantom (rms / rms truth):")
    for j in range(14):
        t = truth["amps"][j] * truth["curves"][j]
        print(f"  sphere {j + 1:2d}: " + "  ".join(
            f"{w} {100 * np.linalg.norm(dig['curves'][j][sl] - t[sl]) / np.linalg.norm(t[sl]):5.1f}%" for w, sl in W.items()))

    centre = np.where(np.arange(14) < 10, t2g, t2n)
    rows = {}
    for name, curves, model, b1v in (("digital, full", dig["curves"][:14], "full", b1s),
                                     ("measured, full", real["curves"][:14], "full", b1s),
                                     ("measured, full, B1 = 1", real["curves"][:14], "full", np.ones(14)),
                                     ("measured, perfect inversion", real["curves"][:14], "perfect_inv", b1s),
                                     ("measured, no slice profile", real["curves"][:14], "no_slice", b1s),
                                     ("measured, simple", real["curves"][:14], "simple", b1s)):
        ref = cached(f"fit_{name.replace(', ', '_').replace(' ', '').replace('=', '')}_K{args.K}.npz",
                     lambda p: np.savez(p, **dict(zip(("best", "edge"), s.fit(curves, t1g, centre, b1v, model)))))
        rows[name] = ref
    truth_t2 = t2_true
    print("\nfitted T1 / T2, % from the reference (digital: from its truth; measured: gold standard 1-10, "
          "NIST T2 11-14); * = optimum on the grid edge")
    print("sphere  ref T1   ref T2 | " + " | ".join(f"{n[:26]:>26s}" for n in rows))
    for j in range(14):
        line = f"{j + 1:5d} {t1g[j] * 1e3:7.0f} {t2_true[j] * 1e3:7.1f} |"
        for n, r in rows.items():
            T1, T2, res = r["best"][j]
            line += f" {100 * (T1 / t1g[j] - 1):+6.1f} {100 * (T2 / truth_t2[j] - 1):+6.1f} ({100 * res:3.1f}%){'*' if r['edge'][j] else ' '} |"
        print(line)


if __name__ == "__main__":
    main()
