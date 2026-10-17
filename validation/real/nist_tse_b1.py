"""Benchmark 2 follow-up: is the remaining TSE error B1+?

Scans a global B1+ scale over 0.60-1.40 (one simulation per value; each sphere's signal
depends only on its own B1+, so the scan gives every sphere's error curve at once), then
compares B1+ models fitted on the even spheres and scored on the odd ones: global scale,
smooth maps, and the per-sphere optimum (an upper bound, not a prediction). Echo shapes only
constrain |B1+ - 1|, which the results reflect.

    uv run python validation/real/nist_tse_b1.py scan  out.npz    # ~17 min on an M4
    uv run python validation/real/nist_tse_b1.py analyse out.npz
"""

import sys
import time

import nist_tse as t
import numpy as np
import torch
from nist import CENTRE, PIXEL, coarsen_rf, phantom, roi_means
from scipy.interpolate import CubicSpline
from scipy.ndimage import center_of_mass
from scipy.optimize import minimize

from blochparty.pulseq import read_pulseq


def scan(path):
    rois = t.sphere_rois()
    real = roi_means(t.real_images(), rois)
    rad = [t.sphere_radius_t1_plane()] * 14
    seq = coarsen_rf(t.last_trs(read_pulseq(t.SEQ), 128 + 1 + 8).to(dtype=torch.float32), 4)
    ph, _ = phantom(rois, rad, t.T1_REF, t.T2_REF, t.imaged_slice(), n_slice=32, sub=1, sub_read=2)
    b1s = np.round(np.arange(0.60, 1.401, 0.05), 3)
    sims = []
    for b in b1s:
        t0 = time.perf_counter()
        sims.append(t.simulate(ph, seq, rois, b1=float(b)))
        print(f"B1 {b:.2f} done in {time.perf_counter() - t0:.0f} s", flush=True)
    np.savez(path, b1=b1s, sims=np.array(sims), real=real)


def analyse(path):
    data = np.load(path)
    b1, sims, real = data["b1"], data["sims"], data["real"]  # (B,), (B, 23, 14), (23, 14)

    def err(s, r):
        a = (s * r).sum(0) / (s * s).sum(0)
        return np.sqrt(((a * s - r) ** 2).sum(0) / (r * r).sum(0))

    E = np.array([err(s, real) for s in sims])  # (B, 14)
    curves = [CubicSpline(b1, E[:, k]) for k in range(14)]
    pos = np.array(
        [[(r - CENTRE) * PIXEL, (c - CENTRE) * PIXEL] for r, c in (center_of_mass(m) for m in t.sphere_rois())]
    )
    even, odd, S = [1, 3, 5, 7, 9], [0, 2, 4, 6, 8], list(range(10))
    fine = np.linspace(b1[0], b1[-1], 801)

    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    nom = E[np.argmin(abs(b1 - 1.0))]
    best = np.array([fine[np.argmin(curves[k](fine))] for k in range(14)])
    print("per-sphere best B1+  :", best[:10])
    print("  (error there, %)   :", 100 * np.array([curves[k](best[k]) for k in range(10)]))
    print("nominal B1+ error (%):", 100 * nom[:10])
    print("sphere (y, z) [mm]   :", np.round(pos[:10] * 1e3).astype(int).tolist())

    def map_value(c, p):  # 1 + c0 + c1 y + c2 z + c3 (y^2 + z^2), y, z in units of 50 mm
        y, zz = p[:, 0] / 0.05, p[:, 1] / 0.05
        return 1 + c[0] + c[1] * y + c[2] * zz + c[3] * (y**2 + zz**2)

    def score(vals, idx):
        return np.mean([curves[k](np.clip(v, b1[0], b1[-1])) for k, v in zip(idx, vals)])

    rows = {"nominal": (nom[odd].mean(), nom[S].mean())}
    g_fit = fine[np.argmin([score([v] * 5, even) for v in fine])]
    rows[f"global B1+ {g_fit:.3f} (fit even)"] = (score([g_fit] * 5, odd), score([g_fit] * 10, S))
    for name, nc in (("map: offset + linear", 3), ("map: offset + linear + quadratic", 4)):
        best_c = None
        for start in np.random.default_rng(0).uniform(-0.3, 0.3, (200, nc)):
            r = minimize(
                lambda c: score(map_value(np.r_[c, np.zeros(4 - nc)], pos[even]), even), start, method="Nelder-Mead"
            )
            if best_c is None or r.fun < best_c[0]:
                best_c = (r.fun, np.r_[r.x, np.zeros(4 - nc)])
        c = best_c[1]
        rows[name] = (score(map_value(c, pos[odd]), odd), score(map_value(c, pos[S]), S))
        print(f"{name}: coefficients {np.round(c, 3)}, B1+ at spheres 1-10 {map_value(c, pos[S])}")
    rows["per-sphere best (upper bound, not a prediction)"] = (
        np.mean([curves[k](best[k]) for k in odd]),
        np.mean([curves[k](best[k]) for k in S]),
    )
    print("\n| B1+ model | held-out odd spheres 1–9 | all spheres 1–10 |\n|---|---|---|")
    for k, (o, a) in rows.items():
        print(f"| {k} | {100 * o:.2f} % | {100 * a:.2f} % |")


if __name__ == "__main__":
    {"scan": scan, "analyse": analyse}[sys.argv[1]](sys.argv[2])
