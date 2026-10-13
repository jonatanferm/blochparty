"""Where the real / simulated gap sits: diagnostics on a scored run's images (score.py's examples.npz).

    uv run python validation/realism/gap.py features  --score ~/data/score_r27   # per-image features, real vs sim
    uv run python validation/realism/gap.py polar     --score ~/data/score_r27   # mean image in head polar coordinates
    uv run python validation/realism/gap.py structure --score ~/data/score_r27   # interior structure power, CSF components
    uv run python validation/realism/gap.py controls  --score ~/data/score_r27   # the benchmark's own resolution

features: head area, anterior-posterior length and left-right width [px], centroid row / column (rows run posterior
to anterior), interior quartiles and p99, white-matter mode, CSF-bright share (> 2.5 x the mode), scalp (outer 6 px),
skull-band minimum (3-9 px), posterior and anterior scalp; for each: real and simulated mean +- SD, the AUC of the
feature alone (> 0.5: larger in the simulations) and its Spearman correlation with the CNN score within each set (what
the classifier's verdict tracks).  polar: each image resampled on rays from the head centroid, radius in units of the
head radius along that ray; the mean real and simulated maps by sector (posterior, right, anterior, left) and radius
band, and their ratio (saved as polar.npz / polar.png next to the score).  structure: power of the interior (beyond
10 px) by radial band of Nyquist relative to the real mean, CSF-bright components 8-25 px under the edge, grouped by
CSF share (real) and CNN score (simulated).  controls: real vs real split by scanner model and by echo spacing, real vs
simulated on one scanner model only, and the real vs simulated AUC under other CNN seeds (needs the real cache and the
fastMRI manifest: run on the machine that scored).

2026-10-13 on r27 (docs/JOURNAL.md): the head centroid sits 2.8 px further anterior in the simulations (AUC 0.83), the
posterior / anterior scalp ratio is 1.48 against the real 1.03 (the receive ring), the interior p99 is above the
scalp's (1.04 vs 0.95: the scalp sets the real scale), and the heads the CNN catches carry 1.36-1.40 x the real coarse
structure power at a CSF share real heads reach with 1.3; scanner-model splits of the real set score 0.42-0.56.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

BANDS = [(0.05, 0.15), (0.15, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.8), (0.8, 1.0)]
SECTORS = {"posterior": range(-6, 7), "right": range(12, 25), "anterior": range(30, 43), "left": range(48, 61)}
RADII = [(0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.86), (0.86, 0.9), (0.9, 0.94), (0.94, 0.98), (0.98, 1.02), (1.02, 1.1)]


def head_mask(img):
    m = ndi.binary_opening(img > 0.12, iterations=1)
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == np.argmax(np.bincount(lab[lab > 0]))
    return ndi.binary_fill_holes(m)


def wm_mode(vals):
    hist, edges = np.histogram(vals[(vals > 0.15) & (vals < 0.6)], 45)
    return edges[np.argmax(hist)] + (edges[1] - edges[0]) / 2


def features(img):
    m = head_mask(img)
    yy, xx = np.nonzero(m)
    d = ndi.distance_transform_edt(m)
    vals = img[d > 12]
    wm = wm_mode(vals)
    top, bot = yy.min(), yy.max()
    return {
        "area": m.sum(), "ap_len": bot - top + 1, "lr_wid": xx.max() - xx.min() + 1, "row": yy.mean(), "col": xx.mean(),
        "q25": np.quantile(vals, .25), "q50": np.quantile(vals, .5), "q75": np.quantile(vals, .75), "p99_in": np.quantile(vals, .99),
        "wm_mode": wm, "csf_share": float((vals > 2.5 * wm).mean()),
        "scalp_0_6": img[m & (d <= 6)].mean(), "band_min_3_9": np.quantile(img[m & (d > 3) & (d <= 9)], 0.1),
        "post_scalp": img[top:top + 5][m[top:top + 5]].mean(), "ant_scalp": img[bot - 4:bot + 1][m[bot - 4:bot + 1]].mean(),
    }


def polar(img, na=72, nr=40, rmax=1.15):
    """(na, nr): rays from the head centroid, angle 0 toward the top rows (posterior), radius / head radius on the ray."""
    m = head_mask(img)
    yy, xx = np.nonzero(m)
    cy, cx = yy.mean(), xx.mean()
    out = np.zeros((na, nr))
    r = np.linspace(0, rmax, nr)
    t = np.arange(0, 90, 0.5)
    for a in range(na):
        th = 2 * np.pi * a / na
        dy, dx = -np.cos(th), np.sin(th)
        ys, xs = cy + t * dy, cx + t * dx
        ok = (ys >= 0) & (ys < img.shape[0] - 1) & (xs >= 0) & (xs < img.shape[1] - 1)
        inside = ndi.map_coordinates(m.astype(float), [ys[ok], xs[ok]], order=1) > 0.5
        R = t[ok][np.nonzero(inside)[0].max()] if inside.any() else t[ok][-1]
        out[a] = ndi.map_coordinates(img, [cy + r * R * dy, cx + r * R * dx], order=1)
    return out


def structure(img):
    m = head_mask(img)
    d = ndi.distance_transform_edt(m)
    inner = d > 10
    vals = img[inner]
    w = np.where(inner, img - vals.mean(), 0.0)
    F = np.abs(np.fft.fftshift(np.fft.fft2(w))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(img.shape[0])) * 2
    rr = np.sqrt(f[:, None] ** 2 + f[None] ** 2)
    power = [F[(rr >= a) & (rr < b)].mean() / inner.sum() for a, b in BANDS]
    level = 0.5 * np.quantile(vals, 0.995)
    band = m & (d > 8) & (d <= 25)
    lab, n = ndi.label(band & (img > level))
    sizes = np.bincount(lab[lab > 0]) if n else np.array([0])
    return power, {"n_per_1000": 1000 * n / band.sum(), "mean_size": float(sizes.mean()), "bright_frac": float((img[band] > level).mean())}, \
        float((vals > 2.5 * wm_mode(vals)).mean())


def auc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float((b[:, None] > a[None]).mean() + 0.5 * (b[:, None] == a[None]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("features", "polar", "structure", "controls"))
    ap.add_argument("--score", type=Path, required=True, help="score.py output directory (examples.npz)")
    ap.add_argument("--fastmri", type=Path, default=None, help="fastMRI directory (controls; default realism.FASTMRI)")
    ap.add_argument("--cache", type=Path, default=Path.home() / "data" / "realism_cache")
    ap.add_argument("--slice", type=int, default=None, help="one slice index only")
    args = ap.parse_args()
    z = np.load(args.score / "examples.npz", allow_pickle=True)
    keep_r = slice(None) if args.slice is None else z["real_slices"] == args.slice
    keep_s = slice(None) if args.slice is None else z["sim_slices"] == args.slice
    real, sim = z["real"][keep_r], z["sim"][keep_s]
    sr, ss = z["real_scores"][keep_r], z["sim_scores"][keep_s]

    if args.mode == "features":
        R, S = [features(i) for i in real], [features(i) for i in sim]
        print(f"{len(R)} real, {len(S)} simulated;  feature: real mean +- sd | sim mean +- sd | AUC(sim > real) | "
              "Spearman with the CNN score (real, sim)")
        for k in R[0]:
            a, b = np.array([f[k] for f in R]), np.array([f[k] for f in S])
            print(f"{k:13s} {a.mean():8.3f} +- {a.std():6.3f} | {b.mean():8.3f} +- {b.std():6.3f} | {auc(a, b):5.2f} | "
                  f"{stats.spearmanr(a, sr).statistic:+5.2f} {stats.spearmanr(b, ss).statistic:+5.2f}")
        print("\nper simulated subject: CNN score, area, AP, LR, csf_share, scalp, post / ant scalp")
        for p in sorted(set(z["sim_groups"][keep_s])):
            idx = np.nonzero(z["sim_groups"][keep_s] == p)[0]
            g = lambda k: np.mean([S[i][k] for i in idx])
            print(f"  {p} {ss[idx].mean():.2f}  {g('area'):6.0f} {g('ap_len'):5.1f} {g('lr_wid'):5.1f} {g('csf_share'):.3f} "
                  f"{g('scalp_0_6'):.3f} {g('post_scalp') / g('ant_scalp'):.2f}")

    elif args.mode == "polar":
        pr, ps = np.stack([polar(i) for i in real]), np.stack([polar(i) for i in sim])
        mr, ms = pr.mean(0), ps.mean(0)
        np.savez(args.score / "polar.npz", real=mr, sim=ms, real_sd=pr.std(0), sim_sd=ps.std(0))
        r = np.linspace(0, 1.15, 40)
        print("mean intensity by sector and radius band (r / head radius): real sim sim/real")
        print("band      " + "".join(f"{a:.2f}-{b:.2f}   " for a, b in RADII))
        for name, idx in SECTORS.items():
            idx = [i % 72 for i in idx]
            row = ""
            for a, b in RADII:
                k = (r >= a) & (r < b)
                vr, vs = mr[idx][:, k].mean(), ms[idx][:, k].mean()
                row += f"{vr:.2f} {vs:.2f} {vs / vr:4.2f}  "
            print(f"{name:10s}" + row)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(1, 3, figsize=(14, 4))
            for a, (m, t) in zip(ax, ((mr, "real"), (ms, "simulated"), (ms / np.maximum(mr, 0.02), "sim / real"))):
                ratio = t == "sim / real"
                im = a.imshow(m.T, origin="lower", aspect="auto", extent=(0, 360, 0, 1.15), vmin=0.6 if ratio else 0,
                              vmax=1.4 if ratio else 0.8, cmap="RdBu_r" if ratio else "gray")
                a.set_title(t); a.set_xlabel("angle (0 posterior, 180 anterior)"); a.set_ylabel("r / head radius")
                plt.colorbar(im, ax=a)
            plt.tight_layout(); plt.savefig(args.score / "polar.png", dpi=110)
        except ImportError:
            pass

    elif args.mode == "structure":
        R, S = [structure(i) for i in real], [structure(i) for i in sim]
        rc, ref = np.array([r[2] for r in R]), np.mean([r[0] for r in R], 0)
        groups = {"real all": R, "real csf > 0.17": [r for r, c in zip(R, rc) if c > 0.17],
                  "real csf < 0.12": [r for r, c in zip(R, rc) if c < 0.12], "sim all": S,
                  "sim score < 0.3": [s for s, c in zip(S, ss) if c < 0.3], "sim score > 0.8": [s for s, c in zip(S, ss) if c > 0.8]}
        print("structure power by band of Nyquist (x the real mean) | CSF components per 1000 px, mean size, bright fraction | csf share")
        print("group              n  " + " ".join(f"{a:.2f}-{b:.2f}" for a, b in BANDS))
        for name, g in groups.items():
            if not g:
                continue
            p = np.mean([x[0] for x in g], 0) / ref
            c = {k: np.mean([x[1][k] for x in g]) for k in g[0][1]}
            print(f"{name:18s} {len(g):3d} " + " ".join(f"{v:9.2f}" for v in p)
                  + f" | {c['n_per_1000']:5.1f} {c['mean_size']:5.1f} {c['bright_frac']:.3f} | {np.mean([x[2] for x in g]):.3f}")

    elif args.mode == "controls":
        import torch
        import realism as RL
        import score as SC
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        root = args.fastmri or RL.FASTMRI
        man = json.loads((root / "manifest.json").read_text())
        man = man["files"] if isinstance(man, dict) and "files" in man else man
        info = {i["name"]: i for i in (man if isinstance(man, list) else [dict(name=k, **v) for k, v in man.items()])}
        slices = sorted(set(z["real_slices"].tolist()))
        items, _ = SC.real_set(args.cache, root, slices)
        model = {n: info[n[5:] + ".h5"]["model"] for n in {i[0] for i in items}}
        esp = {n: info[n[5:] + ".h5"].get("echo_spacing_ms") for n in {i[0] for i in items}}
        main_model = max(set(model.values()), key=list(model.values()).count)
        main_esp = max(set(esp.values()), key=list(esp.values()).count)
        out = []

        def test(name, a, b, seeds=(0, 1)):
            X = np.stack([i[2] for i in a + b]); y = np.array([0] * len(a) + [1] * len(b)); g = np.array([i[0] for i in a + b])
            for seed in seeds:
                s = SC.cnn_scores(X, y, g, epochs=30, device=device, seed=seed)
                res = {"test": name, "seed": seed, "people": [len(set(g[y == 0])), len(set(g[y == 1]))],
                       "cnn_auc": SC.auc(y, s), "cnn_ci": SC.bootstrap(y, s, g)}
                print(json.dumps(res), flush=True)
                out.append(res)

        test(f"real {main_model} vs other models", [i for i in items if model[i[0]] == main_model], [i for i in items if model[i[0]] != main_model])
        test(f"real esp {main_esp} vs other", [i for i in items if esp[i[0]] == main_esp], [i for i in items if esp[i[0]] != main_esp])
        simitems = [(g, s, i) for g, s, i in zip(z["sim_groups"], z["sim_slices"], z["sim"])]
        test(f"real {main_model} only vs simulated", [i for i in items if model[i[0]] == main_model], simitems)
        test("real vs simulated, other CNN seeds", items, simitems, seeds=(1, 2))
        (args.score / "controls.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
