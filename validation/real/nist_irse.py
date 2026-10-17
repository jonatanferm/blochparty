"""Realism benchmark 1: ISMRM/NIST phantom T1 plane, inversion-recovery spin echo, 3 T.

The real scanner's raw data (10 inversion times, 20 coils, Pulseq sequences; see fetch_data.py)
is compared with blochparty simulating the *same .seq files* on a digital model of the phantom
whose relaxation times come from **independent references**, not from fitting this data:

- T1 of the 14 NiCl2 spheres: CaliberMRI/NIST reference values at 3 T and 20 °C, phantom
  version 1 (serial < 43) or 2 (serial >= 43); the dataset does not say which, so both are
  reported (from the RRSG 2020 analysis code, github.com/rrsg2020/analysis, src/nist.py)
- T2 of those spheres: not tabulated anywhere; derived from T1 with the 3 T NiCl2 relaxivity
  ratio (see nist.py). TE being fixed, T2 only scales each sphere's curve here, apart from
  relaxation during the pulses.

Metric (shape): per sphere, the ROI-mean magnitude over the inversion times, after one
least-squares scale per sphere (receiver gain, coil sensitivity, PD and T2 weighting are all
per-sphere constants here), as a relative RMS error. That isolates the physics of the inversion
recovery itself: pulse shapes, slice profile, timing, B1+.

Models compared: the analytic IR formula the authors fitted (without their free offset), the
Bloch simulation of the real .seq files, and the same with a B1+ scale fitted on half of the
inversion times by gradient descent through the simulator and evaluated on the other half.

    uv run python validation/real/nist_irse.py      -> validation/real/RESULTS.md section, outputs/
"""

from __future__ import annotations

import argparse
import functools
import os
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from nist import (
    PIXEL,
    T1_NICL2,
    WATER_T1,
    WATER_T2,
    N,
    coarsen_rf,
    fill_mask,
    fill_phantom,
    fill_phantom_qmc,
    mc_errors,
    phantom,
    phantom_qmc,
    print_mc_table,
    recon,
    recon_complex,
    roi_means,
    shape_error,
    slice_position,
    sphere_radius,
    t2_of_nicl2,
)
from scipy import ndimage

import blochparty as bp
from blochparty.pulseq import read_pulseq

# BLOCHPARTY_BACKEND=core: the float64 reference (slow at this size); default the GPU (float32: Metal, or CUDA on an
# NVIDIA machine); until 2026-10-09 the fused CPU kernel, with the same tables
engine = functools.partial(bp.simulate, engine=os.environ.get("BLOCHPARTY_BACKEND", "gpu"))

DATA = Path(__file__).with_name("data") / "developer_main_site" / "IRSE_T1_mapping"
TI = [50, 75, 100, 125, 150, 250, 1000, 1500, 2000, 3000]  # [ms]
TR, TE = 4.5, 10e-3  # [s]
T1_REF = T1_NICL2
T2_REF = {ver: t2_of_nicl2(t1) for ver, t1 in T1_NICL2.items()}  # [ms], from the relaxivity ratio


def real_images() -> np.ndarray:
    return np.stack([recon(sio.loadmat(DATA / "acq" / f"T1_mapping_raw_TI{ti}ms.mat")["data"]) for ti in TI])


def sim_images(signal: torch.Tensor) -> np.ndarray:
    """(C, n_adc) time-ordered simulated signal -> image, through the same reconstruction."""
    data = signal.detach().reshape(signal.shape[0], N, N).permute(2, 0, 1).cpu().numpy()
    return recon(data)


def sim_complex_image(signal: torch.Tensor) -> np.ndarray:
    """(1, n_adc) single-coil signal -> (N, N) complex image."""
    return recon_complex(signal.detach().reshape(signal.shape[0], N, N).permute(2, 0, 1).cpu().numpy())


def monte_carlo(
    real,
    rois,
    radii,
    x0,
    seqs,
    t1,
    t2,
    *,
    seeds=4,
    n=256,
    fill_margin=8,
    dense=3,
    sparse=0.25,
    wall=1e-3,
    water=(WATER_T1 * 1e3, WATER_T2 * 1e3),
    real_img=None,
):
    """Independent Sobol seeds of spheres (+ water fill), every inversion time: shape errors of the
    power-debiased magnitude with jackknife SEs over seeds (``nist.MCErrors``), and the seed images
    (seeds, n_TI, N, N) for paired comparisons."""
    imgs = []
    for k in range(seeds):
        t0 = time.perf_counter()
        phs = [phantom_qmc(rois, radii, t1, t2, x0, n=n, seed=k)]
        if fill_margin > 0:
            phs.append(
                fill_phantom_qmc(
                    rois,
                    fill_mask(real_img, rois, max(radii) + wall),
                    radii,
                    wall,
                    x0,
                    margin=fill_margin * PIXEL,
                    t1_ms=water[0],
                    t2_ms=water[1],
                    n=n,
                    dense=dense * PIXEL,
                    sparse=sparse,
                    seed=k,
                )
            )
        imgs.append(np.stack([sim_complex_image(sum(engine(ph, s).signal for ph in phs)) for s in seqs]))
        print(
            f"  seed {k}: {' + '.join(str(len(ph)) for ph in phs)} spins, {time.perf_counter() - t0:.0f} s", flush=True
        )
    imgs = np.array(imgs)  # (seeds, n_TI, N, N)
    return mc_errors(imgs, rois, real), imgs


def sphere_rois() -> list[np.ndarray]:
    """The authors' 14 ROI circles, split and ordered as NIST spheres 1..14 (longest T1 first)."""
    ana = sio.loadmat(DATA / "ana" / "t1_by_sphere_filtered.mat")
    lab, n = ndimage.label(ana["total_circle_map"] > 0)
    assert n == 14, n
    t1_map = sio.loadmat(DATA / "ana" / "T1_map.mat")["T1_map"]
    rois = [lab == i for i in range(1, n + 1)]
    return sorted(rois, key=lambda m: -np.median(t1_map[m]))


def analytic(t1_ms, t2_ms) -> np.ndarray:
    t1 = np.asarray(t1_ms)[None] * 1e-3
    ti = np.asarray(TI)[:, None] * 1e-3
    return np.abs(1 - 2 * np.exp(-ti / t1) + np.exp(-TR / t1)) * np.exp(-TE / (np.asarray(t2_ms)[None] * 1e-3))


def simulate_all(phantoms, seqs, b1=None) -> list[torch.Tensor]:
    """Signals for every sequence of one phantom or the sum of several (spheres + water fill)."""
    phantoms = phantoms if isinstance(phantoms, (list, tuple)) else [phantoms]
    out = []
    for s in seqs:
        total = 0
        for ph in phantoms:
            if b1 is not None:
                ph = bp.Phantom(
                    ph.pos, ph.pd, ph.t1, ph.t2, ph.db0, b1=b1 if b1.numel() == len(ph) else b1[:1].expand(len(ph))
                )
            total = total + engine(ph, s).signal
        out.append(total)
    return out


def torch_roi_means(signals, rois) -> torch.Tensor:
    """Differentiable version of roi_means(sim_images(...)) for a single uniform coil."""
    masks = torch.stack([torch.as_tensor(r) for r in rois]).to(torch.float32)  # (14, N, N)
    out = []
    for s in signals:
        k = s[0].reshape(N, N).T  # (samples, lines)
        im = torch.fft.fftshift(torch.fft.ifft2(k), dim=(-2, -1)).abs()
        out.append((masks * im).sum((1, 2)) / masks.sum((1, 2)))
    return torch.stack(out)


def fit_b1(ph, seqs, real, rois, train) -> float:
    """Global B1+ scale minimising the shape error on the training TIs, by gradient descent
    through the Metal adjoint."""
    b = torch.tensor(1.0, requires_grad=True)
    r = torch.as_tensor(real[train], dtype=torch.float32)
    opt = torch.optim.LBFGS([b], lr=1, max_iter=15, line_search_fn="strong_wolfe")
    tr_seqs = [seqs[i] for i in train]

    def closure():
        opt.zero_grad()
        s = torch_roi_means(simulate_all(ph, tr_seqs, b1=b.expand(len(ph))), rois)
        a = (s * r).sum(0) / (s * s).sum(0)
        loss = (((a * s - r) ** 2).sum(0) / (r * r).sum(0)).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(b)


def plot_curves(real, rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 7, figsize=(17, 5.5), sharex=True)
    ti = np.asarray(TI)
    for k, ax in enumerate(axes.flat):
        ax.plot(ti, real[:, k] / np.abs(real[:, k]).max(), "ko", ms=4, label="scanner")
        for (name, _, curve), style in zip(rows, ("C1--", "C0-")):
            a = (curve[:, k] * real[:, k]).sum() / (curve[:, k] ** 2).sum()
            ax.plot(ti, a * curve[:, k] / np.abs(real[:, k]).max(), style, lw=1.5, label=name)
        ax.set_xscale("log")
        ax.set_title(f"sphere {k + 1}: T1 {T1_REF['v2'][k]:.0f} ms", fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.supxlabel("TI [ms]")
    fig.suptitle("NIST phantom, 3 T IR spin echo: scanner vs. blochparty (same .seq, reference T1/T2, nothing fitted)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-slice", type=int, default=128, help="spin positions across the slice")
    p.add_argument("--sub", type=int, default=2, help="in-plane spins per pixel side")
    p.add_argument("--rf-merge", type=int, default=4, help="merge this many 1 us RF samples (1: exact)")
    p.add_argument("--fit-b1", action="store_true", help="also fit a global B1+ scale on 5 TIs (slow)")
    p.add_argument("--t2", type=float, nargs=14, help="override the spheres' T2 [ms]")
    p.add_argument(
        "--fill-margin",
        type=float,
        default=0,
        help="water simulated within this many pixels of a sphere wall (0: none)",
    )
    p.add_argument("--wall", type=float, default=1.0, help="sphere wall thickness [mm]")
    p.add_argument("--water", type=float, nargs=2, default=[WATER_T1 * 1e3, WATER_T2 * 1e3], help="water T1 T2 [ms]")
    p.add_argument(
        "--mc",
        type=int,
        default=0,
        help="Monte Carlo mode: this many Sobol seeds (power-debiased magnitude, error bars; >= 3)",
    )
    p.add_argument("--n", type=int, default=256, help="--mc: Sobol spins per pixel (a power of two)")
    p.add_argument("--save", type=Path, help="--mc: save every row's seed images (.npz) for later paired comparisons")
    p.add_argument("--versions", nargs="+", default=list(T1_NICL2), help="T1 reference versions to run")
    p.add_argument("--figure", action="store_true", help="write nist_irse_curves.png (the figure RESULTS.md shows)")
    args = p.parse_args(argv)

    real_img = real_images()
    rois = sphere_rois()
    real = roi_means(real_img, rois)
    radii = [sphere_radius(real_img, r) for r in rois]
    print(f"sphere radii [mm]: {np.round(np.array(radii) * 1e3, 2)}")
    seqs = [
        coarsen_rf(read_pulseq(DATA / "acq" / f"irse_T1_TI{ti}.seq").to(dtype=torch.float32), args.rf_merge)
        for ti in TI
    ]
    x0 = slice_position(DATA / "acq" / f"irse_T1_TI{TI[0]}.seq")
    print(f"slice centre x0 = {x0 * 1e3:.2f} mm; {len(seqs[0])} segments per sequence")
    train, test = [0, 2, 4, 6, 8], [1, 3, 5, 7, 9]

    if args.mc:
        plain, rows = [], []
        for ver in args.versions:
            t1, t2 = T1_REF[ver], args.t2 or T2_REF[ver]
            plain.append((f"analytic IR formula, T1 {ver}", shape_error(analytic(t1, t2), real)))
            for margin in [0, args.fill_margin] if args.fill_margin > 0 else [0]:
                print(
                    f"T1 {ver}: Monte Carlo, {args.mc} Sobol seeds x {args.n} spins/pixel, water margin {margin} px",
                    flush=True,
                )
                res, imgs = monte_carlo(
                    real,
                    rois,
                    radii,
                    x0,
                    seqs,
                    t1,
                    t2,
                    seeds=args.mc,
                    n=args.n,
                    fill_margin=margin,
                    wall=args.wall * 1e-3,
                    water=args.water,
                    real_img=real_img,
                )
                rows.append((f"Bloch MC{' + water fill' if margin else ''}, T1 {ver}", res, imgs))
        print_mc_table(rows, rois, real, plain)
        if args.save:
            np.savez_compressed(args.save, names=[r[0] for r in rows], **{f"imgs{i}": r[2] for i, r in enumerate(rows)})
        return rows, None
    rows = []
    for ver, t1 in T1_REF.items():
        t2 = args.t2 or T2_REF[ver]
        rows.append((f"analytic IR formula, T1 {ver}", shape_error(analytic(t1, t2), real), analytic(t1, t2)))
        ph, _ = phantom(rois, radii, t1, t2, x0, n_slice=args.n_slice, sub=args.sub)
        phs = [ph]
        if args.fill_margin > 0:
            mask = fill_mask(real_img, rois, max(radii) + args.wall * 1e-3)
            phs.append(
                fill_phantom(
                    rois,
                    mask,
                    radii,
                    args.wall * 1e-3,
                    x0,
                    margin=args.fill_margin * PIXEL,
                    t1_ms=args.water[0],
                    t2_ms=args.water[1],
                    n_slice=args.n_slice,
                    sub=args.sub,
                )
            )
        t0 = time.perf_counter()
        sim = roi_means(np.stack([sim_images(s) for s in simulate_all(phs, seqs)]), rois)
        print(
            f"{ver}: {' + '.join(str(len(q)) for q in phs)} spins, 10 sequences simulated in "
            f"{time.perf_counter() - t0:.0f} s",
            flush=True,
        )
        label = f" + water fill ({args.fill_margin:g} px)" if args.fill_margin > 0 else ""
        rows.append((f"Bloch on the scanner's .seq{label}, T1 {ver}", shape_error(sim, real), sim))
        if args.fit_b1:
            b1 = fit_b1(ph, seqs, real, rois, train)
            sim_b = roi_means(
                np.stack([sim_images(s) for s in simulate_all(ph, seqs, b1=torch.full((len(ph),), b1))]), rois
            )
            rows.append(
                (f"Bloch + B1 {b1:.3f} (fit on 5 TIs, scored on 5), T1 {ver}", shape_error(sim_b, real, test), sim_b)
            )

    lines = ["| model | spheres 1–10 | 1–14 | per sphere 1..14 [%] |", "|---|---|---|---|"]
    for name, err, _ in rows:
        lines.append(
            f"| {name} | {100 * err[:10].mean():.2f} % | {100 * err.mean():.2f} % | "
            f"{' '.join(f'{v:.1f}' for v in 100 * err)} |"
        )
    table = "\n".join(lines)
    print("\n" + table)
    if args.figure:
        plot_curves(
            real, [r for r in rows if r[0].endswith("v2")][:2], Path(__file__).with_name("nist_irse_curves.png")
        )
    return rows, table


if __name__ == "__main__":
    main()
