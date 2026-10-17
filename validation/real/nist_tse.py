"""Realism benchmark 2: ISMRM/NIST phantom T2 plane, multi-echo spin echo (CPMG), 3 T.

The scanner played 256 TRs of a 23-echo train (echo spacing 7 ms, 90° excitation, 180°
windowed-sinc refocusing, crushers on the slice and readout axes), one phase-encode line per
TR, the first 128 TRs discarded. Imperfect refocusing (slice profile, B1+) sends signal through
stimulated-echo pathways: in the data, echo 2 is ~20 % brighter than echo 1 in long-T2 spheres,
which is why the dataset's authors dropped echoes 1-3 before fitting T2.

blochparty simulates the scanner's .seq on a digital T2 plane with reference values only:
T2 of the MnCl2 spheres from the phantom manual (Jiang et al. 2016 for spheres 1 and 5), T1
from the 3 T MnCl2 relaxivity ratio (nist.py). Metric (shape): per sphere, the ROI-mean
magnitude over the 23 echoes after one least-squares scale per sphere.

Models: mono-exponential decay with the reference T2 (what T2 fitting assumes), the Bloch
simulation of the real .seq at nominal B1+, and with a global B1+ scale fitted on the even
spheres and scored on the odd ones (a first digital-twin step).

    uv run python validation/real/nist_tse.py
"""

from __future__ import annotations

import argparse
import functools
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from nist import (
    PIXEL,
    T2_MNCL2,
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
    slice_positions,
    sphere_radius,
    t1_of_mncl2,
)
from scipy import ndimage

import blochparty as bp
from blochparty.pulseq import read_pulseq

# BLOCHPARTY_BACKEND=core: the float64 reference (slow at this size); default the GPU (float32: Metal, or CUDA on an
# NVIDIA machine); until 2026-10-09 the fused CPU kernel, with the same tables
engine = functools.partial(bp.simulate, engine=os.environ.get("BLOCHPARTY_BACKEND", "gpu"))

DATA = Path(__file__).with_name("data") / "developer_main_site" / "TSE_T2_mapping"
SEQ = DATA / "acq" / "tse_multiecho_NIST_t2_sag.seq"
N_ECHO, ESP, N_TR, N_DISCARD = 23, 7e-3, 256, 128  # TRs with readouts
N_TR_ALL = 258  # the file plays 2 passes of (1 dummy TR without ADC + 128 lines)
TE = ESP * np.arange(1, N_ECHO + 1)
T2_REF = T2_MNCL2
T1_REF = t1_of_mncl2(T2_MNCL2)


def real_images() -> np.ndarray:
    """(23, N, N): the authors' reconstruction (MATLAB reshape to samples, coils, echoes, TRs;
    the last 128 TRs are the image)."""
    data = sio.loadmat(DATA / "acq" / "T2_mapping_raw_data.mat")["data"]
    kk = data.reshape(N, 20, N_ECHO, N_TR, order="F")[:, :, :, N_DISCARD:]
    return np.stack([recon(kk[:, :, q, :]) for q in range(N_ECHO)])


def sim_images(signal: torch.Tensor) -> np.ndarray:
    """(C, n_adc) time-ordered signal -> (23, N, N) images from its last 128 readout TRs."""
    s = signal.detach().reshape(signal.shape[0], -1, N_ECHO, N)[:, -N:]  # (C, lines, echoes, samples)
    return np.stack([recon(s[:, :, q, :].permute(2, 0, 1).cpu().numpy()) for q in range(N_ECHO)])


def sphere_rois() -> list[np.ndarray]:
    """The authors' 14 ROI circles ordered as NIST spheres 1..14 by geometry: on this plane the
    outer ring (spheres 1-10) runs counter-clockwise from 0° in 36° steps and the inner ring
    (11-14) continues from next to sphere 10 in ~90° steps, the mirror image of the T1 plane.
    The order is confirmed by the data's decay rates (see nist_tse README notes), not taken from
    any T2 fit."""
    lab, n = ndimage.label(sio.loadmat(DATA / "ana" / "T2_by_sphere_filtered.mat")["total_circle_map"] > 0)
    assert n == 14, n
    rois = [lab == i for i in range(1, n + 1)]
    cs = np.array([ndimage.center_of_mass(r) for r in rois])
    c = cs.mean(0)
    ang = np.degrees(np.arctan2(cs[:, 0] - c[0], cs[:, 1] - c[1])) % 360
    rad = np.hypot(*(cs - c).T)
    outer = [i for i in np.argsort(ang) if rad[i] > 20]
    inner = [i for i in range(n) if rad[i] <= 20]
    near10 = min(inner, key=lambda i: abs((ang[i] - ang[outer[-1]] + 180) % 360 - 180))  # sphere 11 sits by 10
    inner = sorted(inner, key=lambda i: (ang[i] - ang[near10]) % 360)
    assert len(outer) == 10 and len(inner) == 4
    return [rois[i] for i in outer + inner]


def imaged_slice() -> float:
    """The file excites two slices, one per pass (x = +12 mm in TRs 0-128, -12 mm in 129-257);
    the image (the authors' "discard the first 128 excitations") is the second pass's slice."""
    xs = slice_positions(SEQ)
    assert len(xs) == 2, xs
    return xs[1]


def sphere_radius_t1_plane() -> float:
    """Mean sphere radius [m] measured on the T1 plane. The spheres are the same size in both
    arrays, and the half-maximum estimate is unreliable here (weak sphere/fill contrast: it gives
    0 mm for sphere 2 and 9-10 mm elsewhere, vs. 6-8 mm on the T1 plane)."""
    import nist_irse

    img = nist_irse.real_images()
    return float(np.mean([sphere_radius(img, r) for r in nist_irse.sphere_rois()]))


def tr_starts(seq: bp.Sequence) -> list[int]:
    """Segment index of each TR's excitation (dummies included): RF runs after a gap > 1 s."""
    t = np.r_[0, np.cumsum(seq.dt.numpy())]
    rf = np.asarray(seq.is_rf)
    starts = np.flatnonzero(rf & ~np.r_[False, rf[:-1]])
    ends = np.flatnonzero(rf & ~np.r_[rf[1:], False]) + 1
    out = [int(starts[0])] + [int(s) for s, e_prev in zip(starts[1:], ends[:-1]) if t[s] - t[e_prev] > 1.0]
    assert len(out) == N_TR_ALL, len(out)
    return out


def last_trs(seq: bp.Sequence, n_tr: int) -> bp.Sequence:
    """The last ``n_tr`` TRs (counting dummies); earlier TRs only set up the steady state."""
    a = tr_starts(seq)[N_TR_ALL - n_tr]
    return replace(
        seq,
        dt=seq.dt[a:],
        rf=seq.rf[a:],
        grad=seq.grad[a:],
        adc_phase=seq.adc_phase[a:],
        is_rf=seq.is_rf[a:],
        adc=seq.adc[a:],
        spoil=seq.spoil[a:],
        meta={},
    )


def sim_complex_images(signal: torch.Tensor) -> np.ndarray:
    """(1, n_adc) single-coil signal -> (23, N, N) complex images from its last 128 readout TRs."""
    s = signal.detach().reshape(signal.shape[0], -1, N_ECHO, N)[:, -N:]
    return np.stack([recon_complex(s[:, :, q, :].permute(2, 0, 1).cpu().numpy()) for q in range(N_ECHO)])


def monte_carlo(
    real,
    rois,
    radii,
    x0,
    seq,
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
    """Independent Sobol seeds of spheres (+ water fill), each simulated once: shape errors of the
    power-debiased magnitude with jackknife SEs over seeds (``nist.MCErrors``), and the seed images
    (seeds, n_echo, N, N) for paired comparisons."""
    imgs = []
    for k in range(seeds):
        t0 = time.perf_counter()
        phs = [phantom_qmc(rois, radii, T1_REF, T2_REF, x0, n=n, seed=k)]
        if fill_margin > 0:
            mask = fill_mask(real_img, rois, radii[0] + wall)
            phs.append(
                fill_phantom_qmc(
                    rois,
                    mask,
                    radii[0],
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
        total = sum(engine(ph, seq).signal for ph in phs)
        imgs.append(sim_complex_images(total))
        print(
            f"  seed {k}: {' + '.join(str(len(ph)) for ph in phs)} spins, {time.perf_counter() - t0:.0f} s", flush=True
        )
    imgs = np.array(imgs)
    return mc_errors(imgs, rois, real), imgs


def mono_exponential() -> np.ndarray:
    return np.exp(-TE[:, None] / (np.asarray(T2_REF)[None] * 1e-3))


def simulate(phantoms, seq: bp.Sequence, rois, b1: float | None = None) -> np.ndarray:
    """ROI-mean echo curves of one phantom or the sum of several (spheres + water fill: spins are
    independent, so their complex signals add before reconstruction)."""
    phantoms = phantoms if isinstance(phantoms, (list, tuple)) else [phantoms]
    total = 0
    for ph in phantoms:
        if b1 is not None:
            ph = bp.Phantom(ph.pos, ph.pd, ph.t1, ph.t2, ph.db0, b1=torch.full((len(ph),), b1, dtype=ph.pd.dtype))
        total = total + engine(ph, seq).signal
    return roi_means(sim_images(total), rois)


def plot_curves(real, rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True)
    te = TE * 1e3
    for k, ax in enumerate(axes.flat):
        ax.plot(te, real[:, k] / real[:, k].max(), "ko", ms=3.5, label="scanner")
        for (name, _, curve), style in zip(rows, ("C1--", "C0-", "C2-")):
            a = (curve[:, k] * real[:, k]).sum() / (curve[:, k] ** 2).sum()
            ax.plot(te, a * curve[:, k] / real[:, k].max(), style, lw=1.4, label=name)
        ax.set_title(f"sphere {k + 1}: T2 {T2_REF[k]:.0f} ms", fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=6.5, loc="lower left")
    fig.supxlabel("TE [ms]")
    fig.suptitle("NIST phantom, 3 T multi-echo spin echo: scanner vs. blochparty (same .seq, reference T1/T2)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-slice", type=int, default=64, help="spin positions across the slice")
    p.add_argument("--sub", type=int, default=1, help="spins per pixel side along phase encoding")
    p.add_argument("--sub-read", type=int, default=8, help="spins per pixel side along the readout")
    p.add_argument("--rf-merge", type=int, default=4, help="merge this many 1 us RF samples (1: exact)")
    p.add_argument("--warmup", type=int, default=8, help="TRs simulated before the 128 imaging TRs")
    p.add_argument("--b1-grid", type=float, nargs="*", default=[], help="global B1+ scales to fit over")
    p.add_argument(
        "--fill-margin",
        type=float,
        default=5,
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
    p.add_argument("--figure", action="store_true", help="write nist_tse_curves.png (the figure RESULTS.md shows)")
    args = p.parse_args(argv)

    real_img = real_images()
    rois = sphere_rois()
    real = roi_means(real_img, rois)
    radii = [sphere_radius_t1_plane()] * 14
    x0 = imaged_slice()
    n_tr = N + 1 + args.warmup  # 128 lines + this pass's dummy + warm-up TRs of the first pass
    seq = coarsen_rf(last_trs(read_pulseq(SEQ), n_tr).to(dtype=torch.float32), args.rf_merge)
    if args.mc:
        plain = [("mono-exponential, reference T2", shape_error(mono_exponential(), real))]
        rows = []
        for margin in [0, args.fill_margin] if args.fill_margin > 0 else [0]:
            print(f"Monte Carlo: {args.mc} Sobol seeds x {args.n} spins/pixel, water margin {margin} px", flush=True)
            res, imgs = monte_carlo(
                real,
                rois,
                radii,
                x0,
                seq,
                seeds=args.mc,
                n=args.n,
                fill_margin=margin,
                wall=args.wall * 1e-3,
                water=args.water,
                real_img=real_img,
            )
            rows.append((f"Bloch MC{' + water fill' if margin else ''}", res, imgs))
        print_mc_table(rows, rois, real, plain)
        if args.save:
            np.savez_compressed(args.save, names=[r[0] for r in rows], **{f"imgs{i}": r[2] for i, r in enumerate(rows)})
        return rows, real
    ph, _ = phantom(rois, radii, T1_REF, T2_REF, x0, n_slice=args.n_slice, sub=args.sub, sub_read=args.sub_read)
    phantoms = [ph]
    if args.fill_margin > 0:
        mask = fill_mask(real_img, rois, radii[0] + args.wall * 1e-3)
        phantoms.append(
            fill_phantom(
                rois,
                mask,
                radii[0],
                args.wall * 1e-3,
                x0,
                margin=args.fill_margin * PIXEL,
                t1_ms=args.water[0],
                t2_ms=args.water[1],
                n_slice=args.n_slice,
                sub=args.sub,
                sub_read=args.sub_read,
            )
        )
    print(
        f"slice x0 = {x0 * 1e3:.2f} mm, sphere radius {radii[0] * 1e3:.2f} mm, wall {args.wall} mm, "
        f"{' + '.join(str(len(q)) for q in phantoms)} spins, {len(seq)} segments ({n_tr} TRs)",
        flush=True,
    )

    rows = [("mono-exponential, reference T2", shape_error(mono_exponential(), real), mono_exponential())]
    t0 = time.perf_counter()
    sim = simulate(phantoms, seq, rois)
    print(f"simulated in {time.perf_counter() - t0:.0f} s", flush=True)
    label = f" + water fill ({args.fill_margin:g} px)" if args.fill_margin > 0 else ""
    rows.append((f"Bloch on the scanner's .seq{label}, nominal B1+", shape_error(sim, real), sim))
    if args.b1_grid:
        even, odd = [1, 3, 5, 7, 9], [0, 2, 4, 6, 8]  # spheres 2,4,..,10 fit / 1,3,..,9 scored
        fits = []
        for b in args.b1_grid:
            s = simulate(phantoms, seq, rois, b1=b)
            fits.append((shape_error(s, real)[even].mean(), b, s))
            print(f"  B1 {b:.3f}: fit-sphere error {100 * fits[-1][0]:.2f} %", flush=True)
        _, b, s = min(fits, key=lambda f: f[0])
        err = shape_error(s, real)
        rows.append((f"Bloch + B1+ {b:.3f} (fit on even spheres; odd ones scored)", err, s))
        print(f"held-out odd spheres 1-9: {100 * err[odd].mean():.2f} %")

    print("\n| model | spheres 1–10 | odd 1–9 | per sphere 1..14 [%] |\n|---|---|---|---|")
    for name, err, _ in rows:
        print(
            f"| {name} | {100 * err[:10].mean():.2f} % | {100 * err[[0, 2, 4, 6, 8]].mean():.2f} % | "
            f"{' '.join(f'{v:.1f}' for v in 100 * err)} |"
        )
    if args.figure:
        plot_curves(real, rows, Path(__file__).with_name("nist_tse_curves.png"))
    return rows, real


if __name__ == "__main__":
    main()
