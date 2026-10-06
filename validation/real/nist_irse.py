"""Realism benchmark 1: ISMRM/NIST phantom T1 plane, inversion-recovery spin echo, 3 T.

The real scanner's raw data (10 inversion times, 20 coils, Pulseq sequences; see fetch_data.py)
is compared with blochparty simulating the *same .seq files* on a digital model of the phantom
whose relaxation times come from **independent references**, not from fitting this data:

- T1 of the 14 NiCl2 spheres: CaliberMRI/NIST reference values at 3 T and 20 °C, phantom
  version 1 (serial < 43) or 2 (serial >= 43); the dataset does not say which, so both are
  reported (from the RRSG 2020 analysis code, github.com/rrsg2020/analysis, src/nist.py)
- T2 of those spheres at 3 T: Keenan et al., PMC5408299, Table 1 (only affects a per-sphere
  constant here, TE being fixed)

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
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from scipy import ndimage

import blochparty as bp
from blochparty.fused import simulate_fused
from blochparty.pulseq import read_pulseq

DATA = Path(__file__).with_name("data") / "developer_main_site" / "IRSE_T1_mapping"
TI = [50, 75, 100, 125, 150, 250, 1000, 1500, 2000, 3000]  # [ms]
TR, TE = 4.5, 10e-3  # [s]
N, FOV, SLICE = 128, 0.250, 6e-3
PIXEL = FOV / N
CENTRE = 63.5  # pixel index of position 0 in the authors' reconstruction (measured, see check_mapping)

T1_REF = {  # [ms], 3 T, 20 °C
    "v1": [1989, 1454, 984.1, 706, 496, 351.5, 247.13, 175.3, 125.9, 89.0, 62.7, 44.53, 30.84, 21.719],
    "v2": [1883.97, 1330.16, 987.27, 690.08, 484.97, 341.58, 240.86, 174.95, 121.08, 85.75, 60.21, 42.89,
           30.40, 21.44],
}
T2_REF = [581, 406, 292, 203, 143, 97, 71, 51, 37, 26, 20, 14, 13, 11]  # [ms], 3 T


def recon(data: np.ndarray) -> np.ndarray:
    """The authors' reconstruction: per-coil centred 2D iFFT of (samples, coils, lines), RSS."""
    im = np.fft.fftshift(np.fft.ifft2(np.transpose(data, (1, 0, 2))), axes=(-2, -1))
    return np.sqrt((np.abs(im) ** 2).sum(0))


def real_images() -> np.ndarray:
    return np.stack([recon(sio.loadmat(DATA / "acq" / f"T1_mapping_raw_TI{ti}ms.mat")["data"]) for ti in TI])


def sim_images(signal: torch.Tensor) -> np.ndarray:
    """(C, n_adc) time-ordered simulated signal -> image, through the same reconstruction."""
    data = signal.detach().reshape(signal.shape[0], N, N).permute(2, 0, 1).cpu().numpy()
    return recon(data)


def sphere_rois() -> list[np.ndarray]:
    """The authors' 14 ROI circles, split and ordered as NIST spheres 1..14 (longest T1 first)."""
    ana = sio.loadmat(DATA / "ana" / "t1_by_sphere_filtered.mat")
    lab, n = ndimage.label(ana["total_circle_map"] > 0)
    assert n == 14, n
    t1_map = sio.loadmat(DATA / "ana" / "T1_map.mat")["T1_map"]
    rois = [lab == i for i in range(1, n + 1)]
    return sorted(rois, key=lambda m: -np.median(t1_map[m]))


def sphere_radius(images: np.ndarray, roi: np.ndarray) -> float:
    """Physical sphere radius [m] from the real images (geometry only): the half-maximum of the
    sphere-vs-surroundings contrast in the radial profile, at the TI where that contrast is largest."""
    r0, c0 = ndimage.center_of_mass(roi)
    rr, cc = np.indices(roi.shape)
    dist = np.hypot(rr - r0, cc - c0)
    best = None
    for im in images:
        prof = np.array([im[(dist >= d - 0.5) & (dist < d + 0.5)].mean() for d in np.arange(0, 12)])
        inside, outside = prof[:2].mean(), prof[9:].mean()
        if best is None or abs(inside - outside) > best[0]:
            best = (abs(inside - outside), prof, inside, outside)
    _, prof, inside, outside = best
    half = (inside + outside) / 2
    sign = 1 if inside > outside else -1
    k = int(np.argmax(sign * (prof - half) < 0))  # first radius past half-max
    d = k - 1 + (sign * (prof[k - 1] - half)) / (sign * (prof[k - 1] - prof[k]))
    return float(d * PIXEL)


def slice_position(seq_file) -> float:
    """Slice centre along x [m]: the excitation's frequency offset over its slice-select gradient
    (Pulseq positions slices by frequency; with blochparty's default "physical" reading a
    frequency offset f excites at +f/G)."""
    import pypulseq as pp

    s = pp.Sequence()
    s.read(str(seq_file))
    for i in range(1, len(s.block_events) + 1):
        b = s.get_block(i)
        if b.rf is not None and b.gx is not None and getattr(b.gx, "amplitude", None):
            return float(b.rf.freq_offset / b.gx.amplitude)
    raise ValueError("no slice-selective pulse with a trapezoid on x")


def phantom(rois, radii, t1_ms, x0, *, n_slice=48, sub=2, seed=0, dtype=torch.float32) -> tuple[bp.Phantom, torch.Tensor]:
    """Spheres as 3D balls centred on the slice at x0: in-plane spins at sub x sub per pixel
    (stratified jitter), slice positions stratified over 1.5 slice thicknesses either side; the
    slice axis is x, readout y, phase z (from the gradients of the .seq). Returns the phantom and
    each spin's sphere index."""
    g = torch.Generator().manual_seed(seed)
    half = 1.5 * SLICE
    xs = (torch.arange(n_slice, dtype=torch.float64) + torch.rand(n_slice, generator=g, dtype=torch.float64)) \
        / n_slice * 2 * half - half
    pos, sph, t1, t2 = [], [], [], []
    for k, (roi, rad) in enumerate(zip(rois, radii)):
        r0, c0 = ndimage.center_of_mass(roi)
        y0, z0 = (r0 - CENTRE) * PIXEL, (c0 - CENTRE) * PIXEL
        m = math.ceil(rad / PIXEL) + 1
        grid = (torch.arange(-m * sub, m * sub, dtype=torch.float64) + 0.5) * (PIXEL / sub)
        gy, gz = torch.meshgrid(grid, grid, indexing="ij")
        gy = gy + (torch.rand(gy.shape, generator=g, dtype=torch.float64) - 0.5) * (PIXEL / sub)
        gz = gz + (torch.rand(gz.shape, generator=g, dtype=torch.float64) - 0.5) * (PIXEL / sub)
        for x in xs:
            inside = gy**2 + gz**2 <= rad**2 - x**2
            n = int(inside.sum())
            if n == 0:
                continue
            pos.append(torch.stack([torch.full((n,), x0 + float(x), dtype=torch.float64), gy[inside] + y0,
                                    gz[inside] + z0], -1))
            sph.append(torch.full((n,), k))
            t1.append(torch.full((n,), t1_ms[k] * 1e-3, dtype=torch.float64))
            t2.append(torch.full((n,), T2_REF[k] * 1e-3, dtype=torch.float64))
    pos, sph, t1, t2 = (torch.cat(v) for v in (pos, sph, t1, t2))
    pd = torch.full_like(t1, (PIXEL / sub) ** 2 * (2 * half / n_slice) / (PIXEL**2 * SLICE))
    ph = bp.Phantom(pos=pos, pd=pd, t1=t1, t2=t2, db0=torch.zeros_like(t1)).to(dtype=dtype)
    return ph, sph


def coarsen_rf(seq: bp.Sequence, k: int) -> bp.Sequence:
    """Merge every k consecutive RF segments of a pulse into one with the same duration and
    the same RF and gradient integrals (averages weighted by dt). The real files sample RF on a
    1 us raster, which makes RF ~80 % of all segments; the error this introduces is measured in
    the convergence study."""
    if k == 1:
        return seq
    groups, s, n = [], 0, len(seq)
    while s < n:
        e = s + 1
        if seq.is_rf[s]:
            while e < n and e - s < k and seq.is_rf[e] and not seq.adc[e - 1] and not seq.spoil[e - 1]:
                e += 1
        groups.append((s, e))
        s = e
    idx = torch.tensor([a for a, _ in groups])
    seg = torch.repeat_interleave(torch.arange(len(groups)), torch.tensor([b - a for a, b in groups]))
    dt = torch.zeros(len(groups), dtype=seq.dt.dtype).index_add_(0, seg, seq.dt)
    avg = lambda v: torch.zeros(len(groups), v.shape[1], dtype=v.dtype).index_add_(0, seg, v * seq.dt[:, None]) / dt[:, None]
    last = torch.tensor([b - 1 for _, b in groups])
    return replace(seq, dt=dt, rf=avg(seq.rf), grad=avg(seq.grad), adc_phase=seq.adc_phase[last],
                   is_rf=[seq.is_rf[a] for a, _ in groups], adc=[seq.adc[b - 1] for _, b in groups],
                   spoil=[seq.spoil[b - 1] for _, b in groups], meta=dict(seq.meta))


def roi_means(images: np.ndarray, rois) -> np.ndarray:
    """(n_TI, 14) ROI-mean magnitudes."""
    return np.array([[im[r].mean() for r in rois] for im in images])


def shape_error(sim: np.ndarray, real: np.ndarray, idx=None) -> np.ndarray:
    """Per-sphere relative RMS error after one least-squares scale per sphere, over TIs ``idx``."""
    idx = slice(None) if idx is None else idx
    s, r = sim[idx], real[idx]
    a = (s * r).sum(0) / (s * s).sum(0)
    return np.sqrt(((a * s - r) ** 2).sum(0) / (r * r).sum(0))


def analytic(t1_ms) -> np.ndarray:
    t1 = np.asarray(t1_ms)[None] * 1e-3
    ti = np.asarray(TI)[:, None] * 1e-3
    return np.abs(1 - 2 * np.exp(-ti / t1) + np.exp(-TR / t1)) * np.exp(-TE / (np.asarray(T2_REF)[None] * 1e-3))


def simulate_all(ph: bp.Phantom, seqs, b1=None) -> list[torch.Tensor]:
    if b1 is not None:
        ph = bp.Phantom(ph.pos, ph.pd, ph.t1, ph.t2, ph.db0, b1=b1)
    return [simulate_fused(ph, s).signal for s in seqs]


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
    through the fused kernel's adjoint."""
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
    args = p.parse_args(argv)

    real_img = real_images()
    rois = sphere_rois()
    real = roi_means(real_img, rois)
    radii = [sphere_radius(real_img, r) for r in rois]
    print(f"sphere radii [mm]: {np.round(np.array(radii) * 1e3, 2)}")
    seqs = [coarsen_rf(read_pulseq(DATA / "acq" / f"irse_T1_TI{ti}.seq").to(dtype=torch.float32), args.rf_merge)
            for ti in TI]
    x0 = slice_position(DATA / "acq" / f"irse_T1_TI{TI[0]}.seq")
    print(f"slice centre x0 = {x0 * 1e3:.2f} mm; {len(seqs[0])} segments per sequence")
    train, test = [0, 2, 4, 6, 8], [1, 3, 5, 7, 9]

    rows = []
    for ver, t1 in T1_REF.items():
        rows.append((f"analytic IR formula, T1 {ver}", shape_error(analytic(t1), real), analytic(t1)))
        ph, _ = phantom(rois, radii, t1, x0, n_slice=args.n_slice, sub=args.sub)
        t0 = time.perf_counter()
        sim = roi_means(np.stack([sim_images(s) for s in simulate_all(ph, seqs)]), rois)
        print(f"{ver}: {len(ph)} spins, 10 sequences simulated in {time.perf_counter() - t0:.0f} s", flush=True)
        rows.append((f"Bloch on the scanner's .seq, T1 {ver}", shape_error(sim, real), sim))
        if args.fit_b1:
            b1 = fit_b1(ph, seqs, real, rois, train)
            sim_b = roi_means(np.stack([sim_images(s) for s in simulate_all(ph, seqs, b1=torch.full((len(ph),), b1))]),
                              rois)
            rows.append((f"Bloch + B1 {b1:.3f} (fit on 5 TIs, scored on 5), T1 {ver}", shape_error(sim_b, real, test),
                         sim_b))

    lines = [f"| model | spheres 1–10 | 1–14 | per sphere 1..14 [%] |", "|---|---|---|---|"]
    for name, err, _ in rows:
        lines.append(f"| {name} | {100 * err[:10].mean():.2f} % | {100 * err.mean():.2f} % | "
                     f"{' '.join(f'{v:.1f}' for v in 100 * err)} |")
    table = "\n".join(lines)
    print("\n" + table)
    plot_curves(real, [r for r in rows if r[0].endswith("v2")][:2], Path(__file__).with_name("nist_irse_curves.png"))
    return rows, table


if __name__ == "__main__":
    main()
