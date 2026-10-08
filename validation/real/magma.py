"""Benchmark 8 (consistency): Pulseq-CEST APTw Z-spectra of an L-arginine phantom (3 T Prisma fit).

Data: Pulseq-CEST Library MAGMA 2025 phantom DICOMs (Zenodo 10.5281/zenodo.15053086, CC BY 4.0;
``fetch_magma.py``), hybrid 3D GRE images (224 x 184 x 12, 0.98 mm, 5 mm slices) after
(a) the APTw preparation (36 sinc-Gauss pulses of 50 ms, 5 ms gaps, B1rms 2 uT, 1.975 s; 33
offsets -4..4 ppm at 123.2 MHz and an M0 scan at -300 ppm; Trec 3.5 s) and (b) WASABI (one 5 ms
block, 3.7 uT, 31 offsets of +-2 "ppm" at the .seq's nominal 127.73 MHz). Phantom (Schuenke et
al., MAGMA 2025, Table 1): five tubes, 1 mM Gadovist (T1 ~1.5 s), L-arginine 20 / 27 mM without
agarose (tubes 1, 2) and 40 / 60 / 70 mM with 2 % agarose (3-5), pH ~4 (k ~350 Hz). Tube 1 is
the authors' ROI (rows 145-150, columns 95-100 of slice 5 in their V[row, col, slice, offset]).

The tube model parameters (the library's ``phantoms/l-arginin/*.yaml``) are partly tuned against
these measurements (a comment in the 40 mM file), so agreement here is consistency, not
prediction.

    uv run --with pydicom python validation/real/magma.py
"""

from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np

DATA = Path(__file__).with_name("data") / "magma"
SLICE = 5
GAMMA = 42.5764  # Hz / uT
FREQ_APT, FREQ_WASABI = 123.2, 127.7292  # MHz: the .seq files' own ppm scales (Hz offsets are what is played)
APT_PPM = np.concatenate([[-300.0], np.arange(-4, 4.001, 0.25)])
WASABI_PPM = np.concatenate([[-300.0], np.linspace(-2, 2, 31)])


def load(series: str) -> np.ndarray:
    """(offsets, slices, rows, cols) magnitude, instance k = offset k // 12, slice k % 12."""
    import pydicom

    files = sorted(glob.glob(str(DATA / f"PULSEQ_HYBRID_GRE_2_2_5_{series}" / "PULSEQ*" / "*.IMA")))
    ds = sorted((pydicom.dcmread(f) for f in files), key=lambda d: int(d.InstanceNumber))
    n = len(ds) // 12
    v = np.stack([d.pixel_array.astype(np.float64) for d in ds]).reshape(n, 12, ds[0].Rows, ds[0].Columns)
    return v


def tubes(m0: np.ndarray, radius: float = 13.5) -> list[np.ndarray]:
    """Five tube ROIs (discs of radius - 4 voxels) in a slice: the dark tube walls found by matching
    an annulus of the wall radius to the darkness below the container median, five peaks on the
    ring 25-45 voxels from the container centre; ordered by angle from the bottom tube (tube 1,
    the authors' ROI), counter-clockwise in (row, col)."""
    from scipy import ndimage

    sm = ndimage.gaussian_filter(m0, 1.0)
    cont = ndimage.binary_fill_holes(sm > 0.3 * np.percentile(sm, 99))
    c0 = np.argwhere(cont).mean(0)
    cont = ndimage.binary_erosion(cont, iterations=3)
    dark = np.where(cont, np.clip(np.median(sm[cont]) - sm, 0, None), 0)
    yy, xx = np.mgrid[-25:26, -25:26]
    rr = np.hypot(yy, xx)
    ring, inner = np.abs(rr - radius) < 1.2, rr < radius - 2.5
    score = ndimage.correlate(dark, ring / ring.sum() - inner / inner.sum())
    gy, gx = np.mgrid[: m0.shape[0], : m0.shape[1]]
    dist = np.hypot(gy - c0[0], gx - c0[1])
    score = np.where((dist > 25) & (dist < 45), score, -np.inf)
    peaks = []
    for _ in range(5):
        k = np.unravel_index(np.argmax(score), score.shape)
        peaks.append(k)
        score[np.hypot(gy - k[0], gx - k[1]) < 22] = -np.inf
    ang = [np.mod(np.arctan2(b - c0[1], a - c0[0]), 2 * math.pi) for a, b in peaks]
    peaks = [peaks[i] for i in np.argsort(ang)]
    return [np.hypot(gy - a, gx - b) < radius - 4 for a, b in peaks]


def wasabi_model(dw_hz, b1, db0, c, d, tp=5e-3):
    x = dw_hz - db0
    om = np.sqrt((GAMMA * b1) ** 2 + x**2)
    return np.abs(c - d * np.sin(np.arctan2(GAMMA * b1, x)) ** 2 * np.sin(math.pi * om * tp) ** 2)


def wasabi_fit(z: np.ndarray) -> np.ndarray:
    """(b1 [uT], dB0 [Hz], c, d) per spectrum (n, 31) by least squares from a coarse grid start."""
    from scipy.optimize import least_squares

    w = WASABI_PPM[1:] * FREQ_WASABI
    out = []
    for y in z:
        best = None
        for b1 in (3.0, 3.7, 4.4):
            for db0 in (-60, -20, 20, 60):
                r = least_squares(lambda p: wasabi_model(w, *p) - y, (b1, db0, 1.0, 2.0), method="lm")
                if best is None or r.cost < best.cost:
                    best = r
        out.append(best.x)
    return np.array(out)


# the library's phantom models (sim-library/phantoms/l-arginin at master, unchanged since 2024-05-02,
# bbe45dd): (water T1, T2, CEST f, MT f or None); CEST pool 3 ppm, k 350 Hz, T1 1 s, T2 0.1 s; MT
# pool Lorentzian, T2 40 us, k 30 Hz, T1 1 s
TUBE_MODELS = {"20 mM": (1.5, 1.0, 3.6036e-4, None), "27 mM": (1.5, 1.0, 4.8649e-4, None),
               "40 mM": (1.45, 0.05, 7.2072e-4, 0.04), "60 mM": (1.45, 0.05, 1.1e-3, 0.035),
               "70 mM": (1.45, 0.05, 1.3e-3, 0.035)}


def _slice(seq, a, b):
    from dataclasses import replace

    cut = lambda t: None if t is None else t[a:b]  # noqa: E731
    return replace(seq, dt=seq.dt[a:b], rf=seq.rf[a:b], grad=seq.grad[a:b], adc_phase=seq.adc_phase[a:b],
                   is_rf=seq.is_rf[a:b], adc=seq.adc[a:b], spoil=seq.spoil[a:b], b0_offset=cut(seq.b0_offset),
                   rf_freq=cut(seq.rf_freq), meta={})


def coarsen_rf(seq, m: int):
    """Merge runs of m consecutive RF segments into one (duration-weighted mean RF and gradient):
    the .seq's pulses sit on the 1 us RF raster, which the importer steps sample by sample."""
    import torch

    from blochparty.sequence import Sequence

    groups, i, s = [], 0, len(seq)
    while i < s:
        j = i + 1
        if seq.is_rf[i]:
            while j < s and j - i < m and seq.is_rf[j] and not seq.adc[j - 1] and not seq.spoil[j - 1]:
                j += 1
        groups.append((i, j))
        i = j
    idx = torch.tensor([a for a, _ in groups])
    seg = torch.repeat_interleave(torch.arange(len(groups)), torch.tensor([b - a for a, b in groups]))
    dt = torch.zeros(len(groups), dtype=seq.dt.dtype).index_add(0, seg, seq.dt)
    w = seq.dt[:, None]
    rf = torch.zeros(len(groups), 2, dtype=seq.rf.dtype).index_add(0, seg, seq.rf * w) / dt[:, None]
    grad = torch.zeros(len(groups), 3, dtype=seq.grad.dtype).index_add(0, seg, seq.grad * w) / dt[:, None]
    last = [b - 1 for _, b in groups]
    return Sequence(dt, rf, grad, seq.adc_phase[torch.tensor(last)], [seq.is_rf[a] for a, _ in groups],
                    [seq.adc[k] for k in last], [seq.spoil[k] for k in last], {},
                    None if seq.b0_offset is None else seq.b0_offset[idx],
                    None if seq.rf_freq is None else seq.rf_freq[idx])


def rf_frame(seq):
    """The sequence in the frame of its RF (one constant frequency f, as every offset block of a
    CEST .seq): rf * exp(+i 2 pi f t), returning (sequence, f [Hz]); spins then need off-resonance
    db0 - f. A change of rotating frame, so exact; pools see the same RF-minus-spin offsets
    (``Sequence.rf_freq`` derived from the now constant phase is 0). f from the phase slope of the
    RF samples (a pulse turning as exp(-i 2 pi f t) is resonant at offset f)."""
    import torch
    from dataclasses import replace

    t = torch.cumsum(seq.dt, 0) - seq.dt / 2
    c = torch.complex(seq.rf[:, 0], seq.rf[:, 1])
    on = torch.tensor(seq.is_rf) & (c.abs() > 1e-3 * c.abs().max())
    idx = torch.nonzero(on).ravel()
    nxt = idx[1:][(idx[1:] - idx[:-1]) == 1]
    prv = nxt - 1
    dphi = torch.angle(c[nxt] * c[prv].conj())
    f = float(-(dphi / (2 * math.pi * (t[nxt] - t[prv]))).median()) if len(nxt) else 0.0
    c2 = c * torch.exp(2j * math.pi * f * t)
    return replace(seq, rf=torch.stack([c2.real, c2.imag], -1), rf_freq=None), f


def simulate_z(model, db0_hz, rb1, trec_from=None, rf_step=100, frame=True):
    """(33, n) simulated Z per spin (voxel: its WASABI dB0 [Hz] and relative B1) for the APTw .seq:
    Mz at each pseudo-ADC over Mz at the M0 scan. ``trec_from`` None: every offset starts at
    equilibrium (the library's reset); a float: from Mz = that fraction of M0 (the readout's
    unknown saturation) and relaxes through the .seq's recovery time."""
    import torch

    import blochparty as bp
    from blochparty.exchange import BoundPool, ExchangePool
    from blochparty.pulseq import read_pulseq

    t1, t2, fc, fm = model
    full = read_pulseq(DATA / "APTw_3T_001_2uT_36SincGauss_DC90_2s_braintumor.seq")
    seq = full if frame else coarsen_rf(full, rf_step)
    adc = [i for i, a in enumerate(seq.adc) if a]
    n = len(db0_hz)
    one = torch.ones(n, dtype=torch.float64)
    cest = ExchangePool(fc, 350.0, 1.0, 0.1, 3.0 * FREQ_APT)
    bound = None if fm is None else BoundPool(fm * one, 30.0 * one, one, 40e-6 * one, 0.0, lineshape="lorentzian")
    ph = bp.Phantom(torch.zeros(n, 3, dtype=torch.float64), one, t1 * one, t2 * one,
                    torch.as_tensor(db0_hz, dtype=torch.float64), torch.as_tensor(rb1, dtype=torch.float64),
                    bound=bound, pools=(cest,))
    mz, start = [], 0
    from dataclasses import replace

    for k in adc:
        m_init = None
        if trec_from is not None:
            m_init = torch.zeros(n, 3, dtype=torch.float64)
            m_init[:, 2] = trec_from
        block, p = _slice(seq, start, k + 1), ph
        if frame:  # every offset in its own RF frame, then coarse steps of the slow envelope
            block, f = rf_frame(block)
            block = coarsen_rf(block, rf_step)
            p = replace(ph, db0=ph.db0 - f)
        r = bp.simulate(p, block, m_init=m_init)
        mz.append(r.m[:, 2].numpy())
        start = k + 1
    mz = np.array(mz)
    return mz[1:] / mz[0]


def mtr_asym(z, db0_hz, ppm=3.0):
    """Z(-ppm) - Z(+ppm) per voxel after shifting each spectrum by its dB0 (linear interpolation)."""
    w = APT_PPM[1:] * FREQ_APT
    out = []
    for y, d in zip(z.T, db0_hz):
        out.append(np.interp(-ppm * FREQ_APT + d, w, y) - np.interp(ppm * FREQ_APT + d, w, y))
    return np.array(out)


def main():
    apt, was = load("APTW_001_RR_0015"), load("WASABI_RR_0016")
    m0 = apt[0, SLICE]
    rois = tubes(m0)
    za = apt[1:, SLICE] / np.maximum(m0, 1e-9)
    zw = was[1:, SLICE] / np.maximum(was[0, SLICE], 1e-9)
    authors = np.zeros_like(m0, bool)
    authors[145:150, 95:100] = True
    fields, info = [], []
    for i, r in enumerate(rois):
        pars = wasabi_fit(zw[:, r].T)
        b1, db0 = pars[:, 0] / 3.7, pars[:, 1]  # dB0 in Hz: the same for both protocols
        fields.append((db0, b1))
        agar = za[np.argmin(abs(APT_PPM[1:] - 1.5)), r].mean() < 0.6
        info.append((agar, np.median(mtr_asym(za[:, r], db0))))
        print(f"tube {i + 1}: {r.sum()} voxels, centre {np.argwhere(r).mean(0).round(0)}, authors' ROI "
              f"{bool((r & authors).any())}; WASABI rB1 {np.median(b1):.3f} (IQR {np.subtract(*np.percentile(b1, [75, 25])):.3f}), "
              f"dB0 {np.median(db0):+.1f} Hz (IQR {np.subtract(*np.percentile(db0, [75, 25])):.1f}); "
              f"Z(1.5 ppm) {za[np.argmin(abs(APT_PPM[1:] - 1.5)), r].mean():.3f} ({'agarose' if agar else 'no agarose'}), "
              f"MTRasym(3 ppm) {info[-1][1]:+.4f}")
    # assignment: tube 1 is 20 mM (authors' ROI); the other plain tube 27 mM; the agarose tubes
    # 40 / 60 / 70 mM in the order of their measured MTRasym (assumption: monotonic in concentration)
    plain = [i for i in range(5) if not info[i][0] and i != 0]
    agar = sorted([i for i in range(5) if info[i][0]], key=lambda i: info[i][1])
    if len(plain) != 1 or len(agar) != 3:
        raise RuntimeError("expected two plain and three agarose tubes")
    names = {0: "20 mM", plain[0]: "27 mM", agar[0]: "40 mM", agar[1]: "60 mM", agar[2]: "70 mM"}
    print("Z-spectra, measured vs blochparty (voxel-wise WASABI dB0 / B1, library models): rms over 33 offsets "
          "of the ROI mean; MTRasym(3 ppm)")
    q = np.array([1 / 6, 1 / 2, 5 / 6])
    for i, r in enumerate(rois):
        db0, b1 = fields[i]
        meas = za[:, r]
        # the ROI's field spread as a 3 x 3 grid of its dB0 and B1 quantiles (equal weights)
        g0, g1 = np.meshgrid(np.quantile(db0, q), np.quantile(b1, q), indexing="ij")
        for label, trec in (("reset", None), ("Trec, Mz0 = 0", 0.0)):
            cache = DATA / f"sim_tube{i + 1}_{'reset' if trec is None else 'trec'}.npy"
            sim = np.load(cache) if cache.exists() else simulate_z(TUBE_MODELS[names[i]], g0.ravel(), g1.ravel(), trec)
            np.save(cache, sim)
            sim = np.abs(sim)  # the images are magnitudes
            e = meas.mean(1) - sim.mean(1)
            print(f"  tube {i + 1} ({names[i]}, {label}): rms {np.sqrt(np.mean(e**2)):.4f}, max |dZ| {np.abs(e).max():.4f} "
                  f"at {APT_PPM[1:][np.argmax(np.abs(e))]:+.2f} ppm; MTRasym measured {np.median(mtr_asym(meas, db0)):+.4f}, "
                  f"simulated {np.median(mtr_asym(sim, g0.ravel())):+.4f}", flush=True)


if __name__ == "__main__":
    main()
