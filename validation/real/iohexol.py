"""Benchmark 9: CEST saturation-time prediction on an iohexol phantom (3 T Siemens MAGNETOM Vida).

Data: Huemer et al., MRM 2025, Zenodo 10.5281/zenodo.17244380 (CC BY 4.0), selected members fetched
by ``fetch_iohexol.py``: two conventional CEST reference scans of the same phantom (Falcon tubes of
iohexol, CEST at 4.2 ppm, doped with Gadovist for a range of T1; concentrations not given) with
the same 3D Cartesian GRE readout (128 x 128 x 3, 1.5 x 1.5 x 6.7 mm, 4 deg, TR 5.2 ms, centric:
the k-space centre is the first of 384 lines; Trec 3.5 s, M0 at -1500 ppm after 10 s) after
saturation by trains of 100 ms optimal-control pulses with 1 ms gaps: 15 pulses (1.514 s, the
shipped .seq) and 40 pulses (4.04 s; the raw header names
``Ref_00_..._OC_40_100ms_...``, which is not in the archive: rebuilt from the 15-pulse file).
57 offsets -5..6 ppm.

Test: per tube, water T1 / T2 and the iohexol pool(s) calibrated on the 1.5 s Z-spectrum, then the
4 s Z-spectrum predicted by blochparty with nothing refitted.

    uv run python validation/real/iohexol.py [recon|...]
    BLOCHPARTY_BACKEND=metal uv run python validation/real/iohexol.py predict   # on the Apple GPU
"""

from __future__ import annotations

import functools
import math
import re
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).with_name("data") / "iohexol"
SEQ15 = DATA / "Ref_00_3D_res128_alpha_4_OC_15_100ms_1ms_4ppm_Phantom_Trec_3p5.seq"
RAW = {1.5: DATA / "meas_MID00127_FID56966_Ref_Tsat_1p5.dat", 4.0: DATA / "meas_MID00131_FID56970_Ref_Tsat_4.dat"}
N, NPAR, NOFF = 128, 3, 58


def simulator():
    """``blochparty.simulate``, or the Apple GPU kernel with ``BLOCHPARTY_BACKEND=gpu`` (or metal, cuda; float32)."""
    import os

    from blochparty import simulate

    engine = os.environ.get("BLOCHPARTY_BACKEND", "core")
    if engine != "core":
        return functools.partial(simulate, engine=engine)

    return simulate


def offsets_ppm() -> np.ndarray:
    s = SEQ15.read_text(errors="ignore")
    return np.array([float(x) for x in re.search(r"^offsets_ppm (.*)$", s, re.M).group(1).split()])


def encoding() -> tuple[np.ndarray, np.ndarray]:
    """(ky, kz) grid indices of the 384 readouts of one offset, from the .seq's k-space
    (pypulseq), identical for every offset."""
    import pypulseq as pp

    seq = pp.Sequence()
    seq.read(str(SEQ15))
    k = seq.calculate_kspace()[0].reshape(3, -1, N)
    centre = k[:, :, N // 2]
    ky = np.round(centre[1] * 0.192).astype(int)
    kz = np.round(centre[2] * 0.020).astype(int)
    ky, kz = ky.reshape(NOFF, -1), kz.reshape(NOFF, -1)
    if not ((ky == ky[0]).all() and (kz == kz[0]).all()):
        raise RuntimeError("encoding differs between offsets")
    return ky[0], kz[0]


def recon(tsat: float) -> np.ndarray:
    """(offsets, coils, NPAR, N, N) complex images."""
    from blochparty.io import twix  # here, so the module stays numpy-only on import (iohexol_t1 imports it)

    adc, _ = twix.read_adcs(RAW[tsat])
    ky, kz = encoding()
    nc = adc.shape[1]
    k = np.zeros((NOFF, nc, NPAR, N, N), np.complex64)
    adc = adc.reshape(NOFF, -1, nc, N)
    k[:, :, kz + 1, (ky + N // 2) % N, :] = adc.transpose(0, 2, 1, 3)  # ky = +64 is the -64 line
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(k, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1))


def combine(img: np.ndarray) -> np.ndarray:
    """(offsets, NPAR, N, N) magnitude after a coil combination with the M0 image's sensitivities
    (s_c = img_c / rss at M0; S = |sum conj(s_c) img_c|)."""
    m0 = img[0]
    rss = np.sqrt((np.abs(m0) ** 2).sum(0))
    s = m0 / np.maximum(rss, 1e-12 * rss.max())
    return np.abs((s.conj()[None] * img).sum(1))


def tubes(z1: np.ndarray, m0: np.ndarray, n: int = 8) -> list[np.ndarray]:
    """n tube ROIs: connected regions where the +1 ppm image (iohexol's hydroxyl exchange, strong in
    every tube, absent in the bath) is below 0.75, holes filled, eroded by 2 pixels, kept if 40-400
    pixels; the outline is geometric, not a voxel-wise selection on Z."""
    from scipy import ndimage

    inside = ndimage.binary_erosion(m0 > 0.3 * np.percentile(m0, 99), iterations=2)
    lab, k = ndimage.label(ndimage.binary_opening(inside & (ndimage.gaussian_filter(z1, 0.7) < 0.75), iterations=1))
    regs = [ndimage.binary_erosion(ndimage.binary_fill_holes(lab == j), iterations=2) for j in range(1, k + 1)]
    regs = [r for r in regs if 40 < r.sum() < 400]
    if len(regs) != n:
        raise RuntimeError(f"found {len(regs)} tubes")
    return sorted(regs, key=lambda m: tuple(np.argwhere(m).mean(0).round()))


def fields() -> tuple[np.ndarray, np.ndarray]:
    """The authors' WASABI dB0 [ppm] and relative B1 maps of the central partition, transposed into
    our image frame (edge registration: transposed, no shift, correlation 0.74 vs <= 0.67 for the
    other seven orientations)."""
    import scipy.io as sio

    b0 = sio.loadmat(DATA / "meas_MID00122_FID56961_WASABI_3D_dB0_map_NN.mat", squeeze_me=True)["B0_map_NN"][:, :, 1].T
    b1 = sio.loadmat(DATA / "meas_MID00122_FID56961_WASABI_3D_B1_map_NN.mat", squeeze_me=True)["B1_map_NN"][:, :, 1].T
    return b0, b1


def water_shift(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per spectrum (offsets, n) the water-dip position [ppm]: parabola through the minimum and
    its neighbours."""
    i = np.clip(np.argmin(z, 0), 1, len(w) - 2)
    n = np.arange(z.shape[1])
    y0, y1, y2 = z[i - 1, n], z[i, n], z[i + 1, n]
    h = w[i + 1] - w[i]
    return w[i] + h * (y0 - y2) / (2 * (y0 - 2 * y1 + y2))


F_SCANNER = 123.262709  # MHz, lFrequency of both raw headers (2.894 T)
F_SEQ = 127.7292  # MHz: the .seq's ppm scale (B0 3 in its definitions); offsets are played in Hz
TREC, TREC_M0, TR, FLIP, N_LINES = 3.5, 10.0, 5.2e-3, 4.0, 384


def sat_pulse(step: float = 100e-6):
    """The saturation pulse in its RF frame (blochparty's convention), from two consecutive pulses
    of one offset of the 15-pulse .seq: (envelope (k,) complex [Hz] on ``step``, pulse period T [s],
    sign s such that pulse k carries the phase k * s * 2 pi f T in the frame of an offset f [Hz])."""
    import pypulseq as pp
    import torch
    from magma import rf_frame

    from blochparty.pulseq import read_pulseq

    seq = pp.Sequence()
    seq.read(str(SEQ15))
    ids = list(seq.block_events)
    sat = [i for i in ids[:4000] if seq.block_events[i][1] and seq.get_block(i).rf.shape_dur > 0.05]
    p1 = sat[16]  # second offset (the first after M0), its second pulse onwards
    small = pp.Sequence(system=seq.system)
    for i in range(p1, p1 + 3):
        small.add_block(seq.get_block(i))
    fr, f = rf_frame(read_pulseq(small))
    c = torch.complex(fr.rf[:, 0], fr.rf[:, 1]).numpy()
    on = np.array(fr.is_rf)
    idx = np.flatnonzero(on)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    e1, e2 = c[runs[0]], c[runs[1]]
    t = np.cumsum(fr.dt.numpy())
    period = t[runs[1][0]] - t[runs[0][0]]
    dpsi = np.angle(np.vdot(e1, e2))
    cands = {sg: abs(np.angle(np.exp(1j * (dpsi - sg * 2 * math.pi * f * period)))) for sg in (1, -1)}
    sign = min(cands, key=cands.get)
    if cands[sign] > 1e-3:
        raise RuntimeError(f"pulse phase step {dpsi:.4f} does not match +-2 pi f T ({cands})")
    dt_raw = float(fr.dt[runs[0][0]])
    k = int(round(step / dt_raw))
    n = len(e1) // k * k
    env = e1[:n].reshape(-1, k).mean(1)
    return env, period, sign, f, len(e1) * dt_raw


def build_offsets(n_pulses: int, env, period: float, step: float):
    """One offset's events in the RF frame with the pulses' phase steps moved into the gaps: the
    saturation train (``n_pulses`` copies of ``env``, 1 ms gaps carrying a gradient of 1000 Hz/m,
    spins placed at x = phase step / (2 pi G tau)), then ideal spoiling. Returns the prep sequence
    and the readout + recovery sequence (384 hard 4 deg pulses, TR 5.2 ms, spoiled; then Trec)."""
    import blochparty as bp

    gap = period - len(env) * step
    g = 1000.0
    b = bp.SequenceBuilder()
    samples = np.stack([env.real, env.imag], -1)
    for k in range(n_pulses):
        b.rf_waveform(samples, step)
        if k < n_pulses - 1:
            b.delay(gap, grad=(g, 0.0, 0.0))
    b.delay(1e-6).spoil()
    prep = b.build()
    r = bp.SequenceBuilder()
    for _ in range(N_LINES):
        r.rf(1e-6, flip_deg=FLIP, n_steps=1).delay(TR - 1e-6).spoil()
    return prep, r.build(), g, gap


def phantom(par: dict, db0_hz, b1, f_hz, period: float, gap: float, g: float, sign: int):
    """Spins = (tube, offset) pairs in the RF frame of their offset: off-resonance dB0 - f, phase
    step of that offset as a gap rotation (position x), the iohexol pools in Hz at the scanner
    frequency. ``par`` entries are (n_tubes,) tensors: t1, t2, f_nh, k_nh, f_oh, k_oh, dw_oh [ppm];
    the amide pool sits at 4.3 ppm (T1 1 s, T2 10 ms), the hydroxyl pool T1 1 s, T2 10 ms."""
    import torch

    import blochparty as bp
    from blochparty.exchange import ExchangePool

    nt, nf = len(db0_hz), len(f_hz)
    rep = lambda v: torch.as_tensor(v, dtype=torch.float64).reshape(nt, 1).expand(nt, nf).reshape(-1)  # noqa: E731
    f = torch.as_tensor(f_hz, dtype=torch.float64).reshape(1, nf).expand(nt, nf).reshape(-1)
    dpsi = torch.remainder(sign * 2 * math.pi * f * period + math.pi, 2 * math.pi) - math.pi
    pos = torch.zeros(nt * nf, 3, dtype=torch.float64)
    pos[:, 0] = dpsi / (2 * math.pi * g * gap)
    one = torch.ones(nt * nf, dtype=torch.float64)
    pools = (ExchangePool(rep(par["f_nh"]), rep(par["k_nh"]), one, 0.01 * one, 4.3 * F_SCANNER * one),
             ExchangePool(rep(par["f_oh"]), rep(par["k_oh"]), one, 0.01 * one, rep(par["dw_oh"]) * F_SCANNER))
    return bp.Phantom(pos, one, rep(par["t1"]), rep(par["t2"]), rep(db0_hz) - f, rep(b1), pools=pools)


def spectra(par: dict, db0_hz, b1, n_pulses: int, passes: int = 2, step: float = 100e-6, start=None):
    """(n_tubes, 57) simulated Z for a Tsat of ``n_pulses``: Mz before the first readout line of
    each offset over that of the M0 scan; offsets coupled through the readout and Trec, iterated
    ``passes`` times from ``start`` (Mz at the start of each offset's preparation, (n_tubes * 58,);
    default equilibrium). Returns (Z, Mz before readout, start states for the next call)."""
    import torch


    env, period, sign, _, _ = sat_pulse(step)
    prep, ro, g, gap = build_offsets(n_pulses, env, period, step)
    f_hz = offsets_ppm() * F_SEQ
    nt, nf = len(db0_hz), len(f_hz)
    ph = phantom(par, db0_hz, b1, f_hz, period, gap, g, sign)
    t1 = ph.t1
    start = torch.ones(nt * nf, dtype=torch.float64) if start is None else torch.as_tensor(start, dtype=torch.float64)
    for _ in range(passes):
        m0 = torch.zeros(nt * nf, 3, dtype=torch.float64)
        m0[:, 2] = start
        pre = simulator()(ph, prep, m_init=m0).m
        after = simulator()(ph, ro, m_init=pre).m[:, 2].reshape(nt, nf)
        # next offset starts after Trec from this one's readout; the M0 scan after Trec_M0 from the last
        prev = torch.roll(after, 1, dims=1)
        trec = torch.full((nt, nf), TREC, dtype=torch.float64)
        trec[:, 0] = TREC_M0
        rec = 1 - (1 - prev) * torch.exp(-trec / t1.reshape(nt, nf))
        start = rec.reshape(-1)
    mz = pre[:, 2].reshape(nt, nf)
    return (mz[:, 1:] / mz[:, :1]), mz, start


def measured():
    """Per tube: ROI-mean Z (Tsat 1.5 and 4), median WASABI dB0 [Hz] and B1, and per scan the field
    drift [Hz] since WASABI from the bath (the water outside the tubes: no CEST, so its dip is B0)."""
    from scipy import ndimage

    w = offsets_ppm()[1:]
    mags = {t: np.load(DATA / f"ref_tsat{t}.npy") for t in RAW}
    m0 = mags[1.5][0, 1]
    rois = tubes(mags[1.5][1:, 1][np.argmin(abs(w - 1.0))] / np.maximum(m0, 1e-9), m0)
    b0, b1 = fields()
    inside = ndimage.binary_erosion(m0 > 0.3 * np.percentile(m0, 99), iterations=4)
    near = ndimage.binary_dilation(sum(rois).astype(bool), iterations=4)
    bath = inside & ~near & (b1 > 0)
    drift, z = {}, {}
    for t, mag in mags.items():
        zz = mag[1:, 1] / np.maximum(mag[0, 1], 1e-9)
        drift[t] = float(np.median(water_shift(zz[:, bath], w) - b0[bath])) * F_SEQ
        z[t] = np.stack([zz[:, r].mean(1) for r in rois])
    return rois, z, np.array([np.median(b0[r]) for r in rois]) * F_SCANNER, np.array([np.median(b1[r]) for r in rois]), drift, int(bath.sum())


NAMES = ("t1", "t2", "f_nh", "k_nh", "f_oh", "k_oh", "dw_oh")
LOG = {"t1", "t2", "f_nh", "k_nh", "f_oh", "k_oh"}
START = dict(t1=1.0, t2=0.2, f_nh=0.002, k_nh=1000.0, f_oh=0.005, k_oh=1500.0, dw_oh=0.8)
BOUNDS = dict(t1=(0.2, 4.0), t2=(0.01, 3.0), f_nh=(1e-5, 0.05), k_nh=(20.0, 2e4), f_oh=(1e-5, 0.1),
              k_oh=(20.0, 4e4), dw_oh=(0.2, 2.5))


def _to_par(theta: np.ndarray) -> dict:
    import torch

    return {n: torch.as_tensor(np.exp(theta[:, j]) if n in LOG else theta[:, j], dtype=torch.float64)
            for j, n in enumerate(NAMES)}


def _clip(theta):
    for j, n in enumerate(NAMES):
        lo, hi = BOUNDS[n]
        theta[:, j] = np.clip(theta[:, j], *(np.log([lo, hi]) if n in LOG else (lo, hi)))
    return theta


def fit(z_meas: np.ndarray, db0_hz, b1, n_pulses: int = 15, iters: int = 10, h: float = 1e-3, step: float = 200e-6,
        fixed: dict | None = None):
    """Levenberg-Marquardt per tube on (n_tubes, 57) measured Z; the Jacobian by forward differences
    of all parameters at once (each tube's 1 + 7 parameter sets are spins of one simulation). The
    offsets' coupling (readout + Trec) is carried between iterations: one pass each after the first
    (the start states change little with the parameters). 200 us RF steps (3e-4 from 100 us).
    ``fixed``: {name: (n_tubes,) values} held at those values (e.g. T1 from the IR Look-Locker scan)."""
    fixed = fixed or {}
    nt, npar = len(db0_hz), len(NAMES)
    theta = _clip(np.array([[np.log(START[n]) if n in LOG else START[n] for n in NAMES]] * nt, dtype=float))
    for n, v in fixed.items():
        theta[:, NAMES.index(n)] = np.log(v) if n in LOG else v
    free = [j for j, n in enumerate(NAMES) if n not in fixed]
    lam = np.full(nt, 1e-2)
    best, start = None, None
    for it in range(iters):
        nf = len(free)
        sets = np.repeat(theta[:, None, :], nf + 1, 1)
        for q, j in enumerate(free):
            sets[:, q + 1, j] += h
        zs, _, st = spectra(_to_par(sets.reshape(-1, npar)), np.repeat(db0_hz, nf + 1), np.repeat(b1, nf + 1),
                            n_pulses, passes=2 if start is None else 1, step=step, start=start)
        start = st.reshape(nt, nf + 1, -1)[:, :1].expand(nt, nf + 1, -1).reshape(-1)
        zs = zs.numpy().reshape(nt, nf + 1, -1)
        r = zs[:, 0] - z_meas
        jac = (zs[:, 1:] - zs[:, :1]) / h  # (nt, npar, 57)
        cost = (r**2).sum(1)
        if best is not None:  # accept or reject the previous step per tube
            worse = cost > best[1]
            lam = np.where(worse, lam * 10, lam / 3)
            theta = np.where(worse[:, None], best[0], theta)
            r = np.where(worse[:, None], best[2], r)
            jac = np.where(worse[:, None, None], best[3], jac)
            cost = np.minimum(cost, best[1])
        best = (theta.copy(), cost.copy(), r.copy(), jac.copy())
        print(f"  iteration {it}: rms per tube {np.sqrt(cost / r.shape[1]).round(4)}", flush=True)
        delta = []
        for i in range(nt):
            a = jac[i] @ jac[i].T
            delta.append(np.linalg.solve(a + lam[i] * np.diag(np.diag(a) + 1e-12), -jac[i] @ r[i]))
        theta = theta.copy()
        theta[:, free] += np.array(delta)
        theta = _clip(theta)
    return best[0], best[1]


def main(argv):
    what = argv[1] if len(argv) > 1 else "recon"
    if what == "recon":
        w = offsets_ppm()
        for tsat in RAW:
            mag = combine(recon(tsat))
            np.save(DATA / f"ref_tsat{tsat}.npy", mag.astype(np.float32))
            z = mag[1:, 1] / np.maximum(mag[0, 1], 1e-9)
            core = mag[0, 1] > 0.5 * mag[0, 1].max()
            zz = np.median(z[:, core], axis=1)
            print(f"Tsat {tsat} s: images {mag.shape}; median Z over the bright region: min {zz.min():.3f} at "
                  f"{w[1:][np.argmin(zz)]:+.2f} ppm, Z(-5) {zz[0]:.3f}, Z(+6) {zz[-1]:.3f}, Z(4.2) {zz[np.argmin(abs(w[1:] - 4.2))]:.3f}")
    if what in ("fit", "fit_t1"):
        import json

        _, z, db0, b1, drift, _ = measured()
        fixed, name = None, "fit_tsat1.5.json"
        if what == "fit_t1":  # T1 from the IR Look-Locker scan (iohexol_t1.py)
            t1 = json.loads((DATA / "t1_irll.json").read_text())["tubes"]
            fixed, name = {"t1": np.array([t1[str(i + 1)]["T1"] for i in range(len(db0))])}, "fit_tsat1.5_t1irll.json"
        theta, cost = fit(z[1.5], db0 + drift[1.5], b1, fixed=fixed)
        par = {n: v.tolist() for n, v in _to_par(theta).items()}
        (DATA / name).write_text(json.dumps({"par": par, "rms": np.sqrt(cost / 57).tolist()}, indent=1))
        print(json.dumps(par, indent=1))
    if what in ("predict", "predict_t1"):
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _, z, db0, b1, drift, _ = measured()
        name = "fit_tsat1.5.json" if what == "predict" else "fit_tsat1.5_t1irll.json"
        par = {k: np.array(v) for k, v in json.loads((DATA / name).read_text())["par"].items()}
        theta = np.stack([np.log(par[n]) if n in LOG else par[n] for n in NAMES], 1)
        w = offsets_ppm()[1:]
        sim = {}
        for tsat, n in ((1.5, 15), (4.0, 40)):
            sim[tsat] = spectra(_to_par(theta), db0 + drift[tsat], b1, n, passes=3)[0].numpy()
        rms = lambda a, b: np.sqrt(np.mean((a - b) ** 2, axis=1))  # noqa: E731
        fit_rms, pred_rms, same_rms = rms(sim[1.5], z[1.5]), rms(sim[4.0], z[4.0]), rms(z[1.5], z[4.0])
        dz_meas, dz_sim = z[4.0] - z[1.5], sim[4.0] - sim[1.5]
        print("tube: fitted T1 / T2 [s], f_nh, k_nh, f_oh, k_oh, dw_oh | rms: fit at 1.5 s, predicted 4 s, "
              "'no change' 4 s | change 4 - 1.5 s at 4.3 ppm measured / predicted")
        k43 = np.argmin(abs(w - 4.3))
        for i in range(len(db0)):
            print(f"  {i + 1}: {par['t1'][i]:.3f} / {par['t2'][i]:.3f}, {par['f_nh'][i]:.2e}, {par['k_nh'][i]:.0f}, "
                  f"{par['f_oh'][i]:.2e}, {par['k_oh'][i]:.0f}, {par['dw_oh'][i]:.2f} | {fit_rms[i]:.4f}, {pred_rms[i]:.4f}, "
                  f"{same_rms[i]:.4f} | {dz_meas[i, k43]:+.4f} / {dz_sim[i, k43]:+.4f}")
        corr = np.corrcoef(dz_meas.ravel(), dz_sim.ravel())[0, 1]
        print(f"median rms: fit {np.median(fit_rms):.4f}, prediction {np.median(pred_rms):.4f}, no-change {np.median(same_rms):.4f}; "
              f"correlation of the 4 - 1.5 s change over all tubes and offsets {corr:.3f}")
        fig, ax = plt.subplots(2, 4, figsize=(18, 7))
        for i, a in enumerate(ax.ravel()):
            a.plot(w, z[1.5][i], "b.", ms=3, label="measured 1.5 s")
            a.plot(w, sim[1.5][i], "b-", lw=1, label="fitted 1.5 s")
            a.plot(w, z[4.0][i], "r.", ms=3, label="measured 4 s")
            a.plot(w, sim[4.0][i], "r-", lw=1, label="predicted 4 s")
            a.invert_xaxis()
            a.set_ylim(0, 1.05)
            a.set_title(f"tube {i + 1}: rms fit {fit_rms[i]:.3f}, prediction {pred_rms[i]:.3f}", fontsize=9)
        ax[0, 0].legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(Path(__file__).with_name("iohexol_prediction.png" if what == "predict" else "iohexol_prediction_t1irll.png"), dpi=60)
    if what in ("recon", "tubes"):
        w = offsets_ppm()[1:]
        mags = {t: np.load(DATA / f"ref_tsat{t}.npy") for t in RAW}
        m0 = mags[1.5][0, 1]
        rois = tubes(mags[1.5][1:, 1][np.argmin(abs(w - 1.0))] / np.maximum(m0, 1e-9), m0)
        b0, b1 = fields()
        for i, r in enumerate(rois):
            line = []
            for t, mag in mags.items():
                z = mag[1:, 1][:, r] / mag[0, 1][r]
                line.append(f"Tsat {t}: water dip {np.median(water_shift(z, w)):+.3f} ppm, Z(4.2) {np.median(z[np.argmin(abs(w - 4.2))]):.3f}")
            print(f"tube {i + 1} centre {np.argwhere(r).mean(0).round()} ({r.sum()} px): WASABI dB0 {np.median(b0[r]):+.3f} ppm, "
                  f"B1 {np.median(b1[r]):.3f}; " + "; ".join(line))


if __name__ == "__main__":
    main(sys.argv)
