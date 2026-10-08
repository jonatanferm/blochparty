"""Benchmark 7: a measured gradient impulse response (3 T Siemens Prisma) through ``apply_girf``.

Data: Wu, Jaffray, Vannesjo, Uludag, Kasper, ISMRM 2022 #641; Zenodo 10.5281/zenodo.6376737 (CC BY
4.0), fetched by ``fetch_girf.py``. Thin-slice method: two 1 mm slices at +-17 mm on each axis,
a triangle blip (180 T/m/s, 18 amplitudes 9-39.6 mT/m) 2 ms into a 50 ms ADC (5 us dwell), 50
repetitions, reference scans without blips; the output gradient is the difference of the two
slices' phase derivatives, (d phi_1 / dt - d phi_2 / dt) / (2 * 2 pi gamma_bar * 17 mm), and the
common part (d phi_1 / dt + d phi_2 / dt) / 2 the B0 term. Meas1 has positive blips; Meas2, eight
months later, positive and negative blips. Released raw signals are coil-averaged (the authors'
precomputed GIRFs used the coils separately).

Frame (ours, consistent throughout): ADC sample i at t_i = i dwell (the authors' ``roTime``); the
output o_i is the mean gradient over (t_{i-1}, t_i]; the nominal input is the piecewise-linear
waveform through the 10 us raster values starting at 2 ms, as segment averages on the same 5 us
cells. Delays below are in this frame (a different ADC-time convention shifts them all equally).

Steps (all in ``main``):

1. ``outputs``: measured output gradients and B0 terms (Python port of the authors' code).
2. ``estimate``: GIRF H(f) = sum O conj(I) / sum |I|^2 from Meas1, band-limited, vs the authors'
   precomputed Meas1 GIRF.
3. ``predict``: Meas2 outputs (both polarities) predicted through ``apply_girf`` with H from Meas1,
   against ideal gradients and a pure delay; the B0 term as accumulated phase (it turns out to be
   below the drift left after the reference correction).

    uv run python validation/real/girf.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import scipy.io as sio

DATA = Path(__file__).with_name("data") / "girf"
GAMMA_BAR = 42.57e3  # Hz/mT, as the authors' processing (their output scale)
POS = 0.017  # m
DWELL = 5e-6
N = 10002
START = 2e-3  # nominal blip start after ADC start
SLICES = {"x": ("SagL17", "SagR17"), "y": ("CorP17", "CorA17"), "z": ("TraH17", "TraF17")}
F_CUT = 30e3  # band limit of the estimated GIRF (blip spectra carry little power above)


def _raw(meas, name):
    return sio.loadmat(DATA / f"Meas{meas}" / f"{name}.mat", squeeze_me=True)["kspace_all"]


def outputs(meas: int, axis: str, polarity: str = "Positive"):
    """(18, N) output gradient [mT/m] and B0 term [Hz], repetition means, sample i = mean over
    (t_{i-1}, t_i] (i = 0 is zero)."""
    s1, s2 = (_raw(meas, polarity + n) for n in SLICES[axis])  # (N, rep, 18)
    r1, r2 = (_raw(meas, "Ref" + n).mean(1) for n in SLICES[axis])  # (N, nref)
    nb = s1.shape[2]
    ref = np.arange(nb) // 3  # a reference scan before every three blip amplitudes (authors' code)
    rate = []
    for s, r in ((s1, r1), (s2, r2)):
        c = s / r[:, None, ref]
        d = np.diff(np.unwrap(np.angle(c), axis=0), axis=0) / DWELL  # rad/s, (N-1, rep, 18)
        rate.append(np.concatenate([np.zeros((1, *d.shape[1:])), d]).mean(1).T)  # (18, N)
    grad = (rate[0] - rate[1]) / 2 / (2 * math.pi * GAMMA_BAR) / POS
    b0 = (rate[0] + rate[1]) / 2 / (2 * math.pi)
    return grad, b0


def nominal(meas: int = 1) -> np.ndarray:
    """(18, N) nominal input: cell averages over (t_{i-1}, t_i] of the piecewise-linear blip."""
    g = sio.loadmat(DATA / f"Meas{meas}" / "InputGradients.mat", squeeze_me=True)["gradIn_all"]  # (5001, 18), 10 us
    tg = START + 10e-6 * np.arange(g.shape[0])
    edges = DWELL * np.arange(-1, N)  # cell i is (edges[i], edges[i + 1]]
    out = np.zeros((g.shape[1], N))
    for j in range(g.shape[1]):
        cum = np.concatenate([[0], np.cumsum((g[1:, j] + g[:-1, j]) / 2 * 10e-6)])  # exact integral of the linear pieces
        m = np.interp(edges, tg, cum, left=0, right=cum[-1])
        # within a linear piece the integral is quadratic: interpolate it exactly at the cell edges
        for k, e in enumerate(edges):
            if tg[0] < e < tg[-1]:
                n = int((e - tg[0]) // 10e-6)
                tau = e - tg[n]
                slope = (g[n + 1, j] - g[n, j]) / 10e-6
                m[k] = cum[n] + g[n, j] * tau + slope * tau**2 / 2
        out[j] = np.diff(m) / DWELL
    return out


def spectrum(x):
    return np.fft.fft(x, axis=-1)


def estimate(o, i):
    """H on the length-N FFT grid from blip pairs (18, N), band-limited by a raised cosine to F_CUT."""
    so, si = spectrum(o), spectrum(i)
    h = (so * si.conj()).sum(0) / (np.abs(si) ** 2).sum(0)
    f = np.fft.fftfreq(N, DWELL)
    w = np.where(np.abs(f) < 0.8 * F_CUT, 1.0,
                 np.where(np.abs(f) < F_CUT, 0.5 * (1 + np.cos(math.pi * (np.abs(f) - 0.8 * F_CUT) / (0.2 * F_CUT))), 0.0))
    return h * w


def delay_of(h):
    f = np.fft.fftfreq(N, DWELL)
    band = (np.abs(f) < 3e3) & (f != 0)
    return -np.polyfit(f[band], np.angle(h[band]), 1, w=np.abs(h[band]))[0] / (2 * math.pi)


def authors(meas: int, axis: str, kind: str = "Origin") -> np.ndarray:
    """The authors' precomputed GIRF, repetition mean, moved from fftshift order to ours (their
    frame: input point samples at t_i, output over (t_{i-1}, t_i])."""
    m = sio.loadmat(DATA / "CalculatedGIRF" / f"GIRF{kind}_G{axis}_Meas{meas}.mat", squeeze_me=True)
    return np.fft.ifftshift(m["GIRF_FT"].mean(1))


LEAD, TAPS = 100, 800  # impulse response window: -0.5 ms .. +3.5 ms


def to_girf(hs: dict, hb0: dict | None = None):
    """blochparty GIRF (4, 3, TAPS) from transfer functions per axis [mT/m per mT/m] and B0 terms
    per axis [Hz per mT/m] (blochparty's B0 row is Hz per Hz/m: divided by gamma_bar)."""
    import torch

    from blochparty.girf import GIRF

    taps = torch.zeros(4, 3, TAPS, dtype=torch.float64)
    for j, ax in enumerate("xyz"):
        for row, h in ((j, hs.get(ax)), (3, None if hb0 is None else hb0.get(ax))):
            if h is None:
                continue
            imp = np.roll(np.fft.ifft(h).real, LEAD)[:TAPS] / (GAMMA_BAR if row == 3 else 1.0)
            taps[row, j] = torch.from_numpy(imp)
    return GIRF(taps, DWELL, -LEAD * DWELL)


def played(model, i_cells: np.ndarray, axis: str) -> tuple[np.ndarray, np.ndarray]:
    """(gradient [mT/m], B0 [Hz]) segment averages of ``apply_girf`` for nominal cell values on
    one axis: a Sequence of 5 us segments (t_{i-1}, t_i], gradients in Hz/m (gamma_bar)."""
    import torch

    from blochparty.girf import apply_girf
    from blochparty.sequence import Sequence

    s = len(i_cells)
    g = torch.zeros(s, 3, dtype=torch.float64)
    g[:, "xyz".index(axis)] = torch.from_numpy(i_cells) * GAMMA_BAR  # mT/m -> Hz/m
    seq = Sequence(torch.full((s,), DWELL, dtype=torch.float64), torch.zeros(s, 2, dtype=torch.float64), g,
                   torch.zeros(s, dtype=torch.float64), [False] * s, [False] * s, [False] * s)
    out = apply_girf(seq, model)
    return out.grad[:, "xyz".index(axis)].numpy() / GAMMA_BAR, out.b0_offset.numpy()


def errors(pred, meas, window):
    """rms over the window relative to the measured peak, and the moment error at the window end
    relative to the blip moment."""
    a, b = window
    e = pred[:, a:b] - meas[:, a:b]
    peak = np.abs(meas[:, a:b]).max(1)
    rel = np.sqrt((e**2).mean(1)) / peak
    mom = np.abs(e.sum(1)) / np.abs(meas[:, a:b].sum(1))
    return rel, mom


def main():
    import time

    t0 = time.perf_counter()
    win = (int(1.9e-3 / DWELL), int(3.5e-3 / DWELL))  # the blips last 0.1-0.44 ms from 2 ms
    i1, i2 = nominal(1), nominal(2)
    print(f"nominal inputs identical in both measurements: {np.array_equal(i1, i2)}")
    h1, hb1, a1 = {}, {}, {}
    for ax in "xyz":
        o, b = outputs(1, ax)
        h1[ax] = estimate(o, i1)
        hb1[ax] = estimate(b, i1)
        a1[ax] = authors(1, ax)
        f = np.fft.fftfreq(N, DWELL)
        band = np.abs(f) < 10e3
        # the authors' input is point samples at t_i, ours the cell (t_{i-1}, t_i]: half a dwell apart
        diff = np.abs(h1[ax][band] - a1[ax][band] * np.exp(2j * math.pi * f[band] * DWELL / 2)).max()
        print(f"G{ax} Meas1: ours |H(0)| {abs(h1[ax][0]):.4f}, delay {delay_of(h1[ax]) * 1e6:+.2f} us; authors' "
              f"(coils separate, their input frame) delay {delay_of(a1[ax]) * 1e6:+.2f} us; max |H_ours - H_authors| "
              f"< 10 kHz after the half-dwell frame shift {diff:.3f}; B0 term at DC {hb1[ax][0].real:+.3f} Hz per mT/m ({time.perf_counter() - t0:.0f} s)")
    import torch

    from blochparty.girf import GIRF

    full = to_girf(h1, hb1)
    delay = {ax: delay_of(h1[ax]) for ax in "xyz"}
    pure = GIRF.identity(torch.tensor([delay[a] for a in "xyz"], dtype=torch.float64))
    print(f"impulse-response window keeps DC gain {[round(float(full.taps[j, j].sum()), 4) for j in range(3)]}")
    print("Meas2 (8 months later), predicted from Meas1: rms error / peak over 1.9-3.5 ms, median over 18 blips "
          "(max); moment error at 3.5 ms / blip moment")
    for ax in "xyz":
        for pol, sign in (("Positive", 1), ("Negative", -1)):
            o2, b2 = outputs(2, ax, pol)
            rows = {}
            for name, model in (("ideal", None), ("delay", pure), ("GIRF", full)):
                if model is None:
                    pred = sign * i2
                else:
                    pred = np.stack([played(model, sign * i2[j], ax)[0] for j in range(len(i2))])
                rel, mom = errors(pred, o2, win)
                rows[name] = f"{np.median(rel) * 100:5.2f} ({rel.max() * 100:5.2f}) %, moment {np.median(mom) * 100:.2f} %"
            print(f"  G{ax} {pol}: " + "; ".join(f"{k} {v}" for k, v in rows.items()))
    # B0 term: the phase it accumulates; the part that flips with the blip (a GIRF predicts) vs
    # the part that does not (drift left after the reference correction, concomitant fields)
    ten = int(10e-3 / DWELL)
    for ax in "xyz":
        _, bp = outputs(2, ax, "Positive")
        _, bn = outputs(2, ax, "Negative")
        odd = 2 * math.pi * np.cumsum((bp - bn) / 2, axis=1)[:, ten] * DWELL
        even = 2 * math.pi * np.cumsum((bp + bn) / 2, axis=1)[:, ten] * DWELL
        pred = np.array([2 * math.pi * played(full, i2[j], ax)[1][:ten + 1].sum() * DWELL for j in range(len(i2))])
        print(f"  B0 term G{ax}, phase at 10 ms: odd part {np.abs(odd).max():.3f} rad max, predicted {np.abs(pred).max():.3f}, "
              f"rms(odd - predicted) {np.sqrt(np.mean((odd - pred) ** 2)):.3f}; even part {np.abs(even).max():.3f} rad max")
    print(f"done ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
