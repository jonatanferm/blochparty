"""Benchmark 4: inversion recovery NMR of 15 % BSA and 1 mM MnCl2 after rectangular pi pulses
of 22.8-912 us (Asslaender et al. 2022; data and citation in ``fetch_asslaender.py``).

1 T Magritek Spinsolve (43.4 MHz): for each pulse duration T_RF = s x 22.8 us (s = 1, 2, 5,
10, ..., 40), 20 inversion times log-spaced 3 ms - 5 s (plus the authors' timing correction),
30 s repetition; the signal is the first FID point, phased by the last inversion time and
normalised by the overall maximum (the authors' ``load_NMR_data.jl``). The free pool relaxes
with T2* (fitted to the FID) during the pulse. Following the authors' model, a constant factor
scales the free pool's z after the pulse (inversion efficiency), the pools share R1, and the
bound pool exchanges at Rex: blochparty's f = m0s / (1 - m0s), k = (1 - m0s) Rex, PD = 1 - m0s.

Fitted with blochparty (core, float64; the pulse in 50 segments, see below):

* MnCl2: one pool (m0, inversion efficiency, R1);
* BSA: the generalized Bloch model (super-Lorentzian, memory 100 and 300 T2b), the rate model,
  and, as an exact-kernel reference, ``generalized_bloch.simulate_pulse`` (untapered
  Green's function, converged grid) for the pulse with the closed-form recovery.

The authors' published fits (their documentation, MRIgeneralizedBloch.jl) are the comparison.

    uv run python validation/real/fetch_asslaender.py && uv run python validation/real/asslaender_ir.py
"""

from __future__ import annotations

import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm
from scipy.optimize import curve_fit, least_squares

import blochparty as bp
from blochparty import generalized_bloch as gb

DATA = Path(__file__).with_name("data") / "asslaender"
SAMPLES = {"MnCl2": "20210419_1mM_MnCl2", "BSA": "20210416_15%BSA_2ndBatch"}
SCALE = [1, 2] + list(range(5, 41, 5))
T_MIN = 22.8e-6
TRF = np.array(SCALE) * T_MIN
TI = np.exp(np.linspace(np.log(3e-3), np.log(5), 20)) + 12 * T_MIN + (13 * 15.065 - 5) * 1e-6
N_STEPS = 50  # pulse segments: one segment misses the T2 decay while M is transverse (0.5 % here)
F64 = torch.float64

AUTHORS = {  # their documentation's printed fits
    "T2* MnCl2": 0.07497933747410264,
    "T2* BSA": 0.05240554901160865,
    "MnCl2": dict(R1=1.4787326527483415, R1_std=0.0025253461723147427, resid=0.0069842810791774365),
    "BSA gBloch": dict(
        m0=1.095296680512974,
        minv=0.9298345039485287,
        m0s=0.08545256843767216,
        R1=1.1977737700876356,
        T2s=12.95604663754452e-6,
        Rex=71.25408510176928,
        std=[
            0.0013856859195475598,
            0.001224087247397477,
            0.000995259821377364,
            0.001482091207613726,
            6.073187751113117e-7,
            1.9919818510823935,
        ],
        resid=0.004135899495936585,
    ),
    "BSA Graham": dict(resid=0.018342764271148924, T2s=2.8557964813368537e-6, m0s=0.06992813369001911),
}


def load(sample: str, scale: int) -> np.ndarray:
    """(points, inversion times) complex FIDs, the authors' 6 points trimmed at each end."""
    b = (DATA / SAMPLES[sample] / f"ja_IR_v2 ({scale})" / "1" / "data.2d").read_bytes()
    kind, n1, n2, a, c = struct.unpack("<5i", b[12:32])
    assert kind == 501 and a == c == 1  # complex float32
    d = np.frombuffer(b[32 : 32 + 8 * n1 * n2], dtype="<f4").reshape(n2, n1, 2)
    return (d[..., 0] + 1j * d[..., 1]).T[6 : n1 - 6]


def t2star(sample: str) -> tuple[float, float]:
    """Mono-exponential complex FID fit at T_RF = 22.8 us, first inversion time (the authors'
    comment says 5 s, but column 1 is the shortest, 3 ms; the FID shape does not care)."""
    f = load(sample, 1)[:, 0]
    t = 100e-6 * (np.arange(1, len(f) + 1) + 7)
    model = lambda _, a, b, T, w: np.r_[a * np.exp(-t / T) * np.cos(w * t), b * np.exp(-t / T) * np.sin(w * t)]
    p, c = curve_fit(model, None, np.r_[f.real, f.imag], p0=(1, 1, 0.1, 0))
    return p[2], float(np.sqrt(c[2, 2]))


def data(sample: str) -> np.ndarray:
    M = np.zeros((len(TI), len(SCALE)))
    for i, s in enumerate(SCALE):
        d = load(sample, s)[0]
        M[:, i] = (d * np.exp(-1j * np.angle(d[-1]))).real
    return M / M.max()


def model(p, t2f: float, kind: str, memory: float = 100.0, n_steps: int = N_STEPS) -> np.ndarray:
    """Signal (inversion times, pulse durations) with blochparty: the pulse, the inversion
    efficiency applied to the free pool's z, transverse magnetization spoiled (phase cycling),
    then the recovery as a free run."""
    one = lambda v: torch.tensor([v], dtype=F64)
    if kind == "bloch":
        m0, minv, R1 = p
        m0s = 0.0
    else:
        m0, minv, m0s, R1, T2b, Rex = p
    ph = bp.Phantom(pos=torch.zeros(1, 3, dtype=F64), pd=one(1 - m0s), t1=one(1 / R1), t2=one(t2f), db0=one(0.0))
    if kind != "bloch":
        ph = replace(
            ph,
            bound=bp.BoundPool(
                f=one(m0s / (1 - m0s)), k=one((1 - m0s) * Rex), t1=one(1 / R1), t2=one(T2b), model=kind, memory=memory
            ),
        )
    out = np.zeros((len(TI), len(TRF)))
    for i, T in enumerate(TRF):
        r = bp.simulate(ph, bp.SequenceBuilder().rf(T, flip_deg=180.0, n_steps=n_steps).build())
        m = [0.0, 0.0, float(r.m[0, 2]) * minv] + ([float(r.m_bound[0])] if kind != "bloch" else [])
        for j, t in enumerate(TI):
            seq = bp.SequenceBuilder().delay(t - T / 2).build()  # TI is from the pulse centre
            out[j, i] = m0 * float(bp.simulate(ph, seq, m_init=torch.tensor([m], dtype=F64)).m[0, 2])
    return out


def model_reference(p, t2f: float, memory: float | None = None) -> np.ndarray:
    """Exact-kernel reference: ``simulate_pulse`` (untapered super-Lorentzian unless
    ``memory``) on a converged grid (h <= T2b / 16, >= 64 steps), closed-form recovery."""
    m0, minv, m0s, R1, T2b, Rex = p
    m0f = 1 - m0s
    H = np.array([[-R1 - m0s * Rex, m0f * Rex, R1 * m0f], [m0s * Rex, -R1 - m0f * Rex, R1 * m0s], [0, 0, 0]])
    green = gb.green
    if memory is not None:
        gb.green = lambda s, t2, shape, memory_=memory: green(s, t2, shape, memory_)
    try:
        out = np.zeros((len(TI), len(TRF)))
        for i, T in enumerate(TRF):
            n = max(64, int(np.ceil(T / (T2b / 16))))
            r = gb.simulate_pulse(
                gb.rectangular(np.pi, T, T / n),
                T / n,
                t2s=T2b,
                shape="superlorentzian",
                r1s=R1,
                m0s=m0s,
                free=dict(m0=m0f, r1=R1, r2=1 / t2f, dw=0.0),
                k=m0f * Rex,
            )
            z = [r["free"][-1, 2] * minv, r["zs"][-1], 1.0]
            for j, t in enumerate(TI):
                out[j, i] = m0 * (expm(H * (t - T / 2)) @ z)[0]
    finally:
        gb.green = green
    return out


def fit(fun, M, p0, lo, hi, scale):
    f = least_squares(lambda p: (fun(p) - M).ravel(), p0, bounds=(lo, hi), x_scale=scale, diff_step=1e-4)
    s2 = (f.fun @ f.fun) / (f.fun.size - len(p0))
    std = np.sqrt(np.diag(np.linalg.inv(f.jac.T @ f.jac) * s2))
    return f.x, std, float(np.linalg.norm(f.fun) / np.linalg.norm(M)), f.fun.reshape(M.shape)


def main():
    t2 = {s: t2star(s) for s in SAMPLES}
    for s in SAMPLES:
        print(f"T2* {s}: {1e3 * t2[s][0]:.4f} +- {1e3 * t2[s][1]:.4f} ms (authors {1e3 * AUTHORS['T2* ' + s]:.4f})")

    M = data("MnCl2")
    print("\nMnCl2, one pool (m0, inversion efficiency, R1):")
    for n in (1, N_STEPS):
        x, sd, r, res = fit(
            lambda p: model(p, t2["MnCl2"][0], "bloch", n_steps=n),
            M,
            [1, 0.9, 1],
            [0, 0.1, 0.3],
            [np.inf] * 3,
            np.array([1, 1, 1]),
        )
        print(
            f"  pulse in {n:2d} segment(s): R1 {x[2]:.6f} +- {sd[2]:.6f} /s, residual {100 * r:.4f} %"
            f"   per T_RF [%]: {np.round(100 * np.linalg.norm(res, axis=0) / np.linalg.norm(M, axis=0), 2)}"
        )
    a = AUTHORS["MnCl2"]
    print(f"  authors:                R1 {a['R1']:.6f} +- {a['R1_std']:.6f} /s, residual {100 * a['resid']:.4f} %")

    M = data("BSA")
    t2b = t2["BSA"][0]
    p0, lo, hi = [1, 0.932, 0.1, 1, 10e-6, 50], [0, 0.1, 0, 0.3, 1e-9, 10], [np.inf, np.inf, 1, np.inf, 20e-6, 1e3]
    xs = np.array([1, 1, 0.1, 1, 1e-5, 50])
    a = AUTHORS["BSA gBloch"]
    pa = [a[k] for k in ("m0", "minv", "m0s", "R1", "T2s", "Rex")]
    rows = [("authors (their fit)", pa, a["std"], a["resid"])]
    cases = [
        ("blochparty generalized, memory 100", lambda p: model(p, t2b, "generalized", 100.0)),
        ("blochparty generalized, memory 300", lambda p: model(p, t2b, "generalized", 300.0)),
        ("reference, exact kernel", lambda p: model_reference(p, t2b)),
        ("blochparty rate model", lambda p: model(p, t2b, "rate")),
    ]
    print("\nBSA: residual at the authors' parameters:")
    for name, fun in cases[:3]:
        print(f"  {name:36s} {100 * np.linalg.norm(fun(pa) - M) / np.linalg.norm(M):.4f} %")
    for name, fun in cases:
        x, sd, r, _ = fit(fun, M, p0, lo, hi, xs)
        rows.append((name, x, sd, r))
    print(
        "\nBSA fits:                            m0       Minv     m0s              R1 [/s]   T2s [us]       Rex [/s]       residual"
    )
    for name, x, sd, r in rows:
        print(
            f"  {name:36s} {x[0]:.4f}  {x[1]:.4f}  {x[2]:.4f}+-{sd[2]:.4f}  {x[3]:.4f}    {1e6 * x[4]:5.2f}+-{1e6 * sd[4]:.2f}"
            f"   {x[5]:6.1f}+-{sd[5]:4.1f}   {100 * r:.4f} %"
        )
    g = AUTHORS["BSA Graham"]
    print(
        f"  authors, Graham's spectral model: m0s {g['m0s']:.4f}, T2s {1e6 * g['T2s']:.2f} us, residual {100 * g['resid']:.4f} %"
    )


if __name__ == "__main__":
    main()
