"""Benchmark 5: the scanner's own Pulseq spiral multi-echo TSE of a T1MES phantom.

Data: Schuenke P, Kolbitsch C. 2D multi-echo TSE spiral MR raw data and Pulseq sequences.
Zenodo 10.5281/zenodo.18749100 (CC BY 4.0): the `.seq` played on a 3 T Siemens MAGNETOM Cima.X
(PTB Berlin, 16 channels) and the raw data with trajectory (ISMRMRD). TR 4 s, 10 echoes 16 ms
apart, 160 spiral-out interleaves of 91 samples (10 us), FOV 160 mm, 8 mm slice; 640 s.

The T1MES reference values (Captur et al., JCMR 2016) were measured at 1.4 T; agarose T2 has
a field- and echo-spacing-dependent exchange part, and the tubes here measure 35-50 % shorter.
So T2 is fitted per tube, and the test is the echo-train *shape*: the Bloch simulation of the
exact `.seq` (slice profiles, crushers, stimulated echoes) against a mono-exponential.

Simulation: the first three TRs (steady state in the third), 1024 spins across 32 mm (4 slice
thicknesses; converged), in-plane position 0 (the spirals are rewound), the echo at each
readout's first sample (the k-space centre); T1 1.5 s (long-T2 tubes) or 1.0 s (short), which
does not change the shapes. The measurement side is ``spiral_tse_recon.py`` (own process).

    uv run python validation/real/spiral_tse.py
"""

from __future__ import annotations

import functools
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import curve_fit

import blochparty as bp
from blochparty.pulseq import read_pulseq

HERE = Path(__file__).parent
DATA = HERE / "data" / "spiral_tse"
CACHE = DATA / "cache"
NAME = "spiral_tse_tr_4000_te_16_nechos_10_nspirals_160"
RECORD = "https://zenodo.org/api/records/18749100"
TE = 0.016 * np.arange(1, 11)
NZ, EXTENT = 1024, 0.032


def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    rec = json.loads(subprocess.run(["curl", "-sfL", RECORD], capture_output=True, check=True).stdout)
    for f in rec["files"]:
        if f["key"] not in (f"{NAME}.seq", f"{NAME}_with_traj.mrd"):
            continue
        dest = DATA / f["key"]
        want = f["checksum"].removeprefix("md5:")
        if not dest.exists() or hashlib.md5(dest.read_bytes()).hexdigest() != want:
            subprocess.run(["curl", "-sfL", "--retry", "3", f["links"]["self"], "-o", str(dest)], check=True)
            if hashlib.md5(dest.read_bytes()).hexdigest() != want:
                raise RuntimeError(f"checksum mismatch: {dest}")


def first_trs(seq: bp.Sequence, n: int = 3) -> bp.Sequence:
    t = np.cumsum(seq.dt.numpy())
    rf = np.array(seq.is_rf)
    starts = [i for i in range(1, len(rf)) if rf[i] and not rf[i - 1]]
    tr = [starts[0]] + [s for a, s in zip(starts[:-1], starts[1:]) if t[s] - t[a] > 1.0]
    e = tr[n]
    cut = lambda v: None if v is None else v[:e]
    return replace(seq, dt=seq.dt[:e], rf=seq.rf[:e], grad=seq.grad[:e], adc_phase=seq.adc_phase[:e],
                   is_rf=seq.is_rf[:e], adc=seq.adc[:e], spoil=seq.spoil[:e], b0_offset=cut(seq.b0_offset),
                   rf_freq=cut(seq.rf_freq), meta={})


def echoes(seq, t1, t2, b1) -> np.ndarray:
    """(atoms, 10) complex echoes of the last simulated TR at the k-space centre."""
    t1, t2, b1 = (np.atleast_1d(np.asarray(v, float)) for v in (t1, t2, b1))
    z = torch.linspace(-EXTENT / 2, EXTENT / 2, NZ, dtype=torch.float64)
    n = len(t2) * NZ
    rep = lambda v: torch.as_tensor(np.repeat(v, NZ), dtype=torch.float64)
    pos = torch.zeros(n, 3, dtype=torch.float64)
    pos[:, 2] = z.repeat(len(t2))
    ph = bp.Phantom(pos=pos, pd=torch.ones(n, dtype=torch.float64), t1=rep(t1), t2=rep(t2),
                    db0=torch.zeros(n, dtype=torch.float64), b1=rep(b1))
    coils = torch.zeros(n, len(t2), dtype=torch.complex64)
    coils[torch.arange(n), torch.arange(n) // NZ] = 1.0 / NZ
    run = functools.partial(bp.simulate, engine="gpu" if bp.engines.available("gpu") else "core")
    s = run(ph, seq, coils=coils).signal.numpy().reshape(len(t2), -1, 91)[:, :, 0]
    return s[:, -10:]


def residual(m, S):
    a = (S.conj() @ m) / np.einsum("ij,ij->i", S.conj(), S).real
    return np.linalg.norm(m[None] - a[:, None] * S, axis=1) / np.linalg.norm(m), a


def main():
    fetch()
    CACHE.mkdir(parents=True, exist_ok=True)
    rec_path = CACHE / "recon.npz"
    if not rec_path.exists():
        subprocess.run([sys.executable, str(HERE / "spiral_tse_recon.py"), "recon"], check=True)
    rec = np.load(rec_path)
    seq = first_trs(read_pulseq(DATA / f"{NAME}.seq"))
    rows, truth = [], []
    for j, m in enumerate(rec["curves"]):
        (A, T2m), _ = curve_fit(lambda t, A, T: A * np.exp(-t / T), TE, m.real, p0=(m.real[0], 0.1), maxfev=10000)
        mono = np.linalg.norm(m.real - A * np.exp(-TE / T2m)) / np.linalg.norm(m)
        t1 = 1.5 if T2m > 0.08 else 1.0
        T2s = T2m * np.geomspace(0.6, 1.8, 31)
        S = echoes(seq, np.full(31, t1), T2s, np.ones(31))
        r, a = residual(m, S)
        k = int(np.argmin(r))
        rows.append((T2m, mono, T2s[k], r[k]))
        truth.append(a[k] * S[k])
    np.save(CACHE / "truth.npy", np.array(truth))
    dig_path = CACHE / "digital.npz"
    if not dig_path.exists():
        subprocess.run([sys.executable, str(HERE / "spiral_tse_recon.py"), "digital", CACHE / "truth.npy", dig_path],
                       check=True)
    dig = np.load(dig_path)
    floor = [residual(d, t[None])[0][0] for d, t in zip(dig["noisy"], truth)]
    clean = [residual(d, t[None])[0][0] for d, t in zip(dig["clean"], truth)]
    print("tube   mono T2 [ms]  mono resid | Bloch T2 [ms]  Bloch resid | method: noise-free, with noise")
    for j, ((T2m, mono, T2b, rb), fl, cl) in enumerate(zip(rows, floor, clean)):
        print(f"{j + 1:4d}   {1e3 * T2m:9.1f}   {100 * mono:8.2f} % | {1e3 * T2b:10.1f}   {100 * rb:9.2f} % |"
              f" {100 * cl:5.2f} %, {100 * fl:5.2f} %")


if __name__ == "__main__":
    main()
