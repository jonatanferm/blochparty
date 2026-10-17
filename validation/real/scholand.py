"""Benchmark 3: radial single-shot IR-bSSFP of the NIST T2 plane (Scholand et al. 2023).

Data: Scholand N, Wang X, Roeloffs V, Rosenzweig S, Uecker M. Quantitative MRI by nonlinear
inversion of the Bloch equations. MRM 2023;90:520-538. Zenodo doi:10.5281/zenodo.7654462
(version 7654462, CC BY 4.0): ``data_05b_kspace`` (IR-bSSFP), ``data_05b_b1map`` (vendor
preconditioned-TurboFLASH B1 map), ``data_GSM_t1`` / ``data_GSM_t2`` (Cartesian IR-SE and SE
gold standards), all of the same slice, 3 T Siemens Skyra, 20-channel head coil, ~21 C.

Protocol (Zenodo record, paper Table 1, the authors' scripts github.com/mrirecon/bloch-moba
``05b_IR-bSSFP_NIST``, and BART's simulator ``src/simu``): non-selective hyperbolic-secant
inversion (10 ms; BART: B1max 14 uT, beta 800 rad/s, mu 4.9) and a 5 ms spoiler, an
alpha/2 - TR/2 preparation, then 1000 TRs of 4.88 ms, TE 2.44 ms, 45 deg Hamming-windowed sinc
(1 ms, BWTP 4, 5 mm slice), RF phase alternating 0 / pi; one radial spoke per TR (384 samples,
2x oversampled, FOV 200 mm, tiny golden angle pi / (phi + 6)).

Not in any document, so modelled (and varied in the analysis): the exact gradient waveforms
(trapezoids at Skyra-like limits), the gap between the spoiler and the preparation, the
receiver phase (follows the RF phase: the data do not alternate).

    uv run python validation/real/fetch_scholand.py && uv run python validation/real/scholand_irbssfp.py
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import numpy as np
import torch

import blochparty as bp
from blochparty.constants import GAMMA_BAR

DATA = Path(__file__).with_name("data") / "scholand"
TR, TE, FA, T_RF, BWTP, SLICE = 4.88e-3, 2.44e-3, 45.0, 1e-3, 4.0, 5e-3
N_TR, N_READ, BR, FOV = 1000, 384, 192, 0.200
HS = dict(duration=10e-3, b1max=14e-6, beta=800.0, mu=4.9)  # BART pulse_hypsec_defaults
SPOIL = 5e-3
GOLDEN = math.pi / ((1 + math.sqrt(5)) / 2 + 6)  # tiny golden angle #7
SKYRA = bp.System(max_grad=43e-3 * GAMMA_BAR, max_slew=180.0 * GAMMA_BAR)


def readcfl(name: str | Path) -> np.ndarray:
    """A BART .cfl/.hdr pair as a complex64 array (Fortran order, trailing 1s dropped)."""
    name = str(name)
    dims = [int(x) for x in open(name + ".hdr").read().split("\n")[1].split()]
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()
    return np.fromfile(name + ".cfl", dtype=np.complex64).reshape(dims, order="F")


def hypsec(raster: float = 10e-6) -> np.ndarray:
    """(n, 2) [Hz] the hyperbolic-secant inversion as BART defines it: amplitude
    A sech(beta t'), phase mu log(sech(beta t')) + mu log(A), t' from the pulse centre."""
    n = round(HS["duration"] / raster)
    t = (np.arange(n) + 0.5) * raster - 0.5 * HS["duration"]
    a = HS["b1max"] * GAMMA_BAR / np.cosh(HS["beta"] * t)
    ph = HS["mu"] * np.log(1 / np.cosh(HS["beta"] * t))
    return np.stack([a * np.cos(ph), a * np.sin(ph)], -1)


def sequence(
    *,
    n_tr: int = N_TR,
    inversion: str = "hypsec",
    slice_select: bool = True,
    gap: float = 0.0,
    fa: float = FA,
    system: bp.System = SKYRA,
) -> bp.Sequence:
    """The IR-bSSFP as blochparty segments with one ADC sample per TR, at the echo (the time
    of the k-space centre). Gradients only along z (the in-plane readout is balanced and does
    not change the magnetization at the echo). ``inversion``: "hypsec", "perfect" (an
    instantaneous 180) or "none"; ``slice_select`` False makes every pulse non-selective."""
    sb = bp.SequenceBuilder(system=system)
    if inversion == "hypsec":
        sb.rf_waveform(torch.as_tensor(hypsec()), 10e-6)
    elif inversion == "perfect":
        sb.rf(1e-6, flip_deg=180.0)
    sb.delay(SPOIL).spoil()
    if gap:
        sb.delay(gap)

    def make(flip, phase):
        rf, gz, reph = bp.sinc_pulse(
            flip, T_RF, slice_thickness=SLICE, time_bw_product=BWTP, apodization=0.46, phase=phase, system=system
        )
        return (rf, gz, reph) if slice_select else (rf, None, None)

    def play(rf, gz):
        """Append the pulse block; returns the time of the pulse centre."""
        t0 = sb.time
        sb.block(*([gz] if gz is not None else []), rf=rf)
        return t0 + rf.delay + 0.5 * T_RF

    # alpha/2 (phase pi) TR/2 before the first alpha (phase 0), then alternating
    rf, gz, reph = make(fa / 2, math.pi)
    centre = play(rf, gz)
    if reph is not None:
        sb.trapezoid(area=reph)
    nxt = centre + TR / 2
    for i in range(n_tr):
        rf, gz, reph = make(fa, math.pi * (i % 2))
        pre = None if reph is None else bp.trapezoid(area=reph, system=system)
        wait = nxt - rf.delay - 0.5 * T_RF - (pre.duration if pre is not None else 0.0) - sb.time
        if wait < -1e-9:
            raise ValueError("TR too short for the gradients")
        if wait > 1e-9:
            sb.delay(wait)
        if pre is not None:
            sb.block(pre)  # prewinder: the moment from the pulse start to its centre (= the rephase area)
        centre = play(rf, gz)
        if reph is not None:
            sb.trapezoid(area=reph)
        sb.delay(centre + TE - 0.5e-6 - sb.time)
        sb.readout(1, 1e-6, phase=math.pi * (i % 2))
        nxt = centre + TR
    return sb.build()


def slice_spins(n: int = 201, extent: float = 4 * SLICE) -> torch.Tensor:
    """z positions [m] across ``extent`` (the slice profile)."""
    return torch.linspace(-extent / 2, extent / 2, n, dtype=torch.float64)


# ---- simulation side: dictionary and basis, digital-phantom truth, fits (torch; no finufft) ----

CACHE = DATA / "cache"
RF_RASTER = 10e-6  # piecewise-constant RF on 10 us: 3e-4 from the 1 us raster over all 1000 TRs
MODELS = {
    "full": {},
    "perfect_inv": {"inversion": "perfect"},
    "no_slice": {"slice_select": False},
    "simple": {"inversion": "perfect", "slice_select": False},
}


def _seq(**kw) -> bp.Sequence:
    import dataclasses

    return sequence(system=dataclasses.replace(SKYRA, rf_raster=RF_RASTER), **kw)


def signals(t1, t2, b1, df=None, *, n_z: int = 101, batch: int = 64, **model) -> np.ndarray:
    """(n, N_TR) complex echo signals, one per (t1, t2, b1, df) atom, each the mean over n_z
    spins across 4 slice thicknesses (one spin at the centre without slice selection). Runs on
    the GPU when available, else the PyTorch core (slow)."""
    t1, t2, b1 = (np.atleast_1d(np.asarray(v, float)) for v in (t1, t2, b1))
    df = np.zeros_like(t1) if df is None else np.atleast_1d(np.asarray(df, float))
    seq = _seq(**model)
    z = slice_spins(n_z) if model.get("slice_select", True) else torch.zeros(1, dtype=torch.float64)
    nz = len(z)
    run = functools.partial(bp.simulate, engine="gpu" if bp.engines.available("gpu") else "core")
    out = []
    for a in range(0, len(t1), batch):
        m = min(batch, len(t1) - a)
        rep = lambda v: torch.as_tensor(np.repeat(v[a : a + m], nz), dtype=torch.float64)
        pos = torch.zeros(m * nz, 3, dtype=torch.float64)
        pos[:, 2] = z.repeat(m)
        ph = bp.Phantom(
            pos=pos, pd=torch.ones(m * nz, dtype=torch.float64), t1=rep(t1), t2=rep(t2), db0=rep(df), b1=rep(b1)
        )
        coils = torch.zeros(m * nz, m, dtype=torch.complex64)
        coils[torch.arange(m * nz), torch.arange(m * nz) // nz] = 1.0 / nz
        out.append(run(ph, seq, coils=coils).signal.numpy())
    return np.concatenate(out)


def basis(n_keep: int = 24) -> np.ndarray:
    """Temporal basis for the subspace reconstruction, from the union of two dictionaries so it
    does not favour the model under test: non-selective pulses with relative flips 0.15-1.3
    (covering the slice profile's flips) and |df| <= 40 Hz; and the slice-selective sequence with
    both inversion models, B1 0.8-1.2, |df| <= 20 Hz. T1 0.05-4 s, T2 4 ms-2 s (T2 <= T1)."""
    g1 = np.array(
        np.meshgrid(
            np.geomspace(0.05, 4, 20),
            np.geomspace(0.004, 2, 20),
            np.linspace(0.15, 1.3, 10),
            np.linspace(-40, 40, 5),
            indexing="ij",
        )
    ).reshape(4, -1)
    g1 = g1[:, g1[1] <= g1[0]]
    g2 = np.array(
        np.meshgrid(
            np.geomspace(0.05, 4, 20),
            np.geomspace(0.004, 2, 20),
            np.linspace(0.8, 1.2, 5),
            np.linspace(-20, 20, 5),
            indexing="ij",
        )
    ).reshape(4, -1)
    g2 = g2[:, g2[1] <= g2[0]]
    d = [signals(*g1, slice_select=False, batch=128)]
    d += [signals(*g2, inversion=inv) for inv in ("hypsec", "perfect")]
    d = np.concatenate([x / np.linalg.norm(x, axis=1, keepdims=True) for x in d])
    _, _, vh = np.linalg.svd(d, full_matrices=False)
    return vh[:n_keep]


def fit(curves: np.ndarray, t1_ref, t2_centre, b1, model: str = "full", t0: int = 20):
    """Per-sphere (T1, T2) by a coarse then a fine grid: complex amplitude solved in closed
    form, residual over TRs t0.. (the first 0.1 s is reconstruction-limited). Returns
    (14, 3) [T1, T2, relative residual] and a flag per sphere when the optimum is on the grid edge."""
    res, edge = [], []
    for j, m in enumerate(curves):
        T1s = t1_ref[j] * np.geomspace(0.6, 1.6, 15)
        T2s = t2_centre[j] * np.geomspace(0.15, 4.0, 25)
        on_edge = False
        for stage in range(2):
            G = np.array(np.meshgrid(T1s, T2s, indexing="ij")).reshape(2, -1)
            S = signals(G[0], G[1], np.full(G.shape[1], b1[j]), **MODELS[model])[:, t0:]
            a = (S.conj() @ m[t0:]) / np.einsum("ij,ij->i", S.conj(), S).real
            r = (np.linalg.norm(m[t0:][None] - a[:, None] * S, axis=1) / np.linalg.norm(m[t0:])).reshape(
                len(T1s), len(T2s)
            )
            i, k = np.unravel_index(np.argmin(r), r.shape)
            on_edge |= i in (0, len(T1s) - 1) or k in (0, len(T2s) - 1)
            best = (T1s[i], T2s[k], r[i, k])
            T1s, T2s = T1s[i] * np.geomspace(0.9, 1.1, 15), T2s[k] * np.geomspace(0.85, 1.15, 17)
        res.append(best)
        edge.append(on_edge)
    return np.array(res), np.array(edge)
