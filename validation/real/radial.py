"""Benchmark 6: gradient timing of a Pulseq radial scan (Siemens MAGNETOM Cima.X, 2.89 T, PTB).

Data: Zenodo 10.5281/zenodo.14228459 (CC BY 4.0), fetched by ``fetch_radial.py``: a golden-angle
radial 2D GRE (5 slices of 8 mm, 402 spokes each, 256 samples at 10 us, no oversampling, FOV
256 mm), its ``.seq`` and the acquired ISMRMRD file with the nominal trajectory.

1. ``trajectory``: blochparty's Pulseq import gives the file's nominal trajectory (k at every ADC
   sample, sample i at (i + 1/2) dwell).
2. ``delays``: the measured echo shift of every spoke from its nearly antiparallel partner (same
   line, opposite direction: object and coil phase cancel, the delay doubles), fitted by per-axis
   delays d(theta) = dx cos^2 + dy sin^2 + 2 dxy sin cos; fitted on some slices, predicting others.
3. ``simulate``: the same estimator on blochparty simulations of the ``.seq`` with gradient delays
   (``GIRF.identity``): the engine's timing model gives back what was imposed, and the estimator's
   own bias (unequal x / y delays shift antiparallel partners sideways in opposite directions, so
   an off-centre object tilts their cross-phase) is negligible for an object like the measured one.

    uv run python validation/real/radial.py [trajectory|delays|simulate|all]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).with_name("data") / "radial"
SEQ = DATA / "radial_2D_5slices_402spokes_golden_angle.seq"
H5 = DATA / "radial_2D_5slices_402spokes_golden_angle_data_with_traj.h5"
NSL, NSP, NS, NC = 5, 402, 256, 8
DWELL = 10e-6
FOV = 0.256


def load():
    """k-space (slices, spokes, coils, samples) and the file's trajectory (slices, spokes, samples, 2)
    [1 / FOV], in acquisition order."""
    import h5py

    with h5py.File(H5, "r") as f:
        d = f["dataset/data"]
        head = d["head"]
        sl, step = head["idx"]["slice"], head["idx"]["kspace_encode_step_1"]
        k = np.zeros((NSL, NSP, NC, NS), np.complex64)
        traj = np.zeros((NSL, NSP, NS, 2), np.float32)
        for i, (raw, t) in enumerate(zip(d["data"], d["traj"])):
            k[sl[i], step[i]] = raw.view(np.complex64).reshape(NC, NS)
            traj[sl[i], step[i]] = t.reshape(NS, 2)
    return k, traj


def nominal_k(seq) -> np.ndarray:
    """(n_adc, 3) k [1/m] at the ADC samples of a blochparty Sequence, from the last RF segment."""
    g = (seq.grad * seq.dt[:, None]).detach().double().numpy()
    out, kk = [], np.zeros(3)
    for s in range(len(seq)):
        kk = np.zeros(3) if seq.is_rf[s] else kk + g[s]
        if seq.adc[s]:
            out.append(kk.copy())
    return np.array(out)


def trajectory():
    from blochparty.pulseq import read_pulseq

    _, traj = load()
    seq = read_pulseq(SEQ)
    k = nominal_k(seq).reshape(NSL, NSP, NS, 3) * FOV
    # ISMRMRD stores (x, y) in the logical frame of the converter; find the axis map by fit
    best = None
    for sx in (1, -1):
        for sy in (1, -1):
            for swap in (False, True):
                a, b = (k[..., 1], k[..., 0]) if swap else (k[..., 0], k[..., 1])
                err = np.abs(np.stack([sx * a, sy * b], -1) - traj).max()
                if best is None or err < best[0]:
                    best = (err, sx, sy, swap)
    err, sx, sy, swap = best
    print(f"trajectory: max |k_import - k_file| = {err:.2e} / FOV (of max {np.abs(traj).max():.0f}); "
          f"file x = {'+' if sx > 0 else '-'}{'y' if swap else 'x'}, y = {'+' if sy > 0 else '-'}{'x' if swap else 'y'}")
    return best


def angles(traj) -> np.ndarray:
    """(slices, spokes) direction [rad] of each spoke, from its last sample."""
    return np.arctan2(traj[..., -1, 1], traj[..., -1, 0])


def spoke_shifts(k, ang, tol_deg=0.3):
    """Per slice: (theta, shift [samples]) of each spoke with an antiparallel partner within tol.
    The partner is reversed in time; with the readout delayed by d samples the two echoes move
    apart by 2 d, which is the phase slope of the cross-power of their projections."""
    out = []
    x = np.arange(NS) - NS // 2
    for s in range(k.shape[0]):
        proj = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(k[s], axes=-1), axis=-1), axes=-1)  # (spokes, coils, NS)
        th, dl = [], []
        a = ang[s]
        for i in range(k.shape[1]):
            diff = np.abs((a - a[i] - np.pi + np.pi) % (2 * np.pi) - np.pi)
            diff[i] = np.inf
            j = int(np.argmin(diff))
            if diff[j] > math.radians(tol_deg):
                continue
            # partner reversed: sample n of j sits at -k_n of i; shift by one sample for the
            # (i + 1/2) grid: k_n = n - 128 of i, partner's sample 256 - n lies at the same k
            rev = np.roll(k[s, j, :, ::-1], 1, axis=-1)
            pb = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(rev, axes=-1), axis=-1), axes=-1)
            c = (proj[i] * np.conj(pb)).sum(0)
            w = np.abs(c)
            m = w > 0.2 * w.max()
            slope = np.polyfit(x[m], np.unwrap(np.angle(c[m])), 1, w=w[m])[0]
            th.append(a[i])
            dl.append(slope * NS / (2 * np.pi) / 2)
        out.append((np.array(th), np.array(dl)))
    return out


def design(th):
    return np.stack([np.cos(th) ** 2, np.sin(th) ** 2, 2 * np.sin(th) * np.cos(th)], 1)


def fit(pairs):
    th = np.concatenate([p[0] for p in pairs])
    dl = np.concatenate([p[1] for p in pairs])
    coef, *_ = np.linalg.lstsq(design(th), dl, rcond=None)
    return coef, float(np.std(dl - design(th) @ coef))


def delays(k=None, traj=None, quiet=False):
    if k is None:
        k, traj = load()
    pairs = spoke_shifts(k, angles(traj))
    coef, res = fit(pairs)
    if not quiet:
        us = coef * DWELL * 1e6
        print(f"delays (all {sum(len(p[0]) for p in pairs)} pairs): dx {us[0]:+.3f}, dy {us[1]:+.3f}, "
              f"dxy {us[2]:+.3f} us; scatter about the model {res * DWELL * 1e6:.3f} us")
        for train in ((0, 2, 4), (1, 3)):
            test = [s for s in range(NSL) if s not in train]
            c, _ = fit([pairs[s] for s in train])
            for s in test:
                th, dl = pairs[s]
                pred = design(th) @ c
                print(f"  fit slices {train} -> slice {s}: rms(measured - predicted) {np.sqrt(np.mean((dl - pred) ** 2)) * 10:.3f} us, "
                      f"rms(measured - 0) {np.sqrt(np.mean(dl ** 2)) * 10:.3f} us")
    return coef, res, pairs


SLICES_Z = (-0.016, -0.008, 0.0, 0.008, 0.016)  # RF offsets -6250 .. 6250 Hz over 390625 Hz/m


def phantom(kind: str, seed=0):
    """Spins on a jittered 1.5 mm grid (no grid aliasing) in each slice: ``"ellipse"``, an
    asymmetric object (an ellipse 2 cm off centre with a denser disc), or ``"disc"``, a 19 cm disc
    3 / 5 mm off centre like the measured phantom (projection centroids, ``main``)."""
    import torch

    import blochparty as bp

    g = torch.Generator().manual_seed(seed)
    h = 1.5e-3
    ax = torch.arange(-0.1, 0.1, h, dtype=torch.float64)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    xy = torch.stack([xx.ravel(), yy.ravel()], 1) + (torch.rand(xx.numel(), 2, generator=g, dtype=torch.float64) - 0.5) * h
    if kind == "ellipse":
        xy = xy[((xy[:, 0] - 0.02) / 0.08) ** 2 + ((xy[:, 1] + 0.01) / 0.06) ** 2 < 1]
        pd = 1 + 1.5 * (((xy[:, 0] + 0.03) ** 2 + (xy[:, 1] - 0.02) ** 2) < 0.015**2).double()
    else:
        xy = xy[((xy[:, 0] + 0.003) ** 2 + (xy[:, 1] - 0.005) ** 2) < 0.095**2]
        pd = torch.ones(len(xy), dtype=torch.float64)
    pos = torch.cat([torch.cat([xy, torch.full((len(xy), 1), z)], 1) for z in SLICES_Z])
    n = len(pos)
    one = torch.ones(n, dtype=torch.float64)
    return bp.Phantom(pos, pd.repeat(len(SLICES_Z)), 0.8 * one, 0.08 * one, 0 * one)


def ring_coils(pos):
    """(N, C) receive weights of eight 5 cm loops on a 13 cm ring (quasi-static, ``loops.py``)."""
    from blochparty.loops import Loop, fields, receive_sensitivities

    loops = [Loop((0.13 * math.cos(a), 0.13 * math.sin(a), 0.0), (-math.cos(a), -math.sin(a), 0.0), 0.05)
             for a in (2 * math.pi * i / 8 for i in range(8))]
    w = receive_sensitivities(fields(loops, pos)[0])
    return w / w.abs().max()


CASES = (  # (phantom, coils, imposed dx, dy [us])
    ("ellipse", False, 0.0, 0.0), ("ellipse", False, 3.0, 3.0), ("ellipse", False, 3.0, -2.0),
    ("ellipse", False, 0.2, 0.65), ("disc", True, 0.2, 0.65), ("disc", True, 3.0, -2.0))


def simulate():
    """Simulate the .seq with per-axis gradient delays and estimate them back with ``delays``."""
    import time

    import torch

    from blochparty import metal
    from blochparty.girf import GIRF, apply_girf
    from blochparty.pulseq import read_pulseq

    _, traj = load()
    seq = read_pulseq(SEQ)
    print(f"simulate: {len(seq):,} segments")
    for kind, with_coils, dx, dy in CASES:
        t0 = time.perf_counter()
        ph = phantom(kind)
        coils = ring_coils(ph.pos).to(torch.complex64) if with_coils else None
        played = apply_girf(seq, GIRF.identity(torch.tensor([dx, dy, 0.0], dtype=torch.float64) * 1e-6))
        sig = metal.simulate_metal(ph.to(torch.float32), played.to(torch.float32), coils).signal  # (C, n_adc)
        c = sig.shape[0]
        k = sig.to(torch.complex64).numpy().reshape(c, NSL, NSP, NS).transpose(1, 2, 0, 3)
        coef, res, _ = delays(k, traj, quiet=True)
        us = coef * DWELL * 1e6
        print(f"  {kind:7s} {'8 loops' if with_coils else '1 coil ':7s} imposed ({dx:+.2f}, {dy:+.2f}) us -> "
              f"({us[0]:+.3f}, {us[1]:+.3f}), dxy {us[2]:+.3f} us; scatter {res * DWELL * 1e6:.3f} us "
              f"({time.perf_counter() - t0:.0f} s)")


def centroids():
    """Object centre per slice from the RSS projections' centroids c(theta) = x0 cos + y0 sin."""
    k, traj = load()
    ang = angles(traj)
    x = np.arange(NS) - NS // 2  # mm (1 mm samples in image space)
    for s in range(NSL):
        p = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(k[s], axes=-1), axis=-1), axes=-1)
        m = np.sqrt((np.abs(p) ** 2).sum(1))
        m = np.where(m > 0.1 * m.max(), m, 0)
        c = (m * x).sum(1) / m.sum(1)
        a = np.stack([np.cos(ang[s]), np.sin(ang[s])], 1)
        (x0, y0), *_ = np.linalg.lstsq(a, c, rcond=None)
        print(f"  slice {s}: object centre ({x0:+.1f}, {y0:+.1f}) mm, width {(m > 0).sum(1).min()}-{(m > 0).sum(1).max()} mm")


def main(argv):
    what = argv[1] if len(argv) > 1 else "all"
    if what in ("trajectory", "all"):
        trajectory()
    if what in ("delays", "all"):
        delays()
        centroids()
    if what in ("simulate", "all"):
        simulate()


if __name__ == "__main__":
    main(sys.argv)
