"""Benchmark 5, the measurement side (numpy, scipy, h5py, finufft; never torch, see
``scholand_recon.py``): the PTB spiral multi-echo TSE of a T1MES phantom (``spiral_tse.py``).

* ``recon``: per echo, CG-SENSE (30 iterations, Tikhonov 1e-3) with the trajectory stored in the
  ISMRMRD file (matrix-index units, spiral-out from the k-space centre at sample 0: the header's
  centre sample 45 is wrong). Coil maps: ratio method on per-coil least-squares images of echoes
  1-4 together, smoothed (160 interleaves x 91 samples per echo for 160^2: undersampled).
* ROIs: the 3 x 3 tube grid, centres refined by centre of mass, radius 6 px; curves are complex
  ROI means, phased by the first three echoes.
* ``digital``: known curves on discs (2x finer grid) through the same trajectory, coil maps and
  reconstruction, with noise matched to the measured image background: the method's floor.

    uv run python validation/real/spiral_tse_recon.py recon|digital <truth.npy>
"""

from __future__ import annotations

import sys
from pathlib import Path

import finufft
import h5py
import numpy as np
from scipy import ndimage

DATA = Path(__file__).with_name("data") / "spiral_tse"
MRD = DATA / "spiral_tse_tr_4000_te_16_nechos_10_nspirals_160_with_traj.mrd"
CACHE = DATA / "cache"
N, NE, C, NSAMP = 160, 10, 16, 91


def raw():
    with h5py.File(MRD, "r") as f:
        d = f["dataset/data"]
        echo = d["head"]["idx"]["contrast"][:]
        traj = np.stack([t.reshape(NSAMP, 3)[:, :2] for t in d["traj"]]).astype(np.float64)
        data = np.stack([x.reshape(C, NSAMP, 2) for x in d["data"]])
    return traj, (data[..., 0] + 1j * data[..., 1]).astype(np.complex128), echo


def coords(traj, sel):
    k = traj[sel].reshape(-1, 2)
    return 2 * np.pi * k[:, 0] / N, 2 * np.pi * k[:, 1] / N


def coil_images(traj, data, echo, echoes=range(4), iters: int = 20, reg: float = 1e-3):
    """Per-coil least-squares images from the readouts of ``echoes`` together (CG on the normal
    equations; no density compensation needed). A Pipe-style iteration w <- w / |A A^H w| with
    the NUFFT Gram operator does not converge (its kernel is not positive); it gave run-to-run
    different images and coil maps."""
    sel = np.isin(echo, list(echoes))
    x, y = coords(traj, sel)
    yd = np.ascontiguousarray(data[sel].transpose(1, 0, 2).reshape(C, -1))
    p1 = finufft.Plan(1, (N, N), n_trans=C, eps=1e-6)
    p1.setpts(x, y)
    p2 = finufft.Plan(2, (N, N), n_trans=C, eps=1e-6)
    p2.setpts(x, y)
    b = p1.execute(yd)
    lam = reg * np.abs(b).max() / np.abs(yd).max()
    X = np.zeros((C, N, N), complex)
    r, p = b.copy(), b.copy()
    rs = np.vdot(r, r).real
    for _ in range(iters):
        Ap = p1.execute(p2.execute(np.ascontiguousarray(p))) + lam * p
        a = rs / np.vdot(p, Ap).real
        X += a * p
        r -= a * Ap
        rn = np.vdot(r, r).real
        p = r + rn / rs * p
        rs = rn
    return X


def coil_maps(acc):
    rss = np.sqrt((np.abs(acc) ** 2).sum(0))
    sens = np.empty_like(acc)
    den = ndimage.gaussian_filter(rss, 6) + 1e-12
    for c in range(C):
        sens[c] = (ndimage.gaussian_filter(acc[c].real, 6) + 1j * ndimage.gaussian_filter(acc[c].imag, 6)) / den
    return sens / (np.sqrt((np.abs(sens) ** 2).sum(0)) + 1e-12)


def sense(yd, x, y, sens, iters: int = 30, reg: float = 1e-3):
    p1 = finufft.Plan(1, (N, N), n_trans=C, eps=1e-6)
    p1.setpts(x, y)
    p2 = finufft.Plan(2, (N, N), n_trans=C, eps=1e-6)
    p2.setpts(x, y)
    fwd = lambda im: p2.execute(np.ascontiguousarray(sens * im[None]))
    adj = lambda v: (np.conj(sens) * p1.execute(np.ascontiguousarray(v))).sum(0)
    b = adj(yd)
    lam = reg * np.abs(b).max() / np.abs(yd).max()
    X = np.zeros((N, N), complex)
    r, p = b.copy(), b.copy()
    rs = np.vdot(r, r).real
    for _ in range(iters):
        Ap = adj(fwd(p)) + lam * p
        a = rs / np.vdot(p, Ap).real
        X += a * p
        r -= a * Ap
        rn = np.vdot(r, r).real
        p = r + rn / rs * p
        rs = rn
    return X


def tube_centres(img):
    yy, xx = np.indices(img.shape)
    out = []
    for r0 in (46, 79, 112):
        for c0 in (48, 81, 114):
            c = np.array([r0, c0], float)
            for _ in range(5):
                c = np.array(ndimage.center_of_mass(img * (np.hypot(yy - c[0], xx - c[1]) < 9)))
            out.append(c)
    return np.array(out)


def curves(X, centres, radius: float = 6.0):
    yy, xx = np.indices((N, N))
    out = []
    for c in centres:
        r = np.hypot(yy - c[0], xx - c[1]) < radius
        z = np.array([Xe[r].mean() for Xe in X])
        out.append(z * np.exp(-1j * np.angle(z[:3].sum())))
    return np.array(out)


def background_std(X):
    yy, xx = np.indices((N, N))
    m = np.hypot(yy - N / 2, xx - N / 2) > 72
    return float(np.std(np.concatenate([Xe[m].real for Xe in X])))


def recon():
    traj, data, echo = raw()
    sens = coil_maps(coil_images(traj, data, echo))
    X = np.array(
        [
            sense(
                np.ascontiguousarray(data[echo == e].transpose(1, 0, 2).reshape(C, -1)), *coords(traj, echo == e), sens
            )
            for e in range(NE)
        ]
    )
    cents = tube_centres(np.abs(X[0]) + np.abs(X[1]))
    np.savez(CACHE / "recon.npz", X=X, sens=sens, centres=cents, curves=curves(X, cents), noise=background_std(X))


def digital(truth_path, out_path, seed: int = 0):
    """Known (9, NE) curves on radius-8 px discs through the acquisition and reconstruction."""
    rec = np.load(CACHE / "recon.npz")
    truth = np.load(truth_path)
    traj, _, echo = raw()
    sens, cents = rec["sens"], rec["centres"]
    M = 2 * N
    yy, xx = (np.indices((M, M)) + 0.5) / 2 - 0.5
    discs = [(np.hypot(yy - c[0], xx - c[1]) < 8.0).astype(complex) for c in cents]
    sens2 = np.array([ndimage.zoom(s.real, 2, order=1) + 1j * ndimage.zoom(s.imag, 2, order=1) for s in sens])
    rng = np.random.default_rng(seed)

    def run(sigma):
        X = []
        for e in range(NE):
            x, y = coords(traj, echo == e)
            p2 = finufft.Plan(2, (M, M), n_trans=C, eps=1e-7)
            p2.setpts(x / 2, y / 2)
            yd = sum(cv[e] * p2.execute(np.ascontiguousarray(sens2 * d[None])) / 4.0 for d, cv in zip(discs, truth))
            yd = yd + sigma * (rng.standard_normal(yd.shape) + 1j * rng.standard_normal(yd.shape)) / np.sqrt(2)
            X.append(sense(yd, x, y, sens))
        return np.array(X)

    clean = run(0.0)
    unit = background_std(run(1.0) - clean)
    noisy = run(float(rec["noise"]) / unit)
    np.savez(out_path, clean=curves(clean, cents), noisy=curves(noisy, cents))


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "recon":
        recon()
    elif sys.argv[1] == "digital":
        digital(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
