"""Benchmark 3, the measurement side (numpy, scipy and finufft only: finufft and torch each
load an OpenMP runtime and cannot share a process, so this module never imports torch).

* ``gold``: the Cartesian IR-SE / SE gold standards per sphere. Coils are combined with the
  complex ROI signal of a reference image, and the real part is fitted (Gaussian noise, no
  magnitude floor): IR a - b exp(-TI/T1); SE a exp(-TE/T2).
* ``b1``: the vendor B1 map (preconditioned TurboFLASH: arccos(prepared / PD) / 80 deg) and a
  smooth 4th-order polynomial fit to the background water. Inside short-T1 spheres the map reads
  low (0.87 against 1.09 in sphere 14, T1 0.13 s: recovery during the TurboFLASH readout raises
  the prepared image); spheres 1-5 (T1 >= 1.5 s) agree with the fit to ~0.02, so the fit is used.
* ``delays``: gradient delays from pairs of nearly antiparallel steady-state spokes (the phase
  slope of one projection against the reversed other is twice the k-space shift), fitted to
  an anisotropic delay d(theta) = dxx cos^2 + dyy sin^2 + 2 dxy sin cos.
* ``recon``: low-rank subspace reconstruction: images x(t) = sum_k phi_k(t) X_k with a temporal
  basis phi from simulated signals (``scholand.py basis``); 8 SVD-compressed coils, ratio-method
  coil maps from the last 500 spokes, ramp-weighted least squares by conjugate gradients.
* ``curves``: per-sphere complex ROI means (the authors' ROIs; the radial set lists sphere 14
  first) as curves over the 1000 TRs.
* ``digital``: the same acquisition from a digital phantom (sphere discs on a 2x finer grid
  with given curves, measured coil maps, trajectory and noise covariance), to measure what the
  reconstruction itself does to known curves.

    uv run python validation/real/scholand_recon.py gold|b1|delays|recon K|curves K|digital ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import finufft
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit

DATA = Path(__file__).with_name("data") / "scholand"
CACHE = DATA / "cache"
N, NS, NT = 192, 384, 1000  # base resolution, samples per spoke (2x oversampled), spokes
GOLDEN = np.pi / ((1 + np.sqrt(5)) / 2 + 6)
TI = 30e-3 + 0.25 * np.arange(11)
TE_SE = 15e-3 + 0.04 * np.arange(12)


def readcfl(name) -> np.ndarray:
    name = str(name)
    dims = [int(x) for x in open(name + ".hdr").read().split("\n")[1].split()]
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()
    return np.fromfile(name + ".cfl", dtype=np.complex64).reshape(dims, order="F")


def rois(kind: str) -> list[np.ndarray]:
    """Sphere ROIs in sphere order 1..14. The radial set (192^2, transposed to our image
    orientation) lists sphere 14 first, as the authors' reference table does; the Cartesian
    set (256^2) lists sphere 1 first. Verified on the signals: each radial ROI's curve matches
    its sphere's simulated curve best (14 x 14 residual matrix)."""
    files = [DATA / f"rois_{kind}" / f"v{i:03d}" for i in range(14)]
    if kind == "radial":
        return [(readcfl(f).real > 0.5).T for f in files][::-1]
    return [readcfl(f).real > 0.5 for f in files]


# ---- gold standards -------------------------------------------------------------------------

def _cartesian(name: str) -> np.ndarray:
    k = readcfl(DATA / name)  # (512, 256, 1, 20, 1, n)
    k = k.reshape(k.shape[0], k.shape[1], k.shape[3], k.shape[5])
    im = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))[128:384]
    return np.transpose(im, (3, 2, 0, 1))  # (n, coils, 256, 256)


def gold() -> dict:
    out = {}
    masks = rois("cartesian")
    for name, times, ref in (("t1", TI, -1), ("t2", TE_SE, 0)):
        im = _cartesian(f"data_GSM_{name}")
        sig = np.array([im[:, :, r].mean(-1) for r in masks])  # (sphere, n, coils)
        z = np.array([s @ np.conj(s[ref] / np.linalg.norm(s[ref])) for s in sig])
        out[name + "_signal"] = z
    t1, t2 = [], []
    for z1, z2 in zip(out["t1_signal"], out["t2_signal"]):
        y = z1.real / np.abs(z1).max()
        p, c = curve_fit(lambda t, a, b, T: a - b * np.exp(-t / T), TI, y, p0=(1, 2, 1.0), maxfev=20000)
        t1.append((p[2], np.sqrt(c[2, 2]), p[1] / p[0]))
        y = z2.real / np.abs(z2).max()
        p, c = curve_fit(lambda t, a, T: a * np.exp(-t / T), TE_SE, y, p0=(2, 0.05), maxfev=20000)
        t2.append((p[1], np.sqrt(c[1, 1])))
    out["t1"], out["t2"] = np.array(t1), np.array(t2)  # (value, std, [inversion ratio]) [s]
    return out


# ---- B1 ---------------------------------------------------------------------------------------

def b1() -> dict:
    raw = readcfl(DATA / "data_05b_b1map").reshape(304, 192, 20, 2)  # asymmetric echo: 304 of 384
    full = np.zeros((384, 192, 20, 2), complex)
    full[80:] = raw  # the k-space centre is at sample 111 = 191 - 80
    im = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(full, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))[96:288]
    pd, pre = im[..., 0], im[..., 1]
    w = np.conj(pd)
    ratio = ((w * pre).sum(-1) / (w * pd).sum(-1)).real
    kappa = np.degrees(np.arccos(np.clip(ratio, -1, 1))) / 80.0  # 80 deg preparation (authors' script)
    mag = np.abs((w * pd).sum(-1)) ** 0.5
    masks = rois("radial")
    spheres = np.zeros(kappa.shape, bool)
    for r in masks:
        spheres |= ndimage.binary_dilation(r, iterations=9)
    bg = (mag > 0.5 * mag.max()) & ~spheres & (kappa > 0.5)
    yy, xx = np.indices(kappa.shape)

    def poly(yy, xx, deg=4):
        u, v = (xx - N / 2) / (N / 2), (yy - N / 2) / (N / 2)
        return np.stack([u**i * v**j for i in range(deg + 1) for j in range(deg + 1 - i)], -1)

    cf, *_ = np.linalg.lstsq(poly(yy[bg], xx[bg]), kappa[bg], rcond=None)
    fit = poly(yy, xx) @ cf
    return {"kappa": kappa, "fit": fit, "fit_rms": np.std(kappa[bg] - poly(yy[bg], xx[bg]) @ cf),
            "sphere_fit": np.array([fit[r].mean() for r in masks]),
            "sphere_raw": np.array([kappa[r].mean() for r in masks])}


# ---- radial data --------------------------------------------------------------------------------

def kspace() -> np.ndarray:
    return readcfl(DATA / "data_05b_kspace").reshape(NS, 20, NT).astype(np.complex128)  # samples, coils, spokes


def delays(k: np.ndarray, spokes=range(500, NT), tol_deg: float = 0.3) -> np.ndarray:
    ang = np.arange(NT) * GOLDEN
    proj = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(k, axes=0), axis=0), axes=0)
    x = np.arange(NS) - NS // 2
    th, dl = [], []
    spokes = list(spokes)
    for a in spokes:
        d = [(min((ang[b] - ang[a] - np.pi) % (2 * np.pi), (ang[a] + np.pi - ang[b]) % (2 * np.pi)), b)
             for b in spokes if b != a]
        dmin, b = min(d)
        if dmin > np.radians(tol_deg):
            continue
        pb = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(k[::-1, :, b], axes=0), axis=0), axes=0)
        c = (proj[:, :, a] * np.conj(pb)).sum(1)
        w = np.abs(c)
        m = w > 0.2 * w.max()
        slope = np.polyfit(x[m], np.unwrap(np.angle(c[m])), 1, w=w[m])[0]
        th.append(ang[a])
        dl.append(slope * NS / (2 * np.pi) / 2)
    th, dl = np.array(th), np.array(dl)
    A = np.stack([np.cos(th) ** 2, np.sin(th) ** 2, 2 * np.sin(th) * np.cos(th)], 1)
    coef, *_ = np.linalg.lstsq(A, dl, rcond=None)
    return np.append(coef, np.std(dl - A @ coef))  # dxx, dyy, dxy [oversampled samples], residual


def trajectory(coef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) NUFFT coordinates [rad] of every sample, spoke-major. Measured spokes are
    shifted by -d(theta) along the readout (sign fixed by data consistency: +d leaves a 0.78
    relative residual, -d 0.23)."""
    ang = np.arange(NT) * GOLDEN
    d = -(coef[0] * np.cos(ang) ** 2 + coef[1] * np.sin(ang) ** 2 + 2 * coef[2] * np.sin(ang) * np.cos(ang))
    r = ((np.arange(NS) - NS / 2 + 0.5)[None, :] + d[:, None]) / 2  # base-grid units
    return (2 * np.pi / N * r * np.cos(ang)[:, None]).ravel(), (2 * np.pi / N * r * np.sin(ang)[:, None]).ravel()


def compress(k: np.ndarray, n_coils: int = 8):
    dat = np.transpose(k, (1, 2, 0)).reshape(k.shape[1], -1)
    u, _, _ = np.linalg.svd(dat[:, -500 * NS:], full_matrices=False)
    cc = u[:, :n_coils].conj().T
    dat = cc @ dat
    scale = np.abs(dat).max()
    return dat / scale, cc, scale


def ramp() -> np.ndarray:
    w = (np.abs((np.arange(NS) - NS / 2 + 0.5) / 2)[None, :] * np.ones((NT, 1))).ravel() + 0.5
    return w / w.max()


def sensitivities(dat, x, y):
    sel = slice((NT - 500) * NS, None)
    p = finufft.Plan(1, (N, N), n_trans=dat.shape[0], eps=1e-6)
    p.setpts(x[sel], y[sel])
    img = p.execute(np.ascontiguousarray(dat[:, sel] * ramp()[sel]))
    rss = np.sqrt((np.abs(img) ** 2).sum(0))
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(rss > 0.08 * rss.max(), iterations=3))
    sens = img / (rss + 1e-12)
    nm = ndimage.gaussian_filter(mask * 1.0, 3) + 1e-6
    for c in range(len(sens)):
        sens[c] = (ndimage.gaussian_filter(sens[c].real * mask, 3)
                   + 1j * ndimage.gaussian_filter(sens[c].imag * mask, 3)) / nm
    sens *= ndimage.binary_dilation(mask, iterations=4)
    return sens / (np.sqrt((np.abs(sens) ** 2).sum(0)) + 1e-12), mask


def subspace(dat, x, y, sens, basis, iters: int = 200, lam: float = 1e-4):
    """Coefficient images (K, N, N) minimising ||W^1/2 (A X - y)||^2 + lam ||X||^2 (CG)."""
    K, C = basis.shape[0], sens.shape[0]
    phi = basis[:, np.repeat(np.arange(NT), NS)]
    w = ramp()
    p2 = finufft.Plan(2, (N, N), n_trans=C * K, eps=1e-6)
    p2.setpts(x, y)
    p1 = finufft.Plan(1, (N, N), n_trans=C * K, eps=1e-6)
    p1.setpts(x, y)

    def fwd(X):
        v = p2.execute(np.ascontiguousarray((sens[None] * X[:, None]).reshape(K * C, N, N))).reshape(K, C, -1)
        return (phi[:, None, :] * v).sum(0)

    def adj(r):
        im = p1.execute(np.ascontiguousarray((np.conj(phi)[:, None, :] * r[None]).reshape(K * C, -1)))
        return (np.conj(sens)[None] * im.reshape(K, C, N, N)).sum(1)

    b = adj(w * dat)
    X = np.zeros((K, N, N), complex)
    r, p = b.copy(), b.copy()
    rs = r0 = np.vdot(r, r).real
    for _ in range(iters):
        Ap = adj(w * fwd(p)) + lam * p
        a = rs / np.vdot(p, Ap).real
        X += a * p
        r -= a * Ap
        rn = np.vdot(r, r).real
        p = r + rn / rs * p
        rs = rn
    res = np.linalg.norm(np.sqrt(w) * (dat - fwd(X))) / np.linalg.norm(np.sqrt(w) * dat)
    return X, {"cg_residual": np.sqrt(rs / r0), "data_residual": res}


def curves(X, basis, mask) -> np.ndarray:
    """(15, NT) complex ROI-mean curves: spheres 1..14, then the background water (the object
    support eroded by 12 px, minus the sphere ROIs dilated by 12 px)."""
    masks = rois("radial")
    bg = ndimage.binary_erosion(mask, iterations=12) & ~ndimage.binary_dilation(sum(masks) > 0, iterations=12)
    return np.array([[Xk[r].mean() for Xk in X] for r in masks + [bg]]) @ basis[: len(X)]


def noise_covariance(k: np.ndarray) -> np.ndarray:
    """Coil noise covariance from the outermost 32 samples of every spoke (noise only: their
    spread equals that of the spoke-to-spoke differences / sqrt 2)."""
    edge = np.r_[0:16, NS - 16:NS]
    d = (k[edge][:, :, 1:] - k[edge][:, :, :-1]).transpose(1, 0, 2).reshape(k.shape[1], -1) / np.sqrt(2)
    return d @ d.conj().T / d.shape[1]


def digital(curves_true, amps, sens, mask, x, y, cov_compressed, radius: float = 7.0, seed: int = 1):
    """Compressed, normalised k-space of a digital phantom: sphere discs (radius in pixels, at
    the ROI centres) and the remaining support, on a 2x finer grid, each with its curve x amp;
    plus complex Gaussian noise of the given (compressed) covariance."""
    M = 2 * N
    yy, xx = (np.indices((M, M)) + 0.5) / 2 - 0.5
    cents = [np.array(ndimage.center_of_mass(r)) for r in rois("radial")]
    discs = [(np.hypot(yy - c[0], xx - c[1]) < radius).astype(float) for c in cents]
    walls = np.clip(sum((np.hypot(yy - c[0], xx - c[1]) < radius + 1).astype(float) for c in cents), 0, 1)
    support = ndimage.zoom(mask.astype(float), 2, order=1) > 0.5
    regions = discs + [support * (1 - walls)]
    sens2 = np.array([ndimage.zoom(s.real, 2, order=1) + 1j * ndimage.zoom(s.imag, 2, order=1) for s in sens])
    p2 = finufft.Plan(2, (M, M), n_trans=len(sens), eps=1e-7)
    p2.setpts(x / 2, y / 2)  # same k on the finer grid
    t = np.repeat(np.arange(NT), NS)
    out = np.zeros((len(sens), NT * NS), complex)
    for reg, cv, a in zip(regions, curves_true, amps):
        out += a * cv[t][None] * p2.execute(np.ascontiguousarray((sens2 * reg[None]).astype(complex))) / 4.0
    L = np.linalg.cholesky(cov_compressed)
    rng = np.random.default_rng(seed)
    noise = (rng.standard_normal(out.shape) + 1j * rng.standard_normal(out.shape)) / np.sqrt(2)
    return out + L @ noise


def main(argv):
    CACHE.mkdir(parents=True, exist_ok=True)
    cmd = argv[0]
    if cmd == "gold":
        np.savez(CACHE / "gold.npz", **gold())
    elif cmd == "b1":
        np.savez(CACHE / "b1.npz", **b1())
    elif cmd == "prepare":  # delays, compression, coil maps, noise
        k = kspace()
        coef = delays(k)
        x, y = trajectory(coef)
        dat, cc, scale = compress(k)
        sens, mask = sensitivities(dat, x, y)
        cov = cc @ noise_covariance(k) @ cc.conj().T / scale**2
        np.savez(CACHE / "prepared.npz", delays=coef, x=x, y=y, dat=dat, sens=sens, mask=mask, cov=cov)
    elif cmd == "recon":  # recon <basis.npz> <K> <out.npz> [data.npz]
        pre = np.load(CACHE / "prepared.npz")
        basis = np.load(argv[1])["basis"][: int(argv[2])]
        dat = np.load(argv[4])["dat"] if len(argv) > 4 else pre["dat"]
        X, info = subspace(dat, pre["x"], pre["y"], pre["sens"], basis)
        np.savez(argv[3], X=X, curves=curves(X, basis, pre["mask"]), **info)
        print(argv[3], info)
    elif cmd == "digital":  # digital <truth.npz> <out.npz>: truth has curves (15, NT), amps (15,)
        pre = np.load(CACHE / "prepared.npz")
        tr = np.load(argv[1])
        dat = digital(tr["curves"], tr["amps"], pre["sens"], pre["mask"], pre["x"], pre["y"], pre["cov"])
        np.savez(argv[2], dat=dat)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
