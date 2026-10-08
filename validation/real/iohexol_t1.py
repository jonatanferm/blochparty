"""T1 of the iohexol tubes from the archive's IR Look-Locker scan (benchmark 9's independent input).

Radial stack-of-stars after one inversion (Huemer et al.; prepared k-space of the archive,
``data/prepared/phantom/meas_MID00123_FID56962_IR_LL_SOS.h5``: 266 time points x 16 coils x 3
slices x 256 samples (2x oversampled), golden angle; ``Seq_04_IR_LL_FOV_192_SOS3.seq``: 6 deg
every 5 ms over the slab, one spoke per slice every 15 ms). Subspace reconstruction (K = 4 SVD
basis of a Look-Locker dictionary 1 - (1 + kappa) exp(-t / T1*)), tube ROIs from the CEST images,
per-tube fit S = A - B exp(-t R1*), and R1 = R1* + ln(cos(B1 6 deg)) / 5 ms.

numpy + finufft only (finufft and torch cannot share a process on macOS).

    uv run python validation/real/iohexol_t1.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import finufft
import h5py
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
import iohexol as I  # noqa: E402  (numpy-only parts)

H5 = I.DATA / "meas_MID00123_FID56962_IR_LL_SOS.h5"
N, FLIP, TR = 128, 6.0, 5e-3


def load():
    with h5py.File(H5) as f:
        d = f["real_dat"][...] + 1j * f["imag_dat"][...]  # (266, 16, 3, 1, 256)
        traj = f["traj"][...]  # (266, 1, 256, 2), +-1 = Nyquist
        t = np.asarray(f.attrs["time"], float)
    return d[:, :, :, 0], traj[:, 0], t


def basis(t, k=4):
    t1s = np.geomspace(0.05, 4.0, 200)
    kap = np.linspace(0.5, 1.0, 11)
    dic = np.array([1 - (1 + a) * np.exp(-t / x) for x in t1s for a in kap])
    u, s, vt = np.linalg.svd(dic, full_matrices=False)
    return vt[:k], s[:k] / s[0]


def recon(dat, traj, phi, iters=60, lam=1e-3):
    """(K, N, N) subspace coefficients of one slice (dat (T, C, S))."""
    nt, nc, ns = dat.shape
    x = (math.pi * traj[..., 0]).ravel()
    y = (math.pi * traj[..., 1]).ravel()
    w = np.abs(np.linspace(-1, 1, ns))[None, :].repeat(nt, 0).ravel() + 0.5 / ns
    y_data = dat.transpose(1, 0, 2).reshape(nc, -1)
    p1 = finufft.Plan(1, (N, N), n_trans=nc, eps=1e-6)
    p1.setpts(x, y)
    img = p1.execute(np.ascontiguousarray(y_data * w))
    rss = np.sqrt((np.abs(img) ** 2).sum(0))
    mask = ndimage.binary_fill_holes(rss > 0.1 * rss.max())
    sens = img / (rss + 1e-12)
    nm = ndimage.gaussian_filter(mask * 1.0, 3) + 1e-6
    sens = np.stack([(ndimage.gaussian_filter(c.real * mask, 3) + 1j * ndimage.gaussian_filter(c.imag * mask, 3)) / nm for c in sens])
    sens *= ndimage.binary_dilation(mask, iterations=4)
    sens /= np.sqrt((np.abs(sens) ** 2).sum(0)) + 1e-12
    k = phi.shape[0]
    ph = phi[:, np.repeat(np.arange(nt), ns)]
    q2 = finufft.Plan(2, (N, N), n_trans=k * nc, eps=1e-6)
    q2.setpts(x, y)
    q1 = finufft.Plan(1, (N, N), n_trans=k * nc, eps=1e-6)
    q1.setpts(x, y)

    def fwd(X):
        v = q2.execute(np.ascontiguousarray((sens[None] * X[:, None]).reshape(k * nc, N, N))).reshape(k, nc, -1)
        return (ph[:, None] * v).sum(0)

    def adj(r):
        im = q1.execute(np.ascontiguousarray((ph[:, None] * r[None]).reshape(k * nc, -1)))  # phi real
        return (np.conj(sens)[None] * im.reshape(k, nc, N, N)).sum(1)

    b = adj(w * y_data)
    X = np.zeros((k, N, N), complex)
    r, p = b.copy(), b.copy()
    rs = r0 = np.vdot(r, r).real
    for _ in range(iters):
        ap = adj(w * fwd(p)) + lam * np.abs(b).max() * p
        a = rs / np.vdot(p, ap).real
        X += a * p
        r -= a * ap
        rn = np.vdot(r, r).real
        p = r + rn / rs * p
        rs = rn
    return X, rss, float(np.sqrt(rs / r0))


def register(img, ref):
    """The dihedral orientation of ``img`` that best matches ``ref`` (normalised correlation)."""
    best = None
    for tr in (False, True):
        for fy in (False, True):
            for fx in (False, True):
                u = img.T if tr else img
                u = u[::-1] if fy else u
                u = u[:, ::-1] if fx else u
                c = np.corrcoef(u.ravel(), ref.ravel())[0, 1]
                if best is None or c > best[0]:
                    best = (c, tr, fy, fx)
    return best


def orient(x, o):
    _, tr, fy, fx = o
    x = np.swapaxes(x, -1, -2) if tr else x
    x = x[..., ::-1, :] if fy else x
    return x[..., ::-1] if fx else x


def main():
    dat, traj, t = load()
    phi, sv = basis(t)
    w = I.offsets_ppm()[1:]
    mag = np.load(I.DATA / "ref_tsat1.5.npy")
    m0 = mag[0, 1]
    rois = I.tubes(mag[1:, 1][np.argmin(abs(w - 1.0))] / np.maximum(m0, 1e-9), m0)
    _, b1 = I.fields()
    # the central of the slab's three slices, as for the CEST spectra: the edge slices see less than
    # the nominal flip (slab profile), which the Look-Locker correction depends on
    s = 1
    X, rss, cg = recon(dat[:, :, s], traj, phi)
    o = register(rss, m0)
    print(f"slice {s}: CG residual {cg:.3f}; orientation corr {o[0]:.3f} (transpose {o[1]}, flip y {o[2]}, flip x {o[3]})")
    X = orient(X, o)
    out = {}
    for i, r in enumerate(rois):
        curve = phi.T @ X[:, r].mean(1)  # (T,) complex
        curve = (curve * np.exp(-1j * np.angle(curve[-20:].mean()))).real
        fun = lambda p: p[0] - p[1] * np.exp(-t * p[2]) - curve  # noqa: E731
        fit = least_squares(fun, (curve[-1], 2 * curve[-1], 1.5))
        a, bb, r1s = fit.x
        bl = float(np.median(b1[r]))
        r1 = r1s + math.log(math.cos(math.radians(FLIP * bl))) / TR
        deichmann = (bb / a - 1) / r1s  # T1 = T1* (B / A - 1), ideal inversion, no flip angle needed
        out[i + 1] = dict(T1=1 / r1, T1_deichmann=deichmann, T1star=1 / r1s, inversion=bb / a - 1, B1=bl,
                          rel_residual=float(np.sqrt(np.mean(fit.fun**2)) / abs(a)))
        print(f"tube {i + 1}: T1* {1 / r1s:.3f} s; T1 from the flip angle {1 / r1:.3f} s (B1 {bl:.3f}), "
              f"Deichmann {deichmann:.3f} s; fit residual {out[i + 1]['rel_residual']:.4f}")
    (I.DATA / "t1_irll.json").write_text(json.dumps({"slice": int(s), "orientation": [bool(v) for v in o[1:]],
                                                     "tubes": out}, indent=1))


if __name__ == "__main__":
    main()
