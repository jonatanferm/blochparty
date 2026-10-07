"""Generalized Bloch model of the semi-solid (bound) pool (Asslander et al., arXiv 2107.11000).

The rate model of ``exchange.py`` (Graham: W = w1^2 G) saturates the bound pool but never
rotates it, so a short intense pulse cannot invert it, which experiments show it does
(Manning et al.). The generalized Bloch model keeps the rotation: the pool's transverse
magnetization decays not exponentially but with the Green's function G(s) of its lineshape
(the Fourier transform of the absorption line), so, with u = x_s + i y_s in the pool's
rotating frame and the RF w(t) = w_x + i w_y [rad/s],

    u(t)    = -i int_0^t G(t - tau) w(tau) z_s(tau) dtau
    dz_s/dt = Im(conj(w(t)) u(t)) + R1s (m0s - z_s) + exchange,

which is the Bloch equation for the Lorentzian G = exp(-R2 s), Henkelman's steady state
R1 / (R1 + pi w1^2 g(Delta)) under continuous waves, and a rotation by the flip angle in the
hard-pulse limit. Green's functions: Lorentzian exp(-R2 s), Gaussian exp(-(R2 s)^2 / 2),
super-Lorentzian int_0^1 exp(-(R2 s)^2 (3 zeta^2 - 1)^2 / 8) dzeta.

``simulate_pulse`` integrates one RF pulse for single spins (two pools: free water with
ordinary Bloch dynamics, longitudinal exchange with the bound pool) on a uniform grid with a
second-order (Heun, trapezoid-history) scheme: O(n^2) in the number of steps, a reference
for pulse-level dynamics, not yet an imaging engine. The history of the bound pool's
transverse magnetization is assumed to start with the pulse (coherence from earlier pulses
has decayed). Arrays are numpy float64.
"""

from __future__ import annotations

import math

import numpy as np

GREENS = ("lorentzian", "gaussian", "superlorentzian")


MEMORY = 100.0  # super-Lorentzian memory [T2]: G is tapered by exp(-s / (MEMORY T2))


def green(s: np.ndarray, t2: float, shape: str, memory: float | None = None) -> np.ndarray:
    """G(s) for lags s >= 0 [s]; G(0) = 1. ``memory`` [in units of t2] tapers it by
    exp(-s / (memory t2)): a homogeneous broadening of 1 / (2 pi memory t2) that makes the
    super-Lorentzian finite on resonance (its G has a 1/s tail)."""
    x = np.asarray(s, dtype=float) / t2
    taper = 1.0 if memory is None else np.exp(-x / memory)
    return taper * _green(x, shape)


def _green(x: np.ndarray, shape: str) -> np.ndarray:
    if shape == "lorentzian":
        return np.exp(-x)
    if shape == "gaussian":
        return np.exp(-0.5 * x * x)
    if shape == "superlorentzian":
        z, w = np.polynomial.legendre.leggauss(200)
        u0 = 1 / math.sqrt(3)
        out = np.zeros_like(x)
        for a, b in ((0.0, u0), (u0, 1.0)):  # the integrand is smooth; split where 3 zeta^2 - 1 = 0
            zeta = a + (b - a) * 0.5 * (z + 1)
            c = (3 * zeta**2 - 1) ** 2 / 8
            out += 0.5 * (b - a) * (np.exp(-np.multiply.outer(x * x, c)) * w).sum(-1)
        return out
    raise ValueError(f"shape must be one of {GREENS}")


def simulate_pulse(w: np.ndarray, h: float, *, t2s: float, shape: str, r1s: float = 1.0, m0s: float = 1.0,
                   zs0: float | None = None, free: dict | None = None, k: float = 0.0) -> dict:
    """Integrate a pulse given on a uniform grid: ``w`` (n,) complex RF in the bound pool's
    rotating frame [rad/s] at times (i + 1/2) h (piecewise constant). Optional free pool
    ``free`` = dict(m0, r1, r2, dw [rad/s], m=(x, y, z)), exchanging z with rate ``k`` (bound ->
    free) and k m0s/m0f... in BMCTool's convention: free -> bound at k f with f = m0s / m0f.
    Returns the bound pool's z_s and transverse u at the grid points 0..n, and the free pool."""
    n = len(w)
    zs = np.empty(n + 1)
    zs[0] = m0s if zs0 is None else zs0
    G = green(np.arange(n + 1) * h, t2s, shape)
    f = None
    if free is not None:
        f = np.array(free.get("m", (0.0, 0.0, free["m0"])), dtype=float).reshape(1, 3).repeat(n + 1, 0)
        kf = k * m0s / free["m0"]  # free -> bound
    u = np.zeros(n + 1, dtype=complex)
    # history: cell m (RF w_m, constant) contributes -i h/2 w_m [G((j-m) h) z_m + G((j-m-1) h) z_m+1] to u(t_j)
    a_m = np.zeros(n, dtype=complex)  # w_m z_m
    b_m = np.zeros(n, dtype=complex)  # w_m z_m+1

    def u_at(j, a_last=None, b_last=None):
        """u(t_j) from cells 0..j-1; the last cell's terms may be given (predictor)."""
        if j == 0:
            return 0j
        aa, bb = a_m[:j].copy(), b_m[:j].copy()
        if a_last is not None:
            aa[j - 1], bb[j - 1] = a_last, b_last
        return -0.5j * h * ((aa * G[j:0:-1]).sum() + (bb * G[j - 1::-1][:j]).sum())

    def rhs(zj, fj, uj, wj):
        dz = (np.conj(wj) * uj).imag + r1s * (m0s - zj)
        df = None
        if f is not None:
            wx, wy, dw = wj.real, wj.imag, free["dw"]
            x, y, z = fj
            r1f, r2f = free["r1"], free["r2"]
            # dm/dt = m x b, b = (wx, wy, dw) as blochparty (a pulse along +x tips +z to +y)
            df = np.array([y * dw - z * wy, z * wx - x * dw, x * wy - y * wx]) \
                - np.array([r2f * x, r2f * y, r1f * (z - free["m0"])])
            df[2] += -kf * z + k * zj
            dz += kf * z - k * zj
        return dz, df

    for j in range(n):
        wj = w[j]  # RF on [t_j, t_j+1)
        u[j] = u_at(j)
        dz0, df0 = rhs(zs[j], f[j] if f is not None else None, u[j], wj)
        zp = zs[j] + h * dz0
        fp = f[j] + h * df0 if f is not None else None
        up = u_at(j + 1, wj * zs[j], wj * zp)
        dz1, df1 = rhs(zp, fp, up, wj)
        zs[j + 1] = zs[j] + 0.5 * h * (dz0 + dz1)
        if f is not None:
            f[j + 1] = f[j] + 0.5 * h * (df0 + df1)
        a_m[j], b_m[j] = wj * zs[j], wj * zs[j + 1]
    u[n] = u_at(n)
    return {"zs": zs, "u": u, "free": f}


def rectangular(alpha: float, t_rf: float, h: float, delta: float = 0.0) -> np.ndarray:
    """A rectangular pulse of flip angle ``alpha`` [rad] and duration ``t_rf`` on a grid of
    step ``h``, off resonance by ``delta`` [rad/s] in the bound pool's frame."""
    n = max(1, round(t_rf / h))
    t = (np.arange(n) + 0.5) * h
    return (alpha / (n * h)) * np.exp(1j * delta * t)


# ---------------------------------------------------------------- modes for the imaging engines

_MODES: dict = {}


def modes(shape: str, memory: float = MEMORY) -> tuple[np.ndarray, np.ndarray]:
    """G(s) ~ sum_j c_j exp(-p_j s / T2) with complex poles p_j (in units of 1/T2) and weights
    c_j in conjugate pairs (so the sum is real) and sum_j c_j = G(0) = 1.

    ``memory`` [T2] is the super-Lorentzian's taper (ignored for the other shapes); the slowest
    pole sits at 1 / memory. Within the 30 modes, memory 300 still fits its G to 2e-4; 1000
    only to 2e-3. Real data prefer the longer memories (``validation/real/RESULTS.md``,
    benchmark 4: BSA inversion with pulses to 70 T2).

    The Lorentzian is one exact mode. Others: least squares with poles lambda e^{+-i theta},
    theta in {0, 0.6, 1.2}. Super-Lorentzian: the tapered G (``MEMORY``), six lambda
    log-spaced over [1/100, 3], fitted over its whole support (to 800 T2), 30 modes, error
    2.4e-5, weights < 1, lineshape within 0.5 % from 0 to 30 kHz. Fitting the untapered G over
    a finite range instead needed weights of ~5000 that cancel and diverge beyond the range;
    regularising them destroys the cancellations the steep wings need. Gaussian: five lambda
    over [0.3, 6] on [0, 40 T2], 25 modes, error 2e-3. A flat-topped G needs the complex
    poles: positive real exponentials alone leave Lorentzian wings (12-18x the
    super-Lorentzian at 30 kHz)."""
    key = (shape, float(memory)) if shape == "superlorentzian" else shape
    if key in _MODES:
        return _MODES[key]
    if shape == "lorentzian":
        out = (np.array([1.0 + 0j]), np.array([1.0 + 0j]))
    else:
        if shape == "superlorentzian":  # tapered at memory T2; fit its whole support
            x = np.concatenate([np.linspace(0, 4, 400), np.logspace(math.log10(4), math.log10(8 * memory), 600)])
            lams = np.logspace(-math.log10(memory), math.log10(3.0), 6)
            g = green(x, 1.0, shape, memory)
        else:  # Gaussian: gone within ~8 T2; fit to 40 T2 so the tail stays zero
            x = np.concatenate([np.linspace(0, 10, 800), np.linspace(10, 40, 300)[1:]])
            lams = np.logspace(math.log10(0.3), math.log10(6.0), 5)
            g = green(x, 1.0, shape)
        poles = [lam * complex(math.cos(t), math.sin(t)) for lam in lams for t in (0.0, 0.6, 1.2)]
        cols = []
        for p in poles:
            e = np.exp(-p * x)
            cols.append(e.real)
            if p.imag:
                cols.append(e.imag)
        a = np.array(cols).T
        w = 1 / np.maximum(np.abs(g), 1e-4 if shape == "superlorentzian" else 2e-3)
        w[0] = 1e4  # G(0) = 1: the hard-pulse rotation
        coef = np.linalg.lstsq(a * w[:, None], g * w, rcond=None)[0]
        ps, cs, i = [], [], 0
        for p in poles:  # alpha Re(e^{-px}) + beta Im(e^{-px}) = Re((alpha - i beta) e^{-px})
            if p.imag:
                c = 0.5 * (coef[i] - 1j * coef[i + 1])
                ps += [p, p.conjugate()]
                cs += [c, c.conjugate()]
                i += 2
            else:
                ps.append(p)
                cs.append(complex(coef[i]))
                i += 1
        cs = np.array(cs)
        out = (np.array(ps), cs / cs.sum().real)
    _MODES[key] = out
    return out
