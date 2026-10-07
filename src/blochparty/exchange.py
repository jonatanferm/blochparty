"""Magnetization transfer: a bound (semi-solid) pool exchanging with the free water.

The two-pool model of Henkelman et al. (MRM 1993) with Graham's time-dependent saturation
rate, as in BMCTool / Pulseq-CEST: per spin, the free pool keeps its full Bloch dynamics
(Mx, My, Mz; equilibrium PD) and the bound pool only a longitudinal magnetization Mzb
(equilibrium f PD), whose transverse magnetization decays within microseconds. The pools
exchange longitudinal magnetization,

    dMz/dt  = R1 (PD - Mz)     - k f Mz + k Mzb   (+ the RF rotation of the free pool)
    dMzb/dt = R1b (f PD - Mzb) + k f Mz - k Mzb - W Mzb,

with k the exchange rate from the bound to the free pool [Hz] and f = M0b / M0. RF saturates
the bound pool at the rate W = w1^2 G(2 pi Delta) [1/s], w1 = 2 pi |B1| [rad/s] and G the
absorption lineshape in BMCTool's normalisation (G = pi g, with g of unit area over rad/s):

* Lorentzian: G = T2b / (1 + x^2), x = 2 pi Delta T2b;
* super-Lorentzian: G = T2b sqrt(2 pi) int_0^1 exp(-2 x^2 / (3u^2 - 1)^2) / |3u^2 - 1| du,
  which diverges on resonance; it is held at its value at ``BoundPool.cutoff`` closer to
  resonance. The default 1 kHz is the cut of Gloor et al. (MRM 2008, as described by
  Asslander et al., arXiv 2107.11000) for bSSFP qMT; g = G / pi there is 1.5e-5 s at
  T2b = 12 us, the order of the empirically calibrated G(0) = 2.0e-5 s (Soellinger et al.,
  ISMRM 2010: bSSFP qMT, postmortem in-situ brain at 3 T, 6-20 C). BMCTool's own cut is ~150 Hz, with a rectangle-rule
  integral outside it: near the pool (CEST offsets) the two conventions differ by up to 0.12
  in Z (``validation/compare_bmctool.py``). The near-resonance super-Lorentzian is a model
  choice, not physics this model can settle.

Delta is the offset of the RF from the bound pool's resonance: the RF's instantaneous
frequency (from the phase of consecutive RF samples, ``rf_frequency``) minus the spin's
off-resonance (gradients, dB0, field model) minus the pool's shift ``dw``. Slice-selective
pulses therefore saturate the bound pool in other slices by the lineshape at the slice offset:
incidental MT.

Limits: the rate model is accurate for pulses of milliseconds; for short intense pulses
(T_RF below ~0.3 ms, w1 near 1/T2b) it misses the rotation of the bound pool (Asslander et al.;
the generalized Bloch model). Exchange between free pools (myelin water, CEST) is not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import Tensor

SL_CUTOFF = 1e3  # [Hz] default ``BoundPool.cutoff``

LINESHAPES = ("superlorentzian", "lorentzian")


@dataclass
class BoundPool:
    """Bound-pool parameters per spin (tensors of shape (N,), or scalars broadcast to N)."""

    f: Tensor  # pool size relative to the free pool, M0b / M0
    k: Tensor  # exchange rate from the bound to the free pool [Hz] (BMCTool's k)
    t1: Tensor  # [s]
    t2: Tensor  # [s], sets the lineshape width (~10 us in tissue)
    dw: Tensor | float = 0.0  # resonance shift from water [Hz]
    lineshape: str = "superlorentzian"
    cutoff: float = SL_CUTOFF  # [Hz] the super-Lorentzian is held constant closer to resonance

    def __post_init__(self):
        if self.lineshape not in LINESHAPES:
            raise ValueError(f"lineshape must be one of {LINESHAPES}, got {self.lineshape!r}")

    def expand(self, n: int, like: Tensor) -> BoundPool:
        conv = lambda v: torch.as_tensor(v, dtype=like.dtype, device=like.device).expand(n)
        return replace(self, f=conv(self.f), k=conv(self.k), t1=conv(self.t1), t2=conv(self.t2), dw=conv(self.dw))

    def to(self, dtype=None, device=None) -> BoundPool:
        conv = lambda v: v.to(dtype=dtype, device=device) if isinstance(v, Tensor) else v
        return replace(self, f=conv(self.f), k=conv(self.k), t1=conv(self.t1), t2=conv(self.t2), dw=conv(self.dw))

    def __getitem__(self, idx) -> BoundPool:
        pick = lambda v: v[idx] if isinstance(v, Tensor) and v.dim() > 0 else v
        return replace(self, f=pick(self.f), k=pick(self.k), t1=pick(self.t1), t2=pick(self.t2), dw=pick(self.dw))


# ---------------------------------------------------------------- lineshapes

_SL_TABLE: dict = {}
_SL_XMIN, _SL_XMAX, _SL_N = 1e-4, 30.0, 4097


def _sl_integral(x: np.ndarray) -> np.ndarray:
    """F(x) = sqrt(2 pi) int_0^1 exp(-2 x^2 / s^2) / s du, s = |3u^2 - 1|, for x > 0.

    The integrand vanishes (faster than any power) at the magic angle u0 = 1/sqrt(3), where
    s = 0; Gauss-Legendre on [0, u0] and [u0, 1] with nodes clustered towards u0 by
    u = u0 -/+ (u0 or 1-u0) t^4."""
    t, w = np.polynomial.legendre.leggauss(400)
    t, w = 0.5 * (t + 1), 0.5 * w  # on [0, 1]
    x = np.asarray(x, dtype=float)
    u0 = 1 / math.sqrt(3)
    total = np.zeros_like(x)
    for a, sign in ((u0, -1.0), (1 - u0, 1.0)):
        eps = sign * a * t**4  # u - u0
        jac = 4 * a * t**3 * w
        s = 3 * np.abs(eps) * (2 * u0 + eps)  # |3u^2 - 1| = 3 |u - u0| (u + u0), no cancellation
        total += (np.exp(-2 * (x[:, None] / s) ** 2) / s * jac).sum(1)
    return math.sqrt(2 * math.pi) * total


def _sl_table(dtype, device):
    key = (dtype, device)
    if key not in _SL_TABLE:
        lx = np.linspace(math.log(_SL_XMIN), math.log(_SL_XMAX), _SL_N)
        log_f = np.log(np.maximum(_sl_integral(np.exp(lx)), 1e-300))  # in float64: finite in float32 too
        _SL_TABLE[key] = (torch.as_tensor(lx, dtype=dtype, device=device), torch.as_tensor(log_f, dtype=dtype, device=device))
    return _SL_TABLE[key]


def _interp(xq: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Linear interpolation on a uniform grid ``xp``, clamped at its ends; differentiable in xq."""
    h = xp[1] - xp[0]
    pos = ((xq - xp[0]) / h).clamp(0, len(xp) - 1)
    i = pos.detach().floor().long().clamp(max=len(xp) - 2)  # the last cell includes its right end
    a = pos - i
    return fp[i] * (1 - a) + fp[i + 1] * a


def superlorentzian(delta: Tensor, t2: Tensor, cutoff: float = SL_CUTOFF) -> Tensor:
    """G(2 pi delta) [s] for offsets ``delta`` [Hz]; constant for |delta| < ``cutoff``."""
    d = delta.abs().clamp_min(cutoff)
    x = 2 * math.pi * d * t2
    lx, log_f = _sl_table(delta.dtype, delta.device)
    val = torch.exp(_interp(torch.log(x.clamp_min(_SL_XMIN)), lx, log_f))
    return t2 * torch.where(x > _SL_XMAX, torch.zeros_like(val), val)


def lorentzian(delta: Tensor, t2: Tensor) -> Tensor:
    x = 2 * math.pi * delta * t2
    return t2 / (1 + x * x)


def lineshape(pool: BoundPool, delta: Tensor) -> Tensor:
    if pool.lineshape == "lorentzian":
        return lorentzian(delta, pool.t2)
    return superlorentzian(delta, pool.t2, pool.cutoff)


# ---------------------------------------------------------------- RF frequency


def rf_frequency(seq) -> Tensor:
    """(S,) instantaneous RF frequency [Hz] of every segment, in blochparty's convention
    (a pulse turning as exp(-i 2 pi f t) is resonant with spins at offset f): minus the
    phase advance per unit time between consecutive RF segments, averaging the steps to
    both neighbours where they carry RF. Zero where a segment has no RF neighbour. Steps
    are wrapped to (-pi, pi], so |f| up to 1 / (2 dt) is resolved. Returns ``seq.rf_freq``
    instead if it is set."""
    if seq.rf_freq is not None:
        if seq.rf_freq.shape != seq.dt.shape:
            raise ValueError(f"Sequence.rf_freq has shape {tuple(seq.rf_freq.shape)}, expected {tuple(seq.dt.shape)}")
        return seq.rf_freq
    rf, dt = seq.rf, seq.dt
    is_rf = torch.tensor(seq.is_rf, device=dt.device)
    amp2 = rf[:, 0] ** 2 + rf[:, 1] ** 2
    live = is_rf & (amp2 > 0)
    # phase step from segment s to s+1: angle(z_{s+1} conj(z_s))
    re = rf[1:, 0] * rf[:-1, 0] + rf[1:, 1] * rf[:-1, 1]
    im = rf[1:, 1] * rf[:-1, 0] - rf[1:, 0] * rf[:-1, 1]
    ok = live[1:] & live[:-1]
    re = torch.where(ok, re, torch.ones_like(re))  # safe values keep atan2's gradient finite
    im = torch.where(ok, im, torch.zeros_like(im))
    step = -torch.atan2(im, re) / (2 * math.pi * 0.5 * (dt[1:] + dt[:-1]))  # [Hz] between s and s+1
    step = torch.where(ok, step, torch.zeros_like(step))
    okf = ok.to(dt.dtype)
    zero = dt.new_zeros(1)
    total = torch.cat([zero, step]) + torch.cat([step, zero])
    count = torch.cat([zero, okf]) + torch.cat([okf, zero])
    return total / count.clamp_min(1)


# ---------------------------------------------------------------- longitudinal exchange step


def z_exchange(mz: Tensor, mb: Tensor, t, pd: Tensor, r1: Tensor, pool: BoundPool, w: Tensor | None = None):
    """Exact solution of the coupled longitudinal equations over a time ``t`` with constant
    coefficients (saturation rate ``w`` [1/s] of the bound pool, None for none)."""
    kf, kb, r1b = pool.k * pool.f, pool.k, 1.0 / pool.t1
    a11, a12 = -(r1 + kf), kb
    a21, a22 = kf, -(r1b + kb) if w is None else -(r1b + kb + w)
    c1, c2 = r1 * pd, r1b * pool.f * pd
    det = a11 * a22 - a12 * a21  # > 0: both eigenvalues negative
    z1 = -(a22 * c1 - a12 * c2) / det  # steady state -A^-1 c
    z2 = -(a11 * c2 - a21 * c1) / det
    # exp(A t) = e^{m t} [cosh(q t) I + sinh(q t)/q (A - m I)], applied in increment form,
    # z0 + (exp(A t) - I)(z0 - z_ss), with e^{mt} cosh(qt) - 1 from expm1 of the eigenvalues
    # m +/- q (both <= 0): no cancellation for short steps (1 us RF samples), where exp(A t) ~ I
    # and the gradients otherwise lose ~ eps / (t |A|) (3 % in float32), and no overflow for
    # long ones (expm1(2qt) only while 2qt < 1)
    m = 0.5 * (a11 + a22)
    h = 0.5 * (a11 - a22)
    q = torch.sqrt(h * h + a12 * a21)
    chm1 = 0.5 * (torch.expm1((m + q) * t) + torch.expm1((m - q) * t))  # e^{mt} cosh(qt) - 1
    small = 2 * q * t < 1
    lo = torch.exp((m - q) * t)
    e2q = torch.expm1(torch.where(small, 2 * q * t, torch.zeros_like(q)))
    qs = torch.where(q > 0, q, torch.ones_like(q))
    sh = torch.where(small, torch.where(q > 0, lo * 0.5 * e2q / qs, t * torch.exp(m * t)),
                     0.5 * (torch.exp((m + q) * t) - lo) / qs)  # e^{mt} sinh(qt)/q
    d1, d2 = mz - z1, mb - z2
    n1 = chm1 * d1 + sh * (h * d1 + a12 * d2)
    n2 = chm1 * d2 + sh * (a21 * d1 - h * d2)
    return mz + n1, mb + n2


def saturation_rate(pool: BoundPool, rf: Tensor, b1: Tensor | None, delta: Tensor) -> Tensor:
    """W = w1^2 G(2 pi delta) [1/s] for a segment's RF (x, y) [Hz] and offsets ``delta`` [Hz]."""
    w1sq = (2 * math.pi) ** 2 * (rf[0] ** 2 + rf[1] ** 2)
    if b1 is not None:
        w1sq = w1sq * b1 * b1
    return w1sq * lineshape(pool, delta)


# ---------------------------------------------------------------- free pools: Bloch-McConnell


@dataclass
class ExchangePool:
    """A free pool exchanging with the water (CEST pool, myelin water, ...): full Bloch dynamics
    (Mx, My, Mz) at its own shift, equilibrium f PD, exchanging all three components with the
    water at k (pool -> water) and k f (water -> pool), as BMCTool's CEST pools. Its transverse
    magnetization is part of the signal. Tensors of shape (N,) or scalars."""

    f: Tensor | float
    k: Tensor | float  # [Hz], pool -> water
    t1: Tensor | float  # [s]
    t2: Tensor | float  # [s]
    dw: Tensor | float = 0.0  # resonance shift from water [Hz]

    def expand(self, n: int, like: Tensor) -> ExchangePool:
        conv = lambda v: torch.as_tensor(v, dtype=like.dtype, device=like.device).expand(n)
        return ExchangePool(conv(self.f), conv(self.k), conv(self.t1), conv(self.t2), conv(self.dw))

    def to(self, dtype=None, device=None) -> ExchangePool:
        conv = lambda v: v.to(dtype=dtype, device=device) if isinstance(v, Tensor) else v
        return ExchangePool(conv(self.f), conv(self.k), conv(self.t1), conv(self.t2), conv(self.dw))


def _bm_matrix(rf, bz, b1, spins, pools, bound, w_sat):
    """(N, d, d) augmented Bloch-McConnell matrix of one segment: state (Mx, My, Mz) of the water
    and of every free pool, then Mzb of the bound pool (if any), then 1."""
    n, p = bz.shape[0], len(pools)
    nb = 1 if bound is not None else 0
    d = 3 * (p + 1) + nb + 1
    one = d - 1
    A = bz.new_zeros(n, d, d)
    two_pi = 2 * math.pi
    sx, sy = two_pi * rf[0], two_pi * rf[1]
    wx, wy = (sx * b1, sy * b1) if b1 is not None else (sx.expand(n), sy.expand(n))
    params = [(spins.pd, spins.r1, spins.r2, bz)] + [(q.f * spins.pd, 1 / q.t1, 1 / q.t2, bz + q.dw) for q in pools]
    for i, (m0, r1, r2, off) in enumerate(params):
        x, y, z = 3 * i, 3 * i + 1, 3 * i + 2
        wz = two_pi * off
        A[:, x, y] += wz
        A[:, x, z] -= wy
        A[:, y, x] -= wz
        A[:, y, z] += wx
        A[:, z, x] += wy
        A[:, z, y] -= wx
        A[:, x, x] -= r2
        A[:, y, y] -= r2
        A[:, z, z] -= r1
        A[:, z, one] += r1 * m0
    for i, q in enumerate(pools, start=1):
        k_out, k_in = q.k * q.f, q.k  # water -> pool, pool -> water
        for c in range(3):
            A[:, c, c] -= k_out
            A[:, c, 3 * i + c] += k_in
            A[:, 3 * i + c, c] += k_out
            A[:, 3 * i + c, 3 * i + c] -= k_in
    if bound is not None:
        zb = 3 * (p + 1)
        A[:, 2, 2] -= bound.k * bound.f
        A[:, 2, zb] += bound.k
        A[:, zb, 2] += bound.k * bound.f
        A[:, zb, zb] -= 1 / bound.t1 + bound.k + (w_sat if w_sat is not None else 0)
        A[:, zb, one] += bound.f * spins.pd / bound.t1
    return A

