"""Generalized Bloch model (``generalized_bloch.py``) against the three limits its paper claims:
the Bloch equations for a Lorentzian (SciPy ODE; second-order convergence), Henkelman's
steady state under continuous waves for every lineshape (with ``exchange.py``'s lineshapes,
an independent check of their G = pi g normalisation), and the hard-pulse rotation; plus the
free pool's coupling against the two-pool ODE for a Lorentzian bound pool."""

import math

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

from blochparty import generalized_bloch as gb
from blochparty.exchange import lorentzian, superlorentzian

T2 = 10e-6


def test_lorentzian_is_bloch_second_order():
    alpha, trf, delta, r1 = math.pi, 20e-6, 2 * math.pi * 3000, 1.0
    w1 = alpha / trf

    def bloch(t, m):
        wx, wy = w1 * math.cos(delta * t), w1 * math.sin(delta * t)
        x, y, z = m
        return [-x / T2 + wy * z, -y / T2 - wx * z, -wy * x + wx * y + r1 * (1 - z)]

    ref = solve_ivp(bloch, (0, trf), [0, 0, 1], method="DOP853", rtol=1e-12, atol=1e-14, max_step=trf / 2000).y[2, -1]
    err = [abs(gb.simulate_pulse(gb.rectangular(alpha, trf, h, delta), h, t2s=T2, shape="lorentzian", r1s=r1)["zs"][-1]
               - ref) for h in (0.4e-6, 0.2e-6, 0.1e-6)]
    assert err[-1] < 2e-5  # a 20 us pi pulse: w1 = 1.6e5 rad/s, h = 0.1 us
    assert 3.5 < err[0] / err[1] < 5 and 3.5 < err[1] / err[2] < 5, err


@pytest.mark.parametrize("shape", gb.GREENS)
def test_hard_pulse_rotates_the_bound_pool(shape):
    for alpha in (math.pi / 2, math.pi):
        zs = gb.simulate_pulse(gb.rectangular(alpha, 0.05e-6, 0.002e-6), 0.002e-6, t2s=T2, shape=shape)["zs"][-1]
        assert zs == pytest.approx(math.cos(alpha), abs=4e-3)


@pytest.mark.parametrize("shape", gb.GREENS)
def test_continuous_wave_reaches_henkelman(shape):
    w1, delta, r1 = 2 * math.pi * 400, 2 * math.pi * 2500, 600.0
    h, t = 1e-6, 15e-3
    zs = gb.simulate_pulse(gb.rectangular(w1 * t, t, h, delta), h, t2s=T2, shape=shape, r1s=r1)["zs"][-1]
    hz, t2 = torch.tensor(delta / (2 * math.pi), dtype=torch.float64), torch.tensor(T2, dtype=torch.float64)
    x = T2 * delta
    pig = {"lorentzian": float(lorentzian(hz, t2)), "gaussian": T2 * math.sqrt(math.pi / 2) * math.exp(-x * x / 2),
           "superlorentzian": float(superlorentzian(hz, t2, cutoff=1e-9))}[shape]
    assert zs == pytest.approx(r1 / (r1 + w1**2 * pig), abs=2e-4)


def test_free_pool_coupling_matches_two_pool_ode():
    # Lorentzian bound pool: the generalized model is the ordinary two-pool Bloch-McConnell system
    m0f, m0s, k, r1f, r2f, r1s = 1.0, 0.15, 30.0, 1.0, 20.0, 1.0
    alpha, trf, h = math.pi / 2, 200e-6, 0.1e-6
    w = gb.rectangular(alpha, trf, h)
    w1 = alpha / trf
    out = gb.simulate_pulse(w, h, t2s=T2, shape="lorentzian", r1s=r1s, m0s=m0s, k=k,
                            free=dict(m0=m0f, r1=r1f, r2=r2f, dw=2 * math.pi * 50))
    kf = k * m0s / m0f

    def rhs(_, m):
        xf, yf, zf, xs, ys, zs = m
        dw = 2 * math.pi * 50
        return [yf * dw - r2f * xf, zf * w1 - xf * dw - r2f * yf, -yf * w1 - r1f * (zf - m0f) - kf * zf + k * zs,
                -xs / T2, -ys / T2 - w1 * zs, w1 * ys + r1s * (m0s - zs) + kf * zf - k * zs]

    ref = solve_ivp(rhs, (0, trf), [0, 0, m0f, 0, 0, m0s], method="DOP853", rtol=1e-12, atol=1e-14,
                    max_step=trf / 4000).y[:, -1]
    assert np.abs(out["free"][-1] - ref[:3]).max() < 1e-6
    assert abs(out["zs"][-1] - ref[5]) < 1e-6
