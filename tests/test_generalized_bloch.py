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


# ---------------------------------------------------------------- the engine (modes of G)

import blochparty as bp


def single_pool(shape, **kw):
    one = lambda v: torch.full((1,), v, dtype=torch.float64)
    pool = dict(f=1.0, k=0.0, t1=1.0, t2=12e-6, lineshape=shape, model="generalized") | kw
    return bp.Phantom(pos=torch.zeros(1, 3, dtype=torch.float64), pd=one(1.0), t1=one(1.0), t2=one(0.05),
                      db0=one(0.0), bound=bp.BoundPool(**pool))


@pytest.mark.parametrize("shape", ["lorentzian", "superlorentzian"])
@pytest.mark.parametrize("alpha, t_rf", [(math.pi, 10e-6), (math.pi, 100e-6), (math.pi / 2, 300e-6)])
def test_engine_matches_reference_solver(shape, alpha, t_rf, monkeypatch):
    # the engine's Green's function is the tapered fit (generalized_bloch.MEMORY); give the
    # reference the same taper
    mem = gb.MEMORY if shape == "superlorentzian" else None
    orig = gb.green
    seq = bp.SequenceBuilder().rf(t_rf, flip_deg=math.degrees(alpha), n_steps=400).build()
    zb = float(bp.simulate(single_pool(shape), seq).m_bound[0])
    monkeypatch.setattr(gb, "green", lambda s, t2, sh, memory=mem: orig(s, t2, sh, memory))
    h = t_rf / 1600
    ref = gb.simulate_pulse(gb.rectangular(alpha, t_rf, h), h, t2s=12e-6, shape=shape, r1s=1.0)["zs"][-1]
    assert zb == pytest.approx(ref, abs=5e-5)


def test_engine_converges_second_order():
    ph = single_pool("superlorentzian", f=0.15, k=30.0)
    ph = bp.Phantom(ph.pos, ph.pd, ph.t1, ph.t2, torch.full((1,), 120.0, dtype=torch.float64), bound=ph.bound)
    seq = bp.SequenceBuilder().rf(200e-6, flip_deg=150.0, n_steps=10).delay(1e-3).rf(50e-6, flip_deg=90.0, n_steps=5).build()
    run = lambda k: bp.simulate(ph, seq.subdivide_rf(k))
    ref = run(256)
    err = [float(torch.cat([run(k).m.flatten(), run(k).m_bound]).sub(torch.cat([ref.m.flatten(), ref.m_bound])).abs().max())
           for k in (4, 8, 16)]
    assert 3.5 < err[0] / err[1] < 5 and 3.5 < err[1] / err[2] < 5, err


@pytest.mark.parametrize("offset", [2e3, 5e3, 1e4])
def test_off_resonant_saturation_agrees_with_rate_model(offset):
    # long off-resonant saturation is the rate model's regime: both give the same Z (the tapered
    # lineshape differs from the cut one by < 1 % there)
    def z(model):
        one = lambda v: torch.full((1,), v, dtype=torch.float64)
        ph = bp.Phantom(pos=torch.zeros(1, 3, dtype=torch.float64), pd=one(1.0), t1=one(1.0), t2=one(0.05),
                        db0=one(-offset), bound=bp.BoundPool(f=0.15, k=30.0, t1=1.0, t2=12e-6, model=model))
        seq = bp.SequenceBuilder().rf(0.5, amplitude=64.0, n_steps=5000).build()  # 100 us: sub-stepped
        return float(bp.simulate(ph, seq).m[0, 2])

    assert z("generalized") == pytest.approx(z("rate"), abs=2e-3)


def test_engine_gradients():
    ph = single_pool("superlorentzian", f=0.15, k=30.0)
    seq = bp.SequenceBuilder().rf(100e-6, flip_deg=120.0, n_steps=8).delay(2e-3).rf(60e-6, flip_deg=70.0, n_steps=4).build()

    def fn(f, k, t2b_us, rf):
        from dataclasses import replace

        p = replace(ph, bound=replace(ph.bound, f=f, k=k, t2=t2b_us * 1e-6))
        r = bp.simulate(p, replace(seq, rf=rf))
        return r.m, r.m_bound

    args = [torch.tensor([0.15], dtype=torch.float64, requires_grad=True), torch.tensor([30.0], dtype=torch.float64, requires_grad=True),
            torch.tensor([12.0], dtype=torch.float64, requires_grad=True), seq.rf.clone().requires_grad_()]
    assert torch.autograd.gradcheck(fn, args, eps=1e-6, atol=1e-7, rtol=1e-4)


def test_longer_memory_modes_fit_their_taper():
    # the slowest pole follows the memory; 30 modes still fit the tapered G closely
    x = np.linspace(0, 600, 30001)
    p, c = gb.modes("superlorentzian", 300.0)
    fit = (c[None] * np.exp(-p[None] * x[:, None])).sum(1).real
    assert len(p) == 30 and np.abs(fit - gb.green(x, 1.0, "superlorentzian", 300.0)).max() < 5e-4
    assert np.array_equal(gb.modes("superlorentzian")[0], gb.modes("superlorentzian", gb.MEMORY)[0])


@pytest.mark.parametrize("memory", [100.0, 300.0])
def test_engine_memory_matches_reference_solver(memory, monkeypatch):
    # a 1 ms pi pulse (83 T2): the taper matters (the two memories differ by > 2e-3 in z_s)
    t_rf, alpha = 1e-3, math.pi
    seq = bp.SequenceBuilder().rf(t_rf, flip_deg=180.0, n_steps=400).build()
    zb = float(bp.simulate(single_pool("superlorentzian", memory=memory), seq).m_bound[0])
    orig = gb.green
    monkeypatch.setattr(gb, "green", lambda s, t2, sh, memory_=memory: orig(s, t2, sh, memory_))
    h = t_rf / 4000
    ref = gb.simulate_pulse(gb.rectangular(alpha, t_rf, h), h, t2s=12e-6, shape="superlorentzian", r1s=1.0)["zs"][-1]
    assert zb == pytest.approx(ref, abs=5e-5)
    other = float(bp.simulate(single_pool("superlorentzian", memory=400.0 - memory), seq).m_bound[0])
    assert abs(other - zb) > 2e-3
