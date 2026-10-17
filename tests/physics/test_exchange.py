"""Magnetization transfer (bound pool, ``exchange.py``) against independent references: SciPy
quadrature for the lineshape, the matrix exponential for the longitudinal exchange step, the
SciPy ODE solver for the full two-pool Bloch-McConnell equations, and the linear-algebra
steady state under continuous-wave saturation."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.integrate import quad, solve_ivp

from blochparty import Phantom, SequenceBuilder, simulate
from blochparty.exchange import (
    BoundPool,
    lineshape,
    lorentzian,
    modes_lineshape,
    pool_modes,
    rf_frequency,
    superlorentzian,
    z_exchange,
)

from cases import bm_case

f64 = torch.float64


def sl_quad(x):
    u0 = 1 / math.sqrt(3)
    g = lambda u: math.exp(-2 * (x / abs(3 * u * u - 1)) ** 2) / abs(3 * u * u - 1)
    return math.sqrt(2 * math.pi) * (quad(g, 0, u0, limit=500)[0] + quad(g, u0, 1, limit=500)[0])


def test_superlorentzian_matches_quadrature_and_has_unit_area():
    t2 = torch.tensor(12e-6, dtype=f64)
    for hz in (1.5e3, 4e3, 1e4, 3e4, 6e4):
        x = 2 * math.pi * hz * 12e-6
        assert float(superlorentzian(torch.tensor(hz, dtype=f64), t2)) == pytest.approx(12e-6 * sl_quad(x), rel=2e-5)
    d = torch.logspace(-3, 7, 400_001, dtype=f64)  # G = pi g, g of unit area over rad/s
    area = lambda G: 2 * float(torch.trapezoid(G, 2 * math.pi * d)) / math.pi
    assert area(superlorentzian(d, t2, cutoff=1e-9)) == pytest.approx(1, abs=2e-4)
    x_lo, x_hi = (2 * math.pi * float(d[i]) * 12e-6 for i in (0, -1))  # heavy tails: exact truncated area
    assert area(lorentzian(d, t2)) == pytest.approx(2 / math.pi * (math.atan(x_hi) - math.atan(x_lo)), abs=1e-6)
    # held constant inside the cutoff
    assert float(superlorentzian(torch.tensor(10.0, dtype=f64), t2)) == float(
        superlorentzian(torch.tensor(1e3, dtype=f64), t2)
    )


def test_lineshape_without_cut_is_the_tapered_green_functions_transform():
    """cutoff=None: G(2 pi delta) = T2b Re sum_j c_j / (p_j + i x) over the Green's function's modes equals
    the half-line Fourier transform of the tapered super-Lorentzian Green's function itself
    (``generalized_bloch.green``, memory 300 T2b, trapezoid on 8e5 points to 3000 T2b) to 1.3e-4 up to
    10 kHz at T2b = 10 us (the modes fit G to 4e-5; 3.8e-3 at 30 kHz, where G is 2e-3 of its peak), is
    finite on resonance (8.39 T2b), differs from the untapered table by <= 0.75 % up to 10 kHz, is even in
    delta, equals the Lorentzian exactly for a Lorentzian pool, and is differentiable."""
    from blochparty.generalized_bloch import green

    t2 = 10e-6
    pool = BoundPool(f=0.1, k=30.0, t1=1.0, t2=torch.tensor(t2, dtype=f64), cutoff=None)
    x = np.concatenate([np.linspace(0, 10, 200_001), np.linspace(10, 3000, 600_001)[1:]])  # lags in T2b
    g = green(x, 1.0, "superlorentzian", 300.0)
    for hz, tol in ((0.0, 5e-5), (447.0, 1e-5), (1e3, 2e-5), (3e3, 1e-4), (1e4, 3e-4), (3e4, 1e-2)):
        d = torch.tensor(hz, dtype=f64)
        quad = t2 * np.trapezoid(g * np.cos(2 * math.pi * hz * t2 * x), x)
        assert float(lineshape(pool, d)) == pytest.approx(quad, rel=tol), hz
        if 0 < hz <= 1e4:
            assert float(lineshape(pool, d)) == pytest.approx(float(superlorentzian(d, pool.t2, cutoff=1e-9)), rel=1e-2)
    assert float(lineshape(pool, torch.tensor(0.0, dtype=f64))) == pytest.approx(8.389 * t2, rel=1e-3)
    d = torch.linspace(-5e3, 5e3, 11, dtype=f64)
    assert float((lineshape(pool, d) - lineshape(pool, -d)).abs().max()) < 1e-18
    lor = BoundPool(f=0.1, k=30.0, t1=1.0, t2=torch.tensor(t2, dtype=f64), lineshape="lorentzian", cutoff=None)
    assert float((lineshape(lor, d) - lorentzian(d, lor.t2)).abs().max()) == 0.0
    p, c = pool_modes(pool, d)  # T2b as 1e-5 x s so the finite-difference step is small against it
    assert torch.autograd.gradcheck(
        lambda dd, s: modes_lineshape(dd, 1e-5 * s, p, c),
        (d.clone().requires_grad_(), torch.tensor(1.0, dtype=f64, requires_grad=True)),
    )
    with pytest.raises(ValueError, match="cutoff"):
        BoundPool(f=0.1, k=30.0, t1=1.0, t2=t2, cutoff=0.0)


def test_rf_frequency_sign_matches_resonance():
    # a pulse written as exp(-i 2 pi f t) must be resonant with spins at offset +f, and
    # rf_frequency must report +f
    f, dt, k = 1500.0, 10e-6, 200
    t = (torch.arange(k, dtype=f64) + 0.5) * dt
    amp = 90.0 / 360.0 / (k * dt)
    seq = (
        SequenceBuilder()
        .rf_waveform(
            torch.stack([amp * torch.cos(-2 * math.pi * f * t), amp * torch.sin(-2 * math.pi * f * t)], -1), dt
        )
        .build()
    )
    assert torch.allclose(rf_frequency(seq), torch.full((k,), f, dtype=f64))
    ph = Phantom(
        pos=torch.zeros(2, 3, dtype=f64),
        pd=torch.ones(2, dtype=f64),
        t1=torch.full((2,), 1e9, dtype=f64),
        t2=torch.full((2,), 1e9, dtype=f64),
        db0=torch.tensor([f, -f], dtype=f64),
    )
    mz = simulate(ph, seq).m[:, 2]
    assert abs(float(mz[0])) < 2e-3 and float(mz[1]) > 0.9  # 90 deg at +f (a sampled ramp: 6e-4); off resonance at -f
    # carried through subdivision, where the pieces of a sample have no phase step
    assert torch.allclose(rf_frequency(seq.subdivide_rf(3)), torch.full((3 * k,), f, dtype=f64))


def test_z_exchange_matches_matrix_exponential():
    g = torch.Generator().manual_seed(0)
    rnd = lambda *s: torch.rand(*s, generator=g, dtype=f64)
    n = 50
    pool = BoundPool(f=0.05 + 0.2 * rnd(n), k=1 + 60 * rnd(n), t1=0.3 + rnd(n), t2=torch.full((n,), 12e-6, dtype=f64))
    pd, r1, w = 0.5 + rnd(n), 0.5 + 2 * rnd(n), 300 * rnd(n)
    mz, mb, t = rnd(n) * 2 - 1, rnd(n) * 0.2, torch.tensor(0.37, dtype=f64)
    a, b = z_exchange(mz, mb, t, pd, r1, pool, w)
    kf, kb, r1b = pool.k * pool.f, pool.k, 1 / pool.t1
    A = torch.zeros(n, 3, 3, dtype=f64)
    A[:, 0, 0], A[:, 0, 1], A[:, 0, 2] = -(r1 + kf), kb, r1 * pd
    A[:, 1, 0], A[:, 1, 1], A[:, 1, 2] = kf, -(r1b + kb + w), r1b * pool.f * pd
    ref = torch.linalg.matrix_exp(A * t) @ torch.stack([mz, mb, torch.ones(n, dtype=f64)], -1)[..., None]
    assert torch.allclose(a, ref[:, 0, 0], atol=1e-13) and torch.allclose(b, ref[:, 1, 0], atol=1e-13)
    # no exchange, no saturation: plain T1 recovery for both pools
    pool0 = replace(pool, k=torch.zeros(n, dtype=f64))
    a, b = z_exchange(mz, mb, t, pd, r1, pool0, None)
    assert torch.allclose(a, pd + (mz - pd) * torch.exp(-t * r1), atol=1e-14)
    assert torch.allclose(b, pool.f * pd + (mb - pool.f * pd) * torch.exp(-t / pool.t1), atol=1e-14)


def bm_ode(phantom: Phantom, seq):
    """Two-pool Bloch-McConnell equations, segment by segment with SciPy; W from the same
    lineshape and per-segment RF frequency (model inputs), everything else independent."""
    pool = phantom.bound.expand(len(phantom), phantom.pd)
    f_rf = rf_frequency(seq)
    out = []
    for p in range(len(phantom)):
        pd, r1, r2 = float(phantom.pd[p]), 1 / float(phantom.t1[p]), 1 / float(phantom.t2[p])
        f, k, r1b = float(pool.f[p]), float(pool.k[p]), 1 / float(pool.t1[p])
        m = np.array([0.0, 0.0, pd, f * pd])
        for s in range(len(seq)):
            bx, by = seq.rf[s].numpy()
            bz = float(phantom.pos[p] @ seq.grad[s] + phantom.db0[p])
            w = 0.0
            if seq.is_rf[s]:
                delta = torch.tensor(float(f_rf[s]) - bz - float(pool.dw[p]), dtype=f64)
                w = (2 * math.pi) ** 2 * (bx * bx + by * by) * float(lineshape(pool[p], delta))
            b = 2 * math.pi * np.array([bx, by, bz])

            def rhs(_, m):
                dm = np.zeros(4)
                dm[:3] = np.cross(m[:3], b) - np.array([r2 * m[0], r2 * m[1], r1 * (m[2] - pd)])
                dm[2] += -k * f * m[2] + k * m[3]
                dm[3] = r1b * (f * pd - m[3]) + k * f * m[2] - k * m[3] - w * m[3]
                return dm

            m = solve_ivp(rhs, (0, float(seq.dt[s])), m, method="DOP853", rtol=1e-11, atol=1e-13).y[:, -1]
            if seq.spoil[s]:
                m[:2] = 0
        out.append(m)
    return np.array(out)


@pytest.mark.parametrize("shape", ["lorentzian", "superlorentzian"])
@pytest.mark.parametrize("seed", [0, 1])
def test_bloch_mcconnell_matches_ode_second_order(seed, shape):
    ph, seq = bm_case(seed, shape)
    ref = bm_ode(ph, seq)
    err = {}
    for k in (2, 4, 8):
        res = simulate(ph, seq.subdivide_rf(k))
        got = torch.cat([res.m, res.m_bound[:, None]], 1).numpy()
        err[k] = np.abs(got - ref).max()
    assert err[8] < 1e-5, err
    for k in (2, 4):
        assert 3.5 < err[k] / err[2 * k] < 5.0, err
    # the bound pool was actually saturated (not a trivially passing case)
    assert (ref[:, 3] < 0.97 * (ph.bound.f * ph.pd).numpy()).all()


@pytest.mark.slow
def test_cw_saturation_reaches_linear_algebra_steady_state():
    # long continuous-wave saturation: the free pool's Mz approaches the steady state of the
    # 4x4 linear system (the RF is static in its own frame); classic MT (Z-spectrum) physics.
    # Fast relaxation reaches steady state in 2 s. The step (100 us) must not be commensurate
    # with the precession: at 4 kHz a 250 us step turns 2 pi, and the splitting error of every
    # step then adds up instead of averaging out (a 1.7e-3 bias, the same at 1 ms).
    pd, t1, t2, amp = 1.0, 0.2, 0.05, 85.0
    pool = BoundPool(f=0.15, k=40.0, t1=0.2, t2=12e-6, lineshape="superlorentzian")
    for offset in (1500.0, 4000.0):
        ph = Phantom(
            pos=torch.zeros(1, 3, dtype=f64),
            pd=torch.ones(1, dtype=f64),
            t1=torch.full((1,), t1, dtype=f64),
            t2=torch.full((1,), t2, dtype=f64),
            db0=torch.full((1,), -offset, dtype=f64),
            bound=pool,
        )
        res = simulate(ph, SequenceBuilder().rf(2.0, amplitude=amp, n_steps=20000).build())
        w = (2 * math.pi * amp) ** 2 * float(
            superlorentzian(torch.tensor(offset, dtype=f64), torch.tensor(12e-6, dtype=f64))
        )
        b = 2 * math.pi * np.array([amp, 0.0, -offset])
        kf, kb = pool.k * pool.f, pool.k
        A = np.zeros((4, 4))
        A[:3, :3] = -np.array([[0, -b[2], b[1]], [b[2], 0, -b[0]], [-b[1], b[0], 0]])  # m x b = -[b]x m
        A[:3, :3] -= np.diag([1 / t2, 1 / t2, 1 / t1 + kf])
        A[2, 3] = kb
        A[3, 2], A[3, 3] = kf, -(1 / pool.t1 + kb + w)
        ss = np.linalg.solve(A, -np.array([0, 0, pd / t1, pool.f * pd / pool.t1]))
        assert ss[2] < 0.9  # real saturation
        assert float(res.m[0, 2]) == pytest.approx(ss[2], abs=5e-6), offset
        assert float(res.m_bound[0]) == pytest.approx(ss[3], abs=5e-6), offset


@pytest.mark.slow
def test_gradients_reach_bound_pool_parameters():
    ph, seq = bm_case(0, "superlorentzian")
    ph = replace(ph, pos=ph.pos[:2], pd=ph.pd[:2], t1=ph.t1[:2], t2=ph.t2[:2], db0=ph.db0[:2], bound=ph.bound[:2])
    params = [
        ph.bound.f.clone().requires_grad_(),
        ph.bound.k.clone().requires_grad_(),
        (ph.bound.t2 * 1e6).clone().requires_grad_(),
        seq.rf.clone().requires_grad_(),
    ]

    def fn(f, k, t2b_us, rf):  # T2b in us: a finite-difference step stays inside one lineshape-table cell
        p = replace(ph, bound=replace(ph.bound, f=f, k=k, t2=t2b_us * 1e-6))
        res = simulate(p, replace(seq, rf=rf))
        return res.m, res.m_bound

    assert torch.autograd.gradcheck(fn, params, eps=1e-7, atol=1e-6, rtol=1e-4)


def test_no_bound_pool_rejected_by_other_engines():
    ph, seq = bm_case(0, "lorentzian")
    from blochparty.chunked import simulate_chunked

    with pytest.raises(NotImplementedError):
        simulate_chunked(ph, seq)
