"""Free exchanging pools (``Phantom.pools``: CEST, myelin water) in the exact Bloch-McConnell
engine, against independent references: the SciPy ODE solver on the full system written out
by hand, the closed-form two-site exchange of the transverse magnetization (McConnell), and the
split core when the extra pool is inert."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

import blochparty as bp
from blochparty import ExchangePool, Phantom, simulate
from blochparty.exchange import lineshape, rf_frequency
from test_exchange import bm_case
from test_fused import with_readout

f64 = torch.float64


def full_ode(ph: Phantom, seq):
    """Water + free pools + bound pool, segment by segment with SciPy; Mz/Mxy per pool."""
    n = len(ph)
    pools = [q.expand(n, ph.pd) for q in ph.pools]
    bound = ph.bound.expand(n, ph.pd) if ph.bound is not None else None
    f_rf = rf_frequency(seq)
    out = []
    for s_ in range(n):
        pd, r1, r2 = float(ph.pd[s_]), 1 / float(ph.t1[s_]), 1 / float(ph.t2[s_])
        P = [(pd, r1, r2, 0.0, 0.0, 0.0)] + [(float(q.f[s_]) * pd, 1 / float(q.t1[s_]), 1 / float(q.t2[s_]),
                                              float(q.dw[s_]), float(q.k[s_]) * float(q.f[s_]), float(q.k[s_]))
                                             for q in pools]
        m = np.concatenate([[0, 0, p[0]] for p in P] + ([[float(bound.f[s_]) * pd]] if bound is not None else []))
        for s in range(len(seq)):
            bx, by = seq.rf[s].numpy() if seq.is_rf[s] else (0.0, 0.0)
            b1 = float(ph.b1[s_]) if ph.b1 is not None else 1.0
            bz = float(ph.pos[s_] @ seq.grad[s] + ph.db0[s_])
            w = 0.0
            if bound is not None and seq.is_rf[s]:
                delta = torch.tensor(float(f_rf[s]) - bz - float(bound.dw[s_]), dtype=f64)
                w = (2 * math.pi) ** 2 * b1 * b1 * (bx * bx + by * by) * float(lineshape(bound[s_], delta))

            def rhs(_, m):
                dm = np.zeros_like(m)
                for i, (m0, q1, q2, dw, k_out, k_in) in enumerate(P):
                    v = m[3 * i:3 * i + 3]
                    b = 2 * math.pi * np.array([b1 * bx, b1 * by, bz + dw])
                    dm[3 * i:3 * i + 3] += np.cross(v, b) - np.array([q2 * v[0], q2 * v[1], q1 * (v[2] - m0)])
                    if i:  # exchange with the water, all three components
                        dm[3 * i:3 * i + 3] += k_out * m[:3] - k_in * v
                        dm[:3] += k_in * v - k_out * m[:3]
                if bound is not None:
                    f, k, r1b = float(bound.f[s_]), float(bound.k[s_]), 1 / float(bound.t1[s_])
                    dm[2] += -k * f * m[2] + k * m[-1]
                    dm[-1] = r1b * (f * pd - m[-1]) + k * f * m[2] - k * m[-1] - w * m[-1]
                return dm

            m = solve_ivp(rhs, (0, float(seq.dt[s])), m, method="DOP853", rtol=1e-11, atol=1e-13).y[:, -1]
            if seq.spoil[s]:
                for i in range(len(P)):
                    m[3 * i:3 * i + 2] = 0
        out.append(m)
    return np.array(out)


def pool_case(seed, with_bound=True):
    ph, seq = bm_case(seed, "superlorentzian")
    g = torch.Generator().manual_seed(seed + 5)
    rnd = lambda: torch.rand(len(ph), generator=g, dtype=f64)
    amide = ExchangePool(f=0.01 + 0.02 * rnd(), k=50 + 200 * rnd(), t1=torch.ones(len(ph), dtype=f64),
                         t2=0.02 + 0.08 * rnd(), dw=3.5e-6 * 3 * bp.GAMMA_BAR + 100 * rnd())
    myelin = ExchangePool(f=0.15 + 0.05 * rnd(), k=5 + 10 * rnd(), t1=0.4 + 0.2 * rnd(), t2=0.008 + 0.01 * rnd(),
                          dw=10 * rnd())
    return replace(ph, pools=(amide, myelin), bound=ph.bound if with_bound else None), seq


@pytest.mark.parametrize("with_bound", [True, False])
@pytest.mark.parametrize("seed", [0, 1])
def test_exact_engine_matches_ode(seed, with_bound):
    ph, seq = pool_case(seed, with_bound)
    ref = full_ode(ph, seq)
    r = simulate(ph, seq)
    got = [r.m.numpy(), r.m_pools.reshape(len(ph), -1).numpy()]
    if with_bound:
        got.append(r.m_bound[:, None].numpy())
    assert np.abs(np.concatenate(got, 1) - ref).max() < 1e-8


def test_two_site_exchange_free_decay():
    # McConnell: transverse a, b of two exchanging pools decay as exp(M t) with
    # M = [[-(R2a + i wa + kab), kba], [kab, -(R2b + i wb + kba)]], signal a + b
    pd, f, k, t2a, t2b, dwb = 1.0, 0.2, 12.0, 0.08, 0.012, 25.0
    one = lambda v: torch.full((1,), v, dtype=f64)
    ph = Phantom(pos=torch.zeros(1, 3, dtype=f64), pd=one(pd), t1=one(1e6), t2=one(t2a), db0=one(0.0),
                 pools=(ExchangePool(f=f, k=k, t1=1e6, t2=t2b, dw=dwb),))
    dwell = 0.5e-3
    seq = bp.SequenceBuilder().rf(1e-6, flip_deg=90.0).readout(200, dwell).build()
    sig = simulate(ph, seq).signal[0].numpy()
    kab, kba = k * f, k
    M = np.array([[-(1 / t2a + kab), kba], [kab, -(1 / t2b + 2j * math.pi * dwb + kba)]])
    lam, V = np.linalg.eig(M)
    x0 = np.array([1j * pd, 1j * f * pd])  # 90 deg about x: +z -> +y for both pools
    t = (np.arange(200) + 0.5) * dwell + 0.5e-6  # from the centre of the 1 us hard pulse
    ref = np.array([(V @ (np.exp(lam * tt) * np.linalg.solve(V, x0))).sum() for tt in t])
    assert np.abs(sig - ref).max() < 1e-5 * np.abs(ref).max()


def test_inert_pool_matches_split_core():
    ph, seq = bm_case(0, "lorentzian")
    seq = with_readout(seq)
    inert = replace(ph, pools=(ExchangePool(f=0.0, k=0.0, t1=1.0, t2=0.1, dw=0.0),))
    split = simulate(ph, seq.subdivide_rf(8))  # splitting error ~ (dt/8)^2
    exact = simulate(inert, seq)
    assert torch.allclose(exact.signal, split.signal, atol=2e-5) and torch.allclose(exact.m_bound, split.m_bound, atol=2e-6)


# ---------------------------------------------------------------- the fused kernel (fused/bm.h)

POOLF = ("f", "k", "t1", "t2", "dw")


def _pool_grads(sim, ph, seq, coils, **kw):
    req = lambda t: t.detach().clone().requires_grad_()
    spin = {f: req(getattr(ph, f)) for f in ("pos", "pd", "t1", "t2", "db0", "b1")}
    pools = [{f: req(getattr(q, f)) for f in POOLF} for q in ph.pools]
    bound = {f: req(getattr(ph.bound, f).expand(len(ph))) for f in POOLF} if ph.bound is not None else {}
    sq = {f: req(getattr(seq, f)) for f in ("dt", "rf", "grad", "adc_phase")}
    leaves = {**spin, **{f"pool{i}.{k}": v for i, q in enumerate(pools) for k, v in q.items()},
              **{"bound." + k: v for k, v in bound.items()}, **sq, "coils": req(coils)}
    p = Phantom(**spin, pools=tuple(ExchangePool(**q) for q in pools),
                bound=replace(ph.bound, **bound) if ph.bound is not None else None)
    r = sim(p, replace(seq, **sq), leaves["coils"], **kw)
    g = torch.Generator().manual_seed(3)
    loss = ((r.signal * torch.randn(r.signal.shape, dtype=torch.complex128, generator=g)).real.sum()
            + (r.signal.abs() ** 2).sum() + (r.m * torch.randn(r.m.shape, dtype=f64, generator=g)).sum()
            + (r.m_pools**2).sum())
    if r.m_bound is not None:
        loss = loss + (r.m_bound * torch.randn(r.m_bound.shape, dtype=f64, generator=g)).sum()
    return dict(zip(leaves, torch.autograd.grad(loss, list(leaves.values()))))


@pytest.mark.parametrize("with_bound", [True, False])
def test_fused_forward_and_adjoint(with_bound):
    fused = pytest.importorskip("blochparty.fused")
    from test_fused import rel

    ph, seq = pool_case(0, with_bound)
    seq = with_readout(seq)
    ph = replace(ph, b1=torch.linspace(0.8, 1.2, len(ph), dtype=f64))
    coils = bp.birdcage_sensitivities(ph.pos, 3)
    ref = simulate(ph, seq, coils)
    for dtype, tol in ((f64, 1e-12), (torch.float32, 3e-5)):
        out = fused.simulate_fused(ph.to(dtype=dtype), seq.to(dtype=dtype), coils)
        assert rel(out.signal, ref.signal) < tol and rel(out.m, ref.m) < tol and rel(out.m_pools, ref.m_pools) < tol
    g_ref = _pool_grads(simulate, ph, seq, coils)
    g_out = _pool_grads(fused.simulate_fused, ph, seq, coils, checkpoint_every=5)
    errs = {k: rel(g_out[k], g_ref[k]) for k in g_ref}
    assert max(errs.values()) < 1e-10, errs
