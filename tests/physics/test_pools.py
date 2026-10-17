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

from cases import _maps_for, bm_case, pool_case
from helpers import with_readout

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
        P = [(pd, r1, r2, 0.0, 0.0, 0.0)] + [
            (
                float(q.f[s_]) * pd,
                1 / float(q.t1[s_]),
                1 / float(q.t2[s_]),
                float(q.dw[s_]),
                float(q.k[s_]) * float(q.f[s_]),
                float(q.k[s_]),
            )
            for q in pools
        ]
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
                    v = m[3 * i : 3 * i + 3]
                    b = 2 * math.pi * np.array([b1 * bx, b1 * by, bz + dw])
                    dm[3 * i : 3 * i + 3] += np.cross(v, b) - np.array([q2 * v[0], q2 * v[1], q1 * (v[2] - m0)])
                    if i:  # exchange with the water, all three components
                        dm[3 * i : 3 * i + 3] += k_out * m[:3] - k_in * v
                        dm[:3] += k_in * v - k_out * m[:3]
                if bound is not None:
                    f, k, r1b = float(bound.f[s_]), float(bound.k[s_]), 1 / float(bound.t1[s_])
                    dm[2] += -k * f * m[2] + k * m[-1]
                    dm[-1] = r1b * (f * pd - m[-1]) + k * f * m[2] - k * m[-1] - w * m[-1]
                return dm

            m = solve_ivp(rhs, (0, float(seq.dt[s])), m, method="DOP853", rtol=1e-11, atol=1e-13).y[:, -1]
            if seq.spoil[s]:
                for i in range(len(P)):
                    m[3 * i : 3 * i + 2] = 0
        out.append(m)
    return np.array(out)


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
    ph = Phantom(
        pos=torch.zeros(1, 3, dtype=f64),
        pd=one(pd),
        t1=one(1e6),
        t2=one(t2a),
        db0=one(0.0),
        pools=(ExchangePool(f=f, k=k, t1=1e6, t2=t2b, dw=dwb),),
    )
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
    assert torch.allclose(exact.signal, split.signal, atol=2e-5) and torch.allclose(
        exact.m_bound, split.m_bound, atol=2e-6
    )


# ---------------------------------------------------------------- with FieldMaps, motion, diffusion (ROADMAP M1c)


def test_pools_under_field_maps_equal_the_fields_per_spin():
    from blochparty.maps import sample

    ph, seq = pool_case(0, True)
    seq = with_readout(seq)
    maps = _maps_for(ph)
    v = sample(
        maps.channels(), torch.as_tensor(maps.origin, dtype=f64), torch.as_tensor(maps.spacing, dtype=f64), ph.pos
    )
    per_spin = replace(ph, db0=ph.db0 + v[:, 0], b1=v[:, 1] if ph.b1 is None else ph.b1 * v[:, 1])
    a = simulate(ph, seq, maps=maps)
    b = simulate(per_spin, seq, torch.complex(v[:, 2::2], v[:, 3::2]))
    assert (a.signal - b.signal).abs().max() < 1e-14 and (a.m_pools - b.m_pools).abs().max() < 1e-14


def test_constant_rigid_shift_of_pools_is_the_moved_phantom():
    ph, seq = pool_case(1, True)
    seq = with_readout(seq)
    shift = np.array([0.02, -0.01, 0.015])
    rig = bp.Motion(rigid=lambda t: (np.broadcast_to(np.eye(3), (len(t), 3, 3)), np.broadcast_to(shift, (len(t), 3))))
    moving = simulate(replace(ph, motion=rig), seq)
    moved = simulate(replace(ph, pos=ph.pos + torch.as_tensor(shift)), seq)
    assert (moving.signal - moved.signal).abs().max() < 1e-13 and (moving.m_pools - moved.m_pools).abs().max() < 1e-13


@pytest.mark.parametrize("kind", ["velocity", "diffusion"])
def test_inert_pools_moving_match_the_split_core(kind):
    # f -> 0: the exact engine with moving spins equals the single-pool path with the same motion / walk
    ph, seq = bm_case(0, "lorentzian")
    seq = with_readout(seq)
    n = len(ph)
    if kind == "velocity":
        vel = (torch.rand(n, 3, dtype=f64, generator=torch.Generator().manual_seed(7)) - 0.5) * 0.4
        ph = replace(ph, motion=bp.Motion(velocity=vel))
    else:  # gradients 100x (to ~0.4 MHz/m): enough dephasing for the walk to show
        ph = replace(ph, diffusion=torch.full((n,), 3e-9, dtype=f64))
        seq = replace(seq, grad=seq.grad * 100)
    inert = replace(ph, pools=(ExchangePool(f=0.0, k=0.0, t1=1.0, t2=0.1, dw=0.0),))
    # both on the subdivided sequence: the walks are keyed by segment index (splitting error ~ (dt/8)^2)
    fine = seq.subdivide_rf(8)
    split, exact = simulate(ph, fine, seed=5), simulate(inert, fine, seed=5)
    assert torch.allclose(exact.signal, split.signal, atol=2e-6) and torch.allclose(
        exact.m_bound, split.m_bound, atol=2e-6
    )
    still = simulate(replace(inert, motion=None, diffusion=None), fine, seed=5)
    assert (still.signal - exact.signal).abs().max() > 1e-3  # the motion matters


# ---------------------------------------------------------------- with the generalized Bloch bound pool (M1c)


def full_ode_generalized(ph: Phantom, seq):
    """Water + free pools + the generalized bound pool, segment by segment with SciPy, written
    independently of exchange._bm_matrix: the modes u_j complex, du_j/dt = -(p_j / T2b +
    i 2 pi (bz + dw)) u_j + c_j (-wy + i wx) Mzb, dMzb/dt = wy Re S - wx Im S + R1b (f PD - Mzb)
    + exchange, S = sum u_j (generalized_bloch.modes)."""
    from blochparty.exchange import pool_modes

    n = len(ph)
    pools = [q.expand(n, ph.pd) for q in ph.pools]
    bound = ph.bound.expand(n, ph.pd)
    pj, cj = (t.numpy() for t in pool_modes(bound, ph.pd))
    nj = len(pj)
    out = []
    for s_ in range(n):
        pd, r1, r2 = float(ph.pd[s_]), 1 / float(ph.t1[s_]), 1 / float(ph.t2[s_])
        P = [(pd, r1, r2, 0.0, 0.0, 0.0)] + [
            (
                float(q.f[s_]) * pd,
                1 / float(q.t1[s_]),
                1 / float(q.t2[s_]),
                float(q.dw[s_]),
                float(q.k[s_]) * float(q.f[s_]),
                float(q.k[s_]),
            )
            for q in pools
        ]
        f, k, r1b, t2b, dwb = (float(getattr(bound, a)[s_]) for a in ("f", "k", "t1", "t2", "dw"))
        r1b = 1 / r1b
        b1 = float(ph.b1[s_]) if ph.b1 is not None else 1.0
        nf = 3 * len(P)
        m = np.concatenate([np.concatenate([[0, 0, p[0]] for p in P]), [f * pd], np.zeros(2 * nj)])
        for s in range(len(seq)):
            bx, by = (seq.rf[s].numpy() * b1) if seq.is_rf[s] else (0.0, 0.0)
            bz = float(ph.pos[s_] @ seq.grad[s] + ph.db0[s_])
            wx, wy = 2 * math.pi * bx, 2 * math.pi * by

            def rhs(_, m):
                dm = np.zeros_like(m)
                for i, (m0, q1, q2, dw, k_out, k_in) in enumerate(P):
                    v = m[3 * i : 3 * i + 3]
                    w = np.array([wx, wy, 2 * math.pi * (bz + dw)])
                    dm[3 * i : 3 * i + 3] += np.cross(v, w) - np.array([q2 * v[0], q2 * v[1], q1 * (v[2] - m0)])
                    if i:
                        dm[3 * i : 3 * i + 3] += k_out * m[:3] - k_in * v
                        dm[:3] += k_in * v - k_out * m[:3]
                zb = m[nf]
                u = m[nf + 1 :: 2] + 1j * m[nf + 2 :: 2]
                S = u.sum()
                du = -(pj / t2b + 2j * math.pi * (bz + dwb)) * u + cj * (-wy + 1j * wx) * zb
                dm[nf] = wy * S.real - wx * S.imag + r1b * (f * pd - zb) + k * f * m[2] - k * zb
                dm[2] += -k * f * m[2] + k * zb
                dm[nf + 1 :: 2], dm[nf + 2 :: 2] = du.real, du.imag
                return dm

            m = solve_ivp(rhs, (0, float(seq.dt[s])), m, method="DOP853", rtol=1e-10, atol=1e-12).y[:, -1]
            if seq.spoil[s]:
                for i in range(len(P)):
                    m[3 * i : 3 * i + 2] = 0
                m[nf + 1 :] = 0
        out.append(m[: nf + 1])
    return np.array(out)


@pytest.mark.slow
@pytest.mark.parametrize("shape", ["lorentzian", "superlorentzian"])
def test_generalized_bound_pool_with_free_pools_matches_ode(shape):
    ph, seq = pool_case(0, True)
    ph = replace(ph, bound=replace(ph.bound, model="generalized", lineshape=shape))
    keep = slice(0, 2)  # two spins: the ODE carries 2 x 30 mode components for the super-Lorentzian
    ph = replace(
        ph,
        pos=ph.pos[keep],
        pd=ph.pd[keep],
        t1=ph.t1[keep],
        t2=ph.t2[keep],
        db0=ph.db0[keep],
        bound=ph.bound[keep],
        pools=tuple(replace(q, **{a: getattr(q, a)[keep] for a in ("f", "k", "t1", "t2", "dw")}) for q in ph.pools),
    )
    ref = full_ode_generalized(ph, seq)
    out = simulate(ph, seq)
    got = np.concatenate([out.m.numpy(), out.m_pools.numpy().reshape(2, -1), out.m_bound.numpy()[:, None]], 1)
    assert np.abs(got - ref).max() < 2e-8


def test_generalized_with_inert_pools_matches_the_split_core():
    # f -> 0: the exact engine equals the generalized split engine (``_rf_segment_generalized``) to the
    # split's own accuracy (sub-steps <= T2b / 2; for a super-Lorentzian ~1e-6 here)
    ph, seq = bm_case(1, "superlorentzian")
    ph = replace(ph, bound=replace(ph.bound, model="generalized"))
    seq = with_readout(seq)
    inert = replace(ph, pools=(ExchangePool(f=0.0, k=0.0, t1=1.0, t2=0.1, dw=0.0),))
    exact, split = simulate(inert, seq), simulate(ph, seq)
    assert torch.allclose(exact.signal, split.signal, atol=2e-5) and torch.allclose(
        exact.m_bound, split.m_bound, atol=5e-6
    )


@pytest.mark.parametrize("offset", [2e3, 5e3])
def test_generalized_with_pools_approaches_the_rate_model_off_resonance(offset):
    # long off-resonant CW saturation is the rate model's regime (as test_generalized_bloch, without pools);
    # the exact engine takes the 0.5 s pulse as one segment
    def z(model):
        one = torch.ones(1, dtype=f64)
        ph = Phantom(
            pos=torch.zeros(1, 3, dtype=f64),
            pd=one,
            t1=one,
            t2=0.05 * one,
            db0=-offset * one,
            bound=bp.BoundPool(f=0.15, k=30.0, t1=1.0, t2=12e-6, model=model),
            pools=(ExchangePool(f=0.01, k=500.0, t1=1.0, t2=0.01, dw=450.0),),
        )
        return float(simulate(ph, bp.SequenceBuilder().rf(0.5, amplitude=64.0, n_steps=1).build()).m[0, 2])

    assert z("generalized") == pytest.approx(z("rate"), abs=2e-3)


# ---------------------------------------------------------------- the lineshape without a cut (BoundPool.cutoff=None)


def _wm_like(model, db0, cutoff):
    """WM-like tissue at 3 T with an amide and a myelin-water pool; the spin sits at -offset so a
    pulse at zero frequency saturates at +offset from the water."""
    one = torch.ones(1, dtype=f64)
    amide = ExchangePool(f=0.01, k=50.0, t1=1.0, t2=0.02, dw=3.5e-6 * 3 * bp.GAMMA_BAR)
    myelin = ExchangePool(f=0.15, k=10.0, t1=0.5, t2=0.012, dw=0.0)
    return Phantom(
        pos=torch.zeros(1, 3, dtype=f64),
        pd=one,
        t1=one,
        t2=0.07 * one,
        db0=db0 * one,
        bound=bp.BoundPool(f=0.15, k=30.0, t1=1.0, t2=10e-6, model=model, cutoff=cutoff),
        pools=(amide, myelin),
    )


def _pulsed_saturation(n_pulses=36, duration=50e-3, b1_ut=2.0, gap=5e-3):
    b = bp.SequenceBuilder()
    for _ in range(n_pulses):
        b.rf(duration, amplitude=b1_ut * 1e-6 * bp.GAMMA_BAR, n_steps=1).delay(gap)
    return b.build()


@pytest.mark.parametrize("offset", [447.0, 1000.0, 3000.0, 10000.0])
def test_rate_model_without_cut_equals_generalized_for_long_pulses(offset):
    """Pulsed CEST saturation (36 x 50 ms at 2 uT, as the Pulseq-CEST APTw preparation): the rate
    model with the tapered Green's function's lineshape (cutoff=None) equals the generalized Bloch
    model to 5e-5 at every offset, where the default 1 kHz cut differs by 1e-2 in Z at the CEST
    offsets inside it (measured 2026-10-09: 1.15e-2 at 447 Hz, 2.2e-4 at 3 kHz, 8.4e-4 at 10 kHz)."""
    seq = _pulsed_saturation()
    gen = simulate(_wm_like("generalized", -offset, 1e3), seq).m[0]
    nocut = simulate(_wm_like("rate", -offset, None), seq).m[0]
    assert float((nocut - gen).abs().max()) < 5e-5
    if offset < 1e3:
        cut = simulate(_wm_like("rate", -offset, 1e3), seq).m[0]
        assert float((cut - gen).abs().max()) > 5e-3


def test_rate_model_is_wrong_for_short_on_resonance_pulses_with_or_without_cut():
    """Where the generalized model is needed (pulses short against the Green's function's memory,
    300 T2b = 3 ms here, on resonance), neither rate variant reproduces it: 60 hard 0.2 ms 60 deg
    pulses leave the water 1.4e-3 (cut) / 2.3e-2 (no cut) away from the generalized model. The
    cut-free lineshape is for saturation pulses, not a substitute for the generalized model."""
    b = bp.SequenceBuilder()
    for i in range(60):
        b.rf(0.2e-3, flip_deg=60.0, phase=math.pi * (i % 2)).delay(4.8e-3)
    seq = b.build()
    gen = simulate(_wm_like("generalized", 0.0, 1e3), seq).m[0]
    nocut = simulate(_wm_like("rate", 0.0, None), seq).m[0]
    assert float((nocut - gen).abs().max()) > 1e-2
