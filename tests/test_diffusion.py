"""Diffusion as exact Brownian random walks (``Phantom.diffusion``, ``bloch._walk``, ``rng``)
against Stejskal-Tanner (pulsed-gradient spin echo) for isotropic and anisotropic diffusion,
within the Monte Carlo error; plus determinism and agreement of the merged free runs with
segment-by-segment stepping (the same walk, drawn per (seed, spin, segment))."""

import math

import numpy as np
import pytest
import torch

import blochparty as bp
from blochparty.rng import normals, philox

f64 = torch.float64
DELTA_SMALL, DELTA_BIG = 10e-3, 30e-3


def water(n, diffusion, t2=1e6):
    return bp.Phantom(pos=torch.zeros(n, 3, dtype=f64), pd=torch.ones(n, dtype=f64), t1=torch.full((n,), 1e6, dtype=f64),
                      t2=torch.full((n,), t2, dtype=f64), db0=torch.zeros(n, dtype=f64), diffusion=diffusion)


def pgse(g, readout=1):
    """90 - lobe (delta) - 180 - lobe, lobes DELTA_BIG apart; ADC at the echo."""
    b = bp.SequenceBuilder()
    b.rf(10e-6, flip_deg=90.0).delay(DELTA_SMALL, grad=g).delay(DELTA_BIG - DELTA_SMALL - 10e-6)
    b.rf(10e-6, flip_deg=180.0, phase=math.pi / 2).delay(DELTA_SMALL, grad=g).readout(readout, 10e-6)
    return b.build()


def b_value(gnorm):
    return (2 * math.pi * gnorm) ** 2 * DELTA_SMALL**2 * (DELTA_BIG - DELTA_SMALL / 3)


def mc_error(bd, n):
    return math.sqrt(((1 + math.exp(-4 * bd)) / 2 - math.exp(-2 * bd)) / n)


def test_philox_known_answers():
    # Random123 kat_vectors for philox4x32-10
    assert tuple(int(x) for x in philox(0, 0, 0, 0, 0, 0)) == (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)
    assert tuple(int(x) for x in philox(0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344, 0xA4093822, 0x299F31D0)) == (
        0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)
    z = normals(5, np.arange(100000), 7)
    assert np.abs(z.mean(0)).max() < 0.02 and np.abs(z.std(0) - 1).max() < 0.02


@pytest.mark.parametrize("g_mt", [20.0, 40.0, 60.0])
def test_isotropic_stejskal_tanner(g_mt):
    n, d = 100_000, 2e-9
    g = g_mt * 1e-3 * 42.577e6
    s = bp.simulate(water(n, torch.full((n,), d, dtype=f64)), pgse((g, 0.0, 0.0)), seed=3).signal[0, 0]
    bd = b_value(g) * d
    assert abs(abs(float(s.abs())) / n - math.exp(-bd)) < 4 * mc_error(bd, n)


def test_anisotropic_tensor():
    # D = R diag(0.3, 1.0, 2.5) R^T um^2/ms: the attenuation along unit g is exp(-b g^T D g)
    n = 100_000
    th = 0.6
    rot = torch.tensor([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]], dtype=f64)
    dt = rot @ torch.diag(torch.tensor([0.3e-9, 1.0e-9, 2.5e-9], dtype=f64)) @ rot.T
    ph = water(n, dt.expand(n, 3, 3).contiguous())
    gn = 40e-3 * 42.577e6
    for axis in range(3):
        u = torch.zeros(3, dtype=f64)
        u[axis] = 1.0
        s = bp.simulate(ph, pgse(tuple((gn * u).tolist())), seed=11).signal[0, 0]
        bd = b_value(gn) * float(u @ dt @ u)
        assert abs(float(s.abs()) / n - math.exp(-bd)) < 4 * mc_error(bd, n), axis


def test_deterministic_and_merge_independent():
    n = 2000
    ph = water(n, torch.full((n,), 2e-9, dtype=f64), t2=0.08)
    ph = bp.Phantom((torch.rand(n, 3, generator=torch.Generator().manual_seed(0), dtype=f64) - 0.5) * 0.02, ph.pd, ph.t1,
                    ph.t2, ph.db0, diffusion=ph.diffusion)
    seq = pgse((1.2e6, 3e5, -2e5), readout=32)
    a = bp.simulate(ph, seq, seed=4)
    b = bp.simulate(ph, seq, seed=4)
    c = bp.simulate(ph, seq, seed=5)
    d = bp.simulate(ph, seq, seed=4, merge_free_runs=False)
    assert torch.equal(a.signal, b.signal)
    assert (a.signal - c.signal).abs().max() > 1e-3 * a.signal.abs().max()
    assert torch.allclose(a.signal, d.signal, atol=1e-10 * float(a.signal.abs().max()))
    assert torch.allclose(a.m, d.m, atol=1e-12)
