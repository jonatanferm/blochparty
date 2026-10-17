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

from cases import DELTA_BIG, DELTA_SMALL, pgse

f64 = torch.float64


def water(n, diffusion, t2=1e6):
    return bp.Phantom(pos=torch.zeros(n, 3, dtype=f64), pd=torch.ones(n, dtype=f64), t1=torch.full((n,), 1e6, dtype=f64),
                      t2=torch.full((n,), t2, dtype=f64), db0=torch.zeros(n, dtype=f64), diffusion=diffusion)


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


def epg_cpmg_diffusion(alpha, n_echo, esp, d, dk):
    """|echo| of a CPMG train (90x, then refocusing alpha about y), each half echo spacing a
    crusher of dephasing dk [rad/m]; extended phase graphs with isotropic diffusion (Weigel et
    al., JMR 2010): over an interval tau with k -> k + dk, transverse states attenuate by
    exp(-d tau (k^2 + k dk + dk^2 / 3)), longitudinal ones by exp(-d tau k^2). No relaxation."""
    kmax = 2 * n_echo + 4
    fp, fm, z = (np.zeros(kmax, complex) for _ in range(3))
    z[0] = 1.0

    def rf(a, phi):
        nonlocal fp, fm, z
        c2, s2, sa = math.cos(a / 2) ** 2, math.sin(a / 2) ** 2, math.sin(a)
        e = np.exp(1j * phi)
        fp, fm, z = (c2 * fp + e**2 * s2 * fm - 1j * e * sa * z,
                     np.conj(e) ** 2 * s2 * fp + c2 * fm + 1j * np.conj(e) * sa * z,
                     -0.5j * np.conj(e) * sa * fp + 0.5j * e * sa * fm + math.cos(a) * z)

    def interval(tau):
        nonlocal fp, fm, z
        n = np.arange(kmax)
        kp, km = n * dk, -n * dk
        fp = fp * np.exp(-d * tau * (kp**2 + kp * dk + dk**2 / 3))
        fm = fm * np.exp(-d * tau * (km**2 + km * dk + dk**2 / 3))
        z = z * np.exp(-d * tau * kp**2)
        fp = np.concatenate([[0], fp[:-1]])
        fm = np.concatenate([fm[1:], [0]])
        fp[0] = np.conj(fm[0])

    rf(math.pi / 2, 0.0)
    echoes = []
    for _ in range(n_echo):
        interval(esp / 2)
        rf(alpha, math.pi / 2)
        interval(esp / 2)
        echoes.append(abs(fp[0]))
    return np.array(echoes)


def test_cpmg_stimulated_echoes_match_epg_with_diffusion(gpu):
    # crushers make every echo a sum of spin- and stimulated-echo pathways, each attenuated by
    # its own diffusion history; walkers spread uniformly over whole crusher wavelengths
    n, d, esp, n_echo, alpha = 1_000_000, 2.5e-9, 10e-3, 8, math.radians(150)
    g = 2.0e6  # [Hz/m] for esp / 2: dk = 2 pi g esp / 2
    dk = 2 * math.pi * g * esp / 2
    wavelength = 1 / (g * esp / 2)
    x = (torch.arange(n, dtype=f64) + 0.5) / n * 50 * wavelength
    ph = bp.Phantom(pos=torch.stack([x, torch.zeros(n, dtype=f64), torch.zeros(n, dtype=f64)], -1),
                    pd=torch.ones(n, dtype=f64), t1=torch.full((n,), 1e6, dtype=f64), t2=torch.full((n,), 1e6, dtype=f64),
                    db0=torch.zeros(n, dtype=f64), diffusion=torch.full((n,), d, dtype=f64))
    b = bp.SequenceBuilder().rf(1e-6, flip_deg=90.0)
    for _ in range(n_echo):
        b.delay(esp / 2 - 1e-6, grad=(g * esp / 2 / (esp / 2 - 1e-6), 0, 0))
        b.rf(1e-6, flip_deg=math.degrees(alpha), phase=math.pi / 2)
        b.delay(esp / 2 - 1e-6, grad=(g * esp / 2 / (esp / 2 - 1e-6), 0, 0)).readout(1, 2e-6)
    sim = gpu.simulate(ph, b.build(), seed=1).signal[0].abs().numpy() / n
    ref = epg_cpmg_diffusion(alpha, n_echo, esp, d, dk)
    no_diff = epg_cpmg_diffusion(alpha, n_echo, esp, 0.0, dk)
    assert np.abs(no_diff - ref).max() > 0.05  # diffusion matters here
    assert np.abs(sim - ref).max() < 3e-3, (sim, ref)
