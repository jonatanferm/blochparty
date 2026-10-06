"""Drift and physiological fields (``drift.py``): the phase of spins at known positions
under a linear / exponential drift and a respiratory field (uniform plus linear terms)
equals the closed-form integral of the field, in both simulators; parameter gradients
match finite differences."""

import math
from dataclasses import replace

import pytest
import torch

from blochparty import Phantom, SequenceBuilder, simulate
from blochparty.drift import exponential_drift, linear_drift, respiration

fused = pytest.importorskip("blochparty.fused")
SIMS = {"core": simulate, "fused": fused.simulate_fused}


@pytest.fixture(scope="module", autouse=True)
def extension():
    try:
        fused._ext()
    except Exception as e:  # no compiler / ninja: nothing to test
        pytest.skip(f"fused extension unavailable: {e}")


def setup(n=8):
    """Excite, then sample a long gradient-free FID (with a gradient blip in the middle)."""
    b = SequenceBuilder()
    b.rf(1e-6, flip_deg=90.0)
    b.readout(200, 1e-3)
    b.delay(3e-3, grad=(2e3, -1e3, 5e2))
    b.readout(150, 2e-3)
    seq = b.build()
    pos = (torch.rand(n, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) - 0.5) * 0.2
    one = torch.ones(n, dtype=torch.float64)
    return Phantom(pos, one, 1e9 * one, 1e9 * one, 0 * one), seq, torch.eye(n, dtype=torch.complex128)


def adc_times(seq):
    """ADC sample times and the pulse's end [s]."""
    tb = torch.cumsum(seq.dt, 0)
    return tb[torch.tensor(seq.adc)], float(tb[0])


def check(sim, played, seq, ph, coils, integral):
    """Signal ratio vs exp(-i 2 pi int field) for (n_spins, n_adc) integrals [cycles] from the
    end of the pulse. The field is dropped during the pulse, where it would tilt the
    rotation axis rather than add phase."""
    off, grad = played.b0_offset.clone(), played.grad.clone()
    off[0], grad[0] = 0.0, seq.grad[0]
    played = replace(played, b0_offset=off, grad=grad)
    ratio = SIMS[sim](ph, played, coils).signal / SIMS[sim](ph, seq, coils).signal
    expect = torch.exp(-2j * math.pi * integral)
    assert (ratio - expect).abs().max() < 1e-9
    assert (expect - 1).abs().max() > 0.1


@pytest.mark.parametrize("sim", SIMS)
def test_linear_drift(sim):
    ph, seq, coils = setup()
    off, rate, grate = 3.0, 25.0, torch.tensor([40.0, -20.0, 10.0], dtype=torch.float64)  # Hz, Hz/s, Hz/m/s
    t, t0 = adc_times(seq)
    integral = off * (t - t0) + 0.5 * (rate + (ph.pos @ grate)[:, None]) * (t**2 - t0**2)
    check(sim, linear_drift(seq, off, rate, grate), seq, ph, coils, integral)


@pytest.mark.parametrize("sim", SIMS)
def test_exponential_drift(sim):
    ph, seq, coils = setup()
    amp, tau, gamp = 4.0, 0.15, torch.tensor([30.0, 0.0, -15.0], dtype=torch.float64)
    t, t0 = adc_times(seq)
    shape = (t - t0) - tau * (math.exp(-t0 / tau) - torch.exp(-t / tau))  # int_t0^t (1 - exp(-t'/tau))
    integral = (amp + (ph.pos @ gamp)[:, None]) * shape
    check(sim, exponential_drift(seq, amp, tau, gamp), seq, ph, coils, integral)


@pytest.mark.parametrize("sim", SIMS)
def test_respiration(sim):
    ph, seq, coils = setup()
    amp, period, phase = 5.0, 0.21, 0.4
    grad = torch.tensor([20.0, 35.0, -10.0], dtype=torch.float64)  # Hz/m
    t, t0 = adc_times(seq)
    w = 2 * math.pi / period
    shape = (math.cos(w * t0 + phase) - torch.cos(w * t + phase)) / w  # int_t0^t sin(w t' + phase)
    integral = (amp + (ph.pos @ grad)[:, None]) * shape
    check(sim, respiration(seq, amp, period, grad, phase), seq, ph, coils, integral)


def test_parameter_gradients():
    ph, seq, coils = setup(4)
    names = ("amp", "period", "phase", "rate", "tau")
    base = torch.tensor([5.0, 0.21, 0.4, 25.0, 0.15], dtype=torch.float64)

    def loss(p, sim=fused.simulate_fused):
        played = respiration(seq, p[0], p[1], torch.stack([p[0] * 2, -p[0], p[0]]), p[2])
        played = exponential_drift(linear_drift(played, 1.0, p[3]), 3.0, p[4])
        r = sim(ph, played, coils)
        w = torch.randn(r.signal.shape, dtype=torch.complex128, generator=torch.Generator().manual_seed(1))
        return (r.signal * w).real.sum()

    p = base.clone().requires_grad_()
    (g,) = torch.autograd.grad(loss(p), [p])
    q = base.clone().requires_grad_()
    (g_core,) = torch.autograd.grad(loss(q, simulate), [q])
    assert torch.allclose(g, g_core, rtol=1e-10, atol=0)
    for i, name in enumerate(names):
        h = 1e-6 * float(base[i])
        e = torch.zeros(5, dtype=torch.float64)
        e[i] = h
        with torch.no_grad():
            fd = (loss(base + e) - loss(base - e)) / (2 * h)
        assert abs(float(g[i] - fd)) < 1e-6 * abs(float(fd)), name
