"""Gradient impulse responses and eddy currents (``girf.apply_girf``) and the uniform
per-segment field ``Sequence.b0_offset``.

Checked against: k-space of a delayed sequence from NumPy interpolation of the nominal
moment, closed-form exponential responses to a trapezoid (true ramps and the raster
staircase), an FFT (transfer-function) convolution on the raster, the analytic phase of a
B0 eddy current, a sequential reference for the blocked scan, and finite differences for
the eddy parameters through both simulators.
"""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

import blochparty as bp
from blochparty import GAMMA_BAR, Phantom, SequenceBuilder, Sequence, simulate, trapezoid
from blochparty.girf import GIRF, EddyCurrents, _linear_scan, apply_girf
from test_free_runs import random_phantom
from test_fused import rel

fused = pytest.importorskip("blochparty.fused")
SIMS = {"core": simulate, "fused": fused.simulate_fused}
RASTER = 10e-6


@pytest.fixture(scope="module", autouse=True)
def extension():
    try:
        fused._ext()
    except Exception as e:  # no compiler / ninja: nothing to test
        pytest.skip(f"fused extension unavailable: {e}")


def moments(seq):
    """(S + 1,) boundary times and (S + 1, 3) k-space moments."""
    tb = torch.cat([seq.dt.new_zeros(1), torch.cumsum(seq.dt, 0)])
    return tb, torch.cat([seq.grad.new_zeros(1, 3), torch.cumsum(seq.grad * seq.dt[:, None], 0)])


def test_linear_scan_matches_recurrence():
    g = torch.Generator().manual_seed(0)
    for n in (1, 7, 16, 17, 300, 5000):
        log_a = -torch.rand(2, n, dtype=torch.float64, generator=g) * torch.tensor([[0.1], [50.0]], dtype=torch.float64)
        b = torch.randn(2, n, dtype=torch.float64, generator=g)
        x, ref = torch.zeros(2, dtype=torch.float64), []
        for s in range(n):
            x = torch.exp(log_a[:, s]) * x + b[:, s]
            ref.append(x)
        assert rel(_linear_scan(log_a, b), torch.stack(ref, -1)) < 1e-12


@pytest.mark.parametrize("delay", [0.0, 3e-6, 23.5e-6])
def test_pure_delay_shifts_kspace(delay):
    # moment of the played gradient at every segment end = nominal moment at t - delay
    seq = bp.gre_2d(16, ramps=True, slice_thickness=5e-3, spoiling="gradient", n_dummy=0)
    delays = torch.tensor([delay, 2 * delay, -0.5 * delay], dtype=torch.float64)
    played = apply_girf(seq, GIRF.identity(delays))
    tb, k = moments(seq)
    _, kp = moments(played)
    for axis in range(3):
        ref = np.interp(tb.numpy() - float(delays[axis]), tb.numpy(), k[:, axis].numpy(), left=0.0)
        assert np.abs(kp[:, axis].numpy() - ref).max() < 1e-9  # [1/m], k reaches ~1e3
    assert torch.equal(played.b0_offset, torch.zeros(len(seq), dtype=torch.float64))
    if delay == 0.0:
        assert rel(played.grad, seq.grad) < 1e-12
    else:  # an integer-raster delay of a staircase on the raster: the same staircase, shifted
        assert rel(played.grad, seq.grad) > 1e-3


def trapezoid_seq(amp=8e5, tail=3e-3):
    """x trapezoid, then a long gradient-free tail, cut on the raster."""
    b = SequenceBuilder()
    b.block(trapezoid("x", amplitude=amp, flat_time=1e-3, dtype=torch.float64))
    b.delay(tail)
    seq = b.build()
    # cut the tail on the raster too, so the response is sampled finely there
    reps = torch.clamp(torch.round(seq.dt / RASTER), min=1).long()
    rep = lambda t: t.repeat_interleave(reps, dim=0)
    return Sequence(rep(seq.dt / reps), rep(seq.rf), rep(seq.grad), rep(seq.adc_phase),
                    [False] * int(reps.sum()), [False] * int(reps.sum()), [False] * int(reps.sum()))


def ramp_response(t, t1, t2, slope, tau):
    """int_{t1}^{min(t, t2)} slope exp(-(t - t') / tau) dt' (zero before t1)."""
    te = np.minimum(t, t2)
    return np.where(t > t1, slope * tau * (np.exp(-(t - te) / tau) - np.exp(-(t - t1) / tau)), 0.0)


def test_exponential_eddy_on_trapezoid():
    a, tau, amp = 0.03, 0.4e-3, 8e5
    seq = trapezoid_seq(amp, tail=12e-3)
    lobe = trapezoid("x", amplitude=amp, flat_time=1e-3, dtype=torch.float64)
    rise, flat = lobe.rise, lobe.flat
    model = EddyCurrents([0], [0], torch.tensor([a], dtype=torch.float64), torch.tensor([tau], dtype=torch.float64))
    played = apply_girf(seq, model)
    eddy = (played.grad[:, 0] - seq.grad[:, 0]).numpy()
    tb, _ = moments(seq)
    tb = tb.numpy()
    # staircase input: steps h_j at t_j, response -a h_j exp(-(t - t_j) / tau), averaged per segment
    g = np.concatenate([[0.0], seq.grad[:, 0].numpy()])
    steps = np.diff(g)
    ts = tb[:-1]
    t0, t1, dt = tb[:-1, None], tb[1:, None], seq.dt.numpy()[:, None]
    on = ts[None, :] < t1 - 1e-15
    avg = -a * steps[None, :] * tau / dt * (np.exp(-np.clip(t0 - ts[None, :], 0, None) / tau)
                                            - np.exp(-(t1 - ts[None, :]) / tau))
    stair = np.where(on, avg, 0.0).sum(1)
    assert np.abs(eddy - stair).max() < 1e-9 * amp
    # the true trapezoid (linear ramps), at segment midpoints: O((raster / tau)^2) apart
    mid = 0.5 * (tb[:-1] + tb[1:])
    true = -a * (ramp_response(mid, 0.0, rise, amp / rise, tau) + ramp_response(mid, rise + flat, 2 * rise + flat,
                                                                                 -amp / rise, tau))
    assert np.abs(eddy - true).max() < 2e-3 * a * amp
    assert np.abs(eddy).max() > 0.5 * a * amp
    # moment lost by the end of the plateau: int_0^T e dt = -a (amp / rise) tau
    # (rise - tau (exp(-(T - rise) / tau) - exp(-T / tau))) for the ramp-up alone ...
    _, kp = moments(played)
    _, k = moments(seq)
    i = int(np.argmin(np.abs(tb - (rise + flat))))
    T = tb[i]
    up = amp / rise * tau * (rise - tau * (math.exp(-(T - rise) / tau) - math.exp(-T / tau)))
    assert abs(float(kp[i, 0] - k[i, 0]) - (-a * up)) < 2e-3 * a * amp * tau
    # ... and none once the eddy current has decayed (the DC gain is 1)
    assert abs(float(kp[-1, 0] - k[-1, 0])) < 1e-9 * float(k[-1, 0])


def test_convolution_and_transfer_function_agree():
    # random taps (with cross terms and B0 rows) on a random raster waveform: the time-domain
    # GIRF equals an FFT (transfer function) convolution of the raster samples
    gen = torch.Generator().manual_seed(1)
    n_taps, S = 40, 300
    taps = torch.randn(4, 3, n_taps, dtype=torch.float64, generator=gen) * torch.exp(-torch.arange(n_taps) / 8.0)
    g = torch.randn(S, 3, dtype=torch.float64, generator=gen) * 1e5
    seq = Sequence(torch.full((S,), RASTER, dtype=torch.float64), torch.zeros(S, 2, dtype=torch.float64), g,
                   torch.zeros(S, dtype=torch.float64), [False] * S, [False] * S, [False] * S)
    played = apply_girf(seq, GIRF(taps, RASTER))
    n = 1024
    H = torch.fft.rfft(taps, n=n)  # (4, 3, F)
    G = torch.fft.rfft(g.T, n=n)  # (3, F)
    y = torch.fft.irfft((H * G[None]).sum(1), n=n)[:, :S].T  # (S, 4)
    out = torch.cat([played.grad, played.b0_offset[:, None]], 1)
    assert rel(out, y) < 1e-12
    from_tf = apply_girf(seq, GIRF.from_transfer_function(H, RASTER, n=n))
    assert rel(torch.cat([from_tf.grad, from_tf.b0_offset[:, None]], 1), y) < 1e-12


@pytest.mark.parametrize("sim", SIMS)
def test_b0_eddy_gives_analytic_uniform_phase(sim):
    c, tau, amp = 2e-4, 0.8e-3, 8e5  # Hz per Hz/m: ~100 Hz right after the ramp
    b = SequenceBuilder()
    b.rf(1e-6, flip_deg=90.0)
    b.block(trapezoid("x", amplitude=amp, flat_time=1e-3, dtype=torch.float64))
    b.block(trapezoid("x", amplitude=-amp, flat_time=1e-3, dtype=torch.float64))
    b.readout(40, 50e-6)
    seq = b.build()
    model = EddyCurrents([3], [0], torch.tensor([c], dtype=torch.float64), torch.tensor([tau], dtype=torch.float64))
    played = apply_girf(seq, model)
    assert torch.equal(played.grad, seq.grad)
    pos = (torch.rand(6, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(2)) - 0.5) * 0.2
    one = torch.ones(6, dtype=torch.float64)
    ph = Phantom(pos, one, 1e9 * one, 1e9 * one, 0 * one)
    coils = torch.eye(6, dtype=torch.complex128)
    ratio = SIMS[sim](ph, played, coils).signal / SIMS[sim](ph, seq, coils).signal  # (6, n_adc)
    # f_B0 = -c (dg/dt * exp(-t/tau)): steps h_j of the staircase at t_j; phase -2 pi int f_B0
    tb, _ = moments(seq)
    g = torch.cat([torch.zeros(1, dtype=torch.float64), seq.grad[:, 0]])
    steps, tj = torch.diff(g), tb[:-1]
    t_adc = tb[1:][torch.tensor(seq.adc)]
    lag = (t_adc[:, None] - tj[None, :]).clamp(min=0)
    integral = (-c * steps[None, :] * tau * (1 - torch.exp(-lag / tau))).sum(1)  # int_0^t f_B0
    expect = torch.exp(-2j * math.pi * integral)
    assert (ratio - expect[None, :]).abs().max() < 1e-10
    assert (expect - 1).abs().max() > 0.1


def eddy_loss(sim, amp, tau, b0amp, b0tau, delay, **kw):
    ph = random_phantom(30, seed=5)
    seq = bp.gre_2d(8, fov=0.2, ramps=True, n_dummy=1)
    model = [GIRF.identity(delay), EddyCurrents.per_axis(amp, tau, b0amp, b0tau)]
    r = SIMS[sim](ph, apply_girf(seq, model), **kw)
    w = torch.randn(r.signal.shape, dtype=torch.complex128, generator=torch.Generator().manual_seed(9))
    return (r.signal * w).real.sum() + (r.m**2).sum()


def test_eddy_parameter_gradients():
    params = [torch.tensor([[0.02, 0.01], [0.03, 0.0], [0.015, 0.02]], dtype=torch.float64),
              torch.tensor([[0.3e-3, 2e-3], [0.5e-3, 1e-3], [0.2e-3, 4e-3]], dtype=torch.float64),
              torch.tensor([[1e-5], [-2e-5], [3e-5]], dtype=torch.float64),
              torch.tensor([[1e-3], [0.5e-3], [2e-3]], dtype=torch.float64),
              torch.tensor([2e-6, -3e-6, 4e-6], dtype=torch.float64)]
    grads = {}
    for sim in SIMS:
        leaves = [p.clone().requires_grad_() for p in params]
        kw = {"checkpoint_every": 9} if sim == "fused" else {}
        grads[sim] = torch.autograd.grad(eddy_loss(sim, *leaves, **kw), leaves)
    for a, b in zip(grads["core"], grads["fused"]):
        assert rel(b, a) < 1e-10
    for i, idx in [(0, (0, 0)), (0, (1, 1)), (1, (2, 1)), (2, (1, 0)), (3, (0, 0)), (4, (2,))]:
        h = 1e-6 * float(params[i].abs().max())
        plus = [p.clone() for p in params]
        minus = [p.clone() for p in params]
        plus[i][idx] += h
        minus[i][idx] -= h
        with torch.no_grad():
            fd = (eddy_loss("fused", *plus) - eddy_loss("fused", *minus)) / (2 * h)
        got = float(grads["fused"][i][idx])
        assert abs(got - float(fd)) < 1e-5 * max(abs(float(fd)), 1e-3 * float(grads["fused"][i].abs().max())), (i, idx)


def test_chunked_carries_b0_offset():
    from blochparty.chunked import simulate_chunked

    ph = random_phantom(30, seed=6)
    seq = bp.gre_2d(8, fov=0.2, ramps=True, n_dummy=1)
    off = 20.0 * torch.randn(len(seq), dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    grads = []
    for sim in (simulate, lambda *a: simulate_chunked(*a, spin_chunk=7)):
        o = off.clone().requires_grad_()
        r = sim(ph, replace(seq, b0_offset=o))
        grads.append((r.signal, torch.autograd.grad(r.signal.abs().sum(), [o])[0]))
    assert rel(grads[1][0], grads[0][0]) < 1e-12 and rel(grads[1][1], grads[0][1]) < 1e-10
