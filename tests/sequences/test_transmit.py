"""The RF amplifier model (``transmit.py``): linear at low drive, saturating at a_sat (Rapp),
AM/PM phase growing with drive, droop with delivered energy; and its effect on excitation."""

import math

import torch

import blochparty as bp
from blochparty.transmit import amplifier, pulses, rapp

f64 = torch.float64


def test_rapp_limits():
    amp = torch.tensor([1e-3, 1.0, 1e3, 1e6], dtype=f64)
    out = rapp(amp, gain=2.0, a_sat=500.0, p=3.0)
    assert torch.isclose(out[0], torch.tensor(2e-3, dtype=f64), rtol=1e-9)  # linear at low drive
    assert 499.0 < float(out[-1]) <= 500.0 * (1 + 1e-12)  # saturates at a_sat
    assert bool((out[1:] > out[:-1]).all())  # monotonic


def test_ideal_amplifier_is_the_identity():
    seq = bp.gre_2d(8, n_dummy=1)
    assert torch.equal(amplifier(seq).rf, seq.rf)


def test_saturation_lowers_the_flip_and_ampm_rotates_it():
    # a 90 deg hard pulse of amplitude 250 Hz: Rapp output and phase predict the excitation exactly
    t = 1e-3
    seq = bp.SequenceBuilder().rf(t, flip_deg=90.0, n_steps=10).build()
    one = bp.Phantom(pos=torch.zeros(1, 3, dtype=f64), pd=torch.ones(1, dtype=f64), t1=torch.full((1,), 1e9, dtype=f64),
                     t2=torch.full((1,), 1e9, dtype=f64), db0=torch.zeros(1, dtype=f64))
    a_sat, p, phi = 300.0, 2.0, 0.4
    m = bp.simulate(one, amplifier(seq, a_sat=a_sat, p=p, phi_max=phi)).m[0]
    amp = rapp(torch.tensor(250.0, dtype=f64), 1.0, a_sat, p)
    flip = 2 * math.pi * float(amp) * t
    assert math.isclose(float(m[2]), math.cos(flip), abs_tol=1e-12)
    phase = phi * (250.0 / a_sat) ** 2 / (1 + (250.0 / a_sat) ** 2)
    # a pulse along phase psi (rf_x + i rf_y = |B1| e^{i psi}) tips +z to Mxy = i e^{i psi} sin(flip):
    # +x -> +y, +y -> -x
    mxy = complex(float(m[0]), float(m[1]))
    want = 1j * complex(math.cos(phase), math.sin(phase)) * math.sin(flip)
    assert abs(mxy - want) < 1e-12


def test_droop_reduces_late_samples():
    seq = bp.SequenceBuilder().rf(2e-3, amplitude=200.0, n_steps=20).build()
    droop = 1e-3  # [1 / (Hz^2 s)]: 4 Hz^2 s delivered per segment
    amp = amplifier(seq, droop=droop).rf.norm(dim=-1)
    assert bool((amp[1:] < amp[:-1]).all())
    k = torch.arange(20, dtype=f64)
    assert torch.allclose(amp, 200.0 * torch.exp(-droop * (k + 0.5) * 200.0**2 * 1e-4), rtol=1e-12)
    assert pulses(seq) == [(0, 20)]
