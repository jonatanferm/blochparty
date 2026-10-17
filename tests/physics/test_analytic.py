"""Single-spin and few-spin experiments with closed-form answers."""

import math

import torch

from blochparty import Phantom, SequenceBuilder, simulate
from blochparty.reference import inversion_recovery_mz, spoiled_gre_signal


def spins(n=1, t1=1.0, t2=0.1, db0=0.0, pd=1.0):
    f = lambda v: torch.full((n,), v, dtype=torch.float64) if not torch.is_tensor(v) else v
    return Phantom(pos=torch.zeros(n, 3, dtype=torch.float64), pd=f(pd), t1=f(t1), t2=f(t2), db0=f(db0))


def test_flip_angle():
    for flip in [10.0, 45.0, 90.0, 150.0]:
        seq = SequenceBuilder().rf(1e-6, flip_deg=flip).build()
        m = simulate(spins(t1=1e9, t2=1e9), seq).m[0]
        assert torch.allclose(m[2], torch.tensor(math.cos(math.radians(flip)), dtype=m.dtype), atol=1e-12)


def test_rf_phase_and_signal_convention():
    # B1 along +x tips +z to +y, i.e. Mxy = +i; RF phase rotates that by exp(i*phase)
    for phase in [0.0, 0.7, -math.pi / 2]:
        seq = SequenceBuilder().rf(1e-6, flip_deg=90.0, phase=phase).readout(1, 1e-9).build()
        s = simulate(spins(t1=1e9, t2=1e9), seq).signal[0, 0]
        assert abs(s - 1j * complex(math.cos(phase), math.sin(phase))) < 1e-9


def test_fid_decay_and_precession():
    t2, df, dwell, n = 0.05, 37.0, 1e-3, 40
    # a 10 ns pulse is effectively instantaneous (a 1 us one at 37 Hz offset is off by ~3e-5)
    seq = SequenceBuilder().rf(10e-9, flip_deg=90.0).readout(n, dwell).build()
    s = simulate(spins(t2=t2, db0=df, t1=1e9), seq).signal[0]
    t = (torch.arange(n, dtype=torch.float64) + 0.5) * dwell + 5e-9
    expected = 1j * torch.exp(-t / t2) * torch.exp(-2j * math.pi * df * t)
    assert torch.allclose(s, expected, atol=1e-6)


def test_spin_echo_refocuses_off_resonance():
    # 200 spins spread over +-500 Hz: the FID dephases, the echo at 2*tau recovers exp(-2*tau/T2)
    n, t2, tau = 200, 0.08, 10e-3
    db0 = torch.linspace(-500, 500, n, dtype=torch.float64)
    # pulse centres at 0.5 us and tau; the echo forms at 2*tau - 0.5 us
    seq = (
        SequenceBuilder()
        .rf(1e-6, flip_deg=90.0)
        .delay(tau - 2e-6)
        .rf(2e-6, flip_deg=180.0, phase=math.pi / 2)
        .delay(tau - 1.5e-6 - 0.5e-9)
        .readout(1, 1e-9)
        .build()
    )
    s = simulate(spins(n, t2=t2, t1=1e9, db0=db0), seq).signal[0, 0] / n
    assert abs(abs(s) - math.exp(-2 * tau / t2)) < 1e-4
    # and without the refocusing pulse there is (almost) nothing left
    fid = simulate(
        spins(n, t2=t2, t1=1e9, db0=db0),
        SequenceBuilder().rf(1e-6, flip_deg=90.0).delay(2 * tau).readout(1, 1e-9).build(),
    )
    assert abs(fid.signal[0, 0]) / n < 0.05


def test_inversion_recovery():
    t1 = 0.8
    for ti in [0.05, 0.3, 0.555, 2.0]:
        seq = SequenceBuilder().rf(1e-6, flip_deg=180.0).delay(ti - 0.5e-6).build()  # TI from pulse centre
        mz = simulate(spins(t1=t1), seq).m[0, 2]
        assert abs(mz - inversion_recovery_mz(1.0, torch.tensor(t1, dtype=torch.float64), ti)) < 1e-6


def test_spoiled_gre_steady_state():
    t1, t2, tr, te, flip = 0.5, 0.05, 10e-3, 3e-3, 20.0
    b = SequenceBuilder()
    for _ in range(400):
        b.rf(1e-6, flip_deg=flip).delay(te - 0.5e-6).readout(1, 1e-9).delay(tr - te - 0.5e-6 - 1e-9).spoil()
    s = simulate(spins(t1=t1, t2=t2), b.build()).signal[0, -1]
    expected = spoiled_gre_signal(
        1.0, torch.tensor(t1, dtype=torch.float64), torch.tensor(t2, dtype=torch.float64), tr, te, math.radians(flip)
    )
    assert abs(abs(s) - expected) < 1e-5
