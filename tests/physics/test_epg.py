"""Extended Phase Graph reference model: conventions, closed forms, and equivalence
with the isochromat simulator for sequences with stimulated echoes and spoiling.

A voxel of length L is represented by K isochromats evenly spaced along x; one
unit of EPG dephasing is a gradient of area 1/L. The discrete average over the
spins sees EPG order k aliased onto k mod K, so once K exceeds the highest
order reached by the time of an echo the two models agree to round-off.
"""

import math

import pytest
import torch

from blochparty import Phantom, SequenceBuilder, simulate
from blochparty.epg import epg_bssfp, epg_cpmg, epg_grad, epg_init, epg_relax, epg_rf, epg_spoiled_gre, rf_spoil_phases
from blochparty.reference import spoiled_gre_signal

f64 = lambda v: torch.tensor(v, dtype=torch.float64)
L = 1e-3  # voxel length [m]
RF_DUR, DWELL = 1e-6, 1e-6  # gradients are off during RF and ADC, so neither costs accuracy


def ernst(t1, t2, tr, te, flip_deg):
    # float64 tensors throughout: spoiled_gre_signal turns Python floats into float32
    return spoiled_gre_signal(1.0, *(f64(v) for v in (t1, t2, tr, te, math.radians(flip_deg))))


def voxel(k, t1, t2, db0=0.0):
    x = torch.arange(k, dtype=torch.float64) * (L / k)
    full = lambda v: torch.full((k,), v, dtype=torch.float64)
    return Phantom(pos=torch.stack([x, 0 * x, 0 * x], dim=-1), pd=full(1.0), t1=full(t1), t2=full(t2),
                   db0=full(db0))


def unit_grad(duration):
    """Gradient that dephases by one EPG order (area 1/L) over ``duration``."""
    return (1.0 / (L * duration), 0.0, 0.0)


def isochromat_signal(k, t1, t2, seq, db0=0.0):
    return simulate(voxel(k, t1, t2, db0), seq).signal[0] / k


def cpmg_sequence(esp, flips):
    # pulse centres at RF_DUR/2 + {0, esp/2, 3 esp/2, ...}; echoes at RF_DUR/2 + j * esp
    b = SequenceBuilder().rf(RF_DUR, flip_deg=90.0)
    b.delay(esp / 2 - RF_DUR, grad=unit_grad(esp / 2 - RF_DUR))
    half = esp / 2 - RF_DUR / 2 - DWELL / 2
    for j, flip in enumerate(flips):
        b.rf(RF_DUR, flip_deg=flip, phase=math.pi / 2)
        b.delay(half, grad=unit_grad(half)).readout(1, DWELL)
        if j < len(flips) - 1:
            b.delay(half, grad=unit_grad(half))
    return b.build()


def spoiled_gre_sequence(tr, te, flip, phases):
    b = SequenceBuilder()
    rest = tr - te - RF_DUR / 2 - DWELL / 2
    for phi in phases.tolist():
        b.rf(RF_DUR, flip_deg=flip, phase=phi)
        b.delay(te - RF_DUR / 2 - DWELL / 2).readout(1, DWELL, phase=phi)
        b.delay(rest, grad=unit_grad(rest))
    return b.build()


# --- conventions ---------------------------------------------------------------------


def test_rf_phase_precession_and_dephasing_conventions():
    for alpha, phase in [(math.pi / 2, 0.0), (0.4, 0.7), (2.0, -math.pi / 2)]:
        s = epg_rf(epg_init(3), alpha, phase)
        assert abs(s[0, 0] - 1j * complex(math.cos(phase), math.sin(phase)) * math.sin(alpha)) < 1e-15
        assert abs(s[1, 0] - s[0, 0].conj()) < 1e-15  # F-_0 = conj(F+_0) = conj(Mxy)
        assert abs(s[2, 0] - math.cos(alpha)) < 1e-15
    # Mxy evolves as exp(-i 2 pi f t)
    f, t = 37.0, 3e-3
    s = epg_relax(epg_rf(epg_init(1), math.pi / 2), t, 1e9, 1e9, off_resonance_hz=f)
    assert abs(s[0, 0] - 1j * complex(math.cos(2 * math.pi * f * t), -math.sin(2 * math.pi * f * t))) < 1e-11
    # a unit gradient moves the signal into F+_1 (and F-_1 stays empty); the reverse gradient undoes it
    s = epg_rf(epg_init(3), 1.0, 0.3)
    d = epg_grad(s)
    assert d[0, 0] == 0 and abs(d[0, 1] - s[0, 0]) < 1e-15 and d[1, 1] == 0
    assert torch.allclose(epg_grad(d, -1), s, atol=1e-15)


# --- closed forms ------------------------------------------------------------------


def test_cpmg_180_gives_pure_t2_decay():
    t1, t2, esp, n = 0.8, 0.07, 8e-3, 12
    echoes = epg_cpmg(t1, t2, esp, [180.0] * n)
    expected = 1j * torch.exp(-torch.arange(1, n + 1, dtype=torch.float64) * esp / t2)
    assert torch.allclose(echoes, expected.to(echoes.dtype), atol=1e-15)


def test_spoiled_gre_ideal_limits_give_ernst():
    t1, t2, tr, te, flip = 0.5, 0.05, 10e-3, 3e-3, 20.0
    # ideal spoiling: transverse states zeroed every TR
    s = epg_spoiled_gre(t1, t2, tr, te, flip, 400, None, ideal_spoil=True)
    assert abs(s[-1] - 1j * ernst(t1, t2, tr, te, flip)) < 1e-14
    # gradient spoiling only, but T2 << TR so nothing survives to the next pulse
    t2, te = 0.2e-3, 0.1e-3
    s = epg_spoiled_gre(t1, t2, tr, te, flip, 400, None)
    assert abs(s[-1] - 1j * ernst(t1, t2, tr, te, flip)) < 1e-14


def test_rf_spoiling_approaches_ernst_but_gradient_spoiling_alone_does_not():
    # the known behaviour: with T2 comparable to TR, unspoiled transverse coherences
    # change the steady state; 117 deg quadratic RF spoiling brings it close to Ernst
    t1, t2, tr, te, flip = 0.5, 0.05, 10e-3, 3e-3, 15.0
    ss = float(ernst(t1, t2, tr, te, flip))
    rf_spoiled = epg_spoiled_gre(t1, t2, tr, te, flip, 300)
    grad_only = epg_spoiled_gre(t1, t2, tr, te, flip, 300, None)
    assert abs(rf_spoiled[-1] - rf_spoiled[-2]) < 1e-8  # steady state in the demodulated frame
    assert abs(abs(rf_spoiled[-1]) / ss - 1) < 0.01
    assert abs(abs(grad_only[-1]) / ss - 1) > 0.1


@pytest.mark.slow
def test_bssfp_steady_state_closed_form():
    t1, t2, tr, flip = 0.4, 0.1, 5e-3, 40.0
    e1, e2, a = math.exp(-tr / t1), math.exp(-tr / t2), math.radians(flip)
    s = epg_bssfp(t1, t2, tr, flip, 3000)
    on_res = math.sin(a) * (1 - e1) / (1 - (e1 - e2) * math.cos(a) - e1 * e2) * math.exp(-tr / (2 * t2))
    assert abs(s[-1] - 1j * on_res) < 1e-12
    # off-resonance banding profile (Xiang & Hoff, MRM 2014), theta = precession per TR, plus pi for
    # the 180 deg phase cycling: |M| = (1 - E1) sin(a) |1 - E2 e^{-i theta}| / |C cos(theta) + D|
    c, d = e2 * (e1 - 1) * (1 + math.cos(a)), (1 - e1 * math.cos(a)) - (e1 - math.cos(a)) * e2**2
    for df in [0.0, 13.0, 50.0, 99.0, 100.0, -77.0]:
        theta = 2 * math.pi * df * tr + math.pi
        mag = (1 - e1) * math.sin(a) * abs(1 - e2 * complex(math.cos(theta), -math.sin(theta)))
        mag *= math.exp(-tr / (2 * t2)) / abs(c * math.cos(theta) + d)
        assert abs(abs(epg_bssfp(t1, t2, tr, flip, 3000, df)[-1]) - mag) < 1e-12


# --- against the isochromat simulator ------------------------------------------------


def test_cpmg_120_matches_isochromats():
    # non-180 refocusing: echo amplitudes are shaped by stimulated echoes (2nd echo > 1st)
    t1, t2, esp, n = 1.0, 0.1, 10e-3, 8
    flips = [120.0] * n
    epg = epg_cpmg(t1, t2, esp, flips)
    assert abs(epg[1]) > abs(epg[0]) * 1.1
    seq = cpmg_sequence(esp, flips)
    err = {k: (isochromat_signal(k, t1, t2, seq) - epg).abs().max().item() for k in (4, 8, 12, 16, 17, 32)}
    # the last echo has dephasing orders up to 2n = 16: K = 16 still aliases it, K = 17 is exact
    assert err[4] > err[8] > err[12] > err[16] > 1e-7
    assert err[17] < 1e-14 and err[32] < 1e-14


def test_rf_spoiled_gre_matches_isochromats():
    t1, t2, tr, te, flip, n = 0.5, 0.05, 10e-3, 3e-3, 15.0, 150
    epg = epg_spoiled_gre(t1, t2, tr, te, flip, n, 117.0)
    seq = spoiled_gre_sequence(tr, te, flip, rf_spoil_phases(n, 117.0))
    err = {k: (isochromat_signal(k, t1, t2, seq) - epg).abs().max().item() for k in (16, 32, 64, 128, 256)}
    assert err[16] > err[32] > err[64] > err[128]
    assert err[16] > 1e-3 and err[128] < 1e-11 and err[256] < 1e-14


def test_bssfp_off_resonance_matches_isochromat():
    # no gradients, so one isochromat per frequency is exact; this pins the sign of the precession.
    # The only difference is off-resonance during the RF pulse, so it is kept very short.
    t1, t2, tr, flip, n, rf_dur = 0.4, 0.1, 5e-3, 40.0, 400, 1e-10
    half = tr / 2 - rf_dur / 2 - DWELL / 2
    b = SequenceBuilder()
    for j in range(n):
        phi = math.radians((180.0 * j) % 360.0)
        b.rf(rf_dur, flip_deg=flip, phase=phi).delay(half).readout(1, DWELL, phase=phi).delay(half)
    seq = b.build()
    for df in [13.0, -77.0, 160.0]:
        iso = isochromat_signal(1, t1, t2, seq, db0=df)
        assert (iso - epg_bssfp(t1, t2, tr, flip, n, df)).abs().max() < 1e-7


# --- gradients ---------------------------------------------------------------------


def test_gradcheck_flips_and_relaxation():
    def cpmg(flips, t2):
        return torch.view_as_real(epg_cpmg(0.9, t2, 10e-3, flips, excite_flip=80.0))

    def gre(flip, t1, t2):
        return torch.view_as_real(epg_spoiled_gre(t1, t2, 10e-3, 3e-3, flip, 6, 117.0))

    flips = torch.tensor([150.0, 110.0, 130.0], dtype=torch.float64, requires_grad=True)
    t2 = f64(0.08).requires_grad_()
    assert torch.autograd.gradcheck(cpmg, [flips, t2])
    assert torch.autograd.gradcheck(gre, [f64(25.0).requires_grad_(), f64(0.6).requires_grad_(), t2])
