"""Slice-selective windowed-sinc excitation on a line of spins along z: the small-tip
Fourier transform of the RF envelope, a direct ODE integration at 90 deg, flat phase
after the rephasing lobe (ramps included), and the slice position / frequency-offset sign."""

import math

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

import blochparty as bp
from blochparty.sequence import SequenceBuilder, System, sinc_pulse, trapezoid

THK, DUR = 5e-3, 2e-3
f64 = torch.float64


def spins(z):
    pos = torch.zeros(len(z), 3, dtype=f64)
    pos[:, 2] = z
    full = lambda v: torch.full((len(z),), v, dtype=f64)
    return bp.Phantom(pos=pos, pd=full(1.0), t1=full(1e9), t2=full(1e9), db0=full(0.0))


def excite(flip, z, z0=0.0, system=System(), rephase=None):
    """Mxy, Mz along z after the slice-select block and its rephasing lobe."""
    rf, gz, reph = sinc_pulse(flip, DUR, slice_thickness=THK, slice_offset=z0, system=system)
    b = SequenceBuilder(system=system).block(gz, rf=rf)
    b.block(trapezoid(area=reph if rephase is None else rephase, system=system))
    m = bp.simulate(spins(z), b.build()).m
    return torch.complex(m[:, 0], m[:, 1]).numpy(), m[:, 2].numpy(), rf, gz, reph


def small_tip(z, rf, gz):
    """Mxy = i 2 pi int b1(t) exp(i 2 pi G z (t - t_c)) dt over the pulse: the Fourier
    transform of the RF envelope, integrated exactly over each constant raster step."""
    g, dt = float(gz.amplitude[2]), rf.raster
    b1 = torch.complex(rf.signal[:, 0], rf.signal[:, 1]).numpy()
    tau = (np.arange(len(b1)) + 0.5) * dt - rf.center
    z = np.asarray(z)[:, None]
    return 2j * np.pi * (b1 * dt * np.exp(2j * np.pi * g * z * tau)).sum(1) * np.sinc(g * z[:, 0] * dt)


def test_small_tip_profile_is_fourier_transform_of_envelope():
    z = torch.linspace(-3 * THK, 3 * THK, 601, dtype=f64)
    err = {}
    for flip in (2.5, 5.0, 10.0):
        mxy, _, rf, gz, _ = excite(flip, z)
        ft = small_tip(z, rf, gz)
        err[flip] = np.abs(mxy - ft).max() / np.abs(ft).max()
    # the only difference is the small-tip linearisation: relative error alpha^2 / 6
    for flip, e in err.items():
        assert abs(e / (math.radians(flip) ** 2 / 6) - 1) < 0.02, err
    assert err[5.0] < 1.3e-3


def test_flat_phase_after_rephasing_lobe_with_ramps():
    z = torch.linspace(-0.6 * THK, 0.6 * THK, 121, dtype=f64)  # the passband
    mxy, _, rf, gz, reph = excite(5.0, z)
    phase = np.angle(mxy / 1j)  # a real pulse along x: Mxy along +y
    assert np.abs(phase).max() < 1e-3
    assert abs(abs(mxy[60]) - math.sin(math.radians(5.0))) < 1e-6  # exact flip at the slice centre
    # forgetting the fall ramp in the rephaser leaves a linear phase across the slice
    no_ramp = -gz.amplitude * 0.5 * gz.flat
    mxy, *_ = excite(5.0, z, rephase=no_ramp)
    slope = np.polyfit(z.numpy(), np.unwrap(np.angle(mxy / 1j)), 1)[0]
    expected = -2 * math.pi * float(gz.amplitude[2]) * 0.5 * gz.fall  # phase -2 pi z * missing area
    assert abs(slope / expected - 1) < 1e-2  # (to the small-tip nonlinearity at 5 deg)


def ode_profile(z, rf, gz, reph):
    """Direct integration of dM/dt = 2 pi M x (b1x, b1y, G z) through every RF raster step
    (all spins at once), then the moment left after the pulse (rest of the flat top, fall
    ramp, rephasing lobe) from the trapezoid areas, as an exact precession."""
    g, zz = float(gz.amplitude[2]), np.asarray(z)
    n = len(zz)
    m = np.zeros((n, 3))
    m[:, 2] = 1.0
    for bx, by in rf.signal.numpy():
        w = 2 * np.pi * np.stack([np.full(n, bx), np.full(n, by), g * zz], -1)
        f = lambda _, y: np.cross(y.reshape(n, 3), w).ravel()
        m = solve_ivp(f, (0, rf.raster), m.ravel(), method="DOP853", rtol=1e-11, atol=1e-13).y[:, -1].reshape(n, 3)
    flat_left = gz.delay + gz.rise + gz.flat - (rf.delay + rf.duration)
    rest = g * (flat_left + 0.5 * gz.fall) + float(reph[2])
    return (m[:, 0] + 1j * m[:, 1]) * np.exp(-2j * np.pi * zz * rest), m[:, 2]


@pytest.mark.parametrize("z0", [0.0, 3e-3])
def test_90_degree_profile_matches_ode(z0):
    sysm = System(rf_raster=10e-6)  # 200 RF steps keep the ODE quick
    z = torch.linspace(z0 - 2.5 * THK, z0 + 2.5 * THK, 101, dtype=f64)
    mxy, mz, rf, gz, reph = excite(90.0, z, z0, system=sysm)
    mxy_ode, mz_ode = ode_profile(z, rf, gz, reph)
    assert np.abs(mxy - mxy_ode).max() < 1e-9 and np.abs(mz - mz_ode).max() < 1e-9
    # a real 90 deg profile: full excitation in the slice, none outside
    # (off-centre slices: to the stepping of the frequency-offset phase once per 10 us RF step)
    assert abs(mxy[50] - 1j) < (1e-9 if z0 == 0 else 1e-3)
    assert np.abs(mxy[np.abs(z.numpy() - z0) > 1.5 * THK]).max() < 0.02
    # beyond the small-tip regime the half-area rephaser leaves a residual (odd, nearly
    # linear) phase, which the ODE shows too: about 0.2 rad within the slice for this pulse
    inside = np.abs(z.numpy() - z0) < 0.3 * THK
    assert 0.1 < np.abs(np.angle(mxy[inside] / 1j)).max() < 0.3


def test_slice_offset_moves_the_slice_to_plus_z0():
    # the frequency offset f = G z0 is applied as exp(-i 2 pi f (t - t_c)): the slice moves to
    # +z0 with the centred slice's profile and phase (to the stepping of the phase ramp once
    # per 1 us RF step); the opposite sign would excite -z0
    z = torch.linspace(-0.02, 0.02, 801, dtype=f64)
    dz = float(z[1] - z[0])
    centred, *_ = excite(90.0, z)
    for z0 in (4e-3, -7e-3):
        shifted, *_ = excite(90.0, z, z0)
        s = round(z0 / dz)
        assert np.abs(shifted[200 + s:601 + s] - centred[200:601]).max() < 5e-4
        a = np.abs(shifted)
        centre = (z.numpy() * a).sum() / a.sum()
        assert abs(centre - z0) < 0.01 * THK
    # with the phase ramp's sign flipped the slice lands at -z0
    rf, gz, reph = sinc_pulse(90.0, DUR, slice_thickness=THK, slice_offset=4e-3)
    rf.signal[:, 1] = -rf.signal[:, 1]
    b = SequenceBuilder().block(gz, rf=rf)
    b.block(trapezoid(area=reph))
    m = bp.simulate(spins(z), b.build()).m
    a = torch.complex(m[:, 0], m[:, 1]).abs().numpy()
    assert abs((z.numpy() * a).sum() / a.sum() + 4e-3) < 0.01 * THK


def test_selective_gre_excites_only_the_slice():
    # the builder's slice-selective GRE: spins outside the slice give no signal
    seq = bp.gre_2d(8, slice_thickness=THK, slice_offset=2e-3, ramps=True, spoiling="ideal")
    signal = lambda z: bp.simulate(spins(torch.tensor([z], dtype=f64)), seq).signal.abs().max()
    assert signal(2e-3) > 0.2 and signal(2e-3 + 3 * THK) < 1e-2 * signal(2e-3)
    assert signal(-2e-3) < 0.5 * signal(2e-3)  # not mirrored to -z0
