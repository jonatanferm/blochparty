"""Contrast of the realistic builders against independent models: RF- and gradient-spoiled
GRE on isochromats spread across a voxel vs EPG (with convergence in the number of
isochromats K), TSE echo trains vs CPMG EPG, bSSFP vs EPG and the closed-form off-resonance
profile, and the EPI off-resonance shift along the phase-encode direction."""

import math

import pytest
import torch

import blochparty as bp
from blochparty.builders import bssfp_2d, cartesian_recon, epi_2d, tse_2d
from blochparty.epg import epg_bssfp, epg_cpmg, epg_init, epg_relax, epg_rf, epg_spoiled_gre

f64 = torch.float64


def line_of_spins(k, length, t1, t2, axis, db0=0.0):
    """``k`` isochromats evenly spread over ``length`` along ``axis`` (0, 1, 2), centred at 0."""
    x = (torch.arange(k, dtype=f64) + 0.5) * (length / k) - length / 2
    pos = torch.zeros(k, 3, dtype=f64)
    pos[:, axis] = x
    full = lambda v: torch.full((k,), float(v), dtype=f64)
    return bp.Phantom(pos=pos, pd=full(1.0), t1=full(t1), t2=full(t2), db0=full(db0))


def echoes(phantom, seq, n):
    """The k = 0 sample of every readout, averaged over the spins."""
    return bp.simulate(phantom, seq).signal[0].reshape(-1, n)[:, n // 2] / len(phantom)


# --- spoiled GRE vs EPG -----------------------------------------------------------------


def test_rf_and_gradient_spoiled_gre_matches_epg():
    # 117 deg RF spoiling plus a spoiler giving 4 cycles across the voxel per TR (ramps,
    # phase-encode rewinders): K isochromats across one voxel along x see EPG order k
    # aliased onto k mod (K / 4), so they match EPG exactly once K / 4 exceeds the number of
    # TRs (60 here), and converge before that.
    n, fov, n_dummy = 8, 0.2, 52
    t1, t2, tr, flip = 0.5, 0.05, 10e-3, 15.0
    seq = bp.gre_2d(n, fov, flip_deg=flip, tr=tr, spoiling="rf", ramps=True, n_dummy=n_dummy)
    epg = epg_spoiled_gre(t1, t2, tr, seq.meta["te"], flip, n_dummy + n, 117.0)[n_dummy:]
    err = {
        k: (echoes(line_of_spins(k, fov / n, t1, t2, 0), seq, n) - epg).abs().max().item()
        for k in (32, 64, 128, 256, 512)
    }
    assert err[32] > err[64] > err[128] > 1e-9
    assert err[256] < 1e-14 and err[512] < 1e-14


def test_gradient_spoiled_gre_matches_epg():
    # constant RF phase: EPG without RF spoiling; 1 cycle per voxel per TR
    n, fov, n_dummy = 8, 0.2, 52
    t1, t2, tr, flip = 0.5, 0.05, 10e-3, 15.0
    seq = bp.gre_2d(n, fov, flip_deg=flip, tr=tr, spoiling="gradient", spoiler_cycles=1.0, n_dummy=n_dummy)
    epg = epg_spoiled_gre(t1, t2, tr, seq.meta["te"], flip, n_dummy + n, None)[n_dummy:]
    err = {
        k: (echoes(line_of_spins(k, fov / n, t1, t2, 0), seq, n) - epg).abs().max().item() for k in (16, 32, 64, 128)
    }
    assert err[16] > err[32] > 1e-7 and err[64] < 1e-14 and err[128] < 1e-14


# --- TSE vs CPMG EPG ---------------------------------------------------------------------


def test_tse_echo_train_matches_cpmg_epg():
    # crushers of one cycle per voxel on z around every refocusing pulse: K isochromats
    # along z over one crusher cycle are EPG with order k aliased onto k mod K; the last of 8
    # echoes reaches order 16, so K = 17 is exact
    n, fov, t1, t2 = 8, 0.2, 1.0, 0.1
    for flips in ([180.0] * 8, [120.0] * 8, [150.0, 90.0, 110.0, 130.0, 60.0, 170.0, 100.0, 140.0]):
        seq = tse_2d(n, fov, etl=8, refocus_flip_deg=flips, crusher_cycles=1.0)
        epg = epg_cpmg(t1, t2, seq.meta["esp"], flips)
        err = {
            k: (echoes(line_of_spins(k, fov / n, t1, t2, 2), seq, n) - epg).abs().max().item()
            for k in (4, 8, 16, 17, 32)
        }
        if flips[0] == 180.0:  # perfect refocusing: nothing to alias, any K
            assert max(err.values()) < 1e-14
            assert torch.allclose(epg.abs(), torch.exp(-torch.arange(1, 9, dtype=f64) * seq.meta["esp"] / t2))
        else:
            assert err[4] > err[16] > 1e-8, err
            assert err[17] < 1e-14 and err[32] < 1e-14, err
    # stimulated echoes shape the 120 deg train: echo 2 > echo 1
    assert abs(epg_cpmg(t1, t2, 5e-3, [120.0] * 8)[1]) > 1.1 * abs(epg_cpmg(t1, t2, 5e-3, [120.0] * 8)[0])


def test_selective_tse_refocuses_the_slice_centre():
    # sinc excitation and refocusing: at the slice centre the flips are exact, so the
    # echoes follow T2 (relaxation during the 2 ms pulses differs from instantaneous EPG pulses
    # only at second order)
    n, t1, t2 = 8, 1.0, 0.1
    seq = tse_2d(n, etl=8, slice_thickness=5e-3)
    e = echoes(line_of_spins(1, 0.0, t1, t2, 2), seq, n)
    expected = 1j * torch.exp(-torch.tensor(seq.meta["echo_times"], dtype=f64) / t2)
    assert (e - expected).abs().max() < 2e-3


# --- bSSFP -----------------------------------------------------------------------------------


def test_bssfp_matches_epg_with_off_resonance():
    # a single isochromat (no dephasing in a balanced sequence), very short pulses: the EPG
    # bSSFP model applies directly; the off-resonance sign is pinned by the EPG convention
    t1, t2, tr, flip, n, n_dummy = 0.4, 0.1, 5e-3, 40.0, 8, 100
    seq = bssfp_2d(n, 0.2, flip_deg=flip, tr=tr, rf_duration=1e-9, prep=False, n_dummy=n_dummy)
    for df in (0.0, 13.0, -77.0, 160.0):
        s = echoes(line_of_spins(1, 0.0, t1, t2, 0, db0=df), seq, n)
        assert (s - epg_bssfp(t1, t2, tr, flip, n_dummy + n, df)[n_dummy:]).abs().max() < 1e-7


def test_bssfp_alpha_half_preparation_matches_epg():
    # the alpha/2 pulse TR/2 before the train, with the phase of "pulse -1": EPG built from its
    # primitives; on resonance it removes most of the transient oscillation
    t1, t2, tr, flip, n = 0.4, 0.1, 5e-3, 60.0, 16
    seq = bssfp_2d(n, 0.2, flip_deg=flip, tr=tr, rf_duration=1e-9, prep=True)
    a = math.radians(flip)
    state = epg_relax(epg_rf(epg_init(1), a / 2, math.pi), tr / 2, t1, t2)
    ref = []
    for j in range(n):
        phi = math.pi * (j % 2)
        state = epg_relax(epg_rf(state, a, phi), tr / 2, t1, t2)
        ref.append(state[0, 0] * complex(math.cos(phi), -math.sin(phi)))
        state = epg_relax(state, tr / 2, t1, t2)
    s = echoes(line_of_spins(1, 0.0, t1, t2, 0), seq, n)
    assert (s - torch.stack(ref)).abs().max() < 1e-7
    no_prep = echoes(
        line_of_spins(1, 0.0, t1, t2, 0), bssfp_2d(n, 0.2, flip_deg=flip, tr=tr, rf_duration=1e-9, prep=False), n
    )
    jitter = lambda x: x.abs().diff().abs().max()
    assert jitter(s) < 0.05 * jitter(no_prep)


@pytest.mark.slow
def test_bssfp_off_resonance_profile_closed_form():
    # steady state |M| at TE = TR/2 vs off-resonance (Xiang & Hoff, MRM 2014), 180 deg cycling
    t1, t2, flip = 0.4, 0.1, 40.0
    tr = 5e-3
    seq = bssfp_2d(4, 0.2, flip_deg=flip, tr=tr, n_dummy=2500, rf_duration=1e-9)
    e1, e2, a = math.exp(-tr / t1), math.exp(-tr / t2), math.radians(flip)
    c, d = e2 * (e1 - 1) * (1 + math.cos(a)), (1 - e1 * math.cos(a)) - (e1 - math.cos(a)) * e2**2
    dfs = torch.tensor([0.0, 31.0, 0.25 / tr, 0.45 / tr, 0.5 / tr, -0.3 / tr, 1.0 / tr], dtype=f64)
    pos = torch.zeros(len(dfs), 3, dtype=f64)
    one = torch.ones(len(dfs), dtype=f64)
    ph = bp.Phantom(pos=pos, pd=one, t1=t1 * one, t2=t2 * one, db0=dfs)
    # one "coil" per spin separates the spins; the k = 0 sample of the last line
    sig = bp.simulate(ph, seq, coils=torch.eye(len(dfs), dtype=torch.complex128)).signal[:, -2]
    for i, df in enumerate(dfs.tolist()):
        theta = 2 * math.pi * df * tr + math.pi
        mag = (1 - e1) * math.sin(a) * abs(1 - e2 * complex(math.cos(theta), -math.sin(theta)))
        mag *= math.exp(-tr / (2 * t2)) / abs(c * math.cos(theta) + d)
        assert abs(abs(sig[i]) - mag) < 1e-7, (df, abs(sig[i]), mag)
    # banding: signal null at the band (theta = 0 mod 2 pi, i.e. df = 1 / (2 TR))
    assert abs(sig[4]) < 0.1 * abs(sig[0])


# --- EPI -------------------------------------------------------------------------------------


def test_epi_off_resonance_shifts_along_phase_encode():
    # a uniform offset of exactly 2 phase-encode pixels' bandwidth shifts the image by +2 rows
    # (y); along x it moves only f / (readout bandwidth per pixel) = 0.02 px, opposite on
    # alternate lines (the source of the small residual)
    n, fov = 32, 0.22
    maps = bp.shepp_logan_maps(n)
    seq = epi_2d(n, fov)
    df = 2 * seq.meta["bandwidth_per_pixel_pe"]
    assert df / bp.readout_bandwidth_per_pixel(seq) < 0.02
    phantom = lambda off: bp.phantom_from_maps(
        maps["pd"], maps["t1"], maps["t2"], fov, db0=torch.full_like(maps["pd"], off)
    )
    recon = lambda p: cartesian_recon(bp.simulate(p, seq).signal, seq)[0].abs()
    img, ref = recon(phantom(df)), recon(phantom(0.0))
    err = lambda shift: ((img - ref.roll(shift, dims=-2)).norm() / ref.norm()).item()
    assert err(2) < 0.03
    assert min(err(1), err(3), err(-2), err(0)) > 10 * err(2)


# --- gradients through the new builders ------------------------------------------------------


@pytest.mark.slow
def test_autograd_through_realistic_builders():
    # flip angles (excitation, a variable refocusing train, slice-selective sinc) reach the
    # signal through the block assembly; checked against finite differences
    ph = line_of_spins(3, 2e-3, 0.8, 0.06, 2)
    w = torch.randn(16, generator=torch.Generator().manual_seed(0), dtype=torch.complex128)
    project = lambda seq: (bp.simulate(ph, seq).signal[0] * w).sum().real  # one backward per check

    def tse(flips):
        return project(tse_2d(4, 0.2, etl=4, refocus_flip_deg=list(flips.unbind()), tr=0.05))

    def gre(flip):
        return project(
            bp.gre_2d(4, 0.2, flip_deg=flip, spoiling="rf", ramps=True, slice_thickness=5e-3, rf_duration=0.5e-3)
        )

    flips = torch.tensor([150.0, 120.0, 135.0, 170.0], dtype=f64, requires_grad=True)
    assert torch.autograd.gradcheck(tse, [flips])
    assert torch.autograd.gradcheck(gre, [torch.tensor(25.0, dtype=f64, requires_grad=True)])
