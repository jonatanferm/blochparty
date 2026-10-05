"""Shepp-Logan imaging with the realistic builders. Point spins at voxel centres see no
intra-voxel dephasing, so for each builder an independent signal model predicts the
multi-coil k-space exactly: the explicit signal equation (``reference.signal_equation``)
for GRE with ramps and for EPI, a closed-form Mz recursion for SE / TSE, and per-tissue EPG
for bSSFP. The reconstructions are then compared with the tissue-model images; an
RF-spoiled GRE on sub-voxel isochromats against the EPG field of each tissue."""

import math

import pytest
import torch

import blochparty as bp
from blochparty.builders import bssfp_2d, cartesian_kspace, cartesian_recon, epi_2d, se_2d, tse_2d
from blochparty.epg import epg_grad, epg_init, epg_relax, epg_rf, rf_spoil_phases
from blochparty.phantom import LABELS, TISSUES
from blochparty.reference import signal_equation, spoiled_gre_signal

N, FOV = 32, 0.22
f64 = torch.float64


@pytest.fixture(scope="module")
def setup():
    maps = bp.shepp_logan_maps(N)
    phantom = bp.phantom_from_maps(maps["pd"], maps["t1"], maps["t2"], FOV)
    coils = bp.birdcage_sensitivities(phantom.pos, n_coils=4)
    return maps, phantom, coils


def model_kspace(phantom, coils, mxy0, t, k):
    """sum_r C(r) mxy0(r) exp(-t / T2) exp(-i 2 pi k . r) for every sample: ``mxy0`` (N,) or
    (N, n_lines) transverse magnetization at t = 0, ``t`` (n_lines, n_samples), ``k`` (..., 3)."""
    t, k = t.reshape(-1), k.reshape(-1, 3)
    n_lines = mxy0.shape[1] if mxy0.ndim == 2 else 1
    m = (mxy0 if mxy0.ndim == 2 else mxy0[:, None]).repeat_interleave(len(t) // n_lines, dim=1)
    e = torch.exp(-t[None] / phantom.t2[:, None] - 2j * math.pi * (phantom.pos @ k.T))
    return coils.T @ (m * e)


def grid_k(index=None):
    dk = 1 / FOV
    ky, kx = torch.meshgrid(*(2 * [(torch.arange(N, dtype=f64) - N // 2) * dk]), indexing="ij")
    k = torch.stack([kx, ky, torch.zeros_like(kx)], -1).reshape(-1, 3)
    return k if index is None else k[index]


def tissue_image(maps, values):
    """Image with one (complex) value per tissue label."""
    img = torch.zeros(maps["pd"].shape, dtype=torch.complex128)
    for name, v in values.items():
        img[maps["label"] == LABELS.index(name)] = v
    return img


# --- GRE ----------------------------------------------------------------------------------------


def test_gre_with_ramps_matches_signal_equation(setup):
    maps, phantom, coils = setup
    seq = bp.gre_2d(N, FOV, flip_deg=15.0, tr=10e-3, ramps=True, n_dummy=300)
    s = bp.simulate(phantom, seq, coils).signal
    ref = signal_equation(phantom, seq, coils)
    assert (s - ref).abs().max() < 1e-10 * ref.abs().max()
    img = bp.reconstruct(bp.simulate(phantom, seq).signal, seq)[0]
    mxy = spoiled_gre_signal(maps["pd"], maps["t1"], maps["t2"], 10e-3, seq.meta["te"], math.radians(15.0))
    expected = torch.where(maps["pd"] > 0, 1j * mxy, 0)
    assert (img - expected).abs().max() < 2e-3 * expected.abs().max()  # T2 decay during the 320 us readout


def epg_fields(t1, t2, tr, te, flip, n_pulses, increment_deg=117.0):
    """Full EPG state at TE of every TR of an RF-spoiled GRE (one dephasing unit per TR),
    demodulated by the RF phase: a function giving Mxy at dephasing phases theta (S,) for
    each TR, (S, n_pulses). Mxy(theta) = sum_k F+_k exp(-ik theta) + conj(F-_k) exp(ik theta)."""
    a = math.radians(flip)
    phases = rf_spoil_phases(n_pulses, increment_deg)
    state, out = epg_init(n_pulses + 1), []
    for phi in phases.tolist():
        state = epg_relax(epg_rf(state, a, phi), te, t1, t2)
        out.append(state)
        state = epg_grad(epg_relax(state, tr - te, t1, t2))
    states = torch.stack(out)  # (n_pulses, 3, n_states)
    k = torch.arange(n_pulses + 1, dtype=f64)
    demod = torch.exp(-1j * phases)

    def mxy(theta):
        e = torch.exp(-1j * k[None] * theta[:, None])  # (S, n_states)
        return (e @ states[:, 0].T + e[:, 1:].conj() @ states[:, 1, 1:].conj().T) * demod
    return mxy


def test_rf_spoiled_gre_on_subvoxel_spins_matches_epg_field_model(setup):
    # K jittered isochromats per voxel along the spoiler axis x, one spoiler cycle per voxel per
    # TR. EPG describes the continuous transverse field of each tissue exactly; evaluated at
    # every isochromat (theta = 2 pi x / voxel) it predicts each one, for any K, including the
    # higher dephasing states that reach the kx != 0 samples (the voxel-mean F0 alone misses
    # those by ~10 %). Then the image is just the reconstruction of exact k-space.
    maps, phantom, coils = setup
    k, n_dummy, flip, tr = 4, 40, 15.0, 10e-3
    dx = FOV / N
    g = torch.Generator().manual_seed(1)
    rep = lambda x: x.repeat_interleave(k)
    pos = phantom.pos.repeat_interleave(k, dim=0)
    pos[:, 0] += (torch.rand(len(pos), generator=g, dtype=f64) - 0.5) * dx
    sub = bp.Phantom(pos=pos, pd=rep(phantom.pd) / k, t1=rep(phantom.t1), t2=rep(phantom.t2), db0=rep(phantom.db0))
    sub_coils = coils.repeat_interleave(k, dim=0)
    seq = bp.gre_2d(N, FOV, flip_deg=flip, tr=tr, ramps=True, spoiling="rf", spoiler_cycles=1.0, n_dummy=n_dummy)
    te, label = seq.meta["te"], rep(maps["label"][maps["pd"] > 0])
    mxy = torch.zeros(len(sub), N, dtype=torch.complex128)  # at TE of each line
    for name, (pd, t1, t2) in TISSUES.items():
        on = label == LABELS.index(name)
        mxy[on] = pd / k * epg_fields(t1, t2, tr, te, flip, n_dummy + N)(2 * math.pi * pos[on, 0] / dx)[:, n_dummy:]
    t = (torch.arange(N, dtype=f64) - N // 2) * seq.meta["dwell"]  # sample time - TE
    ref = model_kspace(sub, sub_coils, mxy, t.expand(N, N), grid_k())
    sig = bp.simulate(sub, seq, sub_coils).signal
    assert (sig - ref).abs().max() < 1e-10 * ref.abs().max()


# --- SE / TSE ---------------------------------------------------------------------------------


def se_mz(t1, pd, esp, n_refocus, tr):
    """Steady-state Mz before a 90 deg excitation of a CPMG train with exact 180s: Mz is zero
    after the 90, inverted at every refocusing pulse (at esp/2 + j esp), recovers until TR."""
    t1, pd = t1.to(f64), pd.to(f64)
    recover = lambda m, dt: pd + (m - pd) * torch.exp(-dt / t1)
    m, t = torch.zeros_like(t1), 0.0
    for j in range(n_refocus):
        m = -recover(m, esp / 2 + j * esp - t)
        t = esp / 2 + j * esp
    return recover(m, tr - t)


@pytest.mark.parametrize("make", [
    lambda: se_2d(N, FOV, te=15e-3, tr=0.3, n_dummy=1),
    lambda: tse_2d(N, FOV, etl=8, tr=0.6, n_dummy=1),
    lambda: tse_2d(N, FOV, etl=8, tr=0.6, n_dummy=1, ordering="centric", ramps=False),
], ids=["se", "tse_linear", "tse_centric"])
def test_spin_echo_kspace_matches_signal_model(setup, make):
    maps, phantom, coils = setup
    seq = make()
    etl, esp = seq.meta["etl"], seq.meta["esp"]
    mz = se_mz(phantom.t1, phantom.pd, esp, etl, seq.meta["tr"])
    # time of every sample since its shot's excitation (echoes at (e + 1) esp, centred readouts)
    t_echo = torch.tensor(seq.meta["echo_times"], dtype=f64).repeat(N // etl)
    t = t_echo[:, None] + (torch.arange(N, dtype=f64) - N // 2) * seq.meta["dwell"]
    ref = model_kspace(phantom, coils, 1j * mz.to(torch.complex128), t, grid_k(seq.meta["kspace_index"]))
    s = bp.simulate(phantom, seq, coils).signal
    assert (s - ref).abs().max() < 1e-10 * ref.abs().max()
    # the image: the TE-weighted tissue model, blurred by the T2 decay along the echo train
    img = cartesian_recon(bp.simulate(phantom, seq).signal, seq)[0]
    expected = 1j * se_mz(maps["t1"], maps["pd"], esp, etl, seq.meta["tr"]) * torch.exp(-seq.meta["te"] / maps["t2"])
    expected = torch.where(maps["pd"] > 0, expected, 0)
    nrmse = ((img - expected).norm() / expected.norm()).item()
    assert nrmse < (0.01 if etl == 1 else 0.12), nrmse


# --- bSSFP ----------------------------------------------------------------------------------------


def bssfp_epg(t1, t2, tr, flip, n_pulses, prep=True):
    """F0 just after every pulse of a 180 deg phase-cycled train with the alpha/2 prep, demodulated."""
    a = math.radians(flip)
    state = epg_relax(epg_rf(epg_init(1), a / 2, math.pi), tr / 2, t1, t2) if prep else epg_init(1)
    out = []
    for j in range(n_pulses):
        phi = math.pi * (j % 2)
        state = epg_rf(state, a, phi)
        out.append(state[0, 0] * complex(math.cos(phi), -math.sin(phi)))
        state = epg_relax(state, tr, t1, t2)
    return torch.stack(out)


def test_bssfp_kspace_matches_epg_per_voxel(setup):
    # balanced gradients return every point spin to its phase each TR: per voxel, a single
    # EPG isochromat; the readout adds exp(-t/T2) and the k-space encoding
    maps, phantom, coils = setup
    n_dummy, flip = 100, 50.0
    seq = bssfp_2d(N, FOV, flip_deg=flip, n_dummy=n_dummy)
    tr = seq.meta["tr"]
    label = maps["label"][maps["pd"] > 0]
    f0 = torch.zeros(len(phantom), N, dtype=torch.complex128)
    for name, (pd, t1, t2) in TISSUES.items():
        f0[label == LABELS.index(name)] = pd * bssfp_epg(t1, t2, tr, flip, n_dummy + N)[n_dummy:]
    t = tr / 2 + (torch.arange(N, dtype=f64) - N // 2) * seq.meta["dwell"]
    ref = model_kspace(phantom, coils, f0, t.expand(N, N), grid_k())
    s = bp.simulate(phantom, seq, coils).signal
    assert (s - ref).abs().max() < 1e-10 * ref.abs().max()
    # the image: per tissue, the (prep-smoothed) transient at the k-space centre line
    img = bp.reconstruct(bp.simulate(phantom, seq).signal, seq)[0]
    centre = tissue_image(maps, {name: pd * bssfp_epg(t1, t2, tr, flip, n_dummy + N)[n_dummy + N // 2]
                                 * math.exp(-tr / 2 / t2) for name, (pd, t1, t2) in TISSUES.items()})
    assert ((img - centre).norm() / centre.norm()).item() < 0.03


# --- EPI ---------------------------------------------------------------------------------------------


def test_epi_kspace_matches_signal_equation(setup):
    maps, phantom, coils = setup
    seq = epi_2d(N, FOV, flip_deg=90.0)
    s = bp.simulate(phantom, seq, coils).signal
    ref = signal_equation(phantom, seq, coils)
    assert (s - ref).abs().max() < 1e-10 * ref.abs().max()
    img = cartesian_recon(bp.simulate(phantom, seq).signal, seq)[0]
    expected = torch.where(maps["pd"] > 0, 1j * maps["pd"] * torch.exp(-seq.meta["te"] / maps["t2"]), 0)
    # T2 decay along the 15 ms echo train blurs along y
    assert ((img - expected).norm() / expected.norm()).item() < 0.06
    # cartesian_kspace places every sample: the k = 0 sample is the largest
    k = cartesian_kspace(bp.simulate(phantom, seq).signal, seq)[0]
    assert int(k.abs().argmax()) == (N // 2) * N + N // 2
