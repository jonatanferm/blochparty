"""2D imaging: Bloch k-space vs. the explicit signal equation, and reconstructions
vs. the image the tissue maps predict."""

import math

import pytest
import torch

import mrsim
from mrsim.reference import signal_equation, spoiled_gre_signal

N, FOV, FLIP, TR = 32, 0.22, 15.0, 10e-3


@pytest.fixture(scope="module")
def setup():
    maps = mrsim.shepp_logan_maps(N)
    phantom = mrsim.phantom_from_maps(maps["pd"], maps["t1"], maps["t2"], FOV)
    # CSF (T1 = 4 s) needs ~200 dummy TRs to reach steady state at TR 10 ms / 15 deg.
    # A 1 us dwell keeps T2 decay during the readout (which blurs the image along x,
    # ~5e-4 at a 10 us dwell) below the tolerance of the image-domain comparisons.
    seq = mrsim.gre_2d(N, FOV, flip_deg=FLIP, tr=TR, dwell=1e-6, n_dummy=300)
    return maps, phantom, seq


def expected_image(maps, seq):
    mask = maps["pd"] > 0
    mxy = spoiled_gre_signal(maps["pd"], maps["t1"], maps["t2"], TR, seq.meta["te"], math.radians(FLIP))
    return torch.where(mask, 1j * mxy, 0)  # RF phase 0 puts the magnetization along +y


def test_bloch_matches_signal_equation(setup):
    _, phantom, seq = setup
    coils = mrsim.birdcage_sensitivities(phantom.pos, n_coils=4)
    bloch = mrsim.simulate(phantom, seq, coils).signal
    ref = signal_equation(phantom, seq, coils)
    err = (bloch - ref).abs().max() / ref.abs().max()
    assert err < 1e-4, err


def test_recon_matches_tissue_model(setup):
    maps, phantom, seq = setup
    img = mrsim.ifft2c(mrsim.kspace_grid(mrsim.simulate(phantom, seq).signal, seq))[0]
    expected = expected_image(maps, seq)
    assert (img - expected).abs().max() < 1e-4 * expected.abs().max()


def test_multicoil_combinations(setup):
    maps, phantom, seq = setup
    coils = mrsim.birdcage_sensitivities(phantom.pos, n_coils=8, normalize=True)
    imgs = mrsim.ifft2c(mrsim.kspace_grid(mrsim.simulate(phantom, seq, coils).signal, seq))
    expected = expected_image(maps, seq)
    tol = 1e-4 * expected.abs().max()
    assert (mrsim.rss(imgs) - expected.abs()).abs().max() < tol

    grid = mrsim.grid_positions(N, FOV).reshape(-1, 3)
    sens_grid = mrsim.birdcage_sensitivities(grid, n_coils=8, normalize=True).T.reshape(8, N, N)
    assert (mrsim.coil_combine(imgs, sens_grid) - expected).abs().max() < tol


def test_off_resonance_shifts_along_readout(setup):
    # A uniform offset of 2 pixels' bandwidth shifts the image by +2 columns.
    # Own sequence: 10 us dwell keeps the offset (6.25 kHz) small relative to a
    # 0.1 us excitation pulse, so the flip angle is unaffected.
    maps, phantom, _ = setup
    seq = mrsim.gre_2d(N, FOV, flip_deg=FLIP, tr=TR, rf_duration=0.1e-6, n_dummy=300)
    bw = mrsim.readout_bandwidth_per_pixel(seq)
    shifted = mrsim.phantom_from_maps(maps["pd"], maps["t1"], maps["t2"], FOV,
                                      db0=torch.full_like(maps["pd"], 2 * bw))
    recon = lambda p: mrsim.ifft2c(mrsim.kspace_grid(mrsim.simulate(p, seq).signal, seq))[0].abs()
    img, ref = recon(shifted), recon(phantom).roll(2, dims=-1)
    assert (img - ref).abs().max() < 1e-3 * ref.max()


def test_noise_statistics():
    g = torch.Generator().manual_seed(0)
    zero = torch.zeros(2, 200_000, dtype=torch.complex128)
    cov = torch.tensor([[1.0, 0.6], [0.6, 2.0]], dtype=torch.float64)
    noisy = mrsim.add_noise(zero, 0.1, cov, generator=g)
    emp = (noisy @ noisy.conj().T).real / noisy.shape[1] / (2 * 0.1**2)
    assert torch.allclose(emp, cov, atol=0.02)
