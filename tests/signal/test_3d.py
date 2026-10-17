"""3D imaging: the same end-to-end checks as 2D, on a 3D Shepp-Logan and 3D GRE."""

import math

import pytest
import torch

import blochparty as bp
from blochparty.reference import signal_equation, spoiled_gre_signal

N, FOV, FLIP, TR = 12, 0.22, 15.0, 10e-3


@pytest.fixture(scope="module")
def setup():
    maps = bp.shepp_logan_maps(N, ndim=3)
    phantom = bp.phantom_from_maps(maps["pd"], maps["t1"], maps["t2"], FOV)
    # 1 us dwell: T2 decay during the readout stays below the image-domain tolerance
    seq = bp.gre_3d(N, FOV, flip_deg=FLIP, tr=TR, dwell=1e-6, n_dummy=300)
    return maps, phantom, seq


def test_grid_and_phantom_layout():
    pos = bp.grid_positions((4, 5, 6), 0.3)
    assert pos.shape == (4, 5, 6, 3)
    # last axis is x, then y, then z; index n/2 is the origin
    assert torch.allclose(pos[2, 2, 3], torch.zeros(3, dtype=pos.dtype))
    assert pos[0, 0, 1, 0] - pos[0, 0, 0, 0] == pytest.approx(0.3 / 6)
    assert pos[0, 1, 0, 1] - pos[0, 0, 0, 1] == pytest.approx(0.3 / 5)
    assert pos[1, 0, 0, 2] - pos[0, 0, 0, 2] == pytest.approx(0.3 / 4)
    maps = bp.shepp_logan_maps(32, ndim=3)
    assert maps["pd"].shape == (32, 32, 32)
    # every tissue appears, and the head is closed: nothing on the outer faces
    assert set(maps["label"].unique().tolist()) == set(range(len(bp.phantom.LABELS)))
    faces = [maps["label"][0], maps["label"][:, 0], maps["label"][:, :, 0]]
    assert all((f == 0).all() for f in faces)


def test_kspace_ordering(setup):
    _, _, seq = setup
    # prephasers sit between an RF pulse and a readout; their (ky, kz) areas, in units
    # of dk, must step with ky fastest and kz slowest, both centred on n/2
    pre = [s for s in range(1, len(seq) - 1) if seq.is_rf[s - 1] and seq.adc[s + 1]]
    k = (seq.grad[pre] * seq.dt[pre, None])[:, 1:] * FOV
    kz, ky = torch.meshgrid(torch.arange(N) - N // 2, torch.arange(N) - N // 2, indexing="ij")
    expected = torch.stack([ky.reshape(-1), kz.reshape(-1)], dim=-1).to(k.dtype)
    assert torch.allclose(k, expected, atol=1e-9)


def test_bloch_matches_signal_equation(setup):
    _, phantom, seq = setup
    coils = bp.birdcage_sensitivities(phantom.pos, n_coils=4)
    bloch = bp.simulate(phantom, seq, coils).signal
    ref = signal_equation(phantom, seq, coils)
    assert (bloch - ref).abs().max() / ref.abs().max() < 1e-4


def test_recon_matches_tissue_model(setup):
    maps, phantom, seq = setup
    img = bp.reconstruct(bp.simulate(phantom, seq).signal, seq)[0]
    mxy = spoiled_gre_signal(maps["pd"], maps["t1"], maps["t2"], TR, seq.meta["te"], math.radians(FLIP))
    expected = torch.where(maps["pd"] > 0, 1j * mxy, 0)
    assert img.shape == (N, N, N)
    assert (img - expected).abs().max() < 1e-4 * expected.abs().max()


def test_gradients_flow_in_3d():
    phantom = bp.shepp_logan(6, ndim=3)
    flip = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
    seq = bp.gre_3d(6, flip_deg=flip, n_dummy=3)
    loss = bp.reconstruct(bp.simulate(phantom, seq).signal, seq).abs().sum()
    (g,) = torch.autograd.grad(loss, flip)
    eps = 1e-4
    fd = [
        bp.reconstruct(bp.simulate(phantom, bp.gre_3d(6, flip_deg=20.0 + s * eps, n_dummy=3)).signal, seq).abs().sum()
        for s in (1, -1)
    ]
    assert torch.isclose(g, (fd[0] - fd[1]) / (2 * eps), rtol=1e-6)
