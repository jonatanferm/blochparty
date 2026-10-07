"""Loop coils (``loops.py``) against closed forms: the on-axis field of a circular loop, the
far-field dipole, the off-axis vector potential (elliptic integrals); reciprocity conventions;
the sample-loss resistance matrix (symmetric, positive definite, correlation falling with
distance, identical coils fully correlated)."""

import math

import numpy as np
import torch
from scipy.special import ellipe, ellipk

import blochparty as bp
from blochparty.loops import MU0, Loop, b1_plus, fields, receive_sensitivities, resistance

f64 = torch.float64


def test_on_axis_field():
    a = 0.05
    z = torch.tensor([0.0, 0.02, 0.05, 0.1], dtype=f64)
    pts = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], -1)
    B, _ = fields([Loop((0, 0, 0), (0, 0, 1), a, n_seg=512)], pts)
    exact = MU0 * a**2 / (2 * (a**2 + z**2) ** 1.5)
    assert torch.allclose(B[0, :, 2], exact, rtol=1e-4) and B[0, :, :2].abs().max() < 1e-12


def test_far_field_is_a_dipole():
    a = 0.01
    r = torch.tensor([[0.3, 0.1, 0.2], [-0.2, 0.25, -0.15]], dtype=f64)
    B, _ = fields([Loop((0, 0, 0), (0, 0, 1), a, n_seg=512)], r)
    m = torch.tensor([0, 0, math.pi * a**2], dtype=f64)
    d = r.norm(dim=-1, keepdim=True)
    u = r / d
    dip = MU0 / (4 * math.pi) * (3 * (u @ m)[:, None] * u - m) / d**3
    assert torch.allclose(B[0], dip, rtol=2e-3, atol=1e-3 * float(dip.abs().max()))


def test_off_axis_vector_potential():
    a = 0.05
    rho, z = np.array([0.02, 0.06, 0.1]), np.array([0.01, 0.03, -0.04])
    pts = torch.as_tensor(np.stack([rho, np.zeros(3), z], -1))
    _, A = fields([Loop((0, 0, 0), (0, 0, 1), a, n_seg=2048)], pts)
    k2 = 4 * a * rho / ((a + rho) ** 2 + z**2)
    k = np.sqrt(k2)
    a_phi = MU0 / (math.pi * k) * np.sqrt(a / rho) * ((1 - k2 / 2) * ellipk(k2) - ellipe(k2))
    # at (rho, 0, z) the azimuthal direction is +y
    assert np.allclose(A[0, :, 1].numpy(), a_phi, rtol=1e-4) and float(A[0, :, ::2].abs().max()) < 1e-12


def test_reciprocity_conventions():
    # a z-field has no transverse part: no B1+, no receive sensitivity; a field along x gives B1+ = B1- = Bx / 2
    B = torch.tensor([[[0.0, 0.0, 1.0], [2.0, 0.0, 0.0]]], dtype=f64)
    assert torch.allclose(b1_plus(B), torch.tensor([0.0, 1.0], dtype=torch.complex128))
    assert torch.allclose(receive_sensitivities(B)[:, 0], torch.tensor([0.0, 1.0], dtype=torch.complex128))


def test_resistance_matrix():
    # a conducting sphere of saline under three loops: two neighbours and one far away
    g = torch.Generator().manual_seed(0)
    n = 20000
    pts = (torch.rand(n, 3, generator=g, dtype=f64) - 0.5) * 0.2
    pts = pts[pts.norm(dim=-1) < 0.1]
    vol = torch.full((len(pts),), 4 / 3 * math.pi * 0.1**3 / len(pts), dtype=f64)
    sigma = torch.full((len(pts),), 0.6, dtype=f64)
    loops = [Loop((0, 0, 0.12), (0, 0, 1), 0.04), Loop((0.06, 0, 0.11), (0, 0, 1), 0.04),
             Loop((0, 0, -0.12), (0, 0, 1), 0.04), Loop((0, 0, 0.12), (0, 0, 1), 0.04)]
    _, A = fields(loops, pts)
    R = resistance(A, sigma, vol, 64e6)
    assert torch.allclose(R, R.T) and bool((torch.linalg.eigvalsh(R[:3, :3]) > 0).all())
    corr = R / torch.sqrt(torch.diag(R)[:, None] * torch.diag(R)[None, :])
    assert abs(float(corr[0, 3]) - 1) < 1e-12  # identical coils
    assert abs(float(corr[0, 1])) > abs(float(corr[0, 2]))  # neighbours share more of the sample
    assert 0.01 < float(R[0, 0]) < 100  # ohms, a plausible loaded loop at 1.5 T


def test_field_maps_in_a_simulation():
    # a 4-loop array around the phantom: mapped coils sampled at the spins equal the coils computed
    # at the spins, through a whole simulation (to trilinear interpolation on a 1 mm grid)
    from blochparty.loops import field_maps

    loops = [Loop((0.2 * math.cos(t), 0.2 * math.sin(t), 0.0), (math.cos(t), math.sin(t), 0.0), 0.06)
             for t in np.arange(4) * math.pi / 2]
    ph = bp.shepp_logan(32)
    seq = bp.gre_2d(32, n_dummy=2)
    lo = float(ph.pos[:, :2].min()) - 0.002
    nxy = int(math.ceil((float(ph.pos[:, :2].max()) + 0.002 - lo) / 0.001)) + 1
    maps = field_maps(loops, origin=(lo, lo, -0.001), spacing=(0.001, 0.001, 0.001), shape=(3, nxy, nxy), scale=1e3)
    direct = 1e3 * receive_sensitivities(fields(loops, ph.pos)[0])
    a = bp.simulate(ph, seq, maps=bp.FieldMaps(maps.origin, maps.spacing, coils=maps.coils)).signal
    b = bp.simulate(ph, seq, direct).signal
    assert float((a - b).abs().max() / b.abs().max()) < 2e-3
