"""B0 field-map helpers: shim terms are solid harmonics (Laplace's equation), and the closed-form
sphere / cylinder susceptibility fields match an independent Fourier-domain dipole convolution,
Delta f(k) = gamma_bar B0 chi(k) (1/3 - kz^2 / k^2) (Lorentz-corrected, Salomir 2003 /
Marques & Bowtell 2005), on a partial-volume chi grid. A linear shim term shears a GRE image
by exactly f(y) / (bandwidth per pixel)."""

import math
from dataclasses import replace

import pytest
import torch

import blochparty as bp
from blochparty.phantom import GAMMA_BAR, b0_cylinder, b0_shim, b0_sphere
from cases import partial_volume

DCHI, B0 = 9.4e-6, 3.0


def test_shim_terms():
    g = torch.Generator().manual_seed(0)
    pos = (torch.rand(50, 3, generator=g, dtype=torch.float64) - 0.5) * 0.3
    x, y, z = pos.unbind(-1)
    assert torch.equal(b0_shim(pos), torch.zeros(50, dtype=torch.float64))
    assert torch.allclose(b0_shim(pos, offset=3.0, x=10.0, y=-20.0, z=5.0), 3 + 10 * x - 20 * y + 5 * z, atol=1e-13)
    terms = {"z2": z**2 - (x**2 + y**2) / 2, "zx": z * x, "zy": z * y, "x2y2": x**2 - y**2, "xy": x * y}
    for name, expected in terms.items():
        assert torch.allclose(b0_shim(pos, **{name: 7.0}), 7 * expected, atol=1e-13), name
        # every term solves Laplace's equation, as a static field in a source-free region must
        lap = sum(torch.autograd.functional.hessian(lambda p: b0_shim(p[None], **{name: 7.0})[0], pos[i]).trace()
                  for i in range(3))
        assert abs(lap) < 1e-12, name
    # coefficients can be optimised
    c = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    b0_shim(pos, z2=c).sum().backward()
    assert torch.allclose(c.grad, terms["z2"].sum())


def test_sphere_vs_fourier_dipole():
    n, h, a = 96, 1e-3, 9e-3
    ax = (torch.arange(n, dtype=torch.float64) - n // 2) * h
    z, y, x = torch.meshgrid(ax, ax, ax, indexing="ij")
    chi = DCHI * partial_volume(lambda z, y, x: x**2 + y**2 + z**2 <= a * a, (z, y, x), h)
    k = torch.fft.fftfreq(n, d=h, dtype=torch.float64)
    kz, ky, kx = torch.meshgrid(k, k, k, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2
    kernel = 1 / 3 - kz**2 / torch.where(k2 > 0, k2, 1.0)
    kernel[0, 0, 0] = 0.0
    ref = torch.fft.ifftn(torch.fft.fftn(chi) * kernel).real * GAMMA_BAR * B0
    pos = torch.stack([x, y, z], -1).reshape(-1, 3)
    centre = (0.0, 0.0, 0.0)
    out = b0_sphere(pos, centre, a, DCHI, B0).reshape(n, n, n)
    scale = GAMMA_BAR * B0 * DCHI * 2 / 3  # the largest field, at the poles (~176 Hz at 3 T)
    r = pos.norm(dim=-1).reshape(n, n, n)
    for lo, hi in [(0, 0.8), (1.25, 2), (2, 4)]:  # away from the staircased surface
        m = (r >= lo * a) & (r < hi * a)
        assert (out - ref)[m].abs().max() < 0.01 * scale, (lo, hi)
    # the dipole pattern itself: +2 at the poles, -1 at the equator (x scale/2), zero inside
    pts = torch.tensor([[0, 0, 1.0], [1.0, 0, 0], [0, 0.3, 0.2]], dtype=torch.float64) * a * 1.0000001
    assert torch.allclose(b0_sphere(pts, centre, a, DCHI, B0)[:2], torch.tensor([scale, -scale / 2],
                                                                                 dtype=torch.float64), rtol=1e-6)
    assert b0_sphere(pts, centre, a, DCHI, B0)[2] == 0


@pytest.mark.parametrize("theta_deg", [90.0, 54.7356103, 30.0, 0.0])
def test_cylinder_vs_fourier_dipole(theta_deg):
    # an infinite cylinder's field is 2D: convolve in the plane perpendicular to its axis, where
    # k.z = k_u sin(theta) with u along B0's projection on that plane
    th = math.radians(theta_deg)
    axis = torch.tensor([math.sin(th) * math.cos(0.4), math.sin(th) * math.sin(0.4), math.cos(th)],
                        dtype=torch.float64)
    zhat = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    zp = zhat - axis[2] * axis
    u = zp / zp.norm() if zp.norm() > 1e-9 else torch.tensor([1.0, 0, 0], dtype=torch.float64)
    v = torch.linalg.cross(axis, u)
    m, h, a = 512, 0.5e-3, 10e-3
    c = (torch.arange(m, dtype=torch.float64) - m // 2) * h
    U, V = torch.meshgrid(c, c, indexing="ij")
    chi = DCHI * partial_volume(lambda U, V: U**2 + V**2 <= a * a, (U, V), h)
    k = torch.fft.fftfreq(m, d=h, dtype=torch.float64)
    KU, KV = torch.meshgrid(k, k, indexing="ij")
    k2 = KU**2 + KV**2
    kernel = 1 / 3 - (KU * math.sin(th)) ** 2 / torch.where(k2 > 0, k2, 1.0)
    kernel[0, 0] = 0.0
    ref = torch.fft.ifft2(torch.fft.fft2(chi) * kernel).real * GAMMA_BAR * B0
    centre = torch.tensor([0.01, -0.02, 0.005], dtype=torch.float64)
    pos = centre + U[..., None] * u + V[..., None] * v + 0.37 * axis  # anywhere along the axis
    out = b0_cylinder(pos.reshape(-1, 3), centre, a, DCHI, axis=axis, field_strength=B0).reshape(m, m)
    scale = GAMMA_BAR * B0 * DCHI / 2
    rho = (U**2 + V**2).sqrt()
    for lo, hi in [(0, 0.8), (1.25, 2), (2, 4)]:
        mk = (rho >= lo * a) & (rho < hi * a)
        assert (out - ref)[mk].abs().max() < 0.005 * scale, (lo, hi)
    inside = GAMMA_BAR * B0 * DCHI / 6 * (3 * math.cos(th) ** 2 - 1)  # zero at the magic angle
    assert torch.allclose(out[rho < a], torch.full_like(out[rho < a], inside))


def test_linear_shim_shears_gre_image():
    # f = c y with c * (pixel size) = one bandwidth-pixel: row i moves by (i - n/2) pixels along +x
    # (offsets up to 50 kHz: 10 ns pulses keep off-resonance during excitation negligible)
    n, fov = 16, 0.2
    m = bp.shepp_logan_maps(n)
    seq = bp.gre_2d(n, fov, flip_deg=15.0, tr=10e-3, rf_duration=0.01e-6, n_dummy=400)  # CSF at steady state
    bw = bp.readout_bandwidth_per_pixel(seq)
    ph = bp.phantom_from_maps(m["pd"], m["t1"], m["t2"], fov)
    sheared = replace(ph, db0=ph.db0 + b0_shim(ph.pos, y=bw / (fov / n)))
    recon = lambda p: bp.ifft2c(bp.kspace_grid(bp.simulate(p, seq).signal, seq))[0].abs()
    img, ref = recon(sheared), recon(ph)
    expected = torch.stack([ref[i].roll(i - n // 2) for i in range(n)])
    assert (img - expected).abs().max() < 1e-3 * ref.max()
    assert (img - ref).abs().max() > 0.3 * ref.max()
