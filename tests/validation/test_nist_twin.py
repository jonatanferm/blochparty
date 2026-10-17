"""The digital-twin helpers of the NIST benchmarks (validation/real/nist.py, nist_twin.py): the
absolute metric's single scale, the receive profile's exact recovery of a polynomial shading, and
the B1+ model's identifiability from one sagittal plane (what lets a fit on one plane predict the
other)."""

import numpy as np
import torch

from nist import CENTRE, N, absolute_error, receive_profile
from nist_twin import R_B1, b1_field


def test_absolute_error_has_one_scale_for_all_spheres():
    rng = np.random.default_rng(0)
    real = rng.uniform(0.2, 1.0, (10, 14))
    assert np.allclose(absolute_error(3.7 * real, real), 0)  # a global gain costs nothing
    per_sphere = real * np.r_[np.ones(13), 1.5]  # one sphere 50 % brighter: only a per-sphere scale hides it
    err = absolute_error(per_sphere, real, spheres=slice(0, 14))
    assert err[13] > 0.2 and err[:13].max() < 0.1


def test_receive_profile_recovers_a_quadratic_shading():
    rr, cc = np.indices((N, N))
    u, v = (rr - CENTRE) / N, (cc - CENTRE) / N
    shading = 1 + 0.3 * u - 0.2 * v + 0.5 * u * v - 0.4 * u**2 + 0.25 * v**2
    water = (np.hypot(rr - CENTRE, cc - CENTRE) < 50) & ((rr + cc) % 3 == 0)  # sparse pixels, as the fill
    prof = receive_profile(7.0 * shading, water)
    np.testing.assert_allclose(prof, shading / shading[water].mean(), rtol=1e-10)
    assert abs(prof[water].mean() - 1) < 1e-12


def test_b1_model_is_identified_by_one_sagittal_plane():
    theta = torch.tensor([1.08, -0.3, 0.12, 0.05, -0.04], dtype=torch.float64)
    assert float(b1_field(torch.zeros(3, dtype=torch.float64), theta)) == 1.08  # s at the centre
    g = torch.linspace(-0.07, 0.07, 9, dtype=torch.float64)
    yy, zz = torch.meshgrid(g, g, indexing="ij")
    plane = torch.stack([torch.full_like(yy, 0.057), yy, zz], -1).reshape(-1, 3)  # the T1 plane, x = 57 mm
    field = b1_field(plane, theta)
    # the field is linear in (s, s a, s b, s c_y, s c_z): recover them by least squares from this plane alone
    x, y, z = plane.unbind(-1)
    A = torch.stack([torch.ones_like(x), (x**2 + y**2) / R_B1**2, z**2 / R_B1**2, y / R_B1, z / R_B1], -1)
    coef = torch.linalg.lstsq(A, field[:, None]).solution[:, 0]
    assert torch.linalg.matrix_rank(A) == 5
    s = coef[0]
    np.testing.assert_allclose((torch.cat([s[None], coef[1:] / s])).numpy(), theta.numpy(), rtol=1e-9)
    # so the other plane (x = -12 mm) is predicted
    other = plane.clone()
    other[:, 0] = -0.012
    rebuilt = torch.cat([s[None], coef[1:] / s])
    np.testing.assert_allclose(b1_field(other, rebuilt).numpy(), b1_field(other, theta).numpy(), rtol=1e-9)
