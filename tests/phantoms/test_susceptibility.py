"""Independent analytic checks of susceptibility fields and harmonic shimming.

At 3 T and dchi=9.4e-6, the 10-mm sphere in a 64-mm box gives inside/outside
RMS errors 0.908/0.369, 0.804/0.190, 0.439/0.108 Hz at 32^3, 48^3, 64^3
(2x padding, 4^3 boundary samples/voxel; r < 0.7a and 1.4a < r < 2.3a).
Far-field RMS (2a < r < 3a) falls 4.287 -> 0.134 -> 0.0216 Hz for padding
1x -> 2x -> 3x at 64^3. No anti-alias kernel is needed at this accuracy.
Long-cylinder inside/outside RMS at 0/magic/90 degrees: 1.604/1.598,
1.671/0.331, 1.766/0.830 Hz; finite ends and periodic copies remain.
Second-order shimming reduces the specified sphere ROI from 94.111 to
20.615 Hz RMS; a pure harmonic field is removed to < 1e-12 Hz.
No downloaded data or changes to the Bloch engines are needed for these checks.
The QSM Challenge forward-field comparison belongs to M3, once its data is loaded.
"""

import math

import pytest
import torch

from blochparty.constants import GAMMA_BAR
from blochparty.phantom import b0_cylinder, b0_shim, b0_sphere
from blochparty.susceptibility import shim, susceptibility_to_b0
from test_phantom_b0 import partial_volume

DCHI, FIELD = 9.4e-6, 3.0
SCALE = GAMMA_BAR * FIELD * DCHI


def coordinates(shape, voxel, origin=None):
    if origin is None:
        origin = tuple(-(n - 1) * h / 2 for n, h in reversed(list(zip(shape, voxel))))
    axes = [torch.arange(n, dtype=torch.float64) * h + o
            for n, h, o in zip(shape, voxel, reversed(origin))]
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    return torch.stack((x, y, z), -1)


def sphere(n=48):
    h, radius = 0.064 / n, 0.01
    pos = coordinates((n,) * 3, (h,) * 3)
    x, y, z = pos.unbind(-1)
    chi = DCHI * partial_volume(lambda x, y, z: x*x + y*y + z*z <= radius**2,
                               (x, y, z), h)
    ref = b0_sphere(pos, (0, 0, 0), radius, DCHI, FIELD)
    return chi, pos, h, radius, ref


def rms(x):
    return x.square().mean().sqrt().item()


def test_sphere_resolution_and_padding():
    errors = []
    for n in (32, 48, 64):
        chi, pos, h, a, ref = sphere(n)
        radius = pos.norm(dim=-1)
        field = susceptibility_to_b0(chi, (h,) * 3, padding=2)
        masks = [radius < 0.7 * a, (radius > 1.4 * a) & (radius < 2.3 * a)]
        errors.append([rms((field - ref)[m]) for m in masks])
    print("sphere resolution RMS Hz (inside, outside):", errors)
    assert all(errors[-1][i] < errors[0][i] * 0.65 for i in range(2))
    assert max(errors[-1]) < 1.0
    far = (radius > 2 * a) & (radius < 3 * a)
    padding_errors = [rms((susceptibility_to_b0(chi, (h,) * 3, padding=p) - ref)[far])
                      for p in (1, 2, 3)]
    print("sphere padding RMS Hz:", padding_errors)
    assert padding_errors[1] < padding_errors[0] / 5
    assert padding_errors[2] < padding_errors[1]


@pytest.mark.parametrize("theta", [0.0, math.acos(1 / math.sqrt(3)), math.pi / 2])
def test_long_cylinder(theta):
    # A finite 400-mm cylinder of radius 6 mm; inspect only the central 8 mm.
    # Rotate the coordinates into the analytic helper's B0-along-z frame.
    voxel = (0.008, 0.001, 0.001)
    pos = coordinates((64, 64, 64), voxel)
    x, y, z = pos.unbind(-1)
    a = 0.006
    disc = partial_volume(lambda x, y: x*x + y*y <= a*a, (x[0], y[0]), voxel[1], sub=8)
    chi = DCHI * disc[None] * (z.abs() < 0.2)
    s, c = math.sin(theta), math.cos(theta)
    # Long cylinders need more transverse padding than compact spheres:
    # periodic copies cause a nearly uniform offset even far from the ends.
    field = susceptibility_to_b0(chi, voxel, b0_dir=(s, 0, c), padding=(2, 4, 4))
    rotated = torch.stack((c*x - s*z, y, s*x + c*z), -1)
    ref = b0_cylinder(rotated, (0, 0, 0), a, DCHI, axis=(-s, 0, c))
    rho = (x*x + y*y).sqrt()
    masks = [(rho < 0.65*a) & (z.abs() < 0.005),
             (rho > 1.5*a) & (rho < 3*a) & (z.abs() < 0.005)]
    errors = [rms((field-ref)[m]) for m in masks]
    print("cylinder degrees / RMS Hz / interior mean Hz:", math.degrees(theta), errors,
          field[masks[0]].mean().item())
    assert max(errors) < 0.005 * SCALE
    expected = SCALE / 6 * (3*c*c - 1)
    assert abs(field[masks[0]].mean().item() - expected) < 0.002 * SCALE


def test_linearity_background_and_units():
    g = torch.Generator().manual_seed(13)
    a, b = torch.randn(2, 9, 11, 13, generator=g, dtype=torch.float64) * 1e-6
    kwargs = dict(voxel=(0.003, 0.002, 0.001), b0_dir=(1, 2, 3), padding=(1.5, 2, 2.5))
    f = lambda chi, **kw: susceptibility_to_b0(chi, **kwargs, **kw)
    torch.testing.assert_close(f(2*a - b), 2*f(a) - f(b), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(f(a, field_T=1.5), f(a) / 2)
    torch.testing.assert_close(f(a + 0.36e-6, background=0.36e-6), f(a), atol=1e-12, rtol=1e-12)
    assert torch.count_nonzero(f(a, field_T=0)) == 0
    assert torch.count_nonzero(f(torch.full_like(a, 0.36e-6), background=0.36e-6)) == 0
    # Periodic uniform material has only k=0 and hence no relative field.
    assert susceptibility_to_b0(torch.ones(8, 8, 8, dtype=torch.float64), (1, 1, 1),
                                padding=1).abs().max() == 0
    # A sinusoid is an independent exact eigenfunction of the periodic operator.
    z = torch.arange(12, dtype=torch.float64)[:, None, None]
    chi = (1e-6 * torch.cos(2*math.pi*z/12)).expand(12, 10, 8)
    torch.testing.assert_close(susceptibility_to_b0(chi, (1, 2, 3), padding=1),
                               -2/3 * GAMMA_BAR * FIELD * chi, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("shape", [(9, 11, 13), (8, 10, 12)])
def test_joint_rotation(shape):
    chi = torch.randn(shape, generator=torch.Generator().manual_seed(14), dtype=torch.float64) * 1e-6
    voxel, direction = (0.003, 0.002, 0.001), (1.0, 2.0, 3.0)
    field = susceptibility_to_b0(chi, voxel, b0_dir=direction)
    # Quarter turn about z: (x', y', z') = (y, -x, z).
    rotated = susceptibility_to_b0(chi.rot90(1, (1, 2)), (voxel[0], voxel[2], voxel[1]),
                                   b0_dir=(direction[1], -direction[0], direction[2]))
    torch.testing.assert_close(rotated, field.rot90(1, (1, 2)), atol=2e-12, rtol=1e-12)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_shim_removes_harmonics(order):
    shape, voxel, origin = (9, 10, 11), (0.003, 0.002, 0.001), (-0.007, -0.011, -0.017)
    pos = coordinates(shape, voxel, origin)
    coeff = dict(offset=13.0)
    if order >= 1:
        coeff.update(x=41., y=-63., z=27.)
    if order == 2:
        coeff.update(z2=131., zx=-97., zy=183., x2y2=-156., xy=213.)
    df = b0_shim(pos, **coeff)
    mask = pos.norm(dim=-1) < 0.014
    residual, fitted = shim(df, mask, order, voxel=voxel, origin=origin)
    assert residual.abs().max() < 1e-12
    for name, expected in coeff.items():
        assert fitted[name].item() == pytest.approx(expected, abs=2e-9)
    torch.testing.assert_close(df - residual, b0_shim(pos, **fitted), atol=1e-12, rtol=1e-12)


def test_shim_sphere_reduces_field_and_is_orthogonal():
    chi, pos, h, a, _ = sphere(32)
    df = susceptibility_to_b0(chi, (h,) * 3)
    # Tissue region on one side of the inclusion (a source-free region).
    mask = ((pos - torch.tensor([0.014, 0.0, 0.0])).norm(dim=-1) < 0.008)
    mask &= pos.norm(dim=-1) > 1.2*a
    residual, coeff = shim(df, mask, voxel=(h,) * 3)
    print("sphere shim RMS Hz before / after:", rms(df[mask]), rms(residual[mask]))
    assert rms(residual[mask]) < rms(df[mask]) / 3
    for term in coeff:
        basis = b0_shim(pos, **{term: 1.0})[mask]
        assert abs(torch.dot(basis / basis.norm(), residual[mask])) < 1e-10


def test_dtype_and_gradients():
    g = torch.Generator().manual_seed(16)
    chi = (torch.randn(3, 4, 5, dtype=torch.float64, generator=g) * 1e-6).requires_grad_()
    assert torch.autograd.gradcheck(lambda v: susceptibility_to_b0(v, (1, 2, 3)), (chi,))
    field = susceptibility_to_b0(chi.detach().float(), (1, 2, 3))
    assert field.dtype == torch.float32
    torch.testing.assert_close(field.double(), susceptibility_to_b0(chi, (1, 2, 3)),
                               atol=2e-5, rtol=1e-5)
    mask = torch.ones_like(chi, dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda v: shim(v, mask)[0], (chi,))


def test_float32_shim_large_mask():
    # N*eps >= 1 would reject even the constant column with default rank tolerances.
    df = torch.full((1, 1, 2**23 + 1), 17.0, dtype=torch.float32)
    residual, coeff = shim(df, torch.ones_like(df, dtype=torch.bool), order=0)
    assert residual.abs().max() < 2e-4
    assert coeff["offset"].item() == pytest.approx(17.0, abs=2e-4)


@pytest.mark.parametrize("kwargs", [dict(voxel=(1, 0, 1)), dict(voxel=(1, 2)),
    dict(b0_dir=(0, 0, 0)), dict(b0_dir=(1, 2)), dict(padding=0), dict(padding=(1, 2)),
    dict(background=float("nan")), dict(field_T=-1)])
def test_invalid_field_arguments(kwargs):
    options = dict(voxel=(1, 1, 1)) | kwargs
    with pytest.raises(ValueError):
        susceptibility_to_b0(torch.zeros(3, 4, 5, dtype=torch.float64), **options)


def test_invalid_volumes_and_shim_masks():
    for chi in (torch.zeros(3, 4), torch.zeros(0, 3, 4), torch.ones(3, 4, 5, dtype=torch.int64),
                torch.full((3, 4, 5), float("nan"))):
        with pytest.raises(ValueError):
            susceptibility_to_b0(chi, (1, 1, 1))
    df = torch.zeros(3, 4, 5, dtype=torch.float64)
    for mask in (torch.zeros_like(df, dtype=torch.bool), torch.ones_like(df),
                 torch.ones(3, 4, 4, dtype=torch.bool)):
        with pytest.raises(ValueError):
            shim(df, mask)
    mask = torch.zeros_like(df, dtype=torch.bool)
    mask[1] = True
    with pytest.raises(ValueError, match="rank deficient"):
        shim(df, mask)
    with pytest.raises(ValueError):
        shim(df, torch.ones_like(df, dtype=torch.bool), order=3)
