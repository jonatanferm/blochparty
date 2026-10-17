"""Full-wave map import (``em.py``): mariepy's file format and phasor convention against the
rotating-field definition of B1+, the quasi-static loop fields of ``loops.py``, the (x, y, z) ->
(z, y, x) reordering and the fill outside the body."""

import json
import math

import numpy as np
import pytest
import torch

from blochparty import em
from blochparty.loops import Loop, fields
from blochparty.loops import field_maps as loop_maps
from blochparty.maps import sample

c128 = torch.complex128


def write_maps(path, plus, minus, mask, origin, resolution, frequency_hz=127.7e6):
    """A map file as mariepy's ``maps.write`` lays it out (single precision, JSON metadata)."""
    meta = {
        "coil": "test",
        "channels": [f"ch{i}" for i in range(plus.shape[0])],
        "frequency_hz": frequency_hz,
        "drive_unit": "1 A",
        "origin": list(origin),
        "resolution": resolution,
        "frame": "test",
        "bodies": [],
        "mariepy_version": "0",
        "data_licence": "none",
    }
    np.savez_compressed(
        path,
        plus=np.asarray(plus, np.complex64),
        minus=np.asarray(minus, np.complex64),
        mask=np.asarray(mask, bool),
        metadata=np.array(json.dumps(meta)),
    )


def rotating_field(phasor, t, f):
    """Lab-frame field Re(B e^{+j w t}) of a phasor (3,), (len(t), 3)."""
    return (torch.as_tensor(phasor, dtype=c128)[None] * torch.exp(1j * 2 * math.pi * f * t)[:, None]).real


@pytest.mark.parametrize("phase", [0.0, 0.7, -2.1])
def test_transmit_is_the_clockwise_component_with_its_phase(tmp_path, phase):
    # protons (gamma > 0, B0 along +z) precess clockwise seen from +z: a clockwise field of
    # amplitude 1 excites with |B1+| = 1, a counterclockwise one not at all
    f = 1e6
    t = torch.linspace(0, 1 / f, 257, dtype=torch.float64)[:-1]
    e = math.e ** (1j * phase)
    cw, ccw = torch.tensor([1, 1j, 0], dtype=c128) * e, torch.tensor([1, -1j, 0], dtype=c128) * e
    lab = rotating_field(cw, t, f)
    # rotating-frame B1 = (Bx + i By) demodulated by e^{+i w t} (blochparty's Mxy ~ e^{-i w t} frame)
    b1_frame = (torch.complex(lab[:, 0], lab[:, 1]) * torch.exp(1j * 2 * math.pi * f * t)).mean()
    ph = torch.stack([cw, ccw])  # two channels (phasors (C, 3)) at one voxel
    plus, minus = (ph[:, 0] + 1j * ph[:, 1]), (ph[:, 0] - 1j * ph[:, 1])
    write_maps(
        tmp_path / "m.npz",
        plus[:, None, None, None],
        minus[:, None, None, None],
        np.ones((1, 1, 1)),
        (0, 0, 0),
        1e-3,
        f,
    )
    m = em.read_mariepy_maps(tmp_path / "m.npz")
    b1 = [em.transmit_b1(m, torch.tensor(d, dtype=c128))[0, 0, 0] for d in ([1, 0], [0, 1])]
    assert abs(b1[0] - b1_frame) < 1e-6 and abs(b1[0] - e.conjugate()) < 1e-6
    assert abs(b1[1]) < 1e-6


def test_quadrature_drive_picks_the_co_rotating_mode(tmp_path):
    # two linear modes, B along x and along y: plus = Bx + j By = (1, j), minus = (1, -j)
    plus = torch.tensor([1, 1j], dtype=c128)[:, None, None, None].expand(2, 2, 2, 2)
    minus = torch.tensor([1, -1j], dtype=c128)[:, None, None, None].expand(2, 2, 2, 2)
    write_maps(tmp_path / "q.npz", plus, minus, np.ones((2, 2, 2)), (0, 0, 0), 1e-3)
    m = em.read_mariepy_maps(tmp_path / "q.npz")
    d = em.quadrature_drive(m)
    # drive (1, j): B = x + j y, clockwise (see above), minus = 1 + j (-j) = 2, |B1+| = 1
    assert torch.allclose(d, torch.tensor([1, 1j], dtype=c128))
    assert torch.allclose(em.transmit_b1(m, d).abs(), torch.ones(2, 2, 2, dtype=torch.float64))


def test_quasi_static_limit_matches_loops(tmp_path):
    # real (in-phase) loop fields as phasors: em must reproduce loops.field_maps exactly
    loops = [
        Loop((0.08, 0, 0), (-1, 0, 0), 0.04),
        Loop((0, 0.08, 0.01), (0, -1, 0), 0.04),
        Loop((-0.06, -0.05, -0.01), (1, 1, 0), 0.035),
    ]
    origin, h, shape = (-0.03, -0.025, -0.02), 0.005, (9, 11, 13)  # (nz, ny, nx)
    ref = (0.0, 0.0, 0.0)
    expect = loop_maps(loops, origin, (h,) * 3, shape, reference=ref)
    nz, ny, nx = shape
    axes = [origin[i] + h * torch.arange(n, dtype=torch.float64) for i, n in enumerate((nx, ny, nz))]
    xx, yy, zz = torch.meshgrid(*axes, indexing="ij")  # mariepy order (x, y, z)
    B, _ = fields(loops, torch.stack([xx, yy, zz], -1).reshape(-1, 3))
    B = B.reshape(3, nx, ny, nz, 3)
    plus, minus = torch.complex(B[..., 0], B[..., 1]), torch.complex(B[..., 0], -B[..., 1])
    write_maps(tmp_path / "l.npz", plus, minus, np.ones((nx, ny, nz)), origin, h)
    m = em.read_mariepy_maps(tmp_path / "l.npz")
    got = em.field_maps(m, m, drive=torch.ones(3, dtype=c128), reference=ref)
    assert got.b1.shape == expect.b1.shape and got.coils.shape == expect.coils.shape
    assert torch.allclose(got.b1, expect.b1.to(got.b1.dtype), rtol=1e-5)
    assert torch.allclose(got.coils, expect.coils.to(c128), rtol=1e-5, atol=1e-6)


def test_orientation_and_fill(tmp_path):
    # |B1+| = 1 + x / 0.1 on a body that is the half x < 0.01: sampled at (x, y, z) inside it is
    # exact; outside it the edge value continues rather than dropping to zero
    nx, ny, nz, h = 8, 5, 4, 0.01
    origin = (-0.03, -0.02, -0.015)
    x = origin[0] + h * torch.arange(nx, dtype=torch.float64)
    mag = (1 + x / 0.1)[:, None, None].expand(nx, ny, nz)
    mask = (x < 0.01)[:, None, None].expand(nx, ny, nz)
    minus = (2 * mag * mask)[None].to(c128)  # B1+ = conj(minus) / 2
    plus = torch.zeros_like(minus)
    plus[0, :, 2, 1] = 1j  # a receive weight -j / 2 at a single y, z row
    plus = plus * mask
    write_maps(tmp_path / "o.npz", plus, minus, mask.numpy(), origin, h)
    m = em.read_mariepy_maps(tmp_path / "o.npz")
    fm = em.field_maps(m, m, drive=torch.ones(1, dtype=c128), reference=(0.0, 0.0, 0.0))
    assert fm.b1.shape == (nz, ny, nx) and fm.coils.shape == (1, nz, ny, nx)
    o, sp = torch.tensor(origin, dtype=torch.float64), torch.full((3,), h, dtype=torch.float64)
    pts = torch.tensor([[-0.025, 0.003, -0.004], [0.004, -0.017, 0.01], [0.035, 0.0, 0.0]], dtype=torch.float64)
    got = sample(fm.b1[None], o, sp, pts)[:, 0]
    want = 1 + torch.clamp(pts[:, 0], max=0.0) / 0.1  # edge voxel centre x = 0.0
    assert torch.allclose(got, want, atol=1e-6)
    c = fm.coils[0]
    assert abs(c[1, 2, 0] - (-1j)) < 1e-6 and abs(c[1, 2, nx - 1] - (-1j)) < 1e-6 and abs(c[0, 0, 0]) == 0


def test_noise_resistance_round_trip(tmp_path):
    r = np.array([[2.0, 0.3 + 0.1j], [0.3 - 0.1j, 1.5]])
    np.savez(tmp_path / "n.npz", resistance=r, metadata=json.dumps({"unit": "Ohm"}))
    got = em.read_noise_resistance(tmp_path / "n.npz")
    assert torch.allclose(got, torch.as_tensor(r, dtype=c128))
