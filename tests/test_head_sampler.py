"""M5 container and sampler: the DigitalPhantom round trip (bit-identical to to_phantom), the
deformation (exact shifts, partition of unity, Jacobian > 0), tissue draws (means and SDs of the
table over 100 subjects) and sample_head's determinism on BrainWeb subject 04 (skipped without
the data)."""
from dataclasses import fields
import json
import math
from pathlib import Path

import pytest
import torch

from blochparty.anatomy import (DEFAULT_TISSUES, Anatomy, DigitalPhantom, _bspline_weights, deform,
                                deformation_jacobian, draw_tissues, random_deformation, sample_head, to_phantom)

DATA = Path(__file__).parents[1] / "validation" / "anatomy" / "data"


@pytest.mark.parametrize("options", [{}, dict(mt=True, diffusion=True, chi_to_b0=False, t2star="off"),
                                     dict(variability={"t1": .1, "length": .002, "seed": 3}, seed=5)])
def test_digital_phantom_round_trip_is_bit_identical(options):
    a = Anatomy({"wm": torch.full((6, 6, 6), .5, dtype=torch.float64), "gm": torch.full((6, 6, 6), .3, dtype=torch.float64),
                 "fat": torch.full((6, 6, 6), .1, dtype=torch.float64)}, (.002,) * 3, (0, 0, 0))
    ours = DigitalPhantom(a, DEFAULT_TISSUES[3.], 3., options).to_phantom()
    ref = to_phantom(a, DEFAULT_TISSUES, 3., **options)
    for key in ("pos", "pd", "t1", "t2", "db0", "diffusion"):
        x, y = getattr(ours, key), getattr(ref, key)
        assert (x is None and y is None) or torch.equal(x, y), key
    if ref.bound is not None:
        assert all(torch.equal(getattr(ours.bound, k), getattr(ref.bound, k)) for k in ("f", "k", "t1", "t2"))


def test_deform_shifts_exactly_and_keeps_the_identity():
    v = torch.zeros(4, 5, 6, dtype=torch.float64)
    v[1:3, 1:4, 2:5] = torch.rand(2, 3, 3, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    a = Anatomy({"wm": v, "gm": (1 - v) / 2}, (1e-3, 2e-3, 3e-3), (0, 0, 0), chi=v * 1e-6)
    d = torch.zeros(3, 4, 5, 6, dtype=torch.float64)
    assert torch.equal(deform(a, d).fractions["wm"], v)
    d[2] = 3e-3  # one voxel along +x: the new value at x is the old one at x + 1
    b = deform(a, d)
    assert torch.equal(b.fractions["wm"][..., :-1], v[..., 1:]) and not b.fractions["wm"][..., -1].any()
    assert b.chi[..., -1].eq(a.chi_background).all()  # outside the grid: air
    d[2], d[0] = 0, -2e-3  # two voxels along -z
    assert torch.equal(deform(a, d).fractions["gm"][2:], a.fractions["gm"][:2])


def test_bspline_partition_of_unity_and_jacobian():
    w = _bspline_weights(50, 2e-3, 40e-3)
    torch.testing.assert_close(w.sum(1), torch.ones(50, dtype=torch.float64), rtol=0, atol=1e-15)
    shape, voxel = (40, 48, 40), (4e-3,) * 3
    for seed in range(20):
        d, control = random_deformation(shape, voxel, torch.Generator().manual_seed(seed))
        jac = deformation_jacobian(d, voxel)
        assert jac.min() > .5 and jac.max() < 1.5, seed
    # an affine displacement d = A x has the Jacobian det(I + A) everywhere
    A = torch.tensor([[.1, .02, 0], [0, -.05, .03], [.01, 0, .07]], dtype=torch.float64)
    axes = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) * h for n, h in zip(shape, voxel)), indexing="ij")
    d = torch.einsum("kl,lzyx->kzyx", A, torch.stack(axes))
    torch.testing.assert_close(deformation_jacobian(d, voxel), torch.linalg.det(torch.eye(3, dtype=A.dtype) + A)
                               .expand(shape), rtol=1e-12, atol=0)


def test_draws_match_the_table_over_100_subjects():
    table = DEFAULT_TISSUES[3.]
    draws = [draw_tissues(table, torch.Generator().manual_seed(s))[1] for s in range(100)]
    assert draws[7] == draw_tissues(table, torch.Generator().manual_seed(7))[1]
    z = []
    for name, p in table.items():
        for f in fields(p):
            d = getattr(p, f.name)
            if d is None or not d.sd:
                assert all(f.name not in x[name] for x in draws)
                continue
            values = torch.tensor([x[name][f.name] for x in draws], dtype=torch.float64)
            assert abs(float(values.mean()) / d.mean - 1) < .05, (name, f.name)
            if f.name not in ("t2star",):  # clipped at T2
                z.append((values - d.mean) / d.sd)
    z = torch.cat(z)
    print(f"{len(z)} standardised draws: mean {float(z.mean()):+.3f}, SD {float(z.std()):.3f}")
    assert abs(float(z.mean())) < .05 and abs(float(z.std()) - 1) < .05
    tissues, _ = draw_tissues(table, torch.Generator().manual_seed(0))
    assert all(p.t2star is None or p.t2star.mean <= p.t2.mean for p in tissues.values())
    assert tissues["csf"].t1 == table["csf"].t1  # no SD: the mean, untouched


@pytest.mark.slow
def test_sample_head_is_deterministic_and_recorded():
    if not (DATA / "subject04.json").exists():
        pytest.skip("BrainWeb not fetched (validation/anatomy/fetch_brainweb.py)")
    a = sample_head(11, subject=4, root=DATA, voxel=(4e-3,) * 3)
    b = sample_head(11, subject=4, root=DATA, voxel=(4e-3,) * 3)
    c = sample_head(12, subject=4, root=DATA, voxel=(4e-3,) * 3)
    assert all(torch.equal(a.anatomy.fractions[k], b.anatomy.fractions[k]) for k in a.anatomy.fractions)
    assert json.dumps(a.meta) == json.dumps(b.meta) != json.dumps(c.meta)
    assert a.meta["deformation"]["jacobian"][0] > .2 and a.meta["subject"] == 4
    assert a.meta["draws"]["wm"]["t1"] == a.tissues["wm"].t1.mean
    ph = a.to_phantom(spins_per_voxel=1, sampling="categorical", t2star="off")
    assert len(ph) > 0 and math.isfinite(float(ph.db0.abs().max()))
