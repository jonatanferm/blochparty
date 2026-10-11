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
                                deformation_jacobian, draw_tissues, random_deformation, sample_head, sulcal_width,
                                to_phantom)

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


@pytest.mark.skipif(not (DATA / "subject04.json").exists(), reason="BrainWeb subject 04 not fetched")
def test_realistic_head_options():
    """size="anthropometric": the head measures the drawn ANSUR II breadth and length (the random
    deformation adds a few mm), a normal adult brain; "anthropometric_shell" the same size with a
    smaller brain and more skull; yaw_sd turns it without changing volumes;
    age opens the closed sulci ("measured": Jin's widths, more CSF; "conserved": the same CSF at 25 y,
    more with age); the defaults add none of these."""
    from blochparty.anatomy import head_dimensions

    base = sample_head(1, subject=4, root=DATA, voxel=(2e-3,) * 3)
    assert not {"size", "pose", "age"} & set(base.meta)
    sized = sample_head(1, subject=4, root=DATA, voxel=(2e-3,) * 3, size="anthropometric")
    b, l = head_dimensions(sized.anatomy)
    assert abs(b - sized.meta["size"]["breadth"]) < 8e-3 and abs(l - sized.meta["size"]["length"]) < 8e-3
    brain = float((sized.anatomy.fractions["gm"] + sized.anatomy.fractions["wm"]).sum()) * 8e-9  # m^3
    assert 0.9e-3 < brain < 1.4e-3
    kept = sample_head(1, subject=4, root=DATA, voxel=(2e-3,) * 3, size="anthropometric_shell")
    assert kept.meta["size"]["breadth"] == sized.meta["size"]["breadth"]  # the same draw
    b, l = head_dimensions(kept.anatomy)
    assert abs(b - kept.meta["size"]["breadth"]) < 8e-3 and abs(l - kept.meta["size"]["length"]) < 8e-3
    kept_brain = float((kept.anatomy.fractions["gm"] + kept.anatomy.fractions["wm"]).sum()) * 8e-9
    assert 0.9e-3 < kept_brain < brain  # the shell takes its share of the size
    skull = lambda h: float((h.anatomy.fractions["skull"] + h.anatomy.fractions["marrow"]).sum())
    assert skull(kept) > 1.03 * skull(sized)  # the vault thicker; the base, beyond reach, scales
    turned = sample_head(1, subject=4, root=DATA, voxel=(2e-3,) * 3, size="anthropometric", yaw_sd=10.)
    assert turned.meta["size"] == sized.meta["size"] and abs(turned.meta["pose"]["yaw_deg"]) > 0
    for k in ("gm", "wm", "csf"):
        assert float(turned.anatomy.fractions[k].sum()) == pytest.approx(float(sized.anatomy.fractions[k].sum()), rel=0.02)
    old = sample_head(1, subject=4, root=DATA, voxel=(2e-3,) * 3, size="anthropometric", age=70., sulci="measured")
    mean, sd = sulcal_width(70.)
    assert abs(old.meta["age"]["sulcal_width"] - mean) < 4 * sd and 50e-6 < old.meta["age"]["sulcal_csf"] < 250e-6
    assert float(old.anatomy.fractions["csf"].sum()) > float(sized.anatomy.fractions["csf"].sum())
    assert float(old.anatomy.fractions["gm"].sum()) < float(sized.anatomy.fractions["gm"].sum())
    # conserved: the cortex's diffuse CSF gathered into the sheets, the CSF volume kept at 25 y, more at 70 y
    csf = lambda h: float(h.anatomy.fractions["csf"].sum())
    young = sample_head(1, subject=4, root=DATA, voxel=(2e-3,) * 3, size="anthropometric", age=25.)
    assert young.meta["age"]["sulci"] == "conserved" and young.meta["age"]["gathered_csf"] > 5e-6
    assert csf(young) == pytest.approx(csf(sized), rel=2e-3) and young.meta["age"]["sulcal_csf"] == pytest.approx(0, abs=1e-7)
    aged = sample_head(1, subject=4, root=DATA, voxel=(2e-3,) * 3, size="anthropometric", age=70.)
    assert csf(aged) > csf(young) * 1.05


@pytest.mark.skipif(not (DATA / "subject04.json").exists(), reason="BrainWeb subject 04 not fetched")
def test_deep_grey_nuclei():
    """deep_grey: the four nuclei become classes with adult-like volumes (MNI-sized model, both sides:
    caudate and putamen ~7 and ~10 mL; BrainWeb labels much of the thalamus and pallidum white
    matter, which stays), drawn from their own stream (every other class's draw unchanged); a table
    without them refuses."""
    from blochparty import tissuedb
    from blochparty.anatomy import DEEP_GREY

    table = {3.: tissuedb.tissue_table(3.)}
    with pytest.raises(ValueError, match="deep_grey"):
        sample_head(1, subject=4, root=DATA, voxel=(2e-3,) * 3, deep_grey=True)
    plain = sample_head(1, subject=4, root=DATA, voxel=(2e-3,) * 3, table=table)
    deep = sample_head(1, subject=4, root=DATA, voxel=(2e-3,) * 3, table=table, deep_grey=True)
    assert not set(DEEP_GREY) & set(plain.anatomy.fractions) and not set(DEEP_GREY) & set(plain.tissues)
    assert all(deep.meta["draws"][k] == v for k, v in plain.meta["draws"].items())
    assert deep.meta["deep_grey"] == list(DEEP_GREY)
    ml = {k: float(deep.anatomy.fractions[k].sum()) * 8e-3 for k in DEEP_GREY}
    assert 5 < ml["caudate"] < 9 and 7 < ml["putamen"] < 12 and 3 < ml["thalamus"] < 10 and 0 < ml["pallidum"] < 2
    grey = lambda h: sum(float(h.anatomy.fractions[k].sum()) for k in ("gm", *DEEP_GREY) if k in h.anatomy.fractions)
    assert grey(deep) == pytest.approx(grey(plain), rel=1e-5)
    assert deep.tissues["putamen"].t2.mean < deep.tissues["gm"].t2.mean


def test_shell_resize_keeps_the_skull_thickness():
    """A ball of brain (radius 50 mm) in a 7 mm skull, resized by 0.8 about its centre through the
    shell map (sample_head's pull-back for "anthropometric_shell"): the brain's radius scales as
    under plain resizing, the skull keeps its 7 mm (plain resizing: 5.6 mm), to half a voxel."""
    from blochparty.anatomy import _sample_field, _shell_translation

    h, n = 1e-3, 141
    r = torch.linalg.norm(torch.stack(torch.meshgrid(*(torch.arange(n, dtype=torch.float64) * h - .07,) * 3,
                                                     indexing="ij")), dim=0)
    brain = (r < .05).double()
    skull = ((r >= .05) & (r < .057)).double()
    an = Anatomy({"background": 1 - brain - skull, "wm": brain, "skull": skull}, (h,) * 3, (0., 0., 0.))
    c, s = (.07,) * 3, (.8,) * 3
    shell = _shell_translation(an, s, c)
    grid = torch.stack(torch.meshgrid(*(torch.arange(n, dtype=torch.float64) * h,) * 3, indexing="ij"))
    cc = torch.tensor(c, dtype=torch.float64).view(3, 1, 1, 1)
    v = grid - cc
    u = v / .8
    for _ in range(6):
        u = v - _sample_field(shell, cc + u, an.voxel).double()
    out = deform(an, cc + u - grid)
    plain = deform(an, cc + v / .8 - grid)
    line = lambda a, k: float(a.fractions[k][70, 70, 70:].sum()) * h  # along +x from the centre
    assert line(out, "wm") == pytest.approx(line(plain, "wm"), abs=5e-4)  # scaled like plain resizing
    assert line(out, "skull") == pytest.approx(line(an, "skull"), abs=5e-4)  # kept
    assert line(plain, "skull") == pytest.approx(.8 * line(an, "skull"), abs=6e-4)  # scaled


@pytest.mark.skipif(not (DATA / "subject04.json").exists(), reason="BrainWeb subject 04 not fetched")
def test_ventricles_grow_with_age():
    """ventricle_age: subject 04's lateral ventricles (MNI region, 13 mL at 1 mm) scale by Yamada
    2023's age ratio (1 at 30, 1.93 at 70); the head's CSF grows and its brain shrinks by about the
    same volume; 25 years changes nothing; "age" takes the head's age (and needs one)."""
    from blochparty.anatomy import load_brainweb, ventricle_factor, ventricle_mask

    an = load_brainweb(4, DATA, voxel=(2e-3,) * 3)
    assert 4e-6 < float((an.fractions["csf"] * torch.from_numpy(ventricle_mask(an))).sum()) * 8e-9 < 20e-6
    assert ventricle_factor(30.) == 1. and ventricle_factor(50.) == pytest.approx(19.4 / 17.3)
    assert ventricle_factor(70.) == ventricle_factor(85.) == pytest.approx(33.4 / 17.3)
    kw = dict(subject=4, root=DATA, voxel=(2e-3,) * 3, magnitude=0.)
    young, old = sample_head(1, ventricle_age=25., **kw), sample_head(1, ventricle_age=70., **kw)
    plain = sample_head(1, **kw)
    assert young.meta["ventricles"]["factor"] == 1. and torch.equal(young.anatomy.fractions["csf"], plain.anatomy.fractions["csf"])
    csf = lambda h: float(h.anatomy.fractions["csf"].sum()) * 8e-9
    brain = lambda h: float((h.anatomy.fractions["gm"] + h.anatomy.fractions["wm"]).sum()) * 8e-9
    gain = csf(old) - csf(plain)
    assert 0.6 * 0.93 * old.meta["ventricles"]["source_volume"] < gain < 1.4 * 0.93 * old.meta["ventricles"]["source_volume"]
    assert brain(plain) - brain(old) == pytest.approx(gain, rel=.5)
    one = sample_head(1, age=(20., 80.), ventricle_age="age", **kw)  # the head's own age for both
    assert one.meta["ventricles"]["years"] == one.meta["age"]["years"]
    with pytest.raises(ValueError):
        sample_head(1, ventricle_age="age", **kw)


@pytest.mark.skipif(not (DATA / "subject04.json").exists(), reason="BrainWeb subject 04 not fetched")
def test_pitch_tips_the_face_down_and_keeps_volumes():
    """pitch=(20, 0): nose down, so the face's most anterior point rises to the forehead and moves
    back; the pose records the angle; tissue volumes are kept (a rotation); no pitch draws nothing."""
    import numpy as np
    from blochparty.anatomy import _head_mask

    kw = dict(subject=4, root=DATA, voxel=(2e-3,) * 3, magnitude=0., size="anthropometric")
    flat, down = sample_head(1, **kw), sample_head(1, pitch=(20., 0.), **kw)
    assert "pitch_deg" not in flat.meta["pose"] and down.meta["pose"]["pitch_deg"] == pytest.approx(20.)
    front = lambda h: (lambda z, y: (z[y >= y.max() - 1].mean(), y.max()))(*np.nonzero(_head_mask(h.anatomy))[:2])
    (z0, y0), (z1, y1) = front(flat), front(down)
    assert z1 > z0 and y1 < y0
    for k in ("gm", "wm", "csf"):
        assert float(down.anatomy.fractions[k].sum()) == pytest.approx(float(flat.anatomy.fractions[k].sum()), rel=.02)


def _fake_ixi(root, ixi_id=7, scale=0.9):
    """A prepared IXI head on a small grid: nested spheres of BrainWeb's classes, as prepare_ixi.py writes."""
    import numpy as np

    classes = ("background", "csf", "gm", "wm", "fat", "muscle", "skin", "skull", "vessels", "connective", "dura", "marrow")
    n, h = 40, 4e-3
    z, y, x = np.meshgrid(*(np.arange(n) - n / 2 + .5,) * 3, indexing="ij")
    r = np.sqrt(z ** 2 + y ** 2 + x ** 2)
    frac = np.zeros((len(classes), n, n, n), np.float32)
    for name, (lo, hi) in {"wm": (0, 8), "gm": (8, 11), "csf": (11, 13), "skull": (13, 15), "skin": (15, 18)}.items():
        frac[classes.index(name)][(r >= lo) & (r < hi)] = 1
    frac[0][r >= 18] = 1
    origin = (-n / 2 * h,) * 3
    meta = {"ixi_id": ixi_id, "age": 55.0, "sex": "f", "classes": classes,
            "affine_mm": [[scale, 0, 0, 0], [0, scale, 0, 0], [0, 0, scale, 0], [0, 0, 0, 1]],
            "lncc": 0.2, "volumes_ml": {}, "origin": list(origin), "voxel": [h] * 3, "shape": [n] * 3}
    (root / "ixi" / "prepared").mkdir(parents=True)
    np.savez_compressed(root / "ixi" / "prepared" / f"IXI{ixi_id:03d}.npz", fractions=np.round(frac * 255).astype(np.uint8),
                        meta=json.dumps(meta))


def test_ixi_heads(tmp_path):
    """source="ixi": a prepared IXI head loads with its meta, sample_head draws it, size="native" restores its own
    size from the registration's scale (0.9 here: the brain shrinks ~0.73 in volume), and age= / ventricle_age=
    / size="native" for BrainWeb are refused."""
    import numpy as np
    from blochparty.anatomy import ixi_subjects, load_ixi

    _fake_ixi(tmp_path)
    assert ixi_subjects(tmp_path / "ixi") == [7]
    an, meta = load_ixi(7, tmp_path / "ixi")
    total = sum(an.fractions.values())
    assert meta["age"] == 55.0 and float((total - 1).abs().max()) < 1e-5
    kw = dict(root=tmp_path, voxel=(4e-3,) * 3, magnitude=0.)
    head = sample_head(3, source="ixi", **kw)
    assert head.meta["source"] == "ixi" and head.meta["ixi"]["ixi_id"] == 7 and head.meta["subject"] == 7
    native = sample_head(3, source="ixi", size="native", **kw)
    assert native.meta["size"]["factors"] == pytest.approx([0.9] * 3)
    brain = lambda h: float((h.anatomy.fractions["gm"] + h.anatomy.fractions["wm"]).sum())
    assert brain(native) / brain(head) == pytest.approx(0.9 ** 3, rel=0.1)
    with pytest.raises(ValueError):
        sample_head(3, source="ixi", age=50., **kw)
    with pytest.raises(ValueError):
        sample_head(3, size="native", root=tmp_path, voxel=(4e-3,) * 3)
