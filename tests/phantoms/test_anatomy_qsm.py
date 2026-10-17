"""QSM format fixtures; no restricted data or network access in unit tests."""

import json
import math

import numpy as np
import pytest
import torch

import blochparty as bp
from blochparty.anatomy import _file_sha256

nib = pytest.importorskip("nibabel")


def fixture(root, *, orientation="ras", invalid=None):
    shape = (4, 5, 6)  # source x/y/z
    labels = np.full(shape, 14, dtype=np.float32)
    labels[1:3, 1:4, 1:5] = 8
    labels[1, 2, 2] = 1
    labels[2, 3, 3] = 9
    labels[0, 0, 0] = 0
    mask = (labels != 14) & (labels != 0)
    chi = np.where(mask, np.arange(math.prod(shape)).reshape(shape) / 1000, 9.2).astype(np.float32)
    maps = dict(
        chi=chi,
        M0=mask.astype(np.float32) * 0.8,
        R1=mask.astype(np.float32) * 1.2,
        R2star=mask.astype(np.float32) * 35,
        segmentation=labels,
        brain_mask=mask.astype(np.float32),
    )
    if invalid == "labels":
        maps["segmentation"][1, 1, 1] = 17
    if invalid == "rate":
        maps["R1"][1, 1, 1] = 0
    if invalid == "nan":
        maps["chi"][0, 0, 0] = np.nan
    manifest = dict(schema=1, chi_unit="ppm", chi_background_ppm=9.2, spatial_unit="mm", map_field_T=7.0, files={})
    for key, array in maps.items():
        affine = np.diag([0.64, 0.64, 0.64, 1.0])
        affine[:3, 3] = [-1, -2, -3]
        if orientation == "permuted":
            # New axes are -z, +x, +y. Hand-written affine preserves every
            # physical voxel centre; no nibabel orientation inverse in fixture.
            array = array.transpose(2, 0, 1)[::-1].copy()
            affine = np.array([[0, 0.64, 0, -1], [0, 0, 0.64, -2], [-0.64, 0, 0, -3 + 5 * 0.64], [0, 0, 0, 1.0]])
        if invalid == "oblique":
            angle = 0.2
            rotation = np.array(
                [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
            )
            affine[:3, :3] = rotation @ affine[:3, :3]
        if invalid == "grid" and key == "M0":
            affine[0, 3] += 0.1
        path = root / f"{key}.nii.gz"
        image = nib.Nifti1Image(array, affine)
        image.header.set_xyzt_units("meter" if invalid == "unit" else "mm")
        nib.save(image, path)
        manifest["files"][key] = dict(path=path.name, sha256=_file_sha256(path))
    (root / "qsm2.json").write_text(json.dumps(manifest))
    return maps


@pytest.mark.parametrize("orientation", ["ras", "permuted"])
def test_qsm_orientation_units_and_measured_maps(tmp_path, orientation):
    source = fixture(tmp_path, orientation=orientation)
    a = bp.load_qsm2(tmp_path, resolution="native", dtype=torch.float64)
    assert a.shape == (6, 5, 4)
    assert a.voxel == pytest.approx((0.00064,) * 3, abs=1e-10)
    assert a.origin == pytest.approx((-0.001, -0.002, -0.003), abs=1e-10)
    assert a.chi_background == pytest.approx(9.2e-6)
    assert a.map_field_T == 7.0
    for name in ("M0", "R1", "R2star", "brain_mask"):
        torch.testing.assert_close(a.maps[name], torch.from_numpy(source[name].transpose(2, 1, 0).copy()).double())
    torch.testing.assert_close(a.chi, torch.from_numpy(source["chi"].transpose(2, 1, 0).copy()).double() * 1e-6)
    assert a.fractions["caudate"].sum() == 1
    assert a.fractions["gm"].sum() == 1
    torch.testing.assert_close(sum(a.fractions.values()), torch.ones(a.shape, dtype=torch.float64))
    assert "R2" not in a.maps  # measured R2* is not irreversible R2


def test_qsm_conservative_resolution_and_metadata(tmp_path):
    fixture(tmp_path)
    a = bp.load_qsm2(tmp_path, resolution="native", dtype=torch.float64)
    b = bp.load_qsm2(tmp_path, resolution="1mm", dtype=torch.float64)
    for name in a.fractions:
        assert float(b.fractions[name].sum() * math.prod(b.voxel)) == pytest.approx(
            float(a.fractions[name].sum() * math.prod(a.voxel)), rel=1e-12
        )
    assert float(b.maps["M0"].sum() * math.prod(b.voxel)) == pytest.approx(
        float(a.maps["M0"].sum() * math.prod(a.voxel)), rel=1e-12
    )
    assert b.chi_background == a.chi_background
    assert b.map_field_T == 7.0
    crop = b.crop((slice(0, 1), slice(None), slice(None)))
    assert crop.chi_background == a.chi_background and crop.map_field_T == 7.0
    # Air contributes no relaxation-weight signal; coarsening must not lengthen T1.
    assert torch.allclose(b.maps["R1"][b.maps["M0"] > 0], torch.tensor(1.2, dtype=torch.float64), atol=1e-7)


@pytest.mark.parametrize(
    "invalid,match",
    [
        ("labels", "labels"),
        ("rate", "R1"),
        ("nan", "non-finite"),
        ("grid", "physical grid"),
        ("unit", "unit"),
        ("oblique", "oblique"),
    ],
)
def test_qsm_rejects_ambiguous_or_invalid_inputs(tmp_path, invalid, match):
    fixture(tmp_path, invalid=invalid)
    with pytest.raises(ValueError, match=match):
        bp.load_qsm2(tmp_path, resolution="native")


def test_qsm_checksum_and_path_validation(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "qsm2.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["chi"]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        bp.load_qsm2(tmp_path)
    manifest["files"]["chi"]["path"] = "../outside.nii"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inside root"):
        bp.load_qsm2(tmp_path)


def test_challenge_package_does_not_invent_relaxation_maps(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "qsm2.json"
    manifest = json.loads(path.read_text())
    manifest.update(kind="challenge", chi_background_ppm=0)
    manifest.pop("map_field_T")
    manifest["files"] = {k: v for k, v in manifest["files"].items() if k in ("chi", "segmentation")}
    path.write_text(json.dumps(manifest))
    a = bp.load_qsm2(tmp_path, resolution="native")
    assert a.map_field_T is None and a.chi_background == 0
    assert set(a.maps) == {"brain_mask"}
    assert a.maps["brain_mask"].sum() == 24


def test_qsm_reference_background_and_default_mask(tmp_path):
    fixture(tmp_path)
    a = bp.load_qsm2(tmp_path, resolution="native", dtype=torch.float64)
    tissue, scanner = bp.anatomy_field_maps(a, field_T=3.0, shim_order=0)
    mask = a.maps["brain_mask"] > 0.5
    assert abs((tissue.b0 + scanner.b0)[mask].mean()) < 1e-10
    # Changing susceptibility reference including exterior cannot change B0.
    original = tissue.b0.clone()
    a.chi += 2e-6
    a.chi_background += 2e-6
    shifted, _ = bp.anatomy_field_maps(a, field_T=3.0, shim_order=None)
    torch.testing.assert_close(shifted.b0, original, atol=1e-10, rtol=0)
    # Explicit chi works without invented relaxation properties for deep GM.
    assert "caudate" not in bp.DEFAULT_TISSUES[3.0]


def test_measured_density_and_rates_do_not_double_count_partial_volume():
    fractions = {
        "wm": torch.tensor([[[0.25, 0.5, 0.0]]], dtype=torch.float64),
        "gm": torch.tensor([[[0.25, 0.25, 0.0]]], dtype=torch.float64),
    }
    density = torch.tensor([[[0.4, 0.6, 0.0]]], dtype=torch.float64)
    rate = torch.tensor([[[1.0, 1.0, 0.0]]], dtype=torch.float64)
    a = bp.Anatomy(fractions, (0.001,) * 3, (0, 0, 0), maps={"M0": density, "R1": rate}, map_field_T=7.0)
    with pytest.raises(ValueError, match="acquired at 7"):
        bp.to_phantom(a, chi_to_b0=False, t2star="off")
    ph = bp.to_phantom(a, field_T=7.0, chi_to_b0=False, t2star="off", spins_per_voxel=8)
    assert ph.pd.sum() == pytest.approx(1.0, abs=1e-12)
    assert torch.equal(ph.t1, torch.ones_like(ph.t1))
    coarse = a.resample((0.001, 0.001, 0.002))
    ph = bp.to_phantom(coarse, field_T=7.0, chi_to_b0=False, t2star="off", spins_per_voxel=8)
    assert ph.pd.sum() * 2 == pytest.approx(1.0, abs=1e-12)
    assert torch.equal(ph.t1, torch.ones_like(ph.t1))
