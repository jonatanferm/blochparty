"""AHEAD loader on synthetic files with the real files' layout (RPS storage, mm, 0.5 mm): world
positions survive the reorientation exactly, units and the R1/PD mask apply, checksums guard."""

import json

import numpy as np
import pytest
import torch

import blochparty as bp
from blochparty.anatomy import _file_sha256

nib = pytest.importorskip("nibabel")

FILES = {"chi": "q.nii.gz", "R1": "r1.nii.gz", "R2star": "r2.nii.gz", "relative_PD": "pd.nii.gz"}


def fixture(root, shape=(6, 8, 5)):
    """Stored x/y/z = (R, P, S): affine diag (0.5, -0.5, 0.5) mm as the real templates. Voxel
    (i, j, k) of every map holds a value that encodes its world position."""
    i, j, k = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    affine = np.diag([0.5, -0.5, 0.5, 1.0])
    affine[:3, 3] = [-1.0, 2.0, -1.5]
    world = np.stack([i, j, k, np.ones_like(i)], -1) @ affine.T  # mm
    inside = (i > 0) & (j > 0) & (k > 0)
    maps = {
        "chi": np.where(inside, world[..., 0] / 100, 0.0),  # ppm
        "R1": np.where(inside, 0.8 + world[..., 1] / 100, 0.0),
        "R2star": np.where(inside, 30 + world[..., 2], 0.0),
        "relative_PD": np.where(inside, 0.9, 0.0),
    }
    manifest = {"schema": 1, "doi": "10.21942/uva.19646364.v3", "files": {}}
    for key, name in FILES.items():
        img = nib.Nifti1Image(maps[key].astype(np.float32), affine)
        img.header.set_xyzt_units("mm")
        nib.save(img, root / name)
        manifest["files"][key] = {"path": name, "decompressed_sha256": _file_sha256(root / name)}
    (root / "verified_source.json").write_text(json.dumps(manifest))


def test_ahead_orientation_units_and_mask(tmp_path):
    fixture(tmp_path)
    a = bp.load_ahead(tmp_path, resolution="native", dtype=torch.float64)
    assert a.voxel == (5e-4,) * 3 and a.map_field_T == 7.0 and a.chi_background == 0.0
    nz, ny, nx = a.shape
    zz, yy, xx = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) for n in a.shape), indexing="ij")
    x, y, z = (
        (xx * a.voxel[2] + a.origin[0]) * 1e3,
        (yy * a.voxel[1] + a.origin[1]) * 1e3,
        (zz * a.voxel[0] + a.origin[2]) * 1e3,
    )
    m = a.maps["brain_mask"] > 0
    assert int(m.sum()) == 5 * 7 * 4
    assert torch.allclose(a.chi[m], x[m] / 100 * 1e-6)  # ppm -> SI, at its own world x
    assert torch.allclose(a.maps["R1"][m], 0.8 + y[m] / 100) and torch.allclose(a.maps["R2star"][m], 30 + z[m])
    assert torch.allclose(a.maps["M0"][m], torch.full_like(x[m], 0.9)) and float(a.maps["M0"][~m].abs().max()) == 0
    with pytest.raises(ValueError, match="acquired at 7"):
        bp.to_phantom(a, field_T=3.0)


def test_ahead_resampling_and_checksum(tmp_path):
    fixture(tmp_path)
    native = bp.load_ahead(tmp_path, resolution="native", dtype=torch.float64)
    coarse = bp.load_ahead(tmp_path, resolution=(1e-3,) * 3, dtype=torch.float64)
    vol = lambda a: float(a.maps["M0"].sum()) * np.prod(a.voxel)
    assert vol(coarse) == pytest.approx(vol(native), rel=1e-12)
    (tmp_path / "r1.nii.gz").write_bytes(b"")
    with pytest.raises(ValueError, match="checksum mismatch"):
        bp.load_ahead(tmp_path)
