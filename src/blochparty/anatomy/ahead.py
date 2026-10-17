"""The AHEAD 7 T quantitative head (R1, R2*, susceptibility, PD)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .grid import Anatomy
from .qsm import _file_sha256

# Units of the AHEAD v3 templates: the publisher record does not state them. ppm and 1/s
# follow from the values (WM R1 ~0.8 /s and R2* ~ 35 /s at 7 T match Caan et al. 2019, same
# MP2RAGEME method; pallidal QSM ~ +0.1 is ppm, not ppb); checked in validation/anatomy/anatomy_ahead.py.
AHEAD_UNITS = {"chi": 1e-6, "R1": 1.0, "R2star": 1.0, "relative_PD": 1.0}


def load_ahead(root, resolution="1mm", *, dtype=torch.float32) -> Anatomy:
    """Load the AHEAD v3 7 T population-median brain templates (Bazin et al.; figshare
    doi:10.21942/uva.19646364.v3, CC BY 4.0) fetched by validation/anatomy/fetch_ahead.py into
    ``root`` (pins: root/verified_source.json, decompressed SHA-256 checked here).

    A brain-only template: one tissue class ``"brain"`` (where R1 > 0 and PD > 0; QSM can be
    zero inside tissue, so it does not define the mask), maps ``M0`` (relative PD, not absolute
    density), ``R1``, ``R2star`` [1/s] and ``brain_mask``, and ``chi`` (QSM, converted ppm -> SI,
    relative to its own reference, so ``chi_background = 0``: no air or bone around it).
    ``map_field_T = 7``: simulating at another field raises (no field scaling). There is no
    R2 map: ``to_phantom`` needs a table entry ``"brain"`` at 7 T giving T2 (and the T2* policy
    as for other measured R2star maps). Stored RPS; reoriented to RAS without interpolation and
    transposed to (z, y, x). ``resolution``: 'native' (0.5 mm), '1mm' (default, conservative
    cell averages) or a (dz, dy, dx) tuple [m]. Optional dependency: nibabel (``[anatomy]``).
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("load_ahead requires the anatomy extra: pip install 'blochparty[anatomy]'") from exc
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be float32 or float64")
    root = Path(root).resolve()
    manifest = json.loads((root/"verified_source.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("doi") != "10.21942/uva.19646364.v3":
        raise ValueError("unexpected AHEAD manifest")
    arrays, affine = {}, None
    for key in ("chi", "R1", "R2star", "relative_PD"):
        entry = manifest["files"][key]
        path = (root/entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("AHEAD manifest paths must stay inside root")
        if _file_sha256(path) != entry["decompressed_sha256"]:
            raise ValueError(f"checksum mismatch: {path}")
        img = nib.as_closest_canonical(nib.load(path))
        mat = img.affine
        if img.header.get_xyzt_units()[0] != "mm" or not np.allclose(mat[:3, :3], np.diag(np.diag(mat[:3, :3])), atol=1e-6) \
                or (np.diag(mat)[:3] <= 0).any():
            raise ValueError(f"unexpected AHEAD grid: {key}")
        if affine is None:
            affine = mat
        elif not np.allclose(mat, affine, atol=1e-6):
            raise ValueError(f"AHEAD maps do not share a grid: {key}")
        data = img.get_fdata(dtype=np.float64 if dtype == torch.float64 else np.float32)
        arrays[key] = torch.from_numpy(np.ascontiguousarray(data.transpose(2, 1, 0))) * AHEAD_UNITS[key]
    pd, r1, r2s = arrays["relative_PD"], arrays["R1"], arrays["R2star"]
    mask = ((r1 > 0) & (pd > 0)).to(dtype)
    if (pd < 0).any() or (r1 < 0).any() or (r2s < 0).any():
        raise ValueError("AHEAD PD and rates must be nonnegative")
    voxel = tuple((np.diag(affine)[:3]*1e-3)[::-1])
    origin = tuple(affine[:3, 3]*1e-3)
    maps = {"M0": pd*mask, "R1": torch.where(mask > 0, r1, 0.), "R2star": torch.where(mask > 0, r2s, 0.), "brain_mask": mask}
    anatomy = Anatomy({"brain": mask}, voxel, origin, arrays["chi"]*mask, maps, 0.0, 7.0)
    if isinstance(resolution, str):
        options = {"native": None, "1mm": (.001,)*3}
        if resolution not in options:
            raise ValueError("resolution must be native, 1mm, or a voxel tuple")
        resolution = options[resolution]
    return anatomy if resolution is None else anatomy.resample(resolution)
