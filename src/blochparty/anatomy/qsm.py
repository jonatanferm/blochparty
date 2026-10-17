"""The QSM Challenge 2.0 head (susceptibility, segmentation) and the SHA-256 pinning of downloaded files."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .grid import Anatomy

# Label IDs published with the QSM Challenge 2.0 head model; independently
# listed in qsm-forward/label.json, commit 92149383b0f43f47d9a351ceb4f525c0c4ee8299.
QSM_LABELS = {1: "caudate", 2: "globus_pallidus", 3: "putamen", 4: "red_nucleus",
              5: "dentate_nucleus", 6: "substantia_nigra", 7: "thalamus", 8: "wm",
              9: "gm", 10: "csf", 11: "vessels", 12: "fat", 13: "skull",
              14: "background", 15: "muscle", 16: "calcification"}


def _file_sha256(path):
    """Hash decompressed bytes so a gzip header change does not change identity."""
    digest = hashlib.sha256()
    with (gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")) as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_qsm2(root, resolution="1mm", *, dtype=torch.float32) -> Anatomy:
    """Load locally supplied QSM Challenge 2.0 maps using root/qsm2.json pins.

    Optional dependency: pip install 'blochparty[anatomy]'. The full-head source
    directory (chimodel, maps, masks) must be obtained under its access terms;
    validation/anatomy/prepare_qsm2.py records a reviewed local manifest. This
    function never downloads or authenticates. Both chi models can be selected
    by preparing distinct manifests/roots; no filename guessing at load time.

    NIfTI arrays are reoriented to positive RAS axes without interpolation, then
    transposed to z/y/x. Axis-aligned permutations/flips are supported; oblique
    or sheared grids raise rather than silently drop their affine. All maps must
    share a grid after reorientation. Spatial units are declared in the manifest
    (and checked against the header if set). Chi is converted from ppm to SI,
    retaining its original reference and explicit surrounding chi_background.

    ``resolution`` is 'native', '0.64mm', '1mm', or a (dz,dy,dx) tuple in m;
    non-native grids use conservative cell averages, not k-space truncation.
    A manifest with kind='challenge' loads the public challenge ground truth
    (chi and segmentation only): background susceptibility is zero, no measured
    relaxation/density maps are invented, and its brain mask is derived from
    labels 1..11 and 16. It is distinct from the full-head source package.

    Sixteen tissue memberships are retained; original background label 0 joins
    air label 14. M0, R1, R2star and brain_mask are retained as maps. R1/R2star
    belong to the manifest's acquisition field (normally 7 T), not automatically
    to the requested simulation field. There is no measured R2 map in this
    release: full Bloch assembly needs an explicit tissue table for T2, including
    the deep-grey labels. Observed R2star must still follow to_phantom's
    macroscopic/intrinsic separation policy; it is never silently used as R2.
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("load_qsm2 requires the anatomy extra: pip install 'blochparty[anatomy]'") from exc
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be float32 or float64")
    root = Path(root).resolve()
    manifest = json.loads((root/"qsm2.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("chi_unit") != "ppm":
        raise ValueError("unsupported QSM manifest schema or chi units")
    kind = manifest.get("kind", "full-head")
    if kind not in ("full-head", "challenge"):
        raise ValueError("unknown QSM package kind")
    unit = manifest.get("spatial_unit")
    units = {"mm": .001, "meter": 1., "micron": 1e-6}
    if unit not in units:
        raise ValueError("manifest needs an explicit NIfTI spatial_unit")
    scale = units[unit]
    background = float(manifest["chi_background_ppm"])*1e-6
    map_field = float(manifest["map_field_T"]) if kind == "full-head" else None
    arrays = {}
    affine = shape = None
    keys = ("chi", "M0", "R1", "R2star", "segmentation", "brain_mask") if kind == "full-head" else ("chi", "segmentation")
    for key in keys:
        entry = manifest["files"][key]
        path = (root/entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("QSM manifest paths must stay inside root")
        if _file_sha256(path) != entry["sha256"]:
            raise ValueError(f"checksum mismatch: {path}")
        img = nib.load(path)
        header_unit = img.header.get_xyzt_units()[0]
        if header_unit not in ("unknown", unit):
            raise ValueError(f"NIfTI/manifest spatial unit mismatch: {key}")
        img = nib.as_closest_canonical(img)
        mat = img.affine.copy()
        if len(img.shape) != 3 or min(img.shape) < 1 or not np.isfinite(mat).all():
            raise ValueError("QSM maps must have a finite affine and nonempty 3D grid")
        linear = mat[:3, :3]
        if (np.diag(linear) <= 0).any() or not np.allclose(linear, np.diag(np.diag(linear)), rtol=0, atol=1e-6):
            raise ValueError("oblique/sheared QSM grids need explicit common-grid resampling first")
        if affine is None:
            affine, shape = mat, img.shape
        elif img.shape != shape or not np.allclose(mat, affine, rtol=0, atol=1e-5):
            raise ValueError(f"QSM maps do not share a physical grid: {key}")
        data = img.get_fdata(dtype=np.float64 if dtype == torch.float64 else np.float32)
        arrays[key] = torch.from_numpy(np.ascontiguousarray(data.transpose(2,1,0)))
        if not torch.isfinite(arrays[key]).all():
            raise ValueError(f"non-finite QSM map: {key}")
    labels = arrays.pop("segmentation")
    if (labels != labels.round()).any() or (labels < 0).any() or (labels > 16).any():
        raise ValueError("unknown QSM segmentation labels; expected integers 0..16")
    if kind == "full-head":
        if (arrays["M0"] < 0).any() or (arrays["brain_mask"] < 0).any() or (arrays["brain_mask"] > 1).any():
            raise ValueError("QSM M0 must be nonnegative and brain mask in [0,1]")
        for key in ("R1", "R2star"):
            if (arrays[key] < 0).any() or (arrays[key][arrays["M0"] > 0] <= 0).any():
                raise ValueError(f"QSM {key} must be positive where M0 is nonzero")
    else:
        if background != 0:
            raise ValueError("challenge brain-only chi requires zero exterior susceptibility")
        arrays["brain_mask"] = (((labels >= 1) & (labels <= 11)) | (labels == 16)).to(dtype)
    fractions = {name: (labels == label).to(dtype) for label,name in QSM_LABELS.items()
                 if (labels == label).any()}
    fractions["background"] = ((labels == 0) | (labels == 14)).to(dtype)
    chi = arrays.pop("chi")*1e-6
    voxel = tuple((np.diag(affine)[:3]*scale)[::-1])
    origin = tuple(affine[:3,3]*scale)
    anatomy = Anatomy(fractions,voxel,origin,chi,arrays,background,map_field)
    if isinstance(resolution, str):
        options = {"native": None, "0.64mm": (.00064,)*3, "1mm": (.001,)*3}
        if resolution not in options:
            raise ValueError("resolution must be native, 0.64mm, 1mm, or a voxel tuple")
        resolution = options[resolution]
    return anatomy if resolution is None else anatomy.resample(resolution)
