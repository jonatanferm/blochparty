"""Fetch and inspect the four pinned AHEAD v3 median maps; no model conversion.

Source: 10.21942/uva.19646364.v3 (CC BY 4.0). These are brain templates at
7 T, not the QSM challenge full-head phantom. Relative PD is not absolute M0.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from blochparty.anatomy import _file_sha256

HERE = Path(__file__).resolve().parent


def compressed_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(root, *, offline=False):
    manifest = json.loads((HERE / "ahead_source.json").read_text())
    root.mkdir(parents=True, exist_ok=True)
    report = {}
    reference = None
    for role, entry in manifest["files"].items():
        path = root / entry["path"]
        if not path.exists():
            if offline:
                raise FileNotFoundError(path)
            part = path.with_suffix(".part.gz")
            subprocess.run([
                "curl", "-fsSL", "--retry", "2", "--max-time", "300",
                entry["url"], "-o", str(part),
            ], check=True)
            if compressed_sha256(part) != entry["sha256"]:
                raise ValueError(f"download checksum mismatch: {part}")
            part.replace(path)
        if compressed_sha256(path) != entry["sha256"]:
            raise ValueError(f"checksum mismatch: {path}; do not silently refresh pins")
        if _file_sha256(path) != entry["decompressed_sha256"]:
            raise ValueError(f"decompressed checksum mismatch: {path}")
        image = nib.load(path)
        if reference is not None and (
            image.shape != reference.shape
            or not np.array_equal(image.affine, reference.affine)
        ):
            raise ValueError(f"map grid mismatch: {path}")
        reference = image
        values = np.asarray(image.dataobj)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite data: {path}")
        if role != "chi" and np.any(values < 0):
            raise ValueError(f"negative density or rate: {path}")
        report[role] = {
            "shape_xyz": list(image.shape),
            "axis_codes": list(nib.aff2axcodes(image.affine)),
            "spacing": [float(v) for v in image.header.get_zooms()],
            "header_units": list(image.header.get_xyzt_units()),
            "affine": image.affine.tolist(),
            "range": [float(values.min()), float(values.max())],
            "nonzero_voxels": int(np.count_nonzero(values)),
        }
        print(f"Verified {role}: {report[role]}", flush=True)
    (root / "verified_source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE / "data" / "ahead")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    fetch(args.root, offline=args.offline)
