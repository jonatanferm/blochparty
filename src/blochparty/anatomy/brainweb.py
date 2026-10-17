"""BrainWeb's twenty normal heads (fuzzy tissue models)."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .grid import Anatomy, _renormalize, _resample, _triple


def load_brainweb(subject: int, root, *, voxel=None, dtype=torch.float32) -> Anatomy:
    """Load the 20-subject fuzzy models from checksum-pinned raw-short gzip files.

    Run validation/anatomy/fetch_brainweb.py, or obtain the public raw-short files
    manually and provide a subjectNN.json manifest (see validation/anatomy/README.md).
    Subject IDs are 4,5,6,18,20,38,41..54, not consecutive 1..20. Native arrays
    are (362,434,362) at 0.5 mm, z/y/x, positive MINC axes; origin is read from
    the captured header in the manifest. The similarly named simulated T1 image
    has a different 1-mm grid. No flipping/transposing of these raw arrays is needed.

    Raw shorts are 0..4095 memberships (captured little-endian, despite the
    legacy format page claiming big-endian). Quantization can perturb their sum, so all
    12 classes (including background) are jointly normalized where their sum
    exceeds one, before resampling. Deficits remain air, not amplified tissue.
    ``voxel`` downsamples each class as it loads, limiting memory. Returned
    memberships are normalized at native resolution first, preserving volumes.
    Loading never downloads data or silently repairs a checksum mismatch.
    """
    if subject not in (4,5,6,18,20,38,*range(41,55)):
        raise ValueError("invalid BrainWeb 20-model subject ID")
    if dtype not in (torch.float32,torch.float64):
        raise ValueError("dtype must be float32 or float64")
    root = Path(root)
    manifest = json.loads((root/f"subject{subject:02d}.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("subject") != subject:
        raise ValueError("invalid BrainWeb manifest")
    names = ("background","csf","gm","wm","fat","muscle","skin","skull","vessels","connective","dura","marrow")
    shape = tuple(manifest["shape"])
    old = tuple(manifest["voxel"])
    new = old if voxel is None else _triple(voxel,"voxel",True)
    # uint16 maps + one uint32 total, instead of twelve native float64 volumes.
    raw_maps = {}
    storage = np.dtype(manifest["dtype"])
    if storage.str not in ("<u2", ">u2", "|u1") or manifest["scale"] <= 0:
        raise ValueError("unsupported raw encoding")
    total = np.zeros(shape,dtype=np.uint32)
    for name in names:
        entry = manifest["files"][name]
        path = root/entry["path"]
        if path.resolve().parent != root.resolve():
            raise ValueError("manifest paths must name files directly inside root")
        raw = gzip.open(path,"rb").read() if path.suffix == ".gz" else path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"checksum mismatch: {path}")
        if len(raw) != math.prod(shape)*storage.itemsize:
            raise ValueError(f"wrong byte count: {path}")
        array = np.frombuffer(raw,dtype=storage).reshape(shape)
        if array.max() > manifest["scale"]:
            raise ValueError("raw values exceed scale; check byte order")
        raw_maps[name] = array
        total += array
    total = np.maximum(total,manifest["scale"])
    fractions = {}
    for name in names:
        # Divide in float64, then convert only after conservative resampling.
        value = torch.from_numpy(raw_maps.pop(name).astype(np.float64) / total)
        fractions[name] = _resample(value,old,new).to(dtype).clamp(0,1)
    origin = tuple(o+(b-a)/2 for o,a,b in zip(manifest["origin"],reversed(old),reversed(new)))
    return Anatomy(_renormalize(fractions),new,origin)
