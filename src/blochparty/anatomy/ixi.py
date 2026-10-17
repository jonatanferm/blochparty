"""IXI heads prepared from T1, T2 and PD scans (``validation/anatomy/prepare_ixi.py``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .grid import Anatomy, _renormalize, _triple

IXI_MIN_LNCC = 0.15  # registration quality below which a prepared head is not drawn (IXI598, 0.112: CSF all over)


def ixi_subjects(root, prepared=None) -> list[int]:
    """IXI ids with a prepared head under ``root`` (validation/anatomy/prepare_ixi.py writes root/prepared; ``prepared``
    names another directory of prepared heads) whose registration to BrainWeb's frame reached ``IXI_MIN_LNCC`` (the
    rest segmented badly)."""
    out = []
    for p in sorted(Path(prepared if prepared is not None else Path(root) / "prepared").glob("IXI*.npz")):
        if json.loads(str(np.load(p)["meta"]))["lncc"] >= IXI_MIN_LNCC:
            out.append(int(p.stem[3:]))
    return out


def load_ixi(
    subject: int, root, *, voxel=None, dtype=torch.float32, residual: bool = False, prepared=None
) -> tuple[Anatomy, dict]:
    """An IXI 3 T head (Hammersmith, IXI dataset, CC BY-SA 3.0) prepared by validation/anatomy/prepare_ixi.py:
    BrainWeb's 12 classes on BrainWeb's 1 mm MNI grid, segmented from the subject's own T1 and T2 (EM with
    BrainWeb priors, affine registration), so the MNI-based steps (deep grey regions, ventricle prior, slice
    stack) apply. Returns the anatomy (resampled to ``voxel`` if given) and the preparation's meta: age,
    sex, the affine (MNI mm -> the subject's world mm; its scale is the subject's size), volumes. ``residual``: also
    the per-voxel T2 / PD multipliers fitted to the subject's own dual-echo TSE (``t2_scale``, ``pd_scale`` maps, if
    prepared; ``to_phantom`` applies them). ``prepared``: a directory of prepared heads other than root/prepared
    (runs that queue on one machine must not share and swap one directory; LESSONS 2026-10-13)."""
    path = Path(prepared if prepared is not None else Path(root) / "prepared") / f"IXI{int(subject):03d}.npz"
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    raw = data["fractions"]
    fractions = {k: torch.from_numpy(raw[i].astype(np.float32) / 255).to(dtype) for i, k in enumerate(meta["classes"])}
    maps = (
        {}
        if not residual
        else {
            k: torch.from_numpy(data[k].astype(np.float32)).to(dtype)
            for k in ("t1_scale", "t2_scale", "pd_scale")
            if k in data.files
        }
    )
    anatomy = Anatomy(_renormalize(fractions), tuple(meta["voxel"]), tuple(meta["origin"]), maps=maps)
    if voxel is not None and _triple(voxel, "voxel", True) != anatomy.voxel:
        anatomy = anatomy.resample(voxel)
    return anatomy, meta
