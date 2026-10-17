"""The sample specification: what the brain and the scan are; the seed fills in what it leaves open.

The spec (JSON) says what the brain and the scan are; the seed fills in what it leaves open:

    {"anatomy": {"source": "ixi", "prepared": "validation/anatomy/data/ixi/prepared_v16", "subject": null,
                 "lesions": {"count": 3, "size_mm": [3, 10], "region": "periventricular"}},
     "protocol": {"family": "fastmri_t2w", "manifest": "~/data/fastmri_more/manifest.json", "n": 128, "slices": "2-6:2"},
     "scanner": {"receive": "draw", "b1": "birdcage", "noise": "paired"},
     "quality": {"spins": [2, 3, 3]}}

anatomy: an IXI head (``subject`` null: drawn from the prepared directory, or from ``subjects`` if the spec lists a pool), the realism preset's anatomy options
(deep grey, falx, the fitted residual, native size, drawn yaw / pitch) and ``lesions`` (``anatomy.add_lesions``
keywords; null for none; "count": [lo, hi] draws the count per head, 0 meaning none). protocol: ``fastmri_t2w`` pairs the sample with one real scan's header (TR, TE, refocusing
angle, coil count, echo spacing: ``simulate_heads.real_protocols``) and plays its 21-echo train over 16 slices; ``n``
the matrix on the 220 mm field of view; ``slices`` the stack indices (a-b:step) or {"count": k} for k drawn levels.
scanner: ``receive`` "draw" (the elliptical normalised array drawn per head, simulate_heads 2026-10-13; ``receive_scale``
the semi-axes' scale range, re-centred on the 65-scan set's scalp level on 2026-10-15) or "ring"
(the 14 cm ring); ``b1`` "birdcage" or "uniform"; ``noise`` "paired" (the paired scan's background floor and coil
covariance, as score.py adds it) or "none". quality: spins per voxel (z, y, x); 2 x 3 x 3 keeps the sub-voxel grain at
the real scans' level (journal 2026-10-14).
"""

from __future__ import annotations

import copy

DEFAULT_SPEC = {
    "anatomy": {"source": "ixi", "prepared": None, "subject": None, "lesions": None, "deform_mm": 3.0},
    "protocol": {"family": "fastmri_t2w", "manifest": None,
                 "n": 128, "slices": {"count": 3}},
    "scanner": {"receive": "draw", "b1": "birdcage", "noise": "paired", "receive_scale": [1.1, 1.45],
                "coil_radius": 0.11, "coil_radius_lr": 0.095, "coil_gap_mm": 10., "receive_normalise_mm": 30.},
    "quality": {"spins": [2, 3, 3], "sampling": "proportional", "jitter": "inplane", "slab_voxels": 3, "rf_raster": 1e-4},
}


# every key the generator reads (DEFAULT_SPEC's, and the optional ones: anatomy.subjects, a pool of subjects)
KEYS = {"anatomy": {"source", "prepared", "subject", "subjects", "lesions", "deform_mm"},
        "protocol": {"family", "manifest", "n", "slices"},
        "scanner": {"receive", "b1", "noise", "receive_scale", "coil_radius", "coil_radius_lr", "coil_gap_mm",
                    "receive_normalise_mm"},
        "quality": {"spins", "sampling", "jitter", "slab_voxels", "rf_raster"}}


def merged(spec: dict) -> dict:
    """``spec`` over ``DEFAULT_SPEC``, section by section; an unknown section or key raises (a typo would otherwise
    be silently ignored)."""
    out = {k: dict(v) for k, v in DEFAULT_SPEC.items()}
    for k, v in (spec or {}).items():
        if k not in KEYS:
            raise ValueError(f"unknown spec section {k!r}; sections: {sorted(KEYS)}")
        unknown = set(v) - KEYS[k]
        if unknown:
            raise ValueError(f"unknown spec key(s) {sorted(unknown)} in {k!r}; keys: {sorted(KEYS[k])}")
        out.setdefault(k, {}).update(v)
    return out


def default_spec() -> dict:
    """A copy of ``DEFAULT_SPEC`` to edit."""
    return copy.deepcopy(DEFAULT_SPEC)


FALLBACK_PROTOCOLS = [{"file": "fastmri-like (no manifest)", "TR": 6.19, "TE": 0.113, "coils": 16, "refocus_deg": 180., "esp": 10.26e-3}]


def span(s) -> list[int]:
    """"a-b:step" (or "a", "a-b") -> the integers a..b every step."""
    a, _, step = str(s).partition(":")
    lo, _, hi = a.partition("-")
    return list(range(int(lo), int(hi or lo) + 1, int(step or 1)))
