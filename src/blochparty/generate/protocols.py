"""The paired protocol: the real AXT2 scans' headers (``real_protocols``), the 16-slice interleaved stack and the
TSE that plays one of them (``sequence``), and the quality presets (spins per voxel, sampling, RF raster)."""

from __future__ import annotations

import json
from pathlib import Path

import blochparty as bp
from ..sequences.builder import System


FOV, THICKNESS, SLICES = 0.22, 5e-3, 16  # [m], [m], the stack


ACQUIRED_LINES = 396  # phase-encode lines of the real AXT2 scans (encodedSpace y in all 21 headers)


# the real AXT2 echo trains (validation/realism/echo_train.py): their k-space energy steps every 19 lines (ky -161,
# -142, ..., -28, -9.5 | +9.5): 396 lines over 21 echoes in linear order, ky = 0 in echo 11 = TE / the header's echo
# spacing (113 / 10.26 ms and 105 / 9.56 ms)
REAL_ETL = 21


def real_protocols(manifest: Path) -> list[dict]:
    m = json.loads(manifest.read_text())
    return [{"file": k, "TR": v["TR_ms"] * 1e-3, "TE": v["TE_ms"] * 1e-3, "coils": v["kspace_shape"][1],
             "refocus_deg": v.get("flip_deg") or 180.,
             "esp": None if v.get("echo_spacing_ms") is None else v["echo_spacing_ms"] * 1e-3}
            for k, v in sorted(m.items()) if v["contrast"] == "AXT2" and (v.get("field_T") or 0) > 2]


def interleaved(count=SLICES) -> list[int]:
    """Siemens' default interleaved order: slices 1, 3, 5, ... then 2, 4, ... (0-based 0, 2, ..., 1, 3, ...)."""
    return list(range(0, count, 2)) + list(range(1, count, 2))


QUALITY = {  # spins per voxel (z, y, x), sampling, slab half-width [voxels], RF raster [s], jitter
    "full": ((2, 2, 2), "categorical", 3, 1e-4, "stratified"),
    "fast": ((1, 2, 2), "categorical", 3, 1e-4, "stratified"),
    "legacy": ((2, 2, 2), "per_tissue", 4, 1e-5, "stratified"),
    # each voxel's composition exact, z a quadrature through the slice profile: no visible grain
    "real": ((4, 2, 2), "proportional", 3, 1e-4, "inplane"),
    # 4x4x4 in-plane-jittered: the random positions' grain (Immerkaer sigma in flat brain, of p99) falls as
    # 1 / sqrt(spins): 0.026 at 4x2x2, 0.017 at 4x3x3, 0.0136 here; real scans 0.014 with their noise
    "real_fine": ((4, 4, 4), "proportional", 3, 1e-4, "inplane"),
}


def sequence(n: int, tr: float, te: float, slice_order=None, rf_raster=1e-4, spacing=None, readout_os=1,
             refocus_deg=180., ordering="linear", etl=16, esp=None) -> bp.Sequence:
    """The paired protocol. With ``slice_order``: the neighbours as their real pulses, the rest on
    the bound pool only (0.71 % against every pulse exact, ``tse_2d``). ``readout_os``: readout
    oversampling (the real scans: 2). ``refocus_deg``: the refocusing flip angle (the headers'
    ``flipAngle_deg``: 180 in 11 of the 21 3 T scans, 140-179 in the rest). ``ordering``: "linear" (ky = 0 in
    echo 9 of 16), "linear_full" (the n lines take the echoes they have in the real scans' 396-line linear
    acquisition, ``ACQUIRED_LINES``: echoes 6-11, 25 shots) or "reverse_centric" (ky = 0 in echo 16; echo
    spacing te / 16). ``etl`` / ``esp`` (default 16 / None: te / 9, te / 16 reverse-centric): the echo train, e.g.
    the real scans' (``REAL_ETL`` at the header's echo spacing: 19 shots of "linear_full")."""
    multi = {} if slice_order is None else dict(bound_only_beyond=1.5 * spacing)
    if readout_os != 1:
        multi["readout_oversampling"] = readout_os
    if esp is None:
        esp = round(te / (16 if ordering == "reverse_centric" else 9), 5)
    full = dict(acquired_lines=ACQUIRED_LINES) if ordering == "linear_full" else {}
    return bp.tse_2d(n, FOV, etl=etl, esp=esp, ordering="linear" if full else ordering, tr=tr, n_dummy=1, slice_thickness=THICKNESS,
                     refocus_flip_deg=refocus_deg, spoiling="gradient", ramps=True, system=System(rf_raster=rf_raster), slice_order=slice_order,
                     **multi, **full)
