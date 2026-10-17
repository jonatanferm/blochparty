# 2026-10-07 — Alternative anatomy datasets after M3 access limitation

- Continued from `75f2e86`, following the request to find another source/dataset.
  Inspected the NeuroPoly GitHub tree and complete public OSF inventory, the
  Jochmann/Schweser Zenodo ZIP directory/README using range reads, BigBrain-MR's
  metadata/unit notes/example, and AHEAD template v3. No unrestricted complete
  original QSM-source mirror was verified.
- Selected AHEAD v3 (10.21942/uva.19646364.v3, CC BY 4.0) as the most direct
  accessible alternative for brain parameter maps. Downloaded median chi, R1,
  R2* and relative PD: 250,309,250 compressed bytes. All publisher MD5s matched;
  committed compressed/decompressed SHA-256 pins and a repeatable fetch/inspect
  script. Data remain ignored. Four real files share exactly equal 0.5-mm RPS
  grids, shape 394×466×378, and all values are finite; rates/density nonnegative.
- Ran `fetch_ahead.py --offline` successfully against all four actual downloads.
  No simulation/core changes, so no new physics claim, full-suite rerun or default
  snapshot comparison was needed. This checks accessibility/integrity/geometry,
  not physical calibration or scanner agreement.
- Documented source limitations and a concrete route forward: a separate AHEAD
  brain-template adapter, retaining relative-PD meaning and 7 T rates; BrainWeb
  remains the full-head geometry source and QSM retains independent field checks.
  AHEAD lacks skull/air, intrinsic R2 and the original MATLAB ground-truth simulator.
  Its adapter is not implemented and original M3 acceptance criteria remain open.
