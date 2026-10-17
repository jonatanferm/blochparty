# 2026-10-08 — 3 T WM/GM T2* from whole-brain measurements

- Tissue table: 3 T GM 55.9 +- 1.8, WM 51.5 +- 2.2 ms (Shin et al. 2025, MRM, Table 3; 4 subjects,
  FreeSurfer GM/WM) replace occipital Wansapura 1999 (GM 41.6 < WM 48.4 ms). The abstract swaps
  the labels; table and per-subject rows agree with Peters et al. ISMRM 2006 (GM 59.7 > WM 54.6 ms).
- `DEFAULT_R2STAR_ACQUISITION` is now their protocol (1 x 1 x 2 mm, 43 echoes 3-66.18 ms) instead of
  the assumed 1 mm / 5-40 ms. Round-trip test cases moved to 3 / 5 / 8 Hz/mm (8 now clips).
- Long-TE GRE: T2* factor GM/WM 0.92 -> 1.04; total GM/WM 0.72 -> 0.81 (T1 weighting). Snapshot
  bit-identical (no anatomy in it); suite 477 passed.
