# 2026-10-17 — Refactor: tests in area folders, shared cases, one GPU fixture, the tautology audit (plan phase 2)

- **Moves** (`f3ca1b2`, `git mv` only): the 57 test files into `tests/{engines,physics,sequences,phantoms,signal,
  validation,design}/` (12, 11, 10, 14, 4, 5, 1; `tests/docs/` came with phase 9). The four Metal files are
  `test_gpu`, `test_gpu_grad`, `test_gpu_nufft`, `test_gpu_reuse`, since they run on both GPU engines.
- **Rewiring** (`2b955fb`): `tests/cases.py` holds the twelve builders other files imported from test modules
  (`random_case`, `busy_sequence`, `random_phantom`, `pgse`, `bm_case`, `pool_case`, `_maps_for`, `distortion`, `sgp`,
  `partial_volume`, `coordinates`, `sphere`), moved verbatim, so no test imports another test. One `rel` in
  `helpers.py` replaces eleven copies; the `gpu` fixture (metal and cuda, each skipped where absent) is in
  `conftest.py` instead of four files. pyproject's `pythonpath` and `--import-mode=importlib` replace the
  `sys.path.insert` lines of six tests and `tools/snapshot.py`.
- **GPU coverage**: tests that ran Metal inside a core test, or Metal only, take the fixture (anatomy, design ×4,
  restriction, maps, multislice, diffusion, motion ×4 plus a new moving-spins-under-the-field-model case,
  integration, merge_free, phantom_b1, fields_concomitant, phantom_engines, the engine matrix's dtype test).
  Metal items 148 → 179, CUDA items 145 → 179.
- **Audit** (`42af3e5`), all 58 files read for tests that compare the code with itself or cannot fail:
  - `test_magnitude_unchanged` recomputed `mc_magnitude`'s own formula. Replaced by a check against the delta
    method: over 1e6 draws the debiased magnitude sits at A − v/(4A) and the plain |mean| at A + v/(4A)
    (bias 3.3e-3, standard error ~1e-4).
  - `check_kspace` compared `meta["kspace_trajectory"]` with pypulseq's `calculate_kspace`, which is where
    `read_pulseq` takes it from; removed (blochparty's own trajectory is still checked against it to 1e-9).
  - The merged-sequence gradient test only asserted a nonzero gradient; it now equals the unmerged one
    (rel < 1e-9), and the merged parts carry over 10 % of it.
  - Kept on purpose: spec restatements (design report SNR/CNR, soft references, transmit droop) and pinned
    constants with provenance (tissue tables, sulcal width). A static scan for self-comparisons, constant
    asserts and shape-only checks flagged 17 more; all are helper-driven or real checks.
- **Checks**: collected 976 → 1022, the same ids up to the folder prefix and the renames, plus 3 docs tests and the
  GPU items. Core snapshot bit-identical. Quick suite on the laptop: 812 passed, 176 skipped (11 min). Full suite on
  the mini: 838 passed, 178 skipped, 0 failed (14 min, with the design job sharing the GPU).
- **Docs** (`fe270f2`): living documents, docstrings and validation write-ups name `tests/<area>/…`; the journal
  and the archived roadmap keep old paths with a note at the top.
- **A sync mistake, caught** (`58b326a`): the documented `rsync` excluded only `validation/real/data`. The first sync
  of the new layout started on `validation/anatomy/data` (11 GB) and had, at some earlier sync, put the fastMRI
  `.eml` into the mini's mirror where the machine's other user could read it. Stopped before any data file
  landed (checked by birth time; `-cmin` is useless because rsync rewrites metadata), the `.eml` copy deleted, and
  `CLAUDE.md`'s command now excludes the data, `*.eml` and the installers. Moving tests also needs `--delete` on
  `tests/`, or the mirror keeps both copies. Two week-old wait loops of mine on the mini, which matched their own
  `pgrep` and could never exit, were killed.
