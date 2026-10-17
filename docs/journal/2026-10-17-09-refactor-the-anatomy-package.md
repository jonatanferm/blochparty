# 2026-10-17 — Refactor: the anatomy package (plan phase 5)

- **Layout** (`40fac53` moves, `d0a634a` split, `a4d2e0a` docs): the 2095-line `anatomy.py` is `anatomy/` with
  `grid` (Distribution, TissueProperties, Anatomy, resampling, the air and water susceptibilities), `tables` (the
  cited tissue tables, MT and diffusion), the loaders `brainweb`, `qsm` (and the SHA-256 pinning), `ahead`, `ixi`,
  `phantom` (to_phantom, field maps, R2', texture), `edits` (scalp, dermis, lesions, deep grey, falx, ventricles,
  sulci), `deformation` and `sampler` (DigitalPhantom, drawn tissues, anthropometric scaling, sample_head);
  `tissuedb/` inside it, with `blochparty.tissuedb` forwarding. Every top-level statement moved verbatim; each
  module's imports were computed from what its code uses, and module-level import cycles were checked first.
- Two cycles had to be broken: the tissue tables are built at import time from the property classes, while the
  `Anatomy` class used `AIR_CHI` (so the two susceptibility constants live in `grid`); and the ventricle edit calls
  `deform`, which the sampler also needs (so the deformation code is its own module rather than part of the
  sampler, as the plan had it). The plan's `loaders/` folder became four flat modules: one function each.
- `blochparty.anatomy` re-exports every public name and the five private ones other code imports
  (`_file_sha256` for the fetch scripts, four sampler helpers for tests).
- **Checks**: a fingerprint of the tissue tables, tissuedb's 3 T table and one sampled BrainWeb head with every
  draw switched on (lesions, deep grey, falx, dermis, anthropometric size, measured sulci, yaw and pitch: 630,764
  spins, its fractions, field maps and phantom) is identical before and after, and identical run to run.
  tests/phantoms, test_io, test_realism_helpers: 230 passed. The AHEAD and QSM fetch scripts verify their files'
  SHA-256 against the package's pins.
- Found on the way: `tools/tissuedb_report.py` had raised KeyError since the myelin-water fields were added
  (`9680845`); fixed in `2ae6dab`.
- **Full suite on the mini** (phases 3, 5 and 7 together, after all three were committed): 847 passed, 180 skipped,
  0 failed, 13 min.
