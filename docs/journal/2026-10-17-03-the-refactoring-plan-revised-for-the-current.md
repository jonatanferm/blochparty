# 2026-10-17 — The refactoring plan, revised for the current tree

- The plan of 2026-10-11 (draft PR #8, branch `claude/gracious-pasteur-taw1kb`, written at `f983c00`) was never
  merged and is 222 commits behind. Revised as `docs/REFACTOR_PLAN.md` on `main` from a fresh survey at `3663281`:
  `src` 12 802 lines (was 10 074), `metal/__init__.py` 2524 (64 % shader text), `cuda/__init__.py` 863 (552 of
  CUDA), `anatomy.py` 2095 (was 1126), tests 58 files / 976 items with 12 copies of `rel`, 22 scripts calling
  `simulate_metal` by name, 12 validation scripts editing `sys.path`, the journal at ~210 lines a day.
- What the revision changes: the GPU phase now covers both engines (shared `.metal` device code, CUDA kernels as files,
  one `gpu/` driver; verified by byte-identical generated source for Metal's library and all 49 CUDA kernels, so the
  move needs no GPU until the closing check); a new phase 3, one GPU entry point (`simulate_gpu`) so scripts run on
  the L40 as well as the mini; a new phase 4, the M7 training-data generator from `validation/realism/` into
  `blochparty.generate`, with CG-SENSE, the orthonormal FFT pair and line masks into `recon` and the receive-array
  model into `coils`; `anatomy/` gains an `edits.py` for the generator's anatomical edits; journal by month now rather
  than later. Ten phases, about nine working days, seven open decisions for the user (section 9).
- **The author's decisions** (the plan's section 9, same day): tests in subfolders by area; one entry point
  `bp.simulate(..., engine="gpu" | "metal" | "cuda")` with `"core"` the unchanged default (rather than a separate
  `simulate_gpu`); the generator as `blochparty.generate`; `engines/core.py`; `blochparty.metal` / `.cuda` kept as
  forwarding modules; `sequences/`; the journal as one file per entry with `JOURNAL.md` as the index; CI on every push;
  `main` pushed now and after every finished phase (`CLAUDE.md` updated); the design examples' reusable parts into
  `blochparty.design.problems`. Folded into the phases; the estimate is now about ten working days.
