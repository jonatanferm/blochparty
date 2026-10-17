# 2026-10-08 — M0: engine capability matrix, guards, quick test subset, Metal baseline

- **Matrix.** `docs/ENGINES.md`: the four entry points (core, fused, Metal, chunked) against 31
  features and combinations, forward and gradients (248 cells), precision per engine, measured
  agreement, and limits that are not physics. `tests/test_engines.py` parses the table and
  checks every cell on a small case: ✓ forward = agreement with the float64 core; ✓ gradients =
  the core's autograd against a central difference along a random direction (case rebuilt at
  θ ± εv), fused / chunked gradients against the core's autograd; ✗ = `NotImplementedError`.
  251 tests, 8 s. Flipping two cells in the document gave exactly those two failures.
- **Found by the matrix, fixed by raising** (no physics added; snapshot bit-identical):
  the core silently ignored `maps=` with free pools; Metal ran diffusion with exact concomitant
  fields, which the core refuses (so it had no reference); Metal returned results detached from
  the graph when inputs required grad (now `NotImplementedError` unless under `no_grad`; e.g.
  `nist_irse.fit_b1` with `BLOCHPARTY_BACKEND=metal` used to die in an opaque autograd error);
  `simulate_fused` lacked `maps` and `simulate_chunked` lacked `m_init`, `scanner`, `maps`
  (TypeError), now accepted and refused. Gaps left for M1: free pools with `FieldMaps` (M1c),
  diffusion with exact concomitant fields (M1b).
- **Agreement** (worst over the matrix): fused float64 3.6e-14 (gradients 9.7e-14), chunked
  3.5e-16; Metal 1.5e-5 signal / 1.7e-5 m on static spins (strong-gradient concomitant case),
  7.4e-5 signal and 1.1e-3 per-spin m with walking spins. The latter is float32 absolute positions
  (restricted 5 µm cylinders up to 5 mm off-centre): Metal walks absolute positions, the core is
  float64; a walk in offsets would fix most of it. Tolerances 1e-12 / 5e-5 / 2e-4 (walking signal)
  / 3e-3 (walking states).
- **Dead ends on the way, all in the test, not the engines:** finite differences that perturbed
  leaves in place missed tensors derived at build time (GIRF output, `subdivide_rf`, a scaled
  diffusion tensor), a carrier leaf derived from the RF leaf, an asymmetric perturbation of a
  Cholesky-factored tensor (8 % off), a saturated bound pool compared against its own maximum
  (1.5e-4 vs 1e-7 of equilibrium), and a restricted-walk D = 2e-9 at which τmax = 62.5 µs divides
  1 ms exactly, so ±ε both moved the sub-step count 16 → 17 and autograd and the (converged)
  finite difference differed by 10 %. Recorded in LESSONS.
- **Tests.** `slow` marker on 21 tests over ~1 s and on the crosscheck tests (a command-line `-m`
  replaces the one in `addopts`, so `-m "not slow"` would have run them).
  `uv run pytest -m "not slow"`: 714 tests in 45 s. Full suite: 735 passed in 3 min 21 s
  (484 in 3 min 8 s before).
- **Metal baseline** (`benchmarks/history.jsonl`, `--backend metal --repeat 3` at 6b21ccb):
  2d-96 0.02 s, 3d-32 0.06 s, 3d-48 0.45 s, 3d-64 2.40 s (8.2e9 spin-segments/s; the fused CPU
  kernel takes 5.36 s on 3d-64). Two bench fixes: the first Metal case was charged ~80 ms of first-dispatch
  setup (now a warm-up run), and GPU times vary (3d-64 2.4–3.4 s over nine runs), hence
  `--repeat N` keeping the fastest (default 1, unchanged).
- README: an Engines section matching the matrix, Metal in the field-model and B1+ paragraphs,
  the quick-subset and Metal benchmark commands, 12 missing modules in the layout table.
  CLAUDE.md points to the matrix.
- Commits: 236ce4d (engine guards), 97f4d25 (matrix and its test), dd1b38b (slow marker),
  6b21ccb (bench warm-up, `--repeat`), 428ff07 (Metal baseline), then these docs.
