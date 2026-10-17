# 2026-10-09 — M2 finished: DESPOT1 through Metal, adjoint speed, the fused kernel retired

- **Merge.** The CEST branch (cut-free lineshape, `BoundPool(cutoff=None)`) merged as 6db590a, with
  the Metal adjoint extended to it; 814 tests, snapshot bit-identical; worktree and branch removed.
  A test run that took 269 s instead of 16 s right after the merge did not reproduce (19.8 s).
- **DESPOT1 through the GPU adjoint** (f8f6cde). `design.jacobian(method="fd4")`, the fourth-order
  stencil (convergence orders 2 and 4 checked). Through Metal at `rel_step` 5e-2: the pair 0.002°
  from the closed-form grid, its closed-form bound 1.3e-7 above the optimum, 4.7 s. Across steps
  2e-2 to 1e-1 the pair moves 0.0003–0.03° while the bound stays within 3e-6 of the optimum: the
  bound is flat along a2, so the parameter error is the wrong yardstick. Central differences at 1e-2
  had been 0.045° off; the core in float64 gets 0.0002° in 74 s.
- **Adjoint speed** (8af3bee). Per-kernel timing showed the replay at 1.5-2 forwards (it wrote 8
  rows per block start: m and the two-sum low parts, ~106 GB on 3d-48) and the sweep dominated by
  the coil term (adjoint 1.97 s with 8 coils, 0.84 s with a loss on m only). Now the replay keeps m
  only (3 rows, 4 with the bound pool; the sweep restarts each block with zero low parts, one ulp,
  still K1-independent), the samples' source terms are formed once per 4-segment block with the
  coils outermost, and the coil gradients are accumulated during the block recomputation (no extra
  step per sample). Alternating A/B against the old module on 3d-48, 8 coils: backward 2.70 → 2.15
  s (T1), 2.79 → 2.48 s (flip), 3.61 → 3.20 s (all); sweep alone 1.97 → 1.26 s (8 coils), 2.59 →
  1.81 s (16). Gradients within 8e-7 of before. The GPU was shared with another load (2x swings),
  so only alternating comparisons count. `bench.py --grad --backend metal` afterwards (coil, flip
  and T1 gradients): 3d-48 3.2-4.1 s against a 0.42 s forward (8-9x), 3d-64 20.6 s against 3.8 s
  (5.5x); `checkpoint_every` 384 to 4096 changes nothing. The 5x target holds for per-spin
  gradients only; left open.
- **Fused kernel retired.** Tests: shared helpers moved to `tests/helpers.py`; every `SIMS` entry
  and FD check that ran through the fused kernel now runs on the core (the physics tests' job) and
  engine agreement lives in the matrix; fused-only tests deleted; `test_phantom_fused.py` became
  `test_phantom_engines.py` (realistic phantoms through Metal: forward 1e-6 to 3e-6, gradients
  5e-7 to 1.9e-6 per input) and the integration test points at Metal (32000 RF segments: flip
  gradient 8.9e-4, where the core in float32 is 5.4e-3 off; tolerance 3e-3). Design tests: the
  TSE train (1.5 s) and the TR-budget design (36 s) on the core, whose float64 their checks need
  (through Metal the constraint held only to 8e-5); gradient equalities against Metal (1.5e-5,
  4.2e-6, 2.2e-6 measured). Nine test files would have been skipped wholesale by their
  `importorskip("blochparty.fused")`.
- **Validation re-run on Metal**: NIST benchmark 1 table digit for digit (124 s per version
  against 598 s), benchmark 2 at the table's settings digit for digit (64 s against 232 s), and
  incidental MT but for one cell on a rounding boundary (97.2504 %). Benchmark 1's convergence rows
  turned out to predate the T2 correction (16 x 1: 16.72 % now, on both engines, not 14.35 %).
  `examples/gre3d_128.py` 192 s on the GPU (6 min fused).
- **Dead ends and corrections.** A `.double()` on complex coil gradients showed a 1.0 error that
  was the measurement. A rename in `nist_tse.py` made its own `simulate()` recursive (engine now
  `engine`). Wall times of 13 to 88 min for runs with seconds of CPU were the Mac sleeping
  (Maintenance Sleep with 180 s DarkWake windows; pytest's own clock said 338 s for an 88 min
  suite), not the engines and not the swap I suspected first.
- 660 tests (5 min 38 s by pytest's clock; 814 before, the difference the fused kernel's own tests);
  core snapshot bit-identical (the fused entries dropped from it).
- Commits: 6db590a (merge), f8f6cde (DESPOT1 through Metal), 8af3bee (adjoint speed), 2442508
  (fused kernel retired), 5c38a1d (snapshot), f659d4a (docs).
