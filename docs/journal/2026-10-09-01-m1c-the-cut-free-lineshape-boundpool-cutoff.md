# 2026-10-09 — M1c: the cut-free lineshape (`BoundPool(cutoff=None)`) on core and Metal

- Question: free pools + generalized bound pool do not fit the `_bm` kernels' registers. Measured
  first how much the generalized model changes results when free pools are present (core, WM-like
  tissue with amide and myelin-water pools): a 36 × 50 ms, 2 µT CEST saturation train differs by
  1.2e-2 in Z at ±3.5 ppm between rate and generalized model, 5e-6 with a Lorentzian pool, so the
  gap is the near-resonance super-Lorentzian convention (1 kHz cut vs the tapered Green's
  function), not the pool's rotation. Not negligible: 2.3e-3 on MTRasym, a tenth of what
  benchmark 8 measures.
- Fix (A): `cutoff=None` evaluates the rate model's lineshape from the generalized model's modes,
  T2b Re Σ c_j / (p_j + i x) (`exchange.modes_lineshape`); Metal lineshape kind 3 reusing the
  `line` / `log_f` buffers (the table becomes the mode table), no new kernel variant; the fused
  kernel raises. With it rate = generalized to ≤ 5e-5 at 447 Hz to 10 kHz on the saturation train.
  Modes vs quadrature of the tapered Green's function's transform 1.3e-4 to 10 kHz; Metal `_full`,
  `_bm1`, `_bm2` vs core 5e-7; snapshot bit-identical; new engine-matrix row; 11 + 4 + 2 new tests.
- Found on the way: on resonance the rate model needs pulses long against the memory (3 ms), not
  T2b; the cut approximates short hard pulses better than the cut-free lineshape (0.2 ms / 60°:
  1.4e-3 vs 2.3e-2). So the remaining Metal exception is exactly the short-intense-pulse regime,
  and plan (B) for it (operator splitting with the modes in the `_gen` device buffer) plus when it
  would be needed are written into ROADMAP M1c.
- Dead end: a gradcheck of the new lineshape w.r.t. T2b failed with eps 1e-6 on T2b = 1e-5 (a 10 %
  step); scaled multiplier instead.
- Merged into main on 2026-10-09 (branch worktree-taper-lineshape, ae26e83), after the M2 adjoint;
  the merge commit adds the cut-free lineshape to the Metal adjoint (lineshape kind 3 and its
  derivatives in `lineshape_d`), checked by the engine matrix's gradient cell against the core.
