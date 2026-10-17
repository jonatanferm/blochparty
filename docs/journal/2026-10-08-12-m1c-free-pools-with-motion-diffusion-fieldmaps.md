# 2026-10-08 — M1c: free pools with motion, diffusion, FieldMaps and the generalized bound pool

- Core: the free-pool branch of `simulate` moved after the shared spin setup, so the exact engine
  gets walks, motion, the field model and maps, placed per segment for moving spins. The
  generalized Bloch bound pool's modes became 2J rows of the exact matrix. Metal: `_bm*_diff`
  kernels; moving spins not cost-sorted (walks keyed by spin index). Metal still refuses free
  pools + generalized bound pool: with J = 30 modes the Taylor state would carry ~60 extra floats
  per vector, five vectors per thread; left open in ROADMAP M1c.
- Checks: maps = per-spin fields (1e-14); constant rigid shift = moved phantom (1e-13); inert pools
  = single-pool path with the same motion / walk (3e-7); generalized + free pools = a SciPy ODE of
  the mode equations written independently (2e-8); → rate model off resonance (2e-3); Metal = core
  (1e-5) on motion, diffusion, static and moving maps.
- Found: the core's generalized split engine is ~2e-4 off the exact solution in Mzb on a
  Lorentzian random case, because it sub-steps to h ≤ T2b / 2 regardless of segmentation (error
  falls 1.5x, not 4x, per halving); the exact engine matches the ODE to 3e-13. Not changed (the
  split engine is the fast imaging path and Metal mirrors it); an inert free pool routes a phantom
  through the exact engine. Also: subdividing a sequence renumbers its segments and changes the
  diffusion walks (a 1e-4 Monte Carlo difference that first looked like an engine discrepancy);
  `rf(0.5, ...)` is thousands of raster steps unless `n_steps=1`.
- 780 tests in 3 min 33 s; quick subset 755 in 55 s; snapshot bit-identical; plain Metal benchmark
  unchanged (3d-64 2.50 s); shader compile 0.9 s with 10 free-pool kernels per reduction.
- Commits: b8ed082 (feature), then these docs.
