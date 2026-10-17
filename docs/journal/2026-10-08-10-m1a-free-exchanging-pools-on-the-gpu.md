# 2026-10-08 — M1a: free exchanging pools on the GPU; Metal's float32 drift on RF trains

- **Design change against the plan.** The roadmap proposed a float32 matrix exponential per segment
  (scaling and squaring), cached across identical segments. Prototyped in float32 PyTorch against
  the float64 core first: dense `matrix_exp` was 1e-2 off after 2000 segments of benchmark 9's
  saturation train (2e-2 after 6000), 3.5e-4 on a stiff case (k 5 kHz, 0.8 s delays); and
  benchmark 9's shaped pulses never repeat a segment, so the cache would not help. Chosen
  instead: exp(A dt) applied to the state by a Taylor series of the structured A (no matrix),
  sub-steps of norm ≤ 1.5, truncation at 2^-27 of the magnetization, the common precession of
  free segments rotated exactly, increments added by two-sum (3.3e-5 → 2-3e-7 in the prototype).
- **Kernel** (`_bm1` .. `_bm5`, one per free-pool count, up to five so BMCTool's 5-pool library
  model runs): agreement with the core ≤ 1.3e-6 on 1-5 pools with and without the rate-model
  bound pool, both coil reductions, B1+, m_init, the scanner field model, a CEST spectrum with an
  M0 offset at −300 ppm. Performance: first version 5.5 s on benchmark 9's 15-pulse train (464
  spins); the pool count as a runtime loop bound was the main cost (templated: 2.1 s; an
  array-of-pointers suspicion was wrong). Spins are sorted by estimated cost before the dispatch:
  the M0 offset at −1500 ppm costs ~80 sub-steps per RF sample and stalled 31 lanes of every
  simdgroup it sat in (30k spins 18 s → 3.3 s; 464 spins 2.4 → 1.5 s, the M0 spin's own latency).
- **Found on the way: every Metal variant drifted on long RF trains.** Running the BMCTool
  comparison through Metal, the MT path (existing `_full` kernel) was 1.2e-2 off where the core is
  5.8e-7. Measured: plain, MT and generalized Bloch kernels all drift ~1e-3 per 1e4 RF segments of
  CW saturation, because each RF segment rotates m by a tiny angle and rounds all of it. Fixed
  with the same technique: rotation as an increment (Rodrigues with cos = 1 − s2 φ²), relaxation
  and exchange increments, two-sum into a low part folded back at free segments: ≤ 3e-7. Plain
  benchmark within noise. Free segments still drift ~1.5e-5 per 1e5 (left as is: the hot path).
- **Validation through Metal** (`BLOCHPARTY_BACKEND=metal`, new in iohexol.py, magma.py,
  compare_bmctool.py): benchmark 9 reproduces its prediction table digit for digit, Z within
  3.5e-7 of the core; one fit iteration 141 s → 5.7 s, 40-pulse prediction 114 → 8.6 s, the
  whole `predict` in 27 s. Benchmark 8 reproduces its table, Z within 8.3e-6. BMCTool: Lorentzian
  MT 6.5e-7, 5-pool CEST 8.3e-7, 30 offsets 12.8 s → 0.2 s. compare_bmctool.py could not run as a
  script (entry point before a function it calls); fixed.
- Left (ROADMAP M1a): free segments with fast exchange cost sub-steps ∝ k dt (benchmark 9's tube 8,
  k = 40 kHz, ~280 per 5.2 ms TR); they depend only on dt once the precession is factored out,
  so an exact propagator cached per spin would make them one small product.
- Tests: 761 passed in 3 min 18 s; snapshot bit-identical (core untouched).
- Commits: ce67969 (RF drift fix), e93e85d (free pools on Metal), 7419c23 (validation scripts),
  4173cfa (results), then these docs.
