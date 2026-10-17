# 2026-10-07 — Benchmark 4: generalized Bloch on real NMR (Asslaender et al. 2022)

- Cloned the BSA / MnCl2 inversion-recovery data at the paper's commit 77950c4; replicated the
  authors' loader and T2* fits exactly (74.9793 / 52.4055 ms) and fitted their 6-parameter
  global model with blochparty (core, float64), against the numbers in their documentation.
- MnCl2 first fitted 0.839 % residual vs their 0.698 %: a hard pulse in one segment misses the
  T2* decay during the pulse (Strang split around a 180). With 50 segments R1 = 1.478733 ±
  0.002525 /s and 0.6984 %, identical to theirs. Documented in `SequenceBuilder.rf`; the default
  `n_steps=1` is unchanged (bit-identical rule) — a candidate default change.
- BSA, generalized Bloch: 0.4221 % (memory 100) vs the authors' 0.4136 %; m0s 0.0826 vs 0.0855.
  Traced to the super-Lorentzian taper (exp(-s / 100 T2b), 0.5 at the longest pulse): the engine
  equals the tapered reference to 4e-5; the untapered reference fits 0.4127 %, m0s 0.0851. A
  first reference run with a coarse grid gave 0.375 % (spurious; a 15-step Heun pi pulse).
  Added `BoundPool.memory` (core, fused, GPU; modes keyed on memory; default 100 bit-identical,
  snapshot checked); memory 300 fits its kernel to 2e-4 within 30 modes and gives 0.4171 %,
  m0s 0.0842. Rate model: 1.307 % with T2s 1.9 us (their Graham fit: 1.83 %).
- Tests: modes at memory 300, engine vs reference at memory 100 / 300 for a 1 ms pi pulse, GPU
  memory plumbing. 406 tests pass.
