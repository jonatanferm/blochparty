# 2026-10-07 — Two defaults changed (approved): hard-pulse segments, generalized-Bloch memory

- Hard pulses (`SequenceBuilder.rf`, `block_pulse`, constant Pulseq RF pieces) are cut into
  <= 20 us segments (`sequence.MAX_RF_STEP`): the 20 us builder default stays one segment, so
  default builders are unchanged; longer pulses now carry relaxation during the pulse. First tried
  <= 10 deg per segment: 42 test failures, incl. gradchecks (a flip-dependent count is not
  smooth). Builders gained `rf_steps`; EPG / Ernst / signal-equation comparisons and the design
  examples pin `n_steps=1` (their references are instantaneous pulses).
- `BoundPool.memory` / `generalized_bloch.MEMORY` default 300 (was 100). The 300 modes were 1.5e-4
  off at short lags (a 10 us pi pulse missed its 5e-5 tolerance); `modes` now chooses the slowest
  pole (k / memory, k in 1-2) and weight floor per memory: 3.8e-5 at 300, 1.7e-5 at 100. Benchmark
  4 unchanged to 4 digits (m0s 0.0842, 0.4171 %). 406 tests pass.
