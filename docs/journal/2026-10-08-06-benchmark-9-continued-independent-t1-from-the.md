# 2026-10-08 — Benchmark 9 continued: independent T1 from the IR Look-Locker scan

- `iohexol_t1.py`: subspace NUFFT of the archive's IR-LL (K = 6, central slice, fit from 0.15 s):
  T1 0.48-1.46 s, residuals 0.4-1.4 %. K = 4 had 4-13 % (first attempt); edge slices gave T1* above
  the 6 deg / 5 ms bound (lower flip at the slab edges).
- The single-Tsat calibration's T1 was up to 2x short. With T1 fixed to IR-LL and the rest
  recalibrated at 1.5 s, the 4 s prediction: median rms 0.018 (0.022 before, no-change 0.032),
  change correlation 0.81 (0.75), build-up at 4.3 ppm 0.5-1.2x the measured (0.16-0.79x).
