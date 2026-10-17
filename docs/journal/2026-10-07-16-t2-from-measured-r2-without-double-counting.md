# 2026-10-07 — T2* from measured R2* without double counting

- `estimate_macroscopic_r2prime`: per-voxel |sinc(g d t)| from the simulated field's gradient at
  the R2* measurement's voxel size, log-linear fit over its echo times; `to_phantom(t2star="auto")`
  now separates R2' = max(R2* - R2 - macro, 0) when chi is on (was: R2' dropped). Defaults for
  table values 1 mm / 5-40 ms (assumption); AHEAD 0.64 mm / 3-27.5 ms (Caan 2019 TEs).
- Round trip: 20.66 /s in, 20.73 / 20.66 out at 5 / 8 Hz/mm; double counted 22.52 / 25.54, dropped
  16.37 / 19.39. Inconsistent case (15 Hz/mm) clips by design. Fixed along the way: a function name
  shadowed by the existing `macroscopic_r2prime` argument; single-voxel axes in the gradient.
- AHEAD: macroscopic part ~0 (brain-only QSM), so measured R2* contrast now appears by default.
  BrainWeb: small except near air (99th pct 3.1 /s). New visible issue: cortex darker than WM at
  long TE from the occipital T2* proxies; roadmap follow-up.
