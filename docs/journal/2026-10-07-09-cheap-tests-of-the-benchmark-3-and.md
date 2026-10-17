# 2026-10-07 — Cheap tests of the benchmark 3 and 5 remainders; anatomy spec

- Wrote `docs/specs/ANATOMY.md` (chi -> B0, BrainWeb / QSM 2.0 loaders, cited tissue table,
  partial-volume assembly, required checks, file ownership) for a second agent.
- Benchmark 5: MT (≤ 0.1 pt, even at 4x literature f), diffusion (exactly exponential for the
  primary pathway: only a ~2 % T2 shift), readout delay (best at 0 us), sequence balance (CPMG
  exact) all ruled out. Left: hardware (GIRF / RF fidelity) or multi-exponential gel T2.
- Benchmark 3: sphere 4 is B0 (−45 Hz: 7.0 -> 4.0 %). No global gap / inversion efficiency /
  TR shift fits all spheres; per-sphere B0 + B1 cannot fit 6–8. Coil-map smoothing moves T2 of
  spheres 5–8 by 10–30 points while spheres 2–3 stay robust (T1 −5 to −8 %): 5–8 are
  reconstruction-limited; a twin with mismatched smooth coil maps does not reproduce that, so the
  twin is a lower bound. Next for this dataset: model-based reconstruction against k-space.
