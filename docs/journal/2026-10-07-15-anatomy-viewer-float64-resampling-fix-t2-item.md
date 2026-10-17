# 2026-10-07 — Anatomy viewer; float64 resampling fix; T2* item on the roadmap

- `examples/anatomy_viewer.py`: 50 % iso-surfaces of BrainWeb fractions (skin, skull, CSF,
  cortex, white matter, vessels; 1.5 mm, ~1.2 M triangles) plus the simulated 3 T slab (T2w /
  T1w / shimmed B0) placed where it sits, in a local three.js page (not published: no
  BrainWeb redistribution licence).
- Bug found while building it: float64 BrainWeb resampling raised "fractions sum to more than
  one" (cumulative-sum rounding ~1e-14 vs a 3.6e-15 check). Renormalisation after resampling;
  regression test reproduces 1.2e-14 / 2.9e-14 overshoot without it.
- Roadmap: T2* from measured R2* without double counting, as the next physics item.
