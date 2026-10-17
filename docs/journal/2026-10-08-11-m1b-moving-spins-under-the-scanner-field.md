# 2026-10-08 — M1b: moving spins under the scanner field model

- Motion and diffusion under gradient nonlinearity and concomitant fields (leading and exact), core
  and Metal. Moving spins live in true coordinates under the field model; every segment maps them
  to encoding positions and evaluates the concomitant coefficients where they are. Static spins
  unchanged (snapshot bit-identical). Diffusion under the field model changed with it (walks in
  true coordinates; it used encoding coordinates and the start position's coefficients, and
  refused the exact mode). Static FieldMaps are now sampled at true positions.
- Verification: moving spin = chain of static one-segment simulations at mid-segment positions,
  4e-15 (core), 1e-5 per spin (Metal; 3.5-4.8e-6 signal on 300 spins with motion, 4e-5 with
  diffusion at 10 cm). Maxwell-compensated echo: a moving spin (≤ 2 m/s, 15 cm, 32 mT/m, 0.55 T)
  keeps up to 0.36 rad of concomitant phase, matching the exact path integral to 1e-6 with 100 µs
  segments; the mid-segment rule's error falls 4.0x per halving (C dt³/12 from the v² t² term);
  a static spin stays compensated to 1e-9.
- Dead ends / corrections: the Metal side of that echo test first failed at 5e-4: float32 phase
  resolution of ~6400-rad lobes (same with the ideal scanner), not the field model; the core
  gradient check of diffusion + nonlinearity failed at a 1e-6 finite-difference step from loss
  rounding, fixed with Richardson extrapolation at 1e-5.
- 767 tests in 3 min 23 s; plain Metal benchmark unchanged (3d-64 2.45 s).
- Commits: 6c7c9e0 (feature), then these docs.
