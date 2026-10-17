# 2026-10-06 (evening) — short-T2 spheres, benchmark 1 converged, many-coil GPU

- **Wall-thickness sensitivity (benchmark 2, 3 seeds)**: 0.5 mm 2.93 ± 0.16 %, 1.0 mm
  2.20 ± 0.04 %, 1.5 mm 2.10 ± 0.11 %. The wall matters for spheres 9–10 (sphere 10: 11.4 / 6.1 /
  5.0 %) but not for 13–14 (≈ 15–16 % and 22–28 %), so geometry is not their limit.
- **Short-T2 diagnosis**: the measured tails are water, 7–12σ above the noise floor. The simulated
  tail level is wrong in both directions (geometry), and the echo 2 / echo 1 ratio is too low for
  short T2. A spheres-only T2 scan (×1, ×1.25, ×1.5) with a free tail amplitude confirms the
  reference T2 values (longer is worse for 13 of 14 spheres), fixes spheres 9–10 (2.3 %), and
  leaves spheres 11–14 at 6.5–8.7 %, in the first echoes, where T2 ≈ echo spacing (7 ms) and the
  stimulated-echo balance depends on B1⁺ and the refocusing pulse. Open; not reference values.
- **Benchmark 1 at 1024 Sobol spins/pixel** (spheres only, 4 seeds, 22 min GPU): **1.51 ± 0.05 %**,
  matching brute force (1.49 %); the error bar is 4× smaller. What remains is a misfit on spheres
  1–3 (2.4 / 4.0 / 5.2 %), the longest T1, not sampling noise.
- **Metal many-coil reduction**: new kernel `bloch_chunk_mma` sums coils as complex matrix
  products on 8×8 simdgroup matrices (8 samples per block, coils in passes of 48, so no coil
  limit). 48³ GRE: 1 coil 0.41 s, 8 coils 0.42 s (was 1.21 s; CPU 1.98 s), 32 coils 0.82 s (was
  4.57 s), 64 coils 1.59 s. Default from 2 coils. Physics step refactored into a shared inline
  function; 12 Metal tests (direct and matrix paths, partial tiles and blocks, two passes).
- Next: Bloch–McConnell magnetization transfer (`docs/REALISM.md` R2); the short-T2 early-echo
  question goes on the list (B1⁺ map / refocusing pulse at the sphere positions).
