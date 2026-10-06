# Realism benchmark results

## Benchmark 1: NIST phantom T1 plane, 3 T IR spin echo (`nist_irse.py`)

The scanner's own 10 `.seq` files, simulated on a digital phantom whose T1/T2 come from
published references (**nothing fitted to this data**), against the scanner's raw data
(3 T Siemens Prisma Fit, 20 coils; Tong et al. 2022, CC BY 4.0). Metric: per-sphere
inversion-recovery curve shape error (one scale per sphere), see `README.md`.

Measured 2026-10-06, Apple M4, fused kernel, float32, 128 spin positions across ±1.5 slice
thicknesses × 2×2 per pixel (174 k spins), RF merged 4:1:

| model | spheres 1–10 | 1–14 | per sphere 1..14 [%] |
|---|---|---|---|
| analytic IR formula, T1 v1 | 27.30 % | 21.11 % | 31.8 37.5 35.3 33.2 30.0 30.0 25.2 21.5 17.6 10.9 10.8 7.2 3.2 1.4 |
| **Bloch on the scanner's .seq, T1 v1** | **2.67 %** | **2.31 %** | 4.0 7.2 5.1 2.8 1.2 1.3 1.3 0.2 1.7 1.9 2.6 2.1 0.6 0.2 |
| analytic IR formula, T1 v2 | 25.93 % | 19.88 % | 30.2 33.8 35.5 32.2 29.1 28.7 24.1 21.4 15.4 8.8 9.0 5.9 2.9 1.2 |
| **Bloch on the scanner's .seq, T1 v2** | **1.44 %** | **1.24 %** | 2.1 3.3 5.2 1.8 0.5 0.3 0.2 0.3 0.4 0.3 1.3 1.2 0.4 0.2 |

![curves](nist_irse_curves.png)

The analytic formula is what the dataset's authors fitted, plus a free offset that absorbs
exactly the effects it lacks; without that offset it misses by ~26 %. Simulating the actual
sequence (sinc pulses and their slice profiles, the real timing, crushers and spoilers) gets
within ~1.4 % with no free parameter. Reference version 2 fits better than version 1
(consistent with, not proof of, a version-2 phantom; the serial isn't recorded).

### Convergence (T1 v2, RF merged 4:1)

| spins across slice × per pixel side | spins | time | spheres 1–10 | sphere 7 null (scanner 0.038) |
|---|---|---|---|---|
| 16 × 1 | 5.6 k | 20 s | 14.35 % | 0.280 |
| 32 × 1 | 10.9 k | 41 s | 13.63 % | 0.285 |
| 64 × 1 | 21.8 k | 89 s | 2.98 % | 0.054 |
| 32 × 2 | 43.8 k | 174 s | 7.52 % | 0.126 |
| 64 × 2 | 86.5 k | 318 s | 2.16 % | 0.038 |
| 128 × 2 | 174 k | 598 s | **1.44 %** | 0.037 |

Too few spins across the slice leave the crusher dephasing (~1.7 cycles per slice thickness)
unresolved: the magnitude of a few random phasors at a null isn't zero, so nulls stay shallow.
Exact RF (no merging) at 64 × 2: 2.03 % (v2) vs 2.16 % merged, so merging is not what limits.

### What's left (≈ 1–5 % on spheres 1–3, and 1–2 % on 11–12)

Not modelled yet, roughly in order of suspicion: phantom temperature (references are at
20 °C; long-T1 NiCl2 spheres are the most temperature-sensitive), the fill fluid around the
spheres (partial volume and ringing into the ROIs), coil sensitivities and B1⁺ variation
(constant per sphere for the shape metric, but not within a sphere's ROI), sphere wall
thickness. Each is a next step for the digital twin (`docs/REALISM.md`, R7).
