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
| **Bloch on the scanner's .seq, T1 v1** | **2.76 %** | **2.39 %** | 4.2 7.2 5.2 2.8 1.3 1.5 1.4 0.3 1.7 1.9 2.8 2.2 0.6 0.3 |
| analytic IR formula, T1 v2 | 25.93 % | 19.88 % | 30.2 33.8 35.5 32.2 29.1 28.7 24.1 21.4 15.4 8.8 9.0 5.9 2.9 1.2 |
| **Bloch on the scanner's .seq, T1 v2** | **1.49 %** | **1.29 %** | 2.2 3.3 5.3 1.8 0.6 0.4 0.2 0.3 0.4 0.3 1.4 1.2 0.4 0.2 |

**Correction (2026-10-06):** the first version of this table (1.44 % / 2.67 %) used T2 values
that belong to the *MnCl2* (T2-plane) spheres. The NiCl2 spheres' T2 is now derived from their
T1 with the 3 T relaxivity ratio (`nist.py`); the change is small because T2 only scales each
curve at fixed TE.

![curves](nist_irse_curves.png)

The analytic formula is what the dataset's authors fitted, plus a free offset that absorbs
exactly the effects it lacks; without that offset it misses by ~26 %. Simulating the actual
sequence (sinc pulses and their slice profiles, the real timing, crushers and spoilers) gets
within ~1.5 % with no free parameter. Reference version 2 fits better than version 1
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

## Benchmark 2: NIST phantom T2 plane, 3 T multi-echo spin echo (`nist_tse.py`)

The scanner's 23-echo CPMG `.seq` (echo spacing 7 ms, 180° windowed-sinc refocusing, slice and
readout crushers; two slices, the image is the second at x = −12 mm), simulated on a digital
T2 plane: T2 of the MnCl2 spheres from the phantom manual (spheres 1 and 5: Jiang et al. 2016),
T1 from the 3 T relaxivity ratio. Metric: per-sphere echo-train shape (one scale per sphere).

Measured 2026-10-06, Apple M4, fused kernel, float32, 64 spin positions across the slice × 4
per pixel along the readout (85 k spins), RF merged 4:1, 8 warm-up TRs before the imaged pass:

| model | spheres 1–10 | odd 1–9 | per sphere 1..14 [%] |
|---|---|---|---|
| mono-exponential, reference T2 | 10.47 % | 9.27 % | 4.2 4.9 5.8 7.3 5.7 9.5 11.2 15.1 19.6 21.5 32.3 41.2 42.0 53.1 |
| **Bloch on the scanner's .seq, nominal B1⁺** | **2.25 %** | **2.09 %** | 0.7 0.9 1.1 1.3 2.6 0.7 1.0 2.6 5.1 6.6 18.6 24.4 23.7 36.3 |
| Bloch + global B1⁺ 1.144 (fit on even spheres) | 1.90 % | 2.09 % | 0.8 0.7 0.6 0.5 4.0 1.6 2.3 0.8 2.7 5.0 16.5 22.9 21.4 34.6 |

![curves](nist_tse_curves.png)

- **The stimulated-echo signature comes out of the physics**: echo 1 below echo 2 (sphere 2:
  0.85 simulated vs 0.82 measured), and an apparent decay slower than T2 — both missed by the
  mono-exponential model that T2 fitting assumes (why the authors dropped echoes 1–3).
- **Convergence** (nominal B1⁺, spheres 1–10): 21 k spins 2.58 %, 85 k 2.25 %, 171 k 2.37 %,
  343 k 2.35 % (two different splits) — converged at ~2.3 %.
- **B1⁺ as a digital-twin parameter** (`nist_tse_b1.py`): echo shapes constrain |B1⁺ − 1| only,
  and the error is *highest* at nominal B1⁺. A global scale fitted on the even spheres (1.144,
  equivalently ~0.856) helps the fitted spheres but leaves the held-out odd spheres unchanged
  at converged density (2.09 %); smooth low-order maps overfit; a free B1⁺ per sphere (an upper
  bound, not a prediction) reaches 1.6 %. So B1⁺ is not the main remaining error.
- **What is**: (1) the **fill fluid** is not modelled. The deionised water around the spheres has
  T2 ≈ 2 s; at late echoes it blurs into the ROIs of short-T2 spheres (ROIs sit ~0.7 px inside
  the spheres), which is the flat tail on spheres 9–14 — not thermal noise, which is ~15–30× lower.
  (2) Reference-value uncertainty: the worst long-T2 sphere (5) is one whose T2 the manual
  withholds; MnCl2 T2 is temperature-sensitive and the scan temperature is unknown.
