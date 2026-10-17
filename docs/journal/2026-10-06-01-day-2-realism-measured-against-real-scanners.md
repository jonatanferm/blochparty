# 2026-10-06 — Day 2: realism, measured against real scanners

### Strategy
- The owner: the end goal is to be the best and "step out of the shadow" of other
  simulators. Researched what others model (JEMRIS, KomaMRI and MRzero from their source,
  MRiLab, POSSUM, CMRsim, MCMR, Pulseq-CEST, EM-coupled and vector-field simulators) →
  **inventory of ~40 effects** in five physical layers (`docs/REALISM.md`, `974f228`).
- Finding: every simulator owns one layer; none combines them fast and differentiably; **none
  is calibrated against real scanners**. Strategy: a public real-data realism benchmark and a
  differentiably calibrated digital twin.
- Cloud credits down to ~$38: one cloud brief at a time, local work in parallel.

### Field model (cloud, PR #6, `d025dc1`)
- Concomitant fields (leading order + exact |B|), gradient nonlinearity (gradunwarp/Siemens
  coefficients), GIRF/eddy currents, drift and respiration — all opt-in, both kernels,
  differentiable in hardware parameters. Verified on merge; on the real 3 T IR-SE sequence
  concomitant fields change sphere signals by 0.37 % at 3 T and 2.0 % at 0.55 T (the
  expected 1/B0 scaling).

### Benchmark 1: real 3 T NIST phantom, inversion-recovery spin echo (`7c99bd8`)
- Data: Tong et al. 2022 (Pulseq `.seq` + raw 20-coil data, ISMRM/NIST phantom, Prisma Fit).
- Method: simulate the scanner's own `.seq`; phantom T1 from CaliberMRI/NIST references
  (versions 1 and 2, serial unknown); **nothing fitted**; metric = per-sphere
  inversion-curve shape.
- Dead ends on the way: phantom at x = 0 while the slice is at **+57 mm** (RF frequency
  offset) → nonsense; too few spins across the slice → speckle and shallow nulls (14 %).
- **Result: 1.49 %** (v2 references; v1 2.76 %) vs **25.9 %** for the analytic IR formula the
  authors fitted (they needed a free offset). Converged in spin density.
- **Correction** (`0ae53a4`): the T2 first used for the NiCl2 spheres belonged to the MnCl2
  spheres (a misread table). No table gives these cross terms; derived them from 3 T
  relaxivities (Martin et al. 2023), validated by the derived MnCl2 T1s landing in the
  published range. Result moved 1.44 % → 1.49 %; recorded openly in RESULTS.md.

### Lessons and process (`3af579f`)
- Started `docs/LESSONS.md` (owner's request), `tools/snapshot.py` (bit-identity guard) and
  `CLAUDE.md` (working rules). Started this journal.

### Benchmark 2: real 3 T NIST phantom, 23-echo multi-echo spin echo
- Signature to reproduce: echo 2 ~20 % brighter than echo 1 in long-T2 spheres (stimulated
  echoes from imperfect refocusing) — the reason the authors dropped echoes 1–3.
- Sphere identity from geometry (the T2 plane mirrors the T1 plane), confirmed by decay rates
  — the authors' T2 map was unusable for this.
- **Dead end, instructive**: the simulation plateaued at ~61 % error with *growing* echoes at
  every spin density. Ruled out an RF-phase (CPMG) hypothesis, corrected a flawed k-space
  diagnostic of our own, then found the cause: the file images **two slices** (+12 / −12 mm);
  "discard the first 128 excitations" meant "use the second slice", and the phantom sat on the
  first.
- **Result** (converged, nothing fitted): **2.35 %** shape error over 23 echoes vs **10.47 %**
  for mono-exponential decay with the reference T2; sphere 2 echo2/echo1 1.173 vs 1.213 on
  the scanner.
- **First digital-twin step: B1⁺** (`nist_tse_b1.py`). The shape error is *highest* at nominal
  B1⁺ and falls both ways: echo shapes constrain only |B1⁺ − 1|. A global scale fitted on
  even spheres (1.144) does not improve the held-out odd spheres at converged density (2.09 %
  either way); smooth maps overfit; even a free B1⁺ per sphere only reaches 1.6 %. An honest
  negative result: B1⁺ is not the main remaining error.
- **Diagnosis of what is**: short-T2 spheres plateau at late echoes at ~15–30× the thermal noise
  floor — blurring from the unmodelled **fill fluid** (water, T2 ≈ 2 s) into ROIs that sit
  ~0.7 px inside the spheres. Second suspect: T2 reference uncertainty (sphere 5's T2 is one the
  manual withholds; MnCl2 T2 is temperature-sensitive).
- Next: model the fill fluid and sphere walls (both benchmarks).

### Water fill, and Monte Carlo instead of a second engine
- Modelled the fill (deionised water, T2 ≈ 2 s) and sphere walls from image geometry. First
  result was *worse* (2.6 % → 5.8 %). The scanner's late-echo tails matched the simulation
  *without* water, refuting the "water blurs into the ROIs" story for most spheres; ideal
  end-of-TR spoiling changed nothing; the simulated water was heavily speckled even at echo 1:
  Monte Carlo noise from too few spins per pixel, which a magnitude image turns into an upward
  bias that is largest exactly at short-T2 spheres' late echoes.
- Considered analytic voxel integration (state/PDG methods). The owner preferred not to: state
  pruning error, within-voxel uniformity assumptions, a second engine. Decision: keep one exact
  isochromat engine and improve the *sampling and estimation* instead — scrambled-Sobol spin
  placement, importance sampling for the water, independent seeds averaged in the complex
  domain (embarrassingly parallel), magnitude-bias correction from the seed spread, and
  jackknife error bars. Brute-force speckle on a water patch: std/mean 0.38 (64 spins/pixel) →
  0.27 (256) → 0.13 (1024). Validation of the MC estimator against that reference is running.

### Apple GPU kernel (owner: "priority number 1")
- CPU at 60–70 % and the M4's GPU idle for our work. Built `blochparty.metal` with
  `torch.mps.compile_shader` (runtime Metal, no build system): one thread per spin, same physics
  as the CPU kernels, float32. First version 2.5–3× *slower* than the CPU kernel; profiling showed
  spin evolution was already ~4× faster and the per-ADC-sample reduction over spins dominated.
  Block-buffered reductions doubled speed. On the real NIST TSE workload (1 coil, long RF trains,
  23 echoes) it is **5.8× the CPU kernel (40 s vs 232 s)**; on 8-coil synthetic GRE still slower
  — the next optimisation (simdgroup matrix multiply) is noted. 8 tests vs the float64 core.

### Monte Carlo validated; the water fill vindicated
- Water-patch validation on the GPU (8 seeds): Sobol placement falls ≈ N^-0.75 vs N^-1/2 for
  stratified (≈ 12× fewer spins at equal error); the bias-corrected level agrees to ~1 % even at 64
  spins/pixel.
- Benchmark 2 in MC mode: spheres only 2.38 ± 0.06 % (matches brute force); **with the water fill
  2.20 ± 0.04 %**, and the short-T2 spheres roughly halve (sphere 11: 19.4 → 7.8 %, 14: 39.1 →
  22.1 %). The earlier "water makes it worse" (5.75 %) was Monte Carlo bias from undersampling:
  the original hypothesis was right, the first test of it was not.

### State at end of day 2 (for picking up after a context reset)
- **Repo**: `~/projects/blochparty`, private GitHub `jonatanferm/blochparty`, branch `main` only.
  Working rules in `CLAUDE.md`; lessons in `docs/LESSONS.md`; realism plan in `docs/REALISM.md`;
  real-data benchmarks in `validation/real/` (data via `fetch_data.py`, git-ignored).
- **Engines**: PyTorch core (reference, float64-capable), fused C++ CPU kernel with adjoint
  (`simulate_fused`), Apple-GPU forward kernel (`blochparty.metal.simulate_metal`, float32;
  `BLOCHPARTY_BACKEND=metal` for the real-data scripts). Metal is 5.8× the CPU on the 1-coil real
  TSE, slower than CPU with 8 coils (ADC reduction).
- **Results so far** (nothing fitted, reference values only): benchmark 1 (IR-SE) 1.77 ± 0.19 %
  MC (1.49 % brute force) vs 25.9 % analytic; benchmark 2 (23-echo CPMG) 2.20 ± 0.04 % with water
  fill vs 10.5 % mono-exponential. Run with `--mc 4 --n 256 [--fill-margin 8]`.
- **Running when this was written**: benchmark-2 wall-thickness sensitivity (`--wall 0.5` and
  `1.5`, 3 seeds); the 1 mm baseline is the table in `validation/real/RESULTS.md`.
- **Next, in order** (agreed with the owner):
  1. Short-T2 spheres (8–22 % error): read the wall sensitivity; if geometry isn't the limit,
     look at reference values / unmodelled effects.
  2. Benchmark 1 at Sobol n = 1024 to shrink its error bar (spheres only; water not significant).
  3. Metal: many-coil ADC reduction (simdgroup matrix multiply); later adjoint.
  4. Roadmap: Bloch–McConnell magnetization transfer (`docs/REALISM.md` R2), checked against
     BMCTool / Pulseq-CEST and the Pulseq-CEST phantom data.
- **Cloud credits** ≈ $38 left: one cloud brief at a time. The Mac sleeps with the lid closed.
