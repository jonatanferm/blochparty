# Project journal

A chronological record of what we did, why, and how it went — decisions, results with
numbers, dead ends and corrections — kept so the project can be written up later. Newest
entries at the bottom. Hard-won technical lessons are distilled separately in
[LESSONS.md](LESSONS.md); the plan lives in [../ROADMAP.md](../ROADMAP.md) and
[REALISM.md](REALISM.md).

---

## 2026-10-05 — Day 1: from idea to a fast, differentiable, verified simulator

### Scoping
- **Idea**: an MRI simulator — phantom + coils + sequence → k-space. Goals, in the owner's
  words: a fun project, simulated training data, and (agentic) sequence development;
  differentiability "speaks to" the owner.
- **Landscape check**: KomaMRI (Julia, GPU, Pulseq) is closest in spirit; MRzero (PyTorch) is
  the differentiable one; JEMRIS the hardware-realism veteran. Room to add something: a
  Python-native, differentiable simulator with coils/noise as first-class citizens and an
  agent-friendly API.
- **Language**: first recommended JAX (function transforms, fusion, forward mode). The owner
  asked about PyTorch's user base → switched to **PyTorch**: MRI deep-learning ecosystem
  (fastMRI, MONAI, MRzero, torchkbnufft), better Apple-GPU story, and agents write it best.
- **Verification as a principle** from the start: compare against analytic results, an ODE
  solver and an independent signal-equation model; later other simulators and real data.

### MVP (Phase 0)
- Core: isochromat Bloch with exact free precession/relaxation, Rodrigues RF rotations,
  Strang-split relaxation inside RF; sequences as flat piecewise-constant segments
  (continuous parts differentiable, structure plain Python); Shepp-Logan with 3 T tissues;
  analytic birdcage-like coils; 2D spoiled GRE; Cartesian recon.
- **21 tests**: SciPy ODE (rotations exact to 1e-9; measured 2nd-order splitting
  convergence), FID / spin echo / inversion recovery / Ernst closed forms, signal equation
  (1e-13), gradcheck, and gradient descent recovering the Ernst angle and the analytic
  WM/GM-contrast-optimal flip angle (16.109°, to 0.001°).
- Most first test failures were the *tests'* idealisations (T2 blur during readout,
  off-resonant excitation, finite pulse duration, float32 constants) — investigated, not
  loosened.
- Named **Bloch Party** (`blochparty`); moved to `~/projects/blochparty`; private GitHub repo.

### Performance and 3D (Phase 1, moved up front)
- Considered a C++/LibTorch rewrite (owner: performance, shipping without Python, "feels more
  serious"). Measured first: the PyTorch core was memory-bandwidth bound, so LibTorch would
  not help; the win is a fused kernel, which needs no LibTorch. The owner chose to stay in
  Python, with compiled kernels as extensions.
- 3D phantom/builder/recon + benchmark harness (`a430b05`). Baseline on the Apple M4: 3D 64³
  in 233 s; 128³ projected at ~4 h.
- **Closed-form free runs** (all RF-free segments of a TR in one step, any gradient
  waveform): 3.5–9.5× (`e899957`); 64³ → 60–68 s.
- MPS (Apple GPU) ran unchanged but no faster; `torch.compile` only 1.4×/2×.

### First parallel agents
- The owner offered $100 of cloud credits. Three workers were launched; the "remote" option
  silently fell back to **local worktrees** (discovered from paths; noted for the future).
- **EPG reference model** — exact against isochromats once spins outnumber dephasing orders.
- **Memory-bounded backward** (`simulate_chunked`) — 16× less saved memory at N/16.
- **Fused C++ kernel** (forward) — 11–16× over the PyTorch core; 3D 64³ in 5.4 s
  (`26a64b2`). A full **3D 128³** (565 k spins, 2.2 M segments, 8 coils) ran in **6.3 min**
  (`4ff288f`), against the ~4 h projection that morning.

### Cloud workers (owner-launched, claude.ai/code)
- **Pulseq import + MRzero cross-validation** (PR #1): GRE / SE / bSSFP / EPI agree to
  1e-6–4e-5; an exact solver as tie-breaker puts the residual on MRzero's float32 numerics.
- **Hand-written adjoint for the fused kernel** (PR #2, `9484880`): gradients match autograd to
  ~1e-14; ~8× faster than the PyTorch core with gradients. Fixed an Apple-clang `-Werror`
  build issue at merge. Backward costs ~7× forward on the M4 (3.8× on a Xeon).
- **Three parallel Phase 2/4 workers** (PRs #3–#5, integrated `c48a98b`): realistic sequences
  (ramps, slice selection, real spoiling, SE/TSE/bSSFP/EPI), realistic phantoms (sub-voxel
  spins, T2′, B0/B1⁺ maps, fat), and a design toolkit (DESPOT1 reproduces the 1/√2-of-Ernst
  optimum). Integration caught what no single worker could: two different γ constants and a
  150 vs 170 T/m/s slew-limit mismatch.

---

## 2026-10-06 — Day 2: realism, measured against real scanners

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

## 2026-10-06 (evening) — short-T2 spheres, benchmark 1 converged, many-coil GPU

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

## 2026-10-06 (late) — magnetization transfer (R2 begins)

- **Decision**: the short-T2 misfit (spheres 11–14) is parked. Reference T2 values, the wall
  and noise are ruled out; the suspects (local B1⁺, played-vs-written timing) need data we do
  not have. Recorded in `docs/REALISM.md`.
- **Model** (`exchange.py`, `Phantom.bound`, commit 5649814): the Henkelman two-pool model
  with Graham's rate W = ω1²G, as BMCTool/Pulseq-CEST. The bound pool is Mz only, with
  exchange on z. The 2×2 longitudinal step is exact, and split around the free pool's RF
  rotation. W uses each spin's offset from the RF's instantaneous frequency
  (`Sequence.rf_freq`, or from the RF phase), so slice-selective pulses saturate other
  slices. The super-Lorentzian is an exact quadrature table (1e-11 vs SciPy), cut at 1 kHz
  (Gloor) per pool. Literature check: the rate model fails for short intense pulses
  (Assländer, generalized Bloch model), which is a later opportunity.
- **Checks**: matrix exponential; two-pool ODE with second-order convergence (error ratios
  4.4, 4.1); CW steady state from 4×4 linear algebra (5e-6); gradcheck through f, k, T2b and
  the RF. Defaults bit-identical. A commensurate-step trap in the CW test (2π per step at
  4 kHz) gave an identical 1.7e-3 bias at two step sizes, a false "converged".
- **BMCTool** (`validation/compare_bmctool.py`, `validation/RESULTS_BMCTOOL.md`): Lorentzian
  Z-spectra agree to **5.8e-7**, including the −3 ppm asymmetry; through the `.seq` file to
  2.1e-5. The super-Lorentzian differs by up to 0.12 near the pool, purely from the
  near-resonance convention: with a replica of BMCTool's lineshape (1.3e-15 to its function)
  everything agrees to 4e-7, except one offset sitting on BMCTool's own lineshape
  discontinuity at 1 ppm from the pool.
- **Fixes found on the way**: (1) the exact exchange step overflowed (0·∞) for long steps
  and lost float32 gradients for 1 µs steps (3 %); the increment form with expm1 of the
  eigenvalues fixes both (float32 1.5e-5), checked against `matrix_exp`. (2) A block pulse
  without a frequency offset is one RF segment in `read_pulseq`: `max_rf_step=` (opt-in).
- **Next**: the fused CPU kernel forward and adjoint for the bound pool (the 2×2 adjoint is
  derived and prototyped against autograd through `matrix_exp`), then Metal forward.
- **Fused CPU kernel** (5776f82): the bound pool as a fourth component; the exact exchange
  step, with a closed-form adjoint, in place of longitudinal relaxation. Forward 1e-15 vs
  the core; every gradient, pool parameters and RF frequency included, ≤ 7e-13 vs autograd;
  gradcheck. 1.45× the plain kernel once saturated steps are cached on |rf|². Bug found: NaN
  from padding lanes (det A = 0).
- **Apple GPU**: `step_mt` in both coil-reduction kernels (one uniform branch, dummy buffers
  without MT); 2e-6 vs the core, +15 % runtime; coil benchmark unchanged. R2's bound pool is
  now in all three engines; defaults bit-identical throughout.
- Next candidates: an incidental-MT demonstration (multi-slice signal loss in white matter
  vs the literature's 30–45 %); free-pool exchange (myelin water, CEST) in the same engine;
  the generalized Bloch model for short pulses.

## 2026-10-07 (night, autonomous) — incidental MT, free-pool exchange, generalized Bloch

- **Incidental MT** (`validation/incidental_mt.py`, `RESULTS_INCIDENTAL_MT.md`) against Shin et
  al. 2009 (3 T LL-EPI with fat saturation, 1 → 9 slices: ~30 % loss). Nothing tuned. The
  model gives a linear growth with slice count and 97–99 % at 1 slice, with MT 2/3–3/4 of
  the loss for long pulses. The magnitude scales with the unknown fat-sat RF power: 6–19 % for
  10 ms Gaussians at 90–180°; ~30 % needs ~2× the power of 10 ms at 180°. Short pulses are
  ruled out by the measured single-slice value. A plausibility check, not a validation:
  the paper does not give the pulse.
- **Free exchanging pools** (`ExchangePool`, `Phantom.pools`): CEST pools and myelin water, with
  full Bloch dynamics, exchanging all three components with the water, plus the bound pool.
  Exchange couples the pools' transverse magnetization, so the rotation and closed-form
  structure no longer applies. A separate exact engine in the core does one matrix
  exponential of the augmented Bloch–McConnell system per segment, as BMCTool; the signal sums
  all free pools. Checks: SciPy ODE on the system written out by hand (water + amide +
  myelin pools ± bound pool, random RF and gradients, < 1e-8); closed-form two-site exchange
  of a free decay (McConnell; 1e-5, the 1 µs pulse); an inert pool reproduces the split
  core; **BMCTool's own 5-pool CEST model: 8.3e-7 in Z over 50 offsets**, same MTR_asym.
  Cost 0.3 ms per segment with 4 pools, which is fine for Z-spectra but slow for imaging.
  Fused/Metal/chunked refuse pools for now (next: a kernel version).
- **Generalized Bloch model** (`generalized_bloch.py`, `validation/RESULTS_GENERALIZED_BLOCH.md`):
  the bound pool rotated by RF via the Green's function of its lineshape (Assländer et al.).
  A second-order integro-differential solver (first order at first, until the RF was
  integrated cell by cell). It reproduces the paper's three limits independently: Bloch for a
  Lorentzian (SciPy, ratios 4.2/4.1/4.05), Henkelman's steady state for all three lineshapes
  (≤ 2e-4, which also checks exchange.py's G = πg without BMCTool), and z_s → cos α for hard
  pulses; the free-pool coupling matches the two-pool ODE (1e-6). Finding: our rate model
  can't invert the pool for pulses ≤ 100 µs (−0.97 vs 0 at 10 µs, α = π), and even for 1–3 ms
  on-resonance pulses it saturates too little because of the 1 kHz cut. Next: the paper's
  linearized R2,s,l in the imaging engines.
- **State at the end of the night**: tests 334 pass, defaults bit-identical. Open:
  exchange pools in the fused and Metal kernels; generalized Bloch in the engines
  (linearized); incidental MT needs a dataset with its fat-sat pulse; the short-T2 NIST
  question (B1⁺ / timing).

## 2026-10-07 — kernel ports

- **Metal scanner field model** (fbaa2e3): nonlinearity (encoding positions) and concomitant
  fields (leading, exact) in the off-resonance, as kernel.h; vs the core < 1e-4 at 0.55 T.
- **Generalized Bloch in the engines.** The paper's linearized R2,s,l is tabulated per
  rectangular pulse and drops the pool's off-resonance, so it was not used. Instead G(s) is a
  sum of complex exponential modes: the memory integral becomes linear modes u_j, and the RF
  rotates (Mzb, S = Σ u_j) like a Bloch pool (Σ c_j = G(0) = 1) with u_j += c_j ΔS, exact
  and O(J). Fit attempts: positive real exponentials can't make the flat top (Lorentzian
  wings, 12–18× the super-Lorentzian at 30 kHz); complex poles fitted to a finite range
  needed weights of ~5000 that diverge beyond it; ridge destroys the cancellations. What
  works is to define the model with a memory taper e^{−s/(100 T2)} (a ~133 Hz homogeneous
  broadening at 12 µs, a physical regularisation of the singularity) and fit its whole
  support: 30 modes, error 2.4e-5, weights < 1, lineshape within 0.5 % from 0 to 30 kHz.
  Core (`BoundPool(model="generalized")`): matches the reference solver to 1e-5 (including
  −0.969 after a 10 µs π pulse), second-order convergence, and the rate model for
  off-resonant saturation (its regime) to 2e-3; gradcheck. The split needs RF steps ≲ T2b,
  so RF segments are sub-stepped to T2b/2 (100 µs steps were 0.06 off).
- **Fused kernel**: generalized forward (724d4de) and adjoint (b05a4b0). Every input's gradient
  matches autograd to ~1e-14 on the first run.
- **Metal**: the generalized step in both coil kernels. Bugs: `half` is a type in MSL; and
  the generalized arrays (and earlier the MT/field-model state) slowed the plain kernels 10–30 %
  through register pressure. Each kernel is now compiled in three variants (plain / rate
  MT + field model / generalized), and the plain kernel is back at 0.40 s.
- **Free pools in the fused kernel** (ac01fc6, `fused/bm.h`): the exact engine per spin
  (Padé-13 matrix exponential, reused across identical segments). Adjoint by the block-matrix
  Fréchet trick, chained through A's entries. Forward 1e-14; every gradient ~1e-13 vs autograd
  through the core, after one bug (RF gradients credited to segments without RF).
- Not ported: free pools on Metal (a dense ~17×17 exponential per thread and segment, which
  overflows registers; pools are used for spectra and few spins, where the CPU kernel suffices),
  and the Metal adjoint (the whole reverse sweep for every physics variant: the largest port).

## 2026-10-07 — direction: one realistic core, on this Mac

- The owner: realism of the simulator core comes before agentic sequence design or datasets,
  and for now one engine, the fastest on this Mac. Decision: the Metal GPU kernel is the main
  engine and the PyTorch core the reference; the fused CPU kernel is frozen (new physics
  raises there); GPU gradients, the hybrid runner and other engines come later (CLAUDE.md).
- Measured CPU+GPU concurrency on the 48³ GRE: 1.15–1.5× over the faster device (ideal
  1.7–2.3×; shared power budget, bandwidth, one core driving the GPU). Parked.
- Next: moving spins, starting with free diffusion as exact Brownian segments drawn from
  a counter-based RNG (Philox) shared by the core and the GPU.
- **Diffusion** (037492a core; Metal next commit): spins as exact Brownian walkers. Per segment of
  constant gradient the displacement and the bridge integral ∫r dt are jointly Gaussian, so the
  phase is exact without sub-stepping. The normals come from Philox4x32-10 (Random123 known
  answers) keyed on (seed, spin, segment): the core and the GPU draw the same walk and agree spin
  by spin (1e-5), and merged free runs equal segment stepping. Stejskal–Tanner, isotropic and an
  anisotropic tensor: unbiased (2M walkers × 20 seeds on the GPU: 0.54296 ± 0.00009 vs
  0.54297). Speed: 2M walkers through a PGSE in 0.07 s on the GPU, 30× the core.
- **Multi-pathway diffusion**: a CPMG train with crushers and 150° refocusing (spin and
  stimulated echoes, each with its own diffusion history) against extended phase graphs with
  diffusion (Weigel et al. 2010, implemented independently in the test): 1M GPU walkers agree
  with EPG to 1e-4 on all 8 echoes (e.g. 0.9029/0.9248/0.8496 vs 0.9028/0.9247/0.8495), while
  diffusion moves them by up to 0.23. Passed on the first run.
- **Restricted diffusion** (`restriction.py`, `Phantom.restriction`: planes, cylinders,
  spheres per spin; core and Metal). Walls need sub-steps (rms ≤ size / 10, counts from Python
  so the GPU matches) and reflections. First reflection rule, the radial mirror: signals ~1 %
  high and an equilibrium 11 % depleted at the wall (a KS test caught it). Specular reflection
  fixed it: KS within the 5 % level; sphere long-time limit to 3e-4, cylinder 0.3 % (finite
  pulse), planes vs Tanner–Stejskal's series at Δ = 10/25/200 ms within 3e-3, free along the
  cylinder. GPU = core to 1e-5 (same walk, reflections included). Metal's 31-buffer limit was
  hit; the scalars are now one parameter array (plain kernels unchanged at 0.39 s).
- **Flow and motion, positions** (`motion.py`, `Phantom.motion`; core and Metal): r(t) =
  A(t)(r0 + u(t) + v t + d w(t)) + b(t), evaluated at each segment's mid-time (exact for the
  phase of a constant velocity), on top of the diffusion walk in the tissue frame. Checks:
  phase contrast 2πGvτ² exact; a constant rigid transform equals the moved phantom; a jump
  between k-space halves equals the static k-spaces stitched (Fourier shift); inflow in a
  spoiled GRE vs the time-of-flight recursion within 1 % at 0/0.05/0.15 m/s; GPU = core with
  flow, periodic displacement, rotation and diffusion at once. Test bugs on the way: the textbook
  ideal-slice inflow formula was the wrong reference (the sinc's edges tip less, so the static
  signal is 36 % higher); the reference now uses the pulse's own complex profile. The flow
  column must be long enough to feed the slice for the whole run, and the profile must be read
  before the spoiler. Not yet: fields sampled at the moving position (coils, B1⁺, scanner B0),
  and motion with gradient nonlinearity or concomitant fields (raise).
- **Fields where the spin is** (`maps.py`, `simulate(maps=FieldMaps(...))`; core and Metal): grids of
  the scanner B0, B1⁺ and complex coil sensitivities, trilinear; static spins sample once (merged
  free runs kept), moving spins every segment (coils at every ADC sample on the GPU, direct
  reduction). Exactness of trilinear for linear fields gives sharp checks: static linear maps =
  per-spin fields (1e-12); a spin moving through a linear B0 map = a static gradient (1e-12);
  a translated object under mapped coils = the moved static object; GPU = core 1e-6 with flow,
  rigid motion, diffusion and maps together. Plain kernels unchanged (0.38 s).
- **Receive chain** (`receive.py`): oversampled readouts (segments split at the finer sample
  times), a centred windowed-sinc decimation filter (even length for even os, odd for odd: no
  delay), thermal noise with channel covariance, coherent interference tones, spikes, gain
  drift, quantisation; `ReceiveChain.acquire` in the scanner's order. Checks: measured |H| =
  designed |H| to 1e-5 at os 2/3/4 including the band-edge roll-off (−20 % at 0.9 of
  Nyquist), zero phase; noise power through the filter = σ² Σh²; quantisation error step²/12
  and clipping; a tone lands in the image column of its frequency. Test mistake: I expected a
  zipper line, but a tone coherent with the TR lands at one phase-encode point; real zippers are
  lines because interference is incoherent.
- **Transmit chain** (`transmit.py`): the RF amplifier per pulse: Rapp AM/AM (gain, saturation,
  sharpness), Saleh-type AM/PM, droop with delivered energy; ideal by default. Checks: linear and
  saturation limits; a hard pulse's flip and phase after the amplifier predicted exactly
  (Mxy = i e^{iψ} sin α in blochparty's own RF phase); droop law.
- **Loop coils** (`loops.py`): quasi-static Biot–Savart loop arrays: receive conj(B1−) by
  reciprocity, B1⁺ of the driven array, the sample-loss resistance matrix ω²∫σ A_i·A_j dV as noise
  covariance (Roemer), `field_maps` into `FieldMaps`. Checks: on-axis field (1e-4), far-field
  dipole, off-axis vector potential vs elliptic integrals (1e-4), resistance symmetric, positive
  definite, identical coils fully correlated, neighbours more than far ones; mapped coils equal
  direct ones through a whole simulation (2e-3, trilinear on 1 mm).

## 2026-10-07 — Additional scanner-validation data sources

- Searched primary dataset records, papers and repositories for diffusion and CEST phantom
  raw data and complementary validation of implemented physics. Added
  `validation/real/SOURCES.md` and linked it from the existing real-data README.
- Directly read PulseqDiffusion's 21.1 MB phantom MATLAB sample in memory: complex64 k-space
  `(96, 8, 3672)`, 8 channels, 3 slices/directions, 5 nonzero b-values, bmax 1000 s/mm²,
  partial Fourier 0.75. Its current sequence-generator settings differ from the scan, and
  the repository contains no `.seq`; acquisition provenance and independent phantom
  properties remain prerequisites for a forward-prediction benchmark.
- Verified iohexol CEST raw data + full `.seq` definitions, WASABI/IR-LL references and
  repeat scans in Zenodo version 17244380. Inspected the 403-entry ZIP directory using
  small range reads; did not download the 31.6 GB archive. The small MAGMA APTw/WASABI
  release is 53.6 MB of reconstructed DICOMs, useful for spectra but not raw k-space.
- Also catalogued tensor-valued diffusion waveforms/phantoms, a known-diameter taxon phantom
  (current XNAT access unverified), BSA/MnCl2 NMR for generalized Bloch, measured GIRFs,
  NIST IR-bSSFP with B1/reference scans, and a cross-vendor MPRAGE/EPI package. Distinguished
  synthetic data, derived maps and in-vivo-only raw data from measured phantom sources.
- Research/documentation only: no simulator changes or new benchmark results; no bulk data
  retained. Recommended beginning with the small diffusion/CEST sources and resolving their
  metadata gaps, then adding MT/GIRF checks and a held-out raw CEST benchmark.
- Second pass: added the radial and spiral-TSE Pulseq phantom pairs (`.seq` + raw), the Pulseq
  QA package (one sequence, four 3 T scanners), RRSG 2020 T1 data, the multi-site fat-water
  phantom, the QSM 2.0 head phantom and mariepy (full-wave EM maps). Re-ranked: Scholand
  7654462 is the NIST T2 plate with B1 map and pulse-duration variants, so it goes first (the
  short-T2 question). Added routes for what is not public (nonlinearity from grid phantoms,
  filter response from noise spectra, timing by fitting with held-out scans).

## 2026-10-07 — Benchmark 3: radial IR-bSSFP of the NIST T2 plane (Scholand et al. 2023)

- Downloaded Zenodo 7654462 (phantom k-space, vendor B1 map, Cartesian IR-SE / SE gold
  standards; MD5-checked) and the authors' ROIs (bloch-moba, pinned). Correction to the source
  catalogue: the RF pulse-duration variants are volunteer scans; the phantom has one IR-bSSFP.
- No `.seq`: rebuilt the sequence in blochparty from the paper, the authors' scripts and BART's
  simulator (HS inversion with BART's parameters, alpha/2 preparation, Hamming sinc, Skyra-like
  trapezoids); 10 us RF raster is 3e-4 from 1 us; GPU = core to 2e-5.
- Measurement side (`scholand_recon.py`, numpy + finufft): trajectory conventions recovered from
  BART's source and checked on images; anisotropic gradient delays from antiparallel spokes
  (0.64 / 0.78 samples, 0.015 rms); sign from data consistency (0.23 vs 0.78 residual); coil
  compression, ratio coil maps, rank-K subspace CG reconstruction; noise covariance from k-space
  edges. Gold standards refitted with complex coil combination: agree with the authors for
  spheres 1–10 (1–3 %); spheres 11–14 are not measurable with a 15 ms first echo (authors'
  735 ms for sphere 14 = magnitude noise floor). B1 map: low in short-T1 spheres; smooth fit used.
- Dead ends and corrections: (1) assumed the two ROI sets share an order, then a geometric
  mirror mapping — both wrong; the radial set is reversed (signals and the authors' table agree;
  geometry is ambiguous for a ring); (2) the first basis (non-selective signals only) biased the
  digital-phantom fits by up to +41 % T2: truncation; a union basis with slice-selective signals
  fixed it (spheres 2–8 within ±3 % T1, ±11 % T2); (3) stated the B1-map bias for long-T1
  spheres before the ROI order was fixed: it is the short-T1 ones.
- Result (full model, spheres 2, 3, 5, 6): T1 −4 to −10 %, T2 −1 to −14 % vs gold; without the
  slice profile T2 +35 to +55 %; B1 map ~10 % in T2; HS vs perfect inversion ≤ 2 %. Residuals
  4–9 % vs 1.4–2.3 % on the digital twin: an unexplained remainder (B0 unmeasured, coil maps,
  trajectory beyond linear delays, the rebuilt sequence). Spheres 9–14 are out of reach of the
  pipeline, so the short-T2 question from benchmark 2 stays open.

## 2026-10-07 — Benchmark 4: generalized Bloch on real NMR (Asslaender et al. 2022)

- Cloned the BSA / MnCl2 inversion-recovery data at the paper's commit 77950c4; replicated the
  authors' loader and T2* fits exactly (74.9793 / 52.4055 ms) and fitted their 6-parameter
  global model with blochparty (core, float64), against the numbers in their documentation.
- MnCl2 first fitted 0.839 % residual vs their 0.698 %: a hard pulse in one segment misses the
  T2* decay during the pulse (Strang split around a 180). With 50 segments R1 = 1.478733 ±
  0.002525 /s and 0.6984 %, identical to theirs. Documented in `SequenceBuilder.rf`; the default
  `n_steps=1` is unchanged (bit-identical rule) — a candidate default change.
- BSA, generalized Bloch: 0.4221 % (memory 100) vs the authors' 0.4136 %; m0s 0.0826 vs 0.0855.
  Traced to the super-Lorentzian taper (exp(-s / 100 T2b), 0.5 at the longest pulse): the engine
  equals the tapered reference to 4e-5; the untapered reference fits 0.4127 %, m0s 0.0851. A
  first reference run with a coarse grid gave 0.375 % (spurious; a 15-step Heun pi pulse).
  Added `BoundPool.memory` (core, fused, GPU; modes keyed on memory; default 100 bit-identical,
  snapshot checked); memory 300 fits its kernel to 2e-4 within 30 modes and gives 0.4171 %,
  m0s 0.0842. Rate model: 1.307 % with T2s 1.9 us (their Graham fit: 1.83 %).
- Tests: modes at memory 300, engine vs reference at memory 100 / 300 for a 1 ms pi pulse, GPU
  memory plumbing. 406 tests pass.

## 2026-10-07 — Benchmark 5: the scanner's own Pulseq spiral TSE (T1MES, Cima.X)

- Fetched Zenodo 18749100 (`.seq` + ISMRMRD with trajectory; MD5). The `.seq` imports (5.1 M
  segments, 145,600 samples); the first three TRs are simulated (1024 spins across 32 mm:
  converged; B0 +-40 Hz, T1 and refocusing scale have no effect on the shape).
- Reconstruction (`spiral_tse_recon.py`): CG-SENSE per echo. First coil maps came from a
  Pipe-style DCF that does not converge with the NUFFT Gram operator: random maps, and a false
  "tube 3 anomaly" (Bloch 3.8 % vs mono 1.2 %) that I chased through B0, B1, T1 and refocusing-
  scale tests. Per-coil least-squares maps (deterministic) removed it.
- Result: Bloch on the exact `.seq` at nominal B1, T2 fitted per tube: shape residual 1.1–2.7 %
  vs mono-exponential 3.9–5.4 % (short-T2 tubes) and 1.6–2.3 % vs 1.7–2.0 % (long-T2); method floor
  0.2–0.6 % (digital twin). Global B1 fit = 1.00. Remaining < 1 % of echo 1 per echo: odd/even at
  late echoes, echo-1/2 step; candidates MT of agarose, eddy currents, multi-exponential T2.
- T1MES references are 1.4 T values; the 3 T T2s here are 35–50 % shorter (agarose exchange).

## 2026-10-07 — Two defaults changed (approved): hard-pulse segments, generalized-Bloch memory

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

## 2026-10-07 — Cheap tests of the benchmark 3 and 5 remainders; anatomy spec

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
## 2026-10-07 — Anatomy M1: susceptibility fields and shimming

- Started from anatomy spec `a31f4a9` on `codex/anatomy-susceptibility`. Added
  `susceptibility_to_b0` (SI susceptibility → Hz, arbitrary B0 direction, anisotropic
  voxels, configurable padding and explicit surrounding material) and `shim` (order
  0–2 solid harmonics, matching `b0_shim`'s names and physical units). No engine changes.
- Independent closed forms, 3 T, dchi=9.4e-6: 10-mm sphere inside/outside RMS
  0.908/0.369 → 0.804/0.190 → 0.439/0.108 Hz at 32³/48³/64³, with 2× padding;
  far-field RMS 4.287 → 0.134 → 0.0216 Hz at 1×/2×/3× padding. Partial-volume
  sampling plus the standard kernel suffices; an additional anti-alias kernel is deferred.
- Long-cylinder inside/outside RMS at 0°, magic angle, 90°: 1.604/1.598,
  1.671/0.331, 1.766/0.830 Hz, using 4× transverse padding. Interior means
  398.624/0/-199.312 Hz vs infinite-cylinder 400.228/0/-200.114 Hz; finite ends
  and periodic copies remain. A sphere-adjacent ROI shims from 94.111 to 20.615 Hz RMS;
  harmonic inputs vanish below 1e-12 Hz and the residual is orthogonal to every fitted term.
- Corrections: the naive even-grid dipole kernel failed joint object/field rotation by
  ~2 Hz because of pairs of Nyquist axes. Average their signs independently, eliminating
  the offending cross terms (< 2e-12 Hz after correction). A long cylinder initially had
  ~6.1 Hz offset at 2× padding: increased transverse padding, not relaxed tolerances.
  For large float32 shim masks, avoid N*eps rank cutoffs and use the exact mean for order 0.
- Clarified the spec: Anatomy chi uses dimensionless SI (loaders convert ppm by 1e-6);
  padded absolute maps need an explicit air background. Rotation of the head relative to
  B0 requires field recomputation; only translation can carry the susceptibility field
  unchanged, and scanner shims must remain separate. M2–M5 and the QSM cross-check are open.
- Validation: 22 focused tests pass (including linearity, rotation, gradients, dtype,
  malformed inputs and a >8-million-voxel float32 map); before/after snapshots of existing
  default outputs and gradients are bit-identical. Full suite: 390 passed, 37 GPU tests
  skipped in the sandbox, 4 opt-in crosschecks deselected (413.99 s). All 37 GPU skips
  were then covered by 51 passing tests with host GPU access. The final focused rerun
  includes one additional large-map test: 428 distinct default tests covered overall.

## 2026-10-07 — Anatomy M2: BrainWeb partial-volume phantoms

- Continued M1 `d513a43` on `codex/anatomy-susceptibility`; unrelated benchmark commit
  `a3a8a00` arrived during the work and is preserved. Added `Anatomy`, conservative
  overlap-integral resampling/crop/slab, pinned raw-short BrainWeb loading, tissue
  distributions with citations/assumption flags, per-tissue or categorical spin
  assembly, six fat peaks, optional MT/diffusion, and susceptibility/scanner map pairs.
  No engine, `phantom.py`, `maps.py` or real-scanner validation changes.
- Accessed the public download form without accounts or identity fields. Committed
  subject-04 and older SBD manifests only; downloaded/derived volumes stay ignored.
  Corrected the spec's geometry: fuzzy anatomy is 0.5-mm (362,434,362), while the
  newer T1 reference is a separate 1-mm grid. Initially tried raw bytes: documented
  unsigned values were actually signed/offset encoded. Switched to raw shorts, then
  found the actual byte order is little-endian, contrary to the legacy format page.
  Pin decompressed bytes; normalize only overfull membership totals, leaving deficits air.
- Separate tissue spins conserve per-tissue/per-voxel PD to <1e-12; resampling physical
  volume to <1e-6. Mixed-voxel GRE agrees with Ernst to <1e-8. Shifted continuous-box
  Fourier error 0.177089→0.00384746→0.000230445→0.0000143652 for 1/4/16/64 samples/cell.
  Sphere B0 at jittered spins: 0.464157 Hz RMS. GPU/core with B0+MT+diffusion: 3.23e-7.
  Rounded fat weights sum to 0.999: normalize in new assembly to preserve PD.
- T2* policy is explicit: auto omits uncalibrated R2′ when chi is enabled; combining
  observed table T2* and chi needs a supplied macroscopic calibration or intrinsic map.
  K=1 has only the zero Cauchy median and warns. Tissue SDs are metadata, not M4 draws.
  Ex-vivo/other-tissue priors are flagged; only WM/GM have verified 7 T T1/T2* entries.
  Missing requested MT/diffusion parameters and 7 T classes raise.
- Downloaded older T1/T2/PD comparisons: GM/WM relative differences −0.0144/−0.1253/+0.0275%;
  CSF/GM −1.549/+4.733/+2.201%. Newer subject-04 T1 differs −5.44% GM/WM, −21.31% CSF/GM;
  older priors are not established as its exact parameters. Published human 3 T relaxation
  values propagated through these sequences provide a sensitivity check, not measured
  in-vivo ratio validation. Results and limitations: `validation/anatomy/RESULTS.md`.
- Full 2-mm head B0 at 3 T: brain RMS 45.2976→19.1716 Hz after second-order shim;
  residual 1/50/99 percentiles −52.8329/0.5364/51.4486 Hz. Scalar tissue/air/bone
  assumptions, not measured-field agreement; QSM forward-field comparison stays M3.
- Full 1-mm head, categorical K=1: 5,328,321 spins; load 5.807 s, assembly 0.308 s,
  GPU RF/FID 0.686 s (7.77 M spins/s). Phantom 149.19 MB; peak process RSS 3.64 GB;
  end-of-run MPS allocation 1.08 GB (not peak, not additive to unified-memory RSS).
  Benchmark excludes susceptibility/T2′/MT/diffusion and is not full imaging throughput.
- Validation: 21 anatomy tests; full suite with host GPU access **449 passed, 4 opt-in
  crosschecks deselected** in 185.08 s. Before/after existing outputs and gradients are
  bit-identical. M3 QSM, M4 spatial variability and M5 example remain open.


## 2026-10-07 — Anatomy M3: QSM loader and public challenge comparison

- Continued M2 `7a87c19`. Added optional nibabel loading through `load_qsm2`, pinned local
  manifests, 16-label memberships, NIfTI affine permutation/flip handling, ppm→SI/mm→m
  conversion, explicit exterior susceptibility and acquisition field. Reject contradictory
  geometry/units, unknown labels, invalid rates, changed hashes and silent field scaling.
  No engine changes. Full-head files require a locally supplied authorized package.
- Corrected a source assumption: Radboud version 1 metadata declares restrictedAccess and
  RU-DI-HD-1.0, not CC BY. Asked for a local copy while continuing implementation. The user's
  Stage-1 bit.ly link led to a closed Google registration form. Found the challenge author's
  public SURFdrive share via Berkin Bilgic's lab page; read-only directory listings avoided
  downloading its 12.65 GB archive. Retrieved both chi/segmentation ground truths (~107 MB)
  and exploratory frequency/mask files; all volumes stay git-ignored.
- Public GT shapes are (164,205,205) x/y/z with 0.999024391-mm headers. Frequency/mask
  images use exactly 1 mm; do not silently assume equal grid alignment. The latter are
  estimates from simulated noisy multi-echo images and are not exact forward references.
  Public challenge maps omit background susceptibility and contain no R1/R2*/M0.
- Independently executed the two pinned public qsm-forward functions (commit
  `92149383b0f43f47d9a351ceb4f525c0c4ee8299`), unchanged, on both downloaded chi maps:
  3 T, z B0, 2× padding, 1,469,325 brain voxels. Sim1/Sim2 RMS 2.11e-16 / 3.76e-16 Hz,
  maxima 4.47e-15 / 1.38e-14 Hz, after subtracting constant reference differences
  −4.19232e-5 / −1.15128e-5 Hz. Synthetic water/air cases exposed D(0)=1/3 in the reference
  vs zero here (~375 Hz offsets), with <6.83e-13 Hz maximum residual after only demeaning.
  The Python reference is based on Marques 2021; it is not the original MATLAB package.
- Clarified M0 semantics: measured M0 is total voxel density, apportioned among signalling
  memberships; multiplying it by tissue/air fraction again loses mass. Coarsen rates with
  M0 weighting so air cannot lengthen T1. This preserves density volume but remains an
  approximation for multi-exponential relaxation. New synthetic fixtures check these
  invariants and geometry/provenance failures; both real GTs pass fetch/hash/load checks.
- M3 remains open for actual full-head R1/R2*/M0 ingestion and its original MATLAB forward
  check. Their unavailability is explicit in `validation/anatomy/QSM.md`; public-data
  checks do not stand in for those unexecuted steps. M4 and M5 have not started.
- Final validation: **462 passed, 4 opt-in crosschecks deselected** with host Apple GPU
  access in 184.47 s. Thirteen QSM tests are new. Before/after default outputs and
  gradients are bit-identical; both pinned public fetch/load checks pass.


## 2026-10-07 — Alternative anatomy datasets after M3 access limitation

- Continued from `75f2e86`, following the request to find another source/dataset.
  Inspected the NeuroPoly GitHub tree and complete public OSF inventory, the
  Jochmann/Schweser Zenodo ZIP directory/README using range reads, BigBrain-MR's
  metadata/unit notes/example, and AHEAD template v3. No unrestricted complete
  original QSM-source mirror was verified.
- Selected AHEAD v3 (10.21942/uva.19646364.v3, CC BY 4.0) as the most direct
  accessible alternative for brain parameter maps. Downloaded median chi, R1,
  R2* and relative PD: 250,309,250 compressed bytes. All publisher MD5s matched;
  committed compressed/decompressed SHA-256 pins and a repeatable fetch/inspect
  script. Data remain ignored. Four real files share exactly equal 0.5-mm RPS
  grids, shape 394×466×378, and all values are finite; rates/density nonnegative.
- Ran `fetch_ahead.py --offline` successfully against all four actual downloads.
  No simulation/core changes, so no new physics claim, full-suite rerun or default
  snapshot comparison was needed. This checks accessibility/integrity/geometry,
  not physical calibration or scanner agreement.
- Documented source limitations and a concrete route forward: a separate AHEAD
  brain-template adapter, retaining relative-PD meaning and 7 T rates; BrainWeb
  remains the full-head geometry source and QSM retains independent field checks.
  AHEAD lacks skull/air, intrinsic R2 and the original MATLAB ground-truth simulator.
  Its adapter is not implemented and original M3 acceptance criteria remain open.

## 2026-10-07 — Anatomy review, merge, and M3 (AHEAD) / M4 / M5

- Reviewed the second agent's M1-M3 (susceptibility, BrainWeb, QSM): correct kernel and
  shimming, conservative partial-volume assembly, honest limits; 462 tests, defaults
  bit-identical vs main. Merged by replaying its four commits onto main (dropping my duplicate,
  journal conflicts as unions). End-to-end: realistic BrainWeb GRE images at 3 T.
- Fixed the main realism error found in review: CSF/fat used legacy 1.5 T simulator values
  (CSF T2 0.33 s). Now cited: CSF T1 4.4 s (Rooney 2007), T2 2.03 s (Spijkerman 2018), fat
  382/68 ms (de Bazelaire 2004), 1.5 T fat from IT'IS. T2-weighted CSF is now bright.
- M4: within-tissue texture as mean-1 log-normal Gaussian random fields per tissue/property,
  coarse-grid FFT synthesis with exact discrete normalisation; realised SD 95-100 %, correlation
  at lag L within 1.4 %; separate seed; crops share the realisation.
- M3 alternative: `load_ahead` (AHEAD v3 7 T: R1/R2*/PD/QSM); units fixed by Caan 2019 values
  (WM T1 1.14 vs 1.19 s, T2* 28 vs 27 ms) and a positive pallidum; synthetic RPS tests. Found
  that the default T2* policy drops a measured R2* map when chi is on (documented; the
  validation supplies R2' = R2* - R2).
- M5: `examples/anatomy_brain.py` (1.86 M spins, three contrasts, 44 s on the M4 GPU) and
  `validation/anatomy_ahead.py` (7 T GRE showing pallidal/putaminal darkening at TE 20 ms).

## 2026-10-07 — Anatomy viewer; float64 resampling fix; T2* item on the roadmap

- `examples/anatomy_viewer.py`: 50 % iso-surfaces of BrainWeb fractions (skin, skull, CSF,
  cortex, white matter, vessels; 1.5 mm, ~1.2 M triangles) plus the simulated 3 T slab (T2w /
  T1w / shimmed B0) placed where it sits, in a local three.js page (not published: no
  BrainWeb redistribution licence).
- Bug found while building it: float64 BrainWeb resampling raised "fractions sum to more than
  one" (cumulative-sum rounding ~1e-14 vs a 3.6e-15 check). Renormalisation after resampling;
  regression test reproduces 1.2e-14 / 2.9e-14 overshoot without it.
- Roadmap: T2* from measured R2* without double counting, as the next physics item.

## 2026-10-07 — T2* from measured R2* without double counting

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

## 2026-10-07 — Receive phase of loop coils fixed (found while importing full-wave maps)

- Mapping mariepy's circular components onto blochparty showed that nothing in the repo fixed the
  rotation sense of coil phases. A Faraday's-law test (new, `tests/test_loops.py`) shows the
  receive weight is (Bx - iBy)/2; `loops.receive_sensitivities` returned its conjugate, mirroring
  coil phase patterns (magnitudes, RSS and B1+ unaffected). Fixed; defaults unaffected (opt-in).
- On the way, mariepy validated here: B1 maps equal Biot-Savart to 0.3 % at 0.01-0.1 T and depart
  by 14-16 % at 3 T (wavelength); interior E vs Mie for a conducting sphere at 3 T 5.2 % (5 mm) and
  2.5 % (2.5 mm) with the linear basis, 26 % / 10 % with the constant one; port resistance equals
  absorbed power to < 1 %. Noise resistance of a coaxial loop: constant basis wrong at low field
  (no omega^2 scaling), linear basis within 6-9 % of quasi-static at 0.1-1.5 T and +48 % at 3 T
  (converged): quasi-static noise covariance underestimates head-size losses at 3 T.

## 2026-10-07 — Full-wave coil fields imported (mariepy)

- `src/blochparty/em.py`: reads mariepy map files (plus / minus per channel, exp(+j w t), grid in
  x y z) into `FieldMaps`: B1+ = conj(minus)/2 (quadrature drive chosen by size), |B1+| / median;
  receive conj(plus)/2; nearest-inside fill at the skin; noise resistance reader.
  `tests/test_em.py`: rotating-field phase (three phases), quasi-static limit equals
  `loops.field_maps` to 1e-5, orientation and fill. Suite 477 passed.
- `validation/em/run_brainweb.py` (mariepy env, 1b0dc435, 14 min on CPU): BrainWeb 04 at 5 mm,
  3 T. Birdcage: co-rotating mode 1.09 vs 0.17 uT; |B1+| 0.90–1.37x median, centre 1.38x. Eight
  loops: 1.27–1.75 Ohm, centre-sampled E 2–3 % below exact losses (rescaled), noise correlation
  up to 0.40.
- `validation/em/brainweb_demo.py`: T1w GRE with birdcage B1+ and the eight-loop array with
  correlated noise (RSS, Roemer), 37 s; registration check -0.995 (vs -0.93 transposed).
- Limitation kept: `FieldMaps.b1` is real, so transmit phase (pTx) is dropped; roadmap item.

## 2026-10-08 — 3 T WM/GM T2* from whole-brain measurements

- Tissue table: 3 T GM 55.9 +- 1.8, WM 51.5 +- 2.2 ms (Shin et al. 2025, MRM, Table 3; 4 subjects,
  FreeSurfer GM/WM) replace occipital Wansapura 1999 (GM 41.6 < WM 48.4 ms). The abstract swaps
  the labels; table and per-subject rows agree with Peters et al. ISMRM 2006 (GM 59.7 > WM 54.6 ms).
- `DEFAULT_R2STAR_ACQUISITION` is now their protocol (1 x 1 x 2 mm, 43 echoes 3-66.18 ms) instead of
  the assumed 1 mm / 5-40 ms. Round-trip test cases moved to 3 / 5 / 8 Hz/mm (8 now clips).
- Long-TE GRE: T2* factor GM/WM 0.92 -> 1.04; total GM/WM 0.72 -> 0.81 (T1 weighting). Snapshot
  bit-identical (no anatomy in it); suite 477 passed.

## 2026-10-08 — Benchmark 6: radial Pulseq timing (Cima.X)

- `validation/real/fetch_radial.py`, `radial.py`. The import's k at every ADC sample equals the file's
  trajectory to 7e-4 / FOV. Measured effective delays dx +0.20, dy +0.65, dxy +0.05 us (250
  antiparallel pairs, scatter 0.05 us); cross-slice prediction 0.04-0.07 us rms vs 0.42-0.49 us
  without delays.
- Simulating the .seq with imposed delays and re-estimating: exact for zero and equal delays; the
  antiparallel estimator turned out biased for unequal delays with an off-centre object (ellipse:
  (3, -2) -> (3.38, -1.53) us), from opposite sideways shifts of the partners. For a 19 cm disc
  placed like the measured object with 8 loop coils the bias is <= 0.005 us, so the measured values
  stand. Dead end avoided: blaming the GIRF segment averaging (equal delays are exact).
- Not testable here: GIRF shape and B0 terms (no GIRF for this scanner).

## 2026-10-08 — Benchmark 7: measured GIRF (Prisma) through apply_girf

- `validation/real/fetch_girf.py` (1.23 GB, MD5), `girf.py`: Python port of the thin-slice output
  computation; our Meas1 GIRF equals the authors' precomputed one to <= 0.054 below 10 kHz after a
  half-dwell frame shift (delays 1.58 / 1.62 / 4.41 us in our frame, theirs minus 2.5 us).
- Held-out: Meas2 (8 months later, both polarities) predicted from Meas1 through `apply_girf`:
  0.12-0.38 % rms of peak vs 0.67-1.61 % ideal and 0.38-0.63 % pure delay; y worst (its delay moved
  ~1 us). Moments within 0.05 % for all models.
- B0 term: below the drift floor (odd part <= 0.022 rad in 10 ms vs 0.6 rad even part); DC part is
  slice-pair geometry. A units slip (Hz per mT/m used as Hz per Hz/m) caught by the 1e6 % error.
- Finding: blip dephasing (0.53 at k = 371 /m) shows a wider-than-1 mm slice profile.

## 2026-10-08 — CEST: benchmarks 8 (MAGMA, consistency) and 9 (iohexol, Tsat prediction)

- Benchmark 8 (5f6640b): blochparty = BMCTool on the library's APTw .seq to 8e-4; the library's
  L-arginine models miss their phantom (rms 0.05-0.12). New: offsets simulated in their RF frame
  (exact; 100 us steps; 7-8x faster).
- Benchmark 9: raw iohexol references reconstructed with an own twix reader (the prepared .mat
  was ambiguous); offsets batched as spins with the per-pulse phase steps moved into the gaps
  (exact to 3e-5, 58 offsets in 15 s); LM calibration at 1.5 s (rms 0.013 median); prediction of
  4 s: rms 0.022 vs 0.032 for "no change", change correlation 0.75, CEST build-up under-predicted
  1.3-6x (T1 from a single Tsat). Dead ends: a name clash (step) cost one 20-min fit; the
  annulus tube finder picked inter-tube gaps here (tubes taken from the +1 ppm image instead).

## 2026-10-08 — Monte Carlo accuracy and performance review

- Audited spatial stratification/Sobol, tissue and T2′ sampling, NIST estimators, Philox,
  free/restricted diffusion, and Metal execution at baseline `29bae6e`. Findings and staged
  implementation plan: [MONTE_CARLO_REVIEW.md](MONTE_CARLO_REVIEW.md). No simulator or default
  changes; historical scanner benchmark values were not recomputed.
- Confirmed that `mc_magnitude` debiases power, not the subsequent clipped square-root
  magnitude. In 200,000 four-seed trials (mean-noise component SD 1), true amplitudes 0/2/10
  gave corrected means 0.5240/1.6960/9.9485. Both NIST headline SE formulas also omit covariance
  between sphere metrics; recommend jackknifing the aggregate and paired model differences.
- Uniform T2′ diagnostic: 256 deterministic quantiles, 8 seeds, 0–3 T2′, max FID error 0.03818
  despite max seed SD 2.08e-16. Seed replication cannot establish deterministic quadrature
  convergence. Sobol real shifts lose exact dyadic balance; 128-seed analytic Fourier checks
  showed mixed effects on RMSE, so a digital-shift alternative needs protocol-level testing.
- Derived a free-segment bridge-conditioned estimator and checked it using the existing
  Philox samples: complex-variance reductions 1.35/1.54/2.81x at phase variances 0.1/1/4.
  Proposed exact joint-Gaussian free-interval aggregation, pilot allocation, stable sample
  IDs, control variates/multilevel RF corrections, prepared GPU buffers, and restricted-step
  bucketing. These are candidates, not claimed GPU speedups.
- Reproduction: `validation/monte_carlo_audit.py` and its checked-in JSON results. Targeted
  diffusion/restriction/sub-voxel tests: 21 passed, 6 GPU skips initially; one fused case hit
  the sandboxed default compiler cache and passed after redirecting `TORCH_EXTENSIONS_DIR`
  to `/tmp/blochparty-mc-audit-extensions` (22 passed total). MPS was unavailable, so no new
  GPU timings or full NIST runs were performed.

## 2026-10-08 — Benchmark 9 continued: independent T1 from the IR Look-Locker scan

- `iohexol_t1.py`: subspace NUFFT of the archive's IR-LL (K = 6, central slice, fit from 0.15 s):
  T1 0.48-1.46 s, residuals 0.4-1.4 %. K = 4 had 4-13 % (first attempt); edge slices gave T1* above
  the 6 deg / 5 ms bound (lower flip at the slab edges).
- The single-Tsat calibration's T1 was up to 2x short. With T1 fixed to IR-LL and the rest
  recalibrated at 1.5 s, the 4 s prediction: median rms 0.018 (0.022 before, no-change 0.032),
  change correlation 0.81 (0.75), build-up at 4.3 ppm 0.5-1.2x the measured (0.16-0.79x).

## 2026-10-08 — Monte Carlo estimator fixes (from the Monte Carlo review)

A review of the Monte Carlo methods (docs/MONTE_CARLO_REVIEW.md, written against 29bae6e)
flagged the NIST estimators. The parts that change reported numbers are now implemented:

- `nist.mc_power` (unbiased, unclipped |true|^2); `mc_magnitude` is documented as a
  power-debiased approximation (the clip and square root bias it again near nulls). `mc_errors`
  jackknifes the spheres-1–10 mean itself (the old sqrt(Σ se²)/10 dropped covariance), adds the
  jackknife bias-corrected mean and the clipped-pixel fraction per ROI; `mc_paired_difference`
  and a table of paired differences between MC rows sharing seeds; `--save` keeps the seed
  images. Guards: K >= 2 (jackknife of mc_magnitude K >= 3), power-of-two Sobol counts,
  0 < sparse <= 1; the sparse water group generates only its prefix (bit-identical).
- Core bug: an ADC at the end of an RF segment read `spins`, not `step_spins`, so moving spins
  under mapped coils gave zero signal there (the coil placeholders). Test against a moved static
  object; Metal already agreed with the fixed core (9e-7). Snapshot bit-identical.
- Benchmark 1 rechecked with 8 seeds at 256 spins/pixel (RESULTS.md): the historical ±0.19 is
  0.31 with covariance; 4-seed bars differ 0.31 vs 0.19 between disjoint seed sets; clipping is
  ≤ 3 % of ROI pixels. Correction to the earlier reading: the 256 → 1024 shift (1.77 → 1.51 %)
  is mainly MC noise inflating the RMS shape error (1.90 / 1.80 / 1.73 % for 2 / 4 / 8 seeds;
  bias-corrected 1.66 %), not the near-null magnitude bias.

## 2026-10-08 — Roadmap rewritten as a plan; old roadmap archived

- Review of the project state: 110 commits in four days, 484 tests green (3 min 22 s), nine
  real-data benchmarks. The roadmap had become a log: most of its length was verified-numbers
  paragraphs, its open items mixed stale Phase 1 performance ideas (superseded by the Metal
  kernel) with real gaps, and it listed three goals as equals while nearly all work went into
  realism. Two structural points were not written down anywhere: the engine policy ("Metal is
  the engine") leaves the project without GPU gradients, so the differentiable-calibration
  claim in REALISM.md has never been exercised on real data (benchmark 9 fitted by finite
  differences in the core); and anatomy generation, the second pillar, has loaders and texture
  but no sampler.
- `ROADMAP.md` → `docs/ROADMAP_ARCHIVE.md` unchanged (the citable verified record; README and
  REALISM.md now point there for it). New `ROADMAP.md`: the vision (core first on this Mac,
  anatomy generation second, training data and agentic design on top), a testable definition of
  "the core is satisfactory" (Metal parity, Metal adjoint, physics no longer moves the NIST
  benchmark table once the unknown inputs are fitted), working rules for an implementing agent,
  and milestones M0–M8 with design notes, file pointers and verification for each: consolidate
  (engine matrix + raise tests), Metal parity (free pools, motion with the field model), Metal
  adjoint (two-level checkpoints, deterministic per-segment reductions, retire the fused kernel
  after), the NIST digital-twin loop (fit B1+/receive/geometry/timing on one plane, predict the
  other, ablation table against jackknife bars), Pulseq writer with block provenance, anatomy
  sampler + dataset writer + two-sample test, the physics backlog reordered by what training
  data hits first (J-coupled fat, deformable motion, complex B1+/pTx, contrast kinetics,
  orientation-dependent R2*), then training-data/agentic hooks and ports.
- Explicitly not planned: model-based recon for benchmark 3, PDG backend, XCAT, GUI.

## 2026-10-08 — M0: engine capability matrix, guards, quick test subset, Metal baseline

- **Matrix.** `docs/ENGINES.md`: the four entry points (core, fused, Metal, chunked) against 31
  features and combinations, forward and gradients (248 cells), precision per engine, measured
  agreement, and limits that are not physics. `tests/test_engines.py` parses the table and
  checks every cell on a small case: ✓ forward = agreement with the float64 core; ✓ gradients =
  the core's autograd against a central difference along a random direction (case rebuilt at
  θ ± εv), fused / chunked gradients against the core's autograd; ✗ = `NotImplementedError`.
  251 tests, 8 s. Flipping two cells in the document gave exactly those two failures.
- **Found by the matrix, fixed by raising** (no physics added; snapshot bit-identical):
  the core silently ignored `maps=` with free pools; Metal ran diffusion with exact concomitant
  fields, which the core refuses (so it had no reference); Metal returned results detached from
  the graph when inputs required grad (now `NotImplementedError` unless under `no_grad`; e.g.
  `nist_irse.fit_b1` with `BLOCHPARTY_BACKEND=metal` used to die in an opaque autograd error);
  `simulate_fused` lacked `maps` and `simulate_chunked` lacked `m_init`, `scanner`, `maps`
  (TypeError), now accepted and refused. Gaps left for M1: free pools with `FieldMaps` (M1c),
  diffusion with exact concomitant fields (M1b).
- **Agreement** (worst over the matrix): fused float64 3.6e-14 (gradients 9.7e-14), chunked
  3.5e-16; Metal 1.5e-5 signal / 1.7e-5 m on static spins (strong-gradient concomitant case),
  7.4e-5 signal and 1.1e-3 per-spin m with walking spins. The latter is float32 absolute positions
  (restricted 5 µm cylinders up to 5 mm off-centre): Metal walks absolute positions, the core is
  float64; a walk in offsets would fix most of it. Tolerances 1e-12 / 5e-5 / 2e-4 (walking signal)
  / 3e-3 (walking states).
- **Dead ends on the way, all in the test, not the engines:** finite differences that perturbed
  leaves in place missed tensors derived at build time (GIRF output, `subdivide_rf`, a scaled
  diffusion tensor), a carrier leaf derived from the RF leaf, an asymmetric perturbation of a
  Cholesky-factored tensor (8 % off), a saturated bound pool compared against its own maximum
  (1.5e-4 vs 1e-7 of equilibrium), and a restricted-walk D = 2e-9 at which τmax = 62.5 µs divides
  1 ms exactly, so ±ε both moved the sub-step count 16 → 17 and autograd and the (converged)
  finite difference differed by 10 %. Recorded in LESSONS.
- **Tests.** `slow` marker on 21 tests over ~1 s and on the crosscheck tests (a command-line `-m`
  replaces the one in `addopts`, so `-m "not slow"` would have run them).
  `uv run pytest -m "not slow"`: 714 tests in 45 s. Full suite: 735 passed in 3 min 21 s
  (484 in 3 min 8 s before).
- **Metal baseline** (`benchmarks/history.jsonl`, `--backend metal --repeat 3` at 6b21ccb):
  2d-96 0.02 s, 3d-32 0.06 s, 3d-48 0.45 s, 3d-64 2.40 s (8.2e9 spin-segments/s; the fused CPU
  kernel takes 5.36 s on 3d-64). Two bench fixes: the first Metal case was charged ~80 ms of first-dispatch
  setup (now a warm-up run), and GPU times vary (3d-64 2.4–3.4 s over nine runs), hence
  `--repeat N` keeping the fastest (default 1, unchanged).
- README: an Engines section matching the matrix, Metal in the field-model and B1+ paragraphs,
  the quick-subset and Metal benchmark commands, 12 missing modules in the layout table.
  CLAUDE.md points to the matrix.
- Commits: 236ce4d (engine guards), 97f4d25 (matrix and its test), dd1b38b (slow marker),
  6b21ccb (bench warm-up, `--repeat`), 428ff07 (Metal baseline), then these docs.

## 2026-10-08 — M1a: free exchanging pools on the GPU; Metal's float32 drift on RF trains

- **Design change against the plan.** The roadmap proposed a float32 matrix exponential per segment
  (scaling and squaring), cached across identical segments. Prototyped in float32 PyTorch against
  the float64 core first: dense `matrix_exp` was 1e-2 off after 2000 segments of benchmark 9's
  saturation train (2e-2 after 6000), 3.5e-4 on a stiff case (k 5 kHz, 0.8 s delays); and
  benchmark 9's shaped pulses never repeat a segment, so the cache would not help. Chosen
  instead: exp(A dt) applied to the state by a Taylor series of the structured A (no matrix),
  sub-steps of norm ≤ 1.5, truncation at 2^-27 of the magnetization, the common precession of
  free segments rotated exactly, increments added by two-sum (3.3e-5 → 2-3e-7 in the prototype).
- **Kernel** (`_bm1` .. `_bm5`, one per free-pool count, up to five so BMCTool's 5-pool library
  model runs): agreement with the core ≤ 1.3e-6 on 1-5 pools with and without the rate-model
  bound pool, both coil reductions, B1+, m_init, the scanner field model, a CEST spectrum with an
  M0 offset at −300 ppm. Performance: first version 5.5 s on benchmark 9's 15-pulse train (464
  spins); the pool count as a runtime loop bound was the main cost (templated: 2.1 s; an
  array-of-pointers suspicion was wrong). Spins are sorted by estimated cost before the dispatch:
  the M0 offset at −1500 ppm costs ~80 sub-steps per RF sample and stalled 31 lanes of every
  simdgroup it sat in (30k spins 18 s → 3.3 s; 464 spins 2.4 → 1.5 s, the M0 spin's own latency).
- **Found on the way: every Metal variant drifted on long RF trains.** Running the BMCTool
  comparison through Metal, the MT path (existing `_full` kernel) was 1.2e-2 off where the core is
  5.8e-7. Measured: plain, MT and generalized Bloch kernels all drift ~1e-3 per 1e4 RF segments of
  CW saturation, because each RF segment rotates m by a tiny angle and rounds all of it. Fixed
  with the same technique: rotation as an increment (Rodrigues with cos = 1 − s2 φ²), relaxation
  and exchange increments, two-sum into a low part folded back at free segments: ≤ 3e-7. Plain
  benchmark within noise. Free segments still drift ~1.5e-5 per 1e5 (left as is: the hot path).
- **Validation through Metal** (`BLOCHPARTY_BACKEND=metal`, new in iohexol.py, magma.py,
  compare_bmctool.py): benchmark 9 reproduces its prediction table digit for digit, Z within
  3.5e-7 of the core; one fit iteration 141 s → 5.7 s, 40-pulse prediction 114 → 8.6 s, the
  whole `predict` in 27 s. Benchmark 8 reproduces its table, Z within 8.3e-6. BMCTool: Lorentzian
  MT 6.5e-7, 5-pool CEST 8.3e-7, 30 offsets 12.8 s → 0.2 s. compare_bmctool.py could not run as a
  script (entry point before a function it calls); fixed.
- Left (ROADMAP M1a): free segments with fast exchange cost sub-steps ∝ k dt (benchmark 9's tube 8,
  k = 40 kHz, ~280 per 5.2 ms TR); they depend only on dt once the precession is factored out,
  so an exact propagator cached per spin would make them one small product.
- Tests: 761 passed in 3 min 18 s; snapshot bit-identical (core untouched).
- Commits: ce67969 (RF drift fix), e93e85d (free pools on Metal), 7419c23 (validation scripts),
  4173cfa (results), then these docs.

## 2026-10-08 — M1b: moving spins under the scanner field model

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

## 2026-10-08 — M1c: free pools with motion, diffusion, FieldMaps and the generalized bound pool

- Core: the free-pool branch of `simulate` moved after the shared spin setup, so the exact engine
  gets walks, motion, the field model and maps, placed per segment for moving spins. The
  generalized Bloch bound pool's modes became 2J rows of the exact matrix. Metal: `_bm*_diff`
  kernels; moving spins not cost-sorted (walks keyed by spin index). Metal still refuses free
  pools + generalized bound pool: with J = 30 modes the Taylor state would carry ~60 extra floats
  per vector, five vectors per thread; left open in ROADMAP M1c.
- Checks: maps = per-spin fields (1e-14); constant rigid shift = moved phantom (1e-13); inert pools
  = single-pool path with the same motion / walk (3e-7); generalized + free pools = a SciPy ODE of
  the mode equations written independently (2e-8); → rate model off resonance (2e-3); Metal = core
  (1e-5) on motion, diffusion, static and moving maps.
- Found: the core's generalized split engine is ~2e-4 off the exact solution in Mzb on a
  Lorentzian random case, because it sub-steps to h ≤ T2b / 2 regardless of segmentation (error
  falls 1.5x, not 4x, per halving); the exact engine matches the ODE to 3e-13. Not changed (the
  split engine is the fast imaging path and Metal mirrors it); an inert free pool routes a phantom
  through the exact engine. Also: subdividing a sequence renumbers its segments and changes the
  diffusion walks (a 1e-4 Monte Carlo difference that first looked like an engine discrepancy);
  `rf(0.5, ...)` is thousands of raster steps unless `n_steps=1`.
- 780 tests in 3 min 33 s; quick subset 755 in 55 s; snapshot bit-identical; plain Metal benchmark
  unchanged (3d-64 2.50 s); shader compile 0.9 s with 10 free-pool kernels per reduction.
- Commits: b8ed082 (feature), then these docs.

## 2026-10-09 — M1c: the cut-free lineshape (`BoundPool(cutoff=None)`) on core and Metal

- Question: free pools + generalized bound pool do not fit the `_bm` kernels' registers. Measured
  first how much the generalized model changes results when free pools are present (core, WM-like
  tissue with amide and myelin-water pools): a 36 × 50 ms, 2 µT CEST saturation train differs by
  1.2e-2 in Z at ±3.5 ppm between rate and generalized model, 5e-6 with a Lorentzian pool, so the
  gap is the near-resonance super-Lorentzian convention (1 kHz cut vs the tapered Green's
  function), not the pool's rotation. Not negligible: 2.3e-3 on MTRasym, a tenth of what
  benchmark 8 measures.
- Fix (A): `cutoff=None` evaluates the rate model's lineshape from the generalized model's modes,
  T2b Re Σ c_j / (p_j + i x) (`exchange.modes_lineshape`); Metal lineshape kind 3 reusing the
  `line` / `log_f` buffers (the table becomes the mode table), no new kernel variant; the fused
  kernel raises. With it rate = generalized to ≤ 5e-5 at 447 Hz to 10 kHz on the saturation train.
  Modes vs quadrature of the tapered Green's function's transform 1.3e-4 to 10 kHz; Metal `_full`,
  `_bm1`, `_bm2` vs core 5e-7; snapshot bit-identical; new engine-matrix row; 11 + 4 + 2 new tests.
- Found on the way: on resonance the rate model needs pulses long against the memory (3 ms), not
  T2b; the cut approximates short hard pulses better than the cut-free lineshape (0.2 ms / 60°:
  1.4e-3 vs 2.3e-2). So the remaining Metal exception is exactly the short-intense-pulse regime,
  and plan (B) for it (operator splitting with the modes in the `_gen` device buffer) plus when it
  would be needed are written into ROADMAP M1c.
- Dead end: a gradcheck of the new lineshape w.r.t. T2b failed with eps 1e-6 on T2b = 1e-5 (a 10 %
  step); scaled multiplier instead.
- Merged into main on 2026-10-09 (branch worktree-taper-lineshape, ae26e83), after the M2 adjoint;
  the merge commit adds the cut-free lineshape to the Metal adjoint (lineshape kind 3 and its
  derivatives in `lineshape_d`), checked by the engine matrix's gradient cell against the core.

## 2026-10-09 — M2: the Metal adjoint (first version)

- **What.** `simulate_metal` differentiable through a hand-written adjoint on the GPU: the reverse
  sweep of `fused/kernel.h` ported lane by lane (plain Bloch, both concomitant-field modes, the
  rate-model bound pool with its exchange-step reverse and lineshape derivatives), in a
  `torch.autograd.Function`. Scope as planned; moving spins, free pools and the generalized bound
  pool raise. The forward's two-sum low parts now persist across dispatches, which makes the
  forward bit-identical for any chunk size and the checkpoints exact.
- **Accuracy.** Against the core's float64 autograd, per input: 1e-7 to 3e-6 on random sequences,
  2e-5 on a 1600-segment RF-spoiled GRE with 4 coils, ≤ 1.2e-4 with the bound pool, where the fused
  kernel's float32 adjoint is up to 1.5e-3 off (the compensated forward helps). Bit-identical for
  any `checkpoint_every` (64, 128, 640) and run to run. Gradient descent: WM/GM optimum 16.110° vs
  16.109° analytic (0.8 s), Ernst angle to 0.01°. DESPOT1 pair only to 0.05°: its CRB is a
  finite-difference Jacobian of float32 signals (needs `rel_step` 1e-2; at 1e-4 it landed at
  (3.7°, 38°) instead of (4.1°, 23.6°)).
- **Speed (3d-48, same session).** First version: forward 0.49 s, backward 4.2 s. Separate replay
  kernel: small gain. Every 4th state stored, blocks of 4 recomputed into threadgroup memory: the
  replay's DRAM traffic /4 (2.65 → 2.2 s). Sensitivities in registers: slower (occupancy); read from
  memory, coil gradients in registers only when asked for. Per-simdgroup partials instead of a
  barrier per segment. One parameter upload per pass. Now forward + backward 3.3 s on 3d-48 (6.2x
  the forward), 22 s on 3d-64 (~7x), 3-4x on 2d-96 / 3d-32; the 5x target is not met. The machine
  was thermally throttled by then (the 3d-64 forward at 3.4-3.6 s instead of 2.4 s, the old and new
  kernels alike), so only same-session ratios are meaningful.
- **Dead ends.** All gradients but λ and PD came out ~0: idle threads (past the last spin, index
  clamped to 0) overwrote spin 0's replay states. The matrix test's Metal gradient cells all failed
  at first: its loss drew random weights in the output's dtype (float32 vs float64 → different
  numbers). Transcendental caching in the reverse made no measurable difference.
- **Open.** Speed to 5x (next candidates: the ADC term's per-sample coil loop, register pressure of
  the bound-pool reverse, a fused replay+sweep for small intervals). Retiring the fused kernel: it is
  still the only fast float64 path with gradients, which DESPOT1-style designs (finite differences
  inside the objective) need; the core does it in float64, slowly.
- 797 tests (4 min 43 s on the throttled machine); snapshot bit-identical.
- Commits: 2519392 (adjoint), 1ceea1d (bench, examples), then these docs.

## 2026-10-09 — M2 finished: DESPOT1 through Metal, adjoint speed, the fused kernel retired

- **Merge.** The CEST branch (cut-free lineshape, `BoundPool(cutoff=None)`) merged as 6db590a, with
  the Metal adjoint extended to it; 814 tests, snapshot bit-identical; worktree and branch removed.
  A test run that took 269 s instead of 16 s right after the merge did not reproduce (19.8 s).
- **DESPOT1 through the GPU adjoint** (f8f6cde). `design.jacobian(method="fd4")`, the fourth-order
  stencil (convergence orders 2 and 4 checked). Through Metal at `rel_step` 5e-2: the pair 0.002°
  from the closed-form grid, its closed-form bound 1.3e-7 above the optimum, 4.7 s. Across steps
  2e-2 to 1e-1 the pair moves 0.0003–0.03° while the bound stays within 3e-6 of the optimum: the
  bound is flat along a2, so the parameter error is the wrong yardstick. Central differences at 1e-2
  had been 0.045° off; the core in float64 gets 0.0002° in 74 s.
- **Adjoint speed** (8af3bee). Per-kernel timing showed the replay at 1.5-2 forwards (it wrote 8
  rows per block start: m and the two-sum low parts, ~106 GB on 3d-48) and the sweep dominated by
  the coil term (adjoint 1.97 s with 8 coils, 0.84 s with a loss on m only). Now the replay keeps m
  only (3 rows, 4 with the bound pool; the sweep restarts each block with zero low parts, one ulp,
  still K1-independent), the samples' source terms are formed once per 4-segment block with the
  coils outermost, and the coil gradients are accumulated during the block recomputation (no extra
  step per sample). Alternating A/B against the old module on 3d-48, 8 coils: backward 2.70 → 2.15
  s (T1), 2.79 → 2.48 s (flip), 3.61 → 3.20 s (all); sweep alone 1.97 → 1.26 s (8 coils), 2.59 →
  1.81 s (16). Gradients within 8e-7 of before. The GPU was shared with another load (2x swings),
  so only alternating comparisons count. `bench.py --grad --backend metal` afterwards (coil, flip
  and T1 gradients): 3d-48 3.2-4.1 s against a 0.42 s forward (8-9x), 3d-64 20.6 s against 3.8 s
  (5.5x); `checkpoint_every` 384 to 4096 changes nothing. The 5x target holds for per-spin
  gradients only; left open.
- **Fused kernel retired.** Tests: shared helpers moved to `tests/helpers.py`; every `SIMS` entry
  and FD check that ran through the fused kernel now runs on the core (the physics tests' job) and
  engine agreement lives in the matrix; fused-only tests deleted; `test_phantom_fused.py` became
  `test_phantom_engines.py` (realistic phantoms through Metal: forward 1e-6 to 3e-6, gradients
  5e-7 to 1.9e-6 per input) and the integration test points at Metal (32000 RF segments: flip
  gradient 8.9e-4, where the core in float32 is 5.4e-3 off; tolerance 3e-3). Design tests: the
  TSE train (1.5 s) and the TR-budget design (36 s) on the core, whose float64 their checks need
  (through Metal the constraint held only to 8e-5); gradient equalities against Metal (1.5e-5,
  4.2e-6, 2.2e-6 measured). Nine test files would have been skipped wholesale by their
  `importorskip("blochparty.fused")`.
- **Validation re-run on Metal**: NIST benchmark 1 table digit for digit (124 s per version
  against 598 s), benchmark 2 at the table's settings digit for digit (64 s against 232 s), and
  incidental MT but for one cell on a rounding boundary (97.2504 %). Benchmark 1's convergence rows
  turned out to predate the T2 correction (16 x 1: 16.72 % now, on both engines, not 14.35 %).
  `examples/gre3d_128.py` 192 s on the GPU (6 min fused).
- **Dead ends and corrections.** A `.double()` on complex coil gradients showed a 1.0 error that
  was the measurement. A rename in `nist_tse.py` made its own `simulate()` recursive (engine now
  `engine`). Wall times of 13 to 88 min for runs with seconds of CPU were the Mac sleeping
  (Maintenance Sleep with 180 s DarkWake windows; pytest's own clock said 338 s for an 88 min
  suite), not the engines and not the swap I suspected first.
- 660 tests (5 min 38 s by pytest's clock; 814 before, the difference the fused kernel's own tests);
  core snapshot bit-identical (the fused entries dropped from it).
- Commits: 6db590a (merge), f8f6cde (DESPOT1 through Metal), 8af3bee (adjoint speed), 2442508
  (fused kernel retired), 5c38a1d (snapshot), f659d4a (docs).

## 2026-10-09 — M4: Pulseq out

- **Provenance.** `SequenceBuilder` records each call's events and segment span in
  `meta["blocks"]` (plus its `System`); `subdivide_rf` remaps the spans; `read_pulseq` records the
  source block per segment and keeps the source sequence for re-export. `apply_girf` and the drift
  transforms carry `meta` through `replace`.
- **Writer.** `write_pulseq` replays the records, which must rebuild the sequence exactly (a
  sequence changed after building raises), and writes Pulseq 1.5 text itself at full precision:
  pypulseq's writer formats amplitudes as `{:12g}` (six digits, ~5e-7), which alone rules out a
  1e-9 round trip. One Pulseq block per builder block; a block with several ADCs (the EPI train) is
  cut between them on the gradient raster with gradients continuing across the cuts. Raises for
  ideal spoilers (`spoil="drop"`), the ideal `rf` / `readout` calls, instantaneous gradients and
  off-raster timings: flat sequences without provenance are not written, a departure from the plan,
  because instantaneous steps have no Pulseq form and an approximation would not be lossless.
- **Reader.** `read_pulseq` now reads with `remove_duplicates=False`: pypulseq's default rounds the
  libraries (amplitudes six digits, timings 1 us) before merging. All 14 scanner files in
  `validation/real/data` (to 1.1e8 segments) read bit-identically either way (checked on the mini).
- **Numbers** (tests/test_pulseq_write.py): GRE (ramps, slice selection, RF spoiling), TSE, bSSFP,
  EPI: same segment counts, signals equal to 6e-16..5e-14 after write -> read_pulseq, pypulseq's
  `check_timing` passes, k-space at every ADC equals its `calculate_kspace` to 4e-13..5e-11; a
  pypulseq-made file re-exports bit-identically, a scanner file to the last bits (1.7e-18 s in
  segment durations: edges are absolute times from rounded delays).
- **Dead ends.** The file read back with six-digit values: pypulseq's reader, not the writer.
  Block 3 of the EPI file had a 2.16 ms event in a 0.21 ms block: `[GRADIENTS]` and `[TRAP]`
  share one id space. pypulseq hung on the first file: its shape loop never ends at end of file.
  Two float32 slips via `torch.as_tensor(float)` (a block duration off the raster, a receiver
  phase 3e-8 off). The builder's bit-identity test hashed `meta`, so provenance changed its digest
  while the segments did not: the provenance is left out of the hash.
- **Machines.** From today long runs go to a Mac mini (M4, 16 GB, `ssh mini`): the full suite ran
  there in 3 min 42 s (681 tests). The laptop's slow wall clocks overnight were closed-lid sleep
  and a GPU shared with a video-decoding app, not the code.
- Commits: 59fca2a (GIRF response in blocks: the Prisma GIRF on a 4e5-segment sequence needed
  2.8 GB per pair before), 788401e (writer, provenance, tests), f6ca6f1 (docs).

## 2026-10-09 — M3: the digital-twin loop on NIST benchmarks 1 and 2

- **Reduced model for fitting** (`validation/real/nist_twin.py`, abd5675): one column of spins per
  sphere at its ROI centre (slab x one pixel along the readout), only the TR that acquires the
  k-space centre after its warm-up TRs, the two ADC samples either side of k = 0 per echo. 25k
  spins and 1.7e5-2.7e5 segments instead of 85k-174k spins and 4.3e6 segments per series: a
  gradient through the Metal adjoint costs seconds. Against the full simulation: 0.5-1.0 % (shape)
  on benchmark 2's spheres 1-10, <= 1.4 % on benchmark 1's long-T1 spheres 1-4. A 3 x 3 grid of
  columns at half the slab resolution was worse (resolution, not coverage).
- **Which inputs.** Scanned before fitting: per-axis gradient delays of 10 us move nothing on y and
  z and < 0.1 % on x; the roadmap's suspect for benchmark 2, a longer first refocusing interval,
  does nothing without off-resonance and < 0.1 % with 50 Hz (and the files have no gradient-free
  gap to stretch). So the twin's input is B1+ alone: five parameters, axially symmetric about the
  bore so one sagittal plane fixes them (test). Receive maps from the data (the roadmap's coil
  image / RSS) have unit norm and add nothing to an RSS image, so the absolute metric is reported
  with uniform receive and with a smooth receive profile from the water fill (which carries the
  water's transmit shading too).
- **Fits** (reduced model, L-BFGS through the adjoint, gradients checked against differences to
  1e-4..3e-3): on benchmark 1, 1.05 -> 0.50 % in either branch of |B1+ - 1| (field 0.93-1.09 on
  its spheres, 1.21 at isocentre falling off radially: centre brightening); on benchmark 2, 2.26
  -> 1.57-1.66 %, but its field is 0.70-0.91 or 0.99-1.26 on the T1 plane.
- **Full-model predictions** (Monte Carlo, 256 Sobol spins per pixel, 4 seeds, jackknife SEs, on the
  Mac mini). Benchmark 1 from the benchmark-2 fit: 13.57 ± 0.31 % against 1.77 ± 0.31 % nothing
  fitted (the TSE's B1+ is wrong on the other plane); paired ablations around it: water fill
  +0.16 ± 0.08, ideal slab +11.67 ± 0.32, concomitant fields -0.01 ± 0.00, the Prisma GIRF of
  benchmark 7 +0.10 ± 0.07. A refocusing-only amplitude factor on top of the benchmark-1 field
  converges to 1.003 on benchmark 2: the TSE fit's gain is not refocusing calibration but five
  smooth parameters absorbing sphere-specific residuals.
- **Temperature** as an input (Stupic et al., MRM 2021: at 3 T the MnCl2 spheres' T1 +2.7 %/C and
  T2 +1.6 %/C over 16-26 C; NiCl2 insensitive, <= 4 % non-monotonic, so benchmark 1 cannot see it):
  fitted on benchmark 2's reduced model, +2.6 C with B1+ = 1, spheres 1-10 2.26 -> 2.19 %; with
  benchmark 1's B1+ field 20 C stays best (2.15 %). Over the plausible range it moves spheres 1-10
  by < 0.1 point and 11-14 by ~1. What is left on that plane sits in spheres 5 (reference T2
  withheld by the manufacturer, Jiang 2016 instead) and 9-10 (short T2, reached by the water fill).
- **Found on the way.** `write_pulseq` labelled off-centre refocusing pulses as excitations (a phase
  ramp cancels |sum s dt|; fixed by `pulseq.flip_angle`, ede53ab). The Prisma GIRF needed 2.8 GB
  per input/output pair on a NIST sequence (fixed by blocks, 59fca2a).
- **Held-out prediction** (benchmark 2 from benchmark 1's B1+, full model, Mac mini, 90 min):
  1.83 ± 0.03 % against 2.20 ± 0.02 % nothing fitted (paired -0.38 ± 0.04), better on every sphere
  but 1 and 5, sphere 11 7.8 -> 3.8 %. In sample on benchmark 1: 1.77 ± 0.31 -> 1.13 ± 0.25 %
  (paired -0.64 ± 0.16). Ablations around the twin: water fill +0.25 ± 0.07 (benchmark 2) / +0.01
  (benchmark 1), ideal slab +6.8 / +24.5, concomitant fields -0.01 / 0.00, the Prisma GIRF of
  benchmark 7 -0.02 ± 0.03 / -0.03 ± 0.10: what is not in the model moves the prediction less than
  its error bar (condition 3). Spheres 12-14 stay at 13-17 %, sphere 5 at 3.6 %.
- **Absolute metric**: 7-8 % with one global scale and uniform receive, ±1 point with the twin;
  a water-fill receive profile helps where the transmit shading does not double up. Receive
  shading is the input these data cannot separate (needs a B1+ map or PD scan).
- Runs moved to the Mac mini midway (same seeds, nothing-fitted row identical to the laptop's:
  2.20 ± 0.02 %); one prediction ~40-90 min there against ~2.5 h on the laptop.
- Commits: abd5675 (reduced model, fits), 59fca2a (GIRF blocks), ede53ab (flip angle of
  phase-ramped pulses), then the write-up.


## 2026-10-09 — M5: tissue schema, head sampler, dataset writer, all 20 BrainWeb subjects

The parts of M5 that need no outside access, done while the user was away.

- **Schema (P0).** Every signalling BrainWeb class now has cited MT and diffusion, so
  `to_phantom(mt=True, diffusion=True)` works on a whole head (it raised for everything outside
  WM/GM/CSF, and those had illustrative placeholders: f 0.16/0.07, exchange 40/s, T2b 12 us). At
  3 T: WM/GM bound fraction 0.1277 ± 0.0041 / 0.0673 ± 0.0015 (Yarnykh 2016, 8 subjects),
  bound-to-free rate 19.0 /s and T2b 9.7 us (Yarnykh 2012's whole-brain medians); muscle f 0.084 ±
  0.006 and T2b 5.9 ± 0.2 us (Sinclair 2010), its rate assumed (it did not converge in muscle);
  fat and CSF without a semi-solid pool (Chen 2010; CSF MTR 0.01). Diffusivity: WM 0.70, cortical
  GM 0.89 (Helenius 2002), CSF 3.04 (water at 37 C, Holz 2000), muscle 1.78 (Li 2014), blood 1.54
  (Funck 2018) um²/ms, fat 1.83e-5 mm²/s (an upper bound, Weidlich 2018). Skin, connective tissue
  and dura take muscle's values, marrow fat's, flagged; qMT measured at 3 T is flagged at 1.5/7 T.
  The k convention was the trap: Yarnykh's k is free -> bound and his R = k(1-f)/f is our
  `BoundPool.k` (LESSONS). Default `to_phantom` on subject 04 has the same SHA-256 before and
  after; core snapshot bit-identical. `DigitalPhantom` (anatomy, one field's table, `to_phantom`
  options, `meta`) gives `to_phantom` bit for bit.
- **Sampler (P1).** `sample_head(seed)`: one of the 20 subjects, a cubic B-spline displacement
  (control points N(0, (3 mm)²) every 40 mm: ~1 mm SD, Jacobian 0.83-1.21 on subject 04, halved
  until > 0.2), a per-subject table (`draw_tissues`: log-normal with each cited mean and SD; 1800
  standardised draws over 100 subjects have mean -0.03, SD 1.00), the M4 texture and a spin seed,
  all from one `SeedSequence`; `meta` holds every draw and is JSON. Checks: integer shifts exact,
  B-spline weights sum to 1 (1e-15), an affine field's Jacobian equals det(I + A) to 1e-12, equal
  seeds give equal heads. 4 s per head at 2 mm, 16 s at 1 mm with the full-head B0.
- **Dataset writer.** `io.write_dataset` writes ISMRMRD HDF5 through ismrmrd-python (new extra
  `io`): one acquisition per ADC run with its k-space (`sequence.kspace_trajectory`, the M4 test's
  rule moved into the library and vectorised; still equal to pypulseq's `calculate_kspace`),
  encode indices, centre sample, ACQ_IS_REVERSE for EPI's reversed lines; a `/provenance` group
  with the sequence arrays and tagged-JSON meta, the `.seq` file, the head's table and draws,
  coils, noise, seed, engine, git commit and versions. Round trip on GRE/TSE/EPI with 4 coils:
  signal equal to its complex64 cast, k-space assembled from ISMRMRD fields alone equal to
  `cartesian_kspace`, the rebuilt sequence equal and writing the same `.seq` text.
- **All 20 BrainWeb subjects** fetched on the mini (~1.4 GB each) and pinned. The raw files have no
  header and the fetcher writes subject 04's origin for all; the brain centroids agree to 2.6 mm,
  as a shared origin requires. WM 618 ± 42, GM 969 ± 55, CSF 385 ± 43 mL over subjects.
- Full suite on the Mac mini: 701 passed in 3 min 56 s.
- **Example** (`examples/sampled_heads.py`, Mac mini): seeds 0-9 as T1w GRE, T2w TSE and T2*w GRE at
  3 T with scanner-playable sequences, 8 coils and noise; 8.8-10.0 million spins per head, 44-58 s
  per image on Metal, 27 min for 30 ISMRMRD files (148 MB). The first run's GRE files had no `.seq`:
  a round TE leaves a block half a dwell off the 10 us raster (LESSONS); TE 2.505 / 20.005 ms fixed
  it and the run was repeated, so the figure comes from the committed script. `--replot` redraws
  it from the files through `read_dataset`.
- **Left for a decision** (ROADMAP M5, outside the brain): skin/connective/dura relaxation are
  muscle's, marrow's is the legacy fat prior, adipose tissue is pure lipid; changing any moves
  default `to_phantom` output. The realism number needs fastMRI access.
- Commits: 99ef90a (schema values), ecc1ddb (DigitalPhantom, sampler), ae55fae (writer,
  `kspace_trajectory`), 69b2301 (20 subjects), 585e857 (example).

## 2026-10-09 — fastMRI fetch started; tissue database begun (research only)

- fastMRI brain: `validation/realism/fetch_fastmri.py` streams `brain_multicoil_val_batch_0.tar.xz`
  (curl -> tarfile, xz on the fly) and keeps AXT2 (15 GB), AXT1/AXT1PRE (4 + 4) and AXFLAIR (7) on
  the Mac mini (`~/data/fastmri_brain`, outside the repo) with a manifest of field strength,
  protocol, TR/TE and matrix per volume. Tested on a synthetic archive; running at ~5 MB/s (network
  bound, ~2 h). The signed link lives in a mode-600 curl config on the mini, never on a command
  line; `*.eml` is git-ignored. Links expire 2026-11-06.
- Tissue database (user's request): sources located, nothing written yet. IT'IS v4.2 relaxation
  times (T1/T2 at 1.5/3 T, ~30 tissues with own data; others are IT'IS proxies) and v5.0 water
  content (mass %, >100 tissues) come as JSON from the IT'IS site; open full texts with tables:
  Jordan 2013 (MSK 3/7 T, PMC3310288), Edden 2010 (breast 3 T, PMC3154635), Ma 2013 (MRF, PMC3602925),
  Rane 2013 (7 T blood, PMC3561475); abstracts carry Rooney 2007 (T1 power laws, CSF 4.3 s), Lu 2004
  (blood T1 vs Hct at 3 T), Wansapura 1999 (its abstract swaps T2 and T2* labels). Next: schema,
  loader and tests (`src/blochparty/tissuedb/`), then rows.

## 2026-10-10 — Tissue-property database (`blochparty.tissuedb`, branch `tissuedb`)

- **What**: a sourced table of MR tissue properties, package data in `src/blochparty/tissuedb/`:
  `measurements.csv` (1859 rows, one published value each, SI, with source, table/page, field,
  species, state, temperature, method, n and SD kind), `sources.csv` (158, each with DOI or URL),
  `tissues.csv` (126 dotted ids, one per BrainWeb class marked). `query`, `pooled` (human in vivo
  at the field first; every fallback marked assumed; T1/T2/T2* never borrow another field) and
  `tissue_table(field)`, a candidate {class: TissueProperties} that falls back to DEFAULT_TISSUES
  per field. DEFAULT_TISSUES is unchanged (snapshot bit-identical; pickle-equal before/after).
- **Why**: the tissue table is the largest remaining unknown outside the brain (ROADMAP M5 left
  skin/connective/dura/marrow/skull as proxies) and the values in anatomy.py had no machine-readable
  provenance. Switching the default to the candidate is the user's decision.
- **Rows**: the values anatomy.py uses (round trip tested per value); IT'IS v4.2 relaxation
  (T1/T2 at 1.5/3 T) and Gabriel Cole-Cole dielectric properties at 64/128/298 MHz, and v5.0 water
  content (`tools/tissuedb_import_itis.py`, reproducible, 212 blocks kept, 459 skipped as proxies
  or empty, logged with download SHA-256); then eight literature passes (open PMC tables, classic
  abstracts, body relaxometry incl. 0.55 T, head gaps: eye, cortical bone UTE, skin, glands; deep
  GM and QSM; blood vs oxygenation and fat fractions; MT, DTI and PD; organs incl. myocardium and
  5/7 T abdomen). Property x field coverage: T1 463 rows, T2 281, T2* 137, MT 185, MD/FA 165,
  PD 65, PDFF 72, QSM 120, dielectric 300, water content 68.
- **Checks** (`tests/test_tissuedb.py`, 55 tests): schema (keys, units, fractions, field, no
  duplicates), unit magnitudes per property, pooled T2 < T1 and T2* <= T2, T1 rising with field
  within a source, human in-vivo sources within a factor of two (fractions < 0.2: within 0.1), the
  round trip to DEFAULT_TISSUES, Li 2014's 8 muscle MDs reproduce 1.78 +- 0.08 um^2/ms, the
  Cole-Cole evaluation against the IFAC/Gabriel values at 128 MHz (GM 73.5 / 0.59 S/m, WM 52.5 /
  0.34, CSF 84.0 / 2.14, muscle 63.5 / 0.72), and tissue_table(1.5 / 3) through
  to_phantom(mt=True, diffusion=True). Eight documented exceptions, method- or region-driven (MRF
  vs multi-echo T2 in kidney cortex/medulla and muscle, UTE echo times in lung, atlas-ROI vs
  age-fit cortical FA, disc compartments, Peters' CSF-contaminated WM T2*, Ma 2019 lens T1);
  compilations are left out of the T1-rises-with-field check (IT'IS pools other studies per field). A spot check of 8 random agent rows against the PMC XML: all 8 values match.
- **What would change at 3 T** if the candidate became the default (full table in the PR): WM T1
  1084 -> 889 ms and GM 1820 -> 1344 ms (human in vivo replaces ex vivo bovine), WM PD 0.77 -> 0.70,
  fat PDFF 1 -> 0.89, marrow PDFF 1 -> 0.39 (vertebral red marrow pooled with yellow), skull gains
  signal (PD 0.23, T1 237 ms, T2* 0.58 ms), CSF T2* 2.03 s -> 0.20 s (Shin 2025, flow/diffusion
  included), muscle MD 1.78 -> 1.22 um^2/ms (pharyngeal constrictor pooled with thigh). Weak cells:
  WM T2 86 ms is pulled up by SyMRI (flagged), blood MT comes from Stanisz 2006 with a suspect T2B.
- **Corrections found on the way**: Rooney's CSF T1 is 4.3 +- 0.2 s (the code quotes 4.4 via
  Spijkerman); Caan 2019 n = 6 analysed of 8; Jordan JMRI 2013 (the chi citation) is a breast-B0
  paper quoting Schenck 1996; `validation/anatomy_brainweb.py` takes Wansapura's T2 in the
  abstract's (swapped) order. Lessons in LESSONS.md.
- **Dead ends**: Stanisz 2005, Gold 2004, Bojorquez 2017, Sled & Pike 2001, Gloor 2008 and de
  Bazelaire's full text are behind Cloudflare/captcha here; Stanisz's 3 T values were confirmed from
  the group's ISMRM 2006 abstract, the 1.5 T values remain unverified. No skin/scalp T1/T2 at 1.5/3 T
  from a primary source, and no dura, connective-tissue or mucosa relaxometry at all.
- Commits (branch `tissuedb`): ba847f8 (package, IT'IS import, checks), 655ffd4, 67b68d8 (deep GM),
  6050eec (head gaps, blood/fat), 247abc7 (organs), and the MT/diffusion/PD rows with this entry.

## 2026-10-10 — Wansapura 1999 GM/WM T2 swap in the human 3 T comparison

- `validation/anatomy_brainweb.py` propagated Wansapura et al. 1999 (JMRI 9:531, PMID 10232510)
  3 T means through the BrainWeb protocols with T2 WM 110 / GM 80 ms, the abstract's order. Every
  other 3 T measurement has GM > WM (Stanisz 2005 99 / 69 ms), and the abstract also mislabels its
  FLASH T2* values as "T2"; found while building the tissue database (`tissuedb` branch). Now
  WM 80 / GM 110 ms, with the source string saying the order is assigned by physics.
- Full text not read: Wiley serves a bot-verification page (not bypassed), ResearchGate's table
  page returns 403, Europe PMC has no open copy, Bojorquez 2017's review does not quote the values.
- Re-run on the Mac mini (7.5 s): human 3 T GM/WM T2 0.688959 -> 1.561440, PD 0.920534 ->
  1.168634, T1 0.744074 -> 0.745675 (T2' = 1/(1/T2* - 1/T2) moves with T2). Every other result
  in `validation_results.json` is identical. Independent check: the closed-form spin-echo ratio
  (before x exp(2 TE (1/80 - 1/110 ms))) gives 1.561440 and 1.168634 to 6 digits; Ernst x
  exp(-TE/T2*) gives T1 0.745521 (0.02 %, finite T2' sampling).
- RESULTS.md also carried a stale T1 ratio (0.709695): the 2026-10-08 T2* update had moved it to
  0.744074 without a re-run. No core code changed, so no snapshot.
- Commits: 049efdf (script, RESULTS), b7cf67a (LESSONS).
- After the tissuedb merge (#7) its LESSONS line saying the script uses Wansapura's abstract order
  was corrected to the new assignment.

## 2026-10-11 — Refactoring plan for the repository layout (plan only)

- Read the whole tree at `f983c00` and wrote `docs/REFACTOR_PLAN.md`: a phased proposal for a
  clearer layout, structure only, no physics and no public-API change. Nothing implemented.
- What it measures: `metal/__init__.py` 2095 lines, 1465 of them (70 %) Metal Shading Language in
  four templated string literals; `anatomy.py` 1126 lines with five concerns; the core and the Metal
  host each re-derive the same input preparation (defaults, bound-pool expansion, `m_init` split,
  static `FieldMaps` sampling, lineshape / mode tables, the same `ValueError` strings); `rel()`
  defined eight times across tests; five `sys.path.insert` hacks from tests and tools into
  `validation/` and `examples/`; `Sequence.meta` with 28 string keys read in eleven files; the
  README layout table missing `constants.py` and `tissuedb/`.
- What it proposes, by payoff over risk: (1) shaders to `.metal` files and the Metal host split;
  (2) `tests/conftest.py`, one `helpers.py` + `cases.py`, declared `pythonpath`; (3) one study per
  `validation/` folder, generic Monte Carlo estimators and the twix reader into the package;
  (4) an `engines/` subpackage with a shared `prepare` step and one capability table checked
  against `docs/ENGINES.md`; (5) `anatomy/` split with `tissuedb/` inside; (6) `sequences/` with
  the builder DSL out of `sequence.py`, Pulseq read/write split, the `meta` keys documented once;
  (7) generated README layout table, journal by month, ruff; (8) `design/`, optional.
- Rules: `git mv` and edits in separate commits, snapshot bit-identical per commit, the four living
  documents updated per phase, historical documents left alone. Six open decisions listed at the end.
