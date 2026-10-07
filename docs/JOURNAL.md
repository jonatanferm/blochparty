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
