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

## 2026-10-10 — Realism score v0 (fastMRI 3 T T2w vs simulated heads); multi-slice TSE

- **Data**: the fastMRI stream finished with 71 brain volumes (30.9 GB); the 21 fully sampled 3 T
  AXT2 scans (Siemens Skyra/Prisma/Biograph mMR, TSE TR 6.0-7.7 s, TE 105-113 ms, 16 x 5 mm,
  14-20 coils) are the real set. Their stacks run from the lateral ventricles (slice 0) to above
  the vertex in all 21.
- **One reconstruction** (`validation/realism/realism.py`): fastMRI k-space to the 220 mm field of
  view (readout oversampling removed), central 128 x 128 of k-space, RSS, p99 normalisation. It
  first reproduces the files' `reconstruction_rss` (correlation 1.0000000).
- **Matched simulation** (`simulate_heads.py`): each `sample_head` head takes the TR/TE/coil count
  of a paired real scan; stacks placed by the same landmarks (slice 0 at the largest lateral
  ventricles, brain top at slice 13.5; spacing 4.7-5.1 mm); phantom turned so the readout runs
  anterior-posterior as in the real data (checked against the anatomy: r = 0.94; flips 0.82 / 0.35).
  Two first attempts at placing the stack by head area failed (BrainWeb's face and neck dominate
  the area); the landmark rule replaced them.
- **Score** (`score.py`): noise matched to each paired scan's floor (background 1.7 % of p99), a
  small CNN with person-wise class-balanced folds (heads of one BrainWeb subject are one person),
  bootstrap CIs, a histogram baseline, real-vs-real and sim-vs-sim nulls, and a classifier-free
  intensity distance (1-Wasserstein of head pixels).
- **First numbers.** 5 heads (13 slices): CNN AUC 0.998 [0.990, 1.0], histogram 0.91. 7 heads
  (even slices): CNN 0.91 [0.67, 1.0], histogram 0.95, intensity distance 0.050 (real vs real
  0.008, sim vs sim 0.014). What gives the simulations away, by eye and by slice: white matter too
  bright (no multi-slice MT), a thick grey scalp/muscle band (muscle T2 50 ms from excised mouse
  muscle against ~30 ms in human spin-echo studies), weaker coil shading. Per-slice AUC is 1.0 at
  ventricle level and 0.92 near the vertex (more muscle low down).
- **Null caveat**: person-wise CV biases AUC below 0.5 in nulls (histogram real-vs-real 0.20, 0.27
  after class-balanced folds); calibrate small differences with a permutation test (LESSONS).
- **Multi-slice TSE** (`tse_2d(slice_order=...)`, b2fb6af): the other slices' pulses as envelopes on
  a 100 us raster, moved to their slices by `b0_offset = -G z_j`. Fast vs exact phase-ramped pulses:
  0.8 % of the signal at 1/6 of the segments; 7 slices cost white matter 31 % against one slice;
  Metal = core to 3e-6; default sequences unchanged (SHA-256). A 200 us raster is 16 % off. GPU cost
  per slice: x2 for MT, x3 for the other slices' pulses, x6 together.
- **Tissue database** (merged by the other session as PR #7): its pooled 3 T values weight studies
  by n, so one large study can decide a tissue: muscle T2 52 ms (MRF abdomen 74 ms and pharynx
  54 ms outweigh spin-echo studies at 29-32 ms), WM and GM T2 both 86 ms (a SyMRI study, n = 54,
  gives WM 108 ms, GM 80 ms). Flagged to the user; the ablations test it.
- **Running** on the Mac mini (seeds 0-7, slices 0-12:2, 4 spins per tissue and voxel): baseline,
  + MT + 16-slice acquisition, and the same with `tissuedb.tissue_table(3.0)`.
- Commits: e147877 (fetch), b2fb6af (multi-slice TSE), c79c83c (realism v0), dc85124 (ablation
  switches).

## 2026-10-10 — Performance research report (`docs/PERFORMANCE.md`)

- **What**: a read of the engines, the benchmark history and the realism workload, with a cost
  model and estimates for the ways the mathematics can remove work; nothing implemented.
- **Measured on this laptop's 8-core M4 GPU** (realism TSE 128 at the 100 µs RF raster on a
  130k-spin sub-voxel Shepp-Logan, 14 coils): full 1.82 s, ADC flags off 0.97 s, raster-cut ramps
  and crushers merged exactly (36,335 → 19,726 segments, 4e-6 relative) 1.20 s, merged + ADC off
  0.46 s. GRE 3d-48, 8 coils: 1.24 s full, 0.36 s with ADC off. So the coil reduction is 45-70 %
  of the forward, the ramp pieces 25 % of the TSE, RF 10 % (dominant in the multi-slice variant).
  Per spin-segment: cached free 1e-10 s, cache-miss free 2e-10 s, RF 4.4e-10 s, ADC +4-8e-10 s.
- **Segment mix**: 3D GRE is 94-97 % ADC samples; the realism TSE is 46 % raster-cut free
  segments without ADC (154 free runs averaging 77 distinct gradient values each).
- **Ideas, ranked** (gains estimated unless marked): merge ramps exactly (1.5x measured);
  readouts as type-1 NUFFTs along the readout line with time-segmented T2 decay, spreading on the
  same simdgroup matrix products (readout cost ÷ n/(w·M) ≈ 7-14x; 3D GRE 6x); periodic sequences
  (RF-spoiled GRE has period 360/gcd(117, 360) = 40 TRs in the RF frame; affine steady state per
  spin; k-space by 3D NUFFT per residue: 3D GRE 128³ 192 s → 2-4 s); tabulated RF propagators in
  (Δf, B1) with first-order relaxation (multi-slice realism 3x); a reversible adjoint (5-7x →
  ~3x the forward); lattice quadrature instead of jittered spins along the crusher axis (4x on
  spins if the aliasing study holds); half-precision coil reduction, GIRF-friendly phase
  recurrence, interaction-picture free pools; CUDA on rented GPUs for scale.
- Combined estimate for the realism TSE slice: 10-15x (30-60x with the spin reduction); for the
  3D GRE example 50-100x. Suggested order and independent checks in the report.

## 2026-10-10 — Fast realism mode and PERFORMANCE.md items 2.1, 2.4, 2.6, 2.7

- **Why**: the realism ablations took 98-162 s per slice; most testing time went into waiting.
- **Fast mode** (9e1a08a, ee03e81, af4c844): `simulate_heads.py --quality full|fast|legacy`
  (8 or 4 categorical spins per voxel, a ±3 mm slab, RF on 100 µs: image RMS 0.014 / 0.022
  against 27 spins per tissue-voxel, seed-to-seed 0.007); far slices' pulses as one segment each
  with merged delays (42 462 → 23 212 segments); `to_phantom` texture and R2' only where the spins
  are (10.7 → 1.3 s per slab, bit for bit).
- **2.4 merge free stretches** (1cb64c0, `Sequence.merge_free`): exact for static spins (core
  2e-15, Metal 2e-6); realism TSE 36 335 → 19 726 segments; 400k spins, 16 coils: 1.98 → 1.33 s
  single slice, 5.26 → 4.58 s with MT and 16 slices.
- **2.6 fewer spins** (3c80893): systematic tissue assignment = categorical (RMS 0.0139 vs 0.0138
  at 8 spins) and unjittered z-lattices = random spins at equal count; the predicted 4x from
  lattice quadrature along the crusher axis did not appear. Kept as an option, no gain.
- **2.7 overlap** (71da289): the next slice's phantom is built while the GPU runs: fast single
  slice 2.3 s on the laptop (legacy ~128 s).
- **Correction — bound-only pulses in float32** (ebabdaa): the far slices' pulses hid from the water
  by a 1 MHz frame offset (identity in float64). On Metal each 100 µs pulse left ~1e-3 rad, which
  CSF (T2 2 s) accumulated: Metal vs core 0.45 % in signal, 2 % in Mz. `Sequence.bound_only` now
  flags RF that only the bound pool feels (core, Metal forward; chunked, Pulseq export,
  generalized Bloch and free pools refuse it): CSF 6.3e-6, WM with MT 6.8e-7; core vs every pulse
  exact unchanged at 0.71 %. The f1/f2 realism runs started with the old pulses were stopped.
- **2.1 NUFFT readouts** (6b8814c, `simulate_metal(readouts="nufft")`): every readout as a
  type-1 NUFFT (ES kernel w = 6 on a 2x grid, 3 time-segmentation terms) with spreading through
  the coil MMA. One realism slice (16 coils, laptop): 3.83 → 1.51 s (2.5x) fast, 7.53 → 3.10 s
  full; with 16 slices and MT 10.4 → 6.3 s (1.6x, the slices' RF dominates there). Signal 3e-6
  from the per-sample path, images 3e-5. A dead end on the way: the first run detector compared
  dt exactly and found no runs, so "NUFFT vs per-sample" (2.5e-5) was atomic-add nondeterminism
  of the per-sample path twice (LESSONS).
- Defaults bit-identical throughout (snapshots); full suite on the mini: 766 passed before
  bound_only and NUFFT.
- Not started: 2.2 periodicity, 2.3 tabulated pulses (the multi-slice RF share), 2.5 reversible
  adjoint.

## 2026-10-10 — Reused pulse maps (the exact form of PERFORMANCE 2.3); realism suite rerun

- **Where the time went after NUFFT readouts** (one realism slice, 16 coils, laptop, fast): single
  1.50 s (RF 0.34), single + MT 2.41 s (RF 0.90), 16 slices + MT 6.3 s (RF ~4.5; the far
  bound-only pulses 0.1 s, the neighbours' exact pulses triple the stepped RF). The MT kernel is
  slower everywhere (registers). Caching the lineshape per pulse and a series exchange step: 14 %,
  not kept.
- **Dead end — pulses as blocks**: a float64 prototype of 2.3's cheap form (exact rotation of K
  samples with relaxation, exchange and mean saturation around it) is 5e-3 to 1.3e-2 off per pulse
  (WM, GM, muscle), 2e-4 to 1.6e-3 at K = 2. Tables over (Δf, B1) would need the interaction-picture
  expansion and ~60-100 floats per node with the bound pool.
- **What worked** (cedc6ab, `simulate_metal(pulses="reuse")`): a static spin's affine map of a pulse
  is the same at every occurrence; a phase-shifted copy is the map conjugated by Rz exactly. Each
  distinct pulse is stepped once per spin (5 columns) and applied at every occurrence. 1e-7 from
  stepping (TSE, TSE + MT + 16 slices, RF-spoiled GRE + MT). Realism slice: 1.65x single, 2.3x with
  16 slices + MT (full quality 12.9 → 5.6 s; 20.8 s before NUFFT).
- **Suite on the mini** (new code: bound_only, NUFFT; f2 also reuse): pytest 777 passed (783 with the reuse tests). Realism,
  20 heads x 13 slices against 21 fastMRI 3 T T2w scans (CNN AUC, histogram AUC, intensity W1;
  real-vs-real W1 0.0066):
  - f0, single slice, no MT: CNN 0.990 [0.961, 1], histogram 0.965, W1 0.050 (quartiles 0.268 /
    0.360 / 0.548 vs real 0.242 / 0.344 / 0.539).
  - f1, 16 slices + MT: CNN 0.9995 [0.998, 1], histogram 0.968, W1 0.045 (0.213 / 0.304 / 0.502):
    the other slices' pulses and MT darken the slice past the real intensities.
  - f2, 16 slices + MT, `tissuedb.tissue_table(3.0)`: CNN 0.998 [0.993, 1], histogram 0.967, W1
    0.048 (0.222 / 0.308 / 0.501). Its images have no GM/WM contrast: the n-weighted pooled WM and
    GM T2 are both 86 ms (flagged earlier, one SyMRI study dominates).
  - Seen side by side (seed 0 against its paired scan): the real scans' scalp fat is a bright ring
    (TSE "bright fat": rapid refocusing suppresses J-coupling dephasing); the simulation's fat (T2
    68 ms, no J-coupling) is as dim as WM. GM/WM contrast is weaker than real in f0/f1.
- Per head on the mini (13 slices, full quality, ~25 s of it loading and deforming the head): f0 48 s
  (single slice), f1 94 s (NUFFT, stepped pulses), f2 77 s (NUFFT + reused pulses, partly sharing the
  GPU with scoring): ~4 s per multi-slice MT slice, 6-9 s before reuse.
- Commits: ebabdaa (bound_only), 6b8814c (NUFFT), be204b6, b63fb83, 1bd87ee (docs), cedc6ab (reuse),
  6516673 (docs).

## 2026-10-10 — Tissue-database pooling; bright scalp fat

- **Pooling** (cf0d85a): n-weighting let one synthetic-MRI study of 54 elderly subjects (WM T2 108 ms
  > GM 80 ms) set pooled WM and GM T2 to 86 ms each (f2 had no GM/WM contrast). Now each source
  counts once (median, robust spread: the random-effects limit), and T1/T2 come from reference
  methods first (spin-echo decay; inversion/saturation recovery), naming what is left out. 3 T:
  WM T2 74.5, GM 110, muscle 30 (was 52), WM T1 804, GM T1 1276 ms. f3 (f2 with this): CNN AUC
  0.9993, histogram AUC 0.944 (0.967), W1 0.039 (0.048), quartiles 0.230 / 0.346 / 0.552 against
  real 0.242 / 0.344 / 0.539.
- **Fat**: measured where the contrast lives first. Edge profile (mean intensity against depth from
  the head's edge over the interior median, slices 0-8): real scalp peak 2.28 at 3-4 px (1.72 mm),
  f1 a flat 1.53. Fat's T2 68 → 148 ms changed nothing there: BrainWeb labels the scalp
  "Muscle/Skin" (our `skin`, muscle values) and its fat class is 0-24 % of the outer 9 mm.
- **Physics, two parts**: (1) lipids are J-coupled; single-echo SE/STEAM read fat T2 40-75 ms at 3 T,
  a closely spaced refocusing train suppresses the modulation (Henkelman 1992: multi/single-echo
  ratio 1.5 in oil, 1.9-2.4 in fat at 20 ms spacing, ~1 beyond 100 ms) and CPMG at 8 ms reads
  140-159 ms (Keene 2020, four cohorts, EPG fits). A Bloch simulation has no J coupling but plays
  the train, so lipid T2 = the multi-echo-train value (8f6b8ba): pooled fat T2 148 ms. (2) the scalp
  is skin over a subcutaneous fat layer: `layer_scalp` moves BrainWeb's skin class 1.61-5.75 mm deep
  into fat (Li 2024, vertex ultrasound, 30 adults; 2d978be, opt-in `sample_head(scalp="layered")`).
  Laptop A/B, 3 heads: layered 1.26 with fat T2 68 ms, 1.86 with 148 ms.
- **f4** (f3 + layered scalp + fat T2 148 ms), 20 heads x 13 slices: CNN AUC 0.998 [0.992, 1],
  histogram AUC 0.911 [0.823, 0.976], W1 0.041 (quartiles 0.277 / 0.402 / 0.587). Edge profile peak
  1.84 at 3 px (real 2.28), but the band below it (calvarium) is 1.03-1.05 against real 1.56-1.68:
  adult diploe is fatty marrow, and our marrow class has vertebral/knee single-echo values (T2
  50 ms) and one T2 for its water and lipid parts. Next for fat: diploic marrow composition, lipid
  relaxation per species in mixed classes, J modulation (single-echo SE fat is now too bright).
  Also seen: the simulated heads are larger than the real ones and fold a faint scalp line in at
  the field of view's top edge.
- Tests: quick suite 760 passed (laptop). Commits: cf0d85a, 8f6b8ba, 2d978be, 89fa077.

## 2026-10-11 — Visual inspection rounds: what made the simulated heads look fake

- **Method**: the CNN AUC (~0.998) cannot rank fixes, so side-by-side sheets (same recon, noise floor
  and normalisation as score.py) plus a number per tell, and blinded 12-tile sheets (6 real, 6
  simulated, key read after the guesses). Rounds 1-2 on BrainWeb 04 locally, then 20 subjects.
- **Tells found and fixed (each with its number)**:
  - Grey matter too bright (GM/WM 1.91 vs real 1.49): GM T2 rested on one study (110 ms); added the
    Bojorquez 2017 compilation's primaries (Lu 2005 CPMG 83/75 ms, six IR T1 studies; 1f9741d):
    GM T2 96.5 ms; GM/WM over 54 slices 1.54.
  - Heads too large and round (264 vs 189 cm^2 at the ventricles, AP/LR 1.14 vs 1.31): BrainWeb is
    MNI-sized; ANSUR II head breadth/length draws, yaw (real SD 3.8 deg), centred FOV (ff00d4a,
    60603de): 211 cm^2, 1.28.
  - Salt-and-pepper grain: categorical spin draws, and random z through the 5 mm slice profile;
    proportional sampling + z quadrature (`jitter="inplane"`, quality "real").
  - Slice 0 too high (slit ventricles): fitted the real head-area profile and placed slice 0 at the
    foramen of Monro (10 mm below the largest ventricle section), 5 mm spacing.
  - Readout fold-over: 2x readout oversampling as the scanner (b20c148).
  - Skull band too dark (0.26 vs 0.35-0.41): BrainWeb marrow -> yellow marrow (Loevner 2002; 9494d56).
  - Brain tissue ~15 % bright relative to CSF and fat: myelin water as a second free-water species
    (WM 12.3 %, 9680845); body-coil B1+ from the full-wave map (042312f).
- **Dead end**: `open_sulci` (age-dependent sulcal widening, Kochunov 2005) made CSF specks on the
  1 mm grid; the first blinded test (r2) picked every simulated slice by that grain. Left out.
- **Blinded tests**: r2 12/12, r3 12/12 (still obviously fake). Remaining cues: scalp thinner and more
  regular than real, BrainWeb's regular young anatomy (no asymmetry, small sulcal CSF, no pathology),
  uniform white matter, and the contrast gap above.
- **r4** (20 subjects available locally, 6 heads x 9 slices, realistic preset of 9680845/042312f):
  CSF max/WM 4.95 (real 5.21), scalp/WM 2.62 (2.91), GM/WM 1.58 (1.49); blinded 12/12. Head outline
  RMS residual from its ellipse 2.73 % against real 3.32 %; a 9 mm deformation reaches 3.14 % but
  swirls the occipital cortex; not adopted.
- **Deep grey nuclei** (8ead2e0, 30afdd8): BrainWeb calls the caudate, putamen, pallidum and thalamus
  grey matter, so the basal ganglia shone like cortex; at 3 T their iron puts T2 near white
  matter's (Krug 2022: 57-76 ms against cortex 96.5). Labelled by MNI regions, their own tissues
  from the database (nearest-ancestor fallback for what they lack), drawn from their own stream.
  The stack landmark counted gm + wm as brain and jumped 28 mm when the nuclei left gm: fixed.
- **Where the remaining tell sits**: per-ray profiles aligned at each ray's bone minimum (not at a
  fixed depth: that had sent the marrow the wrong way) agree with real from 5 px inward; the excess
  is a bright subarachnoid rim in the first 2-4 px (frontal 0.62-0.64 against 0.39-0.44), with a
  brighter bone band (0.27-0.29 against 0.16-0.21) and a dimmer scalp (0.79 against 0.89 at p90,
  while CSF sets the p99 scale alone: the whole brain ~13 % bright). Variants on the mini (4 heads x
  3 slices each): receive ring at 12 cm or flat, GM T2 85 ms, red/trabecular marrow, the shell-kept
  size, thinning the subarachnoid layer by 1 or 2 mm. Only marrow (bone) and thinning (rim) moved
  their numbers; the full-wave 8-loop head array would make the periphery 1.4-2x brighter, which
  the real scans do not show (they behave as receive-normalised).
- **Shell-kept size** (7603d03): plain resizing thinned the skull and scalp with the head (0.84-0.90);
  `size="anthropometric_shell"` scales only the skull's contents. Skull front/side/back 6.8 -> 7.2,
  2.2 -> 3.0, 5.0 -> 5.5 mm.
- **Flow voids** (f180c83): moving spins through 5 mm, one echo train at 10.26 ms: blood keeps
  0.82 / 0.33 / 0.11 / 0.002 of its static signal at 113 ms for 0.5 / 2 / 5 / 10 cm/s; BrainWeb's
  vessel class (sinuses, large vessels) now leaves no signal; the real superior sagittal sinus is
  dark in every 0.57 mm reconstruction checked.
- **Marrow back to trabecular** (c10c7d2): reverts 9494d56 on the aligned evidence.
- **Ventricles with age** (340dde0, 29d6def): Yamada 2023 lateral-ventricle means as a ratio to 30 y
  (x1.93 at 70); a continuous wall push per side (septum kept), periventricular tissue compressed.
- **Refocusing angle** from each header (0d07df1); headers also give the echo spacing (10.26 ms,
  TE = echo 11) and the true slice thickness (`reconSpace` z 5 mm).
- **Blinded**: r5 (deep grey) 12/12; r6 (shell, flow void) 10/12, the first misses, both at the
  basal-ganglia level; still picked by the bright rim and speckled cortex higher up. r7 (r6 +
  1.5 mm of convexity CSF to inner table, a calibration) 10/12.
- **The "speckled cortex", measured**: CSF-bright components (> half the image's own ventricular
  CSF) in the band 8-25 px under the head edge, slices 2-6: real 8.1 per 1000 px, mean 24 px,
  bright fraction 0.206; r6 6.1, 48 px (the rim joins the sulci), 0.275; r7 8.6, 26 px, 0.232.
  The rim, not the sulci, made the cortex read as fake; the calibration removes 16 % of the CSF
  (245 -> 205 mL), so it stays labelled a calibration (`--sas-mm`), with the age-enlarged
  ventricles giving some back.
- **r8 / r9**: ventricles for ages 20-80, trabecular marrow, protocol refocusing (r8), plus 1.5 mm
  convexity thinning (r9; 4c98996 puts both in the preset). r9 slices 0-8: rim 0.91 / 0.71 / 0.65
  (real 0.75 / 0.52 / 0.57), cortical CSF components 8.5 / 1000 px, 19 px, 0.18 (real 8.1, 24,
  0.21). Blinded r9: 8/12, 10/12, 12/12 = 30/36 (I had seen its six heads by the third sheet).
- **Head shape by height**: simulated heads lost area faster above slice 0 than the real ones. Part
  was my measurement (`lab.head_mask` opened twice and ate the real thin lateral scalp: real
  left-right 143 -> 157 mm at slice 0 with one opening); the rest is tilt. Real routine slices sit
  ~10 deg nose-up to BrainWeb's AC-PC frame: pitch -10 +- 5 deg (1802b56, b7a4ca1) matches the area
  at slices 5 / 8 / 12 / 15 (0.89 / 0.77 / 0.50 / 0.21 vs 0.94 / 0.79 / 0.48 / 0.21).
- **Receive ring at the coil's surface** (1a6d19a): 12.5 cm instead of 18 cm; the scalp top decile
  is as bright as ventricular CSF in the real scans (0.93 of it) and was 0.68 of it in ours.
- **BrainWeb 38's marker** (7332ee8): a detached blob in front of the forehead showed as a dash and
  gave tiles away; detached tissue is dropped.
- Dead ends: membership sharpening (power 2; no better), GM T2 85 ms (no visible change), a 9 mm
  deformation (swirls). The "grain" metric flips sign between slice sets; not used.
- **r11 / r12** (12 new heads each): the 12.5 cm ring (r11) lowered the CNN AUC to 0.967 but the
  blinded test went to 12/12 (bright peripheral cortex, GM/WM 1.73, and a black-looking bone band);
  back to 18 cm with 2 mm thinning (r12): blinded 30/36, CNN 1.0, histogram AUC 0.86, but the
  cortical CSF broke into dots (6f1fe24, then 1.5 mm again: f73fc96).
- **Cleanups**: faint "skin" around BrainWeb 42's neck and fuzzy-classifier haze in the air are
  dropped with the marker (0982579); simulate_heads frees memory between heads (a465208; one
  process grew to 15 GB on the mini and swapped).
- **r13, the scoring protocol of f0-f4** (20 new heads, slices 0-12, realistic preset at f73fc96):
  CNN AUC 0.973 [0.916, 0.999] (f4: 0.998 [0.992, 1]), histogram AUC 0.774 [0.662, 0.879] (0.911),
  intensity W1 0.025 (0.041), quartiles 0.256 / 0.375 / 0.558 against real 0.242 / 0.344 / 0.539.
  Blinded on its first nine heads: 12/12, by the speckled cortex and bright posterior sulcal CSF at
  the high slices: BrainWeb's closed sulci (LESSONS), which no setting here reaches.
- **r14 (12 new heads, slices 1-7, GM T2 83 ms as an experiment)** against r13 on the same slices:
  CNN AUC 0.905 [0.78, 0.99] vs 0.936 [0.82, 0.998], histogram AUC 0.866 vs 0.898, GM/WM 1.54 vs
  1.58 (real 1.49); blinded 12/12 (two tiles by things stuck to the head: BrainWeb 18's marker,
  a thin line in a tilted 42; trimmed, ed15108). The database keeps 96.5 ms until sources robust
  to stimulated echoes are in (LESSONS lists leads).
- **Sulci**: three openers failed (white-matter-distance ridge, Hessian sheets, distance slab), and
  so did using BrainWeb's own faint CSF trace along the closed sulci (real, but too noisy as a
  ridge). The anatomy, not the physics, is what still gives the simulated slices away; the next
  step is a richer anatomy source (needs a download: the user's call) or surface-based sulci.
- Blinded tally this session: r4-r5 12/12; r6-r7 10/12; r9 30/36; r11 12/12; r12 30/36; r13 24/24;
  r14 12/12. Caveat: I have looked at the 21 real scans dozens of times and recognise some; a fair
  human test needs real scans not seen before.
- **Where the CNN sees it** (r13 against real, slices 0-12, the same cross-validated CNN on masked
  images): full 0.973 [0.916, 0.999]; interior beyond 8 px of the head edge 0.991 [0.97, 1.0];
  outer 8 px ring only 0.924 [0.813, 0.985]; centre beyond 25 px 0.831 [0.664, 0.964]. The cortex
  carries most of it, the deep brain least: the same verdict as the eye.
- **r15 (12 new heads, slices 1-7, + `--trace-sulci`, ~13 mL of sheets along BrainWeb's faint CSF
  trace)**: CNN AUC 1.0 [1.0, 1.0] (r13 on the same slices 0.936), histogram AUC 0.86, blinded 12/12.
  No better; the option stays experimental and off. The QSM Challenge 2.0 head (the one other local
  full-head segmentation) has crisp labels but a noisy ribbon and few sulcal lines, and is one person.
  The remaining tell is BrainWeb's cortex; the way on is a population of heads with resolved sulci
  (e.g. HCP or OASIS with FreeSurfer surfaces, IXI with a segmentation), which needs the user's
  decision on downloads and data-use terms.
- **Structure or texture?** The same CNN on blurred images (sigma 1.5 px) still scores 0.979, on
  high-pass residuals alone 0.985: two independent cues. The texture one is ours to fix: in flat
  brain the simulated slices carry 0.022 (Immerkaer sigma, of p99) of pixel-scale texture, the
  same anatomy rendered without spins 0.011, real scans 0.014 with their noise. It is the grain of
  random in-plane spin positions and falls as 1 / sqrt(spins) (4x2x2 0.026, 2x4x4 0.017, 3x4x4
  0.015, 4x4x4 0.0136); subcell centres band and a per-voxel shift makes it worse (0.037).
  In-brain thermal noise is right: real 0.0036 of p99 (each scan's coil covariance from its
  k-space corners, through the same recon), simulated 0.0031. Real coil noise is correlated
  (median |r| 0.09-0.16, up to 0.5) and unequal (2.5-4x), so the background is a few-channel chi
  (std/mean 0.30 raw) but the matching by the background mean still lands the brain's noise.
  Quality "real_fine" (4x4x4) added; r16 runs it.

## 2026-10-11 — Opening BrainWeb's closed sulci; conserved CSF; IXI as the next anatomy

- **Opened sulci** (`open_sulci` rewritten; `sample_head(age=)`): a closed sulcus is grey matter
  reached from two facing white-matter banks, found with the feature transform (neighbouring voxels
  whose nearest white points are >= 3 mm apart and within 60 deg of opposite: the lambda-medial axis)
  on BrainWeb's native 0.5 mm grid, kept where the sheet touches the CSF around the brain, outside the
  deep grey regions; CSF fills the age's width (Jin et al. 2018: 1.27 mm at 47 y, 1.49 at 67, with
  their between-subject SD) either side of the midplane, then the head is resampled (0e2f15f; peak
  memory 6.9 -> 3.8 GB by opening a three-class copy, 24e1db8). Unlike the four earlier openers it
  draws continuous sulcal lines; 0.5 mm bumps and 3 mm grooves open nothing (tests); 20 subjects
  get 117-210 mL at 1.3 mm (MNI size). Subject 04's CSF 254 -> 322 mL (25 y) / 381 mL (75 y, with
  ventricles of the same age; Yamada 2023 287 / 389). Opening brings BrainWeb's grey/white volume
  1.44 -> 1.23 and its pial-to-white distance 2.92 -> 2.35 mm. The ridge opener is gone.
- **Matched pair (local, quality real, 6 heads, slices 1-7)**: the same heads with (A) and without (B)
  opened sulci. Blinded on A: 10/12 (one sim at slice 6 taken for real, one real for sim; r13-r15
  were 12/12). Cortical-band CSF (slices 2-6, components per 1000 px / mean px / bright fraction):
  real 7.2 / 31 / 0.25 (10-90 %: 5.1-9.6 / 11-66 / 0.14-0.33); B 8.8 / 21 / 0.19; A 4.9 / 65 / 0.33,
  i.e. from the real lower half to its upper edge, with young draws (25-48 y). Brain power spectrum
  over real, by band 0.025-0.475 cycles/px: B 1.4-1.5 low, ~1.0 mid, 0.72-0.86 high (soft); A 1.6-1.95
  low and mid, 1.26-1.50 high. GM/WM 1.63 -> 1.59 (real 1.49); the rim under the skull brighter
  (front 0.77 -> 0.82, side 0.70 -> 0.79; real 0.71 / 0.61). Per head the excess follows the subject
  (BrainWeb 04 and 45 already bright, 42 matches real) more than the width.
- **Width x 0.6 (pair C, same heads)**: cortical-band CSF 6.7 / 42 / 0.28 (real 7.2 / 31 / 0.25), the
  spectrum's high bands ~1.0 of real (mid 1.36-1.64); blinded 12/12 nonetheless, by hazy lateral
  patches and a foggy, mottled cortex: no longer the sulci.
- **The fog is BrainWeb's diffuse CSF**: grey-dominant voxels hold 37.6 mL of CSF (3.9 % on average,
  subject 04), which at T2w lifts grey matter ~9 %, the gap between our GM/WM (1.59) and real (1.49).
  It is the blurred trace of the sulci now opened explicitly. Grey matter T2 (96.5 ms, the median of
  Lu 2005 83 ms and Wansapura 1999 110 ms) is the other suspect; the 3 T literature with
  slice-selective multi-echo fits runs to 147 ms (Mumuni 2015), so it needs stimulated-echo-robust
  sources (EMC: Radunsky 2021; Wiggermann 2020 3 T values only in a figure). Parked.
- **Brain-interior intensity quantiles settle the amount** (q70 / q85 over q15 ~ white matter, slices
  1-4; real 1.71 / 2.51): unopened 1.73 / 2.55, Jin x 0.6 1.91 / 2.83, Jin x 1.0 2.07 / 3.22,
  gathered + Jin 2.07 / 3.19 (slices 5-7: real 1.79 / 2.67, unopened 1.70 / 2.27). The real images
  hold about BrainWeb's amount of bright CSF, in lines rather than a rim and diffuse speckle. Hence
  **conserved sulci** (0988ea3, now `sample_head(age=)`'s default): each grey/CSF mixture outside the
  deep nuclei goes to its larger share (`gather_cortical_csf`), the sheets take the width that
  keeps the CSF volume (`open_sulci(volume=)`), widened by Jin's slope (0.011 mm/yr) beyond 30 y;
  `sulci="measured"` keeps Jin's widths. On the six heads: 1.67 / 2.42 and 1.78 / 2.36; cortical-band
  CSF 8.9 / 21 / 0.20 (young draws, 25-48 y, so thin lines). Subject 04 at 25 y: CSF unchanged
  (253.7 mL), 19.9 mL gathered.
- **Blinded, six local heads each (quality real)**: Jin widths 10/12, x 0.6 12/12, gathered + Jin
  12/12, conserved 10/12 (one miss was subject 42's skin wedge at the source grid's front edge, a
  cut of BrainWeb's field of view the pitch lifts into slice 3 in front of the forehead, which I read
  as a real skin marker). What still gives tiles away: a broad light band along the midline at
  slices 6-7 (subjects 04, 45: BrainWeb's posterior medial cortex is a ~15 mm grey slab with no
  white cores, which no opener can split), hazy lateral patches, dense fine lines at slice 7.
- **Checked and not the tell**: head aspect (left-right / front-back 0.81-0.82 at slices 1-7, real
  0.79-0.81), the dark band under the scalp (front 2.5 px vs real 2.0, back 3.5 vs 2.5; before the
  rim calibration it was narrower than real, so later changes widened it, by 0.5-1 px).
- **Anatomy source**: the remaining tells (midline grey slab, hazy patches, fine lines at slice 7) are
  BrainWeb's 1.5 T, 1 mm segmentation. With the user's go-ahead, IXI's 3 T subjects (Hammersmith,
  T1 0.94 x 0.94 x 1.2 mm, T2, ages; CC BY-SA 3.0) are being fetched (`validation/anatomy/fetch_ixi.py`,
  parallel byte ranges: one stream ran at 0.65 MB/s) to segment into BrainWeb's 12 classes on its
  MNI grid (affine registration to a T1-like image from BrainWeb's fractions works: 94 s, det 0.83).
- **r16 (real_fine, 4x4x4 spins, BrainWeb preset)**: two heads died silently on the swapping mini, 10
  heads from 8 subjects; the score's sim-vs-sim control came out 0.008 (anti-learning with 4 people
  per side), so its real-vs-sim AUC 0.07 means nothing (LESSONS). The other measures: in-brain grain
  0.0132 of p99 (real 0.0141, r13 0.0226), so the spin grain is gone; the finest spectral bands
  (0.33-0.48 cycles/px) now at 0.50-0.77 of real (r13 0.74-0.87); blinded 12/12 by the anatomy
  (dark gap under the skull, hazy lateral cortex, midline band).
- **IXI heads** (f86012d): `validation/anatomy/prepare_ixi.py` segments each 3 T subject into
  BrainWeb's 12 classes on its 1 mm MNI grid (affine to a T1-like BrainWeb image, EM on log T1 / log T2
  with BrainWeb priors and a cubic log-bias field; outside the skull, intensity rules: the EM swapped
  classes in the thin head layers). ~3 min, 13 MB per subject; age and sex kept from IXI.xls (ages
  20-82, median 48, 93 of 185 women). `load_ixi`, `sample_head(source="ixi", size="native")`,
  `simulate_heads --source ixi` (no rim calibration, no sulcal or ventricle edits; layered scalp as
  for BrainWeb). Rendered IXI anatomy shows long sulcal lines and a thin midline at the vertex where
  BrainWeb has a grey slab; first simulated IXI heads look like the real scans at slices 5-7. Known
  gap: the T2 covers MNI z <= ~92, so the scalp above is background (above the scored slices).
- **Blinded, six IXI heads (v1, quality real)**: 10/12, with several guesses; the cues were a uniform
  bright scalp ring (fixed 4.14 mm `layer_scalp` everywhere), orbits and frontal sinus at slice 2 (a
  -19.5 deg pitch draw, 2 SD), and a squiggle in front of the forehead (a skin marker in IXI277's own
  scan, attached to the scalp). One IXI tile passed as real.
- **IXI v2 + drawn scalp** (4fb9fed): scalp fat by unmixing the subject's bias-corrected T2 between soft
  tissue (0.9 x WM, temporalis) and its own pure fat (98th percentile of T1-bright head tissue; orbital
  fat 2.3-3.8 x WM in T2), the head one connected piece and, above MNI z 15 mm, each axial section
  opened by an 8 mm disc (the marker goes); `scalp="layered_drawn"` (preset) draws each head's dermis
  and subcutaneous thickness from Li 2024 (1.61 +- 0.23, 4.14 +- 1.10 mm). IXI's own fat alone made the
  scalp dimmer than fastMRI's (the real scalp is as bright as CSF), so the layers stay on top of it.
  26 subjects re-prepared (~3 min each); age trends on v1: GM/ICV 0.484 -> 0.437 from 25 to 75 y
  (r -0.78), CSF/ICV 0.194 -> 0.229 (r 0.69), ICV men 1465, women 1290 mL.
- **r18 stopped** after 6 of 20 v1 heads (230 s per slice on the swapping mini); **r19**: 16 IXI v2 heads,
  one subject each, spins 2x4x4, slices 1-7, then the score with its controls.
- **Blinded, six IXI v2 heads (local, quality real; subjects 12, 148, 202, 253, 334, 538, not looked at
  before)**: 8/12, mostly guesses (two simulated tiles taken for real, two real for simulated). The
  blinded tally on BrainWeb heads was 10-12/12 all day; one 12-tile sheet is weak evidence (8 of 12
  is not significantly above chance), so r19's 16 heads get several sheets and the CNN with its controls.
- **Open: IXI heads narrow a slice too fast.** On the six v2 heads, brain-interior quantiles at slices 5-7
  (q50 / q70 / q85 over q15) are 1.71 / 2.68 / 3.90 against real 1.31 / 1.79 / 2.67 (slices 1-4 closer:
  1.45 / 1.95 / 2.85 vs 1.30 / 1.71 / 2.51), and head area falls faster (slice 7 / slice 1: 0.78 vs
  real 0.86, BrainWeb 0.84). Pitch (-12 to 0 deg) barely moves the anatomy's area profile; a stack
  offset of 15 mm instead of 10 matches it (rms 0.020 vs 0.038 over slices 0-11), but anatomy argues
  for ~8.5 mm at IXI's measured height (Monro ~10 mm below the bodies in MNI, z scale ~0.85). The MNI
  frame is taller than real heads, so BrainWeb heads sized by breadth and length stayed tall and the
  -10 deg pitch was fitted on them. Kept the anatomical offset for r19; its per-slice AUC will say
  whether the top slices carry the difference. Sulcal band statistics now match (7.95 / 34 / 0.27 vs
  7.25 / 31 / 0.25); the back's dark band is wide (4.0 vs 2.5 px).
- **Blinded on r19's first eight heads (v2)**: 10/12; the cues that held were a lighter, hazier white
  matter and cortex (three tiles) and IXI206's scalp split into two arcs. IXI tally: 28/36 (ixiB 10, ixiE
  8, r19a 10): harder than BrainWeb's but above chance.
- **T1 and T2 move apart** (230c44d): IXI's two scans are separate and heads move between them; the best
  integer shift of T2's gradient magnitude onto T1's was 4+ mm in z (IXI206), ~2 mm (IXI012), ~1.4 mm
  (IXI334), 0 (IXI148). Misaligned, the joint EM blurs every tissue boundary (a likely part of the
  haze and the thick cortex: GM/WM volume median 1.43, 1.49 at 30 y) and the head mask doubles the
  scalp. A rigid T2-to-T1 correction (gradient-magnitude correlation, 2 then 1 mm) fixes IXI206's scalp;
  all 26 heads are being re-prepared (v3) for r20. Deep white matter is ~96-98 % white in both IXI and
  BrainWeb, so the haze is not white-matter impurity.
- **r19 (16 IXI v2 heads, one subject each, spins 2x4x4, slices 1-7)**: controls fine this time (real vs
  real 0.44, simulated vs simulated 0.59), so the score stands: CNN AUC 0.979 [0.93, 1.0] at every slice
  (0.976-0.988), histogram AUC 0.965 [0.93, 0.99]: the intensity distribution alone separates them.
  Quartiles of the image (of p99) 0.245 / 0.382 / 0.617 against real 0.216 / 0.299 / 0.491 (W1 0.065;
  BrainWeb runs 0.025): the IXI heads are brighter. IXI's grey-dominant voxels hold 5-7 % CSF (EM
  posteriors, not partial volume; BrainWeb 4.9 %), ~+14 % on grey matter; grey-matter T2 (96.5 ms vs
  Lu 2005's 83) another ~5 %; the high slices more (stack height, open). Variants on four v3 heads
  (baseline, GM T2 83 ms, diffuse CSF < 0.15 in grey given back) are running.
- **Why IXI heads are bright: variants on four v3 heads** (brain-interior q50 / q70 / q85 over q15; real
  slices 1, 3: 1.32 / 1.75 / 2.54, slices 5, 7: 1.32 / 1.79 / 2.66): baseline 1.37 / 1.80 / 2.52 and
  1.50 / 2.10 / 3.33 (the low slices already match; the excess is at the top); grey T2 83 ms 1.28 / 1.61 /
  2.35 and 1.38 / 1.98 / 3.30 (the truth between 83 and 96.5 ms); diffuse CSF < 0.15 in grey given back
  1.34 / 1.71 / 2.40 and 1.43 / 1.98 / 3.25; slice 0 15 mm below the ventricle bodies (three heads)
  1.39 / 1.73 / 2.45 and 1.49 / 1.96 / 3.02, the head's q75 at the top 0.56 -> 0.50 (real 0.50).
  Adopted the 15 mm offset for IXI heads as a calibration (27ae8f1); grey T2 and partial volume open.
- **Partial volume, tried and reverted**: in IXI012's grey/CSF boundary voxels the EM posterior holds 103 mL
  of CSF, linear unmixing of the T2 52 mL (the posterior is close to a step in intensity). Unmixing all
  three classes from two channels and the sum turned noise into white matter (GM/WM 1.48 -> 0.90);
  pairwise unmixing with a dead zone left the total unchanged (CSF reference pulled down by sulcal
  "pure" CSF); ventricle-core references took CSF to 173 mL (11 % of ICV), local references (10 mm) to
  200 mL; on one head the image's high-slice tail improved (q85 3.33 -> 3.09) but the low slices moved
  away from real. With no independent check of which CSF volume is right, v3 keeps the EM posteriors.
- **Blinded, six v3 IXI heads (T2 aligned, 15 mm offset)**: 10/12; both giveaways were slice-7 tiles that
  sat at the wrong height: IXI255 a tiny vertex section (its deep-CSF maximum fell on wide fissure and
  sulcal CSF at 47 mm instead of the ventricle bodies at ~20) and IXI598 a mottled section (a failed
  segmentation: registration lncc 0.112, CSF throughout the brain, 6 % in the white-matter core).
  Fixes (bd47ef2): heads with lncc < 0.15 are not drawn (IXI598, IXI586 rejected; 24 left; the others
  0.16-0.22), and the IXI stack searches the deep-CSF maximum within 12 mm of where MNI's ventricle
  bodies (0, -15, +20 mm) land after the pose (IXI255's slice 0: 28 -> 7 mm). Shape filters (a 2-3 mm
  opening, a central box) did not separate the vertex fissure from the ventricles. IXI tally: 38/48.
- **The mini is off-limits until 23:00 tonight** (user); r20 (16 IXI v3 heads, the 24 that pass QC, the
  landmark prior, 15 mm offset) starts then by a local job that syncs first.
- **The 15 mm offset was wrong** (fa46472): blinded on six heads with the landmark prior and 15 mm, 12/12; the
  simulated slices 1-3 showed basal ganglia, insula and temporal lobes where the real ones show the
  ventricle bodies (1-1.5 slices too low). The 15 mm had been fitted while IXI255's landmark sat 20+ mm
  high; with the prior the area profile prefers 12 mm (rms 0.017, 10 mm 0.029, 15 mm 0.023), but the
  anatomy the eye sees decides: back to 10 mm. IXI tally: 50/60.
- **Blinded, six heads at 10 mm with the landmark prior**: 10/12; the giveaways were slices 6-7 with dense fine
  lines and a hazy midline (three tiles) and one low slice 1 (IXI609). IXI tally: 60/72.
- **Partial volume and grey T2, on the same six heads and seeds** (brain-interior q50 / q70 / q85 over q15;
  real slices 1-3: 1.31 / 1.74 / 2.58, slices 5-7: 1.31 / 1.79 / 2.67): EM posteriors 1.50 / 2.02 / 2.95 and
  1.64 / 2.30 / 3.80; partial-volume unmixing (local references) 1.42 / 1.93 / 2.51 and 1.61 / 2.22 / 3.18;
  plus GM / WM T2 83 / 72 ms 1.37 / 1.84 / 2.49 and 1.55 / 2.12 / 3.25. Both adopted (a7eb14d): unmixing is
  prepare_ixi's default; tissuedb leaves flagged rows out of the pool (only Wansapura 1999's abstract, whose
  GM/WM labels were swapped by assumption), so GM T2 at 3 T is Lu 2005's 83 ms and WM 72 ms (this also
  changes the BrainWeb realism runs' table). The top slices stay ~18 % too bright at the median: IXI's
  cortex grows with height faster than the clinical scans' (open: population age, head shape, stack).
- 24 heads re-prepared with unmixing (v5) for r20, which starts on the mini after 23:00.
- **My blinded sheets are wearing out**: 12/12 on six v5 heads (unmixing, GM T2 83 ms), but these were the six
  subjects I had just looked at in three variants; I now recognise some IXI brains by shape (IXI334's bright
  posterior midline). The CNN (r20) and a fresh observer are the fair judges from here.
- **A quiz for a fresh eye**: `outputs/real_or_simulated.html` (local only: the real tiles are fastMRI), 12 real
  and 12 simulated slices at the final settings (v5 heads, GM T2 83 ms), shuffled, scored in the page; built
  by the scratch quiz.py, its key unseen by me. 24 v5 heads prepared (all pass QC); ~30 more subjects being
  prepared into prepared_more/ for a larger pool. r20 starts on the mini after 23:00.
- **r20 (16 IXI v5 heads, PV unmixing, GM/WM T2 83 / 72 ms, landmark prior, 10 mm offset; one head died on the
  mini, 15 scored)**: CNN AUC 0.959 [0.90, 1.0] (r19 0.979), histogram 0.927 [0.85, 0.98] (0.965), W1 0.042
  (0.065), image quartiles 0.230 / 0.339 / 0.573 vs real 0.216 / 0.299 / 0.491; controls 0.44 / 0.47. By slice
  0.93 (1), 0.93 (2), 0.97 (3-6), 0.98 (7): the low slices are the most realistic. By region (of p99,
  slices 1-3): the band 8-20 px in 0.33 vs real 0.26, the core 0.25 vs 0.21; at slices 5-7 mid and core
  match and the outer ring is brighter (0.41 vs 0.37). Bone-aligned profiles: the rim 2-4 px inside the
  skull is bright again (front 0.55-0.56 vs 0.39-0.46), now from IXI's own subarachnoid CSF, and the lateral
  scalp is too bright (0.80 vs 0.54): `layer_scalp` lays 4 mm of fat over the temporalis, where real scalp
  has little. Simulated heads are also ~7 % larger in area at slices 1-3 (8.2-8.3 vs 7.7 kpx).


## 2026-10-12 — Upside-down heads, the coil and the scalp

- **The simulated heads were upside down against the real ones** (6eccb75). Measuring head shape for the
  bright frontal rim, the real heads' width peaked 0.34-0.44 of their length from the top row and IXI's axial
  T2 heads (oriented by their NIfTI affines) 0.48-0.62 from the front, with no overlap between 21 and 29
  people, and the simulated heads matched IXI. No tilt closes that gap (even 40 deg nose-down left IXI heads
  at 0.44-0.47), so I checked orientation itself: an IXI interior template matches 91 % of 147 real slices
  flipped up-down (r 0.35 vs 0.22) and the simulated ones as displayed (0.32 vs 0.22). fastMRI shows the
  head posterior up; `to_real_orientation` showed ours anterior up since the first score (a mean-image check
  could not see an AP flip; I had read the forehead's dark band as the occiput on the pad). Fixed;
  `validation/realism/orientation.py` keeps the check. r20 rescored, same k-space: CNN AUC 0.998 [0.99, 1.0]
  (was 0.959), histogram 0.927 (orientation-free), controls 0.44 / 0.46. Every earlier front/back finding
  compared our front with the real back: the "bright frontal rim" of r20 is at our front against the real
  front's 0.31-0.33 (still too bright: 0.48-0.52), and the real back is bright.
- **The head lies on the coil** (5600381). With matched orientation the real images fall from back to front
  (white matter q25: front / back 0.89; outer ring 0.71) and ours did not (1.00, 0.82): a head resting on the
  posterior elements. `coil_offset` puts the receive ring's back a gap behind the occiput; re-weighting r20
  by the rings' RSS ratio bracketed the real numbers with a 14 cm ring and a 20-30 mm gap (preset: 25 mm).
  Simulated on IXI012: 0.83 / 0.70.
- **Scalp: IXI's own fat, unmixed against the median** (e40f17d). Next to the real heads, every simulated
  one had a uniform white scalp halo: the preset laid Li 2024's 4 mm vertex fat layer everywhere. Integrated
  scalp signal over white matter (back / side / front): fastMRI 15.1 / 10.3 / 9.8; IXI012 with the layer
  20.3 / 17.1 / 13.1, with its own segmented fat 8.6 / 6.7 / 6.1, because prepare_ixi's pure fat was the
  98th percentile of T1-bright tissue's T2 (3.09 x WM; median 1.41) and unmixed subcutaneous fat to a third.
  With the median: 14.6 / 9.5 / 6.9 (fat 120 -> 274 mL). IXI's raw T2 itself shows a dim scalp (5-6 over
  WM: the late echo of a dual-echo TSE), so it is no direct reference for fastMRI's fat. The preset keeps IXI
  heads' own scalp; BrainWeb heads keep the layer.
- Dead end: a temporalis rule in prepare_ixi (only 2 mm of skin where the soft tissue under the surface is
  thicker than 10 mm) fired across the upper head too, where the outer table is detected in patches; with the
  heads' own fat it is moot. The lateral scalp, with the own fat: 9.5 against 10.3.
- **Where the CNN looks** (scratch regions.py: score.py's test on images masked to a band of distance from the
  head's edge, the rest zeroed). r20, orientation fixed: the head's outer 0-6 px alone (scalp and skull) CNN
  0.998, histogram 0.984; 6-14 px 0.81 / 0.73; deeper than 12 px (brain) 0.79 / 0.79. The tell is the outside
  of the head, not the brain.
- **The dark band at the skull was muscle** (03d0513). The outer band's quantiles in our heads were too dark
  at the bottom (q10 0.056 against 0.136): the dark band at the skull ~4 px wide against ~1.5. The segmented
  bone is thin (IXI012 1.7-5.6 mm along rays); next to it lay 3-7 mm of "muscle", prepare_ixi's label for soft
  tissue deeper than 8 mm, which at T2 30 ms is black at TE 113 ms. That tissue reads 0.53-0.80 x WM in the
  subject's own T2 at every depth (skin-like 0.77, muscle 0.21 at IXI's TR 5725.8 / TE 100 ms by our tissue
  values), so soft tissue is now unmixed between skin and muscle by its T2. IXI012 (muscle 433 -> 290 mL):
  outer band 0.157 / 0.227 / 0.382 / 0.623 / 0.833 against the real 0.136 / 0.214 / 0.375 / 0.616 / 0.859
  (v6 0.056 / 0.130 / 0.227 / 0.470 / 0.764); the side and back profiles close to the real ones, IXI012's
  forehead short of fat.
- The v6 preparation (median fat only) was stopped half-way; 24 heads re-prepared as v7 (both changes) for r21:
  orientation, coil, own scalp fat, soft tissue by T2.
- **r21 (16 IXI v7 heads, the same 16 subjects as r20; orientation fixed, head on the coil's posterior elements,
  own scalp fat with the median reference, soft tissue by T2)**: intensity W1 0.017 (r20 0.042), quartiles 0.222 /
  0.314 / 0.506 vs real 0.216 / 0.299 / 0.491 (r20 0.230 / 0.339 / 0.573), histogram AUC 0.735 [0.59, 0.87] (r20
  0.927), but CNN AUC 1.0 at every slice; controls 0.44 / 0.40. Scalp signal over WM per head inside the real
  range in every sector (back 10-20 vs real 9-23, side 8.5-10.4 vs 7-17, front 7.4-9.4 vs 6-17). By band: the
  outer 6 px CNN 0.944 / histogram 0.89 (r20 0.998 / 0.984), 6-14 px 0.84 / 0.55, interior 0.74 / 0.70: no
  single region carries the cue any more.
- **Phase-encode ordering.** Real AXT2 k-space is symmetric in ky (+ky / -ky 0.95-1.01 from 10 to 190 lines out);
  ours was lopsided (0.6-0.8 at the edges): 128 lines over all 16 echoes (TE 13-200 ms), where the real scans
  encode 396 lines (all 21 headers) and the central 128 come from ~5 echoes. Low-passing both sets across the
  phase axis dropped r21's CNN to 0.92 (across the readout: 0.99). Signal beside the head (phase direction) was
  1.22 x the noise floor against the real 1.09. Two options in tse_2d (6a1a1bf, 88c4c20): reverse-centric
  (symmetric bands, edges first) and acquired_lines (the n lines on the echoes they have in the full linear scan;
  simulate_heads --ordering linear_full, 25 shots, ~2.5x the time). Reverse-centric on IXI012/013: beside the
  head 1.08 (real 1.09), but no more phase-axis detail.
- **Where the remaining fine-detail gap is**: real/sim spectral power along the phase axis rose to 4.6 at
  Nyquist in a central patch but only ~2-2.8 (both axes alike) in patches inside the hemispheres; the central
  excess is the interhemispheric fissure, which in ours is brighter and wider (peak 4.71 x WM, +-1 px 3.59, vs
  real 3.89, 2.68): solid CSF, because neither BrainWeb nor the IXI heads have a falx. White-matter texture
  matches (relative SD 0.087 vs 0.088, same band powers); ventricle walls are ~10 % softer than real (0.418 vs
  0.459 per px: the anatomy's resampling chain). add_falx (0a96dc7): the falx as a 0.91 mm dural sheet in the
  fissure (Walsh 2021), above the callosum's free edge (Kayalioglu 2004), 4-5 mL per head.
- Dead ends here: a GRAPPA hypothesis for the real phase-axis detail (the real k-space noise is flat across
  lines, even / odd 1.00) and physiological ghosting (beside the head the real images carry less, not more).
- **r22 (r21's 16 subjects and seeds + the falx + reverse-centric ordering)**: CNN AUC 0.9996, histogram 0.749, W1
  0.023, quartiles 0.225 / 0.328 / 0.521 (real 0.216 / 0.299 / 0.491); controls 0.44 / 0.41. The fissure now peaks at
  3.92 x WM (real 3.89; r21 4.71), the background beside the head 1.10 x the noise floor (real 1.09; r21 1.22), the outer
  band's quantiles 0.119 / 0.187 / 0.332 / 0.613 / 0.851 (real 0.136 / 0.214 / 0.375 / 0.616 / 0.859). By band, the
  outer 6 px dropped to CNN 0.83 (r21 0.94) but the interior rose to 0.87 (0.74), and low-passing across the phase
  axis no longer helped the classifier (0.998): reverse-centric's early echoes at the k-space edges brighten
  tissue texture and lift the median. Not the scanner's ordering either; r23 (the 396-line linear ordering) runs.
- **The stack climbs too fast** (801752f): at slices 5-7 our heads narrowed faster than the real ones (slice 7 area
  6064 vs 6730 px) and their interior held less white matter (share 0.32-0.37 vs 0.44-0.47; the grey / white
  mode ratio itself matches, 1.17), so the high slices showed extra cortex. Proxy images (each class's spin-echo
  signal, 5 mm slabs, 1.72 mm pixels; at 10 mm they reproduce r22's ventricle drop) put the best stack offset at
  12.5 mm: ventricle cross-section over slices 0-3 211 / 240 / 250 / 223 px against the real 181 / 220 / 272 / 218.
  The 15 mm offset rejected yesterday had been judged by eye on upside-down heads. At slices 4-5 the real ventricles
  stay larger (an older clinical population).
- **Cubic resampling** (82a79cb): deform(order=3) and prepare_ixi onto(order=3); 24 heads re-prepared as v8 (GM / WM /
  CSF volumes within 1 % of v7, skull +10 %, all pass QC); 47 more subjects being prepared as v8 for a larger pool.
- **r23 (the 396-line linear ordering) stopped after 3 heads**: on those heads against r22's, the phase-axis fine
  structure in a central patch came within 1.25-1.75 of real power at high k (reverse-centric 2.0-3.9), hemisphere
  texture slightly closer; the background beside the head 1.17 x the floor (reverse-centric 1.08, real 1.09: a halo of
  the short-T2 scalp tissue along the phase axis, not ghost copies). The 396-line ordering goes into r24.
- **The coil noise is coloured** (9b2abbd): the real scans' background is wider and more skewed than independent
  equal-sigma noise in 16 coils gives (SD / median 0.163, skew 0.41 vs 0.126 / 0.20): their coil SDs span 0.66-1.38
  of the mean and correlate (|r| 0.13-0.17). score.py now estimates each real scan's coil covariance from its image
  corners and colours the paired simulation's noise with it (background 0.158 / 0.35). r21 rescored: CNN AUC 1.0 ->
  0.971 [0.92, 1.0] (by slice 0.96-0.98), controls 0.42 / 0.43; histogram 0.735 -> 0.841 (CIs overlap). The quiz
  pages use the same noise.
- **r24 running** (v8 heads with cubic resampling and pull-back, falx, slice 0 12.5 mm below the ventricle bodies,
  396-line linear ordering; r21's subjects and seeds, ages 20-82, median 52): 14.7 M spins per slice; the mini held
  14 GB and swapped, so simulate_heads --serial (14716ce) builds one slice's phantom at a time (6 GB); ~135 s per slice.
- **r24's first two heads against r21's**: hemisphere texture in the top bands 1.10 / 1.01 of real power (r21 0.65 /
  0.49), ventricle walls 0.453 per px (real 0.459; r21 0.417), the fissure 3.91 / 2.68 at its centre / +-1 px (real
  3.89 / 2.68), phase-axis detail in the central patch ~1.0 of real (r21 up to 3.4). By eye the slices are now hard to
  tell apart; the white matter at the upper slices looks a little mottled.
- **What is left** (two heads): coarse texture 1.4 x real power at 0.3-0.45 of Nyquist; the background beside the head
  1.18 x the floor (real 1.09); the outer band dark at its q25-q50; and at slices 5-7 less of the interior near the
  white-matter mode (0.35 vs 0.44), with the central white matter darker than the peripheral (0.865 vs real 0.93).
  Not the receive coil (3D element rings move the ratio by < 0.01), not the slice geometry (proxy images: 0.39-0.42),
  not IXI's anatomy (its own T2 at those levels: share 0.49-0.59, centre / ring 1.03-1.08), not B1+ alone (the map's
  central overflip is stronger at the low slices, which match). The within-tissue texture (5 % T1/T2 at a 10 mm
  correlation length, an M4 assumption) sits at the scale of the coarse excess; ablations (--no-texture, --b1 uniform;
  01c39ba, 858bbf4) queued after r24. Proxy fit of the pitch at the 12.5 mm offset keeps -10 deg (area rms 0.017).
- **Same-person residual (step 1 of a fitted residual layer)**: IXI012's real T2 (bias-corrected, over its white-matter
  median, on the phantom's 1 mm grid) against the T2 its phantom predicts (each class's spin-echo signal at IXI's TR
  5.726 s / TE 100 ms with our 3 T tissue values, deep grey and falx as simulated; scratch same_person.py). IXI's
  noise: background SD 0.020 x WM. In the brain |log(real / predicted)| median 0.059, RMS 0.160, still 0.071 after
  2 mm smoothing: structure, not noise. Pure white matter: RMS 0.077 raw, 0.046 / 0.023 after 2 / 8 mm smoothing,
  its centre 1-5 % brighter than its periphery at MNI z 25-45 (periventricular and central white matter), the direction
  our high slices lack (centre / ring 0.86-0.90 vs fastMRI 0.93). By class (fraction > 0.9, real / predicted): GM 1.01,
  WM 1.00, CSF 0.92, putamen 1.27, thalamus 1.15, caudate 1.07, dura 1.59, skin 1.15, muscle 0.89, fat 0.67. Outside
  the skull |log ratio| median 0.14 (layer boundaries). Caveat for fitting: some of this is sequence physics, not
  tissue (fat 0.67: IXI's echo spacing does not J-suppress fat as our CPMG T2 assumes), so fat needs its own model and
  T2 / PD need IXI's PD echo to separate.
- **r24 (v8 heads with cubic resampling, cubic pull-back, falx, slice 0 12.5 mm below the ventricle bodies, 396-line
  linear ordering, coloured coil noise; r21's subjects and seeds)**: intensity W1 0.0155 (the closest yet), quartiles
  0.221 / 0.313 / 0.501 vs real 0.216 / 0.299 / 0.491, histogram AUC 0.826 [0.69, 0.93], but CNN AUC 0.9996 (r21
  rescored with the same noise: 0.971); controls 0.44 / 0.42. The targeted measures all moved toward the real ones
  and the classifier still separates the sets: it finds whatever systematic difference remains, so it says where to
  look next rather than how close we are.
- **Where the CNN looks (r24, occlusion of 16 px patches in held-out images per person-wise fold)**: for the simulated
  images the evidence sits at the back of the head (outer 0-6 px: logit drop 300 / 1000 per px with the patch mean,
  366 with only a 2 px blur, against 79-192 at the sides and front; rim 294; interior 144). The fine structure of the
  occipital scalp and skull gives them away.
- **IXI's fat was shifted** (966d8a0). At 0.57 mm the real posterior scalp has a dark dermis outside, fat 5 mm in, fat
  running onto a ~3 mm dark skull line (soft tissue 6.6 mm [5.6-7.6], skull outer surface to its darkest point ~2.0
  mm); ours had fat at the skin and soft tissue against the skull (soft tissue 5.0 mm, then ~4.8 mm to the darkest
  point). Fat's depth below the skin minus skin and muscle's in the prepared heads: -2.6 mm at the back, +0.8 at the
  front, -1.6 at the sides: IXI's T2 displaces fat 1.63 mm posterior (all 24 subjects, IQR 1.1-2.1), a water-fat
  shift along a front-back readout. Undone in prepare_ixi before fat, marrow and soft tissue are assigned. 24 heads
  re-prepared as v9. Dead end on the way: pressing the occipital scalp against the pad (compress_occiput), from an
  edge-to-darkest-bone measure that mixed scalp and skull; the soft tissue alone is thinner than real, not thicker.
- **Ablations at full quality** (IXI012, seed 200, slices 5-7): without texture the central / peripheral white matter
  stays 0.871 (with 0.869; real 0.931), with uniform B1+ 0.900; white-matter mode over the median 0.69 / 0.72 / 0.75
  (real 0.83). The B1+ map explains about half of the central darkening; the rest matches the same-person residual
  (IXI's central white matter 1-5 % brighter than one white-matter class allows), a tissue property.
- IXI's PD volumes fetched (c55897f; 185 Hammersmith subjects, 1.3 GB) for the fitted residual layer.
- **A residual layer fitted to each subject's own scans** (28991d4). prepare_ixi resamples IXI's PD (the dual-echo
  TSE's TE 8 ms echo) with the T2's transform and fits per voxel the T2 and PD multipliers that make the simulator's own
  tissue mixture (deep grey, falx as simulated) reproduce both echoes: T2 from the echo ratio (scale from ventricular
  CSF), PD from the first echo over white matter; water tissue only (not fat, not CSF's unmeasurable T2); 0.7 mm
  smoothing; a cubic trend fitted over pure white matter removed, which also removes the two-point fit's absolute level
  (IXI012's white matter reads 1.18 x our 72 ms: stimulated echoes) and its B1+ pattern (+-15 % base to top; a 20 mm
  high-pass instead left halos around the nuclei); clipped to x0.61-1.65. IXI012: deep grey PD x1.08-1.13, T2
  x0.98-1.09, grey matter PD x1.06, scalp soft tissue T2 x1.52 / PD x0.68. Same-person check (scanner trend removed
  from both): brain |log ratio| median 0.064 -> 0.032, 2 mm-smoothed RMS 0.080 -> 0.072. 24 heads re-prepared as v10
  (fat shift undone + residual maps) for r25.
- **r25 (v10 heads with the residual) stopped after two heads** (IXI012 seed 200, IXI445 seed 210, against r24's same
  heads): the brain moved the right way (central / peripheral white matter 0.873 -> 0.907, real 0.931; white-matter
  share at slices 5-7 0.38 / 0.31 / 0.34 -> 0.47 / 0.34 / 0.42, real 0.47 / 0.44 / 0.44; posterior lumpiness 0.071 ->
  0.054, real 0.042), but the hemispheres' fine texture rose to 1.4-2.4 x real (r24 ~1.0: the echo ratio's noise in the
  maps on top of the assumed texture) and the posterior scalp dimmed (peak 0.65 -> 0.43 of p99, real 0.83: the
  residual fitted the phantom's unshifted scalp fat against the subject's shifted images).
- **Residual v11** (9b8d0f8): fitted inside the skull only, smoothed 1.5 mm within each voxel's largest tissue (log SD
  on IXI012 0.19 -> 0.054 for T2, 0.23 -> 0.065 for PD; same-person 2 mm-smoothed RMS 0.074 vs v10's 0.072 and labels
  only 0.080), fractions bit-identical to v10; simulate_heads drops the assumed GRF texture when the residual is on (the
  fitted maps are the subject's own within-tissue variation). 24 heads re-prepared as v11 for r26.
- **The posterior layers, with r26 waiting for its heads.** At the back of the head, edge-aligned profiles (half the
  scalp peak, 1.72 mm pixels, slices 3-7) put r24's dark minimum at 6-6.5 px and the brain (0.4 of p99) at 9.5 px
  against the real 4.5 and 7.0, while the front read 4.0 against 6.5 and the sides matched (6.5 / 6.5). Not the
  labels: IXI012's own T2 on the MNI grid has the dark zone 7-10 mm deep and its labels reproduce it to ~1 mm. Not the
  pitch: across r24's heads the depths tracked the drawn pitch (r -0.64 / +0.74), but proxy images of the same 16
  heads at fixed pitches -10 to +10 deg (each class's spin-echo signal, 5 mm slabs, the receive ring's RSS; scratch
  pitchfit3.py) moved them < 1 px, and the head-area rms stays best at -10 (0.027 vs 0.033-0.039): pitch kept. Mostly
  the v8 anatomy: IXI012 simulated on v10 (fat shift undone) moves the posterior minimum 5.0 -> 4.5 px and the brain
  8.0 -> 7.5; with fat's chemical shift switched off (all peaks at 0 ppm, scratch sim_noshift.py) the edges move
  another ~0.5 px, the expected 0.56 px (the real scans' shift direction is not settled by one head). The skull band
  stays darker than real (IXI012 0.07, r24 0.12-0.16, real 0.19-0.27). The fastMRI headers (Biograph mMR, 2.89 T)
  give a 384 x 396 encoding, echo spacing 10.26 ms, no readout bandwidth.
- The v11 preparation thrashed the laptop (18 GB swap with the pitch fits running); the last 17 heads moved to the mini
  (~250 s each, two jobs).
- **The real echo train, from the k-space** (f3de52e): the AXT2 line energy (summed over coils and readout, log minus
  a 15-line running median, 21 volumes x slices 1-7) steps every 18.9 lines, at ky -161, -142, ..., -28, -9 and +10:
  each echo fills a block of 19 lines with ky = 0 in the middle of one, so 396 lines over 21 echoes (19 shots) and
  ky = 0 in echo 11; the headers' echo spacing (10.26 ms at TE 113, 9.56 at TE 105) gives TE / spacing = 11.0 for
  both. Ours was a fitted 16-echo train at TE / 9 = 12.6 ms (25 shots): 23 % longer spacing and 30 % fewer refocusing
  pulses per slice (the neighbours' trains drive MT). `simulate_heads --train header` plays the real one (opt-in,
  with --ordering linear_full); validation/realism/echo_train.py records the analysis; the manifests now carry
  echo_spacing_ms. The background beside the head along the phase axis still reads 1.19 x the noise floor on the v10
  IXI012 head (real 1.09), the readout direction 1.00 (real 1.02).
- **r26** = r24's 16 subjects and seeds on v11 heads with the residual (no assumed texture) and the real echo train;
  the laptop first simulates IXI012 (seed 200, slices 3-7) with the real train and no residual against the fitted
  train's run, to separate the two changes.
- **The real echo train removes the phase-axis halo** (IXI012, seed 200, slices 3-7, v10 fractions, no residual; only
  the train differs): noise-free signal beside the head along the phase axis 2.06 / 0.62 / 0.42 / 0.28 x the paired
  noise floor at 1 / 2 / 5 / 9 px (fitted train 2.34 / 1.40 / 1.09 / 0.87; the readout direction 0.46-0.11 for both);
  the noisy background beside the head 1.03 x the floor (fitted 1.19, every run since r21 1.18-1.22; real 1.09). The
  fitted train's 25-line blocks with bigger steps (12.6 ms spacing) put replicas of the head along the phase axis.
  Scalp profiles and the GM / WM mode ratio unchanged (1.17, real 1.17); the white-matter mode over the interior
  median 0.70 (fitted 0.72, real 0.83) — the slice 5-7 composition gap the residual addresses in r26.
- **r26 stopped after two heads** (seeds 200-201 against r24's): the phase-axis halo gone (beside the head 1.03 x the
  floor, r24 1.18, real 1.09), the front / back ring gradient 0.70 (r24 0.45, real 0.71), the front scalp over WM 8.83
  (7.85, real 9.79); but CSF too bright against the brain (interior p95 / p99 3.97 / 5.22 x the median, r24 3.14 /
  4.09, real 3.23 / 4.04; fissure peak 4.36, real 3.89), the outer band dimmer under the higher p99 (q75 / q90 0.518 /
  0.717, real 0.616 / 0.859) and hemisphere texture 1.3-2.0 x real. The train alone (IXI012 A/B, no residual) raised
  CSF's p99 by 7 % and texture by ~6 %; the rest was the PD residual: CSF x1.21-1.28 and grey matter x1.05-1.09 in all
  subjects, IXI's own MT read as PD (1d90f18: PD multipliers re-centred per tissue; T2's kept). 24 heads re-prepared
  as v12 (mini 18, laptop 6) for r27 = r26 on v12.

## 2026-10-13 — r27: CNN AUC 0.875; too much convexity CSF

- **r27** (r24's 16 subjects and seeds; v12 heads: residual inside the skull, smoothed 1.5 mm within tissue, PD
  re-centred per tissue; no assumed texture; the real echo train; coloured noise): CNN AUC **0.875** [0.745, 0.976]
  (r24 0.9996, the first run clearly below 0.96), histogram AUC 0.572 [0.40, 0.73] (r24 0.826), intensity W1 0.0134,
  quartiles 0.215 / 0.303 / 0.498 vs real 0.216 / 0.299 / 0.491; controls 0.44 / 0.46. The laptop's share was moved to
  the mini at the user's request (laptop paused until 22:00); the run finished there at 21:01.
- Against r24 (all 16 heads): background beside the head 1.03 x the floor (1.21, real 1.09), white matter front / back
  0.89 (real 0.89), outer ring front / back 0.61 (0.51, real 0.71), white-matter share at slices 1-5 0.46-0.51 (real
  0.47-0.51). Still off: CSF over the white-matter mode at slices 5-7 6.38 (real 5.25; slices 1-3 5.32 vs 5.12) with
  white matter's share there 0.38 (0.44); the fissure 4.17 (3.89); coarse hemisphere texture 1.5-1.7 x real at 0.15-0.45
  of Nyquist (all structure: the noise adds < 1 % in those patches); behind the scalp the dark minimum 0.13-0.17 at
  5-6 px against the real 0.19-0.27 at 4.5-5.5, the brain 2-3 px later.
- **Where the CNN looks**: masked to the interior (> 12 px) 0.774, to the outer 12 px 0.796 (r20: 0.785 / 0.96):
  no region dominates. Occlusion (logit drop per px x 1000, simulated images): the posterior third carries most
  (rim 6-14 px 268, interior 203, outer 0-6 px 136; blur 284 / 159 / 292), the front and middle thirds 13-117.
- Diagnostic caveat: in 15 % of r27's images (real 7 %) the posterior scalp falls outside lab.head_mask (the skull band
  under its 12 % threshold all round), which shifts edge-aligned profiles for those images.
- **Too much convexity CSF in the IXI labels**: against IXI012's own T2 the labels predicted 30-50 % more CSF-like voxels
  (> 2.5 x WM) than the image shows at MNI z 30-70, their "pure" convexity CSF reading 0.86-0.88 of the prediction
  (1.04 at z 0-30); the unmixing calibrates CSF on the ventricles' T2, and 1 mm convexity voxels are not pure.
  refine_csf (9450c7d) sets each CSF-containing voxel's share from the T2 against the subject's ventricular CSF:
  IXI012 CSF 199 -> 158 mL, same-person brain |log ratio| median 0.055 -> 0.046 (RMS 0.087 -> 0.072), the ratio flat
  with height. Dead end on the way: scalp classes inside the skull (IXI012's intracranial "skin" reads 1.01 x WM in its
  T2: meninges in partial volume, not CSF). 24 heads re-prepared as v13 for r28.
- **Blinded reader on r27**: the quiz page (12 real + 12 simulated slices, scoring noise, seed 7; local only) scored
  7 / 24 by the user: below chance (12), P(<= 7) = 0.032 for guessing, two-sided 0.064. A reader cannot pick the
  simulations out, and whatever cue they used points the wrong way more often than not.
  The reader's cue, in both quizzes: the simulations looked smoother and blurrier, the real scans sharper with more
  fine structure. It picked the simulations out earlier (when our heads had half the real fine-structure power) and
  points the wrong way now: r27's hemisphere structure power is 1.2-1.7 x real at 0.15-0.75 of Nyquist (r24 0.95-1.6),
  ~1.0 at the top band, so the sharper-looking tiles were more often ours.
- **r28 (v13: CSF from the subject's T2) stopped after two heads** (seeds 200-201 against r27's): hemisphere
  structure power 0.81 / 1.18 / 1.01 / 0.97 / 1.13 / 0.99 x real by band (r27 1.04 / 1.56 / 1.37 / 1.34 / 1.48 / 1.28):
  the excess coarse texture, and the reader's inverted sharpness cue, came mostly from the extra sulcal CSF; slice
  5-7 interior p95 / p99 2.83 / 4.00 (real 2.90 / 3.86; r27 3.24 / 4.76), white-matter share at slices 6-7 0.39 / 0.42
  (0.37 / 0.40, real 0.44). But the fissure fell to 3.34 at its centre (real 3.89): the falx counted twice (the subject's
  T2 shows its own; add_falx then drew another). 447eee9: refine_csf models the falx sample_head will add (three
  rounds to settle it; IXI012's fissure 1.06 x its T2 at its own CSF level, v12 0.94, v13 ~1.16). The laptop was
  overloaded for a while by a simulation on top of the IXI preparation; at the user's request everything heavy now
  runs on the mini (r29: v14 preparation, 16 heads and the score in one detached script).

## 2026-10-13 — Milestone: simulated heads that are not obviously fake

Where the realism work stands after r20-r29 (2026-10-12 / 13), against 21 fastMRI 3 T AXT2 scans (slices 1-7, the
same reconstruction, normalisation and coloured coil noise for both sides):
- **By eye: not distinguishable.** A blinded reader (the user) scored 7 / 24 on r27's quiz (chance 12; P(<= 7) =
  0.03). The one cue that had worked before, "the simulations look smoother", now points the wrong way.
- **By statistics:** full-run CNN AUC 1.0 (r20-r24) -> **0.875** [0.745, 0.976] (r27), the intensity histogram test at
  chance (0.572 [0.40, 0.73]); controls 0.44 / 0.46. Masked to the interior or to the outer 12 px the CNN reaches only
  0.77 / 0.80: no single region gives the simulations away any more.
- **What closed the gap** (each found by a targeted measure, fixed in the source, checked independently):
  the display orientation; the receive coil behind the head; IXI's water-fat shift in the segmented scalp; cubic
  resampling (half the real fine structure before); the falx; the slice stack's level; coloured coil noise; a
  residual layer fitted to each subject's own dual-echo scans (intracranial, smoothed within tissue, PD re-centred per
  tissue); the real echo train read off the k-space (21 echoes at 10.26 ms; the phase-axis halo gone); each voxel's CSF
  share from the subject's own T2 (convexity CSF had been 30-50 % too high), with the falx accounted for.
- **What is left, measured:** a posterior skull band ~0.5-1 px deeper and darker than real (minimum 0.13-0.17 vs
  0.19-0.27); the side / front scalp 12-15 % dimmer than real against white matter while the back matches (fastMRI's
  fat brighter than our CPMG fat T2 gives, IXI's dimmer: sequence-dependent fat, i.e. J coupling, ROADMAP M6.1); and
  population: fastMRI's clinical patients carry more deep CSF at slices 3-7 than a 20-82-year spread of IXI volunteers
  (median 554 vs 438 px; IXI subjects over 55: 566; age correlation 0.70), and one real scan has a large lesion.
- **Iteration**: validation/realism/quick.sh (c0fe213) answers in 10 min on the mini (8 heads x slices 2 / 4 / 6,
  8 spins per voxel, score --quick) against ~5.5 h for a full run; its CNN is noisy (+-0.15), its intensity and
  targeted measures are not. Calibration: q27 (r27's heads) 0.774, q29 (v14 heads) 0.809, q29r12 (12 cm coil ring)
  0.769, q29f200 (fat T2 200 ms, a test, not a database change) 0.851 with W1 0.0236 -> 0.0127.
- **Age-matched subjects, quick test (q30)**: the 11 prepared v14 subjects aged 50-82 instead of the 20-82 spread:
  CNN 0.894 [0.77, 0.98] (q29 0.809), W1 0.0207; quartiles 0.200 / 0.280 / 0.492 (real 0.199 / 0.280 / 0.460; the
  median now exact), but hemisphere texture back to 1.1-1.7 x real (older brains' sulcal CSF) and the posterior brain
  at 10 px (real 7, q29 8.5). The real population is not simply IXI's older half; age matching alone is not a win.

## 2026-10-13 — M6.1 J-coupled fat: the model, and why it is not the bright-fat story

- `blochparty.jcoupling` (density matrices of coupled spin fragments under single spin echoes or CPMG trains; ideal
  pulses, strong-coupling Hamiltonian) and a triglyceride fragment model whose composition reproduces the simulator's
  six-peak spectrum (methyl 8.7 %, 2.1 ppm 12.8 %, diallylic 0.4 %, 5.3 ppm 4.8 %: 1.9 double bonds, 0.2
  methylene-interrupted). Tests: no coupling or equivalent spins give no modulation; a weakly coupled AX pair's echo is
  |cos(pi J TE)| to 1 % (the strong-coupling correction, J / dnu); a 2 ms CPMG train keeps a 0.4 ppm pair above 0.95
  where one echo at 100 ms is below 0.8.
- Against measured 3 T fat (validation/jcoupling_fat.py): calibrated to CPMG 145 ms at 8 ms (Keene 2020), the model
  needs an intrinsic T2 of 266 ms and predicts single-echo fat at 222 ms where 48 ms (Jordan 2013) and 68 ms (de
  Bazelaire 2004) are measured; fat at TE 100 ms only 1.13 x brighter in a 10.26 ms train than in one echo (Henkelman
  1992: up to ~2). About half the fat signal is bulk methylene with same-shift neighbours, unmodulated in the model,
  and the coupled groups' modulation partly recovers by TE 160. So the in-vivo SE / CPMG fat difference needs more
  (the methylene chain's many-spin dynamics, or what else fast refocusing suppresses); no echo-spacing fat T2 is
  shipped to the simulator. For the realism question: at ~113 ms the J modulation is 0.687 with 8 ms spacing and
  0.663 with fastMRI's 10.26 ms, so J coupling would make that protocol's scalp fat ~3.5 % dimmer than the CPMG value,
  not the 12-15 % brighter the side / front scalp reads.

## 2026-10-13 — M5: dermis and eyes

- **Tissue values** (3f4cf1b): dermis T2 12 ms (>= its T2* ~10 ms at 1.5 T, Song 1997; assumed at both fields), PD 0.6
  (Richard 1993), muscle's T1 and MT (assumed); vitreous (class eye) T1 4.518 s at 3 T (Simpson 2024), 3.257 s at
  1.5 T and T2 546 ms (Ettl 1994; the 1.5 T T2 used at 3 T, flagged). tissuedb gains the measured rows, a
  stand-alone dermis tissue (as a child of skin it would have changed skin's pooled values) and eye.vitreous as class
  eye. The generator's classes draw from their own stream (other classes' draws unchanged; tested at 1.5 / 3 T); core
  snapshot bit-identical.
- **anatomy.add_dermis / sample_head(dermis=True)**: the outer 1.61 mm (Li 2024) of the head as dermis, by overlap,
  totals conserved. Quick test q32 (v14 heads + dermis) against q29: W1 0.0236 -> 0.0173, quartiles 0.211 / 0.293 /
  0.497 (q29 0.214 / 0.312 / 0.513, real 0.199 / 0.280 / 0.460), side / front scalp over WM 3.66 / 3.50 (3.43 / 3.41,
  real 4.00 / 3.88), but the posterior edge sharper (signal 1 px outside 0.09, q29 0.14, real 0.22); CNN 0.919 (q29
  0.809, +-0.15 at this size). Kept opt-in (`simulate_heads --dermis`), not in the preset.
- **The eyes** (b8b9106): prepare_ixi's orbit_vitreous finds the vitreous (bright T2, dark T1) in the orbits, the
  largest piece per side; IXI012: two globes ~24 x 21 x 27 mm at MNI (-35, 56, -39) / (33, 55, -39), 9.0 mL at its own
  size (before: 80 % skin, 20 % fat). The first version kept 84 voxels instead of ~12,700 (bincount over labels is
  indexed by label; a "+ 1" picked the neighbouring component); the synthetic test also caught a T1 sigmoid that
  capped pure vitreous at 0.87. 24 heads being prepared as v15. No effect on the realism score (fastMRI's stack
  starts above the orbits).
- Not done: dura and connective tissue relaxation (no in-vivo measurements found; muscle's stay, flagged).

## 2026-10-13 — Where the gap sits now (`validation/realism/gap.py`)

Diagnostics on r27's scored images (147 real from 21 scans, 112 simulated from 16 IXI heads, slices 1-7), to decide
the next realism steps; the three scratch analyses became `validation/realism/gap.py` (features, polar, structure,
controls).
- **The benchmark's own resolution** (`controls`): real vs real split by scanner model scores 0.52 / 0.56 (Skyra vs
  Prisma + mMR, two CNN seeds) and 0.46 / 0.42 (Prisma vs Skyra); by echo spacing 0.58 / 0.46 (one scan against
  twenty). The real set's scanners are not told apart, so 0.875 is signal. But the CNN's own seed moves the full
  number: seed 1 gives 0.927 (seed 0 0.875), and against the 11 Skyra scans alone the simulations score 0.99 / 0.94.
  With 21 against 16 people the interval is +-0.12 and the seed adds +-0.05: the next steps need a bigger benchmark
  (more heads, more scans, seeds averaged) before changes below ~0.1 can be read.
- **Per scan and per head**: 18 of 21 real scans score < 0.2 (three Skyra scans 0.93-0.96); the simulated heads are
  bimodal: ixi150 / 277 / 013 / 216 / 130 at 0.07-0.26, ixi255 / 334 / 511 / 167 / 538 / 146 / 205 at 0.89-1.00. The
  caught heads are mostly the older women (58-82 y) with the most CSF (share 0.18-0.27), but ixi277 (61 y) passes and
  real scans with as much CSF (0.22-0.24) score as real: it is how the CSF-rich heads look, not how much CSF they have.
- **Features** (AUC of one feature alone, > 0.5 larger in the simulations): the head centroid row 63.2 +- 1.9 against
  the real 60.4 +- 2.6 (AUC 0.83): our heads sit 2.8 px (4.8 mm) further anterior in the field of view (the real
  operators centre the FOV 6 mm behind the head's centroid); the interior p99 1.04 against 0.95 (0.76): in the real
  scans the scalp fat sets the p99 scale above the CSF, in ours the CSF is the brightest; posterior scalp 0.53 vs 0.47
  (0.68) and anterior scalp 0.36 vs 0.45 (0.28): a posterior / anterior ratio of 1.48 against the real 1.03, the
  receive ring behind the head. Area 0.95 x real (IXI volunteers' heads are smaller), scalp 0.41 vs 0.44, the band
  minimum 0.125 vs 0.140. Within the simulations the CNN score tracks the CSF share (Spearman +0.33) and nothing else
  above 0.25.
- **Polar map** (angle x radius / head radius, mean over images): sim / real 0.72-0.89 at the posterior 0.80-0.94
  (the deep dark band), 1.31 at the posterior edge (0.98-1.02), 0.83-0.93 at the sides' and front's 0.90-1.02 (the
  dim scalp), 1.08-1.14 at the anterior 0.80-0.90; inside 0.8 of the radius everything is within 2-10 %.
- **Structure**: interior power by band relative to the real mean: all simulations 1.16 / 1.28 / 1.25 / 1.13 / 1.03 /
  0.90 (0.05-0.15 ... 0.8-1.0 of Nyquist); the caught heads 1.36 / 1.40 / 1.36 / 1.22 / 1.10 / 0.96 at CSF share 0.196,
  where the real scans with CSF share > 0.17 (mean 0.235) carry 1.39 / 1.29 / 1.29 / 1.24 / 1.25 / 1.21. At a matched
  CSF share ours have ~15-20 % too much coarse structure (0.15-0.45) and ~10 % too little at the top band: the sulcal
  CSF of the older heads is rendered coarser and smoother than real. CSF-bright components under the edge: 14.3 per
  1000 px, mean 9.9 px, bright fraction 0.137 (real 13.9, 11.5, 0.152): count right, pieces smaller.
- The scratch `profile.py` in the mini's data directory shadowed the stdlib again (LESSONS, 2026-10-11); scripts now
  run from their own directory.
- **Not the receive ring (q33)**: the v14 heads with a 0.5 m ring centred on the head (`--coil-radius 0.5
  --coil-gap-mm 405`, a flat receive) keep the posterior / anterior scalp at 1.49 (posterior 0.51, anterior 0.34; real
  0.47 / 0.45), so the asymmetry is the anatomy (our anterior scalp is dim), and the brain reads 20 % too bright against
  the scalp's p99 (quartiles 0.232 / 0.288 / 0.438, white-matter mode 0.229; real 0.197 / 0.239 / 0.365, 0.194): the
  14 cm ring behind the head has been compensating a scalp ~20 % too dim against brain, or a brain too bright. CNN
  0.825 [0.65, 0.96], W1 0.034 (q29, the same heads with the ring: 0.809, 0.024).
- **The anterior scalp is the preparation**: along rays at MNI z 10-45 mm in eight v14 heads (scratch scalp_ap.py),
  the frontal scalp holds 0.9-2.1 mm of labelled fat (IXI334 4.2) in 3.7-7.9 mm of soft tissue, the occipital scalp
  2.5-4.3 mm in 5.3-10.3; the frontal bone band is 0.8-2.6 mm thick where the occipital is 1.8-6.6 (the forehead bone
  is ~6 mm in life). Too little frontal fat makes our anterior scalp 0.36 against the real 0.45; too little frontal
  bone leaves brain and scalp where the real dark band is (the polar map's 1.08-1.14 at the anterior 0.80-0.90).
  Both sit in prepare_ixi's unmixing at the front, to be checked against IXI's own T1 (fat bright, bone dark).
- **What the forehead is in IXI012's own scans** (raw T1 and T2, midline rays from the skin inward at three levels
  above the orbits): skin and fat 2-3 mm (T1 0.5-0.8 x WM), soft tissue 3-4 mm, then a 4-5 mm band bright in T1
  (0.5-0.7) and intermediate in T2 (0.4-0.5) between two dark tables, then CSF: the frontal diploe. Our rules call bone
  only what is dark in both (r1 < 0.35, r2 < 0.45), so the diploe becomes "skin" (7.0 mm of skin on the anterior rays
  against 5.9 at the back; skull 2.4 against 4.5), rendered at skin's 0.77 x WM where the real frontal bone is dark.
  At the occiput the diploe reads 0.2 in T1 and 0.4-0.56 in T2 and comes out half skull, half skin. The frontal
  scalp's fat itself is thin in the T1 (~2 mm), so the anterior scalp deficit is partly real anatomy (IXI's young
  volunteers against fastMRI's patients) and partly the shift-undo at the front. Next: a diploe rule (a band between
  two dark tables, T1-bright) labelled marrow, with marrow's T2w brightness set from the real scans' calvaria
  (frontal dark, occipital brighter: the aligned profiles).
- **Not the bound pool either (q34)**: q33's flat receive with MT off (`--no-mt`, new): white-matter mode 0.332 against
  0.229 with it, so the simulated incidental MT of the 16-slice train takes 31 % off the brain, inside the 30-45 %
  multi-slice MTR reported at 3 T (REALISM.md, [21]); the brain's 20 % excess over the scalp under a flat receive is the
  scalp fat's dimness against brain (the J-coupling route was negative; what is left: adipose water and PD, the
  diploe below), not missing MT. (CNN 0.897, histogram 0.993, W1 0.088: the ablation, not a candidate.)
- **The real calvarium at 0.57 mm** (scratch calv_real.py: 12 scans, slices 2-5, rays +-30 deg, in units of each
  slice's white-matter mode): posterior, from the edge: 0.82 at the skin, the scalp peak 5.0 at 5.7 mm, the bone
  valley 0.81 at 10.3 mm (one valley ~2 mm wide, no separate diploe maximum in the median), then 1.1 / 1.4 / 1.7 / 2.0
  at 11.5-15 mm (cortex and CSF); anterior: 1.09 at the skin, the scalp peak 3.1 at 2.3 mm, the valley only 1.19 at
  5.7 mm, then a slow rise 1.4-2.0 over 7-12 mm: the 5 mm slab cuts the sloping forehead obliquely and smears
  scalp, bone and brain together. Our classes at this protocol (spin-echo signal over WM): fat 3.21, marrow 0.73, skin /
  dura / connective 0.71, muscle 0.16, dermis / skull 0. Relabelling the frontal diploe skin -> marrow would change
  nothing (0.71 -> 0.73): the anterior excess at 0.80-0.90 of the radius is the band's composition and the slab
  geometry, to be tested on the simulated slices against these profiles (item 4 of M5b), not a one-line rule.
- **Found: the frontal bone was labelled skin under displaced fat** (prepare_ixi `--save-intensities`, IXI012's r1 / r2
  on the MNI grid, viewed): in IXI's T2 the scalp's fat is displaced 1.63 mm inward at the forehead and vertex (the
  water-fat shift along the front-back readout), so a 3-voxel band inside the frontal bone reads T1-dark (r1
  0.06-0.21: bone) and T2-bright (r2 1.4-1.8: the fat's ghost). The rule "bone = dark in both" called it skin, which
  the simulator renders at 0.71 x WM: the 7 mm of "skin" on the anterior rays, the 2.4 mm of frontal bone, and the
  anterior band's 1.08-1.14 x real. Fat is T1-bright, so a T1-dark, T2-bright voxel next to bone (within 3 mm; not
  in the eyes) is bone under displaced fat: the T2 factor is waived there. IXI012 (from the saved maps): frontal
  bone 2.4 -> 3.6 mm, occipital 4.5 -> 5.2 mm along the rays; 96 mL of such voxels in the head outside the brain, 58
  of them below MNI z 0 (face, sinuses). The 64-head v16 preparation was stopped (7 heads in) and restarts with the
  rule; the orbit's vitreous is found before the rule so it stays an eye.
- **Placement fixed (q35)**: with the field of view 4.8 mm anterior of the head's box (`--centre-offset-mm`, the preset's
  default now) the v15 heads' centroid row is 60.1 +- 1.2 against the real 60.4 +- 2.6 (that feature's AUC 0.83 -> 0.48);
  CNN 0.872 [0.70, 0.99], W1 0.0245 (q29 0.809 / 0.0236: within the quick CNN's noise). Still there, now the largest
  single features: posterior / anterior scalp 0.59 / 0.32 against 0.47 / 0.45 (AUC 0.76 / 0.20), the quartiles 10-15 %
  high (0.215 / 0.275 / 0.420 vs 0.197 / 0.239 / 0.365) with the scalp as the p99 scale.
- **v16** (the 64 selected subjects, the bone rule, prepared on the mini): IXI012's anterior rays now 3.4 mm of bone (2.2
  before) and 5.4 mm of soft tissue (6.2), posterior 4.8 (4.2); 64 mL of bone under displaced fat with the 3 mm guard.
- **Levels over the white-matter mode** (gap.py features, new: free of the p99 normalisation): real q50 / q75 / CSF
  (interior p99.5) / scalp (outer 6 px p90) = 1.24 / 1.89 / 5.27 / 4.40. With the ring (q35): 1.32 / 2.03 / 4.93 /
  4.07 (the brain +7 %, CSF -6 %, scalp -8 %); with the flat receive (q33): 1.25 / 1.92 / 4.78 / 3.49 (the brain
  right, CSF -9 %, scalp -21 %). So the real images are flat-receive images and the 14 cm ring has been buying the
  scalp's brightness with a brighter periphery of the brain. What is left under a flat receive: the scalp 21 % and
  CSF 9 % dim against white matter. A stronger incidental MT (ours 31 %; 37-40 % would put CSF / WM right and the
  scalp at -13 %) and the fat's own brightness are the candidates; the test that separates a residual peripheral
  receive gain from fat is the non-fat scalp tissue and the bone minimum at the scalp's radius (the real skin edge
  0.82-1.09 x WM at 0.57 mm). Next quick runs: flat receive with the bound-pool fraction scaled 1.3 (`--override`),
  and with fat's T2 at 170 ms, each against these four levels.
- **The real set grows**: the fastMRI link was not expired (a HEAD request is refused, a ranged GET works; LESSONS).
  `fetch_fastmri.py` streams the validation archive again into `~/data/fastmri_more` on the mini (symlinks to the 71
  files already kept, the AXT2 cap raised 15 -> 27 GB: ~40-60 more 3 T AXT2 volumes, 8 GB of disk kept free). The
  queued runs (r30, q36-q39) keep the frozen 21-scan set; r31 (64 v16 heads x 7 slices) scores against the larger one
  and r30 will be re-scored against it for the comparison. score.py's noise-covariance cache now grows with the set.
- **r30: CNN AUC 0.748 [0.594, 0.883]** (24 v15 heads x slices 1-7 at quick quality, 8 spins per voxel, the FOV offset;
  168 simulated against the 147 real slices; r27: 0.875 on 16 v12 heads at 64 spins), histogram 0.549 [chance], W1
  0.0181; controls real-vs-real 0.440, sim-vs-sim 0.527; by slice 0.72-0.77. gap.py on it: the placement (row AUC
  0.54) and every level over the white-matter mode are within noise of real (q50 1.24 vs 1.25, q75 1.93 vs 1.92, CSF
  5.24 vs 5.29, scalp 4.21 vs 4.37; the ring is still in, so the brain's +7 % periphery stays); the head area 0.94 x
  real (0.40); posterior / anterior scalp 0.54 / 0.38 vs 0.47 / 0.45 (0.67 / 0.34: the frontal bone rule is in v16,
  not here); structure by band 1.03 / 1.05 / 0.99 / 0.91 / 0.84 / 0.86 x real: the coarse excess is gone (refine_csf
  and the subjects), the fine bands 15 % short (the reader's "smoother" cue; q36 tests the residual's smoothing, and
  the 8-spin quality adds grain, not smoothness). 105 of the 168 simulated images still score > 0.8 with no feature
  here separating them from the 61 that pass: the cue left is finer than these measures.
- **q36, the residual smoothed 0.75 mm instead of 1.5** (the same 8 heads as q35, re-prepared as v15s): structure by
  band 1.02 / 1.10 / 1.10 / 0.99 / 0.89 / 0.92 x real against q35's 1.02 / 1.08 / 1.05 / 0.92 / 0.82 / 0.86: the fine
  bands up 8 %, the coarse unchanged; CNN 0.792 [0.59, 0.95] (q35 0.872; noise), W1 0.0253 (0.0245). Half the fine
  deficit; a 0.5 mm variant (q40) follows, with the v10 lesson in mind (the unsmoothed residual carried IXI's noise at
  1.4-2.4 x real fine texture). The 64-head v16 preparation keeps 1.5 mm; the default moves with the next preparation.
- **Flat receive on the v15 heads, three ways** (q39 alone, q37 with the bound-pool fractions x1.3, q38 with fat T2
  170 ms; levels over the white-matter mode, real q50 / q75 / CSF / scalp = 1.24 / 1.89 / 5.27 / 4.40): q39 1.30 / 1.98
  / 4.78 / 3.40; q37 1.32 / 2.09 / 5.26 / 3.78; q38 1.30 / 1.98 / 4.77 / 3.60. The stronger MT puts CSF / WM exactly on
  real but lifts q75 / WM 10 % (grey matter loses less to MT than white); fat's T2 moves the scalp 6 %; neither
  reaches it: the scalp stays 14-23 % dim against white matter under a flat receive, and the posterior / anterior
  ratio 1.44 (these heads lack the frontal bone rule). CNN 0.78 / 0.90 / 0.74 (noise), W1 0.040 / 0.029 / 0.036 (the ring's
  q35: 0.025; a flat receive without the scalp fixed is worse on W1). Against the real 0.57 mm layers (skin 0.8-1.1,
  the soft tissue under the fat 1.9, fat 5.0 x WM) our classes give skin 0.71, muscle 0.16, fat 3.2: the scalp's soft
  tissue (galea, loose connective tissue) is 2-3 x too dark and the fat 1.5 x; both are flagged proxies (muscle's
  relaxation for the soft tissue; CPMG fat T2). Next quick runs: skin's T2 90 ms with the MT x1.3 (q41), plus fat 170 ms
  (q42), against the four levels, the edge profile and W1; if they land, the values enter the table as calibrations
  on fastMRI's 0.57 mm profiles, flagged as such.
- **q40, the residual smoothed 0.5 mm**: bands 1.04 / 1.11 / 1.13 / 1.03 / 0.94 / 0.97 x real (q36 at 0.75 mm: ... 0.89 /
  0.92; q35 at 1.5: ... 0.82 / 0.86): the fine structure is within 3-6 % of real, the mid band 13 % over; W1 0.0276
  (q35 0.0245), CNN 0.876 (noise). 0.5 mm becomes prepare_ixi's default for the next preparation (v16, in progress,
  keeps 1.5; r31 runs on it).
- **q41 / q42, the scalp under a flat receive with MT x1.3** (skin T2 90 ms; + fat T2 170 ms): levels over WM
  q50 / q75 / CSF / scalp 1.34 / 2.10 / 5.27 / 3.98 and 1.33 / 2.10 / 5.26 / 4.24 (real 1.24 / 1.89 / 5.27 / 4.40):
  CSF exact, the scalp within 4 % with both changes, but grey matter 8-11 % high over white (5 % at MT x1 in q39, the
  rest from the stronger MT, which darkens WM more than GM); W1 0.038 / 0.033 against the ring's 0.025, the posterior
  scalp 0.58-0.59 vs 0.47 (anatomy; the v15 heads). So the flat receive needs the whole scalp calibration at once and a
  milder MT (x1.15 would put GM / WM within ~1 % and CSF / WM within ~4 %); decided after r31 (64 v16 heads, the ring)
  gives the scaled baseline, as r32 with the flat receive, MT x1.15, skin 90 ms, fat 170 ms, each value flagged as a
  calibration on fastMRI's levels.
- **The scalp at 0.57 mm, simulated against real** (IXI012 and IXI255 of v16 at `--n 384` under a flat receive, slices
  2-4; both sides smoothed 1.1 mm against the simulation's grain of ~2 spins per pixel, ray means from the edge in
  units of the white-matter mode; scratch calv_sim.py): posterior peak 2.05 at 4.6 mm against the real 4.47 at 5.7,
  valley 0.52 at 9.2 against 1.05 at 11.5, the brain beyond 1.7-1.8 against 2.1-2.3; sides 1.81 / 1.00 against 2.98
  / 0.93 (the valley 3 mm deeper in real); anterior 1.02 / 0.64 against 2.78 / 1.18. Everything in the outer two
  centimetres is 1.3-2.7 x brighter in the real scans while the interior's distribution matches a flat receive
  (q39); a 2 x on fat and the soft tissue is beyond their physics (fat's T2 would have to be ~650 ms), so this is
  the receive after all, in its normalised form: the array's sensitivity rises steeply in the last 1-2 cm and the
  scanner's normalisation divides by a smooth estimate that cannot follow it. The 14 cm ring (q35) has the rise but
  spread over the whole brain (+7 % at the periphery of the brain). Model it as the ring's sensitivities divided by
  their own spatially smoothed RSS (`--receive-normalise-mm`, next), and read the hi-res profiles again. The
  calibration rows for skin and fat are not written. Also seen: the real scalp is ~2 mm thicker (patients) and the
  real valley never as dark as ours (0.93-1.18 vs 0.52-1.00).
- **Normalised receive, first geometry (q43: ring 0.12 m, 15 mm behind the occiput, RSS smoothed over 30 mm)**: levels
  q50 / q75 / CSF / scalp over WM 1.29 / 1.97 / 4.77 / 3.77 (flat 1.30 / 1.98 / 4.78 / 3.40; the ring 1.32 / 2.03 /
  4.93 / 4.07; real 1.24 / 1.89 / 5.27 / 4.40): the interior stays flat and the scalp gains 11 %. At 384 px the
  posterior peak goes 2.05 -> 2.72 (real 4.47), the sides 1.81 -> 1.90 (2.98), the front 1.02 -> 1.37 (2.78): the
  right kind of change, a third of the size. A loop's field falls as (a^2 + d^2)^-3/2 (a its radius, d the distance),
  so the real excess of ~2 x over the outer 2 cm needs elements 1-2 cm from the scalp all round, which a circle
  around a 18 x 15 cm head cannot do: an elliptical ring next (q45 / q46), then the geometry fitted on the three
  0.57 mm profiles. The sides also show our temporalis region too dark (0.62-0.76 vs 0.93-1.0 x WM at 8-12 mm:
  muscle at T2 30 ms is near black at TE 113; the real temporalis reads ~1 x WM, partial volume with fat and bone).
- **Elliptical normalised receive (q45: semi-axes 0.11 m AP x 0.095 LR, 10 mm behind the occiput, RSS smoothed 30 mm;
  q46 the same smoothed 20 mm)**: q45 levels 1.31 / 1.99 / 4.84 / 4.10 over WM (real 1.24 / 1.89 / 5.27 / 4.40), W1
  0.0240 (the best so far; the ring 0.0245, flat 0.040), histogram AUC 0.545 (the lowest so far), CNN 0.793; q46
  3.77 at the scalp (a shorter smoothing follows the rise and removes it: the excess kept grows with the smoothing
  length). At 384 px with q45's receive: posterior peak 3.24 (flat 2.05, real 4.47), valley 0.69 (1.05), the brain
  beyond 1.95-2.45 (2.1-2.3); sides 2.17 (2.98); front 2.01 (2.78): the outer 2 cm uniformly 0.72 x real now, the
  interior right. q47 (0.105 x 0.09 m, 8 mm, smoothed 45 mm) next; the seven missing v16 heads are prepared (IXI598
  skipped: a non-finite map), r31 runs on the 57.
- **q47 (0.105 x 0.09 m, 8 mm, smoothed 45 mm)**: scalp 4.58 over WM (real 4.40), W1 0.0161 (the best), histogram
  0.712 (worse than q45's 0.545): the front elements now too close, the anterior interior over-boosted at 384 px
  (3.7-4.5 vs the real 2.9-3.1) while the posterior peak barely moves (3.27 vs q45's 3.24, real 4.47): with this
  model's 1 / d falloff a tighter ring brightens the front interior before it lifts the posterior scalp. A loop's
  own near field, a^2 / (a^2 + d^2)^(3/2), is steeper within a few radii of the element (`element_radius`, next,
  q48); r32 (the 64 v16 heads) is chained behind r31 with q45's receive.
- **q49 / q48 on the v16 heads** (q45's elliptical normalised receive; q48 the same with 4 cm loop elements instead
  of 1 / d): q49 levels 1.26 / 1.98 / 4.78 / 4.52 over WM (real 1.24 / 1.89 / 5.27 / 4.40), histogram AUC 0.432 (the
  first at chance with the scalp right), W1 0.031, CNN 0.847; with the bone rule the anterior scalp is now 0.44
  against the real 0.45 (AUC 0.44; it was 0.32-0.36), the posterior 0.56 vs 0.47 stays. q48: scalp 5.72 (+30 %),
  histogram 0.862; at 384 px its posterior peak lands (4.30 vs 4.47) but the brain beyond it reads 2.3-3.1 (real
  2.0-2.3) and the anterior interior 3.0-4.4 (2.9-3.1): the loop near field boosts the outer brain with the scalp,
  the 1 / d ring under-boosts the scalp alone. The anterior scalp peak stays short in both (2.0 vs 2.78): the
  frontal scalp's composition, not the coil. r32 runs with q45's receive (1 / d ellipse); the element model needs
  the real array's geometry (element size and spacing) rather than another guess.
- **The real set grown to 34 scans** (13 new 3 T AXT2 volumes from the stream; `~/data/fastmri_more`): r30 re-scored
  against it reads CNN 0.904 (0.748 against the 21), controls 0.54 / 0.55, W1 0.0138. The new scans' periphery is
  dimmer: real scalp over WM 4.06 against the old set's 4.37, posterior scalp 0.44 vs 0.47, while r30's ring gives
  4.21 — brighter than this real set. The real scans are not one receive: sites and coils differ, and some may not
  be intensity-normalised at all. The receive calibration must be judged per scanner model / coil count, not on the
  pooled set (scratch realscans.py lists them).
- **The 34 scans one by one** (scratch realscans.py on score_r30m): Biograph mMR 14 coils (5), Prisma 16 / 20 (7),
  Skyra 12 / 16 / 20 (20), TrioTim 4 coils at 120 deg (2); the scalp's bright decile over WM runs 2.1-6.0 (mMR
  3.2-4.6, Prisma 2.7-4.9, Skyra 3.1-6.0, TrioTim 2.1-2.8), posterior / anterior 0.66-2.7, CSF / WM 3.5-6.2; our
  runs sit at one scalp level (4.5 +- 0.4 in q49) and one ratio (1.45). The receive is not one thing across scans
  (coil, positioning, the site's normalisation), so the model draws it per head (`--receive-draw`: semi-axes x
  U(1, 1.25), gap U(5, 30) mm, normalisation length U(20, 60) mm), to be checked on the spread of the scalp level
  (q50) and then as r33 on the 64 heads behind r31 / r32.
- **q50, the drawn receive on 8 v16 heads**: scalp over WM 4.50 +- 0.79 against the real 4.40 +- 0.82 (q49 with one
  receive: 4.52 +- 0.34), posterior / anterior 0.78-2.16 per head (real 0.66-2.7), q50 / q75 over WM 1.24 / 1.94
  (1.24 / 1.89), histogram AUC 0.417, W1 0.036, CNN 0.907 (8 heads; noise, and two of these heads look wrong: ixi175
  with 9033 px of area and a CSF share of 0.012, ixi239 with 4844 px: the proxy-based selection admitted extremes
  that need a screen on the prepared heads (area, CSF share, lncc) before r33's score is read). r33 (64 heads, drawn
  receive) is chained behind r32 and r31b.
- **v16 screened** (the preparation metadata of the 63 heads): lncc 0.195 +- 0.014, CSF 179 +- 54 mL, GM 713 +- 130,
  WM 492 +- 66, skull 305 +- 58, fat 318 +- 101 mL. IXI175 is a failed preparation (CSF 26 mL, refine ratio 0.06,
  GM 1064, fat 880: the one q50 flagged) and IXI586 registered badly (lncc 0.137 < IXI_MIN_LNCC 0.15); both moved to
  prepared_v16_rejected before r31b and r33 list their heads (r32 had listed 64 and skips 175 with an error).
  IXI097 (lncc 0.154) and IXI048 (skull 473 mL) stay. The proxy selection needs these two checks built in
  (select_ixi after preparation: lncc and the CSF refine ratio), noted for M5b item 2.

## 2026-10-14 — The 62-head runs against 34 scans: the CNN at 0.97-1.0

- Overnight on the mini, the v16 heads (62 after the screen) x slices 1-7 at quick quality, scored against the 34
  real scans (`~/data/fastmri_more`): **r31b** (the preset's 14 cm ring) CNN 0.998, W1 0.0161; **r32** (one elliptical
  normalised receive, q45's) 0.967, W1 0.0143; **r33** (the receive drawn per head) 0.9996, W1 0.0170; controls
  real-vs-real 0.536, sim-vs-sim 0.54-0.56 in all three. r30 (24 v15 heads, the ring) reads 0.904 against the same
  real set, 0.748 against the old 21.
- gap.py on all three: every tissue level over white matter within noise of real (q50 1.24-1.25 vs 1.25, q75 1.91-1.96
  vs 1.95, CSF 5.20-5.31 vs 5.16, scalp 4.36-4.46 vs 4.06), placement right (row AUC 0.52), head area 0.92-0.94 x
  real (0.33-0.40); the strongest single features the outer 6 px (0.45-0.46 vs 0.42, AUC 0.66-0.68) and the
  posterior scalp (0.51-0.54 vs 0.44, 0.67-0.73); structure by band 0.91-0.96 / 0.95-1.00 / 0.90-0.95 / 0.85-0.88 /
  0.80-0.83 / 0.82-0.85 x real (v16 is at 1.5 mm residual smoothing; 0.5 mm from v17). Every simulated image scores
  > 0.8: whatever the cue, it is in all of them and finer than these measures.
- The receive draw changed nothing the CNN cares about (r33 = r31b), though it put the scalp's spread right.
- **It is the head count**: 24 of r31b's heads (every fifth left out) against the 34 scans score 0.89, r30's 24 v15
  heads 0.904: the v16 heads and the ring are equivalent at that size, and 62 heads let the CNN learn a cue that is
  in every simulated image. The benchmark has the power M5b item 2 asked for; the cue is now the question.
- **The cue is the quick quality's grain** (gap.py `grain_wm`, Immerkaer's estimate in flat white matter as a fraction
  of p99): real 0.010 +- 0.003; r31b / r30 / q35 at 8 spins per voxel (2 x 2 x 2) 0.015 +- 0.003, that one feature's
  AUC 0.92-0.98; r27 at 64 spins (4 x 4 x 4) 0.010, AUC 0.62. The sub-voxel spin grain (journal 2026-10-11: it falls
  as 1 / sqrt(in-plane spins)) is what 62 heads let the CNN learn; the 24-head runs could not resolve it. So quick.sh's
  CNN number was never the right readout (its W1 and gap.py measures are), and every benchmark run needs the grain at
  the real 0.010: 2 x 3 x 3 spins (18; in-plane 9, 2.25 x the quick cost) should land there. r34 = the 62 v16 heads at
  2 x 3 x 3 with the drawn receive, ~8 h; v17 (the 0.5 mm residual) prepared beside it on the CPU. The receive,
  placement and bone findings stand (their measures are independent of the grain).
