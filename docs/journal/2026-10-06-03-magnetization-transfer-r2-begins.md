# 2026-10-06 (late) — magnetization transfer (R2 begins)

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
