# Lessons: Physics and numerics

- **The generalized Bloch split engine is ~1e-4 accurate, whatever the segmentation.** Its RF
  segments are sub-stepped to h ≤ T2b / 2 and Strang-split there, so `subdivide_rf(k)` barely
  helps (error falling 1.5x per halving of the segments, not 4x): on a Lorentzian random case its
  Mzb was 1.8e-4 off a SciPy ODE of the equivalent Bloch pool, the exact engine (modes as matrix
  rows) 2.6e-13. Adding an inert free pool (f = 0) routes a phantom through the exact engine.
  Source: M1c, 2026-10-08.

- **Subdividing a sequence changes its diffusion walks.** Philox draws are keyed by segment
  index, so `seq.subdivide_rf(8)` renumbers the segments and gives every spin a new path: two runs
  differed by 1e-4 (Monte Carlo), the same subdivided sequence on both sides 2.6e-7. Compare
  engines on one segmentation. Source: M1c, `test_pools.py`, 2026-10-08.

- **Mid-segment positions are exact for G·r, not for the concomitant field.** Moving spins see each
  segment at their mid-time position: exact for terms linear in t, but f_c is quadratic in r, so a
  constant velocity leaves C dt³/12 per segment (6e-3 rad for 5 ms lobes at 2 m/s, 32 mT/m, 0.55 T).
  Checked against the exact path integral (3-point Gauss-Legendre per segment): the error falls
  4.0x per halving of the segments. Source: M1b, `test_fields_concomitant.py`, 2026-10-08.

- **Float32 resolves a segment's phase increment to |φ| 2⁻²⁴.** A 5 ms lobe of 32 mT/m seen from
  15 cm is ~6400 rad, so Metal and the core differed by 5e-4 rad on a moving-spin echo, with the
  ideal scanner too: not the field model, the arithmetic. Source: M1b, 2026-10-08.

- **FieldMaps are physical fields: sample them at true positions.** Static spins sampled the maps at
  their encoding positions (`spin_fields(...).pos`), which differ under a gradient nonlinearity;
  moving spins sampled at their (then encoding) positions too. Both now use true positions.
  Source: M1b, 2026-10-08.

- **Float32 drifts wherever a step rewrites the whole state.** Every Metal variant drifted from the
  float64 core by 1e-3 in Mz per 1e4 RF segments (1e-2 after 1e5 segments of 10 µs CW), because
  each RF segment rotated m by a tiny angle (cos φ ≈ 1 − 1e-5) and rounded all of it. Writing the
  step as an increment (m += s1 w×m + s2 w×(w×m), relaxation and exchange increments) and adding it
  by Knuth's two-sum into a per-spin low part brought it to ≤ 3e-7 at no measurable cost. Free
  segments (exact rotation, `mz += (mz − pd) e1`) still drift ~1.5e-5 per 1e5 segments. The short
  parity tests never saw it; the BMCTool MT comparison through Metal (1.2e-2 vs the core's 5.8e-7)
  did. Source: M1a, 2026-10-08.

- **A float32 matrix exponential per segment fails over a CEST train.** Prototyped against the
  float64 core: dense `matrix_exp` in float32 was 1e-2 off after 2000 segments of benchmark 9's
  saturation (2e-2 after 6000) and 3.5e-4 on a stiff case; a Taylor series applied to the state
  (structured A·v, sub-steps of norm ≤ 1.5, common precession of free segments rotated exactly)
  3.3e-5; with the increment added by two-sum 2-3e-7. The Taylor scheme in float64 was exact to
  3e-11, so all the error was state rounding. Source: M1a prototype, 2026-10-08.

- **The restricted walk is discontinuous in D, and a lucky D sits on a jump.** Sub-steps per
  segment are ceil(dt / tau_max) with tau_max = min over spins of R² / (200 D), so the walk (new
  Philox draws per sub-step) changes discretely with D or R. With D = 2e-9 m²/s and R = 5 µm,
  tau_max = 62.5 µs divides 1 ms and 9 ms segments exactly; a ±1e-6 relative perturbation of a
  per-spin D raises max D either way, so both finite-difference points ran 17 sub-steps where the
  base point ran 16. Autograd (exact for the 16-step walk) and the converged finite difference
  (exact for the 17-step one) then disagreed by 10 %. Autograd through a walk is right at fixed
  sub-step counts; pick parameters away from integer dt / tau_max when differencing. Source:
  `tests/test_engines.py`, 2026-10-08.

- **Compare a saturated pool against the water's scale, not its own.** After a 10 µs hard 180°
  the bound pool's Mz was 8.8e-5 of an equilibrium 0.12; Metal's error was 1.3e-8 (1e-7 of
  equilibrium) but 1.5e-4 relative to the pool's own maximum. `test_metal.py` already divided by
  max |m| of the water; `test_engines.py` does too. Source: `tests/test_engines.py`, 2026-10-08.

- **Susceptibility units do not specify its reference.** QSM full-head chi is referenced
  to water (air around +9.2 ppm), while challenge chi is brain-only with zero exterior;
  neither should inherit the absolute-air value +0.36 ppm blindly. Preserve a separate
  `chi_background` in the same reference through crop/resampling. A constant shift of
  both chi and exterior must leave the field unchanged. (`anatomy.py`, QSM M3)

- **Measured M0 is already voxel density.** Multiplying it by a tissue/air fraction a
  second time underestimates partial-volume signal. Distribute M0 among signal-producing
  memberships, and use M0 weights to average rates when coarsening; zero-density air
  cannot dilute R1. This preserves density integral but not a general multi-exponential
  time course. (`test_measured_density_and_rates_do_not_double_count_partial_volume`)

- **Cross-solver DC conventions can dominate an otherwise exact comparison.** The pinned
  qsm-forward kernel sets D(0)=1/3 while ours uses zero, giving ~375 Hz constant offsets
  for synthetic water/air heads at 3 T. After removing only the ROI mean difference,
  residuals are <6.83e-13 Hz. Match the exact encoded exterior/corner and padding first;
  do not hide disagreement with a slope or spatial fit. (`validation/anatomy_qsm.py`)

- **Fuzzy membership is a density, not subcell geometry.** Keep each tissue's T1/T2 and
  weight its spins by fraction × PD; averaging relaxation constants gives the wrong
  mixture signal. Jitter decouples spin positions from reconstruction, but a sharp
  boundary still needs an independently resolved anatomy grid. Volume-weighted
  resampling preserves sum(fraction) × voxel volume, not the unweighted sum.
  (`anatomy.py`, mixed-voxel Ernst and shifted-box Fourier checks)

- **Observed T2* and susceptibility can double-count dephasing.** There is no universal
  linewidth to subtract from a macroscopic gradient. Require calibrated macroscopic
  R2′ or an already separated intrinsic map before combining both. A single Cauchy
  quantile is the zero median: K=1 cannot produce a finite requested T2′ linewidth.
  (`test_t2prime_quantile_fid_and_no_double_count`, anatomy M2)

- **Published tissue numbers need population and uncertainty labels.** Stanisz 2005
  Table 1 contains ex-vivo animal specimens (human blood), not a normal human atlas.
  Missing SD is unknown, not zero, and applying a 3 T T2 at 7 T is an assumption.
  Keep the older BrainWeb simulator priors separate from current tissue priors.
  (`anatomy.py` provenance; `validation/anatomy/RESULTS.md`)

- **Normalize fat-peak weights before conserving tissue mass.** Existing rounded Ren
  amplitudes sum to 0.999; copying them directly loses 0.1% of PD. Normalize locally
  in new anatomy assembly without changing old phantom defaults. (`test_mt_diffusion_and_fat_bookkeeping`)

- **Even-grid dipole kernels need a Nyquist sign convention.** The usual `(k.b)^2`
  formula followed by `ifft(...).real` breaks joint object/B0 rotation covariance when
  two axes are at Nyquist (~2 Hz in a random small-grid test). Average over the two
  possible signs independently for every Nyquist axis: retain squared terms and zero
  their cross terms. Odd/even-grid quarter turns then agree to < 2e-12 Hz.
  (`susceptibility.py`, `tests/test_susceptibility.py`, anatomy M1)

- **Padding controls periodic copies, not tissue-boundary discretisation.** For a 10-mm
  partial-volume sphere at 3 T, far-field RMS falls 4.287 → 0.134 → 0.0216 Hz at
  1×/2×/3× padding; interior RMS improves 0.908 → 0.439 Hz from 32³ to 64³ at 2×.
  A long cylinder needs more transverse padding (4× here): 2× gave a ~6.1 Hz constant
  error even in its central plane. Subtract the surrounding susceptibility before
  padding absolute maps; zero padding an absolute head map otherwise puts vacuum,
  rather than air, outside the array. (`tests/test_susceptibility.py`; Jordan et al.,
  JMRI 2013, [doi:10.1002/jmri.23762](https://doi.org/10.1002/jmri.23762))

- **A susceptibility field does not simply rotate with the head.** Translation can
  carry the static field with tissue, but rotation changes the susceptibility/B0
  orientation and needs recomputation. Keep the scanner shim in `FieldMaps`, not in
  a moving phantom's combined `db0`. (`susceptibility.py`, `maps.py`)

- **Large float32 shim fits need explicit rank tolerances.** The default N*eps cutoff
  approaches one on multi-million-voxel masks, despite fitting only nine scaled
  harmonics. Set the relative cutoff from the number of columns. For order 0, use the
  mean directly: a one-column QR on >8 million voxels lost ~0.001 Hz even for a constant
  17 Hz map. (`susceptibility.py`, `test_float32_shim_large_mask`)

- **Speckle biases magnitude images upward, worst where the signal is small.** Spins are a
  Monte Carlo sample of the continuum; the complex image's MC noise is ~zero-mean, but
  E|s + n| ≈ |s| + σ²/2|s|. A water fill simulated at sphere-level density (64 spins/pixel)
  made short-T2 spheres' late echoes 2–4× too high (benchmark 2: 2.6 % → 5.8 %); the scanner's
  tails matched the simulation *without* water. Fixes: more spins, independent seeds with the
  power bias removed before a clipped square root (`nist.mc_magnitude`), lower-discrepancy
  placement (`nist.*_qmc`). This does not make magnitude unbiased; see the 2026-10-08
  estimator audit below. Ideal
  spoiling at TR end changed nothing (coherence across TRs was not the cause).

- **Multi-echo shapes constrain |B1⁺ − 1|, not its sign.** Refocusing at 180°·(1 ± δ) gives
  nearly the same echo train, and the 90°'s scaling is absorbed by any per-sphere scale. On
  the real NIST TSE the shape error is *highest* at nominal B1⁺ (2.57 %) and falls both ways
  (0.85: 2.03 %, 1.15: 1.64 %). A single global B1⁺ scale is the wrong digital-twin model for
  a 3 T water phantom (the scanner calibrates the average; the field varies spatially): fit a
  spatial map, and expect sign ambiguity unless the model is smooth or excitation-sensitive
  data is included. (`validation/real/nist_tse.py`)

- **Spins per voxel side must exceed the cycles of dephasing that still carry signal.** Too
  few spins don't fail loudly: the image speckles and signal nulls stay shallow, because the
  magnitude of a few random phasors isn't zero. Real 3 T IR-SE with crushers (~1.7 cycles per
  slice thickness): 16–32 positions across the slice → 14 % shape error; 64 → 3 %; 128 →
  1.4 %. Treat spin density as a convergence parameter and report it.
  (`validation/real/RESULTS.md`; also `validation/RESULTS.md` for RF spoiling: K must exceed
  the dephasing orders present, ~80 cycles per voxel after 40 TRs)

- **The textbook inversion-recovery formula misses real IR spin echo by ~26 %** (curve shape)
  without a free offset term; simulating the actual `.seq` (pulse shapes, slice profiles,
  timing) gets 1.4 % with nothing fitted. Fit models with ad-hoc offsets hide physics.
  (`validation/real/`)

- **Concomitant fields scale as 1/B0**: on the real IR-SE, leading-order Maxwell terms change
  sphere signals by 0.37 % at 3 T and 2.0 % at 0.55 T. Negligible for this sequence at 3 T,
  not at low field. (PR #6 verification)

- **float32 relaxation needs `expm1` forms**: `exp(-dt/T1)` rounds to ~1 for dwell-scale dt, a
  systematic ~2e-3 relative error in the per-step recovery that compounds over TRs. Write
  `m += (m - m0) * expm1(-dt/T)`; the adjoint must propagate λ the same way. (fused kernel)

- **Closed-form free runs work for any piecewise-constant gradients**, not only constant ones:
  phase = 2π(k·r + dB0·t) with cumulative k and t. Terms quadratic in g (concomitant fields)
  need cumulative ∫gᵢgⱼ dt, still exact. (`bloch.py`, PR #6)

- **Strang splitting of relaxation inside RF segments is second order** (error ↓ 4× per halving
  of the RF step), but the coarsest steps are pre-asymptotic (ratios 4.5–9). Without
  relaxation, rotations are exact to 1e-12. (`tests/test_bloch_vs_ode.py`)

- **Many first test failures were the tests' idealisations, not simulator bugs**: T2 decay
  during the readout blurs images (~5e-4 at a 10 µs dwell); an off-resonance comparable to the
  RF bandwidth stops excitation; a 1 µs pulse at 37 Hz offset accrues ~3e-5 of phase during
  the pulse; TI measured from pulse end instead of centre is ~1e-6 off. Investigate before
  loosening a tolerance.

- **±flip angles give identical magnitude images**, so optimizers may land on −α. Compare |α|.

- **Isochromat equivalence with EPG is exact** once isochromats outnumber the dephasing orders
  at the echo; below that, order k aliases to k mod K. (`tests/test_epg.py`)

- **Incidental MT is a large effect**: MTR 33 % (1.5 T) → 46 % (3 T) in white matter in a
  9-slice acquisition; every multi-slice simulation without a bound pool is biased.
  (`docs/REALISM.md` §3.2, ref 21)

- **The super-Lorentzian has no value on resonance; every simulator picks one** (MT,
  2026-10-06). ∫du/|3u²−1| diverges at the magic angle, so G(0) = ∞. Gloor et al. cut it at
  1 kHz (bSSFP qMT); BMCTool switches to a spline at |Δω| < w0 with w0 = γB0 in rad/µs
  *numerically*, i.e. exactly 1 ppm from the pool, and the spline and its outer formula do not
  meet there, so G jumps (a 6e-3 jump in Z right at 1 ppm from the pool). Near the pool, at
  the CEST offsets, the choice moves Z by up to 0.12. Normalisation varies too: BMCTool's G is
  π·g, with g of unit area over rad/s; qMT papers quote g. blochparty makes the cut
  per pool (`BoundPool.cutoff`, default 1 kHz: g there is 1.5e-5 s at T2b = 12 µs, the order of
  the 2.0e-5 s calibrated by Soellinger et al., ISMRM 2010).

- **BMCTool's super-Lorentzian integral is a 101-point rectangle rule** with both endpoints at
  full weight: 0.5–2 % above the exact integral between 1 and 10 kHz, and up to 2× at very
  large offsets, where the u = 1 endpoint dominates. Our table is Gauss–Legendre with nodes
  clustered at the magic angle, written as |3u²−1| = 3|u−u0|(u+u0) because the direct form
  rounds to 0 and gives 0/0 there; it matches SciPy's adaptive quadrature to 1e-11.

- **Strang splitting fails silently when the step is commensurate with the precession**
  (MT CW test, 2026-10-06). At 4 kHz off resonance a 250 µs RF step turns exactly 2π and a
  1 ms step 8π. Relaxation is then applied at the same rotation phase every step, so the
  error stops averaging out: a 1.7e-3 bias in the steady state that was *identical* for
  both steps. My "convergence check" compared two commensurate steps and looked converged.
  Pick non-round step sizes, or compare against an independent steady state. Real RF rasters
  (1–10 µs) are far below this regime.

- **Exact small linear systems need the increment form, or their gradients die in float32**
  (MT exchange step, 2026-10-06). z' = z_ss + e^{At}(z0 − z_ss) is exact, but for a 1 µs RF
  step e^{At} ≈ I, and the adjoint's (I − e^{At})ᵀλ is a difference of nearly equal numbers:
  float32 gradients 3 % wrong (float64 1e-10), checked against autograd through
  `torch.linalg.matrix_exp` of the augmented 3×3. Writing the step as z0 + (e^{At} − I)(z0 − z_ss),
  with e^{mt}cosh(qt) − 1 = ½[expm1((m+q)t) + expm1((m−q)t)], brings float32 to 1.5e-5. The
  first form also overflowed: e^{(m−q)t}·expm1(2qt) is 0·∞ once 2qt > 88 in float32
  (a few-second free run), so expm1 only while 2qt < 1, else the two decaying exponentials.
  My random test matrices also hid this at first: they allowed det A ≈ 0, which the physical
  model never has (det = r1(r1b + k + W) + kf(r1b + W) > 0), so build tests from physical
  parameters.

- **Piecewise-constant RF in a memory integral: integrate cell by cell** (generalized Bloch,
  2026-10-07). Sampling each cell's RF at its left grid point in the history integral made a
  second-order scheme first order (a half-step phase lag of an off-resonant pulse). The
  trapezoid over each cell with that cell's constant RF, G and z at both ends, restored it
  (error ratios 4.2/4.1/4.05 vs the Bloch ODE).

- **The rate model's on-resonance cut matters for imaging pulses, not only near CEST
  offsets.** The generalized Bloch model shows the 1 kHz-cut rate model under-saturating the
  bound pool even for 1–3 ms on-resonance π pulses (0.63 vs 0.46 at 1 ms), and it cannot
  invert it for pulses ≤ 100 µs. Incidental MT from excitation and refocusing pulses is
  therefore underestimated by every rate-model simulator, ours included until the
  linearized model is in the engines.

- **Reflecting walls: test the equilibrium, not only the signal** (restricted diffusion,
  2026-10-07). A radial mirror of the step's end point (r → r(2R − |r|)/|r|) is the obvious
  reflection for spheres and cylinders, and the signal looked nearly right (1 % high at large
  q). But it isn't measure-preserving: the walkers' equilibrium was 11 % depleted within
  one step of the wall (KS distance 2.5× the 5 % level). Specular reflection at the crossing
  point (mirror the rest of the step about the tangent plane) restores a uniform equilibrium,
  and the long-time signals then match to 3e-4. A Kolmogorov–Smirnov test on (r/R)^3 is a
  cheap, sharp check of any reflection rule.

- **A hard pulse in one segment misses relaxation during the pulse** (benchmark 4, 2026-10-07):
  relaxation is Strang-split around the whole rotation, so a 180 deg in one step spends no time
  transverse: Mz after a 912 us pi pulse with T2 = 75 ms is −0.9987 instead of −0.9933
  (~T_RF / (2 T2) lost). Sampled pulses (Pulseq, sinc) take small steps and are fine; give
  `SequenceBuilder.rf` ~10+ `n_steps` when T_RF is not short against T2. Since 2026-10-07 the default
  cuts hard pulses (builder, `block_pulse`, Pulseq block pulses) into <= 20 us segments: by
  *duration*, not rotation — a count that depends on a differentiable flip angle jumps under
  gradcheck's perturbation (a 30 deg pulse becomes 4 steps at 30.000001). Tests and examples
  that compare against instantaneous-pulse references (EPG, Ernst, the signal equation) now
  ask for `n_steps=1` explicitly. Found because MnCl2
  fitted 0.84 % instead of the authors' 0.70 %; with 50 steps it matches to every digit.

- **The generalized-Bloch Green's-function taper is visible in real data**: memory 100 T2b
  (taper 0.50 at 70 T2b, the longest pulse) costs 4 % chi^2 and −3 % in m0s on BSA; 300 halves it.
  The mode fit's slowest pole need not sit at 1 / memory: at memory 300, 1.5 / memory fits
  4x better (3.8e-5); `modes` now picks the layout per memory. Comparing an engine against its own reference solver hides this (they share the taper); only
  the untapered kernel on real data shows it. The reference solver itself needs h <= T2b / 16 and
  >= 64 steps: with 15 steps for a 22.8 us pi pulse its Heun free-pool rotation was 3e-3 off, and
  that error happened to *lower* the residual — a converged reference first, then conclusions.

- **The rate and generalized bound-pool models differ at CEST offsets by a lineshape convention,
  not by dynamics** (2026-10-09). On WM-like tissue with two free pools, 36 × 50 ms pulses at 2 µT
  gave Z 1.2e-2 apart at ±3.5 ppm, yet 5e-6 apart with a Lorentzian pool: the rate model holds the
  super-Lorentzian constant inside its 1 kHz cut, the generalized model's tapered Green's function
  keeps rising toward resonance. The generalized model's lineshape is analytic: a Green's function
  of modes Σ c_j e^{−p_j s/T2b} has the half-line transform T2b Re Σ c_j / (p_j + i x), a sum of
  Lorentzians (`exchange.modes_lineshape`, `BoundPool(cutoff=None)`), and with it the two models
  agree to 5e-5 on saturation pulses. Checked against quadrature of the tapered Green's function
  (1.3e-4 to 10 kHz): the super-Lorentzian's own transform ∫₀^∞ exp(−x²s²/8) cos(kx) dx =
  √(2π)/s · exp(−2k²/s²) is exactly the table's integrand, so the two normalisations agree by
  construction. Which near-resonance convention is physical is still a model choice (BMCTool cuts
  at ~150 Hz, Gloor at 1 kHz, the taper is Assländer's homogeneous broadening); real CEST data
  with MT decide it, not this.

- **The rate model needs pulses long against the Green's function's memory, not just against
  T2b, when played on resonance** (2026-10-09). The taper memory is 300 T2b = 3 ms at T2b 10 µs:
  60 hard 90° pulses of 0.5 to 2 ms stay 2 to 8e-3 from the generalized model with either
  lineshape, 0.2 ms / 60° pulses 1.4e-3 (cut) or 2.3e-2 (no cut), 0.1 ms / 30° 4e-2 or 1e-1. The
  1 kHz cut happens to approximate the short-pulse rotation better than the true long-pulse
  lineshape does, which is why the cut-free lineshape is for saturation pulses only. Off
  resonance by Δ the relevant time is 1 / Δ instead, so 50 ms CEST pulses at 447 Hz are safely
  in the rate regime (ROADMAP M1c; `tests/test_pools.py`).

- **A gradcheck step of 1e-6 on a parameter of 1e-5 is a 10 % perturbation**: check derivatives
  w.r.t. T2b through a unit-scale multiplier (`t2 = 1e-5 * s`), or the mismatch looks like a bug.

- **Judge a float32 design optimum by its objective, not its parameters, when the objective is
  flat** (DESPOT1 through the Metal adjoint, 2026-10-09): the CRB-optimal pair moved by 0.0003° to
  0.03° as the finite-difference step went from 2e-2 to 1e-1, while the closed-form bound at every
  pair found stayed within 3e-6 of its minimum. The Jacobian inside the bound is a difference of
  float32 signals: central differences need steps ~1e-2 (rounding ~1e-6 / step) and then carry
  h² ≈ 1e-4 of truncation, which moved the pair 0.045°; the fourth-order stencil (`design.jacobian(
  method="fd4")`) removes that. Constrained designs are stricter: the augmented Lagrangian found
  the flip through the float32 adjoint (0.003°) but held the scan-time constraint only to 8e-5.

- **Float32 gradient accuracy depends on how many RF segments feed it; measure the core in
  float32 before setting a tolerance** (`tests/test_integration.py`): over 32000 slice-selective
  sinc segments the flip-angle gradient through the Metal adjoint was 8.9e-4 off the float64 core,
  which looked like a bug until the core itself in float32 came out 5.4e-3 off.

- **A ROI mean of a uniform sphere is one column of spins at k = 0** (M3, `nist_twin.py`): spins
  across the slab and one pixel along the readout, at the ROI centre, simulated only through the TR
  that acquires the k-space centre, reproduce the full imaging simulation's ROI curves to 0.5-1.4 %
  at 1/16 of the spins and 1/16-1/25 of the segments, so gradients cost seconds. A 3 x 3 grid of
  columns at half the slab resolution was worse (1-3 %): the slab sampling (crusher dephasing)
  matters more than in-plane coverage.

- **Some scanner imperfections are invisible to magnitude ROI means** (M3 scans): per-axis gradient
  delays of 10 us change nothing on y and z (a readout delay is a k-space shift, a linear phase;
  the k = 0 line has no phase encoding; a crusher pair shifts together) and < 0.1 % on x; a longer
  first refocusing interval does nothing without off-resonance and < 0.1 % with 50 Hz. Scan before
  adding a parameter to a twin.

- **B1+ fits have two branches, and a spatial model removes the asymmetry that seemed to pick one**:
  inversion and refocusing depend on |B1+ - 1| to second order, so the gradient vanishes at 1 (start
  fits at 0.9 or 1.1). A global scan of benchmark 1 looked asymmetric (0.9: 2.2 %, 1.1: 3.7 %), but
  the five-parameter field fits it equally from either side (0.50 %).

- **A field fitted to echo shapes can be absorbing something else** (M3): the TSE's own B1+ fit
  improved it from 2.26 to 1.6 % but predicted the IR-SE plane at 14-19 % (nominal 1.05 %), and a
  refocusing-only amplitude factor on top of the IR-SE-fitted field converged to 1.003. Five smooth
  parameters fit sphere-specific residuals of ten spheres. Fit inputs on the data that constrain
  them physically and judge them on held-out data, never on the fit.

- **The flip angle of a slice-selective pulse off the slice centre is not |sum s dt|**: its phase ramp
  (f = G z0) makes the sum cancel (0.05 deg for a 180 at -12 mm). Demodulate by the pulse's own
  frequency first (`pulseq.flip_angle`: the median phase step, robust to the pi steps of a sinc).

- **A sparse column of test spins keeps coherences a voxel would lose**: spins 0.25 mm apart along
  z let residual transverse magnetization survive the spoilers, and a model that changes only its
  phase looked 5 % wrong; 0.05 mm spacing gave 1.8 %, and the realistic geometry 0.7 %. Validate
  approximations of other slices' or other times' pulses with voxel-dense spins.

- **Fewer spins: the sampling error was not crusher aliasing** (PERFORMANCE 2.6, 2026-10-10):
  against 27 spins per tissue and voxel (seed-to-seed RMS 0.0073) on the realism TSE, systematic
  tissue assignment matched categorical draws (8 spins: 0.0139 vs 0.0138; 4: 0.0206 vs 0.0216)
  and unjittered z-lattices matched random spins at equal count (2: 0.0301, 4: 0.0200, 8 on a
  2x2x2 lattice: 0.0147). The predicted 4x from lattice quadrature along the crusher axis did not
  appear; the error scales as 1/sqrt(K) whatever the placement.

## Monte Carlo estimator audit — 2026-10-08

- **Debiased power is not unbiased magnitude.** `nist.mc_magnitude` correctly subtracts
  estimated variance from squared complex-mean amplitude, then clips and takes a square root.
  The latter operations introduce bias. In 200,000 four-seed Gaussian experiments with
  mean-component SD 1, a true zero becomes 0.524 on average; true amplitude 2 becomes 1.696.
  Retain complex means and unclipped power and calibrate the downstream estimator at nulls.
  Derivation and reproduction: [MONTE_CARLO_REVIEW.md](MONTE_CARLO_REVIEW.md),
  `validation/monte_carlo_audit.py`.

- **Jackknife the statistic that is reported.** Both NIST drivers combine marginal sphere
  SEs as though their reconstructed errors were independent. Shared water and reconstruction
  can correlate ROIs. Jackknife the mean sphere error directly, and pair seeds for model
  differences. A perfectly correlated ten-output check exposes a sqrt(10) underestimate;
  on Benchmark 1 the historical bar was 1.65x too small (next entry).

- **Repeated seeds cannot detect deterministic quantile bias.** For a uniform T2′ voxel,
  permutations preserve the same frequency set. At 256 spins over 0–3 T2′, the audit found
  maximum FID error 0.03818 with seed SD only 2.08e-16. Increase quadrature density separately
  or randomize within frequency-CDF strata when replicate-based error estimates are needed.

- **A real-valued shift is not a Sobol digital shift.** Modulo-one shifts preserve uniform
  marginals but lose exact binary-cell balance. In analytic cube/ball Fourier checks, removing
  the extra shift helped one integrand and slightly hurt the other. Benchmark alternative
  randomizations; do not infer an end-to-end gain from discrepancy alone.

- **Free-bridge phase can be averaged analytically.** Conditional on a free Brownian endpoint,
  its independent constant-gradient bridge multiplies transverse signal by
  exp(-(2*pi)^2 h^3 g^T D g / 12), with g in Hz/m. A single-segment Philox experiment confirmed
  1.35–2.81x lower complex variance over three phase variances. This is an estimator proposal,
  not an implemented engine optimization; RF, reflecting walls and nonlinear spatial fields
  require separate treatment.

- **An RMS error of Monte Carlo curves is biased upward by the MC noise, and error bars do not
  show it**: the NIST shape error (RMS residual after a fitted scale) fell 1.90 → 1.80 → 1.73 %
  as 2 → 4 → 8 seeds were averaged, at a jackknife SE of ~0.2. Compare model errors at equal
  MC noise, or use the jackknife bias correction (`nist.mc_errors`), which removes the 1/K term.
  Also: jackknife the final aggregate, not its parts (sqrt(Σ se²)/n underestimated by 1.65×),
  and 4-seed SEs are themselves noisy (0.31 vs 0.19 on disjoint seed sets). Source: Benchmark 1
  recheck, 2026-10-08.
