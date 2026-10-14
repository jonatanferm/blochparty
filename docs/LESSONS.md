# Lessons learned

A running notebook of things we found out the hard way: physics and numerics that bite, quirks
of formats, data and other tools, performance facts, and process. Add to it whenever something
costs more than a few minutes to figure out. Newest entries go at the top of each section; each
says where it came from so it can be re-checked.

## Physics and numerics

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
- **Metal walks absolute positions in float32.** A walker 5 mm from isocentre is resolved to
  ~3e-10 m, and so is its offset from the compartment's centre: per-spin magnetization of 5 µm
  cylinders agreed with the core to 1.1e-3, the signal to 1.4e-5; spins at 10 cm with a
  gradient nonlinearity gave 7.4e-5 on the signal. Static spins agree to 2e-5. Walking in offsets
  from the start position would remove most of it. Source: `docs/ENGINES.md` Agreement, 2026-10-08.
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
- **Metal kernels take at most 31 buffers** (indices 0–30). Pack the scalars into one
  `constant uint*` array per dispatch rather than one `constant uint&` each.

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

## Pulseq, data and reference values

- **Check what a reference table's columns refer to, not just the numbers.** Jiang et al.
  2016 (PMC5408299) Table 1 gives T1 of the NiCl2 spheres *and T2 of the MnCl2 spheres*;
  benchmark 1 first used those T2 values for the NiCl2 spheres. Nobody tabulates the cross
  terms (T2 of NiCl2, T1 of MnCl2): derive them from the 3 T relaxivities (Martin et al. 2023):
  R2 = R2_w + (r2/r1)(R1 − R1_w). The derived MnCl2 T1s (2454 → 85 ms) land in the published
  80–2480 ms range they weren't fitted to. (`validation/real/nist.py`)
- **NIST phantom planes are mirror images**: T1-plane spheres 1–10 run clockwise on the outer
  ring (−36° steps), T2-plane spheres counter-clockwise; spheres 11–14 on the inner ring
  continue next to sphere 10. Identify spheres by geometry and confirm by decay rates, not by
  a T2 fit: the authors' fitted T2 map is ~2 s almost everywhere on that plane.
- **Multi-echo spin echo with imperfect refocusing**: echo 2 is ~20 % brighter than echo 1 in
  long-T2 spheres (stimulated-echo pathways), which is why T2 fits commonly drop the first
  echoes. (`validation/real/nist_tse.py`)
- **"Discard the first N excitations" may mean "use the other slice".** The NIST TSE file is a
  two-slice acquisition (x = +12 mm in pass 1, −12 mm in pass 2); the authors' recon keeps
  pass 2. A phantom on the first slice found in the file is never excited in the imaged pass:
  near-zero signal whose leftovers looked like growing, mis-centred echoes (61 % error that no
  spin density fixed). Enumerate all slice positions (`nist.slice_positions`) and choose the
  imaged one explicitly; sweep a pulse across x per pass when in doubt.
- **When the simulation plateaus away from reality as spins increase, it's a model or setup
  error, not convergence.** Diagnose by stripping down (one TR, spins only at the slice
  centre: textbook CPMG decay is the expected answer). Core and fused agreeing proves the
  kernels, not the inputs.
- **Track k from the excitation centre, not the TR start**: integrating from the TR start
  showed a fake ±350 /m slice-axis residual at every echo and briefly pointed at the importer.
- **Pulseq files can contain dummy TRs without ADC**: the NIST TSE file plays 2 × (1 dummy +
  128 lines); split TRs by timing and count readout TRs separately.
- **MATLAB-saved arrays reshape with `order="F"`** to reproduce MATLAB's `reshape`.
- **A sphere-radius estimate from sphere/fill contrast fails when contrast is weak** (T2 plane:
  0 mm and 9–10 mm vs 6–8 mm on the T1 plane); reuse the better-conditioned plane's geometry.

- **Slices are positioned by RF frequency offset.** The NIST IR-SE slice is at x = +57 mm
  (15.2 kHz / 266.7 kHz/m); a phantom at x = 0 only sees far sidelobes (tiny, meaningless
  signal). Compute f/G from the `.seq` (`nist_irse.slice_position`). With blochparty's
  default "physical" reading, frequency f excites at +f/G; "mrzero" mirrors it.
- **Pulseq v1.2 files don't mark RF use**, so pypulseq's `calculate_kspace` treats the 180° as
  an excitation and puts spin-echo readouts off-centre (545–1050 /m instead of ±256).
  Don't rely on it for old files; the Bloch simulation doesn't need it. pypulseq 1.5 reads
  v1.2 files with default rasters (warnings only).
- **Real RF is sampled on a 1 µs raster: ~80 % of all segments.** Merging 4 RF samples
  (preserving integrals) is 3× faster and changes results ≤ 0.4 %; merging 10 changes them
  5 %. (`nist_irse.coarsen_rf`)
- **Two gyromagnetic constants exist**: CODATA 42.577478518 MHz/T (`constants.GAMMA_BAR`, use
  for physics) and pypulseq's 42.576 MHz/T (`PULSEQ_GAMMA_BAR`, only for hardware limits so
  gradient designs equal pypulseq's).
- **NIST phantom references**: T1 of the NiCl2 array at 3 T / 20 °C for version 1 (serial < 43)
  and 2 (≥ 43) are in the RRSG 2020 code (`rrsg2020/analysis`, `src/nist.py`); version 2 and
  T2 of the MnCl2 array are the phantom-manual values in Carr et al. 2021 (Table 2); 3 T
  relaxivities for the cross terms in Martin et al. 2023 (Table 1). All in
  `validation/real/nist.py`. The Tong et al. dataset records neither serial nor temperature;
  v2 fits it better (1.5 % vs 2.8 %).
- **Mendeley Data's API rejects Python's urllib** (403); `curl` works.
  (`validation/real/fetch_data.py`)

- **To test a reference value, give the nuisance its own free amplitude** (benchmark 2,
  2026-10-06). The short-T2 spheres' misfit mixed two things: water blurring in (a flat tail
  whose level depends on geometry we only know roughly) and the decay itself. Fitting
  real ≈ a·sim(T2) + b·e^(−TE/T2_water) per sphere removes the geometry and leaves the T2
  question: the reference values held (scaling T2 by 1.25 made 13 of 14 spheres worse), and
  the remaining 6.5–8.7 % on spheres 11–14 sits in the first echoes. Before blaming the
  reference values, also check that a measured tail isn't noise: σ from the air (Rayleigh
  mean/std ≈ 1.9) put these tails 7–12σ above the floor.

- **A block pulse without a frequency offset is one segment** in `read_pulseq` (shaped pulses
  on the RF raster are stepped per sample; constant pieces are not cut). Relaxation is then
  split around one rotation: harmless for a 100 µs hard pulse, but a 1 s CW saturation pulse
  came out 0.6 off in Mz against BMCTool. Opt-in `max_rf_step=` cuts it (defaults stay
  bit-identical).

- **Scholand IR-bSSFP data (Zenodo 7654462), facts not in any document** (benchmark 3,
  2026-10-07): BART trajectory `traj -x 384 -y 1 -s7 -G -t 1000 -D -r`: spoke n at n pi / (phi + 6)
  (tiny golden angle 7; `-D` does not double golden angles since BART v0.4); samples at
  (i - 192 + 1/2) / 2 base-grid units. Measured spokes sit at -d(theta) along the readout (the
  other sign triples the data residual). The receiver phase follows the alternating RF phase.
  The B1 map (304 x 192 asymmetric echo; k-space centre at sample 111 of 384) uses an 80 deg
  preparation (hard-coded in the authors' script).
- **The authors' two ROI sets are ordered differently**: the Cartesian set lists sphere 1 first,
  the radial set sphere 14 first, as their reference table does. Geometry cannot tell: the
  spheres form a near-symmetric ring, so a 123 deg rotation and a mirror both match the centres
  to 0.55 mm. The signals can: a 14 x 14 matrix of curve-vs-simulation residuals is diagonal
  only for the reversed order. First tried the mirror, and every result was nonsense.
- **Vendor preconditioned-TurboFLASH B1 maps read low in short-T1 compartments** (0.87 against a
  smooth 1.09 in a T1 = 0.13 s sphere): T1 recovery during the readout raises the prepared
  image. In a phantom, fit a smooth field to the long-T1 background (agrees with long-T1 spheres
  to 0.02).
- **A single-echo SE series with first TE 15 ms cannot measure T2 < 15 ms**, and the magnitude
  (RSS) fit then returns the noise floor: 735 ms for a 5 ms sphere in the published table.
  Combining coils with a complex reference and fitting the real part removes the floor (25 ± 11 ms:
  unbiased, but still uninformative).

- **Asslaender NMR data**: Magritek `data.2d` = 12 header bytes, int32 501 (complex), size1,
  size2, 1, 1, then complex float32, column-major; the authors trim 6 points at each end. Column 1
  is the *shortest* inversion time (3 ms), although their T2* comment says 5 s. TI carries a
  timing correction 12 x 22.8 us + (13 x 15.065 − 5) us and is counted from the pulse centre.

- **PTB spiral TSE (Zenodo 18749100)**: the ISMRMRD trajectory is in matrix-index units
  (+-80 for 160) and each readout spirals out from the k-space centre at sample 0 — the header's
  `center_sample` 45 is wrong. Each echo of a TR uses a different spiral rotation.
- **T1MES reference values are from a 1.4 T relaxometer** (Captur et al. 2016, Table 2); the
  shared tubes have no 3 T values, and agarose T2 depends on field and echo spacing (here 35–50 %
  below the references). Not an independent 3 T T2 reference.

- **Legacy simulator tissue values are not tissue values** (anatomy, 2026-10-07): BrainWeb's
  MRISIM file gives CSF T2 0.33 s (measured at 3 T: ~2 s) and T1 2.57 s (~4.4 s); used at 3 T it
  made T2-weighted CSF dark, the most visible error in a brain image. Flags marked it as an
  assumption, but nothing stopped it being the default — check defaults against images.
- **The T2* policy can silently drop a measured R2* map**: with susceptibility on, the default
  (`t2star="auto"`) omits R2' to avoid double counting, so AHEAD's measured R2* produced no T2*
  contrast at all until R2' was supplied explicitly. Look at a long-TE image before trusting it.
- **AHEAD v3 states no units**; ppm and 1/s follow from WM R1 ~0.88 /s, R2* ~ 36 /s at 7 T
  (Caan 2019) and a positive pallidum. PMC served a CAPTCHA for the methods paper: do not try to
  get around it; use the publisher record and the values.

- **A sum-to-one check at 16 eps fails after cumulative-sum resampling in float64**: resampling
  complementary fractions by differencing cumulative sums along a 362-voxel axis leaves ~1e-14
  per class, so BrainWeb at 1.5 mm in float64 raised "fractions sum to more than one" (float32
  hid it: its tolerance is 2e-6). Renormalise by max(total, 1) after resampling.

- **A measured T2* is not an intrinsic property once the simulation computes fields**: it
  includes the macroscopic dephasing at the measurement's voxel size and echo times. Separating it
  needs those acquisition parameters (AHEAD: MP2RAGEME, TE 3/11.5/19/27.5 ms); the round trip
  (simulate the acquisition, fit R2*, compare with the input) is the test, and a field strong
  enough to exceed R2* - R2 on its own means the inputs disagree (clip, do not error). Fixing it
  exposed the next issue at once: the occipital WM/GM T2* proxies make cortex dark at long TE.

- **Quasi-static coil checks cannot fix the rotation sense; Faraday's law can** (2026-10-07):
  for real (quasi-static) fields |Bx + iBy| = |Bx - iBy|, so magnitude maps, RSS images and a
  comparison against a full-wave solver at low field all pass with either sign, and a test along a
  single axis is phase-trivial. `loops.receive_sensitivities` returned conj(B1-) and so mirrored
  every coil's phase. A brute-force lab-frame Faraday calculation (precessing dipole, numerical
  -dPhi/dt, demodulation by e^{+i w0 t}) divided by the simulated coil x Mxy must be one constant
  across positions: 2 i w0 with (Bx - iBy)/2, a 137 % spread with the conjugate. With full-wave
  phasors (exp(+j w t)) the same derivation gives transmit conj(minus)/2, receive conj(plus)/2.

- **Full-wave map import** (2026-10-07, `em.py`, `validation/em/`): (1) mariepy leaves which
  circular component excites to the reader; for protons with B0 along +z it is `minus`
  (clockwise), conjugated into blochparty's rotating frame. A birdcage's two quadrature drives
  (1, +-j) differ by ~6x in mean |B1+| on a head (1.09 vs 0.17 uT per uT), so picking the wrong
  sense is obvious in magnitude, but a wrong conjugation is not: test against a rotating field
  with a non-zero phase. (2) The solver stores zeros outside the body; trilinear sampling at the
  skin then mixes in zero B1 / coil weight, so fill the outside with nearest-inside values. (3)
  Noise resistance from E sampled at voxel centres falls 2-3 % below mariepy's exact absorbed
  power (piecewise-linear basis) at 5 mm; rescale the matrix so its diagonal matches. (4) A
  registration check that survives near-symmetric maps: correlate the image ratio (B1 map on /
  off) with the sampled map and with its transpose and mirror (-0.995 vs -0.93 / -0.97).

- **Read the table, not the abstract** (2026-10-08): Shin et al. 2025's abstract gives "white
  matter and gray matter 55.9 ms and 51.5 ms"; Table 3 (columns GM, WM) and every per-subject row
  give GM > WM. Values for the table come from tables, cross-checked against a second source
  (Peters, ISMRM 2006, Table 1 in the PDF: GM 59.7 > WM 54.6 ms at 3 T). A tissue T2* also needs
  its source's voxel size and echo times: the macroscopic part it contains depends on them, and a
  longer echo train (3-66 vs 5-40 ms) clips R2' at gradients half as large.
- **When only the abstract is readable, test its numbers against physics before using them**
  (2026-10-10): Wansapura et al. 1999 (JMRI 9:531) gives T2 "for gray matter and white matter"
  as 80 and 110 ms, opposite to every other 3 T study (GM > WM, Stanisz 2005 99 / 69 ms), and
  also calls its 14-echo FLASH T2* values "T2" and an occipital GM value that is the lower one
  "significantly higher". `validation/anatomy_brainweb.py` used the abstract order since M2
  (2026-10-07); its human-3 T T2-weighted GM/WM ratio was 0.69 (cortex darker than white matter,
  which no T2-weighted image shows) and is 1.56 with GM 110 / WM 80 ms. A predicted contrast
  ratio that inverts the textbook appearance is the quick check. The Wiley full text sits
  behind a bot check and ResearchGate returns 403, so the assignment rests on physics; the
  tissue database (`tissuedb` branch) flags the same rows.

- **The antiparallel-spoke delay estimator is biased by unequal axis delays** (2026-10-08,
  benchmark 6): the partner at theta + pi is shifted along the spoke by the same d(theta) but
  sideways by the opposite amount ((dx - dy) sin cos), so the pair samples two parallel lines; an
  off-centre object then adds a phase tilt to their cross-power. Exact for dx = dy or a centred
  symmetric object; 0.4 us off at |dx - dy| = 5 us for an ellipse 2 cm off centre. Check an
  estimator on simulated data of an object like the measured one before trusting it, and vary
  the object (rotating it flipped the sign of the bias).
- **Modern vendor delay compensation leaves sub-microsecond residuals** (Cima.X: 0.2 / 0.65 us,
  < 0.07 samples at 10 us): a "gradient delay" benchmark on such data checks the timing model and
  the estimator, not a visible artefact.

- **A GIRF is defined in a sampling frame** (2026-10-08, benchmark 7): the thin-slice output is
  the mean gradient between ADC samples, the authors' input point samples of the waveform; the
  two frames are half a dwell apart, so delays (and the phase of H) differ by 2.5 us at 5 us
  dwell. State the frame with any GIRF or delay, and convert before applying a published one.
- **Thin-slice B0 terms are drift-limited**: the slice pair's common-mode phase picks up the drift
  left after reference correction (~10 Hz here) and slice-position asymmetry (a DC term of
  -8 to -14 Hz per mT/m from ~0.25 mm), so check a B0 eddy term with both blip polarities (only the
  odd part is eddy current) and as accumulated phase, not per-sample frequency (noise ~70 Hz).
- **Unit trap in `GIRF` B0 rows**: taps are Hz per (Hz/m) because blochparty gradients are Hz/m;
  a B0 transfer function measured in Hz per (mT/m) must be divided by gamma_bar (here that slip
  inflated the B0 term by 4e4 before it was caught).

- **Pulseq-CEST trains reset the RF phase per pulse** (2026-10-08, benchmarks 8-9): each pulse's
  phase advances by 2 pi f t_p (only while RF is on), so in the frame of the offset f, pulse k
  carries -2 pi f k t_d and, with no spoiling between pulses, that matters near water. Simulate
  each offset in its RF frame (rf * exp(+i 2 pi f t), spins at dB0 - f): exact, slow envelopes,
  100 us steps instead of 1 us. To batch all offsets as spins, move the per-pulse phase step into
  the gaps as a rotation (a gap gradient with spin positions x = dpsi / (2 pi G t_gap)): exact to
  1e-5, 58 offsets in 15 s.
- **Library phantom models are not ground truth**: the Pulseq-CEST L-arginine YAMLs (comments:
  "simplified", MT size "from comparison with measurement") miss their own phantom by rms
  0.05-0.12; the engine meanwhile matches BMCTool to 8e-4. Separate engine checks from model checks.
- **A .seq's ppm scale is its own**: definitions say B0 3 T while the Prisma fit / Vida run at
  2.894 T; offsets are played in Hz (ppm x 127.73), pool shifts must be placed in Hz at the real
  frequency (raw header lFrequency).

- **Prefer the raw data over a dataset's prepared arrays when the bookkeeping does not add up**
  (2026-10-08, benchmark 9): the iohexol archive's prepared Z-spectra had 57 columns for 58
  offsets, a zero normalisation column and the water minimum one column apart between files; a
  60-line twix reader and the .seq's own trajectory gave unambiguous spectra. MATLAB-saved maps
  and masks are column-major: their 2D arrays are the transpose of numpy images (registration
  found it; their masks then still did not match our tubes and were not used).
- **One saturation time does not pin T1 in a CEST fit**: calibrated at 1.5 s, the model predicted
  the 4 s spectra better than "no change" but under-predicted the build-up 1.3-6x; T1 needs an
  independent measurement (or several Tsat) before a Tsat prediction tests the engine.
- **Batch parameter sets as spins**: a Levenberg-Marquardt Jacobian by forward differences costs
  one simulation if every perturbed parameter set is a spin (here 8 tubes x 8 sets x 58 offsets);
  ~7 min per iteration on the CPU core with exchange pools.

- **Look-Locker subspace bases need more components than their singular values suggest**
  (2026-10-08): K = 4 (4th singular value 3 % of the 1st) left per-tube curve fits off by 4-13 %
  and T1 moving with the fit's start time; K = 6 gave 0.4-1.4 % and stable T1. Check a subspace
  reconstruction by the fit residual of the curves it produces, not by the spectrum alone. In a
  3D slab, use the central partition for flip-angle-corrected T1 (the edges see less flip).

- **pypulseq rounds on both sides** (pypulseq 1.5.0, found writing M4's exporter, 2026-10-09):
  its writer formats amplitudes as `{:12g}`, six significant digits (a 1034070 Hz/m gradient
  becomes 1.03407e+06, ~5e-7), and `Sequence.read` defaults to `remove_duplicates=True`, which
  rounds the event libraries before merging them (amplitudes to six digits, timings to 1 us).
  A lossless round trip needs both a full-precision writer and `remove_duplicates=False`. Every
  scanner file in `validation/real/data` (up to 1.1e8 segments) reads bit-identically either way.
- **Pulseq's `[GRADIENTS]` and `[TRAP]` share one id space**: the block table's gradient ids do not
  say which section they mean, so separate counters make a block pick up the wrong event
  (pypulseq then reports events longer than their block).
- **pypulseq's shape reader runs forever if the file ends right after the last shape sample**: its
  line reader returns -1 at end of file, and the sample loop stops only at a blank or comment
  line. Its own files always follow the shapes with `[SIGNATURE]`; write one (md5 of the content
  above it, the preceding newline belonging to the signature).
- **Values in us or ns do not always survive x / 1e-6 * 1e-6**: 60 * 1e-6 is not 6e-05. Writing the
  decimal that the reader's multiplication maps back to the same double (try x / unit, a few
  roundings and neighbours) makes a re-exported file bit-identical; segment edges computed from
  absolute times can still differ in the last bit (1.7e-18 s on a scanner file).

- **qMT papers name their exchange rates differently; convert before tabulating** (M5 tissue
  table, 2026-10-09). Our `BoundPool.k` and `TissueProperties.exchange_rate` are bound -> free.
  Yarnykh's k is free -> bound and his R = k(1-f)/f, with f = M0b/(M0b+M0f) (Yarnykh 2012, Eq. 2
  and 7), so his R is our k: whole-brain median 19.0 /s at 3 T. Henkelman/Sled-Pike's R with
  M0f = 1 is also our k (kf = R·F). Dortch's SIR kmf is bound -> free; its PSR is M0b/M0f, not a
  fraction (Li et al. 2014 muscle PSR 0.097 is f = 0.088). Fat fractions follow the same split:
  "pool size ratio" is F, "MPF"/"bound pool fraction" is f = F/(1+F).
- **Some qMT parameters are not identifiable in vivo**: in human muscle at 3 T the exchange rate
  did not converge in any of 10 subjects (Sinclair et al. 2010), so the muscle rate in the table
  is the brain's, flagged as assumed. Lipid ADCs from DW imaging rise with slice thickness
  (motion): 2.78e-5 at 20 mm, 1.83e-5 mm²/s at 2 mm (Weidlich, ISMRM 2018); take the smallest
  value as an upper bound.
- **Read numbers from the page source, not from a summary of it**: WebFetch's summaries of PMC
  tables dropped columns and invented "not detailed" for values that were there (Li et al. 2014
  ADC, Sinclair et al. 2010 per-subject table). `curl` the PMC HTML and print the `<table>`s and
  the MathML annotations instead.
- **A round TE puts a GRE half a dwell off the gradient raster** (M5 example, 2026-10-09):
  `gre_cartesian` measures TE to the centre of ADC sample n/2, (n/2 + 1/2) dwell after the ADC
  starts, so with a 10 us dwell TE 2.5 ms or 20 ms leaves a fill block 5 us off the 10 us raster
  and `write_pulseq` refuses it; TE 2.505 / 20.005 ms writes and passes `check_timing`. A
  scanner rounds TE itself; the builder does not (changing that would move existing sequences).
- **ismrmrd-python's header arrays ignore whole assignment** (ismrmrd 1.15, writing M5's dataset
  writer): `acq.read_dir = (1., 0., 0.)` leaves the ctypes array unchanged without an error; set
  `acq.read_dir[k]` element by element. Scalars (`acq.sample_time_us`, `acq.idx.kspace_encode_step_1`)
  assign normally. The header model (`ismrmrd.xsd`) requires `experimentalConditions`.

- **Other slices' pulses in a multi-slice simulation** (2026-10-10, `tse_2d(slice_order=...)`):
  a far slice's phase ramp (~28 kHz at 7 cm) aliases on a coarse raster. A first fix played the
  envelopes without the ramp and shifted them by `b0_offset = -G z_j`: 0.8 % off in one test but
  2-3 % off at the slab edges, where spins sit inside a neighbour's band and the shifted frame
  does not reproduce its coherent pathways. What works: the neighbours as their real pulses, and
  every farther pulse as one segment of the same RF energy that only the bound pool sees
  (`Sequence.bound_only`, `rf_freq` at G (z_j - z)): 0.71 % against every pulse exact, 3x a single
  slice on the GPU instead of 6x. 180 deg pulses still nutate water a few kHz outside their band,
  so "band-limited" holds for the far slices, not the neighbours.
- **Hiding a pulse from the water by a frame offset breaks float32** (2026-10-10): the first
  bound-only pulses put the water 1 MHz off resonance for the pulse (an integer number of turns,
  exact in float64). In float32 each 100 us pulse leaves ~1e-3 rad of phase (100 turns at 2^-24),
  harmless for WM (T2 69 ms forgets it) but CSF (T2 2 s) accumulated it over 16 slices' pulses:
  Metal vs core 0.45 % in signal and 2 % in Mz, 3 % in Mz on Shepp-Logan at TR 0.8 s. A flag the
  kernel reads (water precesses, the bound pool saturates) gives 6e-6 (CSF) and 7e-7 (WM). Express
  "this does not act on X" as structure, not as a large number that cancels.
- **A sparse column of test spins keeps coherences a voxel would lose**: spins 0.25 mm apart along
  z let residual transverse magnetization survive the spoilers, and a model that changes only its
  phase looked 5 % wrong; 0.05 mm spacing gave 1.8 %, and the realistic geometry 0.7 %. Validate
  approximations of other slices' or other times' pulses with voxel-dense spins.
- **Count segments by kind before optimising a sequence**: delays padded in pieces and gradient
  ramps of pulses that only matter through their energy made up half of a multi-slice TSE
  (42k -> 23k segments after merging delays and dropping those ramps).
- **Fewer spins: the sampling error was not crusher aliasing** (PERFORMANCE 2.6, 2026-10-10):
  against 27 spins per tissue and voxel (seed-to-seed RMS 0.0073) on the realism TSE, systematic
  tissue assignment matched categorical draws (8 spins: 0.0139 vs 0.0138; 4: 0.0206 vs 0.0216)
  and unjittered z-lattices matched random spins at equal count (2: 0.0301, 4: 0.0200, 8 on a
  2x2x2 lattice: 0.0147). The predicted 4x from lattice quadrature along the crusher axis did not
  appear; the error scales as 1/sqrt(K) whatever the placement.
- **fastMRI brain k-space is (slices, coils, readout x2, phase)** (realism score, 2026-10-10):
  the inverse FFT, then the central rows of the oversampled readout and the central columns of the
  phase direction (to the reconstruction matrix, 384 x 384 over 220 mm), reproduces the files'
  `reconstruction_rss` with correlation 1.0000000. The readout runs anterior-posterior (image
  rows), posterior is up (corrected 2026-10-12: the next entry). The 3 T AXT2 stacks (16 x 5 mm) run
  from the lateral ventricles (slice 0) to above the vertex (slice 15) in all 21 volumes.
- **fastMRI's brain images show the head posterior up; a mean-image correlation cannot tell** (2026-10-12):
  for two days the simulated heads were displayed anterior up against posterior-up real ones. The first
  check correlated mean images (r 0.94 as-is, 0.82 flipped), but an AP flip barely changes a mean oval;
  reading the anatomy by eye failed too (the flattened, dark-banded edge I took for the occiput on the
  pad is the forehead and frontal sinus). What settled it: a head-shape statistic that separated the
  two sets perfectly (the width of the real heads peaks 0.34-0.44 of their length from the top row, of
  IXI's 29 axial T2 heads, oriented by their NIfTI affines, 0.48-0.62 from the front; the values
  mirror each other), and an interior template from IXI (each slice cropped to its head's box,
  central ellipse only): real slices match it flipped in 91 % of 147. Check an orientation against
  data whose orientation is known, with a feature that is not symmetric (`validation/realism/orientation.py`).
- **Unbalanced person-wise folds bias a classifier against the held-out people**: a real-vs-real
  null test gave AUC 0.20 for a histogram logistic regression (CI excluding 0.5) when folds held
  out people without balancing classes; dealing each class's people round-robin into folds moved
  it toward 0.5 but not all the way (0.27). Calibrate small AUC differences with a permutation test.
- **Ask the classifier where it looks before tuning by eye** (2026-10-12): rerunning the realism CNN on images
  masked to bands of distance from the head's edge (the rest zeroed) said at once that the outer 6 px (scalp
  and skull) separated real from simulated (AUC 0.998) and the brain did not (0.79), after days spent on the
  cortex. Then a band's intensity quantiles track a fix in minutes, before a full run.
- **A segmentation label is a contrast claim** (2026-10-12): prepare_ixi called soft tissue deeper than 8 mm
  "muscle", and the simulator gave it muscle's T2 (30 ms, black in a T2 TSE), though the subject's own T2 read
  it at 0.5-0.8 x WM. Check what each label's tissue values predict for the source scan against the source
  scan itself (here: soft tissue unmixed between skin and muscle by its T2).
- **A cropped scan's k-space remembers its echo ordering** (2026-10-12): the realism score keeps the central 128 of
  the real scans' 396 phase-encode lines, so in a linear 16-echo TSE those lines come from ~5 echoes around the
  effective one; our 128-line simulation spread them over all 16 (TE 13-200 ms). Its k-space was lopsided (+ky / -ky
  0.6-0.8 at the edges, the real 0.95-1.01), the point spread complex along the phase axis, and the CNN lost most of
  its edge when both sets were low-passed across that axis. Simulate the acquired matrix's echo-to-line mapping
  (tse_2d(acquired_lines=)), and check a simulated k-space's symmetry against the real one's.
- **Count the resamplings**: an anatomy segmented from a scan resampled onto an atlas grid and then pulled back
  through the pose is interpolated twice; trilinear keeps ~0.6 of the power at an image's Nyquist each time (cubic
  B-spline ~0.9). Our heads had ~half the real fine-structure power and 10 % softer ventricle walls (2026-10-12).
- **An atlas can miss a structure that every real image shows**: neither BrainWeb nor the IXI segmentation has a
  falx (dura 0 at the midline), so the interhemispheric fissure was solid CSF, 20 % brighter at its centre than
  real (2026-10-12; anatomy.add_falx).
- **A segmentation inherits its source scan's artefacts** (2026-10-12): IXI's T2 displaces fat 1.63 mm posterior of the
  water tissue (a water-fat shift along its readout), so our IXI heads had fat at the skin and soft tissue against the
  skull at the back, the reverse of the real layering, and the realism CNN read it there. Check a segmented layer's
  order by direction around the head (fat's depth minus soft tissue's was -2.6 mm at the back, +0.8 at the front,
  -1.6 at the sides): an asymmetry along one axis is the scanner, not anatomy.
- **Measure the thing, not a proxy that mixes two layers** (2026-10-12): "skin to the darkest point of the skull"
  said our posterior scalp was 1.2 mm too thick and led to a compression model; skin to the skull's outer surface said
  the scalp was 1.6 mm too thin and the skull's dark band too wide.
- **A residual fitted per voxel carries the source images' noise, amplified** (2026-10-12): T2 from the ratio of two
  echoes (IXI's TE 8 / 100 ms) turns ~2 % image noise into ~4 % per voxel after 0.7 mm smoothing, and r25's first heads
  had 1.4-2.4 x the real fine texture in the hemispheres. Smooth the fitted map (1.5 mm: log SD 0.19 -> 0.05 on IXI012,
  the structure above 2 mm kept: same-person RMS 0.074 against 0.072), within each tissue so that boundaries the
  labels already carry stay sharp; check a fitted map's power spectrum against the real images' before simulating.
- **Fit a residual only where the model and the source images share geometry** (2026-10-12): outside the skull the
  phantom's fat sits where it belongs (the water-fat shift undone) but the subject's images still show it shifted, so
  the residual "corrected" the scalp layers against misplaced fat and dimmed the posterior scalp peak from 0.65 to
  0.43 of p99 (real 0.83). The residual is now fitted inside the skull only.
- **A correlation across a run's heads is not a parameter's effect** (2026-10-12): in r24 the posterior scalp-to-brain
  depth correlated with each head's drawn pitch (r -0.64, anterior +0.74) and suggested +8 deg instead of -10; the
  same 16 heads as proxy images at fixed pitches -10..+10 moved it by < 1 px (posterior 7.8-8.5, anterior 5.2-6.5).
  The subjects carried the variation. Sweep the parameter on the same heads before changing it.
- **Compare against the same anatomy version** (2026-10-12): r24's heads were v8 (IXI's fat still 1.63 mm posterior),
  the proxies v10 (shift undone); their 1.5 px posterior / anterior disagreement looked like a simulation effect.
  Simulating the v10 head with and without fat's chemical shift moved the edges by ~0.5 px (the expected 0.56 px:
  434 Hz over 128 samples at 10 us); v8 -> v10 moved the posterior dark minimum 5.0 -> 4.5 px (real 4.5).
- **Read the echo train off the real k-space before fitting one** (2026-10-12): in a linear-ordered TSE each echo fills
  a block of adjacent lines (one per shot) and the line energy steps between blocks, so the real AXT2 k-space shows its
  train directly: steps every 19 lines, 21 echoes, ky = 0 in echo 11, at the header's own echo spacing (10.26 ms;
  validation/realism/echo_train.py). Our fitted 16-echo train at TE / 9 = 12.6 ms had bigger steps every 25 lines,
  whose replicas of the head along the phase axis left a plateau of ~1 noise floor beside it (0.87 at 9 px; real
  train 0.28): the background there read 1.19-1.22 x the floor in every run since r21 against the real 1.09, and
  1.03 with the real train. The header had the spacing all along.
- **A residual absorbs whatever its forward model leaves out** (2026-10-12): the PD residual predicted IXI's TE 8 ms
  echo without magnetization transfer, so the subject's own multi-slice MT (white matter darkened most, CSF not at
  all) came out as "PD": CSF x1.21-1.28 and grey matter x1.05-1.09 in every subject, and the simulator added its own MT
  on top (r26: CSF p99 1.28 x r24's). A multiplier that is the same in every subject is the model's gap, not anatomy:
  keep only the within-tissue pattern of a quantity whose forward model is incomplete (re-centred per tissue), or model
  the missing mechanism. T2 from an echo ratio cancels MT and kept its tissue offsets.
- **Partial-volume unmixing needs a pure reference where it unmixes** (2026-10-13): IXI's CSF / grey matter shares were
  unmixed against CSF's intensity from the cores of CSF regions, which near the convexity means the ventricles' (no
  pure CSF within 10 mm). 1 mm sulcal voxels are never pure, so the labels carried 30-50 % more CSF-like voxels than the
  subject's own T2 at MNI z 30-70, and the simulated top slices too much CSF (6.38 x WM vs 5.25) and too little white
  matter. Check a segmentation's fractions against its source image by region (height, depth) before simulating.
- **zsh does not split `$var` into words**: `pytest $files` with a space-separated list ran
  nothing ("no tests ran"); use `${=files}` or list the files.

- **BrainWeb's heads are MNI-sized** (2026-10-10): the 20 models live in stereotaxic space, so every
  head is template-sized and template-shaped: subject 04 is 174 x 207 mm (breadth x length; ANSUR II
  men 154 +- 5.5 x 200 +- 7.0 mm), its brain 1580 cm^3, and simulated heads at the ventricles were
  264 cm^2 against 189 cm^2 in fastMRI, rounder (AP/LR 1.14 vs 1.31) and all upright. Scale each
  head to an anthropometric draw (``sample_head(size="anthropometric")``) and turn it
  (``yaw_sd``); centre the field of view on it as an operator would.
- **Sub-voxel spins make a grain through a 2D slice's profile** (2026-10-10): with the tissue
  composition exact (``sampling="proportional"``; categorical draws add a salt-and-pepper of their
  own), noiseless images still had a pixel-scale grain. More spins in-plane (1x3x3, 1x4x4) did not
  help; more along z did (3x3x3), because random z positions through the steep 5 mm slice profile
  make every pixel's slice integral noisy. z at subcell centres (``jitter="inplane"``, a quadrature)
  fixes it; a full lattice instead makes horizontal banding. 4x2x2 in-plane-jittered is as smooth
  as 3x3x3 at a third of the time.
- **Image statistics can mislead where the eye does not** (2026-10-10): a high-pass "grain" metric
  said the real scans had more fine structure (0.26 vs 0.17), while the simulated ones looked
  grainier. Both were right: real images have crisp CSF lines, simulated ones had sampling grain
  and few sulcal lines. Look first, then build the number that measures what you saw.
- **Where the real slice stack sits** (2026-10-10): fastMRI ships 16 of the protocol's 34 slices
  without positions; the real slice 0 shows frontal horns, septum, caudate heads and thalami
  (foramen of Monro), ~10 mm below the lateral ventricles' largest cross-section. The head-area
  profile alone cannot separate offset from spacing (a valley of equal fits); anatomy picks the
  offset, and then contiguous 5 mm slices fit.
- **fastMRI headers hold more protocol than the manifest kept** (2026-10-11): `reconSpace` z is
  the slice thickness (5 mm; `encodedSpace` z says 7.5, a padded value, not a gap or a thickness),
  `sequenceParameters` has `echo_spacing` (10.26 ms at TE 113, so TE is echo 11; we played 12.6 ms
  and echo 9 of 16), and `flipAngle_deg` is the refocusing angle (180 in 11 of the 21 3 T AXT2,
  140-179 in the rest). No echo-train length anywhere.
- **Measure layers aligned at a landmark, not at a fixed depth** (2026-10-11): a band 5-7 px below
  the head edge said the simulated calvarium was too dark (0.26 vs 0.35-0.41), and BrainWeb's
  marrow became yellow (fatty) marrow. Aligned at each ray's bone minimum, the same images say the
  opposite (bone 0.27-0.29 vs real 0.16-0.21): the scalp and skull thicknesses differ, so a fixed
  depth samples different layers in real and simulated heads. 0.57 mm reconstructions of six real
  heads show a uniformly dark frontal bone, no bright diploe; the mapping went back to the
  unspecified (trabecular) marrow.
- **p99 normalisation couples the scalp to the brain** (2026-10-11): in the real scans the scalp's
  top decile (0.89) is as bright as ventricular CSF (0.87); in ours the scalp was 0.79 and CSF
  1.02, so CSF set the scale alone and the whole brain came out ~13 % brighter. A dim scalp makes
  a bright brain.
- **The bright rim under the skull is the tell** (2026-10-11): blinded r5 12/12 again, picked by a
  continuous bright CSF rim just inside the skull and speckled bright cortex. Aligned at the bone,
  real and simulated profiles agree from 5 px in; the excess is the first 2-4 px (0.62-0.64 vs
  0.39-0.44 frontal). Not the receive ring (12 cm: rim unchanged, white matter periphery worse),
  not grey-matter T2 (85 ms: no change), not marrow. Thinning BrainWeb's subarachnoid layer by
  1.5-2 mm matches, but removes 16-25 % of the CSF, while BrainWeb's total (245 mL after resizing,
  18.5 % of ICV) is already at or below young adults' (Yamada 2023: 287 mL; older 389 mL). Open.
- **BrainWeb labels the deep nuclei grey matter, and much of the pallidum and thalamus white**
  (2026-10-11): with one grey-matter T2 (96.5 ms) the basal ganglia shone like cortex; at 3 T iron
  puts them near white matter (Krug 2022: 57-76 ms). Labelled by MNI regions: caudate 3.3,
  putamen 4.9 mL per side; thalamus 3.3, pallidum 0.3 mL (the rest stays white matter).
- **A morphological opening can eat the real scalp** (2026-10-11): `lab.head_mask` opens the
  thresholded image twice; in real scans the thin, dim lateral scalp at the low slices does not
  survive it, so real heads measured 143 mm left-right at slice 0 (opened once: 157) and their area
  looked flat for 35 mm (it falls like ours, only slower). The simulated scalp survived, so the
  same function biased real against simulated. Check a mask on the images it is used on.
- **Routine axial slices are tilted to BrainWeb's AC-PC frame** (2026-10-11): sampled heads lost
  area faster with height than the real stacks (slice 8: 0.73 vs 0.79 of slice 0; slice 15: 0.15 vs
  0.21). A vertical stretch filled the top too much; tipping the head 10 deg nose up matched all
  of slices 5, 8, 12 and 15 (0.89 / 0.77 / 0.50 / 0.21 against 0.94 / 0.79 / 0.48 / 0.21).
- **BrainWeb's CSF sits in the wrong place for T2w realism** (2026-10-11): at the vertex levels most
  sulci are closed (two grey banks touching, no CSF) and a continuous CSF rim wraps the brain, at
  0.5 mm as at 1 mm; real 3 T scans show a thin outer layer and long sulcal lines. Its 1.5 T source
  scans could not resolve narrow sulcal CSF, so it went to grey matter. Thinning the rim (a
  calibration) removes one tell and exposes the other: the broken sulcal CSF reads as speckle (2 mm:
  10.3 components per 1000 px of 17 px; real 7.2 of 31). Opening the closed sulci failed twice: on
  the ridge of the white-matter distance (salt along the sulci, at 0.5 mm too) and on a Hessian
  sheet filter (2-3 mm blobs, 62 mL in a 50 mm block); and a third time as "grey matter >= 1.8 mm
  from white matter and >= 1.5 mm from CSF", which also hollows grey matter that is only thick
  where cut tangentially (53 mL; busier cortex at 1.72 mm). Needs surface-based sulci (pial
  surfaces) or another anatomy source. One lead: BrainWeb did keep a trace of the closed sulci.
  At 0.5 mm, grey matter dips to ~0.8-0.9 along their midlines, where CSF is 0.02-0.15 (90th
  percentile inside GM 0.08), in continuous lines. But ridges of that trace are as noisy as
  the others (dotted along nearly all grey matter at a 0.03 threshold), and concentrating it
  locally (power 3, 1 mm window) moves too little mass to show. A Hessian sheetness filter on it
  (0.75 mm, non-maximum suppression along the normal) finds thin, plausible pieces, but sparse
  (3-13 mL in a 50 mm block) and still dashed; at 1.72 mm the change is barely visible.
- **Closed sulci are the lambda-medial axis of the non-white region, not the ridge of its distance**
  (2026-10-11): ridges of the distance to white matter fire on every bump of the white surface
  (salt), Hessian sheets on blobs. The feature transform (each voxel's nearest white point) gives a
  robust test instead: two neighbouring voxels whose nearest white points lie >= 3 mm apart and face
  each other (within 60 deg of opposite) straddle a closed sulcus; 0.5 mm bumps make none. On
  BrainWeb 04 at 0.5 mm it draws continuous sulcal lines like real 3 T T2 (12 s). Two traps: the
  voxels beside the sheet must take their midplane distance from the two banks (distance to the
  sheet plus the sheet voxel's own offset under-filled a 30 deg sheet by 16 %); and with only "more
  than 90 deg apart" a 3 mm deep, 6 mm wide groove in the white surface opened too (and BrainWeb got
  173 instead of 137 mL at 1.3 mm). Widths: Jin et al. 2018 (PATH, n ~800) 1.27 mm at 47 y, 1.49 at
  67; Kochunov 2005 with the same definition has a six times steeper slope (0.71 mm per decade); the
  added CSF brings BrainWeb 04's 254 mL to 322 mL at 25 y and 381 mL at 75 y (Yamada 2023: 287 / 389).
- **Measured sulcal widths over-brighten fastMRI T2w; conserve BrainWeb's CSF instead** (2026-10-11):
  on six matched heads (slices 1-7), brain-interior intensity quantiles over the 15th (~white
  matter) at slices 1-4: real q70 / q85 1.71 / 2.51, unopened 1.73 / 2.55, Jin widths x 0.6 1.91 /
  2.83, x 1.0 2.07 / 3.22 (slices 5-7: real 1.79 / 2.67; unopened 1.70 / 2.27). The real images hold
  about BrainWeb's amount of bright CSF, only in lines instead of a rim and diffuse speckle: 3.9 % CSF
  spread through grey-dominant voxels (38 mL, subject 04) lifts T2w grey matter ~9 %. Gathering each
  grey/CSF mixture to its larger share and opening the sheets to the width that keeps the CSF volume
  (plus Jin's widening, 0.011 mm per year beyond 30) gives 1.67 / 2.42 and 1.78 / 2.36. Jin's widths
  are for five major sulci measured on 1 mm T1 (BrainVISA), not every sheet a medial axis finds.
  Opening also brings BrainWeb's grey/white volume (1.44, subject 04; 20-subject mean 1.63) to
  1.23 and its pial-to-white distance from 2.92 to 2.35 mm (adult cortex ~2.5).
- **Segmenting IXI heads: what the EM gets wrong** (2026-10-11): with BrainWeb priors aligned by an affine
  only, a 12-class EM on (log T1, log T2) places the brain well (the subject's intensities decide when
  the CSF/GM/WM priors are shared at 8 mm) but swaps class identities in the thin head layers ("marrow"
  came out dark, "skin" took the scalp fat): rules on bias-corrected intensities relative to the
  subject's white matter do better there. IXI's T1 (3D, sagittal) covers the head, its T2 (axial,
  112 x 1.2 mm) stops ~10 mm short of the vertex. Some scans carry skin markers that a 3 mm closing
  welds to the scalp; above the ears an 8 mm disc opening per axial section removes them. One HTTP
  stream from the IXI server ran at 0.65 MB/s, eight byte ranges at ~8 MB/s together.
- **The MNI frame is taller than real heads** (2026-10-11): IXI subjects registered to BrainWeb's MNI
  frame scale by ~0.85 vertically but ~0.9-1.0 across, so BrainWeb heads sized from ANSUR breadth and
  length (height as their geometric mean) stay tall, and parameters fitted on them (the -10 deg pitch,
  the 10 mm stack offset) absorbed that. IXI heads at their own size narrow a slice faster than the
  real stacks under the same settings.
- **A flagged row should not set a default** (2026-10-11): the 3 T grey-matter T2 (96.5 ms) was the median of
  Lu 2005's CPMG (83 ms) and Wansapura 1999's abstract (110 ms, its GM/WM labels contradicting physics and
  swapped by the curator, flagged). Unresolved flagged rows now stay in the table but do not pool: grey 83,
  white 72 ms (Lu 2005; Krug 2022). On six IXI heads with partial-volume fractions the brain-interior
  quantiles at slices 1-3 went from 1.42 / 1.93 / 2.51 to 1.37 / 1.84 / 2.49 (real 1.31 / 1.74 / 2.58).
- **EM posteriors are not partial volumes** (2026-10-11): a mixture model's posterior is close to a step in
  intensity, so a boundary voxel a little brighter than grey matter reads as mostly CSF; in IXI012's grey/CSF
  boundary voxels the posteriors hold 103 mL of CSF, linear unmixing of the T2 52 mL. Unmixing three classes
  from two channels and the sum is exactly determined and turns noise into spurious white matter (GM/WM
  1.48 -> 0.90); pairwise unmixing needs clean references (sulcal "pure" CSF is partial volume; ventricle
  cores are not) and local ones (residual receive bias). Without an independent CSF volume to check
  against, the posteriors stayed; the simulated images' high slices still carry too bright a tail.
- **zsh does not split words** (2026-10-11): `cmd $args` with args="--a 1 --b 2" passes one argument in zsh
  (the session's shell), so argparse rejected every run of a loop whose output went to /dev/null and three
  variant folders came out empty; use `${=args}`, and keep the logs.
- **Read the score's controls before its AUC** (2026-10-11): r16 (10 heads from 8 BrainWeb subjects,
  two heads lost mid-run on the swapping mini) scored real-vs-simulated CNN AUC 0.07 [0, 0.19] and
  histogram 0.06: inverted, not a success. Its simulated-vs-simulated control (4 subjects against 4)
  gave 0.008 where 0.5 is expected; person-split cross-validation over a handful of people anti-learns
  (each held-out person sits closer to the other class's training people). Score runs need ~15+
  distinct subjects, and a control far from 0.5 voids the headline number.
- **Sub-voxel spins leave a Monte-Carlo grain a CNN reads** (2026-10-11): random in-plane positions
  (and random subcells per tissue) are a point sampling of the anatomy; through the 1.72 mm PSF they
  add pixel-scale texture that falls as 1 / sqrt(spins): flat-brain Immerkaer sigma 0.026 at 5.4 M
  spins per slice, 0.0136 at 21 M, the anatomy itself 0.011, real 0.014. A CNN on high-pass
  residuals alone told real from simulated at AUC 0.985. Regular positions band (aliasing), a
  shared per-voxel shift blotches. Count the grain before trusting a texture comparison.
- **Open: why the real scalp is as bright as CSF** (2026-10-11): in the fastMRI 3 T T2 scans the
  scalp's top decile is 0.93 of ventricular CSF; the tissue values give fat / CSF ~0.63 at TR 6.5 s
  (fat T2 148 ms from CPMG, CSF partly saturated), and ours measure 0.68-0.73. A receive ring at
  the coil's surface (12.5 cm, 1/r) brightens the scalp but also the cortex (GM/WM 1.73 vs real
  1.50), and the full-wave 8-loop map makes the periphery 1.4-2x brighter than the real images
  allow; a blinded test went back to 12/12 with it. Candidates left: a steeper near field than 1/r
  with 3 T central brightening, fat's effective T2 in a 10 ms-spaced train (one 3 T study reports
  371 ms), ventricular CSF dimmed by pulsation.
- **Grey matter T2 at 3 T rests on two numbers** (2026-10-11): the database pools Lu 2005 (CPMG, 83
  ms) and Wansapura 1999 (110 ms, its abstract's GM/WM labels swapped by the curator, flagged).
  A 32-echo 2D train inflates apparent T2 through stimulated echoes, and our engine plays those
  echoes itself, so it needs the intrinsic T2; the real images' GM/WM (1.50 against our 1.58)
  lean the same way. Needs sources robust to stimulated echoes (EMC, nonselective CPMG,
  single-echo series). Leads not yet in the database: Utkur 2025 (MRM, 1 volunteer: 3D GRASE GM
  75.7, WM 59.5 ms); a multi-slice CPMG study with stimulated-echo correction reporting cortical GM
  79-88 ms by lobe (source not traced).
- **A resized grid's margin is empty, not air** (2026-10-11): pull-back resampling fills points
  outside the source grid with zeros in every class, background included, so `1 - background`
  calls the margin tissue and `fill_holes` then swallows the air around the head. Use
  "sum of tissue classes > 0.5" (`_head_mask`).

- **A long simulate_heads run grows** (2026-10-11): one process over many realistic heads reached
  15 GB on the 16 GB mini (allocator caches across heads), which swapped with another user's 5 GB
  app and slowed slices 3-5x. It now frees memory between heads; for long batches, one process per
  head also works.

## Other simulators

- **MRzero**: RF is instantaneous; numerics are float32 (limits agreement to ~1e-5–4e-4 on
  long sequences, mostly from merging states whose k agrees only to float32); default PDG
  pruning thresholds discard most signal of point-isochromat phantoms, and its pre-pass with
  mean T1/T2 prunes long-lived CSF states; ADC frequency offsets are ignored; it tips +z to −y
  (constant −i relative to us). (`validation/RESULTS.md`)
- **KomaMRI (from source, 2026)**: ideal linear gradients only, uniform transmit, no noise
  model; diffusion "under construction", susceptibility commented out; strong on motion,
  flow and speed. Its README roadmap is out of date.
- **JEMRIS**: the hardware-realism leader on CPU, but its GPU port drops diffusion, exchange,
  pTx, MT, concomitant fields, nonlinear gradients and eddy currents.

- **BMCTool / pulseq-cest (1.0.3, read before running)**: per RF sample, a Padé matrix
  exponential of the full Bloch–McConnell matrix (exact for piecewise-constant RF). MT pool:
  Mz only, exchange on z only, `k` from the pool to water (k_ab = f·k), W = ω1²·G with
  G = π·g. Its sequence semantics are not an imaging simulator's: any block with only a z
  gradient is an ideal spoiler, an ADC block records Mz (and by default resets to
  equilibrium), gradients otherwise do nothing, and a frequency offset rotates the frame for the
  pulse's duration, with phase bookkeeping (`accum_phase`) between pulses. Shaped pulses are
  downsampled to `max_pulse_samples` (500 by default). With the Lorentzian lineshape it
  agrees with blochparty to 6e-7 in Z (`validation/compare_bmctool.py`).

## Performance

- **Metal adjoint: memory traffic and occupancy, not arithmetic.** On 3d-48 the first backward took
  4.2 s against a 0.49 s forward. Storing the state before every segment for the reverse sweep wrote
  and read 12 B per spin and segment (~1 s); storing every 4th and recomputing blocks into
  threadgroup memory cut that 4x. Register arrays for 8 coils' sensitivities cost more occupancy than
  the reads they saved (1.9 s against 1.4 s), and the compiler deletes unused arrays, so a variant
  with them only when coil gradients are asked for helped. A per-segment threadgroup reduction with
  two barriers cost ~0.75 s: one simd_sum partial per simdgroup, summed later in order, has none.
  Uploading a small parameter tensor per dispatch costs ~0.3 ms (300 intervals: 0.1 s); one upload
  and a view per dispatch is free (`compile_shader` binds views at their offset). Source: M2.
- **Benchmark only against something measured in the same session.** After hours of GPU work the M4
  ran the 3d-64 forward in 3.4-3.6 s instead of 2.4 s; the M1c and current kernels, alternated in one
  process, timed the same, so it was the machine. Source: M2, 2026-10-09.

- **On the Apple GPU a runtime loop bound over per-thread arrays puts them in memory.** The free-pool
  kernel with the pool count as a runtime value ran 2.7x slower than with it as a template
  parameter (one kernel per count, loops unrolled). An array of pointers into the state, suspected
  first, made no difference. Source: M1a, 2026-10-08.
- **A simdgroup waits for its slowest lane: sort the work.** Free-pool segments cost Taylor
  sub-steps in proportion to |A| dt; a CEST spectrum's M0 offset at −1500 ppm is ~80 sub-steps per
  100 µs RF sample against ~2 for the other offsets. Scattered over the simdgroups, 30k spins took
  18 s; sorted by an estimated cost, 3.3 s. With few spins the slowest spin's sequential work
  bounds the run (~1.5 s for 15k segments, wherever it sits). Source: M1a, 2026-10-08.
- **Metal's default fast math keeps a two-sum.** Adding 3e-8 to 1.0 1e5 times: the naive float sum
  stays 1.0, the two-sum's error term recovers 1.003002 (exact 1.003000); `#pragma METAL fp
  math_mode(safe)` changed nothing (torch 2.14 `compile_shader`, macOS 26). Source: M1a, 2026-10-08.

- **At realistic sizes the PyTorch core is memory-bandwidth bound**, not Python bound (~20 µs
  fixed overhead per segment). Porting the same ops to LibTorch would not help; a fused kernel
  does (11–16× over the closed-form core; 43× over the original at 3D 64³).
- **On the Apple M4**: PyTorch uses 4 of 10 threads by default; eager MPS is no faster than
  CPU for this workload (slow elementwise transcendental ops and K=3 matmuls);
  `torch.compile` gives only 1.4× (CPU) / 2× (MPS); the fused adjoint costs ~7× the forward
  (3.8× on a Xeon); Apple clang can't vectorize the RF-segment adjoint loop (GCC can), which
  `-Werror` turns into a build failure.
- **GIRF/eddy-current models make every segment's gradient unique**, defeating the fused
  kernel's per-segment factor cache: ~6× slower forward on 3D 32³. (PR #6)

- **Metal from Python works with no build step**: `torch.mps.compile_shader(src)` compiles MSL at
  runtime and launches kernels on MPS tensors. Pass `threads=` and `group_size=` explicitly (the
  default launch size is wrong for our layout); threadgroup memory is capped at 32 KB.
- **On the GPU, ADC reductions dominate, not transcendentals**: pure spin evolution ran ~4× the
  CPU kernel's rate (1.7e10 spin-segments/s), but summing every ADC sample over spins (simd_sum,
  two barriers, atomics per sample) made it 3× slower than the CPU. Buffering a block of
  samples in threadgroup memory halved it; a per-segment factor cache changed nothing. With one
  coil (the real-data benchmarks) the GPU is 5.8× the CPU; with 8 coils still ~0.4–0.8×.
- **float32 and large phases**: RF spoiling grows receiver phases to thousands of radians; wrap
  them modulo 2π in float64 before casting to float32.

- **A GPU coil sum is a matrix product; do it on the matrix hardware** (Metal, 2026-10-06). The
  signal of C coils at a block of samples is (C × spins) · (spins × samples). Doing it as 2C
  `simd_sum` shuffle reductions per sample made the cost linear in C: 8 coils 3× slower than 1.
  Buffering 8 samples per simdgroup and multiplying on 8×8 `simdgroup_float8x8` tiles, with
  complex numbers as P = Sr·Mr − Si·Mi, Q = Si·Mr + Sr·Mi (store −Mi as well, so there is no
  elementwise negation of a matrix), makes 1–8 coils cost the same (0.41 s vs 0.42 s, 48³ GRE)
  and 32 coils 5.6× faster. Demodulation e^(−iψ) is the same for every spin, so it moves after
  the sum. The sensitivities go in a zero-padded (2, 8·tiles, grid) layout, so tile loads never
  need bounds checks and padded spins contribute zero.

- **Padding lanes must stay well-posed, not just zero** (fused MT, 2026-10-06). Padded lanes
  got r1 = f = k = 0, so the exchange step's det A = 0 and z_ss = 0/0; their sensitivity is
  zero, but 0·NaN = NaN poisoned every coil sum. The final magnetization looked fine, since
  only real lanes are stored; only the signal showed it. Padding lanes now get unit rates.
- **Cache physics on what it depends on, not on the raw inputs** (fused MT). The saturated
  exchange step depends on the RF through |B1|² only, so caching it on (dt, |rf|², grad,
  offset, f_rf) rather than (rf_x, rf_y) lets RF-spoiled repeats hit. MT then costs 1.45×
  the plain CPU kernel on a 3D GRE (2.8× uncached) and +15 % on the GPU.
- **float32 swallows `n − 1 − 1e-9`**: clamping an interpolation position below the last
  node that way does nothing in float32, so `floor` can reach the last node and read past
  the table. Clamp the cell index to n − 2 instead (and let the last cell include its end).

- **GPU kernels pay for state they never use**: per-thread arrays and structs for optional
  physics (bound pool, field model, 60 floats of generalized-Bloch modes) raise register
  pressure and lower occupancy even when a runtime flag skips them: the plain Metal kernel
  went 0.40 → 0.53–0.65 s. Compile variants (a template parameter, instantiated per kernel
  name) so the common path carries nothing extra. Also: `half` is a type in Metal
  Shading Language; don't name a variable that.

- **A low-rank (subspace) reconstruction biases what it is validated with unless its basis
  spans the true curves**: built from non-selective signals only, the basis biased fitted T2 by
  +11 to +41 % on a digital phantom even with the exact model; adding slice-selective signals
  (and both inversion models, so the basis does not favour one model) brought spheres 2–8 to
  within ±11 %. The first ~0.1 s of a single-spoke-per-frame transient is unrecoverable either way.
  Always push a digital twin of the acquisition (measured coil maps, trajectory, noise) through
  the same pipeline before reading physics into reconstructed curves.

- **A Pipe-style density compensation built from the NUFFT Gram operator diverges**:
  w <- w / |A A^H w| with A A^H a sinc-like (not positive) kernel oscillated for 30 iterations
  (weights spanning 1e-15 to 1e-3), so gridded images and the coil maps derived from them changed
  from run to run. CG-SENSE with those maps produced a plausible-looking but wrong echo train
  (a spurious "anomalous" tube). Use per-coil least squares (no DCF) for coil-map images, or a
  positive kernel, and check reconstructions run to run.

- **A digital twin that shares the reconstruction's coil maps (or anything else estimated from
  the data) hides that stage's error** (benchmark 3): the twin recovered spheres 2–8 within ±11 %
  T2, but re-estimating the real data's coil maps with another smoothing moved the fitted T2 of
  spheres 5–8 by 10–30 points. Generate the twin with independently varied ingredients, and
  still treat its floor as a lower bound when the real object is less clean than the twin.
- **Diffusion with strong crushers: compute b-values, don't Monte-Carlo them** (benchmark 5):
  32k random walkers through ~2000 cycles/m crushers still scattered 2.7 % between seeds, larger
  than the effect sought; the pathway b-values from the gradient waveform answered it exactly.

- **In an adjoint, what is read per sample dominates, not the arithmetic** (Metal adjoint,
  2026-10-09): with 8 coils, reading every spin's 16 sensitivity floats at every ADC sample was
  half of the reverse sweep (adjoint 1.97 s vs 0.84 s with a loss on m only, 3d-48). Forming the
  samples' source terms once per 4-segment block with the coils outermost cut it to 1.26 s; the
  coil gradients need no lambda, so they moved into the block's forward recomputation, which has
  each sample's output at hand (one step per sample saved). Likewise the replay buffer: storing
  the two-sum low parts with the state (8 rows instead of 3 per block start) cost ~25 % of the
  backward, and the recomputation does not need them (one ulp, and block starts are global
  multiples of 4, so results stay independent of `checkpoint_every`).
- **Assert that the fast path ran before comparing it with the slow one** (NUFFT readouts,
  2026-10-10): the first readout-run detector compared dt and gradients exactly, but builder
  segments' dt come from differences of absolute times (9.999999999999999e-06 vs 1e-05) and a
  readout's first sample sits half a dwell in, so it found no runs. "NUFFT vs per-sample" then
  differed by 2.5e-5, which looked like a NUFFT error and was the per-sample path twice: the
  order of float32 atomic adds over threadgroups differs run to run. Compare within 1e-9
  relative, start runs at the second sample, and make the test assert `_readout_runs(seq)`.
- **NUFFT readouts on the realism TSE** (PERFORMANCE 2.1): one realism slice (16 coils, 0.9-1.8 M
  spins, laptop) 3.8 -> 1.5 s (2.5x) single slice, 10.4 -> 6.3 s (1.6x) with 16 slices' pulses and
  MT, where the slices' RF is the larger share; signal 3e-6 from the per-sample path, images 3e-5.
  The kernel's spreading reuses the coil MMA; a threadgroup whose sorted spins span more than
  `NUFFT_WMAX` grid cells (fat, steep B0) falls back to per-sample for that readout.
- **After NUFFT readouts, the multi-slice MT cost is RF steps in the MT kernel** (2026-10-10, one
  realism slice, fast, laptop): single 1.50 s (RF 0.34 s of it), single + MT 2.41 s (RF 0.90 s),
  16 slices + MT 6.3 s (RF ~4.5 s: the neighbours' exact pulses triple the stepped RF segments; the
  far bound-only pulses cost 0.1 s). The MT kernel variant is slower everywhere (per-sample readouts
  3.6 -> 6.2 s: registers), and within its RF step the lineshape and exchange step are half, the
  rotation a fifth. Caching the lineshape per pulse and a series exchange step gave 14 % together:
  not kept. Only fewer RF steps per pulse (PERFORMANCE 2.3) changes the picture.
- **A pulse is not one rotation with relaxation around it** (PERFORMANCE 2.3 prototype, float64):
  the 2 ms 90 and 180 deg pulses of the realism TSE as blocks of K samples (half the block's
  relaxation, exchange and mean saturation, the exact product of its rotations, the other half)
  against per-sample steps: one block 5e-3 (WM, GM) to 1.3e-2 (muscle, T2 30 ms) of the 4D affine
  map, K = 10 worse than one block, K = 2 still 2e-4 to 1.6e-3; CSF 1e-4. Relaxation during the
  rotation has to be expanded in the interaction picture (Dyson/Magnus terms per table node), not
  split off.
- **Reuse what repeats per spin before tabulating what varies across spins** (2026-10-10): tables of
  pulse propagators over (Δf, B1) need a perturbative treatment of relaxation and exchange during
  the pulse; but a static spin sees each pulse of a TSE 150-450 times with the same map, and a
  phase-shifted copy (CPMG, RF spoiling) is that map conjugated by Rz exactly. Stepping 6 distinct
  pulses x 5 columns once per spin instead of 459 pulses: RF work /15, 2.3x on the multi-slice MT
  slice, 1e-7 from stepping. Group pulses by tolerance and a least-squares phase (sum z conj(z0)),
  not by rounding values to a hash: float noise straddles rounding boundaries.
- **A GPU shared with another process gives 2x timing swings** (here a desktop app kept the GPU at
  up to 93 %): the same 8-coil adjoint ran in 1.19 s and 2.60 s back to back. Alternate the old and
  new code in one process (load the old module from a saved copy under another name) and compare
  minima; absolute ratios against the plain forward are only good to ~20 % under such load.

## Process and tooling

- **Scratch scripts named like standard modules break imports** (2026-10-11): `inspect.py` once,
  then a `profile.py` next to a new script made torch's optimizer import fail (`cProfile` imports
  `profile`); Python puts the script's directory first on the path. Keep scratch scripts in their
  own directory, with names no module has.

- **Idle GPU threads alias spin 0: guard every write.** Threads past the last spin read spin 0's
  inputs (index clamped) and must not write. One unguarded store in the adjoint's replay let 255 idle
  threads overwrite spin 0's states with zeros: every gradient but λ and PD came out ~0, which looked
  like broken calculus. Source: M2, 2026-10-09.
- **Finite differences on float32 signals need big steps.** DESPOT1's CRB uses a central-difference
  Jacobian (`rel_step` 1e-4): on float32 signals (~1e-6 rounding after 3700 segments) the bound is
  ~1 % noise and the optimizer landed at (3.7°, 38°) instead of (4.1°, 23.6°); 1e-2 gives 0.05°.
  Source: M2, `examples/design_despot1.py`.
- **A test's random weights must not depend on the engine's output dtype.** `torch.randn(shape,
  dtype=...)` with one generator gives different numbers in float32 and float64: every Metal gradient
  cell of the matrix failed by ~100 % until the loss drew its weights in float64. Source: M2.

- **`SequenceBuilder.rf(duration, ...)` splits a long pulse into raster steps by default.** A
  "single 0.5 s CW segment" made with `rf(0.5, amplitude=...)` was thousands of segments (30 s in
  the exact engine); pass `n_steps=1` for one segment. Source: M1c, 2026-10-08.

- **A finite difference can be noise-limited before it is truncation-limited.** On spins 10 cm out
  under a gradient nonlinearity, a central difference with a 1e-6 relative step agreed with
  autograd to only ~1e-6 (rounding of the loss / step), and a 1e-4 step to 8e-5 (truncation).
  Richardson extrapolation of central differences at 1e-5 and 2e-5 (error O(h⁴)) fixed the check.
  Source: `tests/test_engines.py`, M1b, 2026-10-08.

- **A validation script's entry point goes last.** `compare_bmctool.py` defined `cest_comparison`
  after `if __name__ == "__main__": main()`, so running it as a script died with a NameError
  before the CEST section; found when rerunning it through Metal. Source: M1a, 2026-10-08.
- **A check that patches a Python function does not reach the GPU.** BMCTool's super-Lorentzian
  replica is swapped into `exchange` for one comparison; Metal reads its own lineshape table, so
  under `BLOCHPARTY_BACKEND=metal` that row shows the unpatched 1.2e-1. Source: M1a, 2026-10-08.

- **A finite-difference check must rebuild its inputs.** Perturbing leaf tensors in place misses
  everything derived from them when the case was built (`apply_girf` output, `subdivide_rf`
  pieces, a tensor scaled by a leaf): the GIRF, `rf_freq` and diffusion-scale checks failed for
  that reason alone. And a leaf computed from another leaf (a carrier from `rf_frequency(rf)`)
  carries a dependence the rebuilt case sees and autograd does not. Build cases as functions of
  independent leaves and rebuild at θ ± εv; give each run its own graph. Source:
  `tests/test_engines.py`, 2026-10-08.
- **`pytest -m EXPR` on the command line replaces the `-m` in `addopts`** (the last one wins), so
  `pytest -m "not slow"` would have run the opt-in crosscheck tests: 4 collected with
  `-m "not nothing"`, 0 without. The crosscheck tests are therefore also marked `slow`.
  Source: M0, 2026-10-08.
- **Perturb a symmetric matrix symmetrically.** A finite difference along an asymmetric
  perturbation of a diffusion tensor disagreed with autograd by 8 %: the Cholesky factor reads
  one triangle, its backward returns a symmetrized gradient. Perturb a per-spin scale (or
  D + Dᵀ) instead. Source: `tests/test_engines.py`, 2026-10-08.

- **QSM Challenge releases are not interchangeable.** Radboud's complete head package
  declares restricted access and RU-DI-HD-1.0, despite the initial spec's CC BY guess.
  The old Stage-1 short link is a closed registration form. An author's lab separately
  links a public SURFdrive challenge share with both chi/segmentation ground truths,
  but no R1/R2*/M0 or phantom-generation code. Read-only WebDAV listing avoided a 12.65 GB
  bulk download; needed GT files total ~107 MB. (`validation/anatomy/QSM.md`, 2026-10-07)
- **Nominal resolution is not the NIfTI affine.** QSM challenge GT headers use 0.999024391 mm
  while frequency/mask images use exactly 1 mm. Preserve and check affine geometry, and
  do not silently zip arrays of equal shape as if they occupied identical positions.
  The supplied frequency maps also include phase estimation/noise/downsampling, so are
  not exact convolution references for the distributed GT. (`validation/anatomy/QSM.md`)

- **Verify delivered BrainWeb bytes, not just its legacy format page.** Subject-04
  fuzzy files are 0.5-mm (362,434,362), unlike its separate 1-mm T1 reference. The
  public raw-byte response was signed/offset-encoded despite an unsigned description;
  naive uint8 decoding made the air background look like nearly equal tissue mixtures.
  Raw shorts decoded correctly as little-endian 0…4095, despite a big-endian claim.
  Pin decompressed-byte hashes (gzip headers vary), check range, and record physical
  origins from the download header. (`validation/anatomy/README.md`, 2026-10-07 capture)
- **BrainWeb fuzzy totals need asymmetric normalization.** Clamp overfull class sums
  to unity, but leave deficits as air. Normalizing every tiny positive sum to one
  amplifies background memberships into full tissue. Include the explicit background
  class in the normalization before conservative resampling. (`load_brainweb`)
- **The two BrainWeb releases have different validation coverage.** The newer 20-model
  set publishes only T1 (22/9.2 ms, 30°); older SBD provides T1/T2/PD. Its older tissue
  file reproduces older GM/WM contrasts within 0.13%, but newer subject-04 T1 differs
  by 5.44% GM/WM and 21.31% CSF/GM. Do not present a cross-simulator comparison as
  clinical validation, or assume both releases use identical parameters. (`RESULTS.md`)

- **Guard default behaviour with bit-identity snapshots** before and after any core change:
  `tools/snapshot.py save/compare` (both simulators, outputs and gradients, float32/64).
- **Parallel workers need file ownership**, and integration still finds what no single worker
  could: two different γ constants, and design checks defaulting to 150 T/m/s against
  builders designing at 170 T/m/s (every built sequence was reported as a slew violation).
  Write integration tests across workers' features (`tests/test_integration.py`).
- **Expect README/ROADMAP merge conflicts** from parallel workers; resolve as unions.
- **The Agent tool's "remote" isolation fell back to local worktrees** in this setup; real cloud
  runs are started from claude.ai/code with a written brief. `git add -A` then captured
  `.claude/worktrees/` as embedded repos; it is git-ignored now.
- **Cloud workers may merge despite instructions** (PR #1 was merged under the owner's
  account). Re-verify locally after every merge regardless.
- **GitHub marks a PR merged** when its head commit lands in the base branch by a local merge
  and push, drafts included (it can take a few seconds).
- **The Mac sleeps with the lid closed** (clamshell, especially on battery); long runs pause
  and `caffeinate -i` doesn't prevent it. Watch for wall time ≫ CPU time.
- **Scripts must not overwrite tracked outputs by default**: test runs of `nist_tse.py` replaced
  the published benchmark figure; figures are now written only with `--figure`.
- **`while pgrep -f "<text>"` can wait forever**: the waiting shell's own command line contains
  `<text>`. Wait on a PID (`while kill -0 <pid>`) instead.
- **`cmd | grep` buffers background output** until the command ends; use `python -u` without a
  filter, or filter afterwards.

- **Dataset “raw” does not necessarily mean scanner k-space** (validation-source search,
  2026-10-07). Pulseq-CEST's MAGMA release supplies reconstructed DICOMs; CEST–MRF also
  calls image/signal inputs raw. Verify arrays or archive entries before promising a complete
  acquisition benchmark. PulseqDiffusion's phantom MATLAB file really contains complex64
  k-space, but its current generator defaults differ from the stored acquisition dimensions
  and no scanned `.seq` is included. By contrast, the iohexol CEST release includes both
  Siemens `.dat` and `.seq`. Sources and verification: `validation/real/SOURCES.md`.
- **Pin dataset versions and inspect large ZIP directories before downloading**: Zenodo
  concept DOIs can resolve to newer, substantially larger archives. The iohexol CEST concept
  record resolves to 17244380 (31.6 GB), while an indexed older version is 12.3 GB. HTTP range
  reads verified the actual phantom raw files, sequences and calibration scans from its
  ~63 kB ZIP directory without downloading the archive. `SOURCES.md` records the version.
- **finufft and torch cannot share a process on macOS** (each loads its own libomp: "OMP: Error
  #15", either import order). `KMP_DUPLICATE_LIB_OK` is documented as unsafe; run the NUFFT
  stages in their own process (`scholand_recon.py`) and pass arrays by file.
- **Pipe-buffered Python prints arrive only at exit**: a long driver looks hung under `| grep`;
  check the cache files it writes instead (or use `python -u`).

- **A dataset title or README does not establish that all source maps are public**:
  the NeuroPoly phantom repository and OSF archive expose chi/T2/field derivatives,
  but their inventories lack the original R1/M0 inputs. The Jochmann/Schweser Zenodo
  archive also contains derived images and fields, not the full source package.
  AHEAD template v3 does expose aligned chi/R1/R2*/relative-PD maps under CC BY 4.0.
  Its `r2map` filename means R2*, and its RPS orientation requires a y flip for RAS.
  Relative PD must not be presented as calibrated absolute M0; zero QSM does not
  identify air. BigBrain-MR's TE=0 GRE magnitude also retains TR/flip-angle weighting
  and is not M0. Actual inventories and checks: `validation/anatomy/ALTERNATIVE_SOURCES.md`.

- **Removing a module silently skips every test file that `importorskip`s it**, core tests
  included (2026-10-09, retiring `blochparty.fused`: nine files had `fused =
  pytest.importorskip("blochparty.fused")` plus an autouse fixture at module level, so deleting
  the package would have turned their core checks into skips). Grep for the module name in
  `importorskip` and fixtures before deleting, and compare passed counts, not just failures.
- **`.double()` on a complex tensor drops the imaginary part** (with a warning that is easy to
  miss in a script): a coil-gradient comparison showed relative error 1.0 that was only the
  measurement. Compare complex tensors with `.to(other.dtype)` or `.to(torch.complex128)`.
- **`git stash` in the middle of a rename splits it** into a staged add and an unstaged delete
  (content intact, but confusing); do not stash while a refactor has renames in flight.

- **Wall time is not run time when the Mac sleeps, and `time.perf_counter` does not count the
  sleep** (2026-10-09, overnight with the user away): benchmark 1 took 55 min of wall clock for 251 s
  of simulation, the full suite 88 min while pytest reported 338 s. `pmset -g log` showed the
  machine in Maintenance Sleep with ~180 s DarkWake windows every ~15 min (closed lid or a manual
  sleep; the app's keep-awake prevents idle sleep only), and Python's monotonic clock
  (mach_absolute_time) stops during sleep, so in-script timings stay right while `date` and `time`
  do not. Swap (8.5 GB of 10 GB used, a 6.7 GB test process footprint) looked like the cause and
  was not. Before blaming an engine for a slow wall clock, compare it with `perf_counter` and read
  `pmset -g log | grep -E "Sleep|Wake"`.
- **A results table can predate a correction that the main table got**: benchmark 1's
  convergence rows were computed before the T2 fix (their best row is the old 1.44 %), so the
  16 x 1 row read 14.35 % where the current code gives 16.72 % on both engines. When a rerun
  disagrees with a recorded number, rerun the old code on the same inputs before suspecting the
  new engine.

- **`torch.as_tensor(python_float)` is float32** (the default dtype), not float64: it turned a 6.65 ms
  block duration into 6.6499998793 ms (off the raster) and an RF-spoiling receiver phase into a
  3e-8 error, both in the Pulseq writer. Convert Python numbers with `float()` or an explicit dtype.
- **zsh does not split an unquoted `$var` into words** (bash does): `set -- $job` in a loop over
  "a b c" strings gave one argument, and a chain of background jobs failed at once. Write the
  commands out, or use `${=var}`.

- **A wait loop must also end when its run dies** (2026-10-09): six `until grep ...; do sleep; done`
  loops outlived their runs by hours, because each waited only for success: a run stopped by hand,
  a `time` that prints "total" (zsh) instead of "real", a grep for a line that never follows the
  one it anchored on. Wait on the process (`while kill -0 PID`) or match failure and success alike,
  give the loop a deadline, and stop the loops when the result is in.

- **A presigned S3 link answers HEAD with 403** (2026-10-13): the fastMRI download link looked expired to `curl -I`;
  the signature covers GET only, and a ranged GET (`-r 0-1023`) returned 206. Test signed links with the method they
  were signed for.
- **Freeze the real set while runs are queued** (2026-10-13): score.py reads whatever the fastMRI manifest holds at
  scoring time, so extending the data mid-queue would have given consecutive runs different real sets. The extension
  streams into a second directory of symlinks plus new files (`~/data/fastmri_more` on the mini) and only the runs
  meant for the larger benchmark point `--fastmri` at it.

- **Queued jobs must not share a working directory** (2026-10-13): `quick.sh` swaps `prepared/` (rm, copy) to pick its
  head set, and two chained scripts did the same while r31 was reading heads from it: 50 of 57 heads failed with
  "No such file" and a quick run lost its own heads the same way. Each run names its own input directory
  (`simulate_heads --prepared`), and nothing copies into a directory another process reads.

- **A long-running Metal process can start returning zero signals** (2026-10-14): a generator that simulated 60 heads
  in one process on the mini returned all-zero k-space for 41 of 181 slices from the 30th slice on, with normal
  timings and no error; the same heads regenerated in fresh processes were fine. One head per process, and never
  write a sample without checking the signal is finite and non-zero (`generate.py` now raises).

- **A shared machine's disk is not ours** (2026-10-14): with the mini at 88 % full (42 GB of fastMRI archives, every
  run's raw k-space, a generated set), its owner removed `~/data` wholesale: the real scans, the caches, the runs, the
  181-sample set. The numbers survived in the journal; the data did not. Fetch only what a benchmark needs (the 3 T
  AXT2 subset, not every contrast), delete raw simulations once scored, prune old prepared-head versions, check
  `df -h` before anything that writes gigabytes, and keep whatever matters in the repo or on the laptop.

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

## Tissue-property database — 2026-10-10

- **Getting at full texts.** Through this environment's proxy: Wiley, RSNA, AJR, ScienceDirect and
  the Wayback Machine answer 403 / Cloudflare / captcha; PMC article pages start serving reCAPTCHA
  after ~20 quick requests. What works: PMC OAI-PMH
  (`https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:<digits>&metadataPrefix=pmc`)
  and NCBI `efetch db=pmc` return clean JATS XML for open-access papers AND NIH author manuscripts;
  Europe PMC `fullTextXML` returns HTTP 500 for author manuscripts. In NIH-manuscript JATS the
  tables sit in `<floats-group>`, outside `<body>`. ISMRM proceedings PDFs before ~2010 download
  from `cds.ismrm.org/protected/<yy>MProceedings/PDFfiles/<nnnnn>.pdf` without login. Some
  "tables" are images (Campbell-Washburn 2019 Table 1). Source: the eight row-extraction passes.
- **IT'IS proxies are copies, and the copies reveal them**: Eye Retina = Brain GM, Iris/Ciliary
  body/Diaphragm = Muscle, Choroid/Heart lumen/Placenta = Blood, Connective tissue's water content =
  WM's. A (mean, SD, n, min, max) tuple shared by several names is a proxy; the owner must be chosen
  per table (the Gabriel 1996 dielectric sets have other owners than the relaxation rows: Testis,
  Thyroid, Tendon, Nerve, Stomach, Colon, inflated Lung). `tools/tissuedb_import_itis.py` keeps 212
  blocks and logs 459 skipped. IT'IS "Breast Fat" has T1 0.9-1 s (breast tissue, not adipose), and
  the T1 of WM there falls from 1.0 s at 1.5 T to 0.87 s at 3 T because each field pools other studies.
- **Abstract labels lie**: Wansapura 1999 calls T2* "T2" and lists "GM and WM 80 and 110 ms" in the
  order opposite to physics (GM T2 > WM T2); Shin 2025 swaps WM/GM in the abstract (Table 3 is GM,
  WM); the ISMRM archive page of Peters et al. 2006 lists Hoogenraad first (the PDF has Peters).
  `validation/anatomy_brainweb.py` used the abstract order until 2026-10-10 (now GM .110, WM .080 s;
  see the Wansapura entry under "Pulseq, data and reference values").
- **The CSF T1 the project quotes (4.4 s, via Spijkerman 2018 ref. 13) is 4.3 +- 0.2 s in Rooney
  2007's abstract** (across 3 subjects and all fields; Rooney did not scan at 3 T). Spijkerman's
  2.03 s 3 T CSF T2 is a median from the longest-TE (partial-volume) analysis. Caan 2019 scanned 8
  but analysed 6 subjects.
- **Method beats tissue in many cells**: myocardial T1 depends on MOLLI/ShMOLLI/SASHA, kidney T1 on
  the readout (>200 ms between IR SE-EPI and bFFE, Cox 2017), cortical-bone T1 by a factor ~2.5
  (in vivo UTE-IR vs ex vivo IR-CPMG), MRF T2 runs low in kidney and high in abdominal muscle, VFA
  T1 is 16-28 % above Look-Locker in the same subjects (Krug 2022), blood T2 depends on the CPMG
  spacing, tendon "T2*" on the echo set. The `method` column is part of the value.
- **MT conventions in practice**: Vanderbilt SIR kmf and Bayer's bSSFP kmf already are the
  bound -> free R; Yarnykh/CRI k is free -> bound (R = k(1-f)/f); PSR/F = M0b/M0f -> f = PSR/(1+PSR),
  SD scaled by 1/(1+PSR)^2; the Gloor 2008 bSSFP equation overestimates F by ~40 % (Bayer 2022).
  Stanisz 2006 prints blood T2B 280 +- 50 us next to T2 275 +- 50 ms: probably a typo, kept flagged.
- **QSM numbers carry their reference**: CSF, a deep-WM ROI set to -0.030 ppm, the whole-brain mean
  and phantom ground truth all occur; the database keeps them as `chi_qsm` with the reference in
  `method`, separate from absolute `chi`. Susceptometry papers report cgs: SI = 4 pi cgs.
- **Most blood-oxygenation calibrations are bovine blood** (Lu 2012, Qin 2011, Krishnamurthy 2014,
  Grgac 2013/2017), and Liu 2016's printed T2 coefficients do not reproduce its own 191 ms.
- **Pool studies as equals, and relaxation times by their reference method** (2026-10-10): n-weighted
  pooling let Ndengera 2022 (SyMRI, 54 subjects aged 83 on average) set WM T2 108 ms against 56-80
  ms elsewhere: pooled WM and GM T2 came out 86 ms each, and simulated T2w heads lost all GM/WM
  contrast (realism f2). Between-study differences (method, cohort, region) dwarf sampling error, so
  the random-effects weights are equal: median over sources. T1/T2 from spin-echo decay and
  inversion/saturation recovery first: WM T2 74.5, GM 110, muscle 30 ms (52 with MRF rows). Rows
  without a method cannot be classified and drop out of a tier that has reference rows (de
  Bazelaire's fat rows had an empty method; filled from the paper's protocol).
- **The bright scalp in T2w TSE was anatomy first, fat T2 second** (2026-10-10): real fastMRI T2w
  heads show a scalp peak 2.3x the brain interior 5-7 mm below the edge; the simulation had a flat
  1.6, and changing fat's T2 from 68 to 148 ms changed nothing, because BrainWeb labels the scalp
  "Muscle/Skin" (here `skin`, given muscle values) and the fat class covers 0-24 % of it. Splitting
  that class by depth (dermis 1.61 mm, subcutaneous 4.14 mm, Li 2024 ultrasound: `layer_scalp`) put
  the peak in place (1.26 at fat T2 68 ms), and the J-suppressed fat T2 lifted it to 1.86. Measure
  where a contrast lives (an edge-depth profile, tissue composition per depth band) before tuning a
  parameter. Still off: the calvarial diploe (fatty marrow) is 0.8 vs 1.6 in the real profile.
- **J coupling of the chain ends does not explain bright fat by itself** (2026-10-13, validation/jcoupling_fat.py): a
  density-matrix model of the coupled triglyceride fragments (methyl, carboxyl, allylic, diallylic, glycerol; J 7 Hz),
  calibrated to CPMG fat T2 145 ms at 8 ms spacing, predicts single-echo fat T2 222 ms where 48-68 ms is measured at
  3 T. Half the fat signal is bulk methylene whose neighbours share its shift (no observable J), and the coupled groups'
  modulation oscillates and partly recovers. Check a mechanism against both limits (here SE and CPMG) before turning it
  into a parameter.
- **A Bloch simulator's lipid T2 is the J-suppressed one** (Henkelman 1992; Keene 2020): single-echo
  SE and STEAM read fat T2 40-75 ms at 3 T because J coupling modulates the echo; a train of
  refocusing pulses spaced well below 1/(chemical-shift difference of the coupled groups) suppresses
  it (Allerhand), and CPMG at 8 ms spacing reads 140-159 ms. A Bloch model has no J coupling but
  plays the refocusing train, so it needs the CPMG value; with it, simulated single-echo SE fat is
  too bright (the missing physics, REALISM.md).
- **Fractions of a compartmentalised tissue must not be pooled as one tissue**: cortical-bone pd rows
  of bound and pore water add up, an average of them means nothing; they are separate ids
  (`bone.cortical.bound_water`, `.pore_water`, `.collagen`).
