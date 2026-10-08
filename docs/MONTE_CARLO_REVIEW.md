# Monte Carlo methods in Bloch Party: accuracy, speed, and an improvement plan

**Review date:** 2026-10-08. **Code baseline:** `29bae6e`.

## Main conclusions

Bloch Party already makes several good numerical choices: complex signals are summed before
forming magnitudes; sub-voxel positions are stratified; the NIST experiments use randomized
Sobol points; diffusion uses reproducible Philox counters and includes the Brownian-bridge
phase that an endpoint-only walk would miss. The independent diffusion and equilibrium
checks are particularly valuable. Keep these foundations.

The most urgent work is to make the reported accuracy trustworthy. The current NIST
"magnitude-bias correction" is an unbiased **power** correction followed by a biased nonlinear
transformation. The headline uncertainty calculation discards covariance between spheres.
Deterministic T2′ quadrature has error that repeated seeds cannot reveal. These limitations
matter precisely at inversion nulls, late echoes, and small model differences.

The best next speed improvements should reduce the number of expensive spin histories needed
for a specified error, alongside optimizing the GPU. Start with paired comparisons, better
sample allocation, and a bridge-conditioned diffusion mode. Then investigate control variates,
exact Gaussian aggregation of free diffusion intervals, and multilevel RF corrections.

| Priority | Change | Expected benefit | Evidence / qualification |
|---|---|---|---|
| P0 | Correct estimator terminology; retain complex means and unclipped power; diagnose nulls | Avoid misleading accuracy claims | Algebra and 200,000-trial experiment below |
| P0 | Jackknife the actual aggregate statistic and paired model differences | Correct treatment of correlated errors | Concrete omission in both NIST reporting paths |
| P0 | Separate sample error, time-step bias, geometry truncation, and float32 error | Know which refinement actually helps | Different code paths have different error sources |
| P1 | Validate QMC sample counts, sparse weights, seed identity; benchmark digital shifts | More reliable sampling; possible variance reduction | Current real-valued shifts lose dyadic balance; benefit is integrand-dependent |
| P1 | Allocate spins using pilot variance and runtime, preserving domain support | Spend fewer spins on insensitive regions | Current allocation is a fixed distance heuristic |
| P1 | Integrate out free-segment Brownian-bridge phase | Fewer Gaussian draws and lower variance | Exact under stated assumptions; toy variance reduction 1.35–2.81× |
| P1 | Add persistent spin IDs and reusable simulation buffers | Safe batching, stronger paired comparisons, less setup overhead | Counters currently depend on local spin indices |
| P2 | Control variates / multilevel corrections using the existing engine | Potential large reductions in expensive histories | Requires measured correlation and cost; no speedup claimed yet |
| P2 | Bucket restricted walkers by required step size | Avoid the smallest compartment slowing every walker | Current step limit is a global minimum |
| P2 | Conditional spatial integration and targeted T2′ quadrature | Reduce effective sampling dimension | Useful in restricted physics regimes; needs convergence checks |

These are recommendations, not implemented simulator changes. This review adds only a
reproducible diagnostic script, its results, and documentation.

## Scope and evidence

The review covers the reference and Metal engines, phantom construction, anatomy sampling,
NIST scanner-validation estimators, receive-noise replicas, relevant tests, and existing
performance/convergence records. It does not treat diffusion simulation as MCMC inference:
walkers have Markov dynamics in physical time, but there is no posterior sampling or
Metropolis acceptance step here.

Three evidence levels are distinguished throughout:

1. **Code facts:** observations at the baseline above, with file/function references.
2. **New checks:** small reproducible CPU experiments and existing targeted tests.
3. **Proposals:** derived algorithms or experiments that still need implementation and
   end-to-end validation.

Reproduction from the repository root:

```sh
.venv/bin/python validation/monte_carlo_audit.py
```

The script calls the existing NIST estimators and sampler, the existing phantom builder, and
the existing Philox implementation. Numerical output is saved in
[`validation/monte_carlo_audit_results.json`](../validation/monte_carlo_audit_results.json).
Environment: PyTorch 2.14.1, SciPy 1.18.1, NumPy 2.5.3. MPS was unavailable to this execution
environment, so no new Apple-GPU throughput or real-scanner benchmark is claimed. Previously
recorded GPU results below are historical project evidence.

## 1. What is being sampled, and what already works

| Mechanism | Implementation | Numerical interpretation and assessment |
|---|---|---|
| Regular sub-voxel grid | `phantom.py:phantom_from_maps`, lines 147–160 | Deterministic midpoint quadrature. Cheap and reproducible, but coherent phase aliasing can create false echoes. |
| Stratified spatial jitter | Same function; `anatomy.py:to_phantom` | One uniform point in each sub-cell. Correct volume weights preserve signal mass. Variance reduction is strongest when the response varies smoothly within cells. |
| NIST spatial QMC | `validation/real/nist.py:_sobol_points`, `phantom_qmc`, `fill_phantom_qmc`, lines 223–289 | Scrambled 3D Sobol points plus per-pixel modulo-one shifts. Integrates slice position and in-plane location, with geometric rejection. |
| Sparse water fill | `fill_phantom_qmc`, lines 260–284 | Unequal sample allocation across pixel strata with inverse-count weights. More precisely stratified allocation than an optimized continuous importance density. Correct weights are a strength. |
| Static T2′ offsets | `phantom.py`, lines 162–170; `anatomy.py`, lines 867–883 | Inverse-CDF Cauchy offsets. Default midpoint quantiles are deterministic frequency quadrature, randomly paired with positions. Random draws are a separate option in `phantom_from_maps`. |
| Tissue identity | `anatomy.py`, lines 817–861 | `per_tissue` retains weighted tissue components; `categorical` samples labels. The former avoids label noise at a higher spin count. |
| Fat peaks | `phantom.py`, lines 175–193 | Explicit weighted sum of six species, avoiding categorical peak noise. A useful existing example of summing out a small discrete uncertainty. |
| Free diffusion | `bloch.py:_walk`, lines 210–233; Metal lines 307–342 | Exact joint endpoint and integrated-position sampling for free Brownian motion under a segment's constant linear gradient. Includes anisotropic D through Cholesky factors. |
| Restricted diffusion | `restriction.py`; Metal lines 344–382 | Uniform starts, Gaussian substeps, specular reflection, approximate phase integration. Equilibrium and analytic-limit tests already catch important boundary mistakes. |
| Random streams | `rng.py`; Metal Philox | Counter-based Philox4x32-10 and Box–Muller. Excellent for replay, checkpointing and matching engines without maintaining a huge RNG-state array. |
| Thermal noise replicas | `coils.py:add_noise`; `design.py:image_noise_sigma` | Simulates physical receiver noise through reconstruction. This is a different uncertainty from numerical spin-integration error. |

The NIST code correctly adds sphere and water signals **in the complex domain**, reconstructs
linearly, then averages seed images before taking magnitude. Averaging already-magnituded
seed images would retain a much larger noise floor.

The spatial rejection weights are also correct in principle: rejected points contribute zero
to a fixed-domain quadrature. Do not divide by the number of accepted points; that would
silently change volume normalization.

Historical evidence in [`validation/real/RESULTS.md`](../validation/real/RESULTS.md), under
"Monte Carlo estimation", supports the current direction: the water-patch study reports
roughly N^-0.75 Sobol error over its tested range and about 12× fewer spins than stratified
sampling for a particular error level. Those are empirical, workload-specific results. They
are not a general convergence rate for every pulse train, boundary, or QMC randomization.

## 2. Accuracy findings that deserve attention first

### 2.1 The magnitude correction does not make magnitude unbiased

**Confirmed:** `nist.py:mc_magnitude`, lines 297–305.

For R independent complex image estimates X_r with mean μ, the code computes

\[
\bar X=R^{-1}\sum_r X_r,\qquad
\widehat v=\frac{\sum_r|X_r-\bar X|^2}{R(R-1)},\qquad
\widehat A=\sqrt{\max(|\bar X|^2-\widehat v,0)}.
\]

The identity is `E[|mean|² − v_hat] = |μ|²`. It does **not** imply `E[A_hat] = |μ|`.
This identity needs independent, identically distributed seed estimates with finite variance,
but does not require circular Gaussian error. Clipping and square-rooting reintroduce bias.

A check using 200,000 independent four-seed experiments, with Gaussian errors and standard
deviation 1 in each component of the seed mean, gives:

| True amplitude | Mean raw magnitude | Mean current corrected magnitude | Fraction clipped to zero |
|---:|---:|---:|---:|
| 0 | 1.2497 | 0.5240 | 58.09% |
| 1 | 1.5456 | 0.8459 | 43.98% |
| 2 | 2.2705 | 1.6960 | 18.55% |
| 5 | 5.1004 | 4.8884 | 0.0185% |
| 10 | 10.0497 | 9.9485 | 0% |

The standard errors of the corrected means are 0.0017–0.0026. These are not small numerical
fluctuations. The correction improves some situations, but it leaves a positive null floor
and can overshoot downward for nonzero signal. This experiment tests the estimator under a
known distribution; it does not measure the actual bias of a particular NIST reconstruction.

There is also a useful high-SNR explanation. If each mean-noise component has variance σ²,
raw magnitude has leading bias +σ²/(2A). Subtracting total complex variance 2σ² inside the
square root gives leading bias −σ²/(2A). Subtracting only σ² would cancel the leading circular
Gaussian high-SNR term, but is **not** a universal replacement near nulls or for anisotropic
complex errors.

**Recommended changes:**

- Describe the existing output as a *power-debiased magnitude approximation*.
- Return and save the complex mean, real/imaginary covariance, unclipped power estimate,
  sample counts, and clipping fraction. Preserve negative power estimates for diagnostics.
- Use complex or independently phase-referenced signed signals for simulation convergence
  where possible. Do not change a scanner magnitude-ROI benchmark's estimand without saying so:
  mean pixel magnitude and magnitude of the complex ROI mean are different quantities.
- Where only magnitude comparisons are meaningful, measure residual estimator bias against a
  high-accuracy synthetic reference across the observed SNR range. Use null-specific absolute
  error targets, or a justified noise model, rather than assuming the square-root correction
  has solved the problem.

An equivalent unbiased power statistic is

\[
\widehat P=\frac{\left|\sum_r X_r\right|^2-\sum_r|X_r|^2}{R(R-1)}
=\frac{1}{R(R-1)}\sum_{r\ne s}\operatorname{Re}(X_rX_s^*).
\]

That identity is useful for testing the implementation; it does not remove the magnitude
problem. Numerical MC noise and scanner thermal noise need separate treatment, especially
when the real data use multiple-coil RSS and the simulation uses one uniform coil.

### 2.2 The headline standard errors omit covariance between spheres

**Confirmed:** `nist_irse.py`, line 213, and `nist_tse.py`, line 221, report the mean error of
ten spheres with `sqrt(sum(se[:10]**2))/10`.

That formula assumes zero covariance. The actual variance is

\[
\operatorname{Var}(10^{-1}\textstyle\sum_j e_j)
=10^{-2}\sum_{j,k}\operatorname{Cov}(e_j,e_k).
\]

Reconstruction spreads each source's error across the image; water contributions and shared
randomizations can couple ROIs. Even separately sampled spheres do not establish independent
*reconstructed sphere metrics*. The omitted covariance can have either sign. This review has
not recomputed the historical NIST bars, so their numerical correction is unknown.

In an exact diagnostic with ten perfectly correlated outputs, the current formula returns
0.2041 while jackknifing the aggregate returns 0.6455: an underestimate by √10. This
demonstrates the failure mode, not the size of the error on the real benchmark.

**Fix:** apply the existing jackknife to the complete scalar statistic:

```python
aggregate = lambda imgs: shape_error(roi_means(mc_magnitude(imgs)[0], rois), real)[:10].mean()
estimate, se = jackknife(aggregate, seed_images)
```

For a model comparison, leave out the same seed from both models and jackknife their paired
difference. Merely observing overlapping individual error bars is not a paired significance
analysis. Save seed-level complex images so this can be repaired without rerunning the Bloch
engine; current `monte_carlo` functions return errors, SEs and curves, discarding those images.

### 2.3 Four seeds are a fragile basis for nonlinear uncertainty estimates

`jackknife` computes the full-sample plug-in estimate and a leave-one-out SE. It does not return
the bias-corrected jackknife point estimate. That is defensible, but should be stated.
The clipping in `mc_magnitude`, the square root near zero, and the fitted scaling in
`shape_error` make this statistic less regular than a sample mean.

Guard `mc_magnitude` with R ≥ 2 and jackknifing that function with R ≥ 3, because a
leave-one-out call needs at least two seeds. For a practical validation policy, start with
8–16 independent scrambles when calibrating error bars, then choose production R and N using
measured accuracy per unit cost. This is a proposed starting range, not a coverage guarantee.
A four-replicate Gaussian sample-mean 95% interval uses a t multiplier about 3.18; ± one SE
should never be read as a 95% interval.

The allocation tradeoff between points per scramble and independent scrambles is real;
increasing R can improve uncertainty estimation while reducing integration accuracy at fixed
total work. RQMC error-bar theory remains more delicate than ordinary IID MC. See
[Owen, *Error estimation for quasi-Monte Carlo*](https://arxiv.org/abs/2501.00150).

### 2.4 Sobol is valuable, but the implementation gives up some of its structure

**Confirmed:** `_sobol_points` uses `.random(n)` and adds a real-valued modulo-one shift to
each pixel. `fill_phantom_qmc` rounds `n*sparse` and takes that prefix.

- Require positive power-of-two N and power-of-two retained prefixes in the structured QMC
  mode, and use `random_base2`. The defaults 256 and 64 already satisfy this; arbitrary user
  values do not. SciPy explicitly documents these balance requirements and its LMS+digital
  shift scrambling in the [Sobol API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html).
- A Cranley–Patterson real shift preserves uniform marginals and can be useful, but it does not
  preserve Sobol's exact binary elementary-interval balance. It is not equivalent to an XOR
  digital shift. Benchmark per-pixel digital shifts or independent scrambles before changing
  the default.
- The dense and sparse groups call `_sobol_points` with the **same** seed and restart the
  pixel index. Their corresponding local pixel ranks therefore reuse shifts and points.
  This need not bias the integral, but creates unnecessary dependence. Key streams by stable
  pixel identity and purpose instead.
- Validate `0 < sparse <= 1`. With `sparse > 1`, the code can request `n_pix > n`, generate
  only n points, and still weight each point by `1/n_pix`: a genuine underweighting bug for
  that unsupported-looking but currently unguarded input.
- Generate only the sparse count needed for sparse pixels. The current code allocates an
  `(pixels, n, 3)` array before slicing to `n_pix`. At the default quarter density this is
  four times the required point-array storage for that group. Stream large pixel batches;
  the distance-to-all-spheres temporary also scales with candidates × sphere count.

**New check:** integrate known Fourier transforms of a unit cube and an inscribed radius-0.4
ball, at frequency `(3.25, 1.5, 0.75)` cycles/unit, across 128 independent seeds:

| N=1024 method | Cube complex RMSE | Ball complex RMSE |
|---|---:|---:|
| IID points | 0.03056 | 0.01584 |
| Scrambled Sobol without extra shift | 0.01164 | 0.00792 |
| Current Sobol + real shift | 0.01543 | 0.00751 |

The extra shift improves one test slightly and worsens the other. Across these seeds, about
33% of the finest one-dimensional dyadic bins cease to contain exactly one point after the
shift; unshifted scrambled Sobol retains that property. This supports an A/B experiment,
not a claim that removing shifts must speed up the scanner benchmark. Simply sharing one
unshifted pattern across every pixel would also create strong cross-pixel dependence.

Geometric rejection is unbiased with the existing weights, but the indicator at sphere/water
boundaries reduces smoothness. Together with rapid phase oscillation, this explains why one
should measure convergence rather than promise an ideal smooth-integrand QMC rate.

### 2.5 Deterministic T2′ error is invisible to seed replication

The default offsets are `tan(pi*((j+0.5)/N - 0.5))` in a random permutation. For a uniform
voxel without spatially varying encoding, every seed integrates **the same frequency set**.
The anatomy builder additionally fixes its quantile-shuffle generator to seed 0.

**New check:** N=256, T2′=20 ms, eight seeds, and 1,001 times over 0–60 ms. The maximum complex
FID error relative to `exp(-t/T2′)` is **0.03818 of initial signal**, while maximum seed SD is
**2.08e-16**. This is deterministic quadrature error, not an undetected RNG malfunction.

Keep the quantile option for its reproducibility and often-good accuracy, but report its
N-convergence separately. Add randomized stratified quantiles or a joint scrambled design
over `(x,y,z,frequency-CDF)` when independent replicate error estimates are required. Permuting
fixed quantiles can assess spatial pairing noise; it cannot assess frequency integration bias.

The Cauchy distribution has no finite frequency variance, but the free-precession phasor is
bounded, so a phasor MC variance still exists. Do not clip frequency tails casually: clipping
changes the lineshape. Any truncated quadrature needs a controlled tail contribution. For
gradient-based optimization, separately check derivatives: differentiating a phasor can
introduce unbounded frequency factors even when the signal itself is bounded.

### 2.6 More walkers cannot repair a biased time or geometry discretization

The exact free-walk formula applies to free Brownian motion, constant linear gradients within
a segment, and linear phase accumulation. It does not make all moving-spin physics exact:

- `bloch.py:_run` feeds one effective position into a finite RF rotation. Rotations about
  time-varying axes do not generally commute, so diffusion during RF still needs refinement.
- Sampling B0/B1/coil maps at an effective position is not the same as integrating nonlinear
  maps along the path. Receive sensitivity belongs at the ADC position; inspect this
  separately from phase averaging. The reference RF+ADC path also calls `_sample` with the
  original `spins` rather than `step_spins`, an adjacent mapped-coil consistency issue worth
  a focused test.
- Scanner nonlinear encoding positions and concomitant coefficients are constructed before
  walking. Diffusion with these fields needs a physical-coordinate treatment and independent
  convergence checks; agreement between engines that share an approximation cannot prove it.
- Restricted `walk_segment` reflects endpoints but adds an unconstrained free-bridge term
  to their midpoint. The bridge conditioned on reflection is not a free Gaussian bridge.
  The current approximation needs finite-gradient-duration step convergence, beyond tests of
  equilibrium and the short-gradient-pulse limits. This is a model-of-the-step concern, not
  a measured failure of the existing restricted-signal tests.
- The fixed slab ±1.5 slice thicknesses and finite water margin truncate the integration
  domain. Increasing N within that domain cannot test omitted excited water or slice tails.

Expose a restricted-step refinement factor and sweep it independently of walker count.
Compare finite-width pulse signals with a deterministic Bloch–Torrey/PDE or spectral reference,
not only with another implementation of the same walk. Retain the excellent equilibrium KS
tests. Monitor the eight-bounce cap and assert endpoints are inside instead of silently
accepting an exhausted reflection loop.

### 2.7 Reproducible counters are not yet stable sample identity

The Philox counter includes local spin index, segment, substep, and block. The reference uses
`np.arange(n)` and Metal uses thread index `j`. Reordering or separately simulating batches
changes the mapping from physical spin to random path; separate batches with seed 0 reuse
the same local-index streams. This does not automatically bias a sum, but breaks full-versus-
batched path equality and can add correlation between batches.

Add immutable global spin IDs or a documented spin offset, plus separate stream namespaces
for geometry, diffusion, quantiles, noise, and pilot/production samples. The current spin
counter occupies 32 bits, so define the ID range or extend the layout. IDs must survive
pruning and importance allocation. `simulate_chunked` currently **rejects diffusion**; this
is a prerequisite for future diffusion batching, not a claim that its existing static
backward calculation is wrong.

Different sequence discretizations also change the segment counters. The same integer seed
does not couple coarse and fine Brownian paths. A multilevel method needs explicit shared
increments or a Brownian tree. Counter-based generators make this feasible; the existing
architecture is well suited to it. See the authors' [Random123 implementation](https://github.com/DEShawResearch/random123).

## 3. Improvements with the best accuracy-per-time potential

### 3.1 Analytically average the independent Brownian bridge

**Proposal with a checked single-segment derivation.** This is conditional Monte Carlo: keep
the sampled endpoint and integrate out randomness that need not be explicitly simulated.

For a free segment of duration h, diffusion tensor `D = LLᵀ`, and gradient g in **Hz/m**,
the current implementation uses independent standard Gaussian vectors Z₁ and Z₂:

\[
\Delta r=\sqrt{2h}LZ_1,\quad
\bar r=r_0+\Delta r/2+\sqrt{h/6}LZ_2,\quad
\phi=2\pi h\,g^T\bar r.
\]

Conditioning on the endpoint gives exactly

\[
E[e^{-i\phi}\mid r_0,r_1]
=e^{-i2\pi h g^T(r_0+r_1)/2}
\exp\!\left[-\frac{(2\pi)^2h^3}{12}g^TDg\right].
\]

Replace the bridge draw with this deterministic attenuation during eligible RF-free segments.
Keep the endpoint walk and its correlation with magnetization; do not replace the entire
history by a scalar attenuation after an arbitrary multipath sequence. Because the bridge is
independent of future endpoint-driven evolution and Bloch propagation is linear in state,
the conditional mean can be propagated through subsequent eligible dynamics and RF events.

Initial eligibility should be deliberately narrow: ordinary free diffusion, no reflecting
boundaries, no spatially varying maps or scanner nonlinear fields, and no RF within the
averaged interval. Bound-pool coherent modes require their own consistent treatment. The
output state represents a conditional mean, not an individual physical walk; expose this
as an estimator mode rather than silently changing single-path semantics.

For a single segment starting at zero, total phase variance is v, with 3v/4 from the endpoint
and v/4 from the bridge. The complex variances before/after conditioning are
`1-exp(-v)` and `exp(-v/4)-exp(-v)`. With 300,000 existing-Philox samples:

| v | Exact mean | Conditioned real mean | Raw / conditioned complex variance |
|---:|---:|---:|---:|
| 0.1 | 0.951229 | 0.951238 | 1.353× |
| 1 | 0.606531 | 0.606666 | 1.540× |
| 4 | 0.135335 | 0.135828 | 2.809× |

All conditioned means are within one estimated real-component SE of the exact answer. This
is variance reduction **per walker** in these checks, not a measured runtime multiplier.
The GPU currently draws two Philox blocks and uses three Box–Muller pairs for six normals;
an endpoint-only draw needs one block and two pairs for three normals. A speed benefit is
plausible, but ADC reduction may still dominate.

There is no useful universal gain factor: finely sampled weak-gradient segments may have
tiny bridge variance. Measure the benefit on actual protocols. The conceptual basis for
conditioning, pairing, and control variates is reviewed in
[Owen's Monte Carlo text, chapter 8](https://artowen.su.domains/mc/).

### 3.2 Aggregate whole free diffusion intervals as joint Gaussians

The reference merges free precession algebra but still generates walks for every underlying
segment. Metal also walks segment by segment. In eligible RF-free intervals, Brownian endpoint
displacement and phase under a piecewise-constant gradient are jointly Gaussian and can be
sampled from precomputed covariances.

For an interval `[0,T]`, define `q(u) = 2π ∫_u^T g(t) dt`. Then

\[
\operatorname{Cov}(\Delta r)=2TD,\qquad
\operatorname{Var}(\phi)=2\int_0^T q(u)^TDq(u)\,du,\qquad
\operatorname{Cov}(\Delta r,\phi)=2D\int_0^T q(u)\,du.
\]

Add deterministic phase `2π r₀ᵀ∫g dt`. Those integrals can be evaluated exactly for the
piecewise-constant gradient representation. This can replace many raster steps with one
small joint draw. Split at RF, ADC, spoil boundaries, and any unsupported spatial physics.
For multiple ADCs, preserve their joint phase/endpoint law; independently drawing marginal
ADC phases would give wrong temporal covariance and future state correlations.

Validate against the current fine walk in distribution and against Stejskal–Tanner/EPG,
including unequal gradient lobes and anisotropic tensors. The estimator changes random-path
construction, so same-seed spin-by-spin equality is not the appropriate sole acceptance test.

After this reduction, test randomized QMC for the remaining Gaussian variables. A naive
replacement of all six normals per segment by Sobol coordinates creates an enormous
effective dimension. Instead order coordinates by their contribution to observed phase,
using the joint Gaussian covariance, a principal-component factorization, or a coarse-to-fine
Brownian construction. Apply an inverse-normal transform to points strictly inside `(0,1)`
and use independent scrambles for uncertainty. Keep Philox as the reference and fallback;
high-dimensional path QMC and boundary reflection are not guaranteed wins.

### 3.3 Use common random numbers for comparisons and optimization

Reuse the same physical samples when comparing B1, wall thickness, water inclusion, or
sequence parameters. The IRSE path already reuses a phantom across inversion times; preserve
that useful covariance. For estimates A and B,
`Var(A−B) = Var(A)+Var(B)−2Cov(A,B)`, so positive coupling improves the difference directly.

Use stable candidate-domain points and change membership/weights when geometry changes;
regenerating bounding boxes or dense/sparse groups can destroy the pairing. Report the paired
difference with replicate-level uncertainty. Freeze randomness for smooth optimizer steps,
then evaluate on held-out seeds to detect fitting to numerical noise. Do not treat sample
membership gradients through hard geometric masks as ordinary pathwise derivatives.

Antithetic points `(u,1−u)` or Brownian increments `(Z,−Z)` are a cheap experiment, but not a
default guarantee. For a symmetric Gaussian phase, paired phasors cancel the imaginary part
while duplicating the same cosine real part; at fixed simulation count this need not improve
a real-valued target. Compare variance **times cost**, including the second path.

### 3.4 Allocate by the uncertainty of the actual target

The current dense-near-wall/quarter-density-far-away rule is sensible but blind to echo,
slice excitation, receive sensitivity, and the reconstruction PSF. A small amount of water
outside an ROI can matter disproportionately at a short-T2 late echo.

For independent strata with volume/weight W_h, within-stratum target SD σ_h, per-sample cost
c_h, and counts N_h, ordinary stratified MC has variance `Σ W_h² σ_h²/N_h`. Minimizing that
under a fixed cost gives

\[
N_h\propto W_h\sigma_h/\sqrt{c_h}.
\]

Estimate these quantities in a separate pilot, round production counts to valid QMC sizes,
and keep a nonzero allocation floor. For QMC, fit a local variance law `A_h N_h^(−α_h)` over
two or three densities rather than assuming the IID exponent. If α is common across strata,
the corresponding allocation is proportional to `(α A_h/c_h)^(1/(α+1))`.

Optimize a declared target: a set of complex ROI statistics, echo curves, or a worst-case
image error, not whichever metric happens to look good in the pilot. Account for cross-stratum
covariance if retaining shared randomizations. For complex linear ROI metrics, the adjoint
reconstruction weights can help identify where each spin's k-space contribution matters.

Changing counts with correct weights preserves the fixed domain's integral. Setting weights
to zero outside a smaller water margin does not; that is a separate truncation approximation.
Use independent production samples or an appropriately justified adaptive estimator. Do not
reuse an aggressively selected pilot and then apply fixed-sample confidence formulas blindly.

### 3.5 Reduce geometric and frequency dimensions before adding spins

Several promising options fit the existing isochromat engine:

- **Boundary-aware cells:** split sphere/water boundary pixels and sample smooth interiors
  separately. Where practical, parameterize the allowed slice interval conditional on in-plane
  coordinates, carrying its length as a weight. This removes a rejection dimension without
  renormalizing away physical volume.
- **Anisotropic stratification:** put more cells along slice selection or the dominant phase
  gradient. Existing tuple-valued `spins_per_voxel` already supports this. Derive the need from
  retained coherence pathways, not only the acquired k-space maximum: strong crushers and
  stimulated echoes retain higher spatial orders.
- **Analytic linear-phase integration:** in a rectangular cell with approximately constant
  amplitude and linear phase, integrate that component with a sinc product. Estimate only the
  residual by MC. This is an excellent local control variate; blindly applying a voxel sinc
  after a selective multipath pulse train is not generally valid.
- **T2′ marginalization:** a simple FID has an analytic Lorentzian characteristic function.
  A known coherence pathway uses its signed off-resonance evolution time. Arbitrary selective
  RF and pathway mixtures do not reduce to multiplying the whole signal by `exp(-TE/T2′)`.
  For those, test randomized CDF stratification, tailored quadrature over the relevant time
  window, or a deterministic-frequency/control-variate hybrid.
- **Tissue marginalization:** use weighted `per_tissue` components where rare tissues or high
  tissue contrast matter. `categorical` is a useful memory/speed option, but introduces label
  noise; compare equal wall-time budgets, because full marginalization can multiply spin cost.

### 3.6 Use cheap models as control variates, not substitutes for physics

For full contribution f(X) and cheap correlated contribution g(X) with known integral G,

\[
\widehat I_f=\beta G+\frac1N\sum_i[f(X_i)-\beta g(X_i)].
\]

The residual can have much lower variance. Estimate β on an independent pilot or cross-fit
it. If G is itself estimated, include its uncertainty; treating an approximate deterministic
G as exact can introduce bias.

Candidates include a simple spin-echo/EPG response, a coarse RF calculation with the same
positions, a uniform-field tissue response, a local linear-phase approximation, and free
diffusion as a control for weak restriction. Integrate controls through the same complex
reconstruction and geometry as the full signal. An accurate ROI-mean curve alone is not a
per-spin control with a known integral.

For known G and optimal scalar β, ideal variance reduction is `1/(1−ρ²)`. If evaluating g
costs 10% of f, the illustrative equal-accuracy speedups are about 4.8× at ρ=0.9 and 1.8× at
ρ=0.7. These are planning calculations, not Bloch Party benchmark results; pilot cost and
estimating G reduce them.

A multilevel version telescopes RF/time resolutions:

\[
E[S_L]=E[S_0]+\sum_{\ell=1}^L E[S_\ell-S_{\ell-1}].
\]

Use many cheap coarse samples and fewer expensive paired corrections. Couple static spatial
positions exactly; couple Brownian increments explicitly for diffusion. The final level's
discretization bias still remains. Current `coarsen_rf` is a plausible source of levels for
compatible NIST sequences, but its weighted RF averages are not exact pulse propagators and
its handling of newer per-segment fields must be audited before broader reuse. Preserve RF,
ADC and spoil boundaries. Apply the telescope in complex signal space before magnitude.
The method and cost logic are established in
[Giles' multilevel Monte Carlo work](https://people.maths.ox.ac.uk/~gilesm/mlmc.html);
its usefulness here depends on measured correction variance decay.

## 4. Runtime work that complements better estimators

### Restricted-diffusion scheduling

`tau_max` returns `min(size²/(200D))` over all spins. A single tiny, rapidly diffusing
compartment can dictate substeps for the entire phantom. At equal D, mixing 1 µm and 10 µm
compartments creates a 100× step-limit ratio. That is not a 100× overall speedup promise,
but it is a clear reason to bucket spins by `size²/D`, geometry, and physics flags.

Separate batches reduce GPU divergence and avoid over-refining easy walkers. Preserve global
IDs and sum weighted complex signals before reconstruction. Changing the reference's global
step count changes the realized paths, so validate refined distributions, not just replay.
Distance-to-wall adaptive stepping is a later experiment: a straight segment that does not
cross a wall does not imply the Brownian bridge avoided it. It needs a defensible crossing
treatment, not merely a larger step far from an apparent endpoint collision.

### Setup, reductions, and precision

The Metal engine already buffers ADC reductions, uses SIMD-group sums or matrix products for
many coils, keeps spin state in registers across segment chunks, and specializes physics.
The project has historical evidence that ADC reduction dominated some workloads. Preserve
these strengths and profile by workload before spending effort on Gaussian generation alone.

`simulate_metal` repacks inputs, allocates/uploads buffers, synchronizes, and copies both
signal and final magnetization back to CPU on every call. A reusable prepared-sequence/phantom
object and optional signal-only return could save work across NIST seeds and parameter sweeps.
Cache only immutable inputs and make invalidation explicit. Reuse RF/gradient structure and
group replicas where GPU occupancy permits; measure memory pressure as well as throughput.

Atomic float32 signal accumulation has order-dependent rounding. At destructive-interference
nulls, absolute error relative to the fully coherent signal is more informative than relative
error to a tiny result. Benchmark repeated identical-seed GPU runs separately from different-
seed runs; compare identical sampled phantoms against the float64 reference. A deterministic
two-stage reduction or compensated partial sum is worth testing if that floor becomes visible.
Accumulate independent seed outputs in complex128 on CPU; this helps final averaging but
does not repair errors already accumulated on the GPU.

For diffusion, consider storing displacement relative to a compartment/voxel anchor instead
of repeatedly adding micrometer motion to macroscopic float32 coordinates. Evaluate gradient
phase as anchor plus local displacement where possible. This is a precision proposal requiring
translation-invariance tests, not a newly measured failure.

The reference currently constructs NumPy Gaussian arrays then converts them to tensors per
segment, and materializes `(spins, segments, 3)` effective positions for merged walking runs.
Batching or bounded streaming could reduce its overhead and memory. Avoid turning the
reference into a second opaque performance engine; the primary production investment should
remain Metal, as the repository specifies.

### Noise replicas and gradients

`design.image_noise_sigma` runs 16 reconstruction replicas by default. Report convergence of
the variance estimate, especially for nonlinear RSS at low SNR. For a linear reconstruction
operator A and known covariance Σ, direct propagation `AΣA*` can eliminate replicas when
computationally feasible. Retain replicas for nonlinear reconstruction and validate them
against the linear result in the special case.

Treat anatomical random-field variability as model variability and receiver noise as physical
noise; neither should be conflated with numerical sample error. Use separate seed namespaces.
For differentiable design, validate gradient convergence as well as signal convergence. Fixed
noise helps finite differences and optimizer stability, but hard rejection boundaries,
restriction branches, and changes in integer substep count require special care.

## 5. A concrete validation and implementation sequence

### Stage A — make estimates auditable

1. Save per-seed complex outputs and metadata: sample identity, weights, domain, RF coarsening,
   dtype, backend, sequence hash, model parameters, and runtime.
2. Add minimum-replicate and sampling-parameter guards. Correct the aggregate jackknife and
   paired-difference reporting. Keep the current magnitude result available with honest naming.
3. Add synthetic calibration cases for nulls, weak signals, correlated ROIs, and T2′. Validate
   confidence-interval coverage over repeated independent *experiments*, not merely one seed set.

Acceptance: exact power identities hold numerically; aggregate uncertainty includes covariance;
known estimator biases are quantified; invalid counts fail clearly. Historical results remain
historical until recomputed from retained images or new runs.

### Stage B — compare samplers at equal cost

Use IID, current stratification, current QMC, digital-shift QMC, and pilot allocation on:

| Workload | Independent target | Main failure it probes |
|---|---|---|
| Uniform FID with T2′ | Lorentzian characteristic function | Frequency quadrature bias invisible to seeds |
| Cube/sphere Fourier response | Analytic transforms | Oscillation, geometric discontinuity, weights |
| Free isotropic/anisotropic PGSE | Stejskal–Tanner | Phase moments and diffusion attenuation |
| Imperfect-refocusing echo train | Existing independent EPG diffusion calculation | Multiple pathway histories |
| Restricted finite-width gradients | PDE/spectral solution plus equilibrium checks | Boundary and phase time-step bias |
| NIST IRSE around null; TSE late echoes with water | High-accuracy numerical reference, then scanner comparison | End-to-end estimator and model separation |

For static imaging, sweep powers-of-two N such as 64, 256 and 1024, with 8–16 independently
randomized replicates during calibration. Sweep RF resolution, water margin, slab extent and
restricted step size separately. Avoid two commensurate deterministic grids as the only
convergence evidence. Reuse paired samples when comparing settings.

Report complex bias/RMSE, downstream ROI error, clipping fraction, empirical interval coverage,
peak memory, setup time, warmed simulation time, reconstruction time, and total wall time.
Warm up compilation before timing, synchronize GPU work, and include pilot and control-model
cost in operational comparisons. Use accuracy-versus-time curves; spin-segments/s alone can
reward a method that performs unnecessary work.

### Stage C — implement the first variance-reduction path

Prototype bridge conditioning in the readable reference with explicit eligibility checks,
validate it independently, then implement the Metal counterpart. Next test exact free-interval
Gaussian aggregation. Keep existing defaults and raw-walk behavior available. For any engine
change, follow the repository's snapshot save/compare rule.

Alongside these experiments, add persistent IDs, prepared buffers, sparse-point streaming,
and restricted-walker buckets. These infrastructure
changes make future sampling experiments easier to trust.

### Stage D — fund larger changes with pilot evidence

Implement control variates/multilevel corrections only after measuring both correlation and
cost on a representative protocol. Advance a prototype if it reduces total runtime at a fixed
validated accuracy target without worsening memory or unsupported-physics behavior.
A reasonable initial engineering target is a 2× reduction at equal RMSE; this is a proposed
acceptance target, not a forecast.

Define a numerical error budget before optimizing. For example, for a 0.1-percentage-point
benchmark target, choose a smaller MC uncertainty allowance and separately bound RF, domain,
and reduction errors. Near nulls use an absolute scale tied to initial/peak signal. Do not add
deterministic biases in quadrature unless a probabilistic justification exists. Use a pilot
to select a fixed production budget; repeated unadjusted stopping on an SE threshold does not
automatically preserve nominal coverage.

## 6. Checks performed and limitations

- Ran all five diagnostic experiments in `validation/monte_carlo_audit.py`, including the
  current production magnitude estimator and sampler. The JSON records full numerical results.
- Ran targeted existing diffusion, restriction, and sub-voxel tests. The initial run had
  **21 passed, 6 skipped**, and one fused test blocked by the default compiler-cache location.
  The fused case passed on retry with its cache in a writable temporary directory: **22 passed
  in total, 6 skipped**. The six skips are Apple-GPU-only cases.
- Inspected the implementations and existing independent checks, including Philox known-answer
  vectors, Stejskal–Tanner, anisotropic tensors, EPG echoes, restriction equilibrium, and
  short-gradient-pulse limits.
- Did not rerun the full real-scanner datasets or measure GPU speed. Did not modify simulation
  algorithms, defaults, or published historical benchmark values.

The recommended order is deliberate: first make the error estimate reflect what the code
actually computes, then spend fewer expensive samples to meet that measured error target.
