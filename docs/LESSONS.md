# Lessons learned

A running notebook of things we found out the hard way: physics and numerics that bite, quirks
of formats, data and other tools, performance facts, and process. Add to it whenever something
costs more than a few minutes to figure out. Newest entries go at the top of each section; each
says where it came from so it can be re-checked.

## Physics and numerics

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

## Process and tooling

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
