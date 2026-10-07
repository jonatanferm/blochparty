# Lessons learned

A running notebook of things we found out the hard way: physics and numerics that bite, quirks
of formats, data and other tools, performance facts, and process. Add to it whenever something
costs more than a few minutes to figure out. Newest entries go at the top of each section; each
says where it came from so it can be re-checked.

## Physics and numerics

- **Speckle biases magnitude images upward, worst where the signal is small.** Spins are a
  Monte Carlo sample of the continuum; the complex image's MC noise is ~zero-mean, but
  E|s + n| ≈ |s| + σ²/2|s|. A water fill simulated at sphere-level density (64 spins/pixel)
  made short-T2 spheres' late echoes 2–4× too high (benchmark 2: 2.6 % → 5.8 %); the scanner's
  tails matched the simulation *without* water. Fixes: more spins, independent seeds with the
  bias σ² removed (`nist.mc_magnitude`), lower-discrepancy placement (`nist.*_qmc`). Ideal
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

## Process and tooling

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
