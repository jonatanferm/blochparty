# Bloch Party roadmap

Rewritten 2026-10-08. The phase-by-phase record of what was built and verified before this date
is in [docs/ROADMAP_ARCHIVE.md](docs/ROADMAP_ARCHIVE.md); results against real scanners are in
[validation/real/RESULTS.md](validation/real/RESULTS.md). This file is the plan.

## Vision

The best MR simulator we can build, with the most faithful physics we can muster, judged against
real scanners. Coupled to it, a generator of digital anatomies to run in it. Those two together
are the product; they will later generate synthetic training data for ML models and support
agentic development of MR sequences. A GUI to play with scans is a possible bonus, not a goal.

Order of priorities, now:

1. **The simulator core**, until we are satisfied with it (defined below).
2. **Anatomy generation**, so the core has realistic objects to scan.
3. Training-data generation and agentic sequence development, built on 1 and 2.

We work on one Apple-silicon Mac for now. The Apple GPU kernel is the engine; ports to other
hardware come when other machines exist (M8).

## When is the core satisfactory?

Three conditions, each testable. The core is done when all three hold.

1. **Parity.** The Metal kernel runs everything the PyTorch core runs (since M1a, free pools
   too; since M1b, moving spins under the field model; since M1c, free pools with motion,
   diffusion and `FieldMaps`), with one exception: free pools together with the generalized
   Bloch bound pool, which only the core runs (`docs/ENGINES.md`, M1c).
2. **Gradients.** The Metal kernel has an adjoint for the physics that calibration and sequence
   design need (M2). Without it, "differentiable" holds only for the CPU reference and the
   frozen fused kernel.
3. **Physics no longer moves the benchmark table.** On at least one scanner with two sequences
   (NIST benchmarks 1 and 2), the remaining errors are attributable to unknown inputs (B1+,
   coil sensitivities, temperature, geometry) rather than to missing physics. The test is the
   digital-twin loop (M3): fit the unknown inputs on one sequence, predict the other, and show
   that adding physics changes the prediction less than the error bars.

## Engines

| engine | role | gradients | policy |
|---|---|---|---|
| `blochparty.simulate` (`src/blochparty/bloch.py`) | readable reference; every new effect lands here first, checked on small cases | autograd, `simulate_chunked` | keep small and readable; float64 |
| `blochparty.metal.simulate_metal` (`src/blochparty/metal/__init__.py`) | the engine; all imaging-scale work | none yet (M2) | must agree with the core; float32 |
| `blochparty.fused.simulate_fused` (`src/blochparty/fused/`) | frozen CPU kernel with a hand-written adjoint | adjoint | no new physics (it raises); retired once M2 is verified (see M2) |

## Working rules for whoever implements this

The rules in `CLAUDE.md` apply. Concretely, for every item below:

- **Independent check first.** Name the reference before writing code: a closed form, the SciPy
  ODE (`tests/test_bloch_vs_ode.py: random_case`), EPG (`src/blochparty/epg.py`), the signal
  equation (`src/blochparty/reference.py`), BMCTool (`validation/compare_bmctool.py`), MRzero
  (`validation/compare_mrzero.py`), or real data (`validation/real/`). A feature without a
  reference is not done.
- **Defaults stay bit-identical.** Before the change: `uv run python tools/snapshot.py save
  /tmp/before.pt`; after: `save /tmp/after.pt` and `compare`. New physics is opt-in (a `None`
  default on `Phantom`, `Sequence`, `Scanner` or a builder keyword) and must not change the
  snapshot. If an item needs the snapshot to grow (new engine, new default inputs), add to
  `tools/snapshot.py` in a separate commit and say so in the journal.
- **Core, then Metal, then parity test.** Physics goes into `bloch.py` with a test against the
  reference, then into the Metal shader, then a test in `tests/test_metal.py` comparing the two
  on the same inputs. Float32 tolerance for parity is ~1e-5 relative on signals (what the
  existing tests use); say which tolerance and why.
- **Metal variants.** The shader is compiled in three variants of `advance<V, DIFF>` (plain,
  `_full` for the rate-model bound pool and the field model, `_gen` for generalized Bloch), each
  with and without diffusion/motion (`_diff`). Per-thread state that only one feature needs goes
  into its own variant; the plain kernels lost 10 to 30 % to register pressure when state was
  shared. Measure the plain variant's speed before and after any shader change
  (`benchmarks/bench.py`, record with `--record`).
- **Fused kernel raises.** New `Phantom` or `Sequence` fields must make `simulate_fused` raise
  `NotImplementedError` with a message naming where the feature lives (pattern at
  `src/blochparty/fused/__init__.py:263`).
- **Tests run on this Mac.** `uv run pytest` (about 3.5 min with GPU access). Mark anything over
  ~30 s with a marker and keep the default suite under 5 min. GPU tests skip without MPS, so
  check `blochparty.metal.available()` and do not let a skipped test pass for a GPU change.
- **Record.** `docs/LESSONS.md` for anything that took more than a few minutes to understand.
  `docs/JOURNAL.md`, dated, after each session: what, why, numbers, dead ends, commit hashes.
  Results tables live in the `RESULTS.md` next to the script that produced them; the roadmap
  gets one line and a link.
- **Scripts never overwrite tracked outputs by default** (figures only with `--figure`).

---

## M0. Consolidate (about one day) — done 2026-10-08

Goal: a stranger (or a fresh agent) can tell which engine does what, and the docs match the code.

- [x] `docs/ENGINES.md`: capability matrix of the four entry points (core, fused, Metal,
      chunked) × 31 features and combinations, forward and gradients, precision per engine, and
      the measured agreement (fused 4e-14, Metal 2e-5 static / 7e-5 signal with walking spins).
- [x] `tests/test_engines.py` parses the matrix and checks all 248 cells: ✓ = parity with the
      core (forward; gradients vs the core's autograd, the core's autograd vs finite
      differences), ✗ = `NotImplementedError`. It found two silent cases, now raising: the core
      ignored `maps` with free pools, Metal ran diffusion with exact concomitant fields (no
      reference). Metal now raises when an input requires grad instead of returning detached
      results.
- [x] README: engine section matching the matrix; commands; test count.
- [x] `benchmarks/bench.py --backend metal` on the four cases, recorded in
      `benchmarks/history.jsonl`.
- [x] `slow` marker (21 tests over ~1 s, and the 4 crosscheck tests); `uv run pytest -m "not slow"`
      runs 714 tests in 45 s.

Verify: suite green; snapshot identical; the matrix agrees with the raises.

## M1. Metal parity (3 to 4 days)

### M1a. Free exchanging pools on the GPU — done 2026-10-08

Done, with a different design than planned below: a float32 matrix exponential per segment was
1e-2 off over the 2000 segments of a CEST train, so the kernel applies exp(A dt) to the state by a
Taylor series on sub-steps (no matrix), common precession of free segments factored out exactly,
increments added by two-sum; one kernel per pool count (`_bm1` .. `_bm5`), spins sorted by cost.
Agreement with the core ≤ 1e-6 on the tests, ≤ 3.5e-7 in Z on benchmark 9, 8.3e-6 on benchmark 8,
BMCTool 8.3e-7 (5 pools) and 6.5e-7 (MT); benchmarks 8 and 9 reproduce their tables; benchmark 9's
fit forward 141 s → 5.7 s (`validation/real/RESULTS.md`). Found on the way and fixed: every Metal
variant drifted 1e-2 from the core over 1e5 RF segments (float32 rounding of the whole state each
segment), now ≤ 3e-7. Left: free segments with fast exchange cost sub-steps in proportion to
k dt (benchmark 9's readout, k = 40 kHz over 5.2 ms TRs: ~280); an exact propagator per spin,
cached on dt (free segments only depend on dt once the precession is factored out), would make them
one small matrix-vector product.

The original plan:

The core (`bloch._simulate_exchange`) builds one augmented Bloch–McConnell matrix per segment
(`exchange._bm_matrix`, size d = 3(P+1) + nb + 1 for P free pools and nb ∈ {0, 1} bound pools)
and applies `matrix_exp(A dt)` to the state. Reproduce that on the GPU.

- Design: a fourth `advance` variant `_bm` (its own state: the (d−1)-vector per spin, d ≤ 14,
  i.e. up to three free pools plus a rate-model bound pool; raise above that). Per segment and
  spin: build A·dt in registers or threadgroup memory, exponentiate by scaling-and-squaring with
  a degree-6 Taylor or Padé(3,3) in float32 (choose by testing against `torch.linalg.matrix_exp`
  in float64 on random A·dt with norms up to those a 100 µs RF segment produces at B1 = 20 µT;
  target 1e-6 relative), apply, then sample Σ pools' Mxy at ADC segments, spoil by zeroing the
  transverse rows. Cache the exponential across consecutive identical segments per thread (the
  core keys on dt, rf, grad, b0_offset, rf_freq); CEST saturation trains are long runs of
  identical segments, so this is where the time goes.
- The bound pool's saturation rate during RF (`exchange.saturation_rate`) uses the lineshape
  table already in the `_full` variant; reuse it.
- `simulate_metal` returns `SimResult` with `m_pools` as the core does.
- Benchmark on `validation/real/iohexol.py` (set `BLOCHPARTY_BACKEND=metal`; today its forward
  of 58 offsets × 8 tubes × parameter sets takes 15 s in the core and a Levenberg–Marquardt
  calibration about 7 min). Report core vs Metal time in `RESULTS.md`.

Verify: `tests/test_metal.py`: two-pool and three-pool phantoms with and without a bound pool on
a random RF / gradient / off-resonance sequence and on a CEST saturation + readout, vs the core
≤ 1e-5 relative; the BMCTool comparison (`validation/compare_bmctool.py`) through Metal within
the float32 budget (the core is at 5.8e-7); benchmarks 8 and 9 reproduce their tables.

### M1b. Motion under gradient nonlinearity and concomitant fields — done 2026-10-08

Done as planned, diffusion included: moving spins under the field model live in true coordinates
(motion and walks), and every segment maps them to encoding positions and concomitant
coefficients (core: `bloch._moving_fields`; Metal: `nl_displacement`, `conc_at` in the `_diff`
kernels). Static spins untouched (snapshot bit-identical). A moving spin equals the chain of static
one-segment simulations at each segment's position to 4e-15 (core) and 1e-5 per spin (Metal);
in a Maxwell-compensated echo a moving spin keeps the exact path integral of Bernstein's field,
the midpoint error falling 4x per halving (`tests/test_fields_concomitant.py`). Also: static
`FieldMaps` are now sampled at true positions (they were at encoding positions under a
nonlinearity). Metal's float32 resolves a segment's phase to |φ| 2⁻²⁴ (4e-4 rad for a 5 ms,
32 mT/m lobe at 15 cm): see `docs/ENGINES.md`.

The original plan:

Both engines raise today (`bloch.py:483`, `metal/__init__.py:879`). With motion, the encoding
position f(r(t)) and the concomitant coefficients (quadratic in the position) must be
re-evaluated per segment from the moved position; the free-run closed form then has a
time-varying k·r, which is what the motion path already handles for ideal gradients.

- Core: in `_free_run` / `_rf_segment`, evaluate `fields.spin_fields` terms at the moved
  position per segment (the per-spin `conc` and encoding offset become per segment). Keep the
  static path untouched (snapshot).
- Metal: the `_diff` variants already carry the moved position; add the nonlinearity basis
  evaluation (`fields.GradientNonlinearity`, solid harmonics to the file's order) and the
  concomitant quadratic form per segment under a flag; only when `moving && (mode || nonlin)`.

- Diffusion under the field model today keeps the concomitant coefficients of the start position
  and walks in encoding coordinates (negligible for µm walks), and raises with exact concomitant
  fields; evaluate both per segment at the walked position too, so that row of
  `docs/ENGINES.md` turns ✓.

Verify: a spin moving at constant velocity through a nonlinear field equals a static spin placed
at each segment's position (sequence of one-segment simulations with `m_init`) to 1e-10 (core)
and 1e-5 (Metal); a Maxwell-compensated spin echo stays compensated for a static spin and
acquires Bernstein's velocity-dependent phase for a moving one.

### M1c. Leftovers — done 2026-10-08 but for one Metal item

- [x] Free pools with the generalized Bloch bound pool, **core** (2026-10-08): the modes are
      2J rows of the exact matrix; equal to a hand-written SciPy ODE of the mode equations to
      2e-8 (Lorentzian and super-Lorentzian, with two free pools and the field model off), to the
      generalized split engine with inert pools within the split's accuracy, and to the rate
      model under long off-resonant CW (2e-3). Found: the split engine itself is ~2e-4 off in
      Mzb on a Lorentzian random case (sub-steps ≤ T2b / 2 whatever the segment length); the
      exact engine matches the ODE to 1e-13 there.
- [ ] The same on **Metal** (raises): 2J ≈ 60 extra components per Taylor vector (super-Lorentzian
      J = 30) is beyond the register budget of the `_bm` kernels; either keep the modes in
      thread memory as the `_gen` variant does (slow, but this combination is rare), or decide it
      is not needed. Until then the parity condition has this one exception.
- [x] Free pools with diffusion or motion (2026-10-08): positions, walks and the field model per
      segment in the exact engine, core and Metal (`_bm*_diff` kernels; spins not cost-sorted
      when moving: walks are keyed by spin index). Inert pools equal the single-pool path with the
      same motion or walk to 3e-7; a constant rigid shift equals the moved phantom to 1e-13.
- [x] Free pools with `FieldMaps` (2026-10-08), static and moving, core and Metal: equal to the
      same fields given per spin to 1e-14.

## M2. Metal adjoint (5 to 7 days) — first version done 2026-10-09; speed and the fused kernel's retirement open

Status. `simulate_metal` is differentiable for the planned scope (sequence `dt`, `rf`, `grad`,
`adc_phase`, `b0_offset`, `rf_freq`; phantom `pos`, `pd`, `t1`, `t2`, `db0`, `b1`; coils; `m_init`;
the rate-model bound pool; the scanner field model through the encoding positions and coefficients;
static `FieldMaps`), raising for moving spins, free pools and the generalized bound pool. Checks:
(1) against the core's float64 autograd per input, 1e-7 to 3e-6 on random sequences, 2e-5 on a
1600-segment RF-spoiled GRE, ≤ 1.2e-4 with the bound pool; (2) as accurate as the fused float32
adjoint or better (bound-pool `rf_freq`: 1.2e-4 against its 1.5e-3); (3) bit-identical for any
`checkpoint_every` and every run; (4) the WM/GM optimum to 0.001° and the Ernst angle to 0.01°, but
the DESPOT1 pair only to 0.05°: its bound differentiates float32 signals by finite differences
(`rel_step` 1e-2; the default 1e-4 lands degrees off); (5) forward + backward 6-7x the forward on
3d-48 / 3d-64 (target 5x; 3-4x on the small cases). Not done: the speed target; retiring the fused
kernel, which still is the only fast float64 path with gradients (DESPOT1 to 0.001° needs float64):
a decision, see the journal.

The original plan:

Goal: `simulate_metal` differentiable for the inputs the digital twin and sequence design need,
at imaging scale, verified against the fused adjoint and the core.

Scope (first version): sequence `dt`, `rf`, `grad`, `adc_phase`, `b0_offset`; phantom `pos`,
`pd`, `t1`, `t2`, `db0`, `b1`; complex `coils`; `m_init`; the rate-model bound pool
(`f`, `k`, `t1`, `t2`); the scanner field model's `field_strength` and nonlinearity coefficients.
Out of scope, raising clearly: diffusion, motion, generalized Bloch, free pools, `FieldMaps`
sampled per segment. Losses on `signal` and on `m` (both, as the fused adjoint).

Design, following the fused adjoint (`src/blochparty/fused/kernel.h`, "Adjoint (reverse mode)"
at line 794 and `reverse_mode` / `reverse_bound`): the reverse sweep
λ_{s−1} = A_sᵀ (P_s λ_s + ADC term), with A_s the segment's linear map (rotation, relaxation,
precession) and P_s the spoil projector; parameter gradients from λ_s and m_{s−1}.

- **Checkpoints, two levels.** Forward stores m before every K₁ segments in a device buffer
  (N × S/K₁ × 4 floats). Backward, per coarse interval in reverse order: replay K₁ segments
  forward storing every state in a per-spin device buffer of K₁ states, then sweep backward
  over them. Memory N × (S/K₁ + K₁) × 16 B. For benchmark 2 (85 k spins, 4.1 M segments),
  K₁ = 2048 gives about 5.5 GB; expose K₁ as `checkpoint_every`. Same structure as the fused
  kernel, so results must not depend on K₁ (test).
- **Per-spin gradients** (pos, pd, t1, t2, db0, b1, coils, m_init) accumulate in per-thread
  registers / device memory with no reduction. Coil gradients: dL/dsens_c = Σ_samples h_c·m,
  per spin and coil, as `reverse_mode` does.
- **Per-segment gradients** (dt, rf, grad, adc_phase, b0_offset) need a sum over spins per
  segment. Reduce within the threadgroup (simd_sum, then threadgroup memory), write one partial
  per threadgroup per segment to a buffer (groups × chunk × 9 floats), sum the partials in torch
  in a fixed order. No float atomics: results must be deterministic run to run (test).
- **Dispatch** mirrors the forward: one kernel launch per coarse interval; the ADC term needs
  h = dL/d(raw coil sums) per sample, passed in as a (n_adc × coils × 2) buffer, computed by
  autograd in torch from the loss (`torch.autograd.Function`: forward calls the Metal forward
  and saves the checkpoints; backward receives grad of `signal` and `m`, runs the sweep,
  returns gradients for every tensor input).
- **The bound pool's exchange step** has its own reverse (`ZStep::reverse` in kernel.h); port it
  with the `_full` variant. Keep the plain variant free of pool state.
- **Rodrigues reverse** (`rodrigues_reverse`, kernel.h:844) ports directly; the `expm1`
  relaxation note at kernel.h:1217 applies in float32 even more.

Verify, in this order:
1. Tiny cases vs the core in float64 autograd (`tests/test_fused_grad.py` has the case list:
   random RF / gradient / off-resonance, 8-coil 2D GRE, 3D GRE with dB0, spoils and ADC right
   after RF, `m_init`, losses on signal and/or m). Float32 adjoint vs float64 autograd: expect
   1e-4 relative on well-conditioned gradients; report per input.
2. Vs the fused adjoint in float32 on the same cases (both float32: expect 1e-5 relative), and
   on the bound-pool and field-model cases.
3. Identical gradients for any `checkpoint_every`; identical across two runs (determinism).
4. Gradient descent through it recovers the Ernst angle and the WM/GM optimum (16.109°) to 0.01°
   (`examples/optimize_contrast.py`: add a `--metal` switch next to its `--fused`), and the
   DESPOT1 CRB pair (`examples/design_despot1.py`, which uses the fused adjoint today).
5. `benchmarks/bench.py --grad --backend metal` on the standard cases; record. Target: forward +
   backward within 5× the forward (the fused adjoint is 4 to 7×).

Then retire the fused kernel: once 1 to 4 pass, move `src/blochparty/fused/` out of the package
(`git rm`; the archive and git history keep it), drop its tests, remove it from
`tools/snapshot.py` in a separate commit that re-saves the snapshot and explains why, and update
`docs/ENGINES.md`, README and `CLAUDE.md`. Until then it stays frozen.

## M3. The digital-twin loop on the NIST data (3 to 5 days)

Goal: condition 3 above. Benchmarks 1 (IR-SE, T1 plane) and 2 (23-echo TSE, T2 plane) are the
same phantom on the same Prisma with the same 20-channel coil (Tong et al. 2022). Today both are
compared by shape (one free scale per sphere) and both stop at a list of suspects. Close the loop.

- [ ] **Absolute metric.** Add to `validation/real/nist.py` an absolute comparison: measured vs
      simulated complex coil images (or at least magnitude after the same coil combination),
      with receive sensitivities estimated from the data's own M0-like image (smooth, per coil,
      e.g. low-order polynomial fit of each coil image divided by the RSS inside the phantom)
      and used in the simulation as `FieldMaps.coils`. Report both metrics.
- [ ] **Fit on one plane, predict the other.** Parameters: a low-order B1+ map (solid harmonics
      to order 2, 9 parameters, `b0_shim`-style basis), the receive maps as above (not fitted
      through the simulator, estimated from data), the fill-fluid geometry (sphere positions
      along the slice, wall thickness), the first refocusing interval's timing shift (one scalar,
      benchmark 2's prime suspect). Fit by gradient descent through the Metal adjoint (M2) or,
      if M2 is not ready, through the fused adjoint overnight (its forward on benchmark 2 is
      232 s; a gradient step about 20 min). Reference T1 / T2 are held out: never fitted.
- [ ] **Prediction with error bars.** Predict the other plane with the fitted scanner model.
      Error bars by jackknife over Monte Carlo seeds (`nist.mc_errors`, which keeps covariance).
- [ ] **Ablation table.** With the fitted inputs fixed, switch physics off one at a time (slice
      profile, crushers' dephasing, incidental MT of the sinc pulses with a literature bound
      pool for NiCl₂/MnCl₂ water, which should be ~0, fill fluid, concomitant fields, measured
      GIRF of a Prisma from benchmark 7 as a proxy) and report the change in prediction error
      against the error bar. Condition 3 holds when every row is within the bar or explained.
- [ ] Write it up in `validation/real/RESULTS.md` ("Digital twin") and in `docs/REALISM.md` R7.

Verify: the prediction on the held-out plane beats the nothing-fitted number (1.49 % / 2.25 %
shape; absolute: establish the baseline first) and the twin's parameters are physically plausible
(B1+ within ±20 % at 3 T, timing shift below 100 µs). If the short-T2 spheres 11 to 14 stay at
6 to 9 % with B1+ and timing fitted, that is a physics result: record which physics is missing.

## M4. Pulseq out (1 to 2 days)

Goal: anything built or optimized here runs on a scanner, and the round trip is lossless. This is
the precondition for sequence development and for a linter that talks about Pulseq blocks.

- [ ] Block provenance. `SequenceBuilder.block` records the segment span of every block in
      `seq.meta["blocks"]`; `read_pulseq` records the source block index per segment. Transforms
      (`subdivide_rf`, `apply_girf`, drift) carry `meta` through.
- [ ] `pulseq.write_pulseq(seq, path, system=None)` via pypulseq: for sequences with block
      provenance emit one Pulseq block per recorded block (RF shapes on the RF raster, trapezoids
      where the segments form one, arbitrary gradients otherwise, ADC with the recorded dwell and
      phase); for flat sequences without provenance, emit segment runs between RF and ADC
      boundaries as arbitrary-gradient blocks. Ideal spoil flags cannot be written: raise unless
      `spoil="drop"` is passed, and say so in the docstring.
- [ ] Round-trip tests (`tests/test_pulseq.py`): write → `read_pulseq` → simulate equals the
      original to 1e-9 on GRE (ramps, slice select, RF spoiling), TSE, bSSFP, EPI and an imported
      scanner `.seq` from `validation/real/data`; pypulseq's `check_timing` passes; k-space at every
      ADC sample equals pypulseq's `calculate_kspace` for the written file.

## M5. Anatomy generation v0 and the dataset writer (5 to 8 days)

Goal: a sampler that produces a new, physically consistent head phantom on every call, carrying
every input the engine can use, and a writer that stores what was simulated with full
provenance. This is P0 and P1 of [docs/PHANTOMS.md](docs/PHANTOMS.md) §9 and the start of the
anatomy pillar. The learned parts (P4, P5) come after M6.

- [ ] **Schema (P0).** A `DigitalPhantom` container above `Phantom`: label fractions on an
      anatomy grid (what `anatomy.Anatomy` holds), per-tissue property distributions
      (`anatomy.TissueProperties`), field-strength-dependent tables, and the inputs today's
      engine needs but the tissue table lacks: bound-pool fraction / exchange rate / T2b per
      tissue (literature qMT values at 3 T for WM, GM, CSF, muscle, fat; cite each as the tissue
      table does), diffusivity per tissue, fat fraction. `to_phantom` fills `Phantom.bound`,
      `diffusion` and `fat_fraction` from them, opt-in. Round trip with today's maps bit-identical.
- [ ] **Procedural sampler (P1).** `anatomy.sample(seed, *, subject=None, field_T, voxel)`:
      choose a BrainWeb subject (the 20-subject set is loaded and pinned), apply a random smooth
      diffeomorphic deformation (B-spline displacement field with a magnitude prior; check the
      Jacobian stays positive), sample tissue properties from the cited distributions, apply the
      within-tissue texture (`variability`, exists), compute χ → B0 and shim (exists), and the
      R2* policy (`t2star="separate"`, exists). Deterministic in the seed; metadata records every
      draw.
- [ ] **Outside the brain.** PHANTOMS §6 says what gives synthetic brains away: scalp fat with
      chemical shift, skull and dura, sinuses, eyes. BrainWeb's fuzzy classes cover fat, skull,
      skin, dura, marrow, vessels and CSF; make sure each gets a cited property set and that fat
      classes get `fat_fraction` so the 6-peak model acts on them.
- [ ] **Dataset writer.** `blochparty.io.write_dataset(path, result, seq, phantom_meta, coils,
      noise, seed, engine, versions)` in ISMRMRD-compatible HDF5 (k-space per coil with the
      trajectory from `seq.meta`, acquisition header fields we can fill, and a `provenance` group
      with the sequence as JSON plus the `.seq` from M4, the sampler seed and every property
      draw, the engine and git hash). Reader for round trip.
- [ ] **First realism number (P2 start).** A classifier two-sample test, real vs simulated,
      on fastMRI brain raw k-space through the same reconstruction: a small CNN on magnitude
      images; report AUC. This is the anatomy pillar's benchmark the way `validation/real/` is
      the engine's. Expect a high AUC at first; the point is to have the number move.

Verify: PD conservation and bit-identity of `to_phantom` defaults (existing tests); tissue
statistics within 5 % of their targets over 100 samples; deformation Jacobian > 0 everywhere;
dataset round trip; a 3 T T1w / T2w / T2*w triplet of ten sampled heads in `examples/`.

## M6. Physics backlog, in order (each 1 to 3 days)

Ordered by what training data and sequence design will hit first. Each item ships with the
phantom input that feeds it (M5 schema) and its reference. Each is opt-in, core first, Metal
second, parity test third, fused raises.

1. **J-coupled fat.** Bright fat in TSE (fat T2 appears longer at short echo spacing; SE vs TSE
   fat ratio up to ~2). Model: per fat peak, a small spin-system (two or three coupled spins per
   peak group, J ≈ 7 Hz for the methylene chain) evolved by a density-matrix step under the
   same RF and gradient events, sampled as its transverse coherence; cheapest form is a
   precomputed echo-spacing-dependent apparent T2 per peak from that model, applied in
   `phantom_from_maps` when `j_coupling=True`. Verify: the density-matrix model against
   published SE/TSE fat ratios and the Stables et al. echo-spacing curves; the tabulated
   shortcut against the model; `test_phantom_fat.py` defaults unchanged.
2. **Deformable and physiological motion.** `Motion` today: velocity, displacement × waveform,
   rigid. Add a displacement *field* over time: per-spin trajectories as a sum of basis motions
   (respiratory: a smooth field with a breathing waveform; cardiac: later, with M5's heart).
   Spin history is automatic; the maps are sampled where the spin is (exists). Verify: ghosting
   of periodic motion vs the analytic ghost positions (period / TR), constant-velocity phase vs
   the first moment, a POSSUM-style spin-history case against the core at fine steps.
3. **Complex B1+ and pTx.** `Phantom.b1` is a real scale; `FieldMaps.b1` likewise, so imported
   full-wave maps (`em.py`) lose the transmit phase. Make `b1` complex (phase adds to the RF
   phase per spin) with a per-channel option: `b1` (C_tx, N) complex and `Sequence.rf_shim`
   (S, C_tx) complex drive weights; the effective field is Σ_c w_c b1_c. Default remains a real
   scalar and bit-identical. Verify: Mz = cos(|b1|α) for any phase; a two-channel shim that
   cancels at a point gives zero flip there; the imported mariepy maps' phase (the quadrature
   birdcage's spatial phase) vs the rotating-field phase test in `tests/test_em.py`.
4. **Contrast agents and perfusion.** Time-varying T1 (and T2*, χ) per spin from tracer kinetics
   (Tofts: Ktrans, ve, vp, an AIF) at the per-segment time; `Phantom` gets a `kinetics` field,
   `to_phantom` fills it from a lesion / parenchyma model (M5). Verify: the signal of a spoiled
   GRE vs the closed-form steady state with the instantaneous T1(t) at each TR when T1 changes
   slowly; the concentration-to-R1 relation (relaxivity) by construction.
5. **Orientation-dependent R2\*.** R2*(θ) per tissue from fibre orientation (a diffusion-tensor
   principal direction from M5's anatomy, or a synthetic field) and B0 direction; literature
   curves for WM (Lee et al.) as the model. Verify: the published curve; isotropic tissue
   unchanged.
6. **Smaller items, when touched anyway:** incidental MT validated on data with a known
   fat-saturation pulse; b-matrix with gradient nonlinearity; duty-driven drift; ramp-sampled
   readouts in the builders with gridding reconstruction; SLR pulses; respiration coupled to a
   low-order field model from a breathing phantom (M6.2 and `drift.respiration` together).

## M7. Training data and agentic design (after M5 and M6.1 to 6.3)

The two downstream goals. Only what their first users need; details when we get there.

- [ ] Generation driver: sample anatomy (M5) → sample a protocol family (builders, or a `.seq`
      from a library) → sample scanner imperfections (B1+, B0 shim, GIRF from benchmark 7, coil
      array from `loops.py` / `em.py`, noise covariance) → simulate on Metal → `write_dataset`.
      Throughput target and a nightly run that records spins·segments/s.
- [ ] Sequence spec in: JSON → builder calls (`design.design_report` already produces the report
      out). Linter messages mapped to Pulseq blocks (M4 provenance). Parameterized sequence API
      (which tensors are free, bounds, reparameterizations), hardware projections.
- [ ] A benchmark task suite with scored targets (e.g. "T1w brain at 1 mm in < 2 min, maximize
      WM/GM CNR") so agent progress is a number; every scored sequence must pass Pulseq's checks
      and the M4 round trip.

## M8. Other hardware (when machines exist)

- [ ] CUDA port of the Metal kernel (forward and adjoint), verified against the core and the
      Metal results in float32.
- [ ] CPU fallback for users without a GPU: the PyTorch core is it; `simulate_chunked` for memory.
- [ ] KomaMRI crosscheck and phantom export, if cross-simulator agreement becomes a question
      again.

## Not planned now

Phase-distribution-graph backend; non-Cartesian NUFFT reconstruction inside the package (the
validation scripts have what they need); XCAT / MRXCAT (licensing; prefer open shape models, M5);
a model-based reconstruction for benchmark 3 (tests the reconstruction, not the simulator); a
GUI (the three.js anatomy viewer in `examples/` is a prototype if wanted later).

## Done so far (summary; details in the archive)

Reference core with closed-form free runs and chunked backward; fused CPU kernel with adjoint;
Metal forward with many-coil matrix-product reductions and the field model; sub-voxel spins,
T2′, real spoiling, ramps, slice selection, B1+/B0 maps, 6-peak fat; SE / TSE / bSSFP / EPI
builders; bound pool (rate and generalized Bloch), free pools in core and fused; concomitant
fields, gradient nonlinearity, GIRF / eddy currents, drift and respiration; diffusion (free,
restricted), motion and flow, scanner-frame maps; receive and transmit chains; Biot–Savart and
full-wave coils; BrainWeb / QSM / AHEAD anatomy with texture and R2* separation; Pulseq import;
design penalties, losses, CRB and worked optima; MRzero and BMCTool crosschecks; nine real-data
benchmarks (NIST IR-SE 1.5 %, NIST TSE 2.2 %, radial IR-bSSFP, BSA NMR, spiral TSE, radial
timing, measured GIRF, Pulseq-CEST, iohexol CEST prediction); Monte Carlo estimators with
jackknife error bars. 735 tests.
