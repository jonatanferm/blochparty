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
   Bloch bound pool, which only the core runs (`docs/ENGINES.md`, M1c). Since 2026-10-09 the
   exception is confined to short, intense pulses: for saturation pulses the rate model with
   `BoundPool(cutoff=None)` reproduces the generalized model on the GPU (M1c).
2. **Gradients.** The Metal kernel has an adjoint for the physics that calibration and sequence
   design need (M2, since 2026-10-09; the fused CPU kernel, which had one before, is retired).
3. **Physics no longer moves the benchmark table.** On at least one scanner with two sequences
   (NIST benchmarks 1 and 2), the remaining errors are attributable to unknown inputs (B1+,
   coil sensitivities, temperature, geometry) rather than to missing physics. The test is the
   digital-twin loop (M3): fit the unknown inputs on one sequence, predict the other, and show
   that adding physics changes the prediction less than the error bars.

## Engines

| engine | role | gradients | policy |
|---|---|---|---|
| `blochparty.simulate` (`src/blochparty/bloch.py`) | readable reference; every new effect lands here first, checked on small cases | autograd, `simulate_chunked` | keep small and readable; float64 |
| `blochparty.metal.simulate_metal` (`src/blochparty/metal/__init__.py`) | the engine; all imaging-scale work | hand-written adjoint (M2) | must agree with the core; float32 |
| `blochparty.fused.simulate_fused` (was `src/blochparty/fused/`) | CPU kernel with a hand-written adjoint | adjoint | retired 2026-10-09 after M2; in the git history |

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
- **Metal's adjoint raises** for what it does not differentiate (pattern: `_simulate_grad` in
  `src/blochparty/metal/__init__.py`), and the engine matrix says so.
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
      J = 30) is beyond the register budget of the `_bm` kernels. Resolved for the regime that
      matters by **(A) the cut-free lineshape** (2026-10-09, below); **(B)**, below, is the plan if
      the remaining regime is ever needed. Until then the parity condition has this one exception.
- [x] **(A) `BoundPool(cutoff=None)`** (2026-10-09): the rate model's lineshape is then the
      half-line Fourier transform of the generalized model's tapered Green's function, as a sum of
      Lorentzians over its 30 modes (`exchange.modes_lineshape`; Metal: lineshape kind 3, same
      buffers, no new kernel; the fused kernel raises). Measured first: with the default 1 kHz cut,
      rate and generalized model differed by 1.2e-2 in Z after a 36 × 50 ms, 2 µT saturation train
      at 3.5 ppm on WM-like tissue with two free pools (2.2e-4 at 3 kHz, 8.4e-4 at 10 kHz); with a
      Lorentzian pool they agree to 5e-6, so the gap was the near-resonance convention, not the
      dynamics. Without the cut: ≤ 5e-5 at every offset (9.6e-6 at 447 Hz). Verified: the modes
      sum equals quadrature of the tapered Green's function's transform to 1.3e-4 up to 10 kHz
      (3.8e-3 at 30 kHz, where G is 2e-3 of its peak), is finite on resonance (8.39 T2b), within
      0.75 % of the untapered table from 200 Hz to 10 kHz, even, exact for a Lorentzian pool,
      gradcheck; Metal `_full` and `_bm1`/`_bm2` kernels vs the core 5e-7 (signal), ≤ 1e-7
      (states); snapshot bit-identical; engine matrix row with the fused raise.
      **Where it does not help**: pulses short against the Green's function's memory
      (300 T2b = 3 ms at T2b 10 µs) played on resonance. 60 hard pulses of 0.2 ms / 60° leave
      the water 1.4e-3 (cut) and 2.3e-2 (no cut) from the generalized model; 0.1 ms / 30°:
      3.9e-2 and 1.1e-1; 0.5 to 2 ms / 90°: 2 to 8e-3 either way. There the pool rotates and
      neither rate variant is right (`tests/test_pools.py`).
- [ ] **(B) Generalized bound pool with free pools on Metal by operator splitting**, if needed:
      keep the mode state in the device buffer the `_gen` variant already uses (`U`, 2J × N), and
      per RF sub-step do a half step of the existing mode and bound-pool rotation (`step_gen`'s
      "pool about the RF alone" block), a full Taylor step of water + free pools with the bound
      exchange rows and no saturation rate, then the second half step; Strang, second order. The
      Taylor vector stays at its current size, so no register growth. Verify against the core's
      exact engine (expect ~1e-4, the level of the core's own split engine). It inherits the
      `_gen` cost of sub-steps at T2b / 2, which makes it wrong for 50 ms saturation pulses
      (10⁴ sub-steps each; A covers those) and right for hard pulses ≲ 0.3 ms. **When we will
      need it**: imaging with free pools *and* short intense on-resonance pulses, i.e. MRF or
      bSSFP-style trains of ≲ 0.3 ms hard pulses on tissue with a CEST or myelin-water pool, or
      inversion / refocusing hard pulses in an MT + CEST protocol, at imaging scale (the core
      already runs it exactly for small cases). None of the nine real-data benchmarks is in this
      regime; benchmark 9 (iohexol, no MT) and the Pulseq-CEST protocols use 50 to 100 ms pulses.
      Not recommended: the 60 mode components in thread memory with the exact Taylor step (the
      modes make A dt stiff, so it needs the same sub-steps and is no more accurate than B).
- [x] Free pools with diffusion or motion (2026-10-08): positions, walks and the field model per
      segment in the exact engine, core and Metal (`_bm*_diff` kernels; spins not cost-sorted
      when moving: walks are keyed by spin index). Inert pools equal the single-pool path with the
      same motion or walk to 3e-7; a constant rigid shift equals the moved phantom to 1e-13.
- [x] Free pools with `FieldMaps` (2026-10-08), static and moving, core and Metal: equal to the
      same fields given per spin to 1e-14.

## M2. Metal adjoint (5 to 7 days) — done 2026-10-09

Status. `simulate_metal` is differentiable for the planned scope (sequence `dt`, `rf`, `grad`,
`adc_phase`, `b0_offset`, `rf_freq`; phantom `pos`, `pd`, `t1`, `t2`, `db0`, `b1`; coils; `m_init`;
the rate-model bound pool, with or without the lineshape cut; the scanner field model through the
encoding positions and coefficients; static `FieldMaps`), raising for moving spins, free pools and
the generalized bound pool. Checks: (1) against the core's float64 autograd per input, 1e-7 to 3e-6
on random sequences, 2e-5 on a 1600-segment RF-spoiled GRE, ≤ 1.2e-4 with the bound pool; (2) as
accurate as the fused float32 adjoint or better (bound-pool `rf_freq`: 1.2e-4 against its 1.5e-3;
32000 RF segments: 8.9e-4 where the core in float32 is 5.4e-3 off); (3) bit-identical for any
`checkpoint_every` and every run; (4) the WM/GM optimum to 0.001°, the Ernst angle to 0.01°, the
DESPOT1 pair to 0.0003°–0.03° with its closed-form bound within 3e-6 of the optimum (fourth-order
differences of float32 signals, `design.jacobian(method="fd4")`; the flat bound makes the pair the
wrong yardstick); (5) the 5x target is met only in part: after storing only m in the replay buffer and
forming the coil terms once per 4-segment block (backward −11 to −20 %), forward + backward is ~5x the
forward for per-spin gradients on 3d-48 (8 coils), but 8-9x there with coil and flip-angle gradients
too (`bench.py --grad`: 3.2-4.1 s against a 0.42 s forward) and 5.5x on 3d-64; the interval length
does not change it. Open performance work, not a blocker: the plain forward is very efficient (MMA
coil sums), the reverse sweep is register-bound. The fused kernel is retired (2026-10-09): the design examples whose checks need float64
(constraints to 1e-6, Bloch vs EPG to 1e-12) run on the core, which takes 36 s for the TR-budget
design and 74 s for DESPOT1 instead of seconds; free-pool and generalized-pool gradients are
core-only. Journal: 2026-10-09.

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

## M3. The digital-twin loop on the NIST data (3 to 5 days) — done 2026-10-09

Status. B1+ fitted on benchmark 1 through the Metal adjoint predicts the held-out benchmark 2 at
1.83 ± 0.03 % (shape, spheres 1-10) against 2.20 ± 0.02 % with nothing fitted; in sample it takes
benchmark 1 from 1.77 ± 0.31 to 1.13 ± 0.25 %. The field is plausible (0.93-1.19 on the spheres,
centre-brightened). The reverse direction fails: B1+ fitted on benchmark 2 predicts benchmark 1 at
13.6 % (it absorbs sphere-specific residuals of the echo shapes; a refocusing-only factor and the
phantom temperature explain none of them). Ablations, paired on common Monte Carlo seeds: the water
fill (+0.25 on benchmark 2, 0.0 on benchmark 1) and the slice profile (+7 to +25) matter and are
in the model; concomitant fields (≤ 0.01) and the measured Prisma GIRF of benchmark 7 (≤ 0.1, within
1-1.4 SE) do not, so condition 3 holds for these two sequences. Gradient delays and the first
refocusing interval were scanned out before fitting (< 0.1 %). Of benchmark 2's short-T2 spheres,
11 drops from 7.8 to 3.8 % with the twin but 12-14 stay at 13-17 %: the physics missing there is not
B1+, timing or temperature; candidates are the fill geometry at the sphere edges and the references. Results:
`validation/real/RESULTS.md` ("Digital twin"); journal 2026-10-09.

The original plan:

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

## M4. Pulseq out (1 to 2 days) — done 2026-10-09

Status. `pulseq.write_pulseq` writes a builder sequence losslessly: GRE (ramps, slice selection,
RF spoiling), TSE, bSSFP and EPI simulate identically after write → `read_pulseq` (signal 6e-16 to
5e-14), pypulseq's `check_timing` passes, and k-space at every ADC sample equals its
`calculate_kspace` of the written file (4e-13 to 5e-11). Imported files are written back from their
events: bit-identical for a file pypulseq wrote, last bits (1.7e-18 s) for a scanner file. Two
departures from the plan, both forced by the format: the writer is ours (Pulseq 1.5 at full
precision; pypulseq's writer keeps six significant digits, ~5e-7, and `read_pulseq` now reads
without pypulseq's rounding merge, which leaves every scanner file in `validation/real` bit-identical),
and flat sequences without provenance raise instead of becoming arbitrary-gradient blocks
(instantaneous gradient steps and RF without dead time have no Pulseq form; approximating them
would not be lossless). Before writing, the provenance records are replayed and must rebuild the
sequence exactly, so a changed sequence (e.g. after `apply_girf`) raises. Journal: 2026-10-09.

The original plan:

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

- [x] **Schema (P0).** A `DigitalPhantom` container above `Phantom`: label fractions on an
      anatomy grid (what `anatomy.Anatomy` holds), per-tissue property distributions
      (`anatomy.TissueProperties`), field-strength-dependent tables, and the inputs today's
      engine needs but the tissue table lacks: bound-pool fraction / exchange rate / T2b per
      tissue (literature qMT values at 3 T for WM, GM, CSF, muscle, fat; cite each as the tissue
      table does), diffusivity per tissue, fat fraction. `to_phantom` fills `Phantom.bound`,
      `diffusion` and `fat_fraction` from them, opt-in. Round trip with today's maps bit-identical.
      *Done 2026-10-09:* `anatomy.DigitalPhantom` (anatomy, one field's table, `to_phantom`
      options, `meta`); cited qMT and diffusivity for every signalling class (table in
      validation/anatomy/README.md); round trip bit-identical; defaults unchanged (SHA-256).
- [x] **Procedural sampler (P1).** `anatomy.sample(seed, *, subject=None, field_T, voxel)`:
      choose a BrainWeb subject (the 20-subject set is loaded and pinned), apply a random smooth
      diffeomorphic deformation (B-spline displacement field with a magnitude prior; check the
      Jacobian stays positive), sample tissue properties from the cited distributions, apply the
      within-tissue texture (`variability`, exists), compute χ → B0 and shim (exists), and the
      R2* policy (`t2star="separate"`, exists). Deterministic in the seed; metadata records every
      draw.
      *Done 2026-10-09* as `anatomy.sample_head`: cubic B-spline control points N(0, 3 mm²)
      40 mm apart (displacement SD ~1 mm, Jacobian 0.83-1.21 on subject 04), per-subject
      log-normal draws of every property with a reported SD (`draw_tissues`; 1800
      standardised draws: mean -0.03, SD 1.00), texture and spin seeds from one SeedSequence.
- [x] **Outside the brain.** PHANTOMS §6 says what gives synthetic brains away: scalp fat with
      chemical shift, skull and dura, sinuses, eyes. BrainWeb's fuzzy classes cover fat, skull,
      skin, dura, marrow, vessels and CSF; make sure each gets a cited property set and that fat
      classes get `fat_fraction` so the 6-peak model acts on them.
      *Partly done 2026-10-09:* every signalling class has cited (or flagged-proxy) MT and
      diffusion, and fat and marrow carry fat fraction 1 (6 peaks). Open: relaxation of skin,
      connective tissue and dura is still muscle's and marrow's still BrainWeb's legacy fat prior,
      adipose tissue is pure lipid (real adipose tissue holds some water), and there are no eyes,
      vessels with flow or correlated coil noise. The tissue changes alter default `to_phantom`
      output, so they wait for a decision.
      *2026-10-12 / 13:* IXI-segmented heads (`load_ixi`) bring each subject's own scalp layers
      (fat unmixed from its T1 / T2 with IXI's water-fat shift undone, soft tissue split skin /
      muscle by T2), a falx (`add_falx`), and coloured coil noise in the realism score.
      *2026-10-13, what the rest needs:* (1) skin: measured dermis T2 is ~10-12 ms (Richard 1991 / 1993, Song
      1997), ours 50 ms (muscle's), but in IXI heads "skin" stands for the scalp's non-fat, non-muscle soft tissue,
      unmixed to reproduce each subject's T2; dermis values there would darken the whole scalp, so the dermis wants
      its own thin surface class with the cited values; (2) eyes: tissuedb has lens / aqueous / retina but no
      vitreous (T1 4.3-4.5 s at 3 T, Simpson 2024; T2 546 +- 157 ms only at 1.5 T), and the IXI segmentation labels the
      vitreous "skin" (CSF-like intensities outside the brain), which an orbit-restricted eye class would fix; no
      effect on the realism score (fastMRI's stack starts above the orbits); (3) dura and connective tissue: no
      in-vivo measurements found. The verify list below is covered (tests/test_head_sampler.py: draws over 100
      subjects, Jacobian; tests/test_io.py: round trip; examples/sampled_heads.py: the ten-head triplet).
      *Done 2026-10-13* (3f4cf1b, b8b9106): a dermis class (`add_dermis`, `sample_head(dermis=True)`: the outer
      1.61 mm, T2 12 ms from its 10 ms T2*, flagged) and an eye class (vitreous, T1 4.518 s at 3 T measured, T2 from
      1.5 T flagged; prepare_ixi's `orbit_vitreous` finds the globes, IXI012 9.0 mL). Left: dura and connective
      tissue keep muscle's relaxation (no in-vivo values found); vessels with flow.
- [x] **Dataset writer.** `blochparty.io.write_dataset(path, result, seq, phantom_meta, coils,
      noise, seed, engine, versions)` in ISMRMRD-compatible HDF5 (k-space per coil with the
      trajectory from `seq.meta`, acquisition header fields we can fill, and a `provenance` group
      with the sequence as JSON plus the `.seq` from M4, the sampler seed and every property
      draw, the engine and git hash). Reader for round trip.
      *Done 2026-10-09:* via ismrmrd-python (extra `io`); one acquisition per ADC run with its
      k-space (`sequence.kspace_trajectory`, equal to pypulseq's), encode indices and
      ACQ_IS_REVERSE; round trip exact (complex64 signal, sequence arrays and meta, `.seq` text).
- [x] **First realism number (P2 start).** A classifier two-sample test, real vs simulated,
      on fastMRI brain raw k-space through the same reconstruction: a small CNN on magnitude
      images; report AUC. This is the anatomy pillar's benchmark the way `validation/real/` is
      the engine's. Expect a high AUC at first; the point is to have the number move.
      *Done 2026-10-10 / 13* (`validation/realism/`, docs/JOURNAL.md): CNN AUC 1.0 on the first
      heads, 0.875 on r27 (IXI heads with a fitted residual layer, the real echo train),
      histogram test at chance, a blinded reader 7 / 24; `quick.sh` for 10-minute iterations.
      What is left is M5b.

Verify: PD conservation and bit-identity of `to_phantom` defaults (existing tests); tissue
statistics within 5 % of their targets over 100 samples; deformation Jacobian > 0 everywhere;
dataset round trip; a 3 T T1w / T2w / T2*w triplet of ten sampled heads in `examples/`.

## M5b. Closing the realism gap (revised 2026-10-13)

Where it stands: by eye the simulated T2w slices pass (a blinded reader 7 / 24); the CNN still
separates them, AUC 0.875 on r27, but that number is soft: the CNN's own seed moves it to 0.93,
against the 11 Skyra scans alone it is 0.94-0.99, and 21 real against 16 simulated people give a
+-0.12 interval. The real set's own scanner splits score 0.42-0.58 (`gap.py controls`), so the
remaining gap is real, and `validation/realism/gap.py` says where it sits (journal 2026-10-13).
Items in the order of measured effect over cost; each has its own number to move, checked with
`quick.sh` (10 min) and `gap.py` before a full run.

1. **The scalp against the brain, and placement** (2 days). Posterior / anterior scalp 1.48
   against the real 1.03 (ours: anterior 0.36 vs real 0.45, posterior 0.53 vs 0.47), the sides'
   and front's scalp 0.83-0.93 of real, the interior p99 above the scalp's (1.04; real 0.95, where
   the scalp fat sets the scale). Not the receive ring: with a flat receive (q33, a 0.5 m ring
   centred on the head) the ratio stays 1.49 and the whole brain reads 20 % too bright against the
   scalp (quartiles 0.23 / 0.29 / 0.44 vs real 0.20 / 0.24 / 0.37), so the 14 cm ring behind the
   head has been standing in for scalp fat that is ~20 % too dim against brain, or brain that is
   too bright (incidental MT: measure its size with the bound pool off; fat's water content and
   PD). The anterior deficit is the preparation: in eight v14 heads the frontal scalp carries
   0.9-2.1 mm of labelled fat (one 4.2) against 2.5-4.3 mm at the back, and the frontal bone band
   is 0.8-2.6 mm thick against 1.8-6.6 behind (the real forehead bone is ~6 mm, which also
   explains the anterior band reading 1.08-1.14 x real): prepare_ixi's unmixing at the front,
   where fat, bone and the undone water-fat shift meet; check the labels against IXI's own T1
   (fat bright, bone dark) on the forehead and fix the rule. The head centroid sits
   2.8 px (4.8 mm) further anterior than the real operators place it (AUC 0.83 for that one
   feature): offset the field of view. Verify: scalp / brain and posterior / anterior within 0.1
   of real under a flat receive, p99 inside below the scalp's, centroid row 60 +- 3.
   *2026-10-13:* the placement is done (q35: row 60.1, AUC 0.48; 7a53466); the frontal bone was
   labelled skin under the T2's displaced fat and is bone again (a29d685: IXI012 2.2 -> 3.4 mm),
   in the v16 heads; the scalp / brain level and the posterior / anterior ratio remain. Over the
   white-matter mode (gap.py, journal): with a flat receive the brain's quartiles match real and
   CSF is 9 %, the scalp 21 % dim; the ring had been masking that. Tested (q37-q42): MT x1.3
   puts CSF / WM on real but GM / WM 8-11 % high; skin T2 90 ms and fat T2 170 ms bring the scalp
   within 4 %. *Later that day:* the 0.57 mm profiles showed the whole outer 2 cm 1.3-2.7 x
   brighter in the real scans with the interior flat: the array's sensitivity under the scanner's
   normalisation, not fat (a 2 x on fat needs T2 ~650 ms). Modelled as an elliptical ring hugging
   the head with its RSS divided by its own smoothed RSS (`--receive-normalise-mm`), and, since the
   34 real scans spread 2.1-6.0 in scalp level by coil and site, drawn per head (`--receive-draw`:
   q50 reproduces the real spread, 4.5 +- 0.8). r31 (ring) / r32 (one normalised receive) / r33
   (drawn) on the 64 v16 heads against 34 scans decide the preset; no tissue value was changed.
   Open: CSF / WM 8 % low (MT or CSF inflow), the posterior scalp still +20 %, and the v16 heads
   need a screen (two outliers in area and CSF share came through the proxy selection).
   *2026-10-14:* the 62-head runs (r31b / r32 / r33) against 34 scans read 0.97-1.0 because the
   quick quality's spin grain (8 per voxel: 0.015 vs the real 0.010 in flat white matter) is a cue
   the CNN learns with that many heads (24 heads: 0.89-0.90). Benchmark runs use 2 x 3 x 3 spins
   from r34 on; quick.sh is read by its W1 and gap.py measures, never its CNN.
   *2026-10-15, r34 (62 heads at 2 x 3 x 3 vs 65 compact scans): CNN 0.996 with every level
   over white matter within noise; what it reads is the grain (0.012 vs 0.010: 4 x 4 x 4 fixes it)
   and the scalp level (+12 % against the larger real set's dimmer scalp: re-centre the receive
   draw). The real set is now the compact 65 (journal).*
2. **A benchmark that resolves 0.05** (2 days, plus a data request). More simulated heads: 64 of
   the 185 downloaded IXI subjects, chosen so the heads' size and CSF share follow the real
   distribution (area 0.95 x real today; the CNN's verdict tracks the CSF share), at quick quality
   with all seven slices; more real scans: more 3 T AXT2 volumes from the fastMRI validation archive
   (streamed with the existing link; `fetch_fastmri.py`); the CNN averaged over three seeds. Verify: real-vs-real controls
   at 0.5 +- 0.05; the interval on real vs simulated half as wide.
3. **Coarse structure at matched CSF** (2-3 days). *2026-10-13:* mostly closed by refine_csf and
   the subjects (r30: coarse bands 1.03 / 1.05 / 0.99 x real); the fine bands were 15 % short and
   the residual's smoothing at 0.5 mm instead of 1.5 brings them to 0.94 / 0.97 (q40; the new
   default, from v17 on). The heads the CNN catches (older, CSF-rich)
   carry 1.36-1.40 x the real coarse structure power (0.05-0.45 of Nyquist) at a CSF share that
   real scans reach with 1.3, and 0.9 x at the top band; their CSF pieces are smaller than real
   (9.9 vs 11.5 px). Candidates, each a quick run: the sulcal CSF of older subjects through the
   5 mm slice (IXI's own T2 at 1.2 mm against the real partial volume), the residual's 1.5 mm
   smoothing (the missing fine texture), the slice profile. Verify: power by band within 10 % of
   real at matched CSF share; the caught heads' scores fall.
4. **The posterior band** (1-2 days). *2026-10-16, read from the truth grids (`gap.py composition`):* our
   dark valley sits ~4 mm nearer the brain than the real one and is deeper, because the clinical
   population's posterior scalp + skull is thicker than the IXI volunteers' as segmented (6.6 vs
   5.0 mm of soft tissue, the valley at 10-11 vs 9 mm); the fat is concentrated into a brighter
   peak for the same reason. It is a thickness distribution to draw against the real 0.57 mm
   profiles (the head on the pad, patients vs volunteers), not a tissue value. 0.72-0.89 of real at 0.80-0.94 of the head radius behind
   the brain, 1.31 at the posterior edge: deeper and darker than real. Known: not the labels, not
   the pitch, partly IXI's fat shift (journal 2026-10-12). Open: the occipital diploë (the real
   aligned profile reads 1.6 where ours is 0.8), the dura and CSF under the occiput. Verify with
   the aligned posterior profiles (minimum depth and value within 0.5 px / 0.03 of real).
5. **A second contrast** (3-5 days). The heads must hold across sequences, or the T2w fit is a
   fit. fastMRI's 3 T AXFLAIR (15 volumes on the mini) and AXT1PRE (11) score the same heads under
   inversion recovery and T1 weighting with nothing re-tuned; the residual layer then needs the
   subject's T1 scan too (IXI has it). Expect FLAIR to expose T1 values, CSF nulling and the
   bound pool. Verify: a CNN AUC per contrast, reported together; a blinded reader on FLAIR.
6. **Population**: fastMRI's patients are older, larger-headed and some carry lesions. Subject
   choice (item 2) covers size and CSF; pathology is out of scope until a lesion model exists.

After items 1-2 the realism loop becomes a regression (a nightly quick run with its numbers in
the journal) and M7's generation driver starts; items 3-6 continue beside it. The physics backlog
(M6) is ordered by what M7's protocol families and item 5 need, not by this benchmark, which
cannot see motion, pTx or orientation effects.

## M6. Physics backlog, in order (each 1 to 3 days)

Ordered by what training data and sequence design will hit first (revised 2026-10-13: 6.1 is
closed as a negative result; 6.4 moves up because fastMRI's post-contrast T1w scans give it a
benchmark; 6.2, 6.3 and 6.5 wait for a protocol family that needs them). Each item ships with the
phantom input that feeds it (M5 schema) and its reference. Each is opt-in, core first, Metal
second, parity test third; the Metal adjoint either differentiates it (gradient check against the
core) or raises, and the engine matrix says which.

1. **J-coupled fat.** Bright fat in TSE (fat T2 appears longer at short echo spacing; SE vs TSE
   fat ratio up to ~2). Model: per fat peak, a small spin-system (two or three coupled spins per
   peak group, J ≈ 7 Hz for the methylene chain) evolved by a density-matrix step under the
   same RF and gradient events, sampled as its transverse coherence; cheapest form is a
   precomputed echo-spacing-dependent apparent T2 per peak from that model, applied in
   `phantom_from_maps` when `j_coupling=True`. Verify: the density-matrix model against
   published SE/TSE fat ratios and the Stables et al. echo-spacing curves; the tabulated
   shortcut against the model; `test_phantom_fat.py` defaults unchanged.
   *2026-10-13, model built, shortcut not shipped:* `blochparty.jcoupling` evolves the density matrix of coupled
   fragments (strong-coupling Hamiltonian, hard pulses; tested against the analytic limits) and a triglyceride
   fragment model consistent with the six-peak spectrum. With one intrinsic T2 calibrated to CPMG at 8 ms (145 ms,
   Keene 2020), it predicts single-echo fat T2 222 ms against the 48-68 ms measured at 3 T, and a TSE / SE fat ratio of
   1.13 against Henkelman's ~2 (`validation/jcoupling_fat.py`): the coupled chain ends are only part of the story, the
   bulk methylene (half the signal, unmodulated here) and other mechanisms that fast refocusing suppresses carry the
   rest. Next: measured fat T2 against echo spacing at 3 T (intermediate spacings), or a methylene-chain model; until
   then the CPMG fat T2 stays (tissuedb LIPIDS), and at fastMRI's 10.26 ms the J part alone predicts fat ~3.5 % dimmer
   than at 8 ms.
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

## M7. Training data and agentic design (after M5b items 1-2; revised 2026-10-13)

The two downstream goals. Only what their first users need; details when we get there. The
generation driver comes first: it turns the realism loop into a nightly regression and shows
which physics (M6) the protocol families call for.

- [ ] Generation driver: sample anatomy (M5) → sample a protocol family (builders, or a `.seq`
      from a library) → sample scanner imperfections (B1+, B0 shim, GIRF from benchmark 7, coil
      array from `loops.py` / `em.py`, noise covariance) → simulate on Metal → `write_dataset`.
      Throughput target and a nightly run that records spins·segments/s.
      *Started 2026-10-14:* `validation/realism/generate.py` (spec + seed -> ISMRMRD sample with
      a `/truth` group: clean / noisy k-space, clean image, fractions, residual maps, lesion mask);
      the spec's front door is the brain itself: `anatomy.add_lesions` places white-matter lesions by
      region with cited tissue values. The first consumer is reconstruction: `transfer.py` scores a
      U-Net trained on synthetic data on held-out real fastMRI k-space (zero-fill / synthetic / real
      / synthetic-then-real). *First number, 2026-10-14 (181 samples, 11 held-out people):* zero-fill
      NMSE 0.133 / SSIM 0.569, synthetic-only 0.052 / 0.867, real-only 0.053 / 0.857, synthetic then
      real 0.044 / 0.887. *Three seeds with a fair real baseline, 2026-10-16 (180 + 180 samples, real
      trained on 308 slices):* zero-fill 0.140 / 0.56, synthetic-only 0.054 / 0.85, real-only 0.040 /
      0.89, synthetic then real 0.041 / 0.89 (confirmed on a second set, 2026-10-16). Domain shift
      modest and asymmetric (the synthetic net loses ~12 % crossing to real, the real net ~28 %
      crossing to synthetic, whose task is harder: more mid-band power at the rim). **Lesion test
      (2026-10-16):** every trained net dims lesions ~20 % and doubles its error inside them;
      training on synthetic lesions does not help (0.090 vs 0.075 inside lesions for the real-trained
      net) — the lever is the reconstruction method (a structure-preserving loss, k-space data
      consistency), not more lesion data. Next: that method change, measured by the same lesion
      test; the posterior scalp thickness (M5b item 4); then 320 x 320 and a second contrast.
- [ ] Sequence spec in: JSON → builder calls (`design.design_report` already produces the report
      out). Linter messages mapped to Pulseq blocks (M4 provenance). Parameterized sequence API
      (which tensors are free, bounds, reparameterizations), hardware projections.
- [ ] A benchmark task suite with scored targets (e.g. "T1w brain at 1 mm in < 2 min, maximize
      WM/GM CNR") so agent progress is a number; every scored sequence must pass Pulseq's checks
      and the M4 round trip.
      *First task, 2026-10-16 (exploratory):* lesion detection in a slab under an acquisition
      budget — a standard TSE + SENSE + CNN against a GRE whose flips, TE, TR and k-space lines are
      learned jointly with a net on its raw multi-coil samples through the Metal adjoint
      (`validation/design/lesion_design.py`). Two sweeps at chance (2026-10-16/17): lesions too
      faint, the line mask on the readout axis, a noiseless "SNR 20", a net that memorised heads
      (journal 2026-10-17). Fixed; at full sampling the baseline reads the task at AUC 0.88-0.89
      on 276 slabs and nothing reads 13 uniform lines (R = 5). Third sweep: 50 / 30 / 20 / 100 % on
      744 slabs, with a fixed-GRE arm as the ablation of what the design adds (running).

## M8. Other hardware (when machines exist)

- [x] CUDA port of the Metal kernel (forward and adjoint), verified against the core and the
      Metal results in float32. *Done 2026-10-17:* `blochparty.cuda.simulate_cuda` — Metal's device
      code turned into CUDA by text rules, CUDA kernels (warp-shuffle coil sums), Metal's driver with a
      backend switch, NVRTC at run time. On an RTX 4090 (Runpod): all 68 CUDA cells of the engine matrix
      and the four GPU test files pass at Metal's tolerances; 3d-64 forward 0.51 s (M4 Metal 2.40 s),
      3d-96 (0.24 M spins) 3.4 s at 6.5e10 spin-segments/s, forward + backward 3d-64 2.2 s.
- [ ] CPU fallback for users without a GPU: the PyTorch core is it; `simulate_chunked` for memory.
- [ ] KomaMRI crosscheck and phantom export, if cross-simulator agreement becomes a question
      again.

## Not planned now

Phase-distribution-graph backend; non-Cartesian NUFFT reconstruction inside the package (the
validation scripts have what they need); XCAT / MRXCAT (licensing; prefer open shape models, M5);
a model-based reconstruction for benchmark 3 (tests the reconstruction, not the simulator); a
GUI (the three.js anatomy viewer in `examples/` is a prototype if wanted later).

## Done so far (summary; details in the archive)

Reference core with closed-form free runs and chunked backward; a fused CPU kernel with adjoint
(retired 2026-10-09 after M2); Metal forward with many-coil matrix-product reductions and the
field model, and since M2 its adjoint; sub-voxel spins,
T2′, real spoiling, ramps, slice selection, B1+/B0 maps, 6-peak fat; SE / TSE / bSSFP / EPI
builders; bound pool (rate and generalized Bloch), free pools in core and Metal; concomitant
fields, gradient nonlinearity, GIRF / eddy currents, drift and respiration; diffusion (free,
restricted), motion and flow, scanner-frame maps; receive and transmit chains; Biot–Savart and
full-wave coils; BrainWeb / QSM / AHEAD anatomy with texture and R2* separation; Pulseq import and
lossless export (M4);
design penalties, losses, CRB and worked optima; MRzero and BMCTool crosschecks; nine real-data
benchmarks (NIST IR-SE 1.5 %, NIST TSE 2.2 %, radial IR-bSSFP, BSA NMR, spiral TSE, radial
timing, measured GIRF, Pulseq-CEST, iohexol CEST prediction) and a digital twin across benchmarks 1
and 2 (M3); Monte Carlo estimators with jackknife error bars. 685 tests.
