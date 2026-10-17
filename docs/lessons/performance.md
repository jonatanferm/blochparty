# Lessons: Performance

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
