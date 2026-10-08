# Engines

Which simulator does what. The capability matrix below is checked cell by cell by
`tests/test_engines.py`, which parses this file: edit a cell and the test checks the new claim.

| engine | entry point | device, precision | gradients | role |
|---|---|---|---|---|
| core | `blochparty.simulate` (`src/blochparty/bloch.py`) | any torch device; float32 or float64 | autograd (`checkpoint_every` bounds memory over segments) | the readable reference: new physics lands here first |
| fused | `blochparty.fused.simulate_fused` (`src/blochparty/fused/`) | CPU, all cores; float32 or float64 | hand-written adjoint, checkpointed | frozen: keeps what it has, new physics raises; retired after ROADMAP M2 |
| metal | `blochparty.metal.simulate_metal` (`src/blochparty/metal/`) | Apple GPU; float32 only (inputs are cast, results come back float32 on the CPU) | hand-written adjoint on the GPU, checkpointed (`checkpoint_every`); not for moving spins, free pools or the generalized bound pool (raises) | the engine for imaging-scale work |
| chunked | `blochparty.simulate_chunked` (`src/blochparty/chunked.py`) | as the core | autograd, recomputed one spin chunk at a time | the core with a memory-bounded backward over spins |

## Capability matrix

`✓`: supported, and in `tests/test_engines.py` the engine agrees with the core on a small case
(forward: signal and final magnetization; gradients: the core's autograd against a central
finite difference, the other engines' gradients against the core's autograd). `✗`: raises
`NotImplementedError`. The core's physics is checked against the independent references in the
last column; combinations of features are checked only engine against engine.

| feature | core fwd | core grad | fused fwd | fused grad | metal fwd | metal grad | chunked fwd | chunked grad | reference for the core |
|---|---|---|---|---|---|---|---|---|---|
| basic Bloch: PD, T1, T2, dB0, RF, gradients, ADC, spoils, coils | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | SciPy ODE (`test_bloch_vs_ode.py`), closed forms (`test_analytic.py`, `test_free_runs.py`), EPG (`test_epg.py`), MRzero (`-m crosscheck`) |
| B1+ (`Phantom.b1`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | flip b1·α, signal equation (`test_phantom_b1.py`) |
| FieldMaps, static spins | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | linear maps = the same fields per spin (`test_maps.py`) |
| FieldMaps, moving spins | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | motion through a linear B0 map = a gradient; moved object under mapped coils (`test_maps.py`) |
| bound pool, rate model | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | matrix exponential, CW steady state, SciPy ODE (`test_exchange.py`); BMCTool (`validation/compare_bmctool.py`) |
| bound pool, generalized Bloch | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | reference solver, Henkelman CW limit, rate-model limit (`test_generalized_bloch.py`) |
| free pools (`Phantom.pools`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | SciPy ODE of the full system, McConnell two-site exchange (`test_pools.py`); BMCTool |
| free pools + rate bound pool | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | as above (`test_pools.py`) |
| diffusion, free (isotropic or tensor) | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | Stejskal–Tanner, tensor, CPMG vs EPG with diffusion (`test_diffusion.py`) |
| diffusion, restricted | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | planes (Tanner–Stejskal), sphere and cylinder long-time limits (`test_restriction.py`, through Metal) |
| motion (flow, waveform, rigid) | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | phase contrast, rigid = moved phantom, inflow recursion (`test_motion.py`) |
| concomitant fields, leading order | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | Bernstein's phase, Maxwell-compensated echo (`test_fields_concomitant.py`) |
| concomitant fields, exact | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | \|B\| of the coil field in 50-digit arithmetic (`test_fields_concomitant.py`) |
| gradient nonlinearity | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | point positions vs the displacement field (`test_fields_nonlinearity.py`) |
| GIRF / eddy currents (`apply_girf`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | recurrence, delay shift, analytic B0 eddy phase (`test_girf.py`); a sequence transform, so any engine runs it |
| uniform field per segment (`Sequence.b0_offset`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | analytic drift phases (`test_drift.py`) |
| RF carrier (`Sequence.rf_freq`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | resonance sign (`test_exchange.py`); only the rate-model bound pool reads it |
| many coils (40) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | signal linear in the sensitivities |
| initial magnetization (`m_init`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | as the basic case |
| bound pool + scanner field model | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | combination |
| free pools + scanner field model | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | combination |
| bound pool + diffusion | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | combination |
| generalized Bloch + diffusion | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | combination |
| diffusion + leading-order concomitant fields | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | walks in true coordinates, coefficients per segment where the spin is |
| diffusion + gradient nonlinearity | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | walks in true coordinates, encoded per segment |
| diffusion + exact concomitant fields | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | as above |
| motion + gradient nonlinearity | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | = a static spin at each segment's position, segment by segment (`test_motion.py`) |
| motion + concomitant fields | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | as above; the velocity phase of a Maxwell-compensated echo (`test_fields_concomitant.py`) |
| free pools + generalized Bloch | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | the modes as rows of the exact matrix: a hand-written ODE (`test_pools.py`); Metal open (ROADMAP M1c) |
| free pools + diffusion | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | inert pools = the single-pool path with the same walk (`test_pools.py`) |
| free pools + motion | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | constant shift = the moved phantom; inert pools = the single-pool path (`test_pools.py`) |
| free pools + FieldMaps | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | = the same fields per spin (`test_pools.py`) |

Not in the matrix because they produce ordinary inputs and so run on every engine: phantom
builders (sub-voxel spins, T2′, 6-peak fat, susceptibility B0, anatomy), sequence builders and
`read_pulseq`, and the other sequence transforms (`blochparty.drift`, `Sequence.subdivide_rf`).

## Agreement

Tolerances of `tests/test_engines.py`: max |engine − core| / max |core| per output, the core in
float64 (the bound pool and free pools relative to the water's magnetization: a saturated bound
pool is tiny, so its own maximum is no scale). Observed: the worst case over the matrix on an M4.

| engine | output | tolerance | observed (case) |
|---|---|---|---|
| fused, float64 | signal, magnetization | 1e-12 | 3.6e-14 (generalized Bloch) |
| fused, float64 | gradients vs the core's autograd | 1e-11 | 9.7e-14 (`rf_freq`) |
| chunked, float64 | signal, magnetization | 1e-12 | 3.5e-16 |
| chunked, float64 | gradients | 1e-12 | 1.2e-15 |
| metal, float32 | signal, magnetization, static spins | 5e-5 | 1.4e-5 signal, 1.6e-5 m (leading-order concomitant fields: gradients 50× the base case's, many cycles of phase); free pools ≤ 1e-6 |
| metal, float32 | signal, walking spins | 2e-4 | 7.4e-5 (diffusion + gradient nonlinearity, spins up to 10 cm from isocentre) |
| metal, float32 | per-spin magnetization, walking spins | 3e-3 | 1.1e-3 (5 µm cylinders up to 5 mm from isocentre) |

Why Metal is looser with walking spins: the kernel keeps each walker's absolute position in
float32, so a spin 5 mm from isocentre is resolved to ~3e-10 m, and a compartment's walls (offset
r − centre) to that as well; the per-spin phase carries the rounding, the signal averages it. A
walk in offsets from the start position would remove most of it.

Long sequences (`tests/test_metal.py`): 1e4 CW RF segments agree to 1e-6 and a CEST train to
3.5e-7 in Z (benchmark 9: 15015 segments). RF segments add their increments by two-sum; before
that (until M1a) Metal drifted from the core by 1e-3 per 1e4 RF segments. Free segments are not
compensated: a free run of 1e5 segments drifts by ~1.5e-5 in Mz.

Phase resolution: Metal computes each segment's phase increment in float32, so a segment of
|φ| rad carries ~|φ| 2⁻²⁴ (a 5 ms lobe of 32 mT/m seen from 15 cm, ~6400 rad: 4e-4 rad). Imaging
segments are far smaller; long strong lobes on spins far off centre are where it shows.

Metal's adjoint (float32) against the core's float64 autograd, worst over the matrix: 1.2e-4
(`rf_freq`, where the fused kernel's float32 adjoint is 1.5e-3 off), 8e-5 with the bound pool,
≤ 1.4e-5 elsewhere; on a 1600-segment RF-spoiled GRE with 4 coils 2e-5. Gradients are bit-identical
for any `checkpoint_every` and from run to run (no atomics; per-spin sums flushed every 64
segments). Optimizing through it: the WM/GM flip optimum to 0.001° (`examples/optimize_contrast.py
--metal`), the Ernst angle to 0.01°; the DESPOT1 pair only to 0.05° (`examples/design_despot1.py`:
its bound takes finite differences of float32 signals; the float64 engines reach 0.001°).

The core's own gradients match central differences (relative steps of 1e-5 and 2e-5 along a random
direction, Richardson-extrapolated, the case rebuilt at every point) to 1e-6 relative.

## Limits that are not physics

- Metal, free pools: at most five (`metal.MAX_POOLS`; one kernel per count, static and moving),
  not with the generalized Bloch bound pool. A segment costs Taylor sub-steps in proportion to
  |A| dt: an RF sample far off resonance (a CEST M0 offset at −1500 ppm: ~80 per 100 µs) or fast
  exchange over a long free segment (40 kHz over 5 ms: ~280). Static spins are sorted by that cost
  before the dispatch, since a simdgroup waits for its slowest lane (5× at 30k spins on benchmark
  9), and returned in order; moving spins are not (their walks are keyed by spin index).
- Metal: `mma=False` (one `simd_sum` per coil and sample) takes at most 32 coils; the default for
  two or more coils is the matrix-product reduction, which has no limit. Moving spins in mapped
  coils (`FieldMaps.coils` with motion or diffusion) always use the direct reduction, so at most
  32 mapped coils. Threadgroups of 256 (`group`), segments dispatched in chunks of `chunk`.
- Metal adjoint: `checkpoint_every` must be a multiple of 64 (default ~sqrt(segments)); memory: the
  checkpoints (8 floats per spin per interval) plus one interval's block-start states. Forward +
  backward is 6-7x the plain forward on the 3D benchmark cases (`benchmarks/history.jsonl`).
- Fused: needs a C++20 compiler and `ninja` (built on first use).
- Chunked: `m_init`, `scanner` and `maps` raise; the bound pool and free pools raise.
- Core: free pools step segment by segment with one matrix exponential each, so
  `checkpoint_every` and `merge_free_runs` do not apply to them.
