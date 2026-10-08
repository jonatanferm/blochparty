# Engines

Which simulator does what. The capability matrix below is checked cell by cell by
`tests/test_engines.py`, which parses this file: edit a cell and the test checks the new claim.

| engine | entry point | device, precision | gradients | role |
|---|---|---|---|---|
| core | `blochparty.simulate` (`src/blochparty/bloch.py`) | any torch device; float32 or float64 | autograd (`checkpoint_every` bounds memory over segments) | the readable reference: new physics lands here first |
| fused | `blochparty.fused.simulate_fused` (`src/blochparty/fused/`) | CPU, all cores; float32 or float64 | hand-written adjoint, checkpointed | frozen: keeps what it has, new physics raises; retired after ROADMAP M2 |
| metal | `blochparty.metal.simulate_metal` (`src/blochparty/metal/`) | Apple GPU; float32 only (inputs are cast, results come back float32 on the CPU) | none yet (ROADMAP M2): raises when an input requires grad | the engine for imaging-scale work |
| chunked | `blochparty.simulate_chunked` (`src/blochparty/chunked.py`) | as the core | autograd, recomputed one spin chunk at a time | the core with a memory-bounded backward over spins |

## Capability matrix

`✓`: supported, and in `tests/test_engines.py` the engine agrees with the core on a small case
(forward: signal and final magnetization; gradients: the core's autograd against a central
finite difference, the other engines' gradients against the core's autograd). `✗`: raises
`NotImplementedError`. The core's physics is checked against the independent references in the
last column; combinations of features are checked only engine against engine.

| feature | core fwd | core grad | fused fwd | fused grad | metal fwd | metal grad | chunked fwd | chunked grad | reference for the core |
|---|---|---|---|---|---|---|---|---|---|
| basic Bloch: PD, T1, T2, dB0, RF, gradients, ADC, spoils, coils | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | SciPy ODE (`test_bloch_vs_ode.py`), closed forms (`test_analytic.py`, `test_free_runs.py`), EPG (`test_epg.py`), MRzero (`-m crosscheck`) |
| B1+ (`Phantom.b1`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | flip b1·α, signal equation (`test_phantom_b1.py`) |
| FieldMaps, static spins | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | linear maps = the same fields per spin (`test_maps.py`) |
| FieldMaps, moving spins | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | motion through a linear B0 map = a gradient; moved object under mapped coils (`test_maps.py`) |
| bound pool, rate model | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | matrix exponential, CW steady state, SciPy ODE (`test_exchange.py`); BMCTool (`validation/compare_bmctool.py`) |
| bound pool, generalized Bloch | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | reference solver, Henkelman CW limit, rate-model limit (`test_generalized_bloch.py`) |
| free pools (`Phantom.pools`) | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | SciPy ODE of the full system, McConnell two-site exchange (`test_pools.py`); BMCTool |
| free pools + rate bound pool | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | as above (`test_pools.py`) |
| diffusion, free (isotropic or tensor) | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | Stejskal–Tanner, tensor, CPMG vs EPG with diffusion (`test_diffusion.py`) |
| diffusion, restricted | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | planes (Tanner–Stejskal), sphere and cylinder long-time limits (`test_restriction.py`, through Metal) |
| motion (flow, waveform, rigid) | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | phase contrast, rigid = moved phantom, inflow recursion (`test_motion.py`) |
| concomitant fields, leading order | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | Bernstein's phase, Maxwell-compensated echo (`test_fields_concomitant.py`) |
| concomitant fields, exact | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | \|B\| of the coil field in 50-digit arithmetic (`test_fields_concomitant.py`) |
| gradient nonlinearity | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | point positions vs the displacement field (`test_fields_nonlinearity.py`) |
| GIRF / eddy currents (`apply_girf`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | recurrence, delay shift, analytic B0 eddy phase (`test_girf.py`); a sequence transform, so any engine runs it |
| uniform field per segment (`Sequence.b0_offset`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | analytic drift phases (`test_drift.py`) |
| RF carrier (`Sequence.rf_freq`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | resonance sign (`test_exchange.py`); only the rate-model bound pool reads it |
| many coils (40) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | signal linear in the sensitivities |
| initial magnetization (`m_init`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | as the basic case |
| bound pool + scanner field model | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | combination |
| bound pool + diffusion | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | combination |
| generalized Bloch + diffusion | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | combination |
| diffusion + leading-order concomitant fields | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | combination (coefficients at the start position) |
| diffusion + gradient nonlinearity | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | combination (walks in encoding coordinates) |
| diffusion + exact concomitant fields | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | |
| motion + gradient nonlinearity | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ROADMAP M1b |
| motion + concomitant fields | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ROADMAP M1b |
| free pools + generalized Bloch | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ROADMAP M1c |
| free pools + diffusion | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ROADMAP M1c |
| free pools + motion | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ROADMAP M1c |
| free pools + FieldMaps | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ROADMAP M1c; for static spins, sample the maps into `Phantom.db0` / `b1` and the coils |

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
| metal, float32 | signal, magnetization, static spins | 5e-5 | 1.5e-5 signal, 1.7e-5 m (leading-order concomitant fields: gradients 50× the base case's, many cycles of phase) |
| metal, float32 | signal, walking spins | 2e-4 | 7.4e-5 (diffusion + gradient nonlinearity, spins up to 10 cm from isocentre) |
| metal, float32 | per-spin magnetization, walking spins | 3e-3 | 1.1e-3 (5 µm cylinders up to 5 mm from isocentre) |

Why Metal is looser with walking spins: the kernel keeps each walker's absolute position in
float32, so a spin 5 mm from isocentre is resolved to ~3e-10 m, and a compartment's walls (offset
r − centre) to that as well; the per-spin phase carries the rounding, the signal averages it. A
walk in offsets from the start position would remove most of it.

The core's own gradients match central differences (step 1e-6 relative to each input, the case
rebuilt at both points) to 1e-6 relative.

## Limits that are not physics

- Metal: `mma=False` (one `simd_sum` per coil and sample) takes at most 32 coils; the default for
  two or more coils is the matrix-product reduction, which has no limit. Moving spins in mapped
  coils (`FieldMaps.coils` with motion or diffusion) always use the direct reduction, so at most
  32 mapped coils. Threadgroups of 256 (`group`), segments dispatched in chunks of `chunk`.
- Fused: needs a C++20 compiler and `ninja` (built on first use).
- Chunked: `m_init`, `scanner` and `maps` raise; the bound pool and free pools raise.
- Core: free pools step segment by segment with one matrix exponential each, so
  `checkpoint_every` and `merge_free_runs` do not apply to them.
