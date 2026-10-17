# 2026-10-10 — Fast realism mode and PERFORMANCE.md items 2.1, 2.4, 2.6, 2.7

- **Why**: the realism ablations took 98-162 s per slice; most testing time went into waiting.
- **Fast mode** (9e1a08a, ee03e81, af4c844): `simulate_heads.py --quality full|fast|legacy`
  (8 or 4 categorical spins per voxel, a ±3 mm slab, RF on 100 µs: image RMS 0.014 / 0.022
  against 27 spins per tissue-voxel, seed-to-seed 0.007); far slices' pulses as one segment each
  with merged delays (42 462 → 23 212 segments); `to_phantom` texture and R2' only where the spins
  are (10.7 → 1.3 s per slab, bit for bit).
- **2.4 merge free stretches** (1cb64c0, `Sequence.merge_free`): exact for static spins (core
  2e-15, Metal 2e-6); realism TSE 36 335 → 19 726 segments; 400k spins, 16 coils: 1.98 → 1.33 s
  single slice, 5.26 → 4.58 s with MT and 16 slices.
- **2.6 fewer spins** (3c80893): systematic tissue assignment = categorical (RMS 0.0139 vs 0.0138
  at 8 spins) and unjittered z-lattices = random spins at equal count; the predicted 4x from
  lattice quadrature along the crusher axis did not appear. Kept as an option, no gain.
- **2.7 overlap** (71da289): the next slice's phantom is built while the GPU runs: fast single
  slice 2.3 s on the laptop (legacy ~128 s).
- **Correction — bound-only pulses in float32** (ebabdaa): the far slices' pulses hid from the water
  by a 1 MHz frame offset (identity in float64). On Metal each 100 µs pulse left ~1e-3 rad, which
  CSF (T2 2 s) accumulated: Metal vs core 0.45 % in signal, 2 % in Mz. `Sequence.bound_only` now
  flags RF that only the bound pool feels (core, Metal forward; chunked, Pulseq export,
  generalized Bloch and free pools refuse it): CSF 6.3e-6, WM with MT 6.8e-7; core vs every pulse
  exact unchanged at 0.71 %. The f1/f2 realism runs started with the old pulses were stopped.
- **2.1 NUFFT readouts** (6b8814c, `simulate_metal(readouts="nufft")`): every readout as a
  type-1 NUFFT (ES kernel w = 6 on a 2x grid, 3 time-segmentation terms) with spreading through
  the coil MMA. One realism slice (16 coils, laptop): 3.83 → 1.51 s (2.5x) fast, 7.53 → 3.10 s
  full; with 16 slices and MT 10.4 → 6.3 s (1.6x, the slices' RF dominates there). Signal 3e-6
  from the per-sample path, images 3e-5. A dead end on the way: the first run detector compared
  dt exactly and found no runs, so "NUFFT vs per-sample" (2.5e-5) was atomic-add nondeterminism
  of the per-sample path twice (LESSONS).
- Defaults bit-identical throughout (snapshots); full suite on the mini: 766 passed before
  bound_only and NUFFT.
- Not started: 2.2 periodicity, 2.3 tabulated pulses (the multi-slice RF share), 2.5 reversible
  adjoint.
