# Bloch Party roadmap

Goals, in order of how much they shape design decisions:

1. **A fun, understandable physics project** – small readable core, every feature verified.
2. **Simulated training data** – realistic coils, noise and imperfections; high throughput.
3. **Sequence development, including agentic** – differentiable w.r.t. sequence parameters,
   Pulseq in/out, machine-readable metrics an agent can act on.

**Ground rule: nothing is "done" without an independent check.** Each phase lists what it is
verified against. References, from cheapest to strongest:
closed-form physics → SciPy ODE integration → our own independent signal-equation /
EPG models → other simulators (MRzero, KomaMRI) on identical Pulseq files → real scanner data.

---

## Phase 0 – MVP ✅

- [x] Sequence as flat piecewise-constant segments (`dt`, RF, gradients, ADC, ideal spoil);
      continuous parts are tensors (differentiable), structure is plain Python (control flow)
- [x] Isochromat Bloch core: exact free precession + relaxation, exact RF rotation with
      Strang-split relaxation; `Sequence.subdivide_rf(k)` as the accuracy knob
- [x] Phantom: tissue-labelled Shepp-Logan with 3 T PD/T1/T2, maps → spins
- [x] Coils: analytic birdcage-like receive maps; complex noise with channel covariance
- [x] 2D Cartesian spoiled GRE builder; centred iFFT, RSS, sensitivity-weighted combine
- [x] Gradient checkpointing (`checkpoint_every`) for memory-bounded backprop

**Verified by** (21 tests, `uv run pytest`):
- SciPy ODE on random RF/gradient/off-resonance sequences: 1e-9 without relaxation;
  measured 2nd-order convergence (≈4× per RF-step halving) with relaxation
- Closed form: flip angle, RF phase / signal sign convention, FID, spin echo refocusing,
  inversion recovery, spoiled-GRE (Ernst) steady state
- Independent signal-equation model: multi-coil k-space agrees to ~1e-13
- Image domain: recon equals tissue-model image; RSS / coil combine; off-resonance
  shift of exactly +2 px for 2 px of bandwidth (and ±1 px or the wrong sign fails)
- Autograd vs finite differences (flip, TR, gradient, T1, T2, coil maps); checkpointing
  gives identical gradients; gradient descent recovers the Ernst angle and the analytic
  WM/GM contrast-optimal flip angle (`examples/optimize_contrast.py`, to 0.001°)

Known simplifications: one spin per voxel at voxel centres, ideal spoiling, hard pulses,
instantaneous gradients, analytic (non-EM) coil maps.

---

## Phase 1 – Physical realism

- [ ] Sub-voxel isochromats (jittered), with guidance on spins-per-voxel vs. gradient moment
- [ ] T2′ via Lorentzian (Cauchy-quantile) intra-voxel frequency distribution
- [ ] Real spoiling: gradient spoilers + RF spoiling (117° quadratic phase), replacing ideal spoil
- [ ] Trapezoidal gradients with ramps; ramp-sampling support
- [ ] Slice-selective shaped pulses (sinc, Gaussian) with slice-select/rephase gradients
- [ ] B1⁺ (transmit) maps scaling the RF; B0 maps; chemical shift (multi-peak fat model)
- [ ] Spin-echo / turbo spin-echo / bSSFP / EPI builders

**Verify:** spoiled-GRE steady state with real spoiling vs. Ernst (and vs. EPG for known
deviations); slice profile vs. small-tip Fourier approximation and SLR; T2′: FID decays
with T2*, spin echo with T2; bSSFP off-resonance banding profile vs. closed form;
fat–water shift in pixels vs. bandwidth; EPI distortion vs. B0 map / echo spacing.
Add an independent EPG implementation as a reference model.

## Phase 2 – Pulseq in/out and cross-simulator validation

- [ ] Read `.seq` (via `pypulseq`) → `Sequence`, including block timing / raster
- [ ] Write `Sequence` → `.seq`, so optimized sequences can run on real scanners
- [ ] Phantom export for KomaMRI (`.phantom` / HDF5) and MRzero
- [ ] `validation/` harness: run the same `.seq` + phantom through blochparty, MRzero
      (`pip install MRzeroCore`) and KomaMRI (Julia), store reference outputs, diff in CI

**Verify:** k-space agreement with MRzero/KomaMRI on GRE, SE, TSE, bSSFP, EPI examples
from the PyPulseq repo; round-trip read→write→read is lossless.

## Phase 3 – Performance

- [ ] Closed-form ADC runs: for constant gradients, compute all samples of a readout in
      one vectorized step instead of one segment per sample (expected large speed-up)
- [ ] Merge consecutive free-precession segments; skip spins outside excited slab
- [ ] `torch.compile` of the step kernel; float32 by default with float64 test mode
- [ ] Apple MPS / CUDA devices (complex-free core already); spin chunking
- [ ] Two-pass, chunked-over-spins backward (forward → dL/ds → per-chunk VJP) so
      memory scales with chunk size, not spin count
- [ ] Benchmarks tracked over time (spins × segments / s, memory)

**Verify:** every optimization must reproduce the Phase 0–2 test outputs bit-for-bit
(float64) or within float32 tolerance; benchmark table in this file.

## Phase 4 – Differentiable sequence design

- [ ] Parameterized sequence API: which tensors are free, bounds, reparameterizations
- [ ] Hardware constraints as penalties/projections: Gmax, slew rate, RF peak / SAR proxy,
      PNS model, timing raster
- [ ] Losses: tissue contrast, CNR / SNR efficiency, PSF / resolution, artifact energy,
      parameter-estimation Cramér–Rao bound (for quantitative MRI)
- [ ] Examples: variable flip angle trains for TSE, RF pulse design, MRF-style schedules

**Verify:** recover known optima (Ernst ✅, WM/GM contrast ✅, VFA-TSE schemes from the
literature, SLR-designed pulses); constraint satisfaction checked by Pulseq's own checks.

## Phase 5 – Agentic sequence development

- [ ] Tool-style API: JSON sequence spec in → structured report out (metrics, constraint
      violations by block, images, PSF), deterministic and fast
- [ ] Sequence linter with actionable messages ("slew exceeded in block 14 by 12 %")
- [ ] Split of labour: agent chooses structure (sequence family, ordering, spoiling);
      gradient descent tunes continuous parameters inside it
- [ ] Benchmark task suite with scored targets (e.g. "T1w brain at 1 mm in < 2 min,
      maximize WM/GM CNR"), to measure agent progress

**Verify:** agent-produced sequences pass Pulseq checks and cross-simulator validation
(Phase 2) before any score counts.

## Phase 6 – Training-data generation

- [ ] Realistic phantoms: BrainWeb tissue maps, XCAT / MRXCAT; random augmentations
- [ ] Physically based coils: Biot–Savart loop arrays (low field), imported EM maps (≥3 T)
- [ ] Motion (rigid, then deformable), flow, diffusion
- [ ] Dataset writer (ISMRMRD / HDF5) with full provenance (sequence, phantom, seed)
- [ ] Fast path: signal-equation forward model for cases where Bloch fidelity isn't needed

**Verify:** noise statistics (SNR maps, g-factor) vs. theory; motion artifacts vs.
analytic ghosting for periodic motion; comparison with real fastMRI-style data statistics.

## Ideas parking lot

Phase distribution graph (PDG) backend for speed; magnetization transfer / multi-pool;
non-Cartesian trajectories + NUFFT recon; gradient non-linearity and eddy currents;
concomitant fields.
