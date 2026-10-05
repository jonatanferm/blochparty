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

**Verified by** (21 tests at the end of Phase 0, `uv run pytest`):
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

## Phase 1 – Performance and 3D

Comes first: 3D imaging and most of the realism work (sub-voxel spins multiply the spin
count) are impractical at current speed. 3D is the target workload for this phase.

- [x] 3D: volumetric phantom (3D Shepp-Logan to start), `gre_3d` builder (second
      phase-encode loop on Gz), `ifftnc` / `reconstruct` reconstruction

- [x] Closed-form free runs: every stretch of RF-free segments (prephaser + readout +
      spoiler) is one step; all its ADC samples come from blockwise spins × samples matrix
      products (any gradient waveform, gradients flow to every segment). 3.5–9.5× faster;
      `merge_free_runs=False` keeps the segment-wise path, agreement ~1e-14
- [ ] Merge consecutive free-precession segments; skip spins outside excited slab
- [ ] `torch.compile` of the free-run block: measured 1.4× (CPU) / 2× (MPS) on the 64³
      per-TR math; not wired in yet (compile time, shape-dependent recompiles)
- [ ] Apple MPS / CUDA devices: runs unchanged on MPS, but eager MPS is no faster than CPU
      (elementwise exp/cos/sin and the K=3 matmul are slow there); revisit with fusion
- [ ] Two-pass, chunked-over-spins backward (forward → dL/ds → per-chunk VJP) so
      memory scales with chunk size, not spin count
- [x] Benchmarks tracked over time (spins × segments / s, memory):
      `benchmarks/bench.py [--record]` → `benchmarks/history.jsonl`
- [ ] If pure PyTorch isn't fast enough: a fused compiled kernel (C++ extension; later
      Metal / CUDA) behind the same `simulate()` API — one thread per spin block stepping
      through all segments with magnetization in registers, plus a hand-written adjoint
      backward with checkpointing. The pure-PyTorch simulator stays as its reference
      (forward to ~1e-12 in float64, gradients vs. autograd and finite differences).
      Written against raw arrays so a standalone CLI (no Python) can wrap it later.

**Verify:** every optimization must reproduce the existing test outputs bit-for-bit
(float64) or within float32 tolerance; benchmark table in this file. 3D: Bloch vs.
signal-equation k-space and recon vs. tissue model, as for 2D; small 3D cases also against
the unoptimized simulator.

Targets (to be revisited once measured): 64³ 3D GRE in < 1 min on CPU; 128³ in
< 30 min on CPU and a few minutes on GPU. The work is fundamentally spins × ADC samples
(~4e12 for 128³), so the final factor comes from the fused kernel / GPU.

Baseline (end of Phase 0, Apple M4 CPU, float32, 8 coils, 200 dummy TRs; PyTorch picks
4 threads on this 10-core chip):

| case | spins | segments | time | spin-segments/s |
|---|---|---|---|---|
| 2D 96² | 4 603 | 10 000 | 1.6 s | 2.8e7 |
| 3D 32³ | 8 803 | 37 264 | 7.8 s | 4.2e7 |
| 3D 48³ | 29 831 | 120 208 | 44 s | 8.2e7 |
| 3D 64³ | 70 505 | 278 928 | 233 s | 8.4e7 |

Projected 3D 128³: ~4 h.

After closed-form free runs (same machine, CPU): 2D 96² 0.17 s (9.5×), 3D 32³ 1.2 s (6.3×),
3D 48³ 10.9 s (4.0×), 3D 64³ 68 s (3.5×); projected 128³ ~70 min. Per 64³ TR (~12 ms):
exp/cos/sin ~45 %, phase ~17 %, elementwise products ~20 %, matmuls ~17 %. ~20 µs fixed Python overhead per segment; memory-bandwidth
bound at realistic sizes; ~92 % of segments are single ADC samples.

## Phase 2 – Physical realism

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
✅ Independent EPG reference model (`epg.py`): matches closed forms to 1e-12–1e-15 and the
isochromat simulator exactly once isochromats outnumber the dephasing orders present
(CPMG 120°: K=17 → 3e-16; 117° RF-spoiled GRE: K=151 → 6e-16).

## Phase 3 – Pulseq in/out and cross-simulator validation

- [ ] Read `.seq` (via `pypulseq`) → `Sequence`, including block timing / raster
- [ ] Write `Sequence` → `.seq`, so optimized sequences can run on real scanners
- [ ] Phantom export for KomaMRI (`.phantom` / HDF5) and MRzero
- [ ] `validation/` harness: run the same `.seq` + phantom through blochparty, MRzero
      (`pip install MRzeroCore`) and KomaMRI (Julia), store reference outputs, diff in CI

**Verify:** k-space agreement with MRzero/KomaMRI on GRE, SE, TSE, bSSFP, EPI examples
from the PyPulseq repo; round-trip read→write→read is lossless.

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
