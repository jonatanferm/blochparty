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
- [x] Two-pass, chunked-over-spins backward (`simulate_chunked`): forward without grad,
      then per-chunk recompute + VJP; saved-tensor memory scales with the chunk (16× less
      at N/16), gradients identical to ~1e-16, ~1.5–4× runtime depending on chunk size
- [ ] Reverse sweep over segment blocks (store magnetization at block boundaries,
      `simulate(..., m_init=...)` per block) to remove autograd's per-step bookkeeping
      floor, which `checkpoint_every` does not
- [x] Benchmarks tracked over time (spins × segments / s, memory):
      `benchmarks/bench.py [--record]` → `benchmarks/history.jsonl`
- [x] Fused compiled kernel, forward (`blochparty.fused.simulate_fused`): header-only C++17
      kernel on raw arrays (`fused/kernel.h`, no PyTorch dependency), spin blocks in L1,
      `std::thread` workers, per-segment factor caching, SIMD coil sums; JIT-built PyTorch
      extension. Matches `simulate` to ~2e-14 (float64); float32 error ~1e-6 vs float64
      (10× lower than the PyTorch float32 path, thanks to `expm1` relaxation)
- [x] Fused kernel, backward (`simulate_fused` is now differentiable): hand-written adjoint
      in `kernel.h` (reverse sweep λ ← Aᵀλ + ADC terms per segment, parameter gradients
      from λ and the replayed m), checkpoints every ≈√S segments stored by the forward,
      one interval replayed at a time; per-thread segment accumulators reduced in thread
      order. Verified against autograd through `simulate` for every input (sequence dt /
      RF / gradients / receiver phase, pos / PD / T1 / T2 / dB0, complex coils, `m_init`)
      to ≤ 1e-10 (measured ~1e-14) on random RF / gradient / off-resonance sequences, 8-coil
      2D GRE, 3D GRE with dB0, spoils and ADC right after RF, losses on signal and/or m;
      gradcheck (finite differences) incl. complex coils; standalone C++17 build vs
      finite differences; identical gradients for any checkpoint interval; gradient
      descent through it recovers the WM/GM optimum (16.109°) to 0.002°.
      Apple M4, float32, forward + backward: 2D 96² 0.08 s (PyTorch core 0.32 s); 3D 32³
      0.46 s / 1 GB peak (PyTorch core 3.6 s / 3.5 GB); 3D 64³ 27.5 s.
- [ ] Profile the adjoint on the M4: backward costs ~7× the forward there (3D 64³: 3.3 s
      forward, 27.5 s total) vs ~3.8× on a Xeon; ADC λ-injection and coil-gradient loops
      run at about half the forward sampling speed; the RF-segment loop is not vectorized
      by Apple clang
- [ ] Metal / CUDA ports of the fused kernel (forward and adjoint)

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

Fused kernel (same machine, float32, 10 threads, idle; vs. the closed-form PyTorch core):

| case | PyTorch core | fused | speed-up |
|---|---|---|---|
| 2D 96² | 0.16 s | 0.012 s | ~13× |
| 3D 32³ | 1.14 s | 0.07 s | 16× |
| 3D 48³ | 9.3 s | 0.80 s | 12× |
| 3D 64³ | 60 s | 5.4 s (float64: 12.6 s) | 11× |

Gradients (forward + backward of Σ|signal|² w.r.t. flip angle, T1 map and coil maps; float32,
8 coils, 200 dummy TRs; Intel Xeon Cascade Lake, 4 vCPUs, Linux, GCC 13; peak RSS of the
whole process, ~0.5 GB of which is Python + PyTorch; `benchmarks/bench.py --grad`):

| case | PyTorch core (autograd) | `simulate_chunked` | fused adjoint | fused forward only |
|---|---|---|---|---|
| 2D 96² | 1.65 s, 1.1 GB | 4.2 s, 1.2 GB | 0.15 s, 0.53 GB | 0.03 s |
| 3D 32³ | 19.5 s, 5.1 GB | 41 s, 2.8 GB | 1.0 s, 0.57 GB | 0.27 s |
| 3D 48³ | not run (~50 GB extrapolated) | 258 s, 7.9 GB | 9.1 s, 0.74 GB | 1.8 s |
| 3D 64³ | not run (~300 GB extrapolated) | not run (~17 GB extrapolated) | 46 s, 1.2 GB | 9.4 s |

Targets: 64³ < 1 min ✅ (5.4 s); 128³ < 30 min on CPU ✅ — full run (565k spins × 2.16M
segments, 8 coils, float32) in 6.3 min at 3.3e9 spin-segments/s (`examples/gre3d_128.py`).

## Phase 2 – Physical realism

- [ ] Sub-voxel isochromats (jittered), with guidance on spins-per-voxel vs. gradient moment
- [ ] T2′ via Lorentzian (Cauchy-quantile) intra-voxel frequency distribution
- [x] Real spoiling: gradient spoilers (net moment per TR in cycles per voxel, phase-encode
      rewinders) + RF spoiling (117° quadratic phase, receiver follows), opt-in on the GRE
      builders (`spoiling="gradient" | "rf"`); ideal spoiling stays the default
- [x] Hardware limits (`System`: max gradient / slew per axis, gradient / RF / ADC rasters,
      RF dead time and ringdown) and trapezoidal gradients with ramps (`trapezoid`: by area,
      flat area or amplitude, minimum time or fixed duration, any axis or direction; the
      minimum-time designs equal pypulseq's), arbitrary waveforms (`gradient_waveform`),
      Pulseq-like blocks (`SequenceBuilder.block`) cut on the gradient raster with exact
      per-segment average gradients; `ramps=True` on the GRE builders
- [ ] Ramp-sampled readouts in the builders (ADC on ramps is already exact in k) + gridding recon
- [x] Slice-selective windowed-sinc pulses (`sinc_pulse`: time-bandwidth product, apodization,
      slice offset as a frequency offset) with slice-select gradient and ramp-aware rephaser;
      `slice_thickness=` on the builders (slab in 3D)
- [ ] Gaussian / SLR pulses
- [ ] B1⁺ (transmit) maps scaling the RF; B0 maps; chemical shift (multi-peak fat model)
- [x] Spin-echo / turbo spin-echo / bSSFP / EPI builders (`se_2d`; `tse_2d` with arbitrary
      refocusing flips, ETL, linear / centric ordering; `bssfp_2d` with α/2 preparation and
      phase cycling; `epi_2d` single-shot blipped, blips on the readout ramps)

**Verify:** spoiled-GRE steady state with real spoiling vs. Ernst (and vs. EPG for known
deviations); slice profile vs. small-tip Fourier approximation and SLR; T2′: FID decays
with T2*, spin echo with T2; bSSFP off-resonance banding profile vs. closed form;
fat–water shift in pixels vs. bandwidth; EPI distortion vs. B0 map / echo spacing.
✅ Independent EPG reference model (`epg.py`): matches closed forms to 1e-12–1e-15 and the
isochromat simulator exactly once isochromats outnumber the dephasing orders present
(CPMG 120°: K=17 → 3e-16; 117° RF-spoiled GRE: K=151 → 6e-16).

**Verified** (realistic sequences; `tests/test_sequences_*.py`, 49 tests, ~1 min):
- Gradient moments: trapezoid areas and k-space at every ADC sample of GRE (2D / 3D slab,
  ramps, slice select, spoilers), SE / TSE (moment negated at each refocusing pulse), bSSFP
  (zero net moment per TR, TE = TR/2 to 1e-12 s) and EPI equal the analytic values to ≤ 1e-9 /m;
  an arbitrary waveform with off-raster ADC samples matches its integral. A pypulseq-written
  RF-spoiled, slice-selective GRE (pypulseq's own sinc / readout designs) read with
  `read_pulseq` gives the same moments as pypulseq's `calculate_kspace` (≤ 1e-9) and the same
  simulated signal to 1e-9; the sinc pulse equals `make_sinc_pulse` sample by sample.
- Slice profile: small-tip profiles equal the Fourier transform of the RF envelope up to the
  expected α²/6 (2.5° / 5° / 10°: 3.2e-4 / 1.27e-3 / 5.07e-3); a 90° profile (also off-centre)
  equals SciPy ODE integration to 2e-12; phase flat to 7e-4 rad after the rephaser at 5°
  (dropping the fall ramp from it gives
  the predicted linear phase to 1 %); the slice sits at +z0 for f = G·z0 (flipped sign: −z0).
  At 90° the half-area rephaser leaves the physical ~0.2 rad odd residual phase the ODE shows.
- Spoiling: RF + gradient spoiled GRE (ramps, 4 cycles per voxel per TR, 60 TRs) on K
  isochromats across a voxel vs `epg_spoiled_gre`: K = 16 / 32 / 64 / 128 → 7e-2 / 2.5e-2 /
  5.6e-3 / 2.6e-4, exact (≤ 1e-16) once K / 4 reaches the TR count (K = 240); gradient-only
  spoiling (1 cycle): K = 16 / 32 / 48 → 1e-4 / 4e-5 / 4e-6, K = 64 → 2e-16. On
  Shepp-Logan with 4 jittered isochromats per voxel, the full EPG field evaluated at each
  isochromat predicts the multi-coil k-space to 1e-10.
- TSE vs `epg_cpmg` (ETL 8, crushers of one cycle): 180° exact for any K (4e-15); 120°:
  K = 4 / 8 / 12 / 16 → 0.13 / 0.11 / 7e-3 / 1.4e-5, K = 17 → 8e-16; a variable train
  (60–170°): 0.32 / 1.3e-2 / 2e-4 / 5e-7, K = 17 → 9e-16. SE / TSE Shepp-Logan k-space vs a
  closed-form Mz recursion + signal equation: < 1e-10.
- bSSFP vs `epg_bssfp` with off-resonance 0–160 Hz: ≤ 2e-8 (1 ns pulses); α/2 prep vs EPG
  < 1e-7, with ≥ 20× less transient oscillation; steady-state banding profile vs the closed
  form < 1e-7; Shepp-Logan k-space vs per-tissue EPG < 1e-10. EPI k-space vs the signal
  equation < 1e-10; a uniform offset of 2 PE pixels' bandwidth (1 / (N·esp)) shifts the
  image by +2 rows (residual 0.75 %, from the 0.02 px readout shift; ±1 px or −2 px: ≥ 35 %).
- Default builders: bit-identical sequences to before (sha256); autograd through flip angles
  of the new blocks matches finite differences.

## Phase 3 – Pulseq in/out and cross-simulator validation

- [x] Read `.seq` (via `pypulseq`) → `Sequence`, including block timing / raster
      (`read_pulseq`): blocks split at every event, exact segment-average gradients, shaped
      RF, trapezoids / arbitrary / extended gradients, frequency / phase / ppm offsets
- [ ] Write `Sequence` → `.seq`, so optimized sequences can run on real scanners
- [ ] Phantom export for KomaMRI (`.phantom` / HDF5) and MRzero (the harness builds
      MRzero's `SimData` in memory from blochparty's maps)
- [x] `validation/` harness, MRzero part: the same `.seq` + phantom through blochparty and
      MRzero (`validation/compare_mrzero.py` → `validation/RESULTS.md`, opt-in
      `pytest -m crosscheck`)
- [ ] The same against KomaMRI (Julia); store reference outputs and diff them in CI

**Verified** (`tests/test_pulseq.py`, 13 tests, ~5 s; `pytest -m crosscheck`, 4 tests):
- Pulseq timing and units vs closed forms: FID samples at (i + ½)·dwell after the ADC start
  (half a dwell off would fail by 300×), flip, RF/ADC phase, T2 decay, off-resonance;
  ADC frequency offset tracks off-resonant spins; ppm offsets; ADC phase modulation
- Phase convention pinned by Pulseq's own definition: a sinc pulse with freq_offset = G·z0
  excites the slice at +z0, identical to the centred slice (to 1e-5) with the usual
  zero-phase-at-centre offset; the "mrzero" reading is the exact mirror image (1e-12)
- Gradient moment at every ADC sample vs pypulseq's `calculate_kspace`: GRE with slice
  select 9e-11 /m, EPI with and without ramp sampling 6e-13 /m, arbitrary + extended
  gradients ≤ 1e-9 /m (|k| up to 36 /m)
- A pypulseq GRE equals the same GRE built with `SequenceBuilder` to 1e-10
- pypulseq 2D GRE of Shepp-Logan, imported and reconstructed, equals the tissue-model image
  to 1.1e-4 (T2 decay during the readout)
- MRzero, 32×32, point isochromats, RF-spoiled GRE / spin echo / bSSFP / EPI: k-space
  agreement 2e-6–4e-5, image NRMSE 2e-6–7e-5, after the convention factor −i (fitted to
  1e-5). An exact matrix-exponential Bloch solution on sampled voxels puts the residual on
  MRzero (float32, PDG state merging): blochparty is within 4e-9–6e-7 of it. MRzero's box
  voxels vs. K×K jittered isochromats: 5e-4 k-space at 16×16 on EPI (falls as K⁻²)

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
