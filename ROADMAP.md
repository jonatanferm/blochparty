# Bloch Party roadmap

Goals, in order of how much they shape design decisions:

1. **A fun, understandable physics project** – small readable core, every feature verified.
2. **Simulated training data** – realistic coils, noise and imperfections; high throughput.
3. **Sequence development, including agentic** – differentiable w.r.t. sequence parameters,
   Pulseq in/out, machine-readable metrics an agent can act on.

**Physical realism**: what other simulators model, what nobody models, and the plan to go
beyond them (real-data benchmark, calibrated digital twin) is in
[docs/REALISM.md](docs/REALISM.md).

**Digital phantoms**: generative, physically consistent phantoms, and lifting reconstructed
scans (DICOM) to believable raw multi-coil k-space, are in [docs/PHANTOMS.md](docs/PHANTOMS.md).

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
- [x] Metal forward kernel (`blochparty.metal.simulate_metal`): Apple GPU via
      `torch.mps.compile_shader`, same physics as the CPU paths (float32), block-buffered
      ADC reductions. Agrees with the float64 core like the CPU kernel does in float32 (tests:
      `tests/test_metal.py`). 5.8× the CPU fused kernel on the real NIST TSE (85 k spins, 4.1 M
      segments, 1 coil: 40 s vs 232 s); slower than the CPU on 8-coil GRE benchmarks, where the
      per-sample coil reduction dominates
- [x] Metal many-coil ADC reduction: the coil sum as complex matrix products on 8×8 simdgroup
      matrices; cost nearly flat in the coil count (48³ GRE: 1 coil 0.41 s, 8 coils 0.42 s
      (was 1.21 s; CPU 1.98 s), 64 coils 1.6 s), no coil limit
- [x] Metal: scanner field model (gradient nonlinearity via encoding positions; concomitant fields,
      leading order and exact) in the off-resonance, as the CPU kernel; matches the core to 1e-4
- [ ] Metal: adjoint
- [ ] CUDA port

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

- [x] Sub-voxel isochromats (jittered), with guidance on spins-per-voxel vs. gradient moment
      (`phantom_from_maps(spins_per_voxel=, jitter="stratified")`, PD split over the spins)
- [x] T2′ via Lorentzian (Cauchy-quantile) intra-voxel frequency distribution (`t2prime` map;
      quantiles shuffled per voxel, or seeded random draws)
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
- [x] B1⁺ (transmit) maps scaling the RF (`Phantom.b1`, in the PyTorch core, signal equation,
      `simulate_chunked` and the fused kernel forward + adjoint; `coils.birdcage_b1`); B0 maps
      (`b0_shim`, `b0_sphere`, `b0_cylinder`); chemical shift (6-peak fat model, `fat_fraction`)
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

**Verified** (realistic phantoms: `tests/test_phantom_*.py`, 53 tests, ~20 s):
- Defaults unchanged: one spin per voxel centre, and the fused forward and adjoint are
  bit-identical to before the B1⁺ change (float32 / float64, one thread, 6 sequences incl.
  random RF/gradients/off-resonance, 2D/3D GRE, spoils; also `b1 = 1` vs `None`)
- Sub-voxel spins: layout (one spin per sub-cell, PD conserved); k-space of 3×3 jittered spins
  equals the signal equation for the same spins to 4e-15; gradient-spoiled GRE (no RF spoiling,
  150 TRs, 1 cycle/voxel/TR) vs EPG: regular sub-grid 0.26 (K=2) → 3.7e-7 (64) → 1e-15 (151,
  once K exceeds the orders present); stratified jitter 0.27 → 4e-3 (64) → 3.5e-4 (256), ~K^-1.3
- T2′: spin offsets are the Cauchy quantiles (vs SciPy), random draws pass a KS test; FID vs
  exp(−t/T2*) over 3 T2′ within 0.36/√K, 0.25/√K, 0.15/√K for K = 256, 1024, 4096 (bound
  0.6/√K), T2′ alone within 2/√K out to K·T2′/4; deterministic quantiles beat random draws;
  spin echo with 1024 spins follows exp(−TE/T2) to 1.4e-7 while the FID has decayed with T2*
- B1⁺: Mz = cos(b1·α) to 9e-16 for b1 ∈ [0, 3.7] and any RF phase (core, fused, chunked);
  2D GRE with a centre-brightened birdcage map equals the Ernst signal at b1·α per pixel to
  4.5e-5; `b1` gradient (and all others) of the fused adjoint vs autograd 6e-15–2e-13
  (≤ 1e-10 required), gradcheck through core and fused, chunked backward too; `birdcage_b1`
  solves the Helmholtz equation with the k from SciPy's J0 (1e-5, finite differences)
- B0: shim terms are harmonic (Laplacian 0); sphere and cylinder (θ = 0°, 30°, magic angle,
  90°) vs a Fourier-domain dipole convolution of a partial-volume χ grid within 1 % / 0.5 % of
  the peak field; a linear y shim shears a GRE image by exactly f(y)/bandwidth per row
- Fat: PD / shift bookkeeping (−434.3 Hz main peak at 3 T); fat-only image moves −2 px at
  bandwidth = |Δf|/2 (+2, −1, −3 fail); each peak modulates k-space by exp(−i2πf_p t) (1e-4,
  0.1 µs pulses); water + fat FID at in-/opposed-phase and other TEs equals
  i[(1−f)e^{−t/T2w} + f e^{−t/T2f} Σ a_p e^{−i2πf_p t}] to 2e-7 at 1.5 T and 3 T (fused
  too); single peak gives the two-point Dixon values and fat at +90° at t = 1/(4|Δf|)
- All of it together (sub-voxel + T2′ + fat + B1⁺ + shim + sphere, 2D and 3D, 6 coils): fused
  vs core 1e-12 (float64) / 1e-5 (float32), chunked 1e-12; gradients of every input ≤ 4e-15,
  and back through `phantom_from_maps` to the PD, T1, T2′, fat-fraction, fat-T2, B1⁺, B0 maps
- Cost: 3D 64³ GRE (8 coils, float32, 200 dummies) with 8 spins per voxel (564k spins), fused
  forward 61 s (59 s with T2′ and a B1⁺ map) vs 7.3 s at one spin per voxel, ~2.6e9
  spin-segments/s either way (Intel Xeon, 4 vCPUs)

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

## Phase 2c – Spin system: exchange and magnetization transfer (`docs/REALISM.md` R2)

- [x] Bound (semi-solid) pool: Henkelman two-pool model with Graham's rate W = ω1²G
      (super-Lorentzian with a per-pool cut, Lorentzian), saturated at each spin's offset from
      the RF's instantaneous frequency, so slice-selective pulses cause incidental MT
      (`exchange.py`, `Phantom.bound`). In the PyTorch core, the fused CPU kernel (forward and
      adjoint, 1.45× the plain cost) and the Apple GPU (+15 %)
- [x] Free exchanging pools (`Phantom.pools`: CEST, myelin water): exact Bloch–McConnell
      engine in the PyTorch core (one matrix exponential per segment)
- [x] Generalized Bloch model of the bound pool (`generalized_bloch.py`, Assländer et al.):
      a pulse-level reference solver
- [x] Free pools in the fused kernel (`fused/bm.h`): the same exact engine per spin, forward and
      adjoint (Fréchet derivative of the exponential by the block-matrix trick); ~1e-13 vs autograd
- [ ] Free pools on Metal (a dense per-thread matrix exponential: little gain for the few-spin
      workloads pools are used for); a faster engine for imaging with pools
- [x] Generalized Bloch in the engines (`BoundPool(model="generalized")`): the Green's function as
      complex exponential modes (super-Lorentzian with a 100 T2 memory taper: 30 modes, error
      2.4e-5); under RF the pool rotates like a Bloch pool and each mode takes its share. In the
      core, the fused kernel (forward and adjoint, ~1e-14 vs autograd) and on the Apple GPU (a
      separate kernel variant, so the plain kernels keep their speed)
- [ ] Incidental MT validated on data with a known fat-saturation pulse

**Verified**: lineshape vs SciPy quadrature (1e-11); exchange step vs matrix exponential;
two-pool ODE with second-order convergence; CW steady state; gradcheck; fused vs core
(1e-15 forward, every gradient ≤ 7e-13); Metal vs core (2e-6). **BMCTool**
(`validation/RESULTS_BMCTOOL.md`): Lorentzian Z-spectra to 5.8e-7 (2.1e-5 through the
`.seq` file); its 5-pool CEST model to 8.3e-7; the super-Lorentzian differs only by the
near-resonance convention (with BMCTool's lineshape ≤ 4e-7). Free pools vs the SciPy ODE
(< 1e-8) and closed-form two-site exchange. Generalized Bloch: the Bloch equations for a
Lorentzian (second order), Henkelman's steady state for three lineshapes (≤ 2e-4), the
hard-pulse rotation. Incidental MT vs Shin et al. 2009: the mechanism and trend, with a
magnitude set by an unreported pulse (`validation/RESULTS_INCIDENTAL_MT.md`).

## Phase 2b – Scanner field model (`docs/REALISM.md` R1)

- [x] Concomitant (Maxwell) fields, leading order as a per-spin quadratic form (free runs stay
      closed form via cumulative ∫gᵢgⱼ dt) and exact |B| of the full vector field
      (`fields.Scanner(field_strength=, concomitant="leading" | "exact", z_asymmetry=)`)
- [x] Gradient nonlinearity: per-axis solid-harmonic deviation (gradunwarp's Siemens
      convention), coefficient files (`A( n, m)` `.grad` / `.coef` lines, `alpha_x n m` lines),
      encoding positions f(r) = r + d(r) (`fields.GradientNonlinearity`); ✗ b-matrix
- [x] GIRF and eddy currents as a sequence transform (`girf.apply_girf`): delta-train impulse
      responses on the raster or transfer functions, delays, cross terms and B0 rows;
      multi-exponential eddy currents (amplitudes, time constants differentiable)
- [x] Per-segment uniform field `Sequence.b0_offset` (None = zeros) in both kernels, forward and
      adjoint, and in `simulate_chunked`
- [x] Drift (linear, exponential) and respiratory fields (uniform + linear, periodic) as
      sequence transforms (`drift.py`)
- [ ] Concomitant terms of nonlinear coils; b-matrix with nonlinearity; drift driven by
      gradient duty; mechanical resonance models; fitting a GIRF / eddy model to measured data

**Verified** (`tests/test_fields_*.py`, `tests/test_girf.py`, `tests/test_drift.py`, 48 tests, ~1 min):
- Defaults unchanged: `scanner=None` and no `b0_offset` give bit-identical signals, final
  magnetization and gradients of every input vs `main` (184/184 tensors: core, checkpointed
  core, fused at one thread; float64 / float32; random busy sequence with B1⁺ and coils, 2D
  RF-spoiled GRE with ramps, 3D GRE); a zero-deviation coefficient file is bit-identical too.
- Concomitant: the coil field obeys div B = curl B = 0 (autograd Jacobian traceless and
  symmetric); the quadratic form equals Bernstein's formula to 2e-16 and is the 1/f0 term of
  |B| − f0 − g·r evaluated in 50 digits; exact − leading falls as 1/B0² (ratio per doubling
  3.998 / 3.999, 0.55 → 2.2 T) and equals −t²·(g·r)/(2f0²) to < 5 %; the stable |B| form
  matches 50 digits to 2e-16. Bipolar (phase-contrast) pair on x and z at 0.55 T: phase =
  Bernstein's analytic −2π f_c·2δ to 4e-12 (phases up to 21 rad, core and fused); trapezoids
  equal the raster-staircase ∫g² to 5e-12 and the true ramps to 1e-3. A Maxwell-compensated
  spin echo (equal area and ∫g² either side of the 180°) leaves 4e-13 rad; the uncompensated
  one leaves up to 2.8 rad, equal to the analytic residual to 9e-12. Exact mode vs |B| in 50
  digits 1e-13 at 0.05 / 0.1 T (exact − leading up to 0.33 rad, ratio 3.8–4.2 per halving).
  Fused vs core 2e-14 (signal), gradients of B0, positions, db0, gradients, dt, b0_offset
  ≤ 3e-14 (both modes); d/dB0 vs central differences 1e-9.
- Nonlinearity: basis vs gradunwarp's `siemens_B` (spherical coordinates, SciPy `lpmv`,
  Siemens normalisation) 9e-16 to n = 10; Laplacian of all 242 basis fields to n = 10 ≤ 1.5e-15
  of their second derivatives; synthetic coefficient file in both layouts (R0 in m or mm);
  linear terms act as exact gradient scale errors (1e-12); a 7×7 point grid at z = 4 cm with
  up to 2.4 px distortion images at the analytically mapped positions to 0.002 px (core and
  fused; the ideal scanner at the true grid); coefficient gradients fused vs core 5e-15, vs
  central differences 2e-9 (with concomitant fields on as well).
- GIRF / eddy: a per-axis delay (3 and 23.5 µs, also negative) gives k-space equal to the
  nominal moment interpolated at t − d to 1e-11 /m (|k| ≤ 7000 /m); a single exponential on a
  trapezoid equals the closed-form staircase response to 3e-15 (of a·G) and the true-ramp
  response to 3e-4, the moment lost by the end of the plateau matches the closed form to 1e-6
  relative and vanishes after decay (1e-12 of 896 /m); random taps with cross terms and B0
  rows equal an FFT transfer-function convolution to 2e-14 (also via
  `from_transfer_function`); a B0 eddy current gives the analytic uniform phase to 3e-14
  (core and fused, up to 0.43 rad); eddy amplitude / time-constant / B0-term / delay gradients
  fused vs core 7e-15, vs central differences 2e-6; the blocked scan equals the recurrence
  to 1e-12; `simulate_chunked` carries `b0_offset` (1e-12).
- Drift / respiration: phases equal the closed-form field integrals (uniform + linear terms) to
  ≤ 3e-14 for linear drift (32 rad), exponential drift (16 rad) and respiration (3.4 rad), core
  and fused; parameter gradients (amplitude, period, phase, rate, τ) vs central differences
  1e-6 relative, fused vs core 1e-10.
- Cost (fused, 3D 32³ GRE, 8 coils, float32, 200 dummies, 8.8k spins × 37k segments, 4 vCPUs):
  ideal 0.21 s forward / ~0.9 s forward + backward; concomitant leading 0.20 / 0.87 s, exact
  0.24 / 1.07 s; nonlinearity (order 5) 0.22 / 0.75 s (all within noise: the per-segment factors
  are cached by gradient); time-varying `b0_offset` (respiration) 0.31 / 0.98 s (a uniform
  offset is one rotation per segment on top of the cache); GIRF + eddy currents 1.4 / 4.8 s,
  since every segment's gradient then differs (`apply_girf` itself 0.14 s).

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

## Realism benchmark against real scanners (docs/REALISM.md, R3)

- [x] Benchmark 1 (`validation/real/`): ISMRM/NIST phantom T1 plane, 3 T IR spin echo, the
      scanner's own Pulseq files and raw data (Tong et al. 2022). With reference T1/T2 only
      and nothing fitted, the inversion-recovery curves are predicted to 1.49 % (spheres
      1–10) vs 25.9 % for the analytic formula; converged in spins per slice (`RESULTS.md`)
- [x] Benchmark 2 (`nist_tse.py`): the T2 plane, 23-echo CPMG. Echo-train shapes predicted to
      2.25 % (spheres 1–10) vs 10.5 % for mono-exponential decay, nothing fitted; reproduces
      the stimulated-echo signature. A global B1⁺ fit does not improve held-out spheres
- [x] Monte Carlo estimation (`validation/real/nist.py`): scrambled-Sobol spin placement
      (error ∝ N^-0.75, ~12× fewer spins than stratified), seeds averaged in the complex
      domain, the magnitude bias removed, jackknife error bars. Benchmark 1: 1.51 ± 0.05 %
      (1024 spins/pixel); benchmark 2: 2.20 ± 0.04 %
- [x] Fill fluid (water around the spheres) and sphere walls: halves the short-T2 spheres'
      error in benchmark 2, no effect in benchmark 1; wall thickness scanned (0.5–1.5 mm)
- [ ] Short-T2 spheres 11–14 (6.5–8.7 % in the first echoes): reference T2 values, wall and
      noise ruled out; suspects are local B1⁺ and played-vs-written timing (needs a B1⁺ map)
- [ ] Absolute (not just shape) agreement: coil sensitivities and B1⁺ fitted from the data
      (digital twin, R7), fill fluid, temperature
- [ ] More datasets: RRSG 2020 T1 challenge (multi-site NIST data), Pulseq-CEST phantom data,
      a multi-slice acquisition with its fat-saturation pulse (incidental MT, below)

## Phase 4 – Differentiable sequence design

- [ ] Parameterized sequence API: which tensors are free, bounds, reparameterizations
      (so far by hand, e.g. TR = TR_min + softplus(u) in `examples/design_contrast_tr.py`)
- [x] Hardware constraints as smooth penalties (`blochparty.design`): Gmax and slew per axis,
      peak B1 [µT], RF energy / windowed SAR proxy, approximate PNS (chronaxie-filtered dG/dt),
      scan time; violations named by segment range, time and excess. Verified: units and B1 vs
      closed forms; slew exact on a raster-sampled trapezoid, builders' instantaneous steps
      flagged; windowed RF power vs closed form; PNS reproduces the IEC hyperbolic
      strength–duration law (error < raster / ramp time, converging linearly); penalties zero
      inside the limits, gradcheck outside
- [ ] Timing raster checks; projections onto the constraints; Pulseq's own checks
- [x] Losses: tissue contrast, CNR, SNR efficiency (noise from the coil model by pseudo-replicas:
      σ/√N for the iFFT to 2 %), PSF FWHM / sidelobes (simulated point spin = Dirichlet kernel to
      1e-9, FWHM 1.207 px / sidelobe 21.7 %; T2 blurring = closed-form readout weighting to
      1e-9), Cramér–Rao bound from the simulated Jacobian (= closed-form DESPOT1 CRB to 1e-7, via
      fused autograd, fused finite differences and `simulate` autograd)
- [ ] Artifact energy loss
- [x] Examples with known optima, through the fused kernel (gradients = `simulate`'s to 1e-10):
      DESPOT1 flip pair by CRB (`examples/design_despot1.py`: 4.109° / 23.618° at TR 15 ms,
      T1 1 s, = closed-form grid to 2e-4°, signals at 70.711 % of Ernst = 1/√2 to 1e-6, Deoni
      2003: ≈ 71 %);
      WM/GM contrast with TR free under a scan-time budget (`design_contrast_tr.py`: lands on
      the budget's TR, 16.109° at 10 ms and 25.224° at 25 ms, = closed-form grid to 0.01°);
      low-SAR TSE refocusing train with every echo ≥ 0.6 × the 180° echo
      (`design_tse_flips.py`: 24 % of the RF energy of the cheapest constant train, echoes =
      EPG to 2e-16, train = SciPy SLSQP on EPG to 2e-6°)
- [ ] Examples: RF pulse design, MRF-style schedules

**Verify:** recover known optima (Ernst ✅, WM/GM contrast ✅ (also with TR free ✅), DESPOT1
CRB ✅, low-SAR VFA-TSE vs EPG + SLSQP ✅, SLR-designed pulses); constraint satisfaction checked
by Pulseq's own checks.

## Phase 5 – Agentic sequence development

- [ ] Tool-style API: JSON sequence spec in → structured report out (metrics, constraint
      violations by block, images, PSF), deterministic and fast. Report out exists
      (`design.design_report`, JSON, ~0.6 s for a 64² 8-coil GRE); spec in does not yet
- [ ] Sequence linter with actionable messages ("slew exceeded in block 14 by 12 %"): the report
      names segment ranges ("slew x exceeded in boundaries 2–20 (t = 1.010–1.190 ms) by 12 %");
      mapping segments back to Pulseq blocks is still to do
- [ ] Split of labour: agent chooses structure (sequence family, ordering, spoiling);
      gradient descent tunes continuous parameters inside it
- [ ] Benchmark task suite with scored targets (e.g. "T1w brain at 1 mm in < 2 min,
      maximize WM/GM CNR"), to measure agent progress

**Verify:** agent-produced sequences pass Pulseq checks and cross-simulator validation
(Phase 2) before any score counts.

## Phase 6 – Training-data generation

- [x] Anatomy M1: susceptibility maps → B0 with padded Fourier dipole convolution and
      order 0–2 shimming (`susceptibility_to_b0`, `shim`); sphere inside/outside RMS
      0.439/0.108 Hz at 3 T, long cylinders < 1.77 Hz, harmonic removal < 1e-12 Hz.
      QSM forward-field validation follows in M3 of [the anatomy spec](docs/specs/ANATOMY.md).
- [x] Anatomy M2: checksum-pinned BrainWeb loader, conservative resampling, cited/flagged tissue
      priors and sub-voxel partial-volume spins; separate susceptibility/scanner shim maps.
      PD conservation <1e-12, GPU/core error 3.23e-7; full 1-mm head RF/FID 7.77 M spins/s.
      [Validation and limits](validation/anatomy/RESULTS.md); spatial variability remains M4.
- [ ] Anatomy M3: QSM loader and public Sim1/Sim2 checks implemented (forward RMS
      2.11e-16 / 3.76e-16 Hz vs independent Python reference). Full-source R1/R2*/M0
      validation and original MATLAB check need authorized data; [details](validation/anatomy/QSM.md).
- [x] Anatomy M3 (alternative): `load_ahead` for the open AHEAD v3 7 T templates (measured
      R1/R2*/PD/QSM); units and geometry checked against Caan 2019 / MNI2009b.
- [x] Anatomy M4: within-tissue texture (`variability`), statistics tested.
- [x] Anatomy M5: `examples/anatomy_brain.py` (3 T BrainWeb: T1w GRE, T2w TSE, long-TE GRE) and
      `validation/anatomy_ahead.py` (7 T measured maps).
- [x] **T2* from measured R2* without double counting** (2026-10-07; `t2star="separate"`, the default with chi): a measured R2*
      contains the macroscopic dephasing from the voxel-scale susceptibility field that the
      simulation now computes itself, plus microscopic R2' (vessels, myelin, iron) that it
      does not. Today `t2star="auto"` drops R2' when chi is on, so measured R2* contrast vanishes
      (AHEAD at 7 T: no pallidal darkening until R2' was supplied by hand). Plan: estimate the
      macroscopic part from the simulated field's within-voxel gradients (sinc-type decay,
      Yablonskiy/Fernandez-Seara) at the acquisition's voxel size and TE range, subtract it to
      get microscopic R2' per voxel, and make that the default; check against a multi-echo GRE
      of the same anatomy (simulated R2* fit vs the input map) and the static-dephasing limit.
      Done: round trip within 0.3 % (vs +9 to +24 % double counted, -6 to -21 % dropped).
- [ ] 3 T WM/GM T2* in the tissue table: replace the occipital Wansapura proxies (GM < WM) with
      whole-brain values (Hoogenraad/Peters 2007 reports cortical GM longer than WM; verify numbers).
- [ ] Further realistic phantoms: XCAT / MRXCAT; random augmentations
      (generative plan: [docs/PHANTOMS.md](docs/PHANTOMS.md))
- [ ] Raw k-space from reconstructed scans ("lift", PHANTOMS §5), validated on paired raw/DICOM data
- [ ] Physically based coils: Biot–Savart loop arrays (low field), imported EM maps (≥3 T)
- [ ] Motion (rigid, then deformable), flow, diffusion
- [ ] Dataset writer (ISMRMRD / HDF5) with full provenance (sequence, phantom, seed)
- [ ] Fast path: signal-equation forward model for cases where Bloch fidelity isn't needed

**Verify:** noise statistics (SNR maps, g-factor) vs. theory; motion artifacts vs.
analytic ghosting for periodic motion; comparison with real fastMRI-style data statistics.

## Ideas parking lot

Phase distribution graph (PDG) backend for speed; magnetization transfer / multi-pool;
non-Cartesian trajectories + NUFFT recon (gradient non-linearity, eddy currents and
concomitant fields: done, Phase 2b).
