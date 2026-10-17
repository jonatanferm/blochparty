# Performance research: where the time goes and how the mathematics can remove it

Written 2026-10-10 from reading the engines, the benchmark history and the realism workload,
plus a few measurements on the Metal kernel (this laptop's 8-core M4 GPU; the Mac mini's 10-core
GPU runs the same cases about 1.5-2x faster, the ratios are what matter). Nothing here was
implemented then (see Status); each item names its independent check and an estimate of the gain, marked
**measured** where it was timed and **estimated** where it comes from a cost model.

The short version: the Metal kernel is already a good *isochromat stepper*. The remaining
factor of 10-100x is not in the stepping but in the structure of what is stepped: readouts are
Fourier transforms, pulses are low-rank in their parameters, repeated TRs are a periodic map,
ramps need no raster, and the adjoint need not store or recompute anything. Most of it is exact.

## Status (2026-10-10)

Implemented and measured on the realism TSE (one slice, 128², 16 coils, laptop GPU):

| item | where | measured |
|---|---|---|
| 2.4 merge RF- and ADC-free stretches | `Sequence.merge_free` | 36 335 -> 19 726 segments; 1.98 -> 1.33 s (400k spins); exact (2e-15 core, 2e-6 Metal) |
| 2.6 fewer spins | `to_phantom(sampling="systematic")` | systematic = categorical, lattice = random at equal count: no gain |
| 2.7 overlap phantom build and GPU | `simulate_heads.py` | fast mode 2.3 s per slice (legacy ~128 s) |
| 2.1 NUFFT readouts | `simulate_metal(readouts="nufft")` | 2.5x single slice, 1.6x with 16 slices' pulses and MT; 3e-6 from per-sample |
| 2.3, exact variant: reused pulse maps | `simulate_metal(pulses="reuse")` | on top of 2.1: 1.65x single slice, 2.3x with 16 slices and MT; 1e-7 from stepping |

Along the way the far slices' pulses became `Sequence.bound_only` segments (a frame offset broke
float32: LESSONS). After 2.1 the multi-slice MT slice was RF-bound (~4.5 of 6.3 s). Tabulating
pulses over (Δf, B1) needs the interaction-picture expansion per table node: its cheap form (a
pulse as a few blocks, exact rotation with relaxation and exchange around it) is 0.5-1.3 % off per
pulse for WM and muscle (LESSONS). Instead, per spin: a static spin's map of a pulse is the same at
every occurrence, and a phase-shifted copy is that map turned about z, so each distinct pulse is
stepped once per spin (12 or 20 floats) and applied at every occurrence. Exact, and general (TSE
trains, CPMG, RF spoiling, bSSFP alternation); not for pulses that change every time (MRF flip
trains: tables remain the route there). Realism TSE end to end, full quality, laptop: 16 slices +
MT 20.8 s → 5.6 s per slice (3.7x from 2.1 and the reuse together).
Not started: 2.2 periodicity, 2.5 reversible adjoint (neither speeds the realism TSE).

## 1. Where the time goes today

### 1.1 The segment mix

Counted with the builders (`bp.gre_cartesian`, `bp.gre_3d`, `builders.tse_2d` as
`validation/realism/simulate_heads.py` calls it). "Recompute" is a free segment whose
(dt, g, b0) differs from the previous one, so the kernel's factor cache misses and the thread
evaluates `precise::sin`, `precise::cos` and two `expm1` for it.

| sequence | segments | RF | ADC | free, cache miss, no ADC | free runs |
|---|---|---|---|---|---|
| GRE 3d-64 (benchmark) | 278,928 | 4,296 (1.5 %) | 262,144 (94 %) | 12,488 (4.5 %) | 4,296 |
| GRE 3D 128³ (`examples/gre3d_128.py`) | 2,163,088 | 16,584 (0.8 %) | 2,097,152 (97 %) | 49,352 (2.3 %) | 16,584 |
| realism TSE 128, RF raster 100 µs | 36,335 | 3,060 (8.4 %; 153 pulses of 20) | 16,384 (45 %) | 16,763 (46 %) | 154 |
| realism TSE 128, RF raster 10 µs (`legacy`) | 63,875 | 30,600 (48 %) | 16,384 (26 %) | 16,763 (26 %) | 154 |
| GRE 2D 128, ramps, RF-spoiled, slice-selective, 1 µs RF raster | 495,412 | 456,000 (92 %) | 16,384 (3 %) | 23,028 (5 %) | 229 |

Two things stand out. In 3D GRE essentially every segment is an ADC sample. In the realism TSE,
46 % of the segments are pieces of ramps and crushers cut on the 10 µs gradient raster, each a
cache miss, and they carry no ADC sample; the free runs between pulses average 77 distinct
gradient values.

### 1.2 The cost split, measured

A sub-voxel Shepp-Logan (130,704 spins, 16 per voxel, stratified jitter) through the realism TSE
sequence, and the 3d-48 benchmark phantom (29,831 spins) through its GRE. "ADC off" keeps every
segment and clears the ADC flags (evolution only). "Merged" replaces every stretch of
consecutive RF-free, ADC-free, unspoiled segments by one segment with the same duration and
gradient moment; for static spins without a field model this is exact (the phase at the
segment end is all that matters), and the signals differ by 4e-6 relative (float32 reordering).

| case | full | ADC off | merged | merged + ADC off |
|---|---|---|---|---|
| TSE 100 µs, 1 coil | 1.62 s (2.9e9 spin-seg/s) | 0.92 s | 0.99 s (S 36,335 → 19,726) | 0.48 s |
| TSE 100 µs, 14 coils | 1.82 s | 0.97 s | 1.20 s | 0.46 s |
| TSE 10 µs, 1 coil | 3.90 s | 2.75 s | 2.68 s (S 63,875 → 47,266) | 2.07 s |
| GRE 3d-48, 1 coil | 0.99 s (3.6e9 spin-seg/s) | 0.40 s | 0.95 s | – |
| GRE 3d-48, 8 coils | 1.24 s | 0.36 s | 1.22 s | – |

From the differences, per spin and segment on this GPU: a cached free segment costs about
1e-10 s, a cache-miss free segment 2e-10 s, an RF segment 4.4e-10 s, and an ADC sample
(buffering plus the coil sum) 4-8e-10 s on top of its evolution. So on the realism TSE at the
100 µs raster the budget is roughly: coil reduction 45 %, ramps and crushers 25 %, evolution
through the readouts 10 %, RF pulses 10 %, the rest 10 %. On the 3D GRE the coil reduction is
60-70 % and the evolution through the readouts most of the rest. In the multi-slice variant of the
realism run (every other slice's pulses played, 16x the RF segments) the RF pulses dominate, which
is the x3 the journal records.

Arithmetic intensity, for calibration: the 8-coil 3d-48 reduction is 29,831 × 110,592 × 8 coils
× 8 flop = 2.1e11 flop in about 0.9 s, 0.24 TFLOP/s against a peak of roughly 3.3 TFLOP/s for
this GPU (about 4 for the mini's). Each flush also reads 2 KB of sensitivities per simdgroup per 8
samples: 26 GB per 3d-48 run, which sits between L2 and the 120 GB/s DRAM. The kernel is neither
compute- nor bandwidth-saturated; it is latency- and instruction-bound, which is normal for this
kind of loop and is why restructuring the work pays more than tuning it.

`benchmarks/bench.py` calls spins × ADC samples "the irreducible cost of any Bloch simulator".
Section 2.1 argues it is not.

## 2. The ideas, in order of expected gain

### 2.1 Readouts are Fourier transforms: closed-form free runs + a nonuniform FFT per readout

**The structure.** Within a free run with constant gradient G, spin s at sample i contributes
`a_s · z_s^i` with `z_s = exp(-dt/T2_s) · exp(-i 2π (G·r_s + f_s) dt)` (f_s its off-resonance
including T2′ and fat shifts), `a_s` its coil-weighted transverse magnetization at the run's
start. The core's `_free_run` already uses this (it is the reason the core beats segment
stepping), but evaluates the sum over spins as a dense (spins × samples) product, and the Metal
kernel steps sample by sample and sums with simdgroup matrix products. Both are O(N · n) per
readout. The sum over spins, however, is a nonuniform discrete Fourier transform along the
readout axis evaluated at exactly the Cartesian k-space points, with the decay handled as a
slowly varying envelope:

- off-resonance is a shift along the readout axis, `x_eff = x_s + f_s / |G|`, exact for a line
  in k-space (this is the chemical-shift displacement, used in reverse);
- the T2 decay over a readout of duration T is `exp(-t/T2)` with `T/T2 ≤ 0.06` for T2 ≥ 20 ms
  and a 1.3 ms readout, so a degree-2 Chebyshev expansion in t reproduces it to about 1e-6; spins
  with `T/T2 > 0.3` (cortical bone, lung) keep the direct path or get more terms. This is the
  time-segmentation of field-corrected reconstruction (Noll et al. 1991; Sutton, Noll, Fessler
  2003; Fessler et al. 2005), applied in the forward direction;
- each term is then a type-1 NUFFT (Dutt & Rokhlin 1993; Greengard & Lee 2004; FINUFFT,
  Barnett et al. 2019): spread each spin's weight onto an oversampled 1D grid with a kernel of
  width w (w = 6-8 for 1e-5 to 1e-6), FFT, done. The FFT output *is* the Cartesian readout; no
  interpolation step.

**The cost model.** Per spin and readout of n samples with C coils: today about `n × (8C + 25)`
flop-equivalents (coil MMA plus evolution and buffering per sample); after, about
`w × M × (8C) + 100` with M = 3 decay terms, where the spreading is done with the *same*
simdgroup matrix products as today, (coils × 32 spins) · (32 spins × w·M grid cells), once per
readout instead of once per 8 samples. So the readout cost drops by about `n / (w M)`: 7x for
n = 128, 14x for the 2x-oversampled 256-sample readouts of scanner protocols and Pulseq files,
plus the evolution through the readout collapses from n segments to one closed-form step.

**The GPU design.** For static spins and a fixed readout axis (every Cartesian sequence; EPI
needs the two signs), sort the spins once by `x_eff`; a threadgroup's 256 consecutive spins then
cover a handful of grid cells, so the spreading stays in threadgroup memory and each threadgroup
issues a few hundred device atomics per readout instead of one per spin. Moving spins keep the
direct path (or re-sort per readout, a 1-2 ms radix sort of 1e6 keys). Radial spokes re-sort
per spoke direction. Spirals and other time-varying readouts generalise to a type-3 NUFFT with
time segmentation, the standard forward model of non-Cartesian reconstruction, with a smaller
gain (about 5x) because the spreading is 2D.

**The adjoint** is the type-2 NUFFT (a gather from the gradient grid, no atomics), and the
derivatives with respect to positions, off-resonance and the gradient amplitude come from the
kernel's derivative: the adjoint gets simpler, not harder.

**Independent check.** The current direct kernel on the same spins, to the NUFFT tolerance;
`reference.signal_equation`; and `tests/engines/test_free_runs.py` extended with the NUFFT path.

**Gain (estimated).** Realism TSE: the readout share (45 % + 10 %) shrinks about 7x → 1.6-1.8x
alone, 3x on top of the merge of 2.4. 3D GRE 128³: readouts are 97 % of the segments → about 6x
(192 s → ~30 s). Non-Cartesian benchmarks: 3-5x.

### 2.2 Periodic sequences: simulate one period, encode the rest by FFT

**The structure.** In a 3D RF-spoiled GRE every one of the 16,384 TRs is the same block up to
the phase-encode amplitude and the RF phase. Phase encoding is rewound inside the TR, so it is a
per-spin phase factor `exp(-i 2π (ky y + kz z))` on the readout and nothing else: a Fourier
factor. The RF phase is not periodic (`φ_j = 117° · j(j+1)/2`), but in the frame rotating with
the RF phase the pulse is always along x and the per-TR map becomes
`m' → R_z(-(j+1)·117°) (A m' + b)`, which repeats with period `360 / gcd(117, 360) = 40` TRs.
Free precession and relaxation commute with `R_z`, which is what makes this exact. For TSE
(CPMG phases fixed) the period is one shot; for bSSFP with 180° cycling it is two TRs. So after
the transient, the spin states of the whole scan are 40 (or 1, or 2) vectors per spin, and the
k-space of every line in residue class r is a **3D type-1 NUFFT of the state r** (kx from the
readout, ky and kz from the phase-encode factor, decay by time segmentation as in 2.1).

**The transient** needs no dummy TRs: the 40-TR map is affine per spin, so propagating four
basis vectors (three unit vectors and zero; five with a bound pool) through one period gives
A and b, and the periodic steady state is `(I − A)⁻¹ b`, a 3×3 solve per spin (Hargreaves et al.
2001 did this for bSSFP; Zur et al. 1991 is the RF-spoiling reference). That is exact, where the
200 dummies reach it to about `(E1 cos α)^200 ≈ 1e-4`. A scan that starts away from the steady
state (IR preparation, centric ordering of the first lines, a TSE whose CSF has not recovered in
one TR) is still available exactly: the state before repetition k is `m* + A^k (m_0 − m*)`,
so simulate the first few repetitions directly and the periodic tail by NUFFT.

**What it needs.** A sequence-analysis pass that finds the repetition structure (the builders
know it; for Pulseq files, hash the segment stream with the phase-encode amplitude masked), and
2.1. It does not apply to sequences whose repetitions genuinely differ: motion and flow across
TRs, B0 drift, GIRF tails that differ per line, MRF trains; those keep the per-TR path.

**Gain (estimated).** 3D GRE 128³: 16,384 readout TRs become 40 states, 40 × 3 × 8 3D NUFFTs of
565k spins (about 1 s on this GPU, the 256³ FFTs included) plus 160 TR-equivalents of evolution
without readouts (0.2 s): **192 s → 2-4 s**. 2D TSE 128 in steady state: 8 shots → 1, about 2x
on the evolution; the readouts stay per shot (a 2D NUFFT per echo costs w² per spin, no better
than eight 1D ones). bSSFP and non-spoiled GRE: as 3D GRE.

### 2.3 RF pulses as tabulated propagators in (off-resonance, B1)

**The structure.** During a slice-selective pulse a static spin sees a constant off-resonance
`Δf = G_z z + dB0 + b0_offset` and a constant B1 scale. Without relaxation the net rotation of
the pulse is a function of (Δf, b1) only, smooth in both: in Cayley-Klein form the two
components are polynomials of degree = number of samples in `exp(i 2π Δf dt)` (the Shinnar-Le
Roux representation, Pauly et al. 1991). With relaxation the propagator is a 3×4 affine map
`P(Δf, b1; R1, R2)`, and since `R T_pulse ≤ 0.03` for a 2 ms pulse it is linear in (R1, R2) to
1e-3 and quadratic to 1e-5: `P ≈ P0 + R1 P1 + R2 P2 + (second order)`, each `Pk(Δf, b1)` a table.
The tables come from the core itself (float64, autograd): one Bloch simulation of a few thousand
virtual spins on a (Δf, b1) grid per distinct pulse shape, with `P1, P2` by differentiating in
R1, R2. In the kernel a pulse then costs one table lookup with cubic interpolation in Δf and
linear in b1, about 100 flop, instead of 20 (100 µs raster) or 200 (10 µs) Rodrigues steps with
two precise sines each. The rate-model bound pool fits the same scheme: its saturation over the
pulse is `∫ W(Δf, t) dt` with W quadratic in b1, and the 2×2 exchange step over the pulse is
tabulated in Δf as well, with the water-bound coupling split at the pulse level (Strang, error
`(k T)² ≈ 1e-2` of a 1e-2 effect; sub-blocks of four bring it to 1e-5 if needed).

**Where it does not apply.** Pulses under a time-varying gradient (VERSE, GIRF-distorted slice
selection beyond 1 %), the generalized Bloch pool and free pools during RF, moving spins inside a
pulse: these keep the stepped path. Gradients of the pulse's samples flow through the table
build (a Bloch simulation of virtual spins), gradients of the spin's parameters through the
interpolation.

**Independent check.** The stepped kernel on the same spins at the 1 µs raster; the slice
profile against SLR; `tests/engines/test_integration.py`'s 32,000-segment flip-angle gradient.

**Gain (measured cost, estimated saving).** Realism TSE: RF is 10 % at 100 µs → 1.1x alone, but
after 2.1 and 2.4 it is the largest remaining term (about 40 %), so 1.6x there. Multi-slice
realism (16 slices' pulses, x3 today) → back to about 1.1x the single-slice cost: **3x**. The
`legacy` 10 µs raster and the slice-selective RF-spoiled GRE (92 % RF segments) → 5-10x. It also
makes the raster question moot: the table is built at 1 µs for the cost of 100 µs.

### 2.4 Merge raster-cut ramps and crushers (exact, free)

The builders cut every ramp on the 10 µs gradient raster so that `design.penalties` sees the true
slew, Pulseq export is exact, and ADC samples on ramps (EPI) land at the right k. The simulator
does not need that resolution: for a static spin without a field model the phase at the end of
an RF-free, ADC-free stretch depends only on its total duration and gradient moment. Merging
such stretches into one segment each is exact (4e-6 relative in float32, the reordering) and
turned the realism TSE's 36,335 segments into 19,726 and 1.82 s into 1.20 s (**1.5x measured**,
1.63x with one coil; 2.4x on the evolution-only part). With concomitant fields the merged
segment needs the stretch's `∫ g_i g_j dt` (six numbers, closed form for a linear ramp) instead
of `g_i g_j T`; with diffusion the walk's bridge formula also generalises to a linear gradient
(the effective position gets a `∫ g(t) t dt` term); with GIRF-played gradients the ramp is not
linear but the merge is still exact as long as the moments are integrated numerically on the
raster, which they are. Spoils and `b0_offset` changes end a stretch. This belongs as an opt-in
transform the engines apply to static spins, keeping the raster sequence as the scanner's and the
design checker's representation.

**Independent check.** The unmerged kernel (done above: 4e-6) and the core.

### 2.5 A reversible adjoint: no replay, no recompute

The Metal adjoint stores checkpoints every ~√S segments, replays each interval keeping every
4th state, and recomputes blocks of 4 into threadgroup memory before sweeping them backward:
forward + backward costs 5-7 forward passes. But every Bloch step is exactly invertible:
rotations are orthogonal (`R⁻¹ = Rᵀ`), free precession is a rotation, relaxation is
`m_z,prev = pd + (m_z − pd) e^{+dt/T1}`, the 2×2 exchange step has a closed-form inverse. The
reverse sweep can reconstruct `m_{s−1}` from `m_s` in registers instead of reading it, which
removes the replay kernel, the R buffer and the threadgroup recomputation. Un-relaxing amplifies
rounding by `e^{+t/T2}`, so inversion is stable over an interval of 64 segments × 10 µs ≪ T2
(1 % amplification) but not over a whole scan: keep the checkpoints every 64 segments (as the
per-spin gradient flushes already are), store Mxy at spoils (not invertible) and the bound-pool
state at saturating RF segments (where `W dt` can reach 1), invert everything else. This is the
reversible-network idea (Gomez et al. 2017; MALI, Zhuang et al. 2021) applied to a system whose
steps are exactly invertible rather than approximately. Results stay deterministic.

**Independent check.** The current adjoint (bit-identical for any `checkpoint_every`) and the
core's autograd, as in `tests/engines/test_engines.py`.

**Gain (estimated).** The replay and the in-block recompute are about two of the 5-7 forward
equivalents, and the R traffic was measured as a quarter of the sweep on 3d-48: **forward +
backward from 5-7x to about 3x the forward**, 1.7-2x on gradients. Combined with 2.1 the
adjoint's readout part is a gather, which helps again.

### 2.6 Fewer spins for the same accuracy

Spins are the multiplier on everything above, and the realism run chose 8 per voxel at RMS 0.014
against a 27-spin reference whose own seed-to-seed RMS is 0.0073; the numbers scale as 1/√K, so
this is Monte Carlo error. What the Monte Carlo is integrating, in a crushed TSE, is mostly the
residual of the crushers: a real voxel has 1e15 spins and a 2-cycle crusher leaves nothing, while
K jittered spins leave `1/√K` of the dephased magnetization as coherent noise. Three ways to
spend fewer spins:

- **Quadrature instead of Monte Carlo along the crusher axis.** With spins on a regular lattice
  of spacing h along z, the voxel sum of a configuration with winding c cycles per lattice
  period is the trapezoid rule on a smooth, nearly periodic integrand: exponentially accurate
  (Trefethen & Weideman 2014), and *exactly zero* for every winding that is not a multiple of
  the lattice's Nyquist winding. Jitter was introduced to turn that aliasing into noise; the
  cleaner answer is a lattice fine enough that the aliasing pathways are those whose amplitude
  is already negligible (with near-180° refocusing the amplitude of a pathway falls by
  `sin²(α/2)` per un-refocused lobe). For the realism TSE (crushers of 2 cycles per 5 mm,
  spoiler 4), a 0.5 mm z-lattice (the current 2 spins per 1 mm voxel, un-jittered) resolves
  windings up to 10 cycles across the slice, i.e. five lobes; in-plane the phase encodes are
  rewound and the readout is balanced at the echo, so one spin per voxel in-plane may do. If
  this holds up, 8 jittered spins per voxel → 2 lattice spins at lower error: **4x on spins**.
  The check is the 27-spin reference and the EPG model of the same train
  (`validation/RESULTS.md` already has the aliasing study).
- **Allocation.** Neyman allocation puts spins where the signal variance is: proportional to
  PD × the voxel's dephased fraction, not uniform. Bone, scalp and partial-volume air are
  over-sampled today. Estimated 1.2-1.5x.
- **Partial volume without draws.** `sampling="categorical"` draws each spin's tissue; a voxel
  with fractions (0.5, 0.5) and 8 spins gets a Binomial(8, 0.5) count, a 35 % relative error
  in that voxel's tissue mix. Systematic assignment (round(K f_t) spins per tissue, one draw for
  the residual, as in particle-filter resampling) has O(1/K²) variance. The measured accuracy
  matched `per_tissue` at equal K, so the gain is at boundaries only, but it is free.

The principled end of this road is the phase-distribution graph (Endres et al. 2024, MRzero),
which integrates sub-voxel dephasing, T2′ and free diffusion analytically per configuration
state and needs one state set per voxel instead of K spins; MRzero reports two to three orders
of magnitude over isochromats at equal accuracy for Cartesian sequences. ROADMAP lists it as not
planned, for good reasons (no motion, restricted diffusion, nonlinear field model or exchange in
that representation; a second engine to keep in parity). It remains the one approach that
removes the 1/√K noise outright rather than reducing its constant.

### 2.7 Kernel-level items on the current design

Each is a day or two and measurable immediately; together perhaps 1.5x on the forward, less once
2.1 has removed most of the readout work.

- **Half-precision coil reduction.** Apple GPUs run FP16 at twice the FP32 rate and
  `simdgroup_half8x8` exists. The magnetizations and sensitivities in half (2^-11 relative)
  with the per-tile sums in half and the cross-tile accumulation in float put a noise floor at
  about `5e-4 / √N` of the k-space centre, 5e-7 for 1e6 spins, three orders below the noise the
  realism score adds. Opt-in, since the engine tests want 5e-5 everywhere. 1.3-1.5x on the
  reduction, and half the sensitivity-tile traffic.
- **Sensitivity tiles.** Each flush re-reads 2 KB of sensitivities per simdgroup. Options: 16
  samples per flush with 4 simdgroups per threadgroup (same 32 KB), half-precision tiles, or the
  8-coil tiles resident in registers (16 per thread; the adjoint's experiment said occupancy loses,
  the forward may differ). 1.1-1.3x on the reduction.
- **GIRF-friendly free segments.** A GIRF or eddy-current model makes every readout sample's
  gradient unique and defeats the factor cache (6x on the retired CPU kernel). Two fixes: cache
  the relaxation factors on dt alone (they do not change), and update the precession factor by
  the angle-addition formula with a Taylor `exp(iδ)` for the small change δ between consecutive
  segments (`|δ|` is a few mrad for eddy tails; three terms are exact to 1e-10), with an exact
  refresh every 64 segments. 2-4x on GIRF-played sequences.
- **One `sincos` per rotation** instead of `sin(φ)` and `sin(φ/2)`; cache the RF half-step
  `expm1` factors on dt. About 10 % of the RF step.
- **Dark simdgroups.** Sort spins by distance from the slice; simdgroups whose 32 spins all have
  `|Mxy|` below a threshold at a readout's start skip its emission (a uniform branch). Small with
  the ±3 mm slab already in use.
- **Overlap phantom construction with the GPU.** `to_phantom` of the next slice while the
  current one simulates; up to 1.2x on the realism loop, more once the GPU side is 10x faster.

### 2.8 Free pools: the interaction picture

A CEST M0 offset at −1500 ppm costs ~80 Taylor sub-steps per 100 µs sample because the water's
precession `|ω| dt ≈ 120` sits inside A. The single-pool kernel handles the same situation exactly
with one Rodrigues rotation about the effective field. The free-pool kernel could do the same:
rotate each pool exactly about its own effective field (the fast part), and apply relaxation
and exchange as the slow part in the interaction picture, where the transverse exchange coupling
between pools at different frequencies oscillates and is averaged over the step (first-order
Magnus; Blanes et al. 2009). One or two sub-steps instead of 80 for far-off-resonance spins, which
also removes the need for the cost sort. Check: BMCTool (`validation/compare_bmctool.py`) and
the core's matrix exponential. 5-10x on CEST spectra (niche for the current goals).

### 2.9 Hardware, for scale

A CUDA port of the kernel (ROADMAP M8) on a rented RTX 4090 (82 TFLOP/s FP32, 1 TB/s) or H100
would give 10-20x over the mini's GPU with no algorithmic change, and the items above transfer.
Running the retired CPU kernel alongside the GPU is capped at about 1.5x (the M4's CPU reached
5-6e9 spin-segments/s against the GPU's 8e9) and was retired for maintenance reasons; not worth
reopening before 2.1-2.4.

## 3. Combined estimates per workload

Multiplicative where the items act on different parts of the budget, using the measured shares
of §1.2. "Today" is the mini's GPU where the journal has numbers, this laptop otherwise.

| workload | today | 2.4 merge | + 2.1 NUFFT readouts | + 2.3 tabulated pulses | + 2.2 periodicity | + 2.6 lattice spins |
|---|---|---|---|---|---|---|
| realism TSE, one slice, 14 coils, 1.8 M spins | ~15-20 s | 1.5x | 4-5x | 7-8x | 10-15x | 30-60x |
| realism TSE multi-slice + MT (x6 today) | ~90-120 s | 1.3x | 2x | 6-8x | 10x | 30-40x |
| 3D GRE 128³, 8 coils (`gre3d_128.py`) | 192 s | 1x | 6x | 6x | 50-100x | – |
| 3D GRE adjoint (forward + backward) | 5-7x forward | – | readout adjoint is a gather | – | – | 2.5 reversible: ~3x forward |
| GIRF / eddy-current sequences | 2-6x slower than ideal | 1x | 4x | – | – | 2.7: back to ~1.2x of ideal |
| CEST spectrum, 30k spins, 15k segments | 3.3 s | – | – | – | – | 2.8: 5-10x |

The realism ablations (20 heads × 13 slices, three variants) are hours today; with 2.1-2.4 they
are tens of minutes, and with 2.6 the quality knob moves from "8 spins, RMS 0.014" to
"2 lattice spins, exact crushers".

## 4. Suggested order

1. **2.4 merge** (a day, exact, 1.5x on the realism run; the snapshot check applies as is).
2. **2.1 NUFFT readouts** for static spins on Cartesian lines (one to two weeks with the
   adjoint; the direct kernel stays as the reference and the moving-spin path).
3. **2.3 tabulated pulses** (a week; the table build reuses the core).
4. **2.6 lattice study** (two days of validation against the 27-spin reference and EPG before
   any code).
5. **2.2 periodicity** (a few days once 2.1 exists; the 3D GRE case first, where it is 50-100x).
6. **2.5 reversible adjoint** when gradients become the bottleneck for design work.
7. **2.7** as fillers, each measured against the same-session baseline (LESSONS: benchmark only
   against something measured in the same session).

Every item keeps the defaults bit-identical by construction: 2.4 and 2.1 are opt-in transforms
and kernels, 2.3 a kernel path, 2.2 a sequence analysis that falls back to the per-TR path.

## 5. References

- Dutt & Rokhlin, "Fast Fourier transforms for nonequispaced data", SIAM J. Sci. Comput. 1993.
  Greengard & Lee, "Accelerating the nonuniform fast Fourier transform", SIAM Review 2004.
  Barnett, Magland, af Klinteberg, "A parallel nonuniform fast Fourier transform library based on
  an exponential of semicircle kernel", SIAM J. Sci. Comput. 2019 (FINUFFT); Shih et al.,
  "cuFINUFFT", IPDPSW 2021 (GPU spreading by binning and sorting).
- Noll, Meyer, Pauly, Nishimura, Macovski, "A homogeneity correction method for MRI with
  time-varying gradients", IEEE TMI 1991. Sutton, Noll, Fessler, "Fast, iterative image
  reconstruction for MRI in the presence of field inhomogeneities", IEEE TMI 2003. Fessler, Lee,
  Olafsson, Shi, Noll, "Toeplitz-based iterative image reconstruction for MRI with correction for
  magnetic field inhomogeneity", IEEE TSP 2005 (time and frequency segmentation, min-max bases).
- Zur, Wood, Neuringer, "Spoiling of transverse magnetization in steady-state sequences", MRM
  1991. Crawley, Wood, Henkelman, "Elimination of transverse coherences in FLASH MRI", MRM 1988.
  Hargreaves, Vasanawala, Pauly, Nishimura, "Characterization and reduction of the transient
  response in steady-state MR imaging", MRM 2001 (the affine per-TR map and its steady state).
- Pauly, Le Roux, Nishimura, Macovski, "Parameter relations for the Shinnar-Le Roux selective
  excitation pulse design algorithm", IEEE TMI 1991 (the pulse response as polynomials in the
  off-resonance phasor).
- Hennig, "Multiecho imaging sequences with low refocusing flip angles", JMR 1988; Weigel,
  "Extended phase graphs: dephasing, RF pulses, and echoes", JMRI 2015. Endres, Weinmüller, Dang,
  Zaiss, "Phase distribution graphs for fast, differentiable, and spatially encoded Bloch
  simulations of arbitrary MRI sequences", MRM 2024 (MRzero); see `docs/REALISM.md` for the
  MRzero sources used in this project.
- Gomez, Ren, Urtasun, Grosse, "The reversible residual network: backpropagation without storing
  activations", NeurIPS 2017. Zhuang et al., "MALI: a memory efficient and reverse accurate
  integrator for Neural ODEs", ICLR 2021. Griewank & Walther, "Algorithm 799: revolve", ACM TOMS
  2000 (optimal checkpointing, for comparison).
- Blanes, Casas, Oteo, Ros, "The Magnus expansion and some of its applications", Phys. Rep. 2009.
  Strang, "On the construction and comparison of difference schemes", SIAM J. Numer. Anal. 1968.
- Trefethen & Weideman, "The exponentially convergent trapezoidal rule", SIAM Review 2014.
  Mitchell, "Consequences of stratified sampling in graphics", SIGGRAPH 1996 (jittered sampling
  variance). Higham, "The accuracy of floating point summation", SIAM J. Sci. Comput. 1993
  (for the half-precision error floor).
- Castillo-Passi et al., "KomaMRI.jl: an open-source framework for general MRI simulations with
  GPU acceleration", MRM 2023 (closed-form per-block precession with a dense spin sum, the same
  O(N·n) structure as the core's `_free_run`).
