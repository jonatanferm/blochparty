# Bloch Party 🎉

A small, differentiable MRI simulator in PyTorch: phantom + coils + sequence → raw k-space.
Everyone's invited, as long as they precess.
Every feature is checked against an independent reference (the record: [docs/ROADMAP_ARCHIVE.md](docs/ROADMAP_ARCHIVE.md); the plan: [ROADMAP.md](ROADMAP.md)).
What other simulators do and the realism plan: [docs/REALISM.md](docs/REALISM.md); hard-won
lessons: [docs/LESSONS.md](docs/LESSONS.md); project journal: [docs/JOURNAL.md](docs/JOURNAL.md); comparison with real scanner data:
[validation/real/](validation/real/).

```bash
uv sync
uv run pytest                              # ~480 verification tests, ~3.5 min on an M4
uv run python examples/gre_demo.py         # 8-coil GRE of Shepp-Logan -> outputs/gre_demo.png
uv run python examples/optimize_contrast.py  # gradient descent through the simulator (--fused: C++ adjoint)
uv run python benchmarks/bench.py          # timed 2D/3D workloads (--record to log)
uv run python examples/gre3d_128.py        # full 3D 128^3 with the fused kernel, ~6 min
uv run python benchmarks/bench.py --backend fused  # same, with the fused C++ kernel
uv run python benchmarks/bench.py --grad --backend fused  # forward + backward
```

```python
import blochparty as bp

phantom = bp.shepp_logan(64)
coils = bp.birdcage_sensitivities(phantom.pos, n_coils=8)
seq = bp.gre_2d(64, flip_deg=15.0, tr=10e-3, n_dummy=200)

kspace = bp.simulate(phantom, seq, coils).signal          # (coils, n_adc) complex
image = bp.rss(bp.ifft2c(bp.kspace_grid(kspace, seq)))
```

3D works the same way: `bp.shepp_logan(32, ndim=3)`, `bp.gre_3d(32)`, `bp.reconstruct(...)`.

Anything numeric in a sequence or phantom can be a tensor with `requires_grad=True`
(flip angles, TR, gradient amplitudes, T1/T2/PD/B0/B1⁺/T2′/fat maps, coil maps).

## Realistic phantoms

`phantom_from_maps` turns 2D/3D maps into spins; the options below are off by default (one
spin per voxel centre, results unchanged bit for bit):

```python
from blochparty.coils import birdcage_b1
from blochparty.phantom import b0_shim, b0_sphere

maps = bp.shepp_logan_maps(64)
grid = bp.grid_positions(64, 0.22).reshape(-1, 3)
air = b0_sphere(grid, (0.0, 0.07, 0.0), 0.01, 9.4e-6)    # 1 cm air pocket in tissue, Hz
b0 = (b0_shim(grid, z2=300.0) + air).reshape(64, 64)
phantom = bp.phantom_from_maps(
    maps["pd"], maps["t1"], maps["t2"], fov=0.22, db0=b0,
    spins_per_voxel=4, jitter="stratified", seed=0,   # 2x2 isochromats per voxel, PD / 4 each
    t2prime=torch.full((64, 64), 0.05),               # Lorentzian intra-voxel offsets: T2*
    fat_fraction=(maps["label"] == 1).double(),       # 6-peak fat spins in the "fat" ellipse
    field_strength=3.0,
    b1=birdcage_b1(grid, centre_brightening=0.3).reshape(64, 64))  # transmit scale
```

- **Sub-voxel isochromats** (`spins_per_voxel=K` or per-side counts like `(1, 8)`; regular
  sub-grid or `jitter="stratified"`; in-plane for 2D maps). Spins per voxel side must exceed the cycles of
  dephasing across a voxel that still carry signal (spoilers, stimulated echoes), else
  those configurations alias back as spurious echoes (`validation/RESULTS.md`). Costs
  K× the spins; use `simulate_fused` or `simulate_chunked` for gradients.
- **T2′** (`t2prime` map): each voxel's spins get static offsets from a Lorentzian of
  half-width 1/(2πT2′) (Cauchy quantiles shuffled over the voxel, or `t2prime_sampling="random"`),
  so the FID decays with 1/T2* = 1/T2 + 1/T2′ and a spin echo refocuses it. K spins
  reproduce exp(−t/T2′) to ~0.5/√K of the initial signal over the first few T2′ (needs
  hundreds of spins per voxel for percent accuracy).
- **B0 maps**: `b0_shim` (offset, linear and second-order solid harmonics), `b0_sphere` and
  `b0_cylinder` (closed-form susceptibility fields, Lorentz corrected, B0 along z) return
  Hz at any positions; evaluate them at `phantom.pos` to sample the field at every spin.
- **B1⁺**: `Phantom.b1` (`None` = 1) scales every RF pulse per spin (flip b1·α) in `simulate`,
  `simulate_chunked`, `simulate_fused` (forward and adjoint: `b1` gets gradients) and
  `reference.signal_equation`. `coils.birdcage_b1(pos, centre_brightening, radius)` is a CP
  birdcage field, uniform or with J0(kρ) centre brightening.
- **Fat** (`fat_fraction` map): water keeps PD·(1 − f); one fat spin per peak of `FAT_PEAKS`
  (−3.80 … +0.60 ppm, amplitudes normalised to 1) with PD·f·a_p, fat T1/T2 (`fat_t1`,
  `fat_t2`) and offset ppm·γ̄·B0 (`field_strength`, T): fat sits below water (−434 Hz for
  the main peak at 3 T). Fat spins share their water spin's position, T2′ offset and B1⁺.
  `bp.shepp_logan(n, chemical_shift=True)` turns its fat ellipse into fat.

## Realistic sequences

The default builders are idealised (hard pulse, instantaneous gradients, ideal spoiling) and
stay bit-identical. Opt in to the real versions, built within a `System`'s hardware limits
(default: 40 mT/m, 170 T/m/s, 10 µs gradient / 1 µs RF / 0.1 µs ADC rasters, 100 µs RF dead
time, 30 µs ringdown; gradients in Hz/m):

```python
from blochparty.builders import se_2d, tse_2d, bssfp_2d, epi_2d, cartesian_recon

seq = bp.gre_2d(64, spoiling="rf", ramps=True, slice_thickness=5e-3)  # or spoiling="gradient"
seq = tse_2d(64, etl=8, refocus_flip_deg=[150, 120, 120, ...], ordering="centric")
seq = bssfp_2d(64, flip_deg=50.0)          # alpha/2 prep, 180 deg phase cycling, TE = TR/2
seq = epi_2d(64)                           # single shot, blips on the readout ramps
image = bp.rss(cartesian_recon(bp.simulate(phantom, seq, coils).signal, seq))
```

- `spoiling="gradient"`: a spoiler lobe brings the net readout-axis moment per TR to
  `spoiler_cycles` (default 4) cycles across a voxel, phase encodes are rewound; `"rf"` adds
  quadratic RF spoiling (117°, receiver phase follows). Spoiling acts on intra-voxel dephasing,
  so it needs several isochromats per voxel: with one spin at each voxel centre the spoiler
  is a whole number of turns and RF spoiling only modulates the signal line by line (ghosts).
- `slice_thickness`: windowed-sinc pulse (`time_bw_product`, `apodization`, default 2 ms) with
  slice-select gradient, `slice_offset` via a frequency offset, rephasing lobe including the
  ramps. `ramps=True`: trapezoids (minimum time, as pypulseq designs them).
- `se_2d`, `bssfp_2d` (and GRE) set `meta["kspace_shape"]`, so `bp.reconstruct` works. `tse_2d`
  (line ordering) and `epi_2d` (alternating readouts) set `meta["matrix"]` and
  `meta["kspace_index"]` instead; `cartesian_kspace` / `cartesian_recon` work for all of them.
- Building blocks for your own sequences (`blochparty.sequence`): `trapezoid(...)` (by area,
  flat area or amplitude; any axis or direction), `gradient_waveform(samples, raster)`,
  `block_pulse`, `sinc_pulse`, `ADC`, and `SequenceBuilder.block(*gradients, rf=..., adc=...)`,
  which plays them together like a Pulseq block. Ramps are cut on the gradient raster and each
  segment carries its exact average gradient, so k-space is exact at every ADC sample. Limits
  are checked (per axis) and violations raise.

## Pulseq

`read_pulseq` turns a `.seq` file (or a `pypulseq.Sequence`) into a `Sequence` (needs the
`pulseq` extra: `pip install 'blochparty[pulseq]'`; `uv sync` installs it for development):

```python
from blochparty.pulseq import read_pulseq

seq = read_pulseq("gre.seq", kspace_shape=(64, 64))   # extra keywords go to seq.meta
kspace = bp.simulate(phantom, seq, coils).signal
seq.meta["adc_per_readout"], seq.meta["kspace_trajectory"]  # samples per ADC; k from pypulseq
```

- Blocks are split at every event boundary, so each ADC sample ends a segment; gradients use
  the exact average over each segment, so k-space matches pypulseq's `calculate_kspace` to
  ~1e-10. Shaped RF, trapezoids, arbitrary and extended gradients, delays, frequency and phase
  offsets (including ppm), ADC phase modulation. Labels and triggers are ignored; other
  extensions (rotations, RF shims) raise.
- Phases: Pulseq phase φ gives Mxy = i·e^{−iφ}·sin α (a real pulse still tips +z to +y), so
  that a frequency offset f excites the slice at f/G as Pulseq defines it. MRzero reads the
  phases as written: `read_pulseq(..., convention="mrzero")` does the same (the mirror image).
- Pulseq cannot express ideal spoiling; set `seq.spoil[i] = True` after import if wanted.
- Cross-validation against MRzero: `uv sync --extra mrzero`, then
  `uv run python validation/compare_mrzero.py` (writes [validation/RESULTS.md](validation/RESULTS.md))
  or `uv run pytest -m crosscheck`.

## Scanner field model

Real gradients are not ideal. All of this is opt-in (`scanner=None`, no `b0_offset`: results
unchanged bit for bit), runs in `simulate` and `simulate_fused` (forward and adjoint), and is
differentiable in the hardware parameters, so a scanner's imperfections can be fitted:

```python
from blochparty.fields import GradientNonlinearity, Scanner
from blochparty.girf import GIRF, EddyCurrents, apply_girf
from blochparty.drift import linear_drift, respiration

scanner = Scanner(field_strength=0.55,                      # concomitant fields (leading order)
                  nonlinearity=GradientNonlinearity.from_file("coil.grad"))  # gradunwarp-style
played = apply_girf(seq, [GIRF.identity(delay=[2e-6, 2e-6, 3e-6]),            # per-axis delays
                          EddyCurrents.per_axis([[0.01], [0.01], [0.02]], [[0.5e-3]] * 3,
                                                b0_amplitude=[[1e-5]] * 3, b0_time_constant=[[1e-3]] * 3)])
played = respiration(linear_drift(played, rate=0.01), amplitude=3.0, period=4.0, gradient=(0.0, 20.0, 5.0))
kspace = simulate_fused(phantom, played, coils, scanner=scanner).signal
```

- **Concomitant (Maxwell) fields** (`Scanner(field_strength=B0)`): the transverse field
  components a cylindrical coil must have (div B = curl B = 0) shift the precession
  frequency by f_c = [(gx² + gy²) z² + gz² (x² + y²)/4 − gx gz xz − gy gz yz] / (2 f0)
  (Bernstein; `z_asymmetry` for asymmetric z coils). It is a per-spin quadratic form in g, so
  free runs stay closed form (cumulative ∫gᵢgⱼ dt). `concomitant="exact"` uses |B| of the full
  vector field instead (very low field, checks). Ramps on the raster are staircases: their
  ∫g² is low by (step)²·dt/12 per segment.
- **Gradient nonlinearity** (`GradientNonlinearity(alpha, beta, r0)` or `.from_file`):
  per-axis solid-harmonic deviation of each coil (gradunwarp's Siemens convention, `.grad`
  `A( n, m)` lines or `alpha_x n m` lines); spins are encoded at f(r) = r + d(r). Gradients reach
  the coefficients.
- **GIRF and eddy currents** (`apply_girf(seq, model)`, a sequence transform): `GIRF` impulse
  responses (taps on the gradient raster, or `from_transfer_function`, with delays and cross
  terms, row 4 = B0), `EddyCurrents` multi-exponential terms (amplitudes and time constants
  differentiable, B0 terms included). Each segment gets the exact average of the response to
  the nominal staircase, so k-space is exact at every segment end.
- **Uniform field per segment**: `Sequence.b0_offset` (S,) [Hz] (None = zeros) adds to every
  spin's off-resonance, also during RF; B0 eddy currents and drift land there.
- **Drift and physiology** (`blochparty.drift`): `linear_drift`, `exponential_drift`,
  `respiration` (uniform + linear terms with a period) add exact segment averages to
  `b0_offset` and (for linear fields) to the gradients.

## Memory-bounded gradients

Backprop through `simulate` keeps every spin's state at every segment, so memory grows as
spins × segments. When that no longer fits, swap in `simulate_chunked`: same arguments
and result, but the backward pass recomputes the simulation one spin chunk at a time, so
the graph it holds covers only `spin_chunk` spins.

```python
from blochparty.chunked import simulate_chunked

kspace = simulate_chunked(phantom, seq, coils, spin_chunk=4096).signal
loss(kspace).backward()            # same gradients as with simulate (to ~1e-12 in float64)
```

- Use it for gradients on large phantoms (3D, sub-voxel spins); plain forward runs gain
  nothing from it. It costs about one extra forward pass (~1.5× `simulate` forward+backward)
  plus a fixed per-segment Python cost per chunk, so keep chunks in the thousands of spins.
- `checkpoint_every` bounds the other direction (segments) and combines with it.
- Autograd also keeps a few tens of kB of bookkeeping per segment regardless of spin count,
  so for long sequences that floor, not the chunk, sets the minimum memory.
- First-order gradients only (no double backward).

## Fast gradients: the fused kernel

`blochparty.fused.simulate_fused` takes the same arguments and returns the same result as
`simulate`, ~10-15× faster on CPU, and is differentiable through a hand-written adjoint
(reverse sweep in C++) instead of autograd:

```python
from blochparty.fused import simulate_fused

kspace = simulate_fused(phantom, seq, coils).signal   # JIT-builds the extension on first use
loss(kspace).backward()            # same gradients as with simulate (to ~1e-14 in float64)
```

- Gradients reach everything `simulate` differentiates (sequence, phantom, coils, `m_init`).
- The forward stores every spin's magnetization each `checkpoint_every` segments (default
  ≈ √segments); the backward replays one such interval at a time. Memory is those
  checkpoints, N × 3 × segments / `checkpoint_every` values, with no autograd graph.
  The backward pass costs ~4 forward passes (64³ GRE, 8 coils: 46 s for forward +
  backward on 4 CPU cores, 1.2 GB peak, where autograd through `simulate` would need ~300 GB).
- CPU, float32 / float64, first-order only; needs a C++ compiler and `ninja`.

## Sequence design

`blochparty.design` adds what an optimizer (or an agent) needs around the simulator. Everything
is a differentiable function of the sequence's tensors or of simulated images:

```python
from blochparty import design

hw = design.Hardware(max_grad=40, max_slew=150, max_b1=20)   # mT/m, T/m/s, µT
loss = -design.cnr(image, wm, gm, sigma) + sum(design.penalties(seq, hw).values())
var = design.crb(signal_fn, theta)   # Cramér–Rao bounds from the simulated Jacobian
report = design.design_report(seq, phantom, masks=masks, contrast_pairs=[("wm", "gm")], kspace_sigma=0.05)
report["violations"]  # ["slew x exceeded at 128 boundaries (128 ranges); worst at boundary 402 (t = 2002.685 ms) by 699 % ...", ...]
```

- Constraints, in scanner units: gradient amplitude [mT/m] and slew [T/m/s] per axis, peak
  B1 [µT] (Hz → T via γ̄), RF energy Σ|B1|²·dt and a SAR proxy (mean |B1|², optionally scaled
  and windowed), an approximate PNS estimate (chronaxie-filtered dG/dt; reproduces the IEC
  hyperbolic strength–duration law, but is not a coil-specific model), scan time. Each has a
  smooth penalty (`penalties`) and violation reports naming segment ranges, times and excess.
- A gradient step is taken to ramp over at most one gradient raster time (10 µs): ramps sampled
  on the raster (as `read_pulseq` does) give their true slew, while the builders' instantaneous
  gradients show up as violations instead of being hidden by long segments.
- Metrics: tissue contrast, CNR, SNR and SNR efficiency (noise σ given, or propagated from the
  coil noise model through any reconstruction by pseudo-replicas), PSF and its FWHM / sidelobes
  from a simulated point spin, and the Cramér–Rao bound with the Jacobian w.r.t. tissue
  parameters taken by autograd (`method="autograd"`, differentiable again through `simulate`) or
  central differences (`method="fd"`, differentiable through `simulate_fused`).
- `minimize`: augmented Lagrangian around L-BFGS, so active constraints hold exactly at the end.
- `design_report`: JSON-serialisable timing, constraints with margins, PSF and image metrics.

Optimizations with independently known answers (each is also a test; all run through the
fused kernel, whose gradients match `simulate`'s):

```bash
uv run python examples/design_despot1.py     # VFA T1 mapping: CRB-optimal flips give 70.71 % of Ernst (Deoni 2003)
uv run python examples/design_contrast_tr.py # WM/GM contrast, flip and TR free under a scan-time budget
uv run python examples/design_tse_flips.py   # low-SAR TSE refocusing train, echoes >= target (vs EPG + SLSQP)
uv run python examples/design_report.py      # the report for an 8-coil GRE -> outputs/design_report.json
```

## Layout

| module | contents |
|---|---|
| `sequence.py` | `Sequence` (flat piecewise-constant segments), `SequenceBuilder`, `System` limits, trapezoids, sinc pulses, Pulseq-like blocks |
| `bloch.py` | the simulator core: `simulate()` |
| `chunked.py` | `simulate_chunked()`: same results, spin-chunked memory-bounded backward |
| `fused/` | C++ kernel with a hand-written adjoint, same results and gradients ~10-15× faster than `simulate` on an M4 CPU: `simulate_fused()` (JIT-built on first use) |
| `metal/` | forward kernel on the Apple GPU (`simulate_metal`, float32, runtime-compiled Metal shader); 5.8× the CPU kernel on the real NIST TSE workload |
| `phantom.py` | `Phantom`, tissue table, Shepp-Logan, maps → spins (sub-voxel, T2′, fat, B1⁺), B0 field maps |
| `coils.py` | receive sensitivities, B1⁺ transmit field, thermal noise |
| `builders.py` | ready-made sequences (`gre_2d`, `gre_3d`, `gre_cartesian`, `se_2d`, `tse_2d`, `bssfp_2d`, `epi_2d`) |
| `recon.py` | k-space gridding, centred N-D iFFT, RSS, coil combination |
| `reference.py` | independent models for validation: closed forms, signal equation |
| `epg.py` | independent extended-phase-graph model |
| `pulseq.py` | `read_pulseq()`: `.seq` / pypulseq → `Sequence` |
| `fields.py` | `Scanner` field model: concomitant fields, gradient nonlinearity (solid harmonics, coefficient files) |
| `girf.py` | `apply_girf()`: GIRF / transfer functions, delays, multi-exponential eddy currents (incl. B0 terms) |
| `drift.py` | B0 drift and respiratory fields as sequence transforms |
| `design.py` | hardware / safety constraints and penalties, image metrics, Cramér–Rao bound, `minimize`, `design_report()` |

## Conventions

- **Units (Pulseq-style):** RF in Hz (γ̄·B1), gradients in Hz/m (γ̄·G), off-resonance in Hz,
  time in s, position in m. Gradient area = k-space position in cycles/m.
- **Rotation:** dM/dt = γ M × B with γ > 0. A pulse along +x tips +z to +y; RF phase φ
  puts the magnetization at Mxy = i·e^{iφ}·sin α.
- **Signal:** Mxy = Mx + iMy evolves as e^{−i2πft} at offset f; s(k) = Σ ρ(r)·C(r)·e^{−i2πk·r};
  ADC phase ψ demodulates by e^{−iψ}.
- **Grid:** index i ↔ position (i − n/2)·FOV/n and k = (i − n/2)/FOV; arrays are ordered
  (z, y, x), so the last axis is x. 3D GRE lines run with ky fastest, then kz.
- **ADC timing:** sample i of a readout sits at (i + ½)·dwell after the ADC start (Pulseq).
- **Pulseq:** RF field = conj(Pulseq RF waveform), receiver phase = −(Pulseq ADC phase); see
  `blochparty/pulseq.py` for why and for the MRzero mapping.
