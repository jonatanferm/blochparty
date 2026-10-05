# Bloch Party 🎉

A small, differentiable MRI simulator in PyTorch: phantom + coils + sequence → raw k-space.
Everyone's invited, as long as they precess.
Every feature is checked against an independent reference (see [ROADMAP.md](ROADMAP.md)).

```bash
uv sync
uv run pytest                              # 110 verification tests, ~1.5 min
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
(flip angles, TR, gradient amplitudes, T1/T2/PD maps, coil maps).

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

## Layout

| module | contents |
|---|---|
| `sequence.py` | `Sequence` (flat piecewise-constant segments) and `SequenceBuilder` |
| `bloch.py` | the simulator core: `simulate()` |
| `chunked.py` | `simulate_chunked()`: same results, spin-chunked memory-bounded backward |
| `fused/` | C++ kernel with a hand-written adjoint, same results and gradients ~10-15× faster than `simulate` on an M4 CPU: `simulate_fused()` (JIT-built on first use) |
| `phantom.py` | `Phantom`, tissue table, Shepp-Logan, maps → spins |
| `coils.py` | receive sensitivities, thermal noise |
| `builders.py` | ready-made sequences (`gre_2d`, `gre_3d`, `gre_cartesian`) |
| `recon.py` | k-space gridding, centred N-D iFFT, RSS, coil combination |
| `reference.py` | independent models for validation: closed forms, signal equation |
| `epg.py` | independent extended-phase-graph model |
| `pulseq.py` | `read_pulseq()`: `.seq` / pypulseq → `Sequence` |

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
