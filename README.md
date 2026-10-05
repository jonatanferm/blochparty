# Bloch Party 🎉

A small, differentiable MRI simulator in PyTorch: phantom + coils + sequence → raw k-space.
Everyone's invited, as long as they precess.
Every feature is checked against an independent reference (see [ROADMAP.md](ROADMAP.md)).

```bash
uv sync
uv run pytest                              # 21 verification tests, ~15 s
uv run python examples/gre_demo.py         # 8-coil GRE of Shepp-Logan -> outputs/gre_demo.png
uv run python examples/optimize_contrast.py  # gradient descent through the simulator
```

```python
import blochparty as bp

phantom = bp.shepp_logan(64)
coils = bp.birdcage_sensitivities(phantom.pos, n_coils=8)
seq = bp.gre_2d(64, flip_deg=15.0, tr=10e-3, n_dummy=200)

kspace = bp.simulate(phantom, seq, coils).signal          # (coils, n_adc) complex
image = bp.rss(bp.ifft2c(bp.kspace_grid(kspace, seq)))
```

Anything numeric in a sequence or phantom can be a tensor with `requires_grad=True`
(flip angles, TR, gradient amplitudes, T1/T2/PD maps, coil maps).

## Layout

| module | contents |
|---|---|
| `sequence.py` | `Sequence` (flat piecewise-constant segments) and `SequenceBuilder` |
| `bloch.py` | the simulator core: `simulate()` |
| `phantom.py` | `Phantom`, tissue table, Shepp-Logan, maps → spins |
| `coils.py` | receive sensitivities, thermal noise |
| `builders.py` | ready-made sequences (`gre_2d`) |
| `recon.py` | k-space gridding, centred iFFT, RSS, coil combination |
| `reference.py` | independent models for validation: closed forms, signal equation |

## Conventions

- **Units (Pulseq-style):** RF in Hz (γ̄·B1), gradients in Hz/m (γ̄·G), off-resonance in Hz,
  time in s, position in m. Gradient area = k-space position in cycles/m.
- **Rotation:** dM/dt = γ M × B with γ > 0. A pulse along +x tips +z to +y; RF phase φ
  puts the magnetization at Mxy = i·e^{iφ}·sin α.
- **Signal:** Mxy = Mx + iMy evolves as e^{−i2πft} at offset f; s(k) = Σ ρ(r)·C(r)·e^{−i2πk·r};
  ADC phase ψ demodulates by e^{−iψ}.
- **Grid:** index i ↔ position (i − n/2)·FOV/n and k = (i − n/2)/FOV; rows are y, columns x.
- **ADC timing:** sample i of a readout sits at (i + ½)·dwell after the ADC start (Pulseq).
