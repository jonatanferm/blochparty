# Generalized Bloch model vs. the rate model of magnetization transfer

    uv run python validation/mt/generalized_bloch.py
    uv run pytest tests/physics/test_generalized_bloch.py

`blochparty.generalized_bloch` implements the generalized Bloch model of Assländer et al.
([arXiv 2107.11000](https://arxiv.org/abs/2107.11000)): the bound pool's transverse
magnetization decays with the Green's function of its lineshape, and the pool is rotated by RF,
not only saturated. Solver: a uniform grid, Heun with a cell-trapezoid history, second order,
O(n²) per pulse; a reference for pulse-level dynamics.

## The paper's three limits, checked independently

- **Lorentzian = Bloch equations** (SciPy ODE, off-resonant π pulse): second-order convergence
  (error ratios 4.2, 4.1, 4.05); 2.3e-8 at h = 50 ns for a 50 µs pulse.
- **Continuous waves → Henkelman's steady state** R1/(R1 + πω1²g(Δ)) for the Lorentzian,
  Gaussian and super-Lorentzian lineshapes to ≤ 2e-4, with `exchange.py`'s lineshapes: this
  also confirms their G = πg normalisation independently of BMCTool.
- **Hard pulses rotate**: z_s → cos α (π/2, π) for all three lineshapes.
- The coupling to the free pool (z exchange) matches the two-pool Bloch–McConnell ODE for a
  Lorentzian bound pool to 1e-6.

## Where the rate model fails

Isolated bound pool, super-Lorentzian, T2s = 12 µs, on-resonance rectangular pulses
(as Fig. 3 of the paper); the rate model is `exchange.py`'s (W = ω1²G, the
super-Lorentzian held at its 1 kHz value on resonance):

| T_RF | α = π/2: generalized | rate model | α = π: generalized | rate model |
|---|---|---|---|---|
| 10 µs | +0.012 | +0.000 | -0.973 | +0.000 |
| 30 µs | +0.086 | +0.022 | -0.813 | +0.000 |
| 100 µs | +0.360 | +0.319 | -0.364 | +0.010 |
| 300 µs | +0.638 | +0.683 | +0.030 | +0.218 |
| 1000 µs | +0.836 | +0.892 | +0.458 | +0.633 |
| 3000 µs | +0.928 | +0.963 | +0.737 | +0.859 |

- **Short pulses (≤ 100 µs) invert** the bound pool (−0.97 after a 10 µs π pulse); the rate
  model can only saturate it, to 0. That is the effect Manning et al. measured.
- **Even millisecond on-resonance pulses differ**: the rate model with the 1 kHz cut
  saturates too little (α = π, 1 ms: +0.63 vs +0.46; 3 ms: +0.86 vs +0.74). The cut that
  avoids the super-Lorentzian's singularity underestimates the on-resonance saturation the
  full lineshape gives. So incidental MT from on-resonance imaging pulses (excitation,
  refocusing) is underestimated by the rate model. Off-resonant saturation pulses (MT and
  CEST preparation, fat saturation) are its regime and agree with Graham.

## Next

Bring it into the imaging engines with the paper's linearized form: a Bloch–McConnell
system with (x_s, z_s) and an effective R2,s,l(T_RF/T2s, α), tabulated from this solver,
for on-resonance pulses. It is exact for hard pulses and continuous waves, and the paper
gives ~2 µs per rectangular pulse.
