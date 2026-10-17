# 2026-10-07 (night, autonomous) — incidental MT, free-pool exchange, generalized Bloch

- **Incidental MT** (`validation/incidental_mt.py`, `RESULTS_INCIDENTAL_MT.md`) against Shin et
  al. 2009 (3 T LL-EPI with fat saturation, 1 → 9 slices: ~30 % loss). Nothing tuned. The
  model gives a linear growth with slice count and 97–99 % at 1 slice, with MT 2/3–3/4 of
  the loss for long pulses. The magnitude scales with the unknown fat-sat RF power: 6–19 % for
  10 ms Gaussians at 90–180°; ~30 % needs ~2× the power of 10 ms at 180°. Short pulses are
  ruled out by the measured single-slice value. A plausibility check, not a validation:
  the paper does not give the pulse.
- **Free exchanging pools** (`ExchangePool`, `Phantom.pools`): CEST pools and myelin water, with
  full Bloch dynamics, exchanging all three components with the water, plus the bound pool.
  Exchange couples the pools' transverse magnetization, so the rotation and closed-form
  structure no longer applies. A separate exact engine in the core does one matrix
  exponential of the augmented Bloch–McConnell system per segment, as BMCTool; the signal sums
  all free pools. Checks: SciPy ODE on the system written out by hand (water + amide +
  myelin pools ± bound pool, random RF and gradients, < 1e-8); closed-form two-site exchange
  of a free decay (McConnell; 1e-5, the 1 µs pulse); an inert pool reproduces the split
  core; **BMCTool's own 5-pool CEST model: 8.3e-7 in Z over 50 offsets**, same MTR_asym.
  Cost 0.3 ms per segment with 4 pools, which is fine for Z-spectra but slow for imaging.
  Fused/Metal/chunked refuse pools for now (next: a kernel version).
- **Generalized Bloch model** (`generalized_bloch.py`, `validation/RESULTS_GENERALIZED_BLOCH.md`):
  the bound pool rotated by RF via the Green's function of its lineshape (Assländer et al.).
  A second-order integro-differential solver (first order at first, until the RF was
  integrated cell by cell). It reproduces the paper's three limits independently: Bloch for a
  Lorentzian (SciPy, ratios 4.2/4.1/4.05), Henkelman's steady state for all three lineshapes
  (≤ 2e-4, which also checks exchange.py's G = πg without BMCTool), and z_s → cos α for hard
  pulses; the free-pool coupling matches the two-pool ODE (1e-6). Finding: our rate model
  can't invert the pool for pulses ≤ 100 µs (−0.97 vs 0 at 10 µs, α = π), and even for 1–3 ms
  on-resonance pulses it saturates too little because of the 1 kHz cut. Next: the paper's
  linearized R2,s,l in the imaging engines.
- **State at the end of the night**: tests 334 pass, defaults bit-identical. Open:
  exchange pools in the fused and Metal kernels; generalized Bloch in the engines
  (linearized); incidental MT needs a dataset with its fat-sat pulse; the short-T2 NIST
  question (B1⁺ / timing).
