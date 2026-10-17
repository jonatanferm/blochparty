# 2026-10-07 — kernel ports

- **Metal scanner field model** (fbaa2e3): nonlinearity (encoding positions) and concomitant
  fields (leading, exact) in the off-resonance, as kernel.h; vs the core < 1e-4 at 0.55 T.
- **Generalized Bloch in the engines.** The paper's linearized R2,s,l is tabulated per
  rectangular pulse and drops the pool's off-resonance, so it was not used. Instead G(s) is a
  sum of complex exponential modes: the memory integral becomes linear modes u_j, and the RF
  rotates (Mzb, S = Σ u_j) like a Bloch pool (Σ c_j = G(0) = 1) with u_j += c_j ΔS, exact
  and O(J). Fit attempts: positive real exponentials can't make the flat top (Lorentzian
  wings, 12–18× the super-Lorentzian at 30 kHz); complex poles fitted to a finite range
  needed weights of ~5000 that diverge beyond it; ridge destroys the cancellations. What
  works is to define the model with a memory taper e^{−s/(100 T2)} (a ~133 Hz homogeneous
  broadening at 12 µs, a physical regularisation of the singularity) and fit its whole
  support: 30 modes, error 2.4e-5, weights < 1, lineshape within 0.5 % from 0 to 30 kHz.
  Core (`BoundPool(model="generalized")`): matches the reference solver to 1e-5 (including
  −0.969 after a 10 µs π pulse), second-order convergence, and the rate model for
  off-resonant saturation (its regime) to 2e-3; gradcheck. The split needs RF steps ≲ T2b,
  so RF segments are sub-stepped to T2b/2 (100 µs steps were 0.06 off).
- **Fused kernel**: generalized forward (724d4de) and adjoint (b05a4b0). Every input's gradient
  matches autograd to ~1e-14 on the first run.
- **Metal**: the generalized step in both coil kernels. Bugs: `half` is a type in MSL; and
  the generalized arrays (and earlier the MT/field-model state) slowed the plain kernels 10–30 %
  through register pressure. Each kernel is now compiled in three variants (plain / rate
  MT + field model / generalized), and the plain kernel is back at 0.40 s.
- **Free pools in the fused kernel** (ac01fc6, `fused/bm.h`): the exact engine per spin
  (Padé-13 matrix exponential, reused across identical segments). Adjoint by the block-matrix
  Fréchet trick, chained through A's entries. Forward 1e-14; every gradient ~1e-13 vs autograd
  through the core, after one bug (RF gradients credited to segments without RF).
- Not ported: free pools on Metal (a dense ~17×17 exponential per thread and segment, which
  overflows registers; pools are used for spectra and few spins, where the CPU kernel suffices),
  and the Metal adjoint (the whole reverse sweep for every physics variant: the largest port).
