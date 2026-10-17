# 2026-10-13 — M5: dermis and eyes

- **Tissue values** (3f4cf1b): dermis T2 12 ms (>= its T2* ~10 ms at 1.5 T, Song 1997; assumed at both fields), PD 0.6
  (Richard 1993), muscle's T1 and MT (assumed); vitreous (class eye) T1 4.518 s at 3 T (Simpson 2024), 3.257 s at
  1.5 T and T2 546 ms (Ettl 1994; the 1.5 T T2 used at 3 T, flagged). tissuedb gains the measured rows, a
  stand-alone dermis tissue (as a child of skin it would have changed skin's pooled values) and eye.vitreous as class
  eye. The generator's classes draw from their own stream (other classes' draws unchanged; tested at 1.5 / 3 T); core
  snapshot bit-identical.
- **anatomy.add_dermis / sample_head(dermis=True)**: the outer 1.61 mm (Li 2024) of the head as dermis, by overlap,
  totals conserved. Quick test q32 (v14 heads + dermis) against q29: W1 0.0236 -> 0.0173, quartiles 0.211 / 0.293 /
  0.497 (q29 0.214 / 0.312 / 0.513, real 0.199 / 0.280 / 0.460), side / front scalp over WM 3.66 / 3.50 (3.43 / 3.41,
  real 4.00 / 3.88), but the posterior edge sharper (signal 1 px outside 0.09, q29 0.14, real 0.22); CNN 0.919 (q29
  0.809, +-0.15 at this size). Kept opt-in (`simulate_heads --dermis`), not in the preset.
- **The eyes** (b8b9106): prepare_ixi's orbit_vitreous finds the vitreous (bright T2, dark T1) in the orbits, the
  largest piece per side; IXI012: two globes ~24 x 21 x 27 mm at MNI (-35, 56, -39) / (33, 55, -39), 9.0 mL at its own
  size (before: 80 % skin, 20 % fat). The first version kept 84 voxels instead of ~12,700 (bincount over labels is
  indexed by label; a "+ 1" picked the neighbouring component); the synthetic test also caught a T1 sigmoid that
  capped pure vitreous at 0.87. 24 heads being prepared as v15. No effect on the realism score (fastMRI's stack
  starts above the orbits).
- Not done: dura and connective tissue relaxation (no in-vivo measurements found; muscle's stay, flagged).
