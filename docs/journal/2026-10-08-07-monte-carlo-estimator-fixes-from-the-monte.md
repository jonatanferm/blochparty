# 2026-10-08 — Monte Carlo estimator fixes (from the Monte Carlo review)

A review of the Monte Carlo methods (docs/MONTE_CARLO_REVIEW.md, written against 29bae6e)
flagged the NIST estimators. The parts that change reported numbers are now implemented:

- `nist.mc_power` (unbiased, unclipped |true|^2); `mc_magnitude` is documented as a
  power-debiased approximation (the clip and square root bias it again near nulls). `mc_errors`
  jackknifes the spheres-1–10 mean itself (the old sqrt(Σ se²)/10 dropped covariance), adds the
  jackknife bias-corrected mean and the clipped-pixel fraction per ROI; `mc_paired_difference`
  and a table of paired differences between MC rows sharing seeds; `--save` keeps the seed
  images. Guards: K >= 2 (jackknife of mc_magnitude K >= 3), power-of-two Sobol counts,
  0 < sparse <= 1; the sparse water group generates only its prefix (bit-identical).
- Core bug: an ADC at the end of an RF segment read `spins`, not `step_spins`, so moving spins
  under mapped coils gave zero signal there (the coil placeholders). Test against a moved static
  object; Metal already agreed with the fixed core (9e-7). Snapshot bit-identical.
- Benchmark 1 rechecked with 8 seeds at 256 spins/pixel (RESULTS.md): the historical ±0.19 is
  0.31 with covariance; 4-seed bars differ 0.31 vs 0.19 between disjoint seed sets; clipping is
  ≤ 3 % of ROI pixels. Correction to the earlier reading: the 256 → 1024 shift (1.77 → 1.51 %)
  is mainly MC noise inflating the RMS shape error (1.90 / 1.80 / 1.73 % for 2 / 4 / 8 seeds;
  bias-corrected 1.66 %), not the near-null magnitude bias.
