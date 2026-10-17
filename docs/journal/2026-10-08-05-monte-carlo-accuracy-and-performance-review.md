# 2026-10-08 — Monte Carlo accuracy and performance review

- Audited spatial stratification/Sobol, tissue and T2′ sampling, NIST estimators, Philox,
  free/restricted diffusion, and Metal execution at baseline `29bae6e`. Findings and staged
  implementation plan: [MONTE_CARLO_REVIEW.md](../MONTE_CARLO_REVIEW.md). No simulator or default
  changes; historical scanner benchmark values were not recomputed.
- Confirmed that `mc_magnitude` debiases power, not the subsequent clipped square-root
  magnitude. In 200,000 four-seed trials (mean-noise component SD 1), true amplitudes 0/2/10
  gave corrected means 0.5240/1.6960/9.9485. Both NIST headline SE formulas also omit covariance
  between sphere metrics; recommend jackknifing the aggregate and paired model differences.
- Uniform T2′ diagnostic: 256 deterministic quantiles, 8 seeds, 0–3 T2′, max FID error 0.03818
  despite max seed SD 2.08e-16. Seed replication cannot establish deterministic quadrature
  convergence. Sobol real shifts lose exact dyadic balance; 128-seed analytic Fourier checks
  showed mixed effects on RMSE, so a digital-shift alternative needs protocol-level testing.
- Derived a free-segment bridge-conditioned estimator and checked it using the existing
  Philox samples: complex-variance reductions 1.35/1.54/2.81x at phase variances 0.1/1/4.
  Proposed exact joint-Gaussian free-interval aggregation, pilot allocation, stable sample
  IDs, control variates/multilevel RF corrections, prepared GPU buffers, and restricted-step
  bucketing. These are candidates, not claimed GPU speedups.
- Reproduction: `validation/monte_carlo_audit.py` and its checked-in JSON results. Targeted
  diffusion/restriction/sub-voxel tests: 21 passed, 6 GPU skips initially; one fused case hit
  the sandboxed default compiler cache and passed after redirecting `TORCH_EXTENSIONS_DIR`
  to `/tmp/blochparty-mc-audit-extensions` (22 passed total). MPS was unavailable, so no new
  GPU timings or full NIST runs were performed.
