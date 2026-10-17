# Monte Carlo diagnostics

`monte_carlo_audit.py` reproduces the small CPU-only diagnostics of [docs/MONTE_CARLO_REVIEW.md](../../docs/MONTE_CARLO_REVIEW.md) (the debiased magnitude, the jackknife of an aggregate, quasi-Monte-Carlo spin placement, T2' offsets and a
bridge between estimators: five experiments) with `blochparty.montecarlo`; it prints the numbers recorded in [monte_carlo_audit_results.json](monte_carlo_audit_results.json).

    uv run python validation/monte_carlo/monte_carlo_audit.py
