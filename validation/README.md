# Validation

One folder per study: what it checks, against which independent reference, how to run it, and its results.
The unit tests are in `tests/`; these are the larger comparisons and the experiments.

| study | checks | run | results |
|---|---|---|---|
| [mrzero/](mrzero/) | k-space and images against MRzero (point isochromats, voxel dephasing) on GRE, spin echo, bSSFP and EPI | `uv run python validation/mrzero/compare_mrzero.py` (the `mrzero` extra); `pytest -m crosscheck` | [RESULTS.md](mrzero/RESULTS.md) |
| [bmctool/](bmctool/) | Z-spectra with a bound pool and free pools against BMCTool | `uv run --with bmctool==1.0.3 python validation/bmctool/compare_bmctool.py` | [RESULTS.md](bmctool/RESULTS.md) |
| [mt/](mt/) | the generalized Bloch model against the rate model; incidental MT in multi-slice imaging against Shin et al. | `uv run python validation/mt/generalized_bloch.py`, `.../incidental_mt.py` | [generalized Bloch](mt/RESULTS_GENERALIZED_BLOCH.md), [incidental MT](mt/RESULTS_INCIDENTAL_MT.md) |
| [jcoupling/](jcoupling/) | whether J coupling explains bright fat at 3 T (fat T2 by echo spacing) | `uv run python validation/jcoupling/jcoupling_fat.py` | [README](jcoupling/README.md) |
| [monte_carlo/](monte_carlo/) | the Monte Carlo estimators' diagnostics (`blochparty.montecarlo`) | `uv run python validation/monte_carlo/monte_carlo_audit.py` | [results](monte_carlo/monte_carlo_audit_results.json), [review](../docs/reviews/2026-10-08-monte-carlo.md) |
| [anatomy/](anatomy/) | anatomies and their tissue properties: BrainWeb, QSM Challenge, AHEAD, IXI | [README](anatomy/README.md) | [RESULTS.md](anatomy/RESULTS.md) |
| [em/](em/) | full-wave coil fields (B1+, receive maps, noise covariance) | [README](em/README.md) | [RESULTS.md](em/RESULTS.md) |
| [real/](real/) | real scanner data: NIST phantom T1 / T2 benchmarks, CEST, GIRF, radial and spiral trajectories | [README](real/README.md) | [RESULTS.md](real/RESULTS.md) |
| [realism/](realism/) | can a classifier tell simulated fastMRI T2w slices from real ones; sim-to-real transfer; the generator's command line | [README](realism/README.md) | [docs/REALISM.md](../docs/REALISM.md), the journal |
| [design/](design/) | sequence design by lesion classification under a k-space budget | [README](design/README.md) | the journal |

Scripts import their own folder's modules directly (Python puts a script's folder on the path) and everything else
from `blochparty`; the tests reach the few they import through `pyproject.toml`'s `pythonpath`.
