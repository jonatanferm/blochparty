# 2026-10-17 — Refactor: validation, one study per folder (plan phase 8)

- **Moves** (`68612fc`): `validation/mrzero/` (compare_mrzero + RESULTS), `bmctool/`, `mt/` (generalized Bloch,
  incidental MT), `jcoupling/`, `monte_carlo/`; the three `anatomy_*.py` into `anatomy/`; `io.py` is `io/ismrmrd.py`
  and `validation/real/twix.py` is `io/twix.py`.
- **`blochparty.montecarlo`** (`137384f`): `sobol_points`, `mc_power`, `mc_magnitude`, `jackknife` from
  `validation/real/nist.py` (which imports them back); their tests are `tests/signal/test_montecarlo.py`, the NIST
  ones stay. The Monte Carlo audit imports the package instead of reaching into `validation/real`.
- **Paths and imports** (`d07d31d`): the anatomy scripts find their data next to them; fourteen `sys.path.insert`
  lines that added a script's own folder go; the tests' `pythonpath` names `validation/mrzero` instead of
  `validation`. `iohexol.py` imports `twix` inside `recon()`: imported at the top, `blochparty.io` brought torch in
  next to finufft's OpenMP in `iohexol_t1.py`, which then aborted (`OMP: Error #15`).
- **READMEs** (`a2f128d`): `validation/README.md` lists every study with what it checks, its command and its
  results; seven folders got their first README. Every relative link in the living documents resolves (one the
  lessons split had broken now points one level up).
- Checks: every moved or edited script runs from the repository root (`--help`, or its quick run where it takes no
  options; `nist_twin` and `iohexol` dispatch on a subcommand and never had `--help`); `pytest -m crosscheck` finds
  `compare_mrzero` in its new folder and passes; the mini's mirror lost its stale copies of the moved files through
  a previewed `rsync --delete`.
- **Full suite on the mini** (phases 4, 8 and 10 together, with the test fix): 859 passed, 180 skipped, 0 failed, 13 min.
