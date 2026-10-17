# 2026-10-17 — Refactor: the design package and its worked problems (plan phase 10)

- **`design/`** (`87f1800` moves, `3e41172` split): `hardware`, `metrics` (with the AUC and bootstrap of phase 4),
  `estimation`, `optimize`, `report`, every statement of `design.py` moved verbatim; `blochparty.design` exports all
  of them as before.
- **`design/parameters.py`**: the lesion experiment's `learnable_gre` and `Design` (flips, TE, TR, line positions as
  constrained parameters) with their acquisition constants; `lesion_design.py` imports them; a 19-line design gives
  identical sequences and summaries before and after.
- **`design/problems/`**: `despot1`, `contrast_tr`, `tse_flips`, the three example scripts without their `main()`;
  the examples are thin command lines and reach their reference optima as before (DESPOT1 at 70.71 % of the Ernst
  signal; the contrast/TR budget optimum; the TSE train within 7e-7 deg of SLSQP on EPG, 24.1 % of the constant
  train's RF energy). `tests/design` imports the package and `examples` left the tests' `pythonpath`.
- Checks: design tests 16 passed; the widened core snapshot bit-identical.
- **Full suite on the mini** (phases 4, 8 and 10 together, with the test fix): 859 passed, 180 skipped, 0 failed, 13 min.
