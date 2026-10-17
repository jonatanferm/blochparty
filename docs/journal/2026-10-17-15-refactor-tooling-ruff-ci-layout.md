# 2026-10-17 — Refactor: ruff, CI, the generated layout table (plan phase 9, tooling)

- **Layout table** (`33ef727`): `tools/layout.py` writes README.md's layout table from the module
  docstrings (one row per top-level module or package, forwarding modules left out); `tests/docs` runs `--check`
  and now also checks that every relative link in the living documents resolves, which caught the Monte Carlo
  review's two links when it moved to `docs/reviews/2026-10-08-monte-carlo.md`.
- **ruff** (`6c1473a`): line length 120; E, F, I, B, with seven rules ignored for the codebase's style, each with its
  reason in pyproject (B023's closures in loops checked case by case: all are called within their iteration). Of
  1124 findings, the ones kept were fixed: imports sorted, unused imports, a dead assignment in `receive`, two
  mid-file imports, a blind `pytest.raises(Exception)` narrowed to the checksum's ValueError, a `System()` default
  argument, and `simulate_heads`' closure (`8905a89`: its loop freed arrays with `del`, which made the linter think
  the closure's names undefined; it rebinds them to None now).
- **Formatting** (`f4eb759`, formatting only, listed in `.git-blame-ignore-revs`): `ruff format` over the code, 185
  files; Markdown excluded (ruff 0.17 formats code blocks in documents, which would undo their alignment; three
  documents it had touched were restored).
- **CI** (`6362c93`): GitHub Actions on every push and pull request: ruff check and format check (ruff 0.17.1,
  pinned there and in the dev group), the quick tests on Ubuntu with CPU torch. Simulated first on a clean
  `git archive` of HEAD (no untracked data): lint and format clean, 828 passed, 185 skipped, 0 failed.
- Checks for the formatting: the widened snapshot bit-identical, the GPU sources byte-identical, the quick suite on
  the laptop 835 passed, 178 skipped.
- Found on the way: a plain `uv sync` removes the extras (it makes the environment exact); `CLAUDE.md`'s command
  now says so and lists the anatomy extra.
