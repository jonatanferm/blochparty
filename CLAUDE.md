# Bloch Party

Differentiable MRI simulator: PyTorch core (`src/blochparty/bloch.py`, the reference
implementation) and an Apple-GPU kernel with a hand-written adjoint (`src/blochparty/metal/`, the
fast engine). The fused C++ CPU kernel was retired on 2026-10-09 (in the git history).
Goal: the most physically realistic simulator, judged against real scanners
(`docs/REALISM.md`); roadmap in `ROADMAP.md`.

## Working rules

- **Nothing is done without an independent check**: closed forms, the SciPy ODE solver, EPG,
  the signal equation, MRzero (`validation/`), or real data (`validation/real/`).
- **Defaults stay bit-identical.** New physics is opt-in. Before and after a core change run
  `uv run python tools/snapshot.py save <file>` and `compare <before> <after>`.
- **One fast engine: Apple GPU (`blochparty.metal`).** New physics goes into the PyTorch core
  (the readable reference, checked on small cases) and the Metal kernel (forward, and the adjoint
  where calibration or design needs it), which must agree with it. Other engines and the CPU+GPU
  hybrid come later.
- **`docs/ENGINES.md` is the capability matrix**, checked cell by cell by `tests/test_engines.py`:
  when an engine gains or refuses a feature, change its cell (and add a row and a case for a new
  feature).
- **Record what you learn in `docs/LESSONS.md`**: anything that took more than a few minutes
  to figure out (physics, numerics, data/format quirks, tool behaviour, process), with where
  it came from.
- **Commit without asking.** Commit finished, verified work directly on `main`, split into
  logical commits (what and why, with numbers), and put the hashes in the journal. Push only
  when asked.
- **Keep `docs/JOURNAL.md` current**: after each milestone or work session, append a dated
  entry — what was done and why, results with numbers, dead ends and corrections, commit
  hashes — so the project can be written up later.

## Commands

```bash
uv sync --extra pulseq --extra mrzero
uv run pytest                      # full suite (GPU tests skip without MPS)
uv run pytest -m "not slow"        # quick subset (< 1 min) while iterating
uv run pytest -m crosscheck        # opt-in MRzero agreement
uv run python benchmarks/bench.py [--backend metal] [--grad] [--record]
uv run python validation/real/fetch_data.py && uv run python validation/real/nist_irse.py
```

## Conventions

Pulseq units (RF Hz, gradients Hz/m, s, m); dM/dt = γ M×B, a pulse along +x tips +z to +y;
Mxy = Mx + iMy evolves as e^{−i2πft}; arrays ordered (z, y, x); ADC sample i at (i+½)·dwell.
Physical γ is `constants.GAMMA_BAR` (CODATA); `PULSEQ_GAMMA_BAR` only for hardware limits.
