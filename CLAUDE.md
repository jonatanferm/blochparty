# Bloch Party

Differentiable MRI simulator: PyTorch core (`src/blochparty/engines/core.py`, the reference
implementation) and a GPU engine with a hand-written adjoint (`src/blochparty/engines/gpu/`, the fast
engine: one driver, Metal on Apple GPUs and CUDA on NVIDIA), both behind `bp.simulate(..., engine=)`.
The fused C++ CPU kernel was retired on 2026-10-09 (in the git history).
Goal: the most physically realistic simulator, judged against real scanners
(`docs/REALISM.md`); roadmap in `ROADMAP.md`.

## Working rules

- **Nothing is done without an independent check**: closed forms, the SciPy ODE solver, EPG,
  the signal equation, MRzero (`validation/`), or real data (`validation/real/`).
- **Defaults stay bit-identical.** New physics is opt-in. Before and after a core change run
  `uv run python tools/snapshot.py save <file>` and `compare <before> <after>`.
- **One fast engine: the GPU driver (`blochparty.engines.gpu`), Metal and CUDA backends.** New physics
  goes into the PyTorch core (the readable reference, checked on small cases) and the GPU device code
  (`engines/gpu/shaders/*.metal`, shared by both backends; kernel-side changes in `metal/*.metal.in` and
  `cuda/*.cu.in`), forward and the adjoint where calibration or design needs it, which must agree with it.
  Other engines and the CPU+GPU hybrid come later.
- **`docs/ENGINES.md` is the capability matrix**, checked cell by cell by `tests/engines/test_engines.py`:
  when an engine gains or refuses a feature, change its cell (and add a row and a case for a new
  feature).
- **Record what you learn in `docs/lessons/<topic>.md`** (index: `docs/LESSONS.md`): anything that took more
  than a few minutes to figure out (physics, numerics, data/format quirks, tool behaviour, process), with where
  it came from.
- **Commit without asking.** Commit finished, verified work directly on `main`, split into
  logical commits (what and why, with numbers), and put the hashes in the journal. Push `main`
  after every finished phase of `docs/REFACTOR_PLAN.md` (the user's request, 2026-10-17);
  otherwise push only when asked.
- **Keep the journal current**: after each milestone or work session, add an entry file
  `docs/journal/<date>-<nn>-<slug>.md` (nn: its place in the day) and its line in the index `docs/JOURNAL.md`:
  what was done and why, results with numbers, dead ends and corrections, commit hashes, so the project can be
  written up later.

## Commands

```bash
uv sync --extra pulseq --extra mrzero --extra io
uv run pytest                      # full suite (GPU tests skip without MPS)
uv run pytest -m "not slow"        # quick subset (< 1 min) while iterating
uv run pytest tests/engines        # one area (tests/<area>/: engines, physics, sequences, ...)
uv run pytest -m crosscheck        # opt-in MRzero agreement
uv run python benchmarks/bench.py [--backend gpu] [--grad] [--record]   # gpu: CUDA if present, else Metal
uv run python validation/real/fetch_data.py && uv run python validation/real/nist_irse.py
```

## Machines

Long runs (the full suite, GPU validation and benchmarks) go to the Mac mini (`ssh mini`, Apple
M4, 16 GB): a mirror at `~/projects/blochparty` with uv at `~/.local/bin/uv`. Sync with
`rsync -az --exclude .git --exclude .venv --exclude __pycache__ --exclude outputs --exclude .claude --exclude
validation/real/data --exclude validation/anatomy/data --exclude '*.eml' --exclude '*.dmg' --exclude '*.pkg' ./
mini:projects/blochparty/` (copy only the data a job needs; the anatomy data is 11 GB, the `.eml` holds signed
links; after moving or deleting tests also `rsync -az --delete tests/ mini:projects/blochparty/tests/`), run with
`ssh mini 'cd ~/projects/blochparty && ~/.local/bin/uv run pytest -q'`, long jobs under `nohup`
with a log. Commit from this checkout only. Small runs here are fine.

## Conventions

Pulseq units (RF Hz, gradients Hz/m, s, m); dM/dt = γ M×B, a pulse along +x tips +z to +y;
Mxy = Mx + iMy evolves as e^{−i2πft}; arrays ordered (z, y, x); ADC sample i at (i+½)·dwell.
Physical γ is `constants.GAMMA_BAR` (CODATA); `PULSEQ_GAMMA_BAR` only for hardware limits.
