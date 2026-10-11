# Refactoring plan: a clearer layout for Bloch Party

Written 2026-10-11 from a read of the whole tree at `f983c00`. Status: **proposal, nothing
implemented**. It is a plan for structure only: no physics changes, no public-API renames, every
default bit-identical (`tools/snapshot.py` before and after each phase, as `CLAUDE.md` requires).

## 1. Summary

What is already good and must survive the refactor:

- One flat public API, `import blochparty as bp`, with ~70 names re-exported from `__init__.py`.
- One data model shared by every engine: `Phantom`, `Sequence`, `FieldMaps`, `Scanner`, `SimResult`,
  with optional physics as `None`-default fields. Optional dependencies (pypulseq, nibabel, h5py,
  ismrmrd, scipy) are all imported lazily inside functions, so the package imports with torch alone.
- `docs/ENGINES.md` as a capability matrix *parsed by* `tests/test_engines.py`: the document cannot
  go stale. The same trick is proposed below for the README layout table.
- The snapshot tool, the Pulseq conventions block, the journal/lessons discipline.

What hurts, with numbers (section 2 has the evidence):

1. `src/blochparty/metal/__init__.py` is 2095 lines, 70 % of it Metal Shading Language inside four
   Python string literals, templated by `@SUFFIX@`/`@V@`/`@DIFF@`/`@NP@`/`@NC@` replacements. It is
   the most edited source file in the repository and has no editor support, no syntax check, no
   readable diffs.
2. The core (`bloch.simulate`) and the Metal host (`metal._forward`) each re-derive the same
   things from the same inputs by hand: default coils and B1, `spin_fields`, bound-pool expansion,
   the `m_init` split with its bound column, static-vs-moving `FieldMaps` sampling, `rf_frequency`,
   lineshape and generalized-Bloch mode tables, the walk setup, even the same `ValueError` strings.
   Every new physics item (ROADMAP M6 has five) must be threaded through both by hand.
3. `anatomy.py` (1126 lines) does five jobs: literal tissue tables with citations, three dataset
   loaders with SHA pins, the `Anatomy` grid container, `to_phantom` (205 lines), and the generative
   head sampler. A second tissue table (`tissuedb/`) sits beside it as a separate package.
4. `sequence.py` (660) holds both the `Sequence` dataclass and the whole builder DSL; `pulseq.py`
   (636) holds reader and writer; the two import each other through five lazy imports and share a
   near-verbatim `_subdivide`. `Sequence.meta` is a dict with 28 distinct string keys read in
   eleven source and test files, with no single place that says what they mean.
5. Tests: 51 flat files, no `conftest.py`; `rel()` is defined eight times; `test_engines.py`
   imports case builders from six other test files; four test files `sys.path.insert` their way
   into `validation/` and `examples/`; `tools/snapshot.py` imports `tests.test_bloch_vs_ode`.
6. `validation/` mixes eight loose scripts and three `RESULTS_*.md` at the top with four study
   folders that each have their own `RESULTS.md`; `validation/real/nist.py` (460 lines) is a shared
   library of Monte Carlo estimators and phantom fillers reached by six scripts and two tests
   through `sys.path` hacks (`nist_twin.py` alone has three).
7. Docs: `docs/JOURNAL.md` is 1289 lines after three days (32 dated entries); `docs/LESSONS.md` is
   965. The README's hand-written layout table already omits `constants.py` and `tissuedb/`.
   There is no formatter or linter configuration (101 source lines exceed 120 columns, the de facto
   limit).

What I propose, in order of payoff over risk:

| phase | what | kind | risk |
|---|---|---|---|
| 1 | shaders to `.metal` files, Metal host split into forward / adjoint / library | mechanical | low |
| 2 | tests: `conftest.py`, one `helpers.py` + `cases.py`, declared `pythonpath`, optional subfolders | mechanical | low |
| 3 | `validation/` as one study per folder; generic estimators and the twix reader into the package | mechanical | low |
| 4 | `engines/` subpackage with a shared `prepare` step and one capability table | structural | medium |
| 5 | `anatomy/` subpackage (tables, loaders, grid, to_phantom, sampler, tissuedb) | mechanical | low |
| 6 | `sequences/` subpackage: builder DSL out of `sequence.py`, Pulseq read/write split, `meta` keys documented in one module, cycle removed | mostly mechanical | low–medium |
| 7 | docs and tooling: generated layout table, journal by month, ruff, result-file naming | hygiene | low |
| 8 | `design/` subpackage (hardware, metrics, estimation, report) | mechanical | low, optional |

Phases 1–3 and 7 can be done in a day each and do not touch physics. Phase 4 is the one real
design change and the one with the most lasting payoff; it should land before ROADMAP M6.1 so the
five M6 items only have to be added to one preparation path.

## 2. Evidence

Sizes at `f983c00` (lines; `.py` and `.md` only):

| area | lines | files | note |
|---|---|---|---|
| `src/blochparty` | 10 074 | 32 | 30 flat modules + `metal/` + `tissuedb/` |
| `tests` | 8 683 | 51 | 341 test functions, ~685 with parameters |
| `validation` | 8 857 | 54 | scripts, result tables, pinned manifests |
| `docs` | 4 365 | 8 | plus `README.md` 381 and `ROADMAP.md` 546 at the root |
| `examples` | 940 | 10 | three of them are imported by `tests/test_design.py` |
| `tools`, `benchmarks` | 602 | 4 | |

Largest source modules and what they hold:

| module | lines | concerns |
|---|---|---|
| `metal/__init__.py` | 2095 | shader text 1465 (`_SOURCE` 704, `_KERNELS` 250, `_ADJOINT_COMMON` 328, `_ADJOINT_KERNEL` 183); host: `_forward` 252, `_MetalSimulate` 115, `_simulate_grad` 52, library build, cost ordering |
| `anatomy.py` | 1126 | tissue tables + citations (~200), `load_brainweb` / `load_qsm2` / `load_ahead` (~225), `Anatomy` + resampling, `to_phantom` 205, `DigitalPhantom` + deformation + `draw_tissues` + `sample_head` |
| `sequence.py` | 660 | `Sequence` dataclass + `kspace_trajectory` (~130); `System`, `Trapezoid`, `GradientWaveform`, `RFPulse`, `ADC`, `trapezoid`, `sinc_pulse`, `SequenceBuilder` 191 |
| `pulseq.py` | 636 | `read_pulseq` 154 + helpers; `_Writer` 117, `write_pulseq`, `_write_imported` |
| `design.py` | 617 | hardware limits + PNS + checks (~160); image metrics + PSF (~90); Jacobian / Fisher / CRB; `minimize`; `design_report` 78 |
| `bloch.py` | 596 | the core: single-pool stepping, free pools, orchestration in `simulate` (126) |
| `builders.py` | 568 | ready-made sequences |

Import structure (from `grep` of relative imports): the package is nearly a DAG. The cycles are
dodged with lazy imports at `sequence.py:67` (→ `exchange.rf_frequency`), `sequence.py:127`
(→ `pulseq.flip_angle`), `pulseq.py` ×4 (→ `sequence`), `exchange.py:341`
(→ `generalized_bloch.modes`), `design.py:351` (→ `bloch.simulate`), `io.py:189` (→ `anatomy`,
`pulseq`), and six in `metal/__init__.py`. The clusters the imports already form are: the data
model (`phantom`, `sequence`, `maps`, `motion`, `restriction`, `exchange`, `generalized_bloch`,
`fields`), the engines (`bloch`, `chunked`, `metal`), sequence making (`sequence`'s builder half,
`builders`, `pulseq`, `girf`, `drift`, `transmit`), phantom making (`anatomy`, `tissuedb`,
`susceptibility`, `coils`, `loops`, `em`), and the rest (`receive`, `recon`, `io`, `reference`,
`epg`, `design`).

Duplication between the core and the Metal host (same logic, both hand-written):

| step | `bloch.simulate` | `metal._forward` |
|---|---|---|
| default coils / B1 | yes | yes |
| `spin_fields(scanner, pos)` | yes | yes |
| `maps`: static sampled once, moving per segment; "either `coils` or `maps`" check | yes | yes (same message) |
| `bound.expand(n, pd)`, `pool.f * pd` default, `m_init` 3 vs 4 columns, "m_init has a bound-pool column" check | yes | yes (same message) |
| `rf_frequency(seq)` for the rate model | yes | yes |
| super-Lorentzian table / modes of the generalized model / cut-free modes | via `exchange` | re-derived from `exchange._sl_table`, `generalized_bloch.modes` |
| walk: `sqrt(D)` or Cholesky, restriction `tau_max`, `start_offsets` | yes | yes (adjoint path) |
| free-pool guards (`MAX_POOLS`, generalized + pools) | n/a / raises | raises |

Test-suite sharing today: `helpers.py` (`rel`, `with_readout`, `head_3d`); case builders imported
across files: `test_bloch_vs_ode.random_case` (used by 4 files), `test_free_runs.busy_sequence` /
`random_phantom` (3), `test_exchange.bm_case` (2), `test_diffusion.pgse`, `test_pools.pool_case`,
`test_fields_nonlinearity.distortion`, `test_restriction.sgp`, `test_susceptibility.sphere` (1
each). `sys.path.insert` in `test_nist_estimators`, `test_nist_twin` (→ `validation/real`),
`test_crosscheck_mrzero` (→ `validation`), `test_design` (→ `examples`).

## 3. Target layout

Principle: **group by what a contributor is doing** (running a simulation, building a sequence,
building a phantom, checking a result), keep the shared data-model dataclasses flat at the top
where every subpackage can import them without cycles, and keep the public API exactly as it is
through re-exports in `blochparty/__init__.py`.

```
src/blochparty/
  __init__.py            the flat API, same names as today (re-exports only)
  constants.py  rng.py
  phantom.py  sequence.py  maps.py  motion.py  restriction.py        ← the data model: what is simulated
  exchange.py  generalized_bloch.py  fields.py  result.py            (result.py: SimResult, out of bloch.py)
  susceptibility.py  coils.py  loops.py  em.py                        ← phantom-side inputs (B0 from chi, coil maps)
  receive.py  recon.py  io/                                          ← after the signal: receive chain, recon, files
      ismrmrd.py (today's io.py)   twix.py (from validation/real)
  reference.py  epg.py                                               ← independent models for checks

  engines/                                                           ← how it is simulated
      __init__.py        simulate, simulate_chunked, simulate_metal (re-exports), available()
      prepare.py         NEW: Prepared = one normalized case both engines consume (section 5)
      capabilities.py    NEW: what each engine accepts; one NotImplementedError site
      core.py            today's bloch.py (stepping only)
      chunked.py
      metal/
          __init__.py    simulate_metal, available
          forward.py     buffers, dispatch (today's _forward, _cost_order, _take, _field_model_buffer)
          adjoint.py     _MetalSimulate, _simulate_grad, default_checkpoint_every
          library.py     template expansion + torch.mps.compile_shader, importlib.resources
          shaders/
              common.metal            today's _SOURCE (compiles standalone)
              kernels.metal.in        today's _KERNELS (templated entry points)
              adjoint_common.metal    today's _ADJOINT_COMMON
              adjoint.metal.in        today's _ADJOINT_KERNEL

  sequences/                                                         ← building and transforming sequences
      __init__.py        SequenceBuilder, System, pulses, gre_2d … (re-exports)
      builder.py         System, Trapezoid, GradientWaveform, RFPulse, ADC, trapezoid, sinc_pulse,
                         block_pulse, block_duration, SequenceBuilder, flip_angle (today's sequence.py second half
                         + pulseq.flip_angle)
      library.py         today's builders.py
      meta.py            the Sequence.meta keys: names as constants, one table of meanings
      pulseq/
          read.py  write.py  (today's pulseq.py)
      girf.py  drift.py  transmit.py   (Sequence → Sequence transforms; unchanged)

  anatomy/                                                           ← building phantoms from anatomy
      __init__.py        Anatomy, DigitalPhantom, to_phantom, sample_head, loaders … (re-exports)
      grid.py            Distribution, TissueProperties, Anatomy, conservative resampling
      tissues.py         DEFAULT_TISSUES, BRAINWEB_TISSUES, the citation strings, _mt_and_diffusion
      loaders/
          brainweb.py  qsm.py  ahead.py   (SHA-pinned loaders; nibabel stays lazy)
      phantom.py         to_phantom, anatomy_field_maps, estimate_macroscopic_r2prime, gaussian_random_field
      sampler.py         DigitalPhantom, B-spline deformation, draw_tissues, sample_head
      tissuedb/          moved here unchanged (package + CSVs + README)

  design/                                                            ← optional, phase 8
      hardware.py  metrics.py  estimation.py  optimize.py  report.py
```

What deliberately stays where it is: `ROADMAP.md`, `CLAUDE.md`, `README.md` at the root (cited
everywhere); `benchmarks/`, `tools/`, `examples/`; `docs/ROADMAP_ARCHIVE.md` and the journal as
historical records (their old paths are not rewritten, see section 8).

Naming notes and alternatives:

- `sequence.py` (the dataclass) next to `sequences/` (the builders) is the one near-collision in
  this layout. The alternative is to call the package `builders/`
  (`blochparty.builders.SequenceBuilder`, `blochparty.builders.pulseq`), which reads well for
  the DSL and the library but less well for the transforms. Decide at phase 6; nothing earlier
  depends on it.
- `engines/core.py` versus keeping the name `bloch.py`: `bloch.py` is cited in `CLAUDE.md`,
  `ROADMAP.md`, `docs/ENGINES.md` and the README as "the reference". Keeping `engines/bloch.py`
  costs nothing and keeps those sentences true; `core.py` matches the matrix's column name. Either
  is fine; I lean to `engines/core.py` and updating the four living documents.
- `io/` as a package shadows nothing in the stdlib at package level (`blochparty.io` already
  exists as a module today).

## 4. Phases

Every phase: branch, `git mv` for history, `uv run python tools/snapshot.py save` before and
`compare` after, `uv run pytest -m "not slow"` locally and the full suite on the mini, then
update the four living documents (`README.md` layout, `CLAUDE.md` paths, `ROADMAP.md` paths in
the working rules and open items, `docs/ENGINES.md` entry-point column), a journal entry with the
commit hashes, and a line in `docs/LESSONS.md` if anything surprised.

### Phase 1. Metal: shaders as files, host split (about half a day)

- Move the four string literals to `metal/shaders/*.metal` (plain) and `*.metal.in` (templated),
  loaded with `importlib.resources.files(__package__) / "shaders"`. Hatchling already ships
  non-Python files under the package (the tissuedb CSVs rely on it).
- `library.py`: the variant expansion that `_lib()` does today, unchanged logic, plus a
  `source()` function returning the final text so a test can assert it is byte-identical to
  today's concatenation (one-off check, then deleted).
- `forward.py` / `adjoint.py` as listed above; `__init__.py` keeps `simulate_metal`, `available`,
  `MAX_COILS`, `MAX_POOLS`, `ADJOINT_COILS`, `MMA_MIN_COILS`, `N_SPIN_GRADS`.
- Gains: syntax highlighting; on the mini `xcrun -sdk macosx metal -c shaders/common.metal`
  checks the untemplated files without a GPU run; diffs of a kernel change are diffs of a kernel.
- Check: `tests/test_metal.py`, `tests/test_metal_grad.py`, `tests/test_engines.py` on the mini;
  `benchmarks/bench.py --backend metal` before and after (compile time is cached per process, so
  no runtime change is expected; say so with numbers).
- Documents: `ROADMAP.md` "Metal variants" and "Metal's adjoint raises" rules name
  `src/blochparty/metal/__init__.py`; `docs/ENGINES.md` entry column; README layout row.

### Phase 2. Tests: shared fixtures in one place, no path hacks (half a day)

- `tests/conftest.py`: the `metal` fixture (`importorskip` + `available()`), a `f64` helper,
  marker registration moved from `pyproject.toml` if preferred.
- `tests/cases.py`: `random_case`, `busy_sequence`, `random_phantom`, `pgse`, `bm_case`,
  `pool_case`, `distortion`, `sgp`, `sphere`, moved out of the test modules that own them today
  (those modules import them back). `tests/helpers.py` keeps `rel` (one definition), `with_readout`,
  `head_3d`; the seven local `rel` copies go.
- `pyproject.toml`: `[tool.pytest.ini_options] pythonpath = ["tests", "validation", "validation/real", "examples"]`
  replaces the four `sys.path.insert` calls in tests and the one in `tools/snapshot.py` (which then
  imports `cases`). This is a declared hack instead of five undeclared ones; phase 3 removes the
  need for `validation/real` by promoting the generic code.
- Optional: subfolders mirroring the package, `tests/engines/`, `tests/physics/`,
  `tests/sequences/`, `tests/phantoms/`, `tests/signal/`, `tests/validation/`, so that
  `uv run pytest tests/engines` is the natural subset. The prefix naming (`test_phantom_*`,
  `test_sequences_*`, `test_fields_*`) already does most of this; weigh the ~50 renames against
  that. `docs/ENGINES.md` cites test files by basename only, so the matrix is unaffected either way.
- Check: `uv run pytest` collects the same ~685 items (`--collect-only -q | tail -1` before and
  after).

### Phase 3. Validation: one study per folder, shared code into the package (half a day)

Top-level `validation/` today: `compare_mrzero.py` + `RESULTS.md`, `compare_bmctool.py` +
`RESULTS_BMCTOOL.md`, `generalized_bloch.py` + `RESULTS_GENERALIZED_BLOCH.md`, `incidental_mt.py`
+ `RESULTS_INCIDENTAL_MT.md`, `monte_carlo_audit.py` + `..._results.json`, three `anatomy_*.py`
whose results live in `validation/anatomy/RESULTS.md`, and the folders `anatomy/`, `em/`, `real/`,
`realism/`.

- One folder per study, each with `README.md` (what and how to run), `RESULTS.md`, the scripts:
  `validation/mrzero/`, `validation/bmctool/`, `validation/mt/` (generalized Bloch + incidental
  MT), `validation/monte_carlo/`, `validation/anatomy/` (absorbs the three loose scripts),
  `validation/em/`, `validation/real/`, `validation/realism/`. A top-level `validation/README.md`
  lists them with one line each and the command.
- Promote generic code out of `validation/real/nist.py`: the Monte Carlo estimators (`mc_power`,
  `mc_magnitude`, `jackknife`, `MCErrors`, `mc_errors`, `mc_paired_difference`) and the Sobol /
  QMC phantom filling (`_sobol_points`, `fill_phantom_qmc`) become `blochparty/montecarlo.py`
  (tested by today's `test_nist_estimators.py`, renamed `test_montecarlo.py`); `twix.py` becomes
  `blochparty/io/twix.py`. What stays in `nist.py` is NIST-specific: reference relaxation values,
  sphere geometry, the authors' reconstruction, ROI metrics.
- `nist_twin.py`'s three `sys.path.insert` calls and `iohexol.py`'s two go; scripts import their
  siblings as plain modules (a script's own directory is on `sys.path` when run), and the two tests
  that need `nist` / `nist_twin` use the declared `pythonpath` from phase 2 or move next to the
  scripts under `validation/real/tests/` with `testpaths` extended.
- Rule to write into `validation/README.md` (it is already practice): results tables live in the
  `RESULTS.md` beside the script that made them; the roadmap gets a line and a link.
- Check: each moved script runs with `--help` (or its cheapest flag) from the repository root;
  `uv run pytest -m crosscheck` still finds `compare_mrzero`.

### Phase 4. `engines/`: one preparation path, one capability table (two to three days)

The structural change. Section 5 has the design. Order inside the phase, each step its own commit
with a snapshot compare:

1. `git mv` `bloch.py` → `engines/core.py`, `chunked.py` → `engines/chunked.py`, `metal/` →
   `engines/metal/`; `SimResult` → `result.py`; re-exports so `bp.simulate`, `bp.simulate_chunked`,
   `blochparty.metal.simulate_metal` keep working (`blochparty/metal/__init__.py` as a one-line
   shim for the latter, since `ROADMAP.md`, `docs/ENGINES.md`, the README and every script use that
   path; or update them all and skip the shim).
2. `engines/prepare.py`: extract from `core.simulate` exactly the pieces listed in section 2's
   duplication table, with the core consuming the result. Snapshot must be bit-identical; the
   order of floating-point operations inside each extracted piece does not change, only where the
   code lives.
3. The Metal host consumes the same `Prepared` for the pieces it duplicates (defaults, expansion,
   `m_init` split, static-maps sampling, tables) and keeps its own float32 packing, cost ordering
   and buffer layout. Check with `test_engines.py` on the mini: tolerances unchanged.
4. `engines/capabilities.py`: each engine declares the `Prepared` features it accepts for forward
   and for gradients; one `check(prepared, engine, grad)` raises the `NotImplementedError`s that
   `chunked.py` (four sites) and the Metal host and adjoint (three sites plus `_simulate_grad`'s
   `present` list) raise today, with one message format. `test_engines.py` gains a cheap check that
   the declared sets equal the ✓/✗ cells of `docs/ENGINES.md`; its behavioural checks stay.
5. `chunked.py` consumes `Prepared` too (it already rejects most features; the table makes that
   explicit).

Check: snapshot bit-identical after steps 2, 3 and 5; the full suite on the mini after each step;
`benchmarks/bench.py` and `--backend metal` before and after the phase (preparation is a
negligible share of runtime, say so with numbers).

### Phase 5. `anatomy/` (half a day)

- Split as in section 3: `grid.py`, `tissues.py`, `loaders/{brainweb,qsm,ahead}.py`, `phantom.py`,
  `sampler.py`; `tissuedb/` moves under `anatomy/`. The `__init__.py` re-exports every name the
  package root exports today, so `bp.load_brainweb`, `bp.sample_head`, `bp.DEFAULT_TISSUES` are
  unchanged, and `from blochparty import tissuedb` keeps working through a re-export.
- `tissues.py` and `tissuedb/` side by side make the open decision (switch `DEFAULT_TISSUES` to
  `tissuedb.tissue_table`, journal 2026-10-10) a one-directory diff; the decision itself is not
  part of this plan.
- Check: `test_anatomy*.py`, `test_head_sampler.py`, `test_tissuedb.py`, `test_io.py`; the
  SHA-256 round-trip tests in `validation/anatomy/` pin the loaders' output; `tools/tissuedb_report.py`
  and `tools/tissuedb_import_itis.py` run.

### Phase 6. `sequences/` (one day)

- `sequence.py` keeps `Sequence`, `_remap_spans`, `kspace_trajectory`, `subdivide_rf`, `to`;
  everything from `System` down moves to `sequences/builder.py`. `pulseq.flip_angle` moves to the
  builder module, which removes the `sequence → pulseq` lazy import; `pulseq/read.py` and
  `pulseq/write.py` import `sequences.builder` normally, which removes the four lazy imports the
  other way. One `_subdivide` (the two differ only in whether the input points are included;
  one keyword).
- The `sequence → exchange` lazy import in `subdivide_rf` stays unless `rf_frequency` moves to a
  small `rf.py` next to the data model; low value, note it and move on.
- `sequences/meta.py`: the 28 `Sequence.meta` keys as named constants with a table of meaning,
  unit and who writes / reads each (`te`, `tr`, `esp`, `kspace_shape`, `kspace_index`, `blocks`,
  `matrix`, `dwell`, `pulseq_source`, `pulseq_fingerprint`, `echo_times`, `system`, `subject`,
  `deformation`, `block_segments`, `segment_block`, `adc_per_readout`, …). The dict stays a dict
  (it round-trips through JSON in `io`), but `recon.kspace_grid`, `design`, `io`, `chunked` and the
  builders read the constants. A test asserts every key written in `src/` appears in the table.
- `builders.py` → `sequences/library.py`; `girf.py`, `drift.py`, `transmit.py` move unchanged.
- Check: `test_sequences_*.py`, `test_pulseq*.py`, `test_girf.py`, `test_drift.py`,
  `test_transmit.py`, `test_io.py`; snapshot (the builders are in it).

### Phase 7. Docs and tooling (half a day, can run alongside any phase)

- `tools/layout.py`: prints the README layout table from each module's first docstring line (the
  docstrings are already written to serve as that), and `--check` fails if `README.md` differs.
  A test calls `--check`, the way `test_engines.py` checks `ENGINES.md`. The table then cannot
  omit `constants.py` or `tissuedb/` again.
- Journal by month: `docs/journal/2026-10.md` (today's file moved and renamed), `docs/JOURNAL.md`
  becomes a one-line-per-entry index with links, or is dropped in favour of the folder; `CLAUDE.md`
  points at "the current month's file". At the present rate (about 400 lines a day) the single
  file is unusable within weeks.
- Lessons: keep the one file while sections stay under ~300 lines; the rule for splitting
  (`docs/lessons/<topic>.md`) goes into `CLAUDE.md` now. The two dated sections at the end
  ("Monte Carlo estimator audit", "Tissue-property database") fold into the topic sections.
- One-off documents get a dated folder: `docs/reviews/2026-10-08-monte-carlo.md` (today's
  `MONTE_CARLO_REVIEW.md`); `docs/specs/ANATOMY.md` → `docs/ANATOMY.md` (the only spec; `PHANTOMS.md`
  and `REALISM.md` are specs too and live at the top).
- `ruff`: `[tool.ruff] line-length = 120`, `select = ["E", "F", "I", "B"]`, `uv run ruff check`
  and `uv run ruff format --check` added to the quick commands in `CLAUDE.md`; one formatting commit
  per phase so that formatting and moves never mix in a diff with logic. `uv run ruff check` also
  catches the duplicate names and unused imports that moves leave behind.
- Optional: a GitHub Actions job running `ruff` and `pytest -m "not slow"` on CPU for pull requests
  (the repository already takes PRs, e.g. the `tissuedb` merge). GPU tests skip there; the mini
  remains the real runner, as `CLAUDE.md` says.

### Phase 8. `design/` (half a day, optional)

`design.py` is 617 lines with four concerns (hardware limits and PNS; image metrics and PSF;
Jacobian, Fisher information, CRB; `minimize` and `design_report`). Split only if ROADMAP M7
("sequence spec in", "parameterized sequence API", "benchmark task suite") starts adding to it;
until then the single module is tolerable.

## 5. Design of the two non-mechanical pieces

### 5.1 `engines/prepare.py`

```python
@dataclass(frozen=True)
class Prepared:
    # spins, after defaults and expansion (dtype/device of the caller's phantom)
    pos: Tensor; pd: Tensor; r1: Tensor; r2: Tensor; db0: Tensor; b1: Tensor
    sens: Tensor | None          # (N, C) complex, None = one uniform coil
    fields: SpinFields           # encoding positions + concomitant coefficients (static spins)
    moving_field: tuple | None   # (nonlinearity, f0, z_asymmetry) when spins move under the field model
    maps: MapsPrepared | None    # grid, origin, spacing, has_coils; sampled already if spins are static
    bound: BoundPool | None      # expanded per spin
    bound_tables: BoundTables | None   # rf_freq, lineshape table or modes, kind, cutoff
    pools: tuple[ExchangePool, ...]    # expanded per spin
    walk: Walk | None            # sqrt(D) / Cholesky factor, seed, index, restriction + tau_max + start offsets
    motion: tuple | None         # (Motion, per_segment, start)
    m0: tuple[Tensor, Tensor, Tensor, Tensor | None]   # mx, my, mz, mb after the m_init split
    steps: list[tuple[int, int]] # _steps(seq, merge) — the core's; Metal ignores it
    features: frozenset[str]     # names from the ENGINES.md feature column that this case uses
```

`prepare(phantom, seq, coils, m_init, scanner, maps, seed, merge_free_runs) -> Prepared` holds
every `ValueError` of today's two paths once. The core's `_Spins` NamedTuple is built from it in
three lines; the Metal host packs its float32 buffers from it. `features` is what
`capabilities.check` tests against.

Bit-identity argument: every extracted expression is the same tensor expression evaluated in the
same dtype in the same order; only the enclosing function changes. The snapshot compare after
each commit is the proof, not the argument.

What does not move into `prepare`: the Metal cost ordering (`_cost_order`, `_take`), the hi/lo
float32 split, buffer layouts, `checkpoint_every` policy, and the core's `checkpoint` chunking.
Those are engine-specific by nature.

### 5.2 `engines/capabilities.py`

```python
FORWARD  = {"core": ALL, "chunked": BASIC | {"B1+", "GIRF / eddy currents", "uniform field per segment", "many coils (40)"},
            "metal": ALL - {"free pools + generalized Bloch"}}
GRADIENT = {"core": ALL, "chunked": FORWARD["chunked"],
            "metal": ALL - {"FieldMaps, moving spins", "bound pool, generalized Bloch", "free pools", "diffusion", "motion", ...}}

def check(prepared: Prepared, engine: str, grad: bool) -> None:
    missing = prepared.features - (GRADIENT if grad else FORWARD)[engine]
    if missing: raise NotImplementedError(f"{engine}: {', '.join(sorted(missing))} not implemented ({'gradients' if grad else 'forward'}); use blochparty.simulate")
```

The feature names are the `docs/ENGINES.md` row names, so `test_engines.read_matrix()` can assert
`FORWARD[e] == {row for row, cells in matrix.items() if cells[(e, "forward")] == "✓"}` for each
engine. Today the matrix is enforced only behaviourally (run the case, expect a raise); this adds
the declaration the behaviour must match, and gives the three engines one wording.

## 6. Tests, validation, tools, examples

- `examples/` doubles as test code: `tests/test_design.py` imports `design_contrast_tr`,
  `design_despot1`, `design_tse_flips`. Either keep that with the declared `pythonpath` (phase 2)
  or move the reusable parts of those three scripts into `blochparty.design` and leave the scripts
  as thin drivers. The second is cleaner; the first is zero work. Decide at phase 8.
- `tools/snapshot.py` imports `random_case` from a test file; after phase 2 it imports
  `tests.cases`. Consider widening the snapshot to the ready-made sequences and `phantom_from_maps`
  options the refactor touches (it covers `gre_2d` and `shepp_logan` today); a wider snapshot is
  the cheapest insurance for phases 5 and 6. Add in its own commit, per the roadmap rule.
- `benchmarks/history.jsonl` stays; record one run before phase 1 and one after phase 4 so the
  table shows the refactor cost nothing.

## 7. Documents to update in each phase

Living documents (update): `README.md` (layout table, code snippets with paths), `CLAUDE.md`
(paths in the first paragraph and the rules), `ROADMAP.md` (working rules, the engines table, open
items that name files), `docs/ENGINES.md` (entry-point column), `docs/LESSONS.md` where a lesson
points at a file, `validation/*/README.md`, `src/blochparty/tissuedb/README.md`.

Historical documents (do not rewrite): `docs/JOURNAL.md` and `docs/ROADMAP_ARCHIVE.md`. Add one
line at the top of each: "paths before 2026-10-xx refer to the layout described in
`docs/REFACTOR_PLAN.md` §2". `validation/*/RESULTS.md` tables keep their script names as run
commands; update the command line, not the numbers.

## 8. Rules for the refactor

1. **No physics, no API.** Every public name in `blochparty/__init__.py` stays importable from
   `blochparty`, with the same signature. Module paths change; that is the whole change.
2. **Bit-identical defaults** per `CLAUDE.md`: snapshot before and after every commit that moves
   code the snapshot covers; widen the snapshot first (own commit) where it does not.
3. **Moves and edits in separate commits.** `git mv` first (history follows), then the edit that
   fixes imports, then any formatting. A reviewer should be able to `git diff -M` and see no logic
   change in a move commit.
4. **One phase per branch, merged when the mini is green.** Phases 1, 2, 3, 7 are independent of
   one another; 4 wants 1 and 2 first; 5 and 6 want 2; 8 last.
5. **Parity work follows the rules that exist.** Phase 4 step 3 is a Metal change; the roadmap's
   "measure the plain variant's speed before and after any shader change" applies even though the
   shader text is only moved.
6. **Journal every phase** with numbers (test count, snapshot result, bench delta) and hashes.

## 9. Not doing, and open decisions

Not doing: renaming public functions; merging the core's and Metal's stepping code (they are
different algorithms for different hardware); a plugin registry for engines (three engines, one
file of declarations is enough); moving `ROADMAP.md` or `CLAUDE.md`; splitting `bloch.py`'s
stepping functions further; touching `uv.lock` beyond adding `ruff` to the dev group.

Open for the author to decide, in the order the phases need them:

1. Phase 2: test subfolders, or keep flat prefix names?
2. Phase 4: `engines/core.py` or `engines/bloch.py`; keep a `blochparty.metal` shim or update
   every reference?
3. Phase 6: package name `sequences/` (next to `sequence.py`) or `builders/`?
4. Phase 7: journal by month from now, or also retro-split the existing file?
5. Phase 7: add the CPU-only GitHub Actions job?
6. Phase 8 and `examples/`: move the design examples' reusable code into the package, or keep
   importing scripts in tests?
