# Refactoring plan: a clearer layout for Bloch Party

First written 2026-10-11 from a read of the tree at `f983c00` (draft PR #8, branch `claude/gracious-pasteur-taw1kb`).
**Revised 2026-10-17 at `3663281`**, 222 commits later, for the tree as it is now; this version supersedes the one
on the PR branch. Status: **proposal**. Two small pieces exist already, made for the CUDA port: the GPU driver's
backend switch (`b8afd98`) and a GPU-engine test fixture (`b5cb509`). It is a plan for structure only: no physics
changes, no public-API renames (one addition is proposed, phase 3), every default bit-identical
(`tools/snapshot.py` before and after each phase, as `CLAUDE.md` requires).

## 0. What changed since the first version

- **A second GPU engine.** `blochparty.cuda` (`b5cb509`) reuses Metal's device code by turning
  `metal._SOURCE` / `_ADJOINT_COMMON` into CUDA with text rules, and Metal's Python driver through a backend object
  (`metal._Metal` / `cuda._Cuda`). Two engines now depend on shader text held in Python strings, and `cuda` reaches
  into `metal`'s private names (`_SOURCE`, `_ADJOINT_COMMON`, `_simulate`, `MAX_POOLS`, `ADJOINT_COILS`). The shader
  phase is no longer about one engine's readability; it is the shared source of two.
- **`anatomy.py` nearly doubled** (1126 → 2095 lines, 50 top-level names): the head generator's anatomical edits
  (scalp layering, dermis, lesions, falx, deep grey, ventricles, sulci, anthropometric scaling) and the IXI loader.
- **The M7 training-data generator lives in `validation/realism/`** as scripts that import one another through
  `sys.path` (`simulate_heads.py` 630 lines, `generate.py` 297, `score.py` 309, `transfer.py` 270, `gap.py` 294).
  The user named "a generator where you can say, generate a brain with this type of lesion" a big part of the product
  (journal 2026-10-14); today it cannot be imported as a library.
- **`validation/design/`** (the lesion-classification experiment) re-implements reconstruction pieces the package
  lacks: an orthonormal centred FFT pair, CG-SENSE, line masks.
- **Long runs go to cloud GPUs now** (an L40 on Runpod, the user's default since 2026-10-17), but 22 scripts call
  `simulate_metal` by name and so run only on Apple GPUs.
- Tests grew to 58 files, 397 test functions, 976 collected items; `rel()` is defined 12 times (was 8).
- `docs/JOURNAL.md` is 2729 lines (63 dated entries in 13 days, ~210 lines a day); `docs/LESSONS.md` is 1395.
- Unchanged: the `sequence` / `pulseq` import cycle, `design.py`, no formatter or linter (172 source lines and 72 test
  lines over 120 columns), no CI.

## 1. Summary

What is good and must survive the refactor:

- One flat public API, `import blochparty as bp`, with ~70 names re-exported from `__init__.py`; `blochparty.metal`
  and `blochparty.cuda` as the GPU import paths every script and document uses.
- One data model shared by every engine (`Phantom`, `Sequence`, `FieldMaps`, `Scanner`, `SimResult`, optional physics
  as `None`-default fields); optional dependencies imported lazily.
- `docs/ENGINES.md` as a capability matrix parsed by `tests/test_engines.py`, now with a `cuda` column.
- One GPU driver for two backends: a feature added to the Metal kernels and the driver needs only its kernel-side
  text for CUDA, and the two engines cannot drift apart in what they compute from a phantom and a sequence.
- The snapshot tool, the Pulseq conventions block, the journal and lessons discipline.

What hurts, with numbers (section 2 has the evidence):

1. **GPU source in strings.** `metal/__init__.py` is 2524 lines, 64 % of it Metal Shading Language in five string
   literals; `cuda/__init__.py` (863) adds 552 lines of CUDA in seven more and derives CUDA from Metal's strings by
   regular expressions. No editor support, no standalone compile check, kernel diffs read as Python string edits.
2. **Core and GPU driver duplicate the input handling**: default coils and B1, `spin_fields`, bound-pool expansion,
   the `m_init` split, static-versus-moving `FieldMaps` sampling, `rf_frequency`, lineshape and mode tables, the walk
   setup, the same `ValueError` texts. Every ROADMAP M6 item must be threaded through both.
3. **The generator is not a library.** Seven realism scripts import each other through `sys.path.insert`; the
   pieces a user of the generator needs (the realism preset's anatomy steps, the receive-array model with intensity
   normalisation, the protocol from a manifest, truth on the acquisition grid, spec + seed → sample) sit next to
   benchmark-only code (fastMRI fetching, CNN scoring, gap diagnostics).
4. **Scripts name an engine.** 22 scripts in `validation/` and `examples/` call `simulate_metal`; none can run on
   the L40 without editing.
5. **`anatomy.py` does six jobs**: tissue tables, four dataset loaders, the grid container, `to_phantom` (246
   lines), the anatomical edits, the head sampler (`sample_head` 233).
6. **Tests**: no `conftest.py` (the GPU fixture is pasted into four files); 12 copies of `rel`; case builders imported
   across test modules (`bm_case` by 6, `random_case` and `distortion` by 5 each); six test files and
   `tools/snapshot.py` reach into `validation/` or `examples/` through `sys.path`.
7. **Validation layout**: nine loose scripts and four `RESULTS*.md` at the top next to five study folders; 12 scripts
   edit `sys.path`; `auc` is written three times, noise addition twice beside the package's `coils.add_noise`.
8. **Docs**: the journal grows ~210 lines a day; the lessons' "Pulseq, data and reference values" section is 544
   lines; four dated lesson sections sit outside the topic sections.

What I propose, in order of payoff over risk:

| phase | what | kind | risk | checks |
|---|---|---|---|---|
| 1 | GPU engines: shaders and kernels as files, one `gpu/` driver, Metal and CUDA backends | mechanical | low | text identity, mini, L40 |
| 2 | tests: `conftest.py`, `cases.py`, one `rel`, a declared `pythonpath` | mechanical | low | collected items |
| 3 | one GPU entry point (`simulate_gpu`); the 22 scripts stop naming Metal | small addition | low | mini, L40 |
| 4 | the generator into the package (`blochparty.generate`), recon helpers into `recon` | structural | medium | sample identity, mini |
| 5 | `anatomy/` subpackage | mechanical | low | snapshot, tests |
| 6 | `engines/` with a shared `prepare` step and one capability table | structural | medium | snapshot, mini, L40 |
| 7 | `sequences/` subpackage, Pulseq read / write split, `meta` keys in one module | mostly mechanical | low–medium | snapshot |
| 8 | `validation/` as one study per folder; estimators and twix into the package | mechanical | low | scripts' cheapest runs |
| 9 | docs and tooling: journal by month, lessons by topic, ruff, generated layout table, CI | hygiene | low | — |
| 10 | `design/` subpackage | mechanical | low, optional | tests |

Phases 1–3 make the GPU code reviewable and every script portable to the L40, and need a day and a half together.
Phase 4 serves the product direction (ROADMAP M7). Phase 6 is the one real design change of the core; it should land
before ROADMAP M6.1 so the M6 items are added to one preparation path.

## 2. Evidence

Sizes at `3663281` (lines, `.py` and `.md`; at `f983c00` in brackets):

| area | lines | files | note |
|---|---|---|---|
| `src/blochparty` | 12 802 (10 074) | 34 (32) | 30 flat modules + `metal/` + `cuda/` + `tissuedb/` |
| `tests` | 10 068 (8 683) | 58 (51) | 397 test functions, 976 collected items |
| `validation` | 12 514 (8 857) | 68 (54) | scripts, result tables, pinned manifests; `realism/` 2235, `design/` 634 |
| `docs` | 6 384 (4 365) | 9 (8) | plus `README.md` 385, `ROADMAP.md` 697, `CLAUDE.md` 56 |
| `examples` | 940 | 10 | three imported by `tests/test_design.py` |

Largest source modules:

| module | lines | concerns |
|---|---|---|
| `metal/__init__.py` | 2524 | shader text 1603 (`_SOURCE` 706, `_KERNELS` 345, `_ADJOINT_COMMON` 327, `_ADJOINT_KERNEL` 182, `_PULSE_KERNEL` 43); the GPU driver for both backends: `_forward` 383, `_MetalSimulate` 115, `_simulate_grad` 54, `_reused_pulses` 48, `_readout_runs` 29; library build; `_Metal` backend |
| `anatomy.py` | 2095 | tissue tables + citations (`_tables` 101, `_mt_and_diffusion` 42); loaders `load_brainweb` 59, `load_qsm2` 106, `load_ahead` 58, `load_ixi` + `ixi_subjects`; `Anatomy` 101 + resampling; `to_phantom` 246, field maps, r2′; edits `layer_scalp`, `add_dermis`, `add_lesions` 89, `label_deep_grey`, `add_falx` 56, ventricles (4 functions), sulci (`open_sulci` 105 + 4); deformation, `draw_tissues`, anthropometric scaling, `sample_head` 233 |
| `cuda/__init__.py` | 863 | CUDA text 552 (prelude, warp helpers, five kernel families); the Metal-to-CUDA text rules; NVRTC and driver API (`_Driver` 91); `_Cuda` backend |
| `sequence.py` | 704 | `Sequence` + `kspace_trajectory`; `System`, waveforms, pulses, `SequenceBuilder` |
| `builders.py` | 697 | ready-made sequences |
| `pulseq.py` | 639 | reader, writer |
| `design.py` | 617 | hardware limits, image metrics, CRB, `minimize`, `design_report` |
| `bloch.py` | 603 | the core |

Import structure: still nearly a DAG, cycles dodged by lazy imports in `metal/__init__.py` (6), `pulseq.py` (4),
`sequence.py` (2), `io.py` (2), `exchange.py`, `design.py`, `builders.py`, `loops.py`. New edge: `cuda` → `metal`
private names.

Duplication between the core and the GPU driver (`metal._forward`, used by both GPU backends): unchanged from the
first version, with every row now serving two engines.

| step | `bloch.simulate` | GPU driver |
|---|---|---|
| default coils / B1 | yes | yes |
| `spin_fields(scanner, pos)` | yes | yes |
| `maps`: static sampled once, moving per segment; "either `coils` or `maps`" | yes | yes (same message) |
| `bound.expand`, `pool.f * pd` default, `m_init` 3 vs 4 columns | yes | yes (same message) |
| `rf_frequency(seq)` for the rate model | yes | yes |
| lineshape table / generalized-Bloch modes / cut-free modes | via `exchange` | re-derived from `exchange._sl_table`, `generalized_bloch.modes` |
| walk: `sqrt(D)` or Cholesky, restriction `tau_max`, `start_offsets` | yes | yes |

Test-suite sharing: `helpers.py` (`rel`, `with_readout`, `head_3d`, `gpu_engines`, `GPU`); case builders imported
across files: `test_exchange.bm_case` (6), `test_bloch_vs_ode.random_case` (5, and `tools/snapshot.py`),
`test_fields_nonlinearity.distortion` (5), `test_pools.pool_case` / `_maps_for` (3), `test_free_runs.busy_sequence`
/ `random_phantom` (3), `test_diffusion.pgse` (2), `test_metal_grad.compare` (1), `test_restriction.sgp`,
`test_susceptibility.sphere` (1 each). `sys.path.insert` in `test_design` (→ `examples`), `test_nist_estimators`,
`test_nist_twin` (→ `validation/real`), `test_crosscheck_mrzero` (→ `validation`), `test_realism_compact`,
`test_realism_helpers` (→ `validation/realism`, `validation/anatomy`).

Validation: loose at the top `compare_mrzero.py`, `compare_bmctool.py`, `generalized_bloch.py`, `incidental_mt.py`,
`jcoupling_fat.py`, `monte_carlo_audit.py` (+ its json), three `anatomy_*.py` whose results are in
`validation/anatomy/RESULTS.md`, and `RESULTS.md`, `RESULTS_BMCTOOL.md`, `RESULTS_GENERALIZED_BLOCH.md`,
`RESULTS_INCIDENTAL_MT.md`; folders `anatomy/` (8 scripts), `design/` (2), `em/` (3), `real/` (24), `realism/` (9).
Scripts that edit `sys.path`: `monte_carlo_audit.py`, four in `real/`, seven in `realism/`. Helpers written more than
once: `auc` (`realism/score.py`, `realism/gap.py`, `design/lesion_design.py`), noise addition (`realism/generate.py`,
`design/lesion_design.py`, and `coils.add_noise` in the package), centred FFTs (`realism/realism.py`,
`design/lesion_design.py`; the package has only `recon.ifftnc`), undersampling masks (`realism/transfer.py`,
`design/lesion_design.py`).

## 3. Target layout

Principle: group by what a contributor is doing (running a simulation, generating data, building a sequence or a
phantom, checking a result); keep the shared data model flat at the top; keep every public name importable from
`blochparty` through re-exports, and `blochparty.metal` / `blochparty.cuda` as import paths.

```
src/blochparty/
  __init__.py            the flat API (re-exports only), plus simulate_gpu (phase 3)
  constants.py  rng.py
  phantom.py  sequence.py  maps.py  motion.py  restriction.py       ← the data model
  exchange.py  generalized_bloch.py  fields.py  result.py           (result.py: SimResult, out of bloch.py)
  susceptibility.py  coils.py  loops.py  em.py                      ← phantom-side inputs (coils.py gains the
                                                                       receive-array model of the realism preset)
  receive.py  recon.py  io/                                         ← after the signal; recon.py gains the
      ismrmrd.py (today's io.py)  twix.py (from validation/real)      orthonormal FFT pair, CG-SENSE, line masks
  montecarlo.py          estimators and QMC filling from validation/real/nist.py
  reference.py  epg.py  jcoupling.py                                ← independent models for checks

  engines/                                                          ← how it is simulated
      __init__.py        simulate, simulate_chunked, simulate_gpu, available()
      prepare.py         one normalized case every engine consumes (phase 6)
      capabilities.py    what each engine accepts; one NotImplementedError site (phase 6)
      core.py            today's bloch.py
      chunked.py
      gpu/
          __init__.py    simulate_gpu: CUDA if available, else Metal; the backend protocol
          driver.py      today's metal._forward, _simulate, _readout_runs, _reused_pulses, _cost_order, _take,
                         _field_model_buffer
          adjoint.py     today's _MetalSimulate, _simulate_grad, default_checkpoint_every
          shaders/
              common.metal          today's _SOURCE: device code shared by both backends
              adjoint_common.metal  today's _ADJOINT_COMMON
          metal/
              __init__.py   simulate_metal, available, the METAL backend, library build
              kernels.metal.in  pulse.metal.in  adjoint.metal.in   (Metal kernels, templated as today)
          cuda/
              __init__.py   simulate_cuda, available, the CUDA backend
              translate.py  the text rules from shaders/*.metal to CUDA (today's _shared_source)
              rtc.py        NVRTC and the driver API (today's _Driver)
              prelude.cuh  common.cuh   kernels.cu.in  pulse.cu.in  replay.cu.in  adjoint.cu.in  seg_reduce.cu
  metal/__init__.py  cuda/__init__.py   one-line shims: blochparty.metal and blochparty.cuda keep working

  generate/                                                         ← training data (ROADMAP M7)
      __init__.py        default_spec, sample, write
      spec.py            DEFAULT_SPEC, merging and checking a spec (today's generate.DEFAULT_SPEC, merged)
      heads.py           the realism preset's anatomy for a seed (today's head_for and simulate_heads'
                         drop_detached, trace_sulci, thin_convexity_csf, flow_void, centre, turn)
      protocols.py       protocols from a manifest (real_protocols, sequence, interleaved)
      sample.py          spec + seed → simulated, noised sample with truth on the acquisition grid
                         (today's generate, truth_on_grid, kspace_on_fov)

  sequences/             builder DSL, library (today's builders.py), meta.py, pulseq/{read,write}.py,
                         girf.py, drift.py, transmit.py                (as in the first version)

  anatomy/
      __init__.py        re-exports (bp.load_brainweb, bp.sample_head, bp.DEFAULT_TISSUES unchanged)
      grid.py            Distribution, TissueProperties, Anatomy, conservative resampling
      tables.py          the tissue tables and citations, _mt_and_diffusion
      loaders/           brainweb.py  qsm.py  ahead.py  ixi.py   (SHA-pinned; nibabel stays lazy)
      phantom.py         to_phantom, anatomy_field_maps, estimate_macroscopic_r2prime, gaussian_random_field
      edits.py           layer_scalp, add_dermis, add_lesions, label_deep_grey, add_falx, ventricle_*, sulci
      sampler.py         DigitalPhantom, deformation, draw_tissues, anthropometric scaling, sample_head
      tissuedb/          moved here unchanged

  design/                optional (phase 10)
```

```
validation/
  README.md              one line per study and its command
  mrzero/  bmctool/  mt/ (generalized Bloch + incidental MT)  jcoupling/  monte_carlo/
  anatomy/  em/  real/
  realism/               the fastMRI realism benchmark: fetch_fastmri, score, gap, transfer, orientation,
                         echo_train; simulate_heads becomes a thin driver of blochparty.generate
  design/                the lesion-classification experiment
```

Naming notes: `sequence.py` next to `sequences/` is the one near-collision (alternative: `builders/`); `engines/core.py`
or `engines/bloch.py` (`bloch.py` is cited as "the reference" in `CLAUDE.md`, `ROADMAP.md`, `ENGINES.md`, the
README); `generate/` or `datagen/` or `synth/`. All three are open (section 9).

## 4. Phases

Every phase: its own commits on `main` (moves, then import fixes, then formatting; never mixed),
`tools/snapshot.py save` before and `compare` after where the snapshot covers the code, `uv run pytest -m "not
slow"` locally, the full suite on the mini, the four living documents updated (`README.md` layout,
`CLAUDE.md` paths, `ROADMAP.md` paths, `docs/ENGINES.md` entry points), a journal entry with hashes and numbers,
a lesson if anything surprised. GPU-touching phases also run the CUDA cells of the engine matrix on an L40 pod
(about 15 minutes, ~$0.20; the four GPU test files add 30 minutes of CPU-bound references, ~$0.40).

### Phase 1. GPU engines: sources as files, one driver (one day)

- Move the five Metal strings to `engines/gpu/shaders/*.metal` (shared device code) and
  `engines/gpu/metal/*.metal.in` (templated kernels), the seven CUDA strings to `engines/gpu/cuda/*.cuh` /
  `*.cu.in` / `*.cu`, loaded with `importlib.resources` (hatchling already ships package data: the tissuedb CSVs).
- `cuda/translate.py` reads `shaders/*.metal` instead of `metal._SOURCE`; the CUDA backend no longer imports Metal's
  private names. `driver.py` / `adjoint.py` hold the shared driver; `_MetalSimulate` becomes `_GPUSimulate`.
- **The check that needs no GPU**: before the move, dump Metal's concatenated library source (what `_lib()` passes to
  `torch.mps.compile_shader`) and `cuda.kernel_source(name)` for all 49 kernel names; after the move, assert both are
  byte-identical. Equal source in, equal kernels out. Then the GPU suites on the mini and the CUDA matrix on an L40 as
  the closing check, and `bench.py --backend metal` before and after (no runtime change expected; say so with
  numbers).
- Gains: syntax highlighting; `xcrun -sdk macosx metal -c shaders/common.metal` and the host-C++ stub check of the CUDA
  text (docs/LESSONS.md, "The CUDA port") as standalone compile checks; a kernel change diffs as a kernel change.
- Documents: `ROADMAP.md` rules that name `src/blochparty/metal/__init__.py`, `docs/ENGINES.md` entry column, the
  CUDA module docstring, README.

### Phase 2. Tests: shared fixtures in one place, no path hacks (half a day)

- `tests/conftest.py`: the `gpu` fixture (today pasted into four files), `f64`.
- `tests/cases.py`: `random_case`, `busy_sequence`, `random_phantom`, `pgse`, `bm_case`, `pool_case`, `_maps_for`,
  `distortion`, `sgp`, `sphere`, `compare` (GPU gradients); the owning modules import them back. One `rel` in
  `helpers.py`; the eleven copies go.
- `pyproject.toml`: `pythonpath = ["tests", "examples", "validation", "validation/real", "validation/realism",
  "validation/anatomy"]` replaces the `sys.path.insert` calls in six tests and `tools/snapshot.py` (which then imports
  `cases`); phases 4 and 8 shrink the list.
- Optional: test subfolders (`tests/engines/`, ...); the prefix names already group most files.
- Check: `uv run pytest --collect-only -q | tail -1` gives 976 before and after.

### Phase 3. One GPU entry point (half a day)

- `simulate_gpu(phantom, seq, coils=None, *, engine="auto", **kw)`: `simulate_cuda` when a CUDA device is there,
  else `simulate_metal`, else a clear error; `engine="metal" | "cuda"` to choose. Same arguments and results as both.
  The one public addition of this plan (`bp.simulate_gpu` and `blochparty.engines.simulate_gpu`).
- The 22 scripts in `validation/` and `examples/` that call `simulate_metal` call `simulate_gpu`; `bench.py` gains
  `--backend gpu`; `CLAUDE.md`'s command block and the README name it.
- Check: each script's cheapest run on the mini; the realism quick run and the lesion experiment's smoke run on an L40.

### Phase 4. The generator into the package (two days)

1. **Reconstruction helpers into `recon`** (own commit): the orthonormal centred FFT pair (`fft2c`, and an
   orthonormal `ifft2c` beside today's), CG-SENSE with given maps, line masks (uniform, variable density, the
   transfer test's random mask). Tests: the adjoint identity (today checked to 1e-7 in the design experiment),
   SENSE recovering a phantom at R = 2, the masks' line counts. `auc` and a bootstrap go to `design.py`'s metrics
   (or `blochparty.metrics`).
2. **The receive-array model into `coils.py`** (own commit): `coil_offset`, `normalised_sensitivities`, the drawn
   elliptical array of the realism preset; `b1_plus` into `transmit.py` if it is a transmit model. Tests in
   `test_receive.py`.
3. **`generate/`**: `spec.py` (`DEFAULT_SPEC`, merging), `heads.py` (the preset's anatomy for a seed), `protocols.py`,
   `sample.py` (`sample(spec, seed, engine="auto")` → k-space, noise, truth on the grid; `write` through `io`).
   `validation/realism/generate.py` and `simulate_heads.py` become thin command-line drivers; `score.py`, `gap.py`,
   `transfer.py`, `orientation.py`, `echo_train.py` stay as the benchmark and import the package.
- Check: **sample identity** — the same spec and seed written before and after give the same HDF5 datasets (one head,
  a few seconds on the mini; float32 atomics in the GPU coil sums make Metal runs differ in the last bits from run to
  run, so compare the noiseless truth bit for bit and the k-space to 1e-6); `test_realism_*`; one transfer-test run
  reading an existing set gives the same numbers.
- Risk: `simulate_heads.main` is 264 lines mixing the command line and the pipeline; split by what `generate.py`
  already calls (it imports `simulate_heads` as a module today) before moving anything else.

### Phase 5. `anatomy/` (half a day)

As section 3; the edits (scalp, dermis, lesions, falx, deep grey, ventricles, sulci) get their own module because the
generator adds to them more than to anything else in the package. Check: `test_anatomy*.py`, `test_head_sampler.py`,
`test_tissuedb.py`, `test_io.py`, the loaders' SHA round trips in `validation/anatomy/`, the tissuedb tools.

### Phase 6. `engines/`: one preparation path, one capability table (two to three days)

1. `git mv` `bloch.py` → `engines/core.py`, `chunked.py` → `engines/chunked.py`; `SimResult` → `result.py`;
   re-exports so `bp.simulate` and `bp.simulate_chunked` keep working.
2. `engines/prepare.py`: extract from `core.simulate` the pieces of section 2's duplication table, the core consuming
   the result; snapshot bit-identical (same expressions in the same order, only the enclosing function moves).
3. The GPU driver consumes the same `Prepared` for the pieces it duplicates and keeps its float32 packing, cost
   ordering and buffer layout; both GPU backends follow at once. Check: `test_engines.py` on the mini (Metal) and an
   L40 (CUDA), tolerances unchanged.
4. `engines/capabilities.py`: each engine (core, chunked, metal, cuda) declares the `Prepared` features it accepts
   for forward and gradients; one `check()` raises every `NotImplementedError` with one wording; `test_engines.py`
   asserts the declaration equals the ✓/✗ cells of `docs/ENGINES.md`.
5. `chunked.py` consumes `Prepared` too.
- Check: snapshot after steps 2, 3, 5; full suite on the mini after each step; `bench.py` on CPU and both GPUs before
  and after (preparation is a negligible share of runtime: say so with numbers).

### Phase 7. `sequences/` (one day)

As in the first version: `Sequence` stays in `sequence.py`; `System` and everything below it, and
`pulseq.flip_angle`, to `sequences/builder.py`, which removes the `sequence` ↔ `pulseq` lazy imports; one `_subdivide`;
`sequences/meta.py` with the `Sequence.meta` keys as constants and one table of meaning (a test asserts every key
written in `src/` is in it); `builders.py` → `sequences/library.py`; `girf.py`, `drift.py`, `transmit.py` moved.
Check: `test_sequences_*`, `test_pulseq*`, `test_girf`, `test_drift`, `test_transmit`, `test_io`, snapshot (the
builders are in it).

### Phase 8. `validation/`: one study per folder (half a day)

- Folders as section 3, each with `README.md` and `RESULTS.md`; a top-level `validation/README.md`.
- Promote the generic code of `validation/real/nist.py` (Monte Carlo estimators, Sobol filling) to
  `blochparty/montecarlo.py` (tested by today's `test_nist_estimators.py`, renamed) and `twix.py` to
  `blochparty/io/twix.py`; the realism scripts already import the package after phase 4.
- The 12 scripts' `sys.path.insert` calls go: a script's own directory is on the path when it runs; tests use the
  declared `pythonpath`.
- Check: each moved script runs with `--help` or its cheapest flag from the repository root; `pytest -m crosscheck`
  still finds `compare_mrzero`.

### Phase 9. Docs and tooling (half a day, alongside any phase)

- **Journal by month, now**: `docs/journal/2026-10.md` (today's file, moved), `docs/JOURNAL.md` an index of one line
  per entry; `CLAUDE.md` points at the current month. At ~210 lines a day the single file is past use already.
- **Lessons by topic**: `docs/lessons/{physics,data,performance,process,gpu,design}.md`; the four dated sections
  fold into their topics; the 544-line "Pulseq, data and reference values" section splits into Pulseq and data.
- One-off documents in dated folders: `docs/reviews/2026-10-08-monte-carlo.md` (today's `MONTE_CARLO_REVIEW.md`).
- `tools/layout.py` generates the README layout table from module docstrings, with `--check` run by a test (as
  `test_engines.py` checks `ENGINES.md`).
- `ruff`: `line-length = 120`, `select = ["E", "F", "I", "B"]`; one formatting commit per phase, never mixed with
  moves or logic; `uv run ruff check` and `ruff format --check` in `CLAUDE.md`'s commands.
- CI: a GitHub Actions job running `ruff` and `pytest -m "not slow"` on CPU (GPU tests skip there; the mini and the
  L40 stay the GPU runners). Only useful once `main` is pushed regularly: today local `main` is 222 commits ahead of
  `origin/main`.

### Phase 10. `design/` (half a day, optional)

`design.py` (617 lines: hardware limits and PNS, image metrics, CRB, `minimize`, `design_report`) splits only when
ROADMAP M7's "parameterized sequence API" starts adding to it; the lesion experiment's `learnable_gre` and `Design`
(flips, TE, TR and line positions as constrained parameters) are the first candidate for `design/parameters.py`.
`tests/test_design.py` importing three example scripts is decided here: move their reusable parts into the package,
or keep the declared `pythonpath`.

## 5. Design of the non-mechanical pieces

### 5.1 The GPU layout

The backend protocol that exists today stays the interface: `name`, `device`, `lib()` (kernels called as
`kernel(*buffers, threads=, group_size=)`), `synchronize()`, `max_coils`. The driver knows nothing else about a backend.
The shared device code is Metal Shading Language (`shaders/common.metal`, `adjoint_common.metal`), compiled as it is
by Metal and through `cuda/translate.py`'s rules (address spaces dropped, vector types and precise math mapped onto
`prelude.cuh`, scalar math to its float forms) by NVRTC. New physics goes into the shared files once; a kernel-side
change goes into both `*.metal.in` and `*.cu.in`; the engine matrix checks both against the core.

Not proposed: writing the device code in a neutral dialect (macros for both languages). The text rules are about 20 lines,
checked by 49 compiled kernels and 68 matrix cells; a dialect would make the Metal source, the one most edited, harder
to read.

### 5.2 `engines/prepare.py`

```python
@dataclass(frozen=True)
class Prepared:
    pos: Tensor; pd: Tensor; r1: Tensor; r2: Tensor; db0: Tensor; b1: Tensor   # after defaults and expansion
    sens: Tensor | None          # (N, C) complex, None = one uniform coil
    fields: SpinFields           # encoding positions + concomitant coefficients (static spins)
    moving_field: tuple | None   # (nonlinearity, f0, z_asymmetry) when spins move under the field model
    maps: MapsPrepared | None    # grid, origin, spacing, has_coils; sampled already if spins are static
    bound: BoundPool | None      # expanded per spin
    bound_tables: BoundTables | None   # rf_freq, lineshape table or modes, kind, cutoff
    pools: tuple[ExchangePool, ...]
    walk: Walk | None            # sqrt(D) / Cholesky factor, seed, restriction + tau_max + start offsets
    motion: tuple | None
    m0: tuple[Tensor, Tensor, Tensor, Tensor | None]   # mx, my, mz, mb after the m_init split
    steps: list[tuple[int, int]] # the core's merge plan; the GPU driver ignores it
    features: frozenset[str]     # ENGINES.md feature names this case uses
```

`prepare(...)` holds every `ValueError` of today's two paths once. What does not move into it: the GPU cost ordering,
the hi / lo float32 split, buffer layouts, `checkpoint_every` policy, the core's checkpoint chunking.

### 5.3 `engines/capabilities.py`

As in the first version, with `cuda` added (today identical to `metal`). The feature names are the `docs/ENGINES.md`
row names, so `test_engines.read_matrix()` checks the declaration against the document cell by cell.

### 5.4 `blochparty.generate`

```python
from blochparty import generate
spec = generate.default_spec()                      # today's DEFAULT_SPEC: IXI v17 heads, the realism preset
spec["anatomy"]["lesions"] = {"count": [2, 4], "region": "periventricular", "size_mm": [5, 12]}
s = generate.sample(spec, seed=7, engine="auto")    # dict: kspace (C, n, n), noise floor, truth {class: (n, n)}, meta
generate.write(s, "sample_7.h5")                    # ISMRMRD with a /truth group, as today
```

A spec is data (JSON), so a run is reproducible from its spec and seed, as `generate.py` already guarantees; the
package version adds no behaviour, only the import path and a documented `spec` schema (`spec.py` checks unknown
keys).

## 6. Verification per phase

| check | how | cost | phases |
|---|---|---|---|
| core bit-identity | `tools/snapshot.py save` / `compare` (widen it first where a phase moves uncovered code) | seconds | 4–7 |
| GPU source identity | Metal's library text and `kernel_source` of all 49 CUDA kernels, byte for byte | seconds, no GPU | 1, 3, 6 |
| Metal | the GPU test files and the matrix on the mini | ~15 min | 1, 3, 4, 6 |
| CUDA | the matrix's cuda cells on an L40 (the four GPU files too when the driver changes) | $0.20–0.60 | 1, 3, 6 |
| test inventory | `pytest --collect-only -q`: 976 items | seconds | 2, 8 |
| generated data | same spec + seed before and after: truth bit for bit, k-space to 1e-6 | seconds on the mini | 4 |
| scripts | each moved script's cheapest run | minutes | 3, 4, 8 |
| speed | `bench.py` on CPU, Metal and CUDA before and after | minutes | 1, 6 |

## 7. Documents to update in each phase

Living documents: `README.md`, `CLAUDE.md`, `ROADMAP.md`, `docs/ENGINES.md` (entry points), `docs/LESSONS.md`
where a lesson names a file, `validation/*/README.md`, `src/blochparty/tissuedb/README.md`, the CUDA and Metal module
docstrings. Historical documents (`docs/JOURNAL.md` or its monthly files, `docs/ROADMAP_ARCHIVE.md`) are not
rewritten; one line at the top of each says paths before the refactor refer to section 2's layout.

## 8. Rules for the refactor

1. **No physics, no API removal.** Every public name stays importable from `blochparty` with the same signature;
   `blochparty.metal` and `blochparty.cuda` stay import paths. The only addition is `simulate_gpu` (phase 3).
2. **Bit-identical defaults** per `CLAUDE.md`; for GPU code, identical generated source (section 6).
3. **Moves and edits in separate commits**: `git mv` first, then the import fixes, then formatting; `git diff -M` of a
   move commit shows no logic change.
4. **Commits on `main`, one phase at a time** (`CLAUDE.md`: commit finished, verified work directly on `main`), each
   phase finished with the mini green and, for GPU phases, the L40 green. Phases 1, 2 and 9 are independent; 3 wants
   1; 4 wants 2 and 3; 5 wants 2; 6 wants 1 and 2; 7 wants 2; 8 wants 4; 10 last.
5. **Parity rules still apply**: a phase that touches the GPU driver measures the plain kernel's speed before and after
   (ROADMAP's shader rule), even when the text only moved.
6. **Journal every phase** with numbers (test count, snapshot result, bench delta, pod cost) and hashes.

## 9. Not doing, and open decisions

Not doing: renaming public functions; merging the core's and the GPU kernels' stepping (different algorithms for
different hardware); an engine plugin registry (four engines, one table of declarations); a neutral shader dialect
(section 5.1); moving `ROADMAP.md` or `CLAUDE.md`; touching `uv.lock` beyond adding `ruff`.

Open for the author, in the order the phases need them:

1. Phase 2: test subfolders, or keep the flat prefix names?
2. Phase 3: is `simulate_gpu` the name (alternatives: `simulate_fast`, `bp.simulate(..., engine="gpu")`)?
3. Phase 4: `generate/`, `datagen/` or `synth/`?
4. Phase 6: `engines/core.py` or `engines/bloch.py`; keep the `blochparty.metal` / `.cuda` shims for good?
5. Phase 7: `sequences/` (next to `sequence.py`) or `builders/`?
6. Phase 9: retro-split the journal into months, or start the monthly files now and leave October whole? Add CI, and
   start pushing `main` (222 commits are local only)?
7. Phase 10: move the design examples' reusable code into the package, or keep importing scripts in tests?

Estimated total: about nine working days for phases 1–9, of which phases 1–3 (a day and a half) and 9 (half a day) pay
off immediately.
