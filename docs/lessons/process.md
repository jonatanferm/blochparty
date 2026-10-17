# Lessons: Process and tooling

- **Mendeley Data's API rejects Python's urllib** (403); `curl` works.
  (`validation/real/fetch_data.py`)

- **Read numbers from the page source, not from a summary of it**: WebFetch's summaries of PMC
  tables dropped columns and invented "not detailed" for values that were there (Li et al. 2014
  ADC, Sinclair et al. 2010 per-subject table). `curl` the PMC HTML and print the `<table>`s and
  the MathML annotations instead.

- **zsh does not split `$var` into words**: `pytest $files` with a space-separated list ran
  nothing ("no tests ran"); use `${=files}` or list the files.

- **zsh does not split words** (2026-10-11): `cmd $args` with args="--a 1 --b 2" passes one argument in zsh
  (the session's shell), so argparse rejected every run of a loop whose output went to /dev/null and three
  variant folders came out empty; use `${=args}`, and keep the logs.

- **Scratch scripts named like standard modules break imports** (2026-10-11): `inspect.py` once,
  then a `profile.py` next to a new script made torch's optimizer import fail (`cProfile` imports
  `profile`); Python puts the script's directory first on the path. Keep scratch scripts in their
  own directory, with names no module has.

- **Finite differences on float32 signals need big steps.** DESPOT1's CRB uses a central-difference
  Jacobian (`rel_step` 1e-4): on float32 signals (~1e-6 rounding after 3700 segments) the bound is
  ~1 % noise and the optimizer landed at (3.7°, 38°) instead of (4.1°, 23.6°); 1e-2 gives 0.05°.
  Source: M2, `examples/design_despot1.py`.

- **A test's random weights must not depend on the engine's output dtype.** `torch.randn(shape,
  dtype=...)` with one generator gives different numbers in float32 and float64: every Metal gradient
  cell of the matrix failed by ~100 % until the loss drew its weights in float64. Source: M2.

- **`SequenceBuilder.rf(duration, ...)` splits a long pulse into raster steps by default.** A
  "single 0.5 s CW segment" made with `rf(0.5, amplitude=...)` was thousands of segments (30 s in
  the exact engine); pass `n_steps=1` for one segment. Source: M1c, 2026-10-08.

- **A finite difference can be noise-limited before it is truncation-limited.** On spins 10 cm out
  under a gradient nonlinearity, a central difference with a 1e-6 relative step agreed with
  autograd to only ~1e-6 (rounding of the loss / step), and a 1e-4 step to 8e-5 (truncation).
  Richardson extrapolation of central differences at 1e-5 and 2e-5 (error O(h⁴)) fixed the check.
  Source: `tests/test_engines.py`, M1b, 2026-10-08.

- **A validation script's entry point goes last.** `compare_bmctool.py` defined `cest_comparison`
  after `if __name__ == "__main__": main()`, so running it as a script died with a NameError
  before the CEST section; found when rerunning it through Metal. Source: M1a, 2026-10-08.

- **A finite-difference check must rebuild its inputs.** Perturbing leaf tensors in place misses
  everything derived from them when the case was built (`apply_girf` output, `subdivide_rf`
  pieces, a tensor scaled by a leaf): the GIRF, `rf_freq` and diffusion-scale checks failed for
  that reason alone. And a leaf computed from another leaf (a carrier from `rf_frequency(rf)`)
  carries a dependence the rebuilt case sees and autograd does not. Build cases as functions of
  independent leaves and rebuild at θ ± εv; give each run its own graph. Source:
  `tests/test_engines.py`, 2026-10-08.

- **`pytest -m EXPR` on the command line replaces the `-m` in `addopts`** (the last one wins), so
  `pytest -m "not slow"` would have run the opt-in crosscheck tests: 4 collected with
  `-m "not nothing"`, 0 without. The crosscheck tests are therefore also marked `slow`.
  Source: M0, 2026-10-08.

- **Perturb a symmetric matrix symmetrically.** A finite difference along an asymmetric
  perturbation of a diffusion tensor disagreed with autograd by 8 %: the Cholesky factor reads
  one triangle, its backward returns a symmetrized gradient. Perturb a per-spin scale (or
  D + Dᵀ) instead. Source: `tests/test_engines.py`, 2026-10-08.

- **QSM Challenge releases are not interchangeable.** Radboud's complete head package
  declares restricted access and RU-DI-HD-1.0, despite the initial spec's CC BY guess.
  The old Stage-1 short link is a closed registration form. An author's lab separately
  links a public SURFdrive challenge share with both chi/segmentation ground truths,
  but no R1/R2*/M0 or phantom-generation code. Read-only WebDAV listing avoided a 12.65 GB
  bulk download; needed GT files total ~107 MB. (`validation/anatomy/QSM.md`, 2026-10-07)

- **Nominal resolution is not the NIfTI affine.** QSM challenge GT headers use 0.999024391 mm
  while frequency/mask images use exactly 1 mm. Preserve and check affine geometry, and
  do not silently zip arrays of equal shape as if they occupied identical positions.
  The supplied frequency maps also include phase estimation/noise/downsampling, so are
  not exact convolution references for the distributed GT. (`validation/anatomy/QSM.md`)

- **Verify delivered BrainWeb bytes, not just its legacy format page.** Subject-04
  fuzzy files are 0.5-mm (362,434,362), unlike its separate 1-mm T1 reference. The
  public raw-byte response was signed/offset-encoded despite an unsigned description;
  naive uint8 decoding made the air background look like nearly equal tissue mixtures.
  Raw shorts decoded correctly as little-endian 0…4095, despite a big-endian claim.
  Pin decompressed-byte hashes (gzip headers vary), check range, and record physical
  origins from the download header. (`validation/anatomy/README.md`, 2026-10-07 capture)

- **BrainWeb fuzzy totals need asymmetric normalization.** Clamp overfull class sums
  to unity, but leave deficits as air. Normalizing every tiny positive sum to one
  amplifies background memberships into full tissue. Include the explicit background
  class in the normalization before conservative resampling. (`load_brainweb`)

- **The two BrainWeb releases have different validation coverage.** The newer 20-model
  set publishes only T1 (22/9.2 ms, 30°); older SBD provides T1/T2/PD. Its older tissue
  file reproduces older GM/WM contrasts within 0.13%, but newer subject-04 T1 differs
  by 5.44% GM/WM and 21.31% CSF/GM. Do not present a cross-simulator comparison as
  clinical validation, or assume both releases use identical parameters. (`RESULTS.md`)

- **Guard default behaviour with bit-identity snapshots** before and after any core change:
  `tools/snapshot.py save/compare` (both simulators, outputs and gradients, float32/64).

- **Parallel workers need file ownership**, and integration still finds what no single worker
  could: two different γ constants, and design checks defaulting to 150 T/m/s against
  builders designing at 170 T/m/s (every built sequence was reported as a slew violation).
  Write integration tests across workers' features (`tests/test_integration.py`).

- **Expect README/ROADMAP merge conflicts** from parallel workers; resolve as unions.

- **The Agent tool's "remote" isolation fell back to local worktrees** in this setup; real cloud
  runs are started from claude.ai/code with a written brief. `git add -A` then captured
  `.claude/worktrees/` as embedded repos; it is git-ignored now.

- **Cloud workers may merge despite instructions** (PR #1 was merged under the owner's
  account). Re-verify locally after every merge regardless.

- **GitHub marks a PR merged** when its head commit lands in the base branch by a local merge
  and push, drafts included (it can take a few seconds).

- **The Mac sleeps with the lid closed** (clamshell, especially on battery); long runs pause
  and `caffeinate -i` doesn't prevent it. Watch for wall time ≫ CPU time.

- **Scripts must not overwrite tracked outputs by default**: test runs of `nist_tse.py` replaced
  the published benchmark figure; figures are now written only with `--figure`.

- **`while pgrep -f "<text>"` can wait forever**: the waiting shell's own command line contains
  `<text>`. Wait on a PID (`while kill -0 <pid>`) instead.

- **`cmd | grep` buffers background output** until the command ends; use `python -u` without a
  filter, or filter afterwards.

- **Dataset “raw” does not necessarily mean scanner k-space** (validation-source search,
  2026-10-07). Pulseq-CEST's MAGMA release supplies reconstructed DICOMs; CEST–MRF also
  calls image/signal inputs raw. Verify arrays or archive entries before promising a complete
  acquisition benchmark. PulseqDiffusion's phantom MATLAB file really contains complex64
  k-space, but its current generator defaults differ from the stored acquisition dimensions
  and no scanned `.seq` is included. By contrast, the iohexol CEST release includes both
  Siemens `.dat` and `.seq`. Sources and verification: `validation/real/SOURCES.md`.

- **Pin dataset versions and inspect large ZIP directories before downloading**: Zenodo
  concept DOIs can resolve to newer, substantially larger archives. The iohexol CEST concept
  record resolves to 17244380 (31.6 GB), while an indexed older version is 12.3 GB. HTTP range
  reads verified the actual phantom raw files, sequences and calibration scans from its
  ~63 kB ZIP directory without downloading the archive. `SOURCES.md` records the version.

- **finufft and torch cannot share a process on macOS** (each loads its own libomp: "OMP: Error
  #15", either import order). `KMP_DUPLICATE_LIB_OK` is documented as unsafe; run the NUFFT
  stages in their own process (`scholand_recon.py`) and pass arrays by file.

- **Pipe-buffered Python prints arrive only at exit**: a long driver looks hung under `| grep`;
  check the cache files it writes instead (or use `python -u`).

- **A dataset title or README does not establish that all source maps are public**:
  the NeuroPoly phantom repository and OSF archive expose chi/T2/field derivatives,
  but their inventories lack the original R1/M0 inputs. The Jochmann/Schweser Zenodo
  archive also contains derived images and fields, not the full source package.
  AHEAD template v3 does expose aligned chi/R1/R2*/relative-PD maps under CC BY 4.0.
  Its `r2map` filename means R2*, and its RPS orientation requires a y flip for RAS.
  Relative PD must not be presented as calibrated absolute M0; zero QSM does not
  identify air. BigBrain-MR's TE=0 GRE magnitude also retains TR/flip-angle weighting
  and is not M0. Actual inventories and checks: `validation/anatomy/ALTERNATIVE_SOURCES.md`.

- **Removing a module silently skips every test file that `importorskip`s it**, core tests
  included (2026-10-09, retiring `blochparty.fused`: nine files had `fused =
  pytest.importorskip("blochparty.fused")` plus an autouse fixture at module level, so deleting
  the package would have turned their core checks into skips). Grep for the module name in
  `importorskip` and fixtures before deleting, and compare passed counts, not just failures.

- **`.double()` on a complex tensor drops the imaginary part** (with a warning that is easy to
  miss in a script): a coil-gradient comparison showed relative error 1.0 that was only the
  measurement. Compare complex tensors with `.to(other.dtype)` or `.to(torch.complex128)`.

- **`git stash` in the middle of a rename splits it** into a staged add and an unstaged delete
  (content intact, but confusing); do not stash while a refactor has renames in flight.

- **Wall time is not run time when the Mac sleeps, and `time.perf_counter` does not count the
  sleep** (2026-10-09, overnight with the user away): benchmark 1 took 55 min of wall clock for 251 s
  of simulation, the full suite 88 min while pytest reported 338 s. `pmset -g log` showed the
  machine in Maintenance Sleep with ~180 s DarkWake windows every ~15 min (closed lid or a manual
  sleep; the app's keep-awake prevents idle sleep only), and Python's monotonic clock
  (mach_absolute_time) stops during sleep, so in-script timings stay right while `date` and `time`
  do not. Swap (8.5 GB of 10 GB used, a 6.7 GB test process footprint) looked like the cause and
  was not. Before blaming an engine for a slow wall clock, compare it with `perf_counter` and read
  `pmset -g log | grep -E "Sleep|Wake"`.

- **A results table can predate a correction that the main table got**: benchmark 1's
  convergence rows were computed before the T2 fix (their best row is the old 1.44 %), so the
  16 x 1 row read 14.35 % where the current code gives 16.72 % on both engines. When a rerun
  disagrees with a recorded number, rerun the old code on the same inputs before suspecting the
  new engine.

- **`torch.as_tensor(python_float)` is float32** (the default dtype), not float64: it turned a 6.65 ms
  block duration into 6.6499998793 ms (off the raster) and an RF-spoiling receiver phase into a
  3e-8 error, both in the Pulseq writer. Convert Python numbers with `float()` or an explicit dtype.

- **zsh does not split an unquoted `$var` into words** (bash does): `set -- $job` in a loop over
  "a b c" strings gave one argument, and a chain of background jobs failed at once. Write the
  commands out, or use `${=var}`.

- **A wait loop must also end when its run dies** (2026-10-09): six `until grep ...; do sleep; done`
  loops outlived their runs by hours, because each waited only for success: a run stopped by hand,
  a `time` that prints "total" (zsh) instead of "real", a grep for a line that never follows the
  one it anchored on. Wait on the process (`while kill -0 PID`) or match failure and success alike,
  give the loop a deadline, and stop the loops when the result is in.

- **A presigned S3 link answers HEAD with 403** (2026-10-13): the fastMRI download link looked expired to `curl -I`;
  the signature covers GET only, and a ranged GET (`-r 0-1023`) returned 206. Test signed links with the method they
  were signed for.

- **Freeze the real set while runs are queued** (2026-10-13): score.py reads whatever the fastMRI manifest holds at
  scoring time, so extending the data mid-queue would have given consecutive runs different real sets. The extension
  streams into a second directory of symlinks plus new files (`~/data/fastmri_more` on the mini) and only the runs
  meant for the larger benchmark point `--fastmri` at it.

- **Queued jobs must not share a working directory** (2026-10-13): `quick.sh` swaps `prepared/` (rm, copy) to pick its
  head set, and two chained scripts did the same while r31 was reading heads from it: 50 of 57 heads failed with
  "No such file" and a quick run lost its own heads the same way. Each run names its own input directory
  (`simulate_heads --prepared`), and nothing copies into a directory another process reads.

- **A shared machine's disk is not ours** (2026-10-14): with the mini at 88 % full (42 GB of fastMRI archives, every
  run's raw k-space, a generated set), its owner removed `~/data` wholesale: the real scans, the caches, the runs, the
  181-sample set. The numbers survived in the journal; the data did not. Fetch only what a benchmark needs (the 3 T
  AXT2 subset, not every contrast), delete raw simulations once scored, prune old prepared-head versions, check
  `df -h` before anything that writes gigabytes, and keep whatever matters in the repo or on the laptop.

- **Fine-tuning a model in place renames nothing** (2026-10-16): `train(net_s, real_data)` returned the same object, so
  every later reference to the "synthetic" net (its in-domain score, the lesion test, the saved weights) was the
  fine-tuned one, and two rows of a table agreed to the digit without anyone noticing for two days. Copy before
  fine-tuning, and treat two identical rows as a bug, not a coincidence.
