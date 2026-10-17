# 2026-10-17 — Refactor: the training-data generator into the package (plan phase 4)

- **Reconstruction and metrics** (`577d2b4`): the lesion and transfer experiments' helpers into the package,
  verbatim: `recon.fft2c` / `ifft2c_ortho`, `uniform_mask`, `variable_density_mask`, `random_mask`, `cg_sense`;
  `design.auc` and `design.bootstrap`. On random inputs the old copies and the package give identical masks, SENSE
  images, AUCs, intervals and undersampled k-space. New tests check properties, not formulas: the pair is unitary,
  `<Ax, y> = <x, A^H y>` for the SENSE encoding, CG-SENSE unfolds R = 2 to under 1 % (ten times closer than zero
  filling), the masks take the lines they promise, the AUC equals the rank formula.
- **The receive model** (`ab3eeb0`): `coils.coil_offset` and `coils.normalised_sensitivities`; a new test: a
  profile spanning 1.8x over the object's centre comes out flat to 15 %.
- **`blochparty.generate`** (`a568e4c`): `spec` (DEFAULT_SPEC, `merged()` now refusing unknown sections and keys,
  `default_spec`, `span`), `protocols` (the paired real protocols and their TSE), `heads` (the preset's head per
  seed, its clean-up, the stack, `b1_plus` now given the anatomy root), `sample` (`samples()`, an iterator of
  per-slice samples, and `write()`, the HDF5 part of the old `generate()`). `validation/realism/generate.py` is its
  command line with the real scans' floors and covariances; `simulate_heads.py` keeps its ablation options and
  imports the moved code. Departures from the plan: `simulate_heads.main` stays the ablation driver rather than
  becoming thin (its options are experiments the spec does not cover), `b1_plus` (a head-specific full-wave map) went
  to `generate/heads.py` rather than `transmit.py`, and `sample()` is `samples()`, an iterator (one head gives
  several slices).
- **Sample identity** on the mini (one IXI v17 head, seed 3, one 64 x 64 slice; with the fallback protocol and with
  the real fastMRI protocols):

  | dataset | before vs after | before vs before (two runs of the old code) |
  |---|---|---|
  | truth: fractions, lesion, t2/pd scale | identical | identical |
  | truth: kspace, kspace_clean | rel 1.0e-7 | rel 5e-8 to 1e-7 |
  | truth: image_clean | rel 4e-8 to 9e-8 | rel 9e-8 |
  | raw data (ISMRMRD) | rel 7e-8 | rel 7e-8 |
  | spec.json, sequence, phantom provenance | identical | identical |

  The GPU's coil sums use atomics, so the k-space is never bit-reproducible; after equals before at that level.
  Not done: the plan's transfer-test rerun on an existing set: none is left on the mini (the sets went with
  `~/data`), and its helpers were shown identical instead.
- **Caught by the full suite, not by my subsets**: `test_meta`'s source scan read every `meta = {...}` under `src/`,
  so the generator's sample metadata failed it (`71b2bc5` scopes it to the sequence code). Full suite on the mini
  after phase 4: 857 passed, 1 failed (that one), 180 skipped.
- **Full suite on the mini** (phases 4, 8 and 10 together, with the test fix): 859 passed, 180 skipped, 0 failed, 13 min.
