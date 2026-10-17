# 2026-10-17 — Refactor: one entry point, bp.simulate(..., engine=) (plan phase 3)

- **The entry point** (`07b8d5e`): `blochparty.simulate` is `engines.simulate`. `engine="core"` (default) calls
  `bloch.simulate` with the same arguments (core snapshot bit-identical); `"chunked"` is `simulate_chunked`;
  `"metal"` and `"cuda"` the GPU engines; `"gpu"` resolves to CUDA where an NVIDIA GPU is, else Metal
  (`engines.resolve`, `engines.available`). Another engine's options raise TypeError naming the engine they belong to
  (`readouts=` on the core: "an option of engine='gpu'"); `merge_free_runs=False` and a seed on the chunked engine
  raise. `tests/engines/test_entry.py` picks every argument so that dropping it would change the result, then
  checks each engine name against its function: a wiring test that cannot pass by accident.
- **The scripts** (`0011a8e`): the 21 example and validation scripts that called `simulate_metal` call
  `bp.simulate(..., engine="gpu")`; `BLOCHPARTY_BACKEND` takes an engine name; `--metal` is `--gpu` (alias kept);
  `nist_twin` frees memory on whichever GPU ran; `bench.py --backend gpu`.
- **Checks**: quick suite on the laptop 818 passed, 178 skipped. Each script on the mini through a wrapper of
  `bp.simulate` that logs every call's engine (10 min cap; a capped run counts once its GPU calls succeeded):

  | result | scripts |
  |---|---|
  | ran through, every call on Metal via `engine="gpu"` | design_despot1 (320 calls), optimize_contrast, gre3d_128 (141 s), anatomy_brain, anatomy_viewer, sampled_heads, anatomy_ahead, anatomy_brainweb, compare_bmctool (`--with bmctool`), em/brainweb_demo, incidental_mt (104), nist_irse, nist_tse (528 s), nist_twin check (44), radial simulate, magma (340, `--with pydicom`), spiral_tse, simulate_heads (one slice), lesion_design smoke (16 slabs, 34 calls) |
  | the converted function called directly | iohexol's `simulator()` (GPU against the core, < 1e-4: the prediction needs two 400 MB raw files the mini lacks), scholand's `signals()` (its driver found every result cached) |

  First failures that were not the change: `bmctool` and `pydicom` are outside the project's dependencies; the
  mini lacked radial's data file and magma's DICOM images (copied for the run, removed afterwards with the magma
  run's caches); `--starts` takes `flip:TE:TR` strings, not a count.
- **On CUDA** (secure RTX 4090 again: no L40 on offer; 20 min, $0.17, removed afterwards), where `engine="gpu"`
  resolves to CUDA: `test_entry` 5 passed (the Metal case skips); `bench.py --backend gpu` 3d-64 0.45 s; design_despot1
  (336 calls), optimize_contrast, gre3d_128 (16 s, against 141 s on the shared mini), the lesion smoke run (62 calls),
  incidental_mt (its first calls; stopped after 150 s, as its Python sequence building with tiny torch operations
  crawls on 32 threads), and the realism generator's one-slice sample (no manifest: the fallback protocol; an IXI
  head, 8.6 M spins, 8.9 s on the GPU, the HDF5 sample with its truth group written and finite). fastMRI-derived
  files stayed off the pod. GPU load: 100 % and 412 W at peak, idle while the CPU builds phantoms and sequences.
- **Full suite on the mini** (phases 3, 5 and 7 together, after all three were committed): 847 passed, 180 skipped,
  0 failed, 13 min.
