# 2026-10-17 — Refactor: GPU device code as files, one driver for Metal and CUDA (plan phase 1)

- **Layout** (`461cf04` moves, `8ee7ac3` split, `d2ceac7` docs): `src/blochparty/engines/gpu/` holds the shared
  Python driver (`driver.py`: buffers, dispatch, NUFFT readouts, reused pulses; `adjoint.py`: the reverse sweep,
  `_MetalSimulate` renamed `_GPUSimulate`; `__init__.py`: the backend protocol, `source()`, `_simulate`), the
  device code both backends share (`shaders/common.metal`, `adjoint_common.metal`), and one folder per backend:
  `metal/` (`kernels.metal.in`, `pulse.metal.in`, `adjoint.metal.in`, the library build, `simulate_metal`) and
  `cuda/` (`prelude.cuh`, `common.cuh`, five kernel templates, `translate.py` with the text rules, `rtc.py` with
  NVRTC and the driver API, `simulate_cuda`). CUDA now reads `shaders/*.metal` instead of Metal's private strings.
- `blochparty.metal` and `blochparty.cuda` replace themselves in `sys.modules` by the new modules: one object
  under two names, so private helpers and monkeypatching behave as before. Only two tests changed: they patch
  driver constants, which now live in the driver.
- **Checks**:

  | check | result |
  |---|---|
  | Metal library text, CUDA programs of all 49 kernels | byte-identical before and after (560,911 and 3,317,243 bytes) |
  | Metal forward, laptop M4, 3d-64 | 2.84 s before, 2.84 s after |
  | Metal gradient, 3d-48 | 2.47 s before and after |
  | CUDA forward, RTX 4090 pod, 3d-96 | 3.04 s before and after |
  | CUDA gradient, 3d-64 | 1.56 s before and after |
  | CUDA tests on the pod (`-k cuda`) | 179 passed, 9 min |
  | full suite on the mini (Metal) | 839 passed, 179 skipped, 0 failed, 15 min |

  The cubin cache keys (a hash of the source) matched, so the pod's "after" runs loaded the cubins compiled
  "before". The wheel ships the device-code files (hatchling includes package data).
- **The pod**: no secure-cloud L40 was on offer, so a secure RTX 4090 ($0.89/h against $0.82; the same Ada sm_89
  as the L40, and the card the port was verified on) ran it: 25 min, $0.16, removed afterwards. GPU load:
  100 % and 375 W in the benchmark, idle most of the test run (the float64 references are CPU-bound).
- **Found on the way**: `test_realism_helpers` failed collection where `nibabel` is missing (the optional
  anatomy extra) and stopped the whole pod run; it skips now (`e2c894f`). Relative imports inside functions are
  invisible to ruff's undefined-name check: four `from ..maps import sample`-style imports pointed one level too
  shallow after the move and failed only when a gradient ran. Not tried: `xcrun metal -c shaders/common.metal`,
  as this machine has only the Command Line Tools, without the Metal compiler.
