# 2026-10-17 — Refactor: engines/ with one preparation step and one capability table (plan phase 6)

- **Moves** (`4dc3c6c`, `f918ab5`): `bloch.py` is `engines/core.py`, `chunked.py` is `engines/chunked.py`,
  `SimResult` is in `result.py`; `blochparty.bloch` and `blochparty.chunked` forward to the new modules (tests
  and scripts import them and patch the core's private constants).
- **The snapshot first** (`597c116`): it covered the basic GRE, a 3D case and the random ODE cases, none of what
  this phase touches. It now also holds the core's outputs and gradients on the 34 cases of the engine matrix
  (maps, bound and free pools, diffusion, motion, the field model, m_init, ...): 7 s, bit for bit run to run.
- **`engines/prepare.py`** (`ef50840`): the input checks every engine made on its own (coils from both `coils` and
  `maps`; an `m_init` bound-pool column without a bound pool; restricted diffusion with a tensor) and what they
  derive alike, in the phantom's dtype: `spin_fields`, the moving flag, the moving-spin field model, where walkers
  start, the expanded bound pool, the starting magnetization. The core uses it with its own expressions moved,
  not rewritten. One behaviour change: a 4-column `m_init` without a bound pool is refused on the free-pool path
  too, where both engines silently dropped the fourth column.
- **`engines/capabilities.py`**: ten features, what each engine runs forward and with gradients (`SUPPORTED`), the
  combinations it refuses although it runs each part (`REFUSED`: free pools with generalized Bloch and bound-only
  RF with either on the GPU; bound-only RF with generalized Bloch in the core), and `check()` with one wording
  ("cuda engine: gradients through motion (Phantom.motion) is not implemented; use ..."). It replaces eleven
  `NotImplementedError` sites in the core, the GPU forward, the adjoint and the chunked engine (the GPU's five-pool
  size limit stays in the driver). A new test checks the declaration against every cell of `docs/ENGINES.md` for
  all four engines without running them, so the CUDA column is checked on every machine.
- **The GPU side** (`8b0b984`): forward and adjoint take their checks and shared inputs from `prepare` (the float64
  copies the forward makes are the same values as before); the forward keeps its own float64 sampling of static
  maps and its diffusion factors, so a float32 phantom loses nothing.
- **Checks**:

  | check | result |
  |---|---|
  | widened snapshot (core, 34 matrix cases with gradients) | bit-identical after every step |
  | Metal library and the 49 CUDA programs | byte-identical |
  | CPU core, mini, 3d-32 | 1.05 s before, 1.05 s after |
  | Metal, mini, 3d-64 | 2.07 s before and after |
  | CUDA, L40, 3d-96 | 3.43-3.46 s before, 3.45-3.48 s after (run-to-run noise) |
  | CUDA tests on the L40 (`-k cuda`) | 181 passed |
  | full suite on the mini | 849 passed, 180 skipped, 0 failed, 13 min |

- **The L40 pod** cost $1.22 instead of ~$0.40: torch started 128 threads on a container allowed 27 CPUs (`nproc`
  says 256), so the float64 references crawled until the run was restarted with `OMP_NUM_THREADS=27`; and the pod
  then idled an hour after the tests, because its removal waited on this laptop, which had gone to sleep. Both are
  lessons in `docs/lessons/gpu.md`. Utilization: 100 % and 296 W at peak, idle through the CPU-bound references.
