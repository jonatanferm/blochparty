# 2026-10-17 — The CUDA port (ROADMAP M8): `blochparty.cuda.simulate_cuda`

- **Approach**: one physics source. The Metal shader's device code (`metal._SOURCE` and `_ADJOINT_COMMON`, ~1000
  lines: segment steps with two-sum low parts, the exchange step, lineshapes, the Bloch-McConnell Taylor series,
  generalized-Bloch modes, Philox walks, restriction, maps, the field model, the reverse sweep) is turned into CUDA
  C++ by text rules in `cuda._shared_source()`. The rules drop `thread` / `device` / `threadgroup`, map `constant
  float*` to `const float*`, rename `float3`, `int3`, `uint2`, `uint4`, `float2` onto `_PRELUDE`'s structs with the
  operators used, and map `precise::sin` and the scalar math to the `f` functions. NVRTC's `-default-device` needs no
  `__device__` annotations. Written for CUDA: the kernels (`bloch_chunk[_mma]`, `bloch_replay`, `bloch_adjoint_cN`,
  `pulse_maps`, `seg_reduce`, same names and buffer lists as Metal). The coil sums use a warp reduce-scatter
  (each lane forms 16 products for 8 coils; four butterfly steps and a final pair add, 16 shuffles, leave each
  coil's sum in one lane), in full float32 rather than TF32 tensor cores. The Python driver is Metal's with a backend
  object (`metal._Metal` / `cuda._Cuda`: device, kernels, synchronize, coil cap). NVRTC (ctypes) compiles one kernel
  per program on first use, ~1.1 s each (102 registers, 64 B local memory for `bloch_chunk_mma`), cached on disk.
- **Before the GPU**: all 49 kernel variants compiled as host C++ against a page of CUDA stubs (`clang++
  -fsyntax-only`), which caught nothing because the transformed library type-checked as written; the converted
  Metal tests (a `gpu` fixture over the available engines) and the refactored driver passed on the laptop's Metal
  (78 tests, 605 s), so Metal was unchanged before any CUDA ran.
- **Runpod**: no community-cloud RTX 4090 with a public IP was free (CUDA 12.8, 12.4 or 12.1 images); a secure-cloud
  4090 at $0.89/h ran (driver 595, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, torch 2.8.0+cu128,
  NVRTC 12.8, reused through `uv venv --system-site-packages`). SSH needed `pod update --ports 22/tcp`. About 45 minutes
  in all, $0.59; the pod was removed afterwards (balance $9.41).
- **Verification on the 4090** (everything passed on the first compile; no kernel was changed on the pod):
  - first case, a 16x16 GRE with 8 coils against the float64 core: signal 7.2e-7 and magnetization 2.4e-6, for both
    kernels and for 1 coil;
  - `tests/test_engines.py`, the new cuda column: all 68 cells (34 features x forward and gradients, the refusals
    included) in 52 s;
  - `test_metal.py`, `test_metal_grad.py`, `test_metal_nufft.py`, `test_metal_reuse.py` on CUDA: 78 passed in 28 min,
    CPU-bound by the float64 references (17 cores busy, GPU at 0 %, which the user noticed and asked about);
  - worst over the matrix vs the core: static spins 1.6e-5 (leading-order concomitant fields), walking signal 7.7e-5,
    walking per-spin states 1.1e-3 (restricted diffusion), adjoint 1.3e-4 (`rf_freq`), 5.3e-5 with the bound pool,
    ≤ 1.4e-5 elsewhere. Metal's recorded worst cases: 1.6e-5, 7.4e-5, 1.1e-3, 1.2e-4;
  - against Metal directly on eight cases (RF-spoiled GRE, TSE with the bound pool, the same with NUFFT readouts and
    reused pulses, diffusion, free pools, motion, exact concomitant fields, 40 coils): signals 5e-7 to 1.7e-5,
    magnetization up to 1.2e-4 (the 1300-segment RF-spoiled train).
- **Speed** (forward, fastest of 3; utilization logged per second with `nvidia-smi`):

  | case | spins | segments | RTX 4090 | spin-segments/s | Metal, M4 (2026-10-08) |
  |---|---|---|---|---|---|
  | 2d-96 | 4603 | 10000 | 0.013 s | 3.6e9 | 0.016 s |
  | 3d-32 | 8803 | 37264 | 0.045 s | 7.4e9 | 0.057 s |
  | 3d-48 | 29831 | 120208 | 0.14 s | 2.6e10 | 0.45 s |
  | 3d-64 | 70505 | 278928 | 0.51 s | 3.9e10 | 2.40 s |
  | 3d-96 | 238187 | 922000 | 3.38 s | 6.5e10 | not measured |

  Small cases are launch-bound on both machines. The 3d-96 runs hold the GPU at 100 % and ~387 W of 450 W; between
  runs the GPU waits 4-5 s for the CPU to build the phantom and sequence. Forward + backward: 3d-32 0.32 s, 3d-48 0.75 s,
  3d-64 2.23 s (4.4x its forward, as on Metal).
- **Commits**: b8afd98 (the driver's backend), b5cb509 (the port, the cuda column, the GPU fixture, bench `--backend
  cuda`, README, ENGINES).
