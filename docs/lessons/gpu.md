# Lessons: GPU engines

- **Metal walks absolute positions in float32.** A walker 5 mm from isocentre is resolved to
  ~3e-10 m, and so is its offset from the compartment's centre: per-spin magnetization of 5 µm
  cylinders agreed with the core to 1.1e-3, the signal to 1.4e-5; spins at 10 cm with a
  gradient nonlinearity gave 7.4e-5 on the signal. Static spins agree to 2e-5. Walking in offsets
  from the start position would remove most of it. Source: `docs/ENGINES.md` Agreement, 2026-10-08.

- **Metal kernels take at most 31 buffers** (indices 0–30). Pack the scalars into one
  `constant uint*` array per dispatch rather than one `constant uint&` each.

- **Idle GPU threads alias spin 0: guard every write.** Threads past the last spin read spin 0's
  inputs (index clamped) and must not write. One unguarded store in the adjoint's replay let 255 idle
  threads overwrite spin 0's states with zeros: every gradient but λ and PD came out ~0, which looked
  like broken calculus. Source: M2, 2026-10-09.

- **A check that patches a Python function does not reach the GPU.** BMCTool's super-Lorentzian
  replica is swapped into `exchange` for one comparison; Metal reads its own lineshape table, so
  under `BLOCHPARTY_BACKEND=metal` that row shows the unpatched 1.2e-1. Source: M1a, 2026-10-08.

- **A long-running Metal process can start returning zero signals** (2026-10-14): a generator that simulated 60 heads
  in one process on the mini returned all-zero k-space for 41 of 181 slices from the 30th slice on, with normal
  timings and no error; the same heads regenerated in fresh processes were fine. One head per process, and never
  write a sample without checking the signal is finite and non-zero (`generate.py` now raises).

## The CUDA port — 2026-10-17

- **Port a shader by sharing its device code, not by copying it** (`src/blochparty/engines/gpu/cuda/`): Metal Shading Language is
  C++14 with address-space qualifiers, vector types and a few built-ins. Dropping `thread` / `device` /
  `threadgroup`, mapping `constant float*` to `const float*`, renaming `float3` & co. onto small structs with the
  operators the code uses, and `precise::sin` & co. onto `sinf` turned the whole device library (~1000 lines: steps,
  exchange, lineshapes, walks, field model, reverse sweep) into CUDA that compiled at the first try and agreed with
  the core at Metal's tolerances in all 68 matrix cells. Only the kernels were written anew, because that is where
  the two GPUs differ (simdgroup matrices, threadgroup memory, atomics). NVRTC's `-default-device` makes every
  unannotated function a device function, so the shared text needs no `__device__`.

- **Map scalar math to the float functions explicitly**: CUDA resolves `sqrt(x)` for a float through overloads that
  may land on the double version, and double runs at 1/64 rate on consumer GPUs (an RTX 4090). The port rewrites
  `sqrt`, `fabs`, `ceil`, `exp`, ... to their `f` forms in the shared text (`floor` stays overloaded: it is also
  called on a vector).

- **Check the C++ before renting a GPU**: no NVIDIA compiler runs on macOS, but the generated CUDA compiles as host
  C++ against a page of stubs (`__shfl_xor_sync`, `atomicAdd`, `__syncthreads`, `blockIdx`, `__shared__` as `static`):
  `clang++ -fsyntax-only` over all 49 kernel variants caught the type and overload errors locally; NVRTC then compiled
  each in ~1 s on the pod.

- **Runpod**: community-cloud RTX 4090s with a public IP (needed for plain SSH, rsync and `--wait`) were all taken
  on a Saturday midday; a secure-cloud 4090 ($0.89/h against $0.34) was there at once. `pod create` without
  `--ports 22/tcp` publishes no SSH port (`--wait` then times out after 10 min); `pod update --ports 22/tcp` fixes
  it. A secure 4090 for 45 minutes of verification and benchmarks cost $0.59. Reuse the image's PyTorch
  (`uv venv --system-site-packages`): the project ran on torch 2.8 there, nothing to download.

- **Know what a GPU test run measures**: the engine tests compare tiny cases (6-300 spins) with the float64 core on
  the CPU, so a 28-minute GPU test run kept the GPU at 0 % and 17 CPU cores busy (the user saw no GPU use and asked).
  Log `nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv -l 1` under `setsid` from the start, and show
  load with a case big enough to fill the GPU (3d-96: 100 % and ~387 W for 3.4 s per run; between runs the GPU waits
  4-5 s for the CPU to build the phantom and sequence).
