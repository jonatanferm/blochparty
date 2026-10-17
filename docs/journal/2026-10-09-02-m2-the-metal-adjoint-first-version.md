# 2026-10-09 — M2: the Metal adjoint (first version)

- **What.** `simulate_metal` differentiable through a hand-written adjoint on the GPU: the reverse
  sweep of `fused/kernel.h` ported lane by lane (plain Bloch, both concomitant-field modes, the
  rate-model bound pool with its exchange-step reverse and lineshape derivatives), in a
  `torch.autograd.Function`. Scope as planned; moving spins, free pools and the generalized bound
  pool raise. The forward's two-sum low parts now persist across dispatches, which makes the
  forward bit-identical for any chunk size and the checkpoints exact.
- **Accuracy.** Against the core's float64 autograd, per input: 1e-7 to 3e-6 on random sequences,
  2e-5 on a 1600-segment RF-spoiled GRE with 4 coils, ≤ 1.2e-4 with the bound pool, where the fused
  kernel's float32 adjoint is up to 1.5e-3 off (the compensated forward helps). Bit-identical for
  any `checkpoint_every` (64, 128, 640) and run to run. Gradient descent: WM/GM optimum 16.110° vs
  16.109° analytic (0.8 s), Ernst angle to 0.01°. DESPOT1 pair only to 0.05°: its CRB is a
  finite-difference Jacobian of float32 signals (needs `rel_step` 1e-2; at 1e-4 it landed at
  (3.7°, 38°) instead of (4.1°, 23.6°)).
- **Speed (3d-48, same session).** First version: forward 0.49 s, backward 4.2 s. Separate replay
  kernel: small gain. Every 4th state stored, blocks of 4 recomputed into threadgroup memory: the
  replay's DRAM traffic /4 (2.65 → 2.2 s). Sensitivities in registers: slower (occupancy); read from
  memory, coil gradients in registers only when asked for. Per-simdgroup partials instead of a
  barrier per segment. One parameter upload per pass. Now forward + backward 3.3 s on 3d-48 (6.2x
  the forward), 22 s on 3d-64 (~7x), 3-4x on 2d-96 / 3d-32; the 5x target is not met. The machine
  was thermally throttled by then (the 3d-64 forward at 3.4-3.6 s instead of 2.4 s, the old and new
  kernels alike), so only same-session ratios are meaningful.
- **Dead ends.** All gradients but λ and PD came out ~0: idle threads (past the last spin, index
  clamped to 0) overwrote spin 0's replay states. The matrix test's Metal gradient cells all failed
  at first: its loss drew random weights in the output's dtype (float32 vs float64 → different
  numbers). Transcendental caching in the reverse made no measurable difference.
- **Open.** Speed to 5x (next candidates: the ADC term's per-sample coil loop, register pressure of
  the bound-pool reverse, a fused replay+sweep for small intervals). Retiring the fused kernel: it is
  still the only fast float64 path with gradients, which DESPOT1-style designs (finite differences
  inside the objective) need; the core does it in float64, slowly.
- 797 tests (4 min 43 s on the throttled machine); snapshot bit-identical.
- Commits: 2519392 (adjoint), 1ceea1d (bench, examples), then these docs.
