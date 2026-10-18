# 2026-10-18 — Lesion design: the frozen-design control (arm F), three seeds on an L40

- The open question of entry 2026-10-17-16: C's 400 joint steps also train G's net 400 more steps, so C minus G mixed
  the learned sequence with extra training. Arm F (2d15e46) runs C's steps from the same net on the same batches,
  noise and replay (one generator) with the sequence frozen at G's; with the design rates at 0, C reproduced F's
  losses and scores exactly (12 slabs, MPS), and on the pod F's and C's step-0 losses agree in every job. Three jobs
  side by side on one secure-cloud L40, `--seed 0, 1, 2`, `--mask-seed 0` (the same 32 lines as every 50 %-Gaussian
  run), arms G, F, C, everything else as before (SNR 20, RSS front, 100 epochs, 400 steps of 8 slabs, replay 4).
- Test AUC, 129 slabs (G0: the G net that F and C start from):

  | seed | G (mean of three) | G0 | F: frozen | C: learned | C - F | F - G0 |
  |---|---|---|---|---|---|---|
  | 0 | 0.734 | 0.721 | 0.734 | 0.817 | +0.083 | +0.013 |
  | 1 | 0.731 | 0.722 | 0.725 | 0.730 | +0.005 | +0.003 |
  | 2 | 0.736 | 0.759 | 0.776 | 0.806 | +0.030 | +0.017 |
  | mean | 0.734 | | 0.745 | 0.784 | **+0.039** | +0.011 |

  C - F: sd 0.040, t(2) = 1.70, p = 0.23. Accuracy at the training-set threshold: C 0.78 / 0.72 / 0.71, F 0.73 / 0.74 /
  0.75; train AUC C 0.86 / 0.82 / 0.88, F 0.83 / 0.78 / 0.86.
- **The extra training is not the gain**: F adds +0.011 on average to its starting net, about a fifth of C's +0.050.
  What is left, +0.039, is the sequence's share; it is positive in all three seeds but ranges +0.005 to +0.083, so
  three paired seeds make it suggestive, not established. C's test accuracy is not better than F's (two of three
  lower): the gain is in the ranking, not at the threshold chosen on the training set.
- **The sequence is the same every time** (now five C runs): flips 16.1 deg on average (spreads 12.8-19.0, 14.3-18.0,
  14.6-17.8), TE 60.1-61.2 ms, TR 78.9-79.9 ms from 70, the lines nearly where they started (std 13.1 cycles). The
  variable part is how much a given network gains from it.
- **Same seed, different result**: seed 1 repeats entry 2026-10-17-16's job on another L40 host (10 CPU threads
  instead of 15, three jobs sharing the GPU). The G start net's train AUC was 0.768 against 0.786, G's third net 0.746
  against 0.709, and C 0.730 against 0.758. Training on CUDA here is not bit-reproducible (lessons/design.md), so the
  run-to-run spread includes that; only pairs inside one job (C and F from one net, one stream) are clean. Five C runs
  so far: 0.803, 0.758, 0.817, 0.730, 0.806 (mean 0.783); G's means 0.720-0.736.
- Next, a cleaner test of the sequence alone: freeze C's final sequence and train fresh nets on it with G's recipe
  (several seeds, ~70 s each on the L40), against G's fresh nets on the start sequence; no fine-tuning path in it.
- Cost: the jobs took 20 min, but the pod ran 2 h 7 min ($1.75; $7.46 -> $5.71). The laptop's lid closed six
  minutes after the pod came up (clamshell sleep on battery), so the launcher and the watcher ran only in its
  2-second dark wakes about every 17 minutes: the jobs started 33 min late, the copy back broke mid-transfer, and the
  pod went at the 01:46 wake. The three `results.json` and `arm_C.pt` were lost with it; the logs (every arm's final
  numbers and C's design summary) came back and are parsed into `~/data/design8_ctrl/summary.json` on the mini,
  next to the logs (lessons/gpu.md). GPU 22 % mean, 100 % peak, 281 W.
- Commits: 2d15e46 (arm F), this entry and its lessons.
