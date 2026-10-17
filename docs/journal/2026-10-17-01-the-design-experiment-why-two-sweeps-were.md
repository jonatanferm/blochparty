# 2026-10-17 — The design experiment: why two sweeps were at chance, and the third

- **The second sweep (conspicuous lesions, 372 slabs, 40 epochs, 300 steps)** was at chance too: f = 1.0 A 0.668
  (accuracy 0.42), B 0.556, C 0.466; f = 0.2 A 0.507, B 0.463, C 0.528; f = 0.1 0.44-0.51; f = 0.05 0.48-0.51; C's
  loss wandered 0.45-0.75 and its line positions drifted without a direction. Three diagnostics on the mini, in the
  order run:
  1. *Noise*: `--snr 20` as defined (sigma = mean |sample| of the reference GRE / 20) gave an image SNR of ~1200 in
     the brain (k-space noise std 0.36 against a brain median of 6.5 and a pixel noise std of 0.006); the lesion
     excess was 300-800 noise units. Both sweeps were effectively noiseless; the noise was not the cause.
  2. *The training path*: the image CNN overfits the six local smoke slabs through arm A's pipeline (loss 0.004 in
     300 steps), so inputs and labels agree.
  3. *Learnability* (`design_diag.py`, 276 slabs after the 0 < lesion px < 4 ones are dropped, full sampling,
     60-100 epochs): the lesions are 1.62 x the brain median (10-90 %: 1.32-1.93), peak 2.08 x, but the peak is
     below the brain's 90th percentile in 16 % of the positives (sulcal CSF), so the task is context. Arm A's net as
     run (mean pooling, no augmentation) reached train AUC 0.98 / test 0.62: it memorised the heads. Max-pooling
     over space alone 0.59, with flip + shift augmentation 0.58 (underfits at 16-32-64 channels), with the
     ambiguous slabs dropped 0.70 (two seeds 0.69-0.70), and 32-64-128 channels 0.79-0.88 at 60 epochs, 0.88-0.91 at
     100. The raw path (`design_diag_raw.py`): a CNN on the raw multi-coil samples 0.58 (train loss 0.30: it
     overfits); the same net behind the encoding's adjoint (a per-coil zero-filled inverse DFT, 1e-6 from `ifft2c`
     on the grid, differentiable in continuous line positions) 0.89 as a root-sum-of-squares image and 0.84 as 16
     real/imaginary coil channels; the full SENSE image 0.88.
- **And a bug the diagnostics exposed**: the line mask was applied to the last k-space axis, which is the readout
  (k-space is (coil, ky, kx); checked by blurring one axis' periphery and by the head's A-P extent along the rows). "20 %
  of the lines" had dropped 80 % of the readout samples in arms A and B (no scan-time meaning, SENSE unfolding along x).
  Fixed in all three places (the mask, CG-SENSE's operator, the raw lines). With phase-encode undersampling the picture
  is stark (276 slabs, noiseless, 100 epochs): f = 1.0 SENSE 0.88 / adjoint-RSS 0.89 / adjoint-coils 0.84 / raw 0.58;
  f = 0.2 (13 lines) 0.62 / 0.56 / 0.54 / 0.54 with the training loss stuck at 0.69; f = 0.1 all 0.55-0.58. Thirteen
  uniformly spaced lines through an 8-element ring is R = 5 in one direction: SENSE cannot unfold it, and neither net
  learns anything from it in 276 slabs. The regime where the baseline degrades rather than dies is 100 -> 20 %.
- **Changes (c0bfaf1)**: phase-encode masks; `--snr` as the reference GRE's image SNR in the head (sigma = N x median
  head pixel / snr); the CNN max-pools over space, 32-64-128 channels, flip + shift augmentation, 100 epochs, cosine
  learning rate, three seeds for the fixed arms, accuracy at the train-best threshold; ambiguous slabs left out
  (`--min-lesion-px 4`: 100 of 744); the raw arms read the encoding's adjoint (`--front rss|coils|raw`); a fixed-GRE
  arm G (25 deg, TE 6, TR 15 ms, K lines evenly spread) as the ablation of what the design adds; arm C starts from G's
  net and trains jointly for 400 simulated steps with a replay buffer (4 extra network updates per step on the last
  256 acquisitions); TR = TE + the readout tail + an extra, so every design is feasible (the old independent (8, 120)
  ms range allowed TR < TE). `lesion_slabs.py --draws 3` added two more lesion draws per head and level
  (incremental, the index merged): 744 slabs, 449 with a lesion, 6.9 GB; 515 train / 129 test after the drop.
- **The third sweep** (`design3.sh rss 0.5 0.3 0.2 1.0`, started 00:14): fractions 0.5, 0.3, 0.2 and 1.0, in that order,
  front `rss`, SNR 20 (the TSE's image SNR is ~2.7 x that), 100 epochs x 3 seeds for A, B, G, 400 steps for C. Each
  fraction is 1-2 h on the mini (the TSE acquisition of a slab is 0.45 s; a joint step with 8 slabs at 64 lines 10 s).
- **50 % of k-space, the cold start (design3_rss_f0.5, 515 train / 129 test slabs, SNR 20):** A (TSE + CG-SENSE + CNN)
  AUC 0.81 (seeds 0.73 / 0.83 / 0.87; accuracy 0.76), B (the same 32 TSE lines through the adjoint) 0.72 (0.68-0.75),
  G (the generic GRE, 25 deg / TE 6 / TR 15 ms, 32 lines evenly spread) 0.50 on all three seeds, C from that start
  0.48 after 400 joint steps (train AUC 0.52): TE 5.97 -> 6.14 ms, TR 15.0 -> 17.4 ms, the flips spread to 16-32 deg,
  the line positions unchanged (std 18.7 cycles throughout). The reason is physics, not the optimiser's step: at 25 deg
  / TR 15 ms the lesion (T1 1.36 s, PD 0.85) is 16 % darker than white matter, the same as grey matter, so the net
  cannot learn it, and a chance-level net gives the design parameters a gradient that is noise (the per-line flips
  random-walk, TE does not go anywhere). The TSE's 15 s against the GRE's 0.6-0.7 s is the price difference the
  arms do not yet pay for. Changed (0421b16): `--starts` evaluates six generic GRE contrasts (25/6/15, 10/6/15,
  25/20/30, 15/30/40, 15/45/55, 15/60/70 as flip deg / TE ms / TR ms) with one seed each, the best training AUC is
  arm G (refitted with three seeds) and arm C's start — a coarse search before the gradient, standard for a non-convex
  design; TE's range widened to 80 ms. The achievable GRE contrast for this lesion is T2*: at TE 30 ms lesion / WM
  ~1.3 (25 deg) to 1.5 (8 deg), at TE 60 ms ~2.3, against the TSE's 1.6 — so the design arm has something to find.
  Relaunched at 01:26 as `design4.sh design4 rss 0.3 0.2 1.0 0.5` (the cold-start 50 % results kept).
- **30 % of k-space, multi-start (design4_rss_f0.3, 19 lines):** A 0.62 (0.58 / 0.62 / 0.67; accuracy 0.71 = the
  negative class), B 0.60 (0.59-0.61). The six GRE starts, one seed each (test / train AUC): 25/6/15 0.52 / 0.54,
  10/6/15 0.52 / 0.56, 25/20/30 0.52 / 0.57, 15/30/40 0.52 / 0.55, 15/45/55 0.50 / 0.61, 15/60/70 0.54 / 0.62; the best
  on the training set (TE 60 ms) refitted with three seeds as G, and C from it: after 400 joint steps TE 60 -> 64.7 ms,
  TR 70 -> 78 ms, flips 15 -> 16.4 deg (11.7-22.5), the lines' spread 18.7 -> 19.5 cycles; test AUC 0.53 (train 0.54).
  Nineteen lines, uniform (R = 3.4 in one direction), is already past what any reader here can use: SENSE with exact
  maps at 0.62, the zero-filled adjoint of any GRE contrast at chance. The 20 % run was moved to the end of the queue
  (100 % and 50 % first, then 20 %, then arms B and G with the 16 coil channels at 50 / 30 %: `reorder.sh`).
- **The multi-start sweep's table (design4_rss_f*, 515 train / 129 test slabs, 92 test positives, SNR 20, test AUC,
  three seeds for A, B, G; G = the best of the six GRE starts on the training set, which was 15 deg / TE 60 / TR 70 ms
  at every fraction):**

  | fraction (lines) | A: TSE + CG-SENSE + CNN | B: TSE lines + adjoint + CNN | G: GRE TE 60 + adjoint + CNN | C: joint from G | scan A / G |
  |---|---|---|---|---|---|
  | 100 % (64) | 0.86 (0.84-0.88) | 0.80 (0.74-0.85) | **0.95** (0.94-0.96) | 0.66 | 30 s / 5.0 s |
  | 50 % (32) | 0.81 (0.73-0.87) | 0.72 (0.68-0.75) | 0.61 (0.57-0.66) | 0.52 | 15 s / 2.8 s |
  | 30 % (19) | 0.62 (0.58-0.67) | 0.60 (0.59-0.61) | 0.54 (0.53-0.55) | 0.53 | 8.9 s / 1.9 s |
  | 20 % (13) | 0.61 (0.58-0.63) | 0.57 (0.56-0.59) | 0.52 (0.52-0.53) | 0.49 | 6.1 s / 1.5 s |

  The six starts at 100 % (test / train AUC): 25/6/15 0.49 / 0.59, 10/6/15 0.51 / 0.54, 25/20/30 0.52 / 0.55, 15/30/40
  0.86 / 0.92, 15/45/55 0.95 / 0.96, 15/60/70 0.95 / 0.98 — the lesion is a T2* lesion for a GRE (its T2* is 0.8 x its
  T2 here, 160 ms against white matter's ~50), invisible below TE 20 ms and conspicuous from TE 45 ms. With the 16
  coil channels instead of the root-sum-of-squares (one seed): B 0.58 / G 0.55 at 50 %, 0.60 / 0.55 at 30 % — worse.
  Three results. (1) *At full sampling the raw path wins*: a 5 s spoiled GRE at TE 60 ms, read by a CNN from the
  zero-filled adjoint of its multi-coil samples with no reconstruction, detects the lesion at 0.95 where the 30 s
  clinical TSE through SENSE reads 0.86 — but the choice that did it was the contrast, found by a six-point search,
  not by the gradient. (2) *Undersampling hurts the raw arms more than SENSE*: evenly spread lines at R = 2 fold the
  head onto itself in the zero-filled image, the root-sum-of-squares discards the coil phase that could unfold it,
  and the CNN does not learn to unfold from 515 slabs (nor from the 16 coil channels), while SENSE with the exact
  maps holds 0.81 at 50 %; below 30 % nothing reads 64 x 64 through 8 coils, SENSE included. (3) *The joint phase
  did not help anywhere*: from G's converged net (0.95 at 100 %) it ended at 0.66 with the design nearly where it
  started (TE 60 -> 58.5 ms, flips 15 -> 16.6, lines unchanged, the per-step loss 0.3-1.1) — the net's fine-tuning
  (Adam 3e-4 from scratch, batch-norm statistics from batches of 8) wrecked it; at 50 / 30 / 20 % the start was
  already at chance. Fixed: the joint phase keeps batch-norm and dropout fixed and moves the net at 3e-5
  (`--net-lr`), the loss is logged as a 20-batch mean; rerunning C at 100 % and 50 % (`queue3.sh` -> design5_rss_f*).
- **Why the TE 60 ms GRE reads the lesion better than the TSE (closed-form check against the simulated slab):** with the
  table's 3 T values (WM T1 0.89 s / T2* 48 ms / PD 0.70; GM 1.45 / 58 / 0.83; CSF 4.28 / 204 / 1.0) and the task's
  lesion (T1 1.64 s, T2 200 ms, T2* 160 ms assumed as 0.8 x T2, PD 0.85), the spoiled-GRE signal equation at 15 deg /
  TE 60 / TR 70 ms gives, relative to WM: GM 1.25, CSF 1.74, lesion 2.35 — the lesion is the brightest tissue in the
  brain, because the short TR saturates CSF (T1 4.3 s) while TE 60 ms keeps the lesion's long T2*. The spin-echo
  equation at TE 113 / TR 6000 ms gives GM 1.44, lesion 3.24, CSF 4.9: the lesion sits far below the CSF it has to be
  told from. The simulated smoke slab agrees: lesion peak / brain 90th percentile 1.31 (GRE) against 0.94 (TSE), lesion
  mean / brain median 1.49 against 1.21 (partial volume at 3.4 mm pulls both down from the closed form). So the
  machinery found the contrast that fits *this* lesion's assumed T2*; whether a real lesion is T2*-long is a question
  for the literature (recorded as assumed in lesion_slabs), not a claim of this experiment.
- **The joint phase, second fix (4baab75):** with the net held gently (lr 3e-5, batch-norm fixed) the joint loss at
  100 % still went 0.19 -> 0.69 (20-batch mean) within 20 steps. Adam moves every parameter by ~lr per step whatever
  the gradient's signal-to-noise, and a central line moves 32 cycles per unit of its tanh-raw parameter, so at
  design lr 1e-2 the central lines wandered ~0.3 cycles a step under a net trained on the grid (|ky| < 4: 8 -> 10
  lines in 40 steps). The line positions now have their own rate (`--ky-lr 1e-3`) and the flips / TE / TR 3e-3:
  the joint loss holds at 0.189 / 0.193 / 0.210 at steps 0 / 20 / 40, with the design moving slowly (flips 15.0 ->
  15.2 deg, spread 14.7-15.6; TE 60.0 ms; TR 70.0 -> 70.5 ms). The broken run's log is kept as
  `design5_rss_f1.0_designlr1e-2.log`. Queued after it on the mini: arm C at 50 %, then every arm with
  variable-density lines (`--sampling gauss`) at 50 % and 30 % (`design6_gauss_f*`).
- **Arm C at 100 % with the separate design rates (design5_rss_f1.0):** the joint phase now holds G's network: test AUC
  0.950 (accuracy 0.87, train AUC 0.986) against G's 0.950 (three seeds 0.943-0.960). The 20-batch loss stayed at
  0.17-0.22 over 400 steps; the design moved little and slowly: flips 15.0 -> 16.5 deg (spread 14.8-19.8), TE 60.0 ->
  60.8 ms, TR 70.0 -> 74.0 ms, the line positions unchanged (std 18.6 cycles, the same 8 lines within 4 cycles of the
  centre). The joint gradient no longer harms anything, and it finds nothing to gain around this start in 400 steps
  of 8 slabs: the contrast that matters was chosen by the six-point search. Still queued on the mini: C at 50 %,
  then every arm with variable-density lines at 50 % and 30 % (`design6_gauss_f*`).
