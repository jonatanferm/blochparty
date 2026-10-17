# 2026-10-14 — M7 started: the lesion model, the generation driver, the transfer test

- **Lesions** (25e1a04): `anatomy.add_lesions` places ellipsoidal, soft-edged white-matter lesions by region
  (periventricular within 10 mm of the lateral ventricles, deep, juxtacortical within 3 mm of cortex, or anywhere in
  white matter), the major axis log-uniform between the size bounds, orientation uniform, centres at least a spacing
  apart; they replace white matter only, every voxel's total conserved, each recorded (centre, semi-axes, rotation,
  volume). `sample_head(lesions=...)` applies them in MNI space before the deformation from their own stream (the other
  classes' draws unchanged, tested), `simulate_heads --lesions`, and the generator's spec draws the count per head.
  The class `lesion` has West 2014's 1.5 T values (QRAPMASTER: T1 917 +- 93, T2 125 +- 14 ms, PD 0.82 +- 0.04;
  controls' WM 606 / 78 ms / 0.66) in the database; at 3 T the table's white matter scaled by the measured ratios
  (T1 x1.51, T2 x1.61; assumed, flagged), bound pool 0.6 x WM (F reduced in lesions in every qMT study, York 2022;
  MPF 7.2 % in lesions vs ~12 % WM, 2024) and diffusivity 1.2e-9 (plaque MD 1.23-1.24e-3 mm2/s), both assumed with
  sources. Literature with 3 T lesion T1 / T2 tables (Hagiwara 2017, Bonnier 2014 for lesions) was behind paywalls or
  reCAPTCHA; West's open-access table is the measured anchor. Core snapshot bit-identical; 116 tests.
- **The driver** (`validation/realism/generate.py`): a spec (anatomy with lesions, protocol family from the real
  headers, scanner with the receive drawn per head and the paired scan's coloured noise, quality) and a seed give one
  head's slices as ISMRMRD files via `write_dataset` (provenance: the phantom's draws, git commit) with a `/truth`
  group: clean and noisy k-space, the clean image, the tissue fractions and residual maps on the acquisition grid, the
  lesion mask, every draw as JSON; a JSONL manifest. The truth grid's orientation was checked numerically against the
  simulated image (a T2w proxy of the fractions correlates 0.74 in the display orientation, 0.68 flipped; the column
  direction was wrong at first). The Pulseq export inside `write_dataset` fails on the multi-slice train's
  bound-pool-only RF segments and is recorded as `seq_error` (the sequence itself is stored). Smoke test on the laptop:
  32 x 32, one slice, two lesions, 21 s.
- **The transfer test** (`validation/realism/transfer.py`): 4 x Cartesian undersampling with an 8 % centre at the
  realism resolution; a 0.5 M-parameter U-Net from the zero-filled RSS to the fully sampled RSS; four numbers on
  held-out real people: zero-filling, trained on synthetic only, trained on real (other people), synthetic then real.
- On the mini, first in line (r34 paused at 9 heads and resumed after it: the transfer number matters more than the
  realism confirmation): 100 v16 heads x 3 drawn slices (lesions 0-4 per head, 2 x 3 x 3 spins, the 34 real
  protocols) as `~/data/gen_t2w` (~7 h), then `transfer_1`; the v17 preparation continues on the CPU beside it.
- **First samples and a dry run**: the generator writes a sample every ~45-110 s per slice on the mini (8-9 M spins at
  2 x 3 x 3), heads loading in between; the first run ended after 16 samples with no traceback (a hard exit while the
  v17 preparation ran beside it on the CPU), so the relaunch runs serial (one phantom in memory) and logs its exit
  status, and the set is capped at 60 heads / 180 samples to land today. transfer.py ran end to end on the 16 (40
  epochs): zero-fill NMSE 0.133 / SSIM 0.569, synthetic-only 0.200 / 0.662, real-only 0.050 / 0.875, synthetic-then-real
  0.067 / 0.846: the pipeline works; the numbers wait for the set.
- **The set, and the first transfer number**: 181 samples by 14:55 (60 heads x 3 slices, capped), but 41 of them carry
  an all-zero clean k-space: the one long-running generator process began returning nothing from the GPU after ~30
  slices (seed 15 on; denser later: three of three slices of heads 48, 54, 59), timings and logs normal, the same head
  regenerated alone fine. The driver now refuses to write a sample with zero or non-finite signal, generation runs
  one head per process (as the realism runs always did), and the 20 affected heads are being regenerated. On the 140
  good samples (`transfer_1`, 40 epochs, 4 x undersampling, 11 held-out real people): zero-fill NMSE 0.133 / SSIM 0.569;
  **trained on synthetic only 0.166 / 0.756**; trained on real (23 other people) 0.053 / 0.857; synthetic then real
  0.058 / 0.847; the synthetic net on synthetic 0.259 / 0.765. The synthetic net sharpens real images (SSIM 0.57 ->
  0.76) but gets their intensities wrong (NMSE up): a scale or contrast mismatch to read before anything else, and its
  own in-domain NMSE (0.26) says the synthetic task is harder than the real one or the net is undertrained at 140
  samples. Next: the full 181, zero-fill and the real net scored on synthetic (in-domain floors), the error maps.
- **Why the synthetic net sharpened but did not de-alias**: its error on real images sat in the mid and bright bands
  and at both scales, no global scale (a least-squares rescale made NMSE worse), and the error maps showed the
  zero-fill's horizontal streaks still there. The undersampling mask was applied to the simulated k-space's last axis
  before the display transpose, so the synthetic training inputs aliased along the other image axis than the real
  test inputs (streak-direction ratio 0.76 against the real 1.42; after transposing the k-space into the display
  orientation first, 1.81). transfer.py fixed (7f904a7); the test reruns on the regenerated 181.
- **The first transfer number, M7's benchmark** (`~/data/transfer_1` on the mini; 181 synthetic samples from 60 v16
  heads with 0-4 lesions each, 2 x 3 x 3 spins, the 34 real protocols, the receive drawn per head, the paired scans'
  coloured noise; 4 x Cartesian undersampling with an 8 % centre at 128 x 128; a 0.5 M-parameter residual U-Net, 40
  epochs; 11 held-out real people, 77 slices): zero-fill NMSE 0.133 / SSIM 0.569; **trained on synthetic data only
  0.052 / 0.867**; trained on real data (23 other people, 161 slices) 0.053 / 0.857; synthetic then real fine-tuning
  0.044 / 0.887. In domain: the synthetic net on synthetic 0.053 / 0.885, zero-fill on synthetic 0.172 / 0.518 (the
  synthetic task is the harder one), the real net on synthetic 0.071 / 0.846. A reconstruction net that never saw a
  real scan matches one trained on real scans, and pre-training on synthetic beats real-only by 17 % in NMSE. Caveats:
  one seed (a second runs as transfer_1s1), a small net and matrix, a real training set of 161 slices (the regime
  synthetic data is for), the same 34 scans' protocols and noise covariances feed both sides (not their images).
- **The mini's `~/data` is gone** (22:22, its owner, after the disk reached 88 % with 42 GB of fastMRI archives, every
  run's raw k-space and the generated set): the 34 + 37 real scans, the realism cache and noise covariances, r27-r34
  and their scores, `gen_t2w` and `transfer_1*`, the v15s / v16 / v17 preparation logs. The repo's prepared heads
  survived (v14-v17; the obsolete versions pruned, the repo's data dir 9.8 -> 7.1 GB, 72 GB free). Every number is in
  the journal; the second-seed transfer run and r34 were lost unread. Standing rule from the user: keep the mini's disk
  use small (memory `mini-disk-budget`, LESSONS). The real set will come back as a compact form, the 128 x 128 coil
  k-space per slice that the realism score and the transfer test actually use (~1.5 GB for the 34 scans instead of
  16 GB of raw files), streamed and compacted file by file.
- **The compact real set** (6beffdc): a scan is kept as its central 128 x 128 coil k-space per slice plus the header's
  facts (`realism.coil_kspace`, `.npz`, ~27-40 MB) and the loaders, floors and noise covariances are keyed by the
  fastMRI file name; `fetch_fastmri.py --compact 128 --min-field 2` streams the archive, compacts each 3 T AXT2 volume
  and deletes the raw file, drops the 1.5 T ones (the first: 0.55 GB -> 27 MB). Streaming into `~/data/fastmri_c` on
  the mini overnight (0.2 GB/min); r34 (62 v16 heads at 2 x 3 x 3, the drawn receive, one head per process) is
  chained behind it and the v17 preparation resumed on the CPU. echo_train.py still wants raw files (the k-space
  line energies): it ran once and its result is in the journal.
