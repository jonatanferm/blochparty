# 2026-10-16 — The transfer number with a fair real baseline, three seeds

- `night2`: 60 v17 heads x 3 drawn slices at 2 x 4 x 4 spins with 1-4 lesions each (`gen_les`, 180 samples, 331 min)
  and the same heads without lesions (`gen_nol`, 180), the receive drawn per head with the re-centred scale, the 65
  real protocols and their noise; three seeds of the transfer test against the 65 compact scans: real-only now trains
  on 308 slices of 44 people and is tested on 147 slices of 21 (4 x undersampling, 128 x 128, the 0.5 M-parameter
  residual U-Net, 40 epochs). NMSE / SSIM, mean over the seeds (spread in NMSE +- 0.001-0.002):
  zero-fill 0.140 / 0.56; **synthetic only 0.054 / 0.85**; synthetic without lesions 0.054 / 0.83; **real only 0.040 /
  0.89**; synthetic then real 0.041 / 0.89; the synthetic net on held-out synthetic 0.052 / 0.88; zero-fill on synthetic
  0.170 / 0.49; the real net on synthetic 0.052 / 0.87.
- Reading: with twice the real training data the real-only net pulls ahead (0.040 against 0.054; on 2026-10-14 with
  161 slices they tied at 0.053 / 0.052), and pre-training on synthetic no longer adds to it (0.041). The cross-domain
  numbers are symmetric and equal to the in-domain ones (synthetic net: 0.054 on real, 0.052 on synthetic; real net:
  0.040 on real, 0.052 on synthetic), so no domain shift is visible in either direction; what separates them is the
  synthetic task itself: its zero-fill floor is 0.170 against the real 0.140, the synthetic images carry more
  high-frequency energy than the real crops, and a net that learns to reconstruct them lands at ~0.052 wherever it is
  tested. The question for the next round is therefore the synthetic images' fine content (grain at 2 x 4 x 4,
  the 0.5 mm residual, the absence of the scanner's raw filter on a 320-line acquisition cropped to 128), not the
  anatomy. Synthetic data still takes a reconstruction 90 % of the way from zero-fill to the real-trained net with no
  real scan seen.
- **The lesion test has no statistics yet**: of the 18 held-out synthetic samples only one carried 4 or more lesion
  pixels (1-4 lesions of 3-12 mm rarely meet three 5 mm slices at 1.72 mm pixels), so its numbers swing seed to seed
  and say nothing. It needs an evaluation-only set with many, larger lesions (count 4-6, 6-15 mm) and nets saved
  (transfer.py did not keep them, and the chain deleted the raw samples by the disk rule, so the ablation cannot be
  rescored without regenerating; both are fixed in the code for the next run).
- **Where the synthetic task's higher floor comes from** (r34's scored images, 434 synthetic at 2 x 3 x 3 against the
  455 real, power along the phase axis as a share of each image's total): the always-sampled centre (0-8 % of
  Nyquist) holds 0.714 of a real image's power and 0.656 of ours; 8-25 % holds 0.163 real / 0.205 ours; 25-50 %
  0.093 / 0.111; the top two bands equal or lower in ours (0.022 / 0.021, 0.0086 / 0.0074). So it is not the grain:
  our images carry ~25 % more mid-scale power (12-35 mm features) and less smooth large-scale shading than the real
  crops, and a 4 x mask therefore removes more of them (zero-fill proxy on the magnitude images 0.167 vs 0.155, the
  same direction as the k-space floors 0.170 vs 0.140). Inside the brain the isotropic bands are near 1 (gap.py), so
  the excess sits at the head's rim: the scalp-fat / black-bone / cortex sequence is a stronger mid-frequency feature
  in ours than in the real scans (the real bone band is never as dark, the real scalp smoother), and the real images
  keep a broader shading our normalised receive flattens. That is a simulator question (the rim's rendering, the
  shading the scanner's normalisation leaves), testable with quick runs and gap.py, not with another night of
  generation.
- **Today's plan, agreed**: (1) the lesion run on the mini (`night3.sh`, started 11:26): the training set with lesions
  regenerated on 50 subjects (60 heads x 3 slices, 2 x 4 x 4), an evaluation set of 20 heads x 3 slices on 12 held-out
  subjects with 4-6 lesions of 6-15 mm each (`spec_leval.json`; the generator's spec now takes a `subjects` pool),
  three seeds of the transfer test with `--lesion-eval`, nets and sets kept (~5 GB); (2) the rim through quick runs
  against q51: q53 gives the skull a dim signal (T2 40 ms, PD 0.25: the diploe's worth, an experiment by override),
  q54 draws the normalisation length 40-120 mm instead of 20-60 (more of the array's shading left), both read by
  gap.py's band minimum, scalp levels and polar map. The no-lesion ablation set is dropped: it was null on real data
  and the lesion question compares the synthetic net against the real-trained one.
- **Rim, first two variants** (v17 heads, quick quality, against q51; gap.py polar in units of each image's p99, sim /
  real by radius band): q53 (skull T2 40 ms, PD 0.25) left the posterior band at 0.72 / 0.65 / 0.74 of real at
  0.80-0.94 of the head radius (q51 0.75 / 0.71 / 0.73): at TE 113 a 40 ms T2 keeps 6 % of the signal, nothing to
  see; q54 (normalisation length drawn 40-120 mm) improved W1 (0.0161 vs 0.0191) but brightened the scalp (4.34 vs
  the real 3.90; q51 4.09) and the posterior scalp (0.52 vs 0.44) without touching the band. The maps put the rim's
  excess in two places: the posterior band 30 % too dark (0.34 / 0.24 / 0.34 against the real 0.46 / 0.34 / 0.47) and
  the posterior edge at 0.98-1.02 of the radius 38 % too bright (0.21 vs 0.15), with the sides' band also at 0.77-0.80.
  Next: q55 with a skull signal that shows (T2 70 ms, PD 0.3: ~0.3 x WM, the diploe's worth, an experiment) and q56
  with the dermis on (the outer 1.6 mm dark; on 2026-10-13 it over-sharpened the edge, before the receive changes).
- **Rim, second pair** (against q51): q55 (skull T2 70 ms, PD 0.3) brightened the posterior scalp (0.54 vs the real
  0.44) and the brain's median over WM (1.34 vs 1.24) and left the inner band darker (0.68 / 0.61 at 0.80-0.90 of the
  radius) while 0.90-0.94 rose to 0.82; W1 0.0151 (the best yet) with a worse histogram AUC. q56 (the dermis) put the
  band minimum exactly on real (0.138) but the outer edge brighter still (1.55 at 0.98-1.02; sides 1.16-1.31). One
  class at a time does not converge because the p99 normalisation couples every change; the generated samples carry
  the tissue fractions on the image grid, so the rim's composition by radius is read directly (scratch rimcomp.py).
- **The rim read from the truth grids** (`gap.py composition`, 30 generated v17 samples at 2 x 4 x 4, clean images over
  p99, against the real polar levels; posterior sector by band of the head radius): 0.70-0.80: sim 0.43 / real 0.48
  (cortex); **0.80-0.86: sim 0.26 / real 0.46**, where ours already holds skull 0.21, skin 0.13, muscle 0.08 beside
  gm 0.29; 0.86-0.90: 0.28 / 0.34 (skull 0.32, skin 0.21); 0.90-0.94: 0.46 / 0.47 (fat 0.29); 0.94-0.98: 0.66 / 0.54
  (fat 0.46: a thinner, brighter fat layer); 0.98-1.02: 0.20 / 0.15 (skin 0.49, the dim surface on both sides). The
  sides: 0.80-0.86 0.28 / 0.41 with skull 0.29. So the dark valley sits one band (~4 mm of a 90 mm radius) nearer the
  brain than in the real scans and the real valley is shallower: the clinical population's posterior scalp + skull is
  thicker than the IXI volunteers' as segmented (the 0.57 mm profiles said 6.6 vs 5.0 mm of soft tissue and the valley
  at 10-11 vs 9 mm), and a thinner scalp also concentrates the fat into a brighter peak. No relaxation value fixes a
  thickness; q56's dermis only moved the head-mask edge. The item is a scalp / skull thickness draw against the real
  profiles (population), on the roadmap under M5b item 4; the rim work stops here for today.
- **night3: the lesion run** (60 training heads on 50 subjects with 1-4 lesions, 180 samples, 357 min; 20 evaluation
  heads on 12 held-out subjects with 4-6 lesions of 6-15 mm, 60 samples; three seeds; sets and nets kept, 7.7 GB):
  the transfer numbers repeat yesterday's (synthetic-only 0.055, real-only 0.039, synthetic then real 0.042; the
  synthetic net on held-out synthetic 0.048-0.051, zero-fill on synthetic 0.158-0.159). The targets' power, synthetic
  over real by sixth of Nyquist: 0.96 / **1.32** / 1.13 / 0.98 / 0.90 / 0.90: the mid-band excess (the rim) and a 10 %
  deficit at the top, as read from r34 this morning.
  **The lesion test** (held-out synthetic samples with lesion pixels; NMSE inside the lesion / in the rest of the
  brain; the reconstructed lesion level over the true one), per seed: the synthetic net 0.055 / 0.045 (0.83), 0.101 /
  0.046 (0.78), 0.084 / 0.043 (0.77); the real net 0.063 / 0.046 (0.80), 0.107 / 0.046 (0.79), 0.095 / 0.042 (0.73);
  zero-fill 0.052-0.081 / 0.11 (0.86-0.97). Both nets err about twice as much inside lesions as elsewhere and dim them
  by 17-27 %; the net that saw lesions in training is consistently a little better than the real-trained one (8-12 %
  lower lesion error, 3-4 points of level) but far from keeping them; zero-filling keeps the level and loses the detail.
  Only 15 of the 60 evaluation samples hit a lesion with 4 or more pixels (lesions anywhere in white matter against
  three 5 mm slices of 1.72 mm pixels), and the numbers move with the mask: re-evaluated below with the saved nets on
  the full set over several masks and by lesion size.
- **Correction, the same evening**: transfer.py fine-tuned the synthetic net in place, so in every run so far the
  in-domain "synthetic on synthetic" score, the lesion test's "synthetic" row and the saved `net_synthetic.pt` were the
  **fine-tuned** net's (the synthetic-on-real headline was evaluated before fine-tuning and stands; the two lesion
  rows above that agree to the digit are the same net). The claim that the net which saw lesions does a little better
  is therefore unsupported; re-evaluating the saved nets on the full evaluation set over 8 masks gives real 0.070 /
  0.081 / 0.074 and fine-tuned 0.066 / 0.081 / 0.091 inside lesions (brain 0.040), lesion levels 0.78-0.81 for both,
  zero-fill 0.102 at level 0.98: so far the only firm lesion finding is that both trained nets dim lesions by ~20 % and
  double their error there while zero-filling keeps the level. Fixed (a copy is fine-tuned), three seeds rerunning as
  transfer_4 with the pure synthetic net.
- **transfer_4, the corrected three seeds** (the pure synthetic net kept apart from its fine-tuned copy; the same data as
  night3). On real test slices (NMSE / SSIM, three seeds): zero-fill 0.140 / 0.56; synthetic only 0.056, 0.051, 0.057 /
  0.86; real only 0.040, 0.037, 0.041 / 0.89; synthetic then real 0.042, 0.040, 0.043 / 0.88. In domain, now of the
  right nets: the synthetic net on held-out synthetic 0.048, 0.048, 0.052 (mean 0.049) against its 0.055 on real; the
  real net on synthetic 0.049, 0.048, 0.051 against its 0.039 on real; zero-fill on synthetic 0.159. So there is a
  domain shift after all, modest and asymmetric: the synthetic net loses ~12 % crossing to real, the real net ~28 %
  crossing to synthetic (the harder task: more mid-band power, the rim); the earlier "cross-domain equals in-domain"
  rested on the fine-tuned net's score and is withdrawn.
  **The lesion test, with statistics** (the full evaluation set's 15 lesion-bearing samples x 8 undersampling masks,
  lesion NMSE / brain NMSE / reconstructed lesion level, three seeds): the pure synthetic net 0.099 / 0.045 / 0.75,
  0.080 / 0.041 / 0.79, 0.091 / 0.044 / 0.80 (mean 0.090, level 0.78); the real-trained net 0.070 / 0.040 / 0.79,
  0.081 / 0.039 / 0.78, 0.074 / 0.039 / 0.79 (0.075, 0.79); synthetic then real 0.066, 0.081, 0.091 (0.079, 0.79);
  zero-fill 0.102 / 0.112 / 0.98. **Training on synthetic lesions does not help**: the net that saw them is worse
  inside lesions than the one that never did (0.090 vs 0.075) and no better in level; every trained net dims lesions
  by ~20 % (small and medium alike, 0.76-0.85 by size) and doubles its error there, while zero-filling keeps the level
  and the noise. At 4 x with a 0.5 M-parameter image-domain U-Net, lesion fidelity is set by the reconstruction's
  pull toward smooth normal tissue, not by what the training set contained: the lever is the method (a loss that
  penalises losing small bright structure, or data consistency in k-space), not more lesion data. That is the first
  number the lesion generator has produced, and it says something about reconstruction, not about the generator.
