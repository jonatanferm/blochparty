# Lessons: Sequence design

## Sequence design by classification — 2026-10-17

- **A design gradient through line positions does sampling design** (2026-10-18, the split of the learned GRE): the
  whole per-line gain (+0.07-0.09 AUC) came from moving lines off the Cartesian grid by ~0.3 cycles, and the learned
  flip pattern added nothing. The moved lines lowered the zero-filled point-spread function's sidelobes below 99.9 %
  of random jitters of the same size, which raise them. Check what a learned trajectory does to the PSF before
  reading it as physics, and compare it with random perturbations of the same size, not only with the start.

- **The gain was in the per-line freedom, not the contrast point** (2026-10-18, split_design.py): of the +0.10 AUC a
  learned GRE gave over its start, the per-line part (a mild band pattern of flips across k-space and sub-cycle line
  shifts) carried 85-95 % at the start's own TE, TR and scan time; the learned flip level, TE and TR alone gave
  +0.02-0.03. Searching a few global sequence parameters finer would not have found it. When a learned design wins,
  split it into what a conventional parameter search could reach and what only the extra freedom can, and test
  each part with fresh nets.

- **Judge a learned acquisition with a reader trained from scratch on it** (2026-10-18, arms C, F, L): the joint run's
  own network, fine-tuned from the start sequence's for 400 steps, scored 0.73-0.82 (mean 0.78 over five runs), and
  the matched control F said the sequence added only +0.04. Fresh nets trained on the same learned sequences with the
  baseline's recipe scored 0.842 and 0.841 (ten seeds each, +0.100 over the baseline, 10 of 10 seeds). The fine-tuned
  net carries the start sequence's habits and too little training on the new images; the joint run's AUC
  undersells what it found. Seed-pair the comparison (same initialisation, order and noise) and resample subjects,
  not slabs, for the interval.

- **The same seed does not give the same network on CUDA** (2026-10-18, lesion_design.py seed 1 run twice on two
  L40 hosts): the G net trained on cached k-space (no simulation in the loop) reached train AUC 0.786 one time and
  0.768 the next, another G net 0.709 and 0.746, and arm C 0.758 and 0.730. The runs differed in CPU threads (15
  against 10) and GPU sharing, and torch's CUDA defaults allow non-deterministic convolution algorithms; which one
  moved it was not isolated. So a seed's spread includes non-determinism, and only comparisons inside one job are
  clean: arm F reuses C's starting net and C's generator, and its gap to C is the paired number.

- **A repeat with a new seed must leave what it holds fixed alone** (validation/design/lesion_design.py, the second
  seed of arm C): `--seed` set the networks' initialisation and also drew the Gaussian line mask, so "the same run
  with another seed" would have scanned other lines and mixed the mask's luck into the seed spread. `--mask-seed`
  now draws the mask on its own; the repeat checked its `ky_cycles` against the first run's before reading any AUC.
  The same goes for anything a seed reaches indirectly (caches keyed by it, the data split).

- **Know which k-space axis is the phase encode before you undersample it** (validation/design/lesion_design.py, the
  first two sweeps): every Cartesian k-space here is (coil, ky, kx) — `cartesian_kspace` fills the TSE's
  `kspace_index` line by line and the GRE's `kspace_shape` is (lines, readout) — and `ifft2c` of it has rows along +y.
  The first design script applied the line mask to the *last* axis, so "20 % of the lines" dropped readout samples
  (which saves no scan time and unfolds along x), and CG-SENSE was consistent with itself but modelled the wrong
  acquisition. The check that settles it: zero everything but the central lines of one axis and see which image
  direction blurs; the head is also longer in A-P (rows) than L-R. The transfer test had the same class of bug a day
  earlier (masking before the display transpose). Orientation tests are cheap; do one whenever a mask meets an array.

- **A "noise level" must be stated in the image domain to mean anything**: sigma = (mean |sample| of a reference
  acquisition) / 20 gave an image SNR of ~1200 (the mean |k| is set by the centre of k-space; the noise spreads over
  N^2 pixels). Both null sweeps were effectively noiseless. `--snr` is now the reference GRE's head median over its
  pixel noise standard deviation (sigma = N x median / snr for `ifft2c`'s 1/N^2 normalisation).

- **A small-lesion detector needs max-pooling, augmentation, width and epochs, and a clean label**: on 276 slabs the
  image CNN behind SENSE at full sampling went from AUC 0.62 (mean pooling, 40 epochs, no augmentation: train AUC 0.98,
  it memorised the heads) to 0.89 (max-pooling over space, flip + shift augmentation, 32-64-128 channels, 100 epochs,
  the 0 < lesion px < 4 slabs left out); each piece alone moved it 0.03-0.08 and the width most. Lesions here are
  1.6x the brain median but 16 % peak below the brain's 90th percentile (sulcal CSF): the task is context, and
  the net has to see it enough times with the lesion in enough places. Before any comparison of acquisitions, show
  the baseline can learn the task at full sampling.

- **A CNN cannot read a small lesion out of raw k-space, and does not need to**: on the same data the raw-sample net
  reached AUC 0.58 (and overfit); the encoding's adjoint (a parameter-free per-coil zero-filled inverse DFT, 1e-6 from
  `ifft2c` on the grid and differentiable in continuous line positions) put the same net at 0.75-0.83. The raw arms
  keep "no reconstruction" (no iterations, no maps, no regulariser) by reading the adjoint, which is the first layer any
  sensible raw network would learn anyway.
