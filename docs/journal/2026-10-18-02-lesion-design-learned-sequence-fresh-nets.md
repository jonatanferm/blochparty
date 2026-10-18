# 2026-10-18 — Lesion design: the learned sequence read by fresh nets (arm L), ten seeds

- The cleaner test proposed in entry 01: hold a learned sequence fixed and train fresh nets on it with G's recipe, so
  no fine-tuning path is involved. Arm L (56d6cca) loads a saved `arm_C.pt`, caches its k-space and calls the same
  `fit` as G: for each seed L's net and G's share initialisation, order, noise and dropout, and differ only in the
  sequence. Checked first: L on a file holding G's own start sequence reproduced G's runs (same AUCs, test scores
  within 3e-6, 12 slabs, MPS). The two learned sequences that were saved: run 1's C (`design6_gauss_f0.5`, MPS,
  seed 0) and run 2's (`design7_gauss_f0.5_seed1`, CUDA, seed 1); both learned on the training slabs only, and no
  sequence was chosen on the test set. Ten seeds (five jobs of two on one L40), the same 32 Gaussian lines, SNR 20.
- Test AUC, 129 slabs (12 test subjects):

  | arm (scan time) | mean of 10 nets | spread | ensemble of 10 | minus G, paired by seed |
  |---|---|---|---|---|
  | G: start sequence, 15 deg, TE 60, TR 70 ms (2.8 s) | 0.742 | 0.708-0.786 (sd 0.030) | 0.769 | |
  | L: run 1's learned sequence (3.2 s) | **0.842** | 0.816-0.870 (sd 0.016) | 0.863 | +0.100, 10 of 10 |
  | L: run 2's learned sequence (3.2 s) | **0.841** | 0.828-0.863 (sd 0.010) | 0.862 | +0.099, 10 of 10 |

  Seed-paired t = 13.4 for both. Resampling the 12 test subjects (2000 times, every net's scores kept with their
  slab): L minus G +0.100 with 95 % interval [+0.057, +0.145] and [+0.050, +0.155]; for the ensembles +0.092
  [+0.043, +0.152] and +0.093 [+0.039, +0.157]. Accuracy 0.774 and 0.771 against 0.736; train AUC 0.918 and 0.909
  against 0.828.
- **The learned sequence is clearly better than its start, and fine-tuning undersold it**: C's own nets, fine-tuned
  from G's for 400 steps, reached 0.78 on average (five runs); fresh nets on the same sequences reach 0.84, about the
  15 s TSE with CG-SENSE at the same lines (arm A, 0.846 in entry 2026-10-17-07), at 3.2 s. The two sequences, learned
  from different nets on different engines, give the same AUC to 0.001. Caveat on the budget: the budget is in
  lines (32 of 64), and the learned sequences also took 13 % more time (TR 79 against 70 ms).
- What the sequence does is not yet clear. A look at six local lesion slabs (noise-free adjoint images) found no
  simple story: absolute lesion-minus-surround contrast -8 % for run 1's sequence and +25 % for run 2's, head signal
  not higher. The per-line flips (14-19 deg, a k-space weighting) and the TE/TR point are confounded in L; the next
  run separates them: the learned TE/TR with flat flips on G's lines, and the learned per-line flips and lines at G's
  TE/TR.
- Runs and cost: the pod (secure L40, 15 CPUs) deleted itself through the Runpod API after the mini had fetched and
  checked the five results (the new pipeline, lessons/gpu.md): copy 6 min, caches 77 s, jobs 8.5 min, pod about
  18 min, $0.24 ($5.71 -> $5.47). Results with every net's test scores in the mini's `~/data/design9_fixed/`.
- Commits: 56d6cca (arm L), this entry and its lessons.
