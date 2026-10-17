# 2026-10-10 — Realism score v0 (fastMRI 3 T T2w vs simulated heads); multi-slice TSE

- **Data**: the fastMRI stream finished with 71 brain volumes (30.9 GB); the 21 fully sampled 3 T
  AXT2 scans (Siemens Skyra/Prisma/Biograph mMR, TSE TR 6.0-7.7 s, TE 105-113 ms, 16 x 5 mm,
  14-20 coils) are the real set. Their stacks run from the lateral ventricles (slice 0) to above
  the vertex in all 21.
- **One reconstruction** (`validation/realism/realism.py`): fastMRI k-space to the 220 mm field of
  view (readout oversampling removed), central 128 x 128 of k-space, RSS, p99 normalisation. It
  first reproduces the files' `reconstruction_rss` (correlation 1.0000000).
- **Matched simulation** (`simulate_heads.py`): each `sample_head` head takes the TR/TE/coil count
  of a paired real scan; stacks placed by the same landmarks (slice 0 at the largest lateral
  ventricles, brain top at slice 13.5; spacing 4.7-5.1 mm); phantom turned so the readout runs
  anterior-posterior as in the real data (checked against the anatomy: r = 0.94; flips 0.82 / 0.35).
  Two first attempts at placing the stack by head area failed (BrainWeb's face and neck dominate
  the area); the landmark rule replaced them.
- **Score** (`score.py`): noise matched to each paired scan's floor (background 1.7 % of p99), a
  small CNN with person-wise class-balanced folds (heads of one BrainWeb subject are one person),
  bootstrap CIs, a histogram baseline, real-vs-real and sim-vs-sim nulls, and a classifier-free
  intensity distance (1-Wasserstein of head pixels).
- **First numbers.** 5 heads (13 slices): CNN AUC 0.998 [0.990, 1.0], histogram 0.91. 7 heads
  (even slices): CNN 0.91 [0.67, 1.0], histogram 0.95, intensity distance 0.050 (real vs real
  0.008, sim vs sim 0.014). What gives the simulations away, by eye and by slice: white matter too
  bright (no multi-slice MT), a thick grey scalp/muscle band (muscle T2 50 ms from excised mouse
  muscle against ~30 ms in human spin-echo studies), weaker coil shading. Per-slice AUC is 1.0 at
  ventricle level and 0.92 near the vertex (more muscle low down).
- **Null caveat**: person-wise CV biases AUC below 0.5 in nulls (histogram real-vs-real 0.20, 0.27
  after class-balanced folds); calibrate small differences with a permutation test (LESSONS).
- **Multi-slice TSE** (`tse_2d(slice_order=...)`, b2fb6af): the other slices' pulses as envelopes on
  a 100 us raster, moved to their slices by `b0_offset = -G z_j`. Fast vs exact phase-ramped pulses:
  0.8 % of the signal at 1/6 of the segments; 7 slices cost white matter 31 % against one slice;
  Metal = core to 3e-6; default sequences unchanged (SHA-256). A 200 us raster is 16 % off. GPU cost
  per slice: x2 for MT, x3 for the other slices' pulses, x6 together.
- **Tissue database** (merged by the other session as PR #7): its pooled 3 T values weight studies
  by n, so one large study can decide a tissue: muscle T2 52 ms (MRF abdomen 74 ms and pharynx
  54 ms outweigh spin-echo studies at 29-32 ms), WM and GM T2 both 86 ms (a SyMRI study, n = 54,
  gives WM 108 ms, GM 80 ms). Flagged to the user; the ablations test it.
- **Running** on the Mac mini (seeds 0-7, slices 0-12:2, 4 spins per tissue and voxel): baseline,
  + MT + 16-slice acquisition, and the same with `tissuedb.tissue_table(3.0)`.
- Commits: e147877 (fetch), b2fb6af (multi-slice TSE), c79c83c (realism v0), dc85124 (ablation
  switches).
