# 2026-10-14 — The 62-head runs against 34 scans: the CNN at 0.97-1.0

- Overnight on the mini, the v16 heads (62 after the screen) x slices 1-7 at quick quality, scored against the 34
  real scans (`~/data/fastmri_more`): **r31b** (the preset's 14 cm ring) CNN 0.998, W1 0.0161; **r32** (one elliptical
  normalised receive, q45's) 0.967, W1 0.0143; **r33** (the receive drawn per head) 0.9996, W1 0.0170; controls
  real-vs-real 0.536, sim-vs-sim 0.54-0.56 in all three. r30 (24 v15 heads, the ring) reads 0.904 against the same
  real set, 0.748 against the old 21.
- gap.py on all three: every tissue level over white matter within noise of real (q50 1.24-1.25 vs 1.25, q75 1.91-1.96
  vs 1.95, CSF 5.20-5.31 vs 5.16, scalp 4.36-4.46 vs 4.06), placement right (row AUC 0.52), head area 0.92-0.94 x
  real (0.33-0.40); the strongest single features the outer 6 px (0.45-0.46 vs 0.42, AUC 0.66-0.68) and the
  posterior scalp (0.51-0.54 vs 0.44, 0.67-0.73); structure by band 0.91-0.96 / 0.95-1.00 / 0.90-0.95 / 0.85-0.88 /
  0.80-0.83 / 0.82-0.85 x real (v16 is at 1.5 mm residual smoothing; 0.5 mm from v17). Every simulated image scores
  > 0.8: whatever the cue, it is in all of them and finer than these measures.
- The receive draw changed nothing the CNN cares about (r33 = r31b), though it put the scalp's spread right.
- **It is the head count**: 24 of r31b's heads (every fifth left out) against the 34 scans score 0.89, r30's 24 v15
  heads 0.904: the v16 heads and the ring are equivalent at that size, and 62 heads let the CNN learn a cue that is
  in every simulated image. The benchmark has the power M5b item 2 asked for; the cue is now the question.
- **The cue is the quick quality's grain** (gap.py `grain_wm`, Immerkaer's estimate in flat white matter as a fraction
  of p99): real 0.010 +- 0.003; r31b / r30 / q35 at 8 spins per voxel (2 x 2 x 2) 0.015 +- 0.003, that one feature's
  AUC 0.92-0.98; r27 at 64 spins (4 x 4 x 4) 0.010, AUC 0.62. The sub-voxel spin grain (journal 2026-10-11: it falls
  as 1 / sqrt(in-plane spins)) is what 62 heads let the CNN learn; the 24-head runs could not resolve it. So quick.sh's
  CNN number was never the right readout (its W1 and gap.py measures are), and every benchmark run needs the grain at
  the real 0.010: 2 x 3 x 3 spins (18; in-plane 9, 2.25 x the quick cost) should land there. r34 = the 62 v16 heads at
  2 x 3 x 3 with the drawn receive, ~8 h; v17 (the 0.5 mm residual) prepared beside it on the CPU. The receive,
  placement and bone findings stand (their measures are independent of the grain).
