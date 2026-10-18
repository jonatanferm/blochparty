# 2026-10-18 — Lesion design: what in the learned sequence helps (TE/TR against the per-line design)

- Entry 02 found fresh nets reading C's learned sequences at AUC 0.84 against 0.74 for the start sequence, with the
  contrast point (flip level, TE, TR) and the per-line design (each line's flip and position) learned together.
  `validation/design/split_design.py` (f6a0e68) splits a learned sequence into its **TE/TR part** (the learned flip
  level, TE and TR with flat flips on the start's lines: what a finer search over flip, TE and TR could find) and its
  **per-line part** (the learned per-line flip pattern, shifted to the start's 15 deg, and the learned line positions,
  at the start's TE 60 / TR 70 ms); checked exact to 2e-6 deg. Both saved sequences split, all six run as arm L
  next to G in the same jobs (ten seeds, seed-paired), on a secure RTX 4090 (no secure L40 was free).
- Test AUC, mean of 10 nets, and each minus G (seed-paired; 95 % interval from resampling the 12 test subjects,
  `validation/design/paired_report.py`):

  | sequence (scan time) | run 1's design | run 2's design |
  |---|---|---|
  | G: start, 15 deg, TE 60, TR 70 ms (2.8 s) | 0.734 | 0.734 |
  | TE/TR part (3.2 s) | 0.755: +0.022 [+0.006, +0.039], 8/10 seeds | 0.765: +0.031 [+0.018, +0.045], 9/10 |
  | per-line part (2.8 s) | **0.836: +0.102** [+0.048, +0.156], 10/10 | **0.821: +0.087** [+0.045, +0.132], 10/10 |
  | full learned sequence (3.2 s) | 0.840: +0.106 [+0.061, +0.153], 10/10 | 0.839: +0.105 [+0.057, +0.158], 10/10 |
  | interaction (L - T - P + G) | -0.017 [-0.036, +0.003] | -0.013 [-0.027, +0.003] |

- **The per-line design carries the gain**: 85-95 % of it at the start's TE and TR, and at the start's scan time
  (2.8 s), so the extra 13 % of scan time the full design took is not what pays. The TE/TR part is a real but small
  +0.02-0.03; the two are slightly less than additive. A finer search over flip, TE and TR would have found about a
  quarter of the gain at most; the rest needs a design with a free parameter per line.
- **What the per-line design is**: two things, both small. The lines moved off the grid by a fraction of a cycle
  (rms 0.28 and 0.36 cycles, at most 0.68 and 0.85), and the flips form a mild band pattern: 15.2 deg on average
  within 5 cycles of the centre, 16.5-16.7 deg at 5-15 cycles, 15.6-15.7 deg beyond (correlation of the two runs'
  patterns, line by line, 0.54). Which of the two matters is the next split (the flip pattern on the grid lines,
  and the line shifts with flat flips).
- Replication: G 0.734 here against 0.742 in entry 02, the full sequences 0.840 and 0.839 against 0.842 and 0.841,
  on another card type and host.
- Cost: the pod deleted itself after the mini fetched the five results; 13 CPUs (2 threads per job), caches
  3.6 min, jobs 17 min, $0.43 ($5.47 -> $5.04). Results with every net's scores and the four split designs in the
  mini's `~/data/design10_split/`.
- Commits: f6a0e68 (split_design.py), and paired_report.py with this entry.
