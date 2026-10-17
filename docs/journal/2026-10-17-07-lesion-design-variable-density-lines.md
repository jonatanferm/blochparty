# 2026-10-17 — Lesion design: variable-density lines at 50 % and 30 % (design5, design6)

- The runs queued in the morning finished on the mini (`~/data/design5_rss_f0.5`, `design6_gauss_f0.5`,
  `design6_gauss_f0.3`; SNR 20, RSS front, the separate design and line rates of 4baab75). Test AUC, 129 test
  slabs; G is the mean of three seeds (spread about ±0.012), A and B of three, C one run:

  | arm (scan time) | 50 % uniform | 50 % Gaussian | 30 % uniform | 30 % Gaussian |
  |---|---|---|---|---|
  | A: TSE lines, CG-SENSE, CNN (15 s; 8.9 s at 30 %) | 0.810 | 0.846 | 0.624 | 0.715 |
  | B: TSE lines, adjoint, CNN | 0.716 | 0.720 | 0.599 | 0.635 |
  | G: fixed GRE, best of six starts (2.8 s; 1.9 s) | 0.610 | 0.734 | 0.555 | 0.552 |
  | C: G's GRE learned jointly with the CNN | 0.597 | **0.803** | 0.525 | 0.562 |

  (Uniform: `design4_rss_f*` for A, B, G and the first C; `design5_rss_f0.5` for C with the separate rates.)
- **Variable-density lines are what the GRE arms needed.** At 50 % the Gaussian mask lifts G from 0.610 to 0.734:
  the evenly spread lines left the centre of k-space as thin as the edge, and the CNN reads a 3-10 mm lesion from
  the low frequencies. With those lines the joint design adds +0.07 (C 0.803 against G 0.734, outside G's seed
  spread), five times faster than the TSE arm A (0.846), and above B, which sees the same kind of adjoint image
  from 5x the scan time. C moved flips 15.0 -> 15.9 deg (spread 14.4-18.8), TE 60.0 -> 61.0 ms, TR 70 -> 79 ms and
  spread the lines further (ky -21..29, std 13.1, six within 4 cycles of the centre).
- **At 30 % nothing a GRE does helps**: G 0.552 and C 0.562 are near chance on both masks; A keeps 0.715 with the
  Gaussian mask (0.624 uniform), where SENSE has a densely sampled centre to calibrate from.
- What it shows for the M7 goal ("push the acquisition fraction down"): at half the lines, a learned 3 s GRE
  recovers most of what a 15 s TSE with SENSE gives the classifier; at 30 % the classifier needs SENSE. One C run
  per cell: a second seed of C at 50 % Gaussian is the next check before writing it up.
