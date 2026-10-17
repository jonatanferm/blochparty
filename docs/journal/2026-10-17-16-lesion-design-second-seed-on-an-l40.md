# 2026-10-17 — Lesion design: a second seed of the learned GRE at 50 % Gaussian, on an L40

- The check the 50 %-Gaussian result was waiting for (entry 07: C 0.803 against G 0.734, one C run). Arms G and C
  again with networks seeded 1-3 (`--seed 1`, G three seeds, C fine-tuning G's first net) on **the same lines**:
  `--mask-seed 0` (new, 7ea5af8) keeps the first run's Gaussian mask, and the repeat's `ky_cycles` matched the first
  run's line for line; the same split (515 train / 129 test slabs, 100 ambiguous left out), SNR 20, RSS front, 100
  epochs, 400 joint steps. Run on a secure-cloud L40 (Runpod) with CUDA training (7ea5af8 also moves the networks to
  CUDA where there is one); the slabs went straight from the mini to the pod (lessons/gpu.md).
- Test AUC, 129 slabs:

  | | first run (seeds 0-2) | second run (seeds 1-3) |
  |---|---|---|
  | G: fixed GRE (15 deg, TE 60, TR 70 ms; 2.8 s), mean of three | 0.734 (0.724-0.749) | 0.720 (0.709-0.731) |
  | the G net C starts from | 0.729 | 0.721 |
  | C: the GRE learned jointly with the CNN (3.2 s) | 0.803 | 0.758 |
  | C minus its starting net | +0.074 | +0.037 |

  C's accuracy was 0.736 both times; its train AUC 0.865 and 0.804.
- **The gain holds and halves.** Over the two seeds C is ~0.78 against G's ~0.73 (+0.056 paired with the net each C
  starts from); the second seed's +0.037 is still above G's seed spread (±0.011), but one more C seed would not have
  to fall far to sit inside it. Entry 07's +0.07 was the lucky end.
- **The design is reproducible; the spread is the network's.** The two C runs ended at nearly the same sequence:
  flips 15.9 and 16.1 deg (14.4-18.8 and 14.0-18.4), TE 61.0 and 61.1 ms, TR 78.9 and 79.0 ms, lines over ky
  -21.3..29.1 both times (std 12.9 and 13.1 cycles, six and seven within 4 cycles of the centre). Different nets,
  same direction for the scan: longer TR, slightly larger flips, the lines pushed outward.
- **What is not yet separated**: C's 400 joint steps also train the net 400 more steps. A control that fine-tunes
  G's net for the same steps with the design frozen would say how much of +0.04-0.07 is the sequence; it is the next
  run before writing this up.
- Cost and time: the job 12 min 15 s (C's 400 joint steps in 435 s, about 5x the mini; each G net ~70 s); the pod
  ~23 minutes in all, $0.30 ($7.82 -> $7.52), GPU at a mean 21 % (peak 93 %, 231 W) over the logged time including
  the copy. Pod removed; results in the mini's `~/data/design7_gauss_f0.5_seed1/` next to `design6_gauss_f0.5/`.
- Commits: 7ea5af8 (`--mask-seed`, CUDA training), this entry and the two lessons.
