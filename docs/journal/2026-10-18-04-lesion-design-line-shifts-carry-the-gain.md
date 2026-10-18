# 2026-10-18 — Lesion design: the sub-cycle line shifts carry the gain, by lowering the aliasing

- Entry 03 put the learned sequence's gain in its per-line part (each line's flip and position at the start's TE
  and TR). `split_design.py` (f6d35af) now also writes that part's two halves: **flips** (the learned flip pattern,
  shifted to 15 deg, on the start's grid lines) and **shifts** (the learned line positions with flat 15 deg flips).
  Both saved sequences split; the halves and the whole per-line part ran as arm L next to G, ten seeds, seed-paired,
  on a secure RTX 4090 (no secure L40 was free), started with the new `validation/design/pod/launch.sh`.
- Test AUC, mean of 10 nets (2.8 s each, TE 60 / TR 70 ms), minus G with the subject-bootstrap 95 % interval:

  | | run 1's design | run 2's design |
  |---|---|---|
  | G (grid lines, flat 15 deg) | 0.738 | 0.738 |
  | flips only | 0.736: -0.002 [-0.014, +0.014], 5/10 seeds | 0.748: +0.010 [+0.002, +0.019], 5/10 |
  | **shifts only** | **0.825: +0.087** [+0.041, +0.136], 10/10 | **0.812: +0.074** [+0.032, +0.126], 10/10 |
  | both (the per-line part) | 0.826: +0.088 [+0.042, +0.140], 10/10 | 0.809: +0.071 [+0.034, +0.118], 10/10 |

  The per-line part repeats entry 03 within its spread (0.836 and 0.821 there, on another 4090 host).
- **The line shifts are the whole per-line effect; the learned flip pattern does nothing.** Lines moved off the grid
  by rms 0.28 and 0.36 cycles (at most 0.68 and 0.85), with no trend across k-space (correlation with ky +0.07); the
  two runs' shifts correlate 0.43 line by line.
- **What the shifts do: they lower the zero-filled image's aliasing.** The adjoint's 1-D point-spread function along
  the phase-encode axis (`|sum_l exp(2 pi i ky_l y / 64)| / 32` at the pixel centres): with the main lobe (|y| <= 1 px)
  left out, G's lines give a peak sidelobe of 0.231 and an rms of 0.1028; run 1's shifts 0.222 and 0.0951, run 2's
  0.149 and 0.0764. Random jitter of the same size (rms 0.3 cycles, 5000 draws) raises the sidelobes (rms 0.1247 on
  average, the lowest of 5000 0.0903): the learned shifts sit below 99.9 % (run 1) and all (run 2) of them. The
  network's gradient through the line positions found a sampling with less coherent aliasing, the incoherence
  criterion of compressed-sensing sampling design, without being told about it. The sidelobe level does not rank
  the two runs (run 2's lower sidelobes, the smaller gain), so it is not the whole account.
- Next, two arms in one run: lines optimised for low PSF sidelobes alone (no simulation, no network: is it only PSF
  design?), and random sub-cycle jitter (does any off-grid sampling help the CNN, or only the learned one?).
- Cost: the pod deleted itself after the mini fetched the five results: copy 12 min (the mini's link was slower
  this time), caches 3.6 min, jobs 17 min, $0.51 ($5.04 -> $4.54). Results in the mini's `~/data/design11_parts/`.
- Commits: f6d35af (the split halves, `validation/design/pod/`), this entry.
