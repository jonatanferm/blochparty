# 2026-10-08 — Benchmark 7: measured GIRF (Prisma) through apply_girf

- `validation/real/fetch_girf.py` (1.23 GB, MD5), `girf.py`: Python port of the thin-slice output
  computation; our Meas1 GIRF equals the authors' precomputed one to <= 0.054 below 10 kHz after a
  half-dwell frame shift (delays 1.58 / 1.62 / 4.41 us in our frame, theirs minus 2.5 us).
- Held-out: Meas2 (8 months later, both polarities) predicted from Meas1 through `apply_girf`:
  0.12-0.38 % rms of peak vs 0.67-1.61 % ideal and 0.38-0.63 % pure delay; y worst (its delay moved
  ~1 us). Moments within 0.05 % for all models.
- B0 term: below the drift floor (odd part <= 0.022 rad in 10 ms vs 0.6 rad even part); DC part is
  slice-pair geometry. A units slip (Hz per mT/m used as Hz per Hz/m) caught by the 1e6 % error.
- Finding: blip dephasing (0.53 at k = 371 /m) shows a wider-than-1 mm slice profile.
