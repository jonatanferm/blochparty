# 2026-10-08 — Benchmark 6: radial Pulseq timing (Cima.X)

- `validation/real/fetch_radial.py`, `radial.py`. The import's k at every ADC sample equals the file's
  trajectory to 7e-4 / FOV. Measured effective delays dx +0.20, dy +0.65, dxy +0.05 us (250
  antiparallel pairs, scatter 0.05 us); cross-slice prediction 0.04-0.07 us rms vs 0.42-0.49 us
  without delays.
- Simulating the .seq with imposed delays and re-estimating: exact for zero and equal delays; the
  antiparallel estimator turned out biased for unequal delays with an off-centre object (ellipse:
  (3, -2) -> (3.38, -1.53) us), from opposite sideways shifts of the partners. For a 19 cm disc
  placed like the measured object with 8 loop coils the bias is <= 0.005 us, so the measured values
  stand. Dead end avoided: blaming the GIRF segment averaging (equal delays are exact).
- Not testable here: GIRF shape and B0 terms (no GIRF for this scanner).
