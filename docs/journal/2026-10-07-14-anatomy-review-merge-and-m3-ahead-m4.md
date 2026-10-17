# 2026-10-07 — Anatomy review, merge, and M3 (AHEAD) / M4 / M5

- Reviewed the second agent's M1-M3 (susceptibility, BrainWeb, QSM): correct kernel and
  shimming, conservative partial-volume assembly, honest limits; 462 tests, defaults
  bit-identical vs main. Merged by replaying its four commits onto main (dropping my duplicate,
  journal conflicts as unions). End-to-end: realistic BrainWeb GRE images at 3 T.
- Fixed the main realism error found in review: CSF/fat used legacy 1.5 T simulator values
  (CSF T2 0.33 s). Now cited: CSF T1 4.4 s (Rooney 2007), T2 2.03 s (Spijkerman 2018), fat
  382/68 ms (de Bazelaire 2004), 1.5 T fat from IT'IS. T2-weighted CSF is now bright.
- M4: within-tissue texture as mean-1 log-normal Gaussian random fields per tissue/property,
  coarse-grid FFT synthesis with exact discrete normalisation; realised SD 95-100 %, correlation
  at lag L within 1.4 %; separate seed; crops share the realisation.
- M3 alternative: `load_ahead` (AHEAD v3 7 T: R1/R2*/PD/QSM); units fixed by Caan 2019 values
  (WM T1 1.14 vs 1.19 s, T2* 28 vs 27 ms) and a positive pallidum; synthetic RPS tests. Found
  that the default T2* policy drops a measured R2* map when chi is on (documented; the
  validation supplies R2' = R2* - R2).
- M5: `examples/anatomy_brain.py` (1.86 M spins, three contrasts, 44 s on the M4 GPU) and
  `validation/anatomy_ahead.py` (7 T GRE showing pallidal/putaminal darkening at TE 20 ms).
