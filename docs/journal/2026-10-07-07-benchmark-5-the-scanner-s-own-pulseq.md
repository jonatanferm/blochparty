# 2026-10-07 — Benchmark 5: the scanner's own Pulseq spiral TSE (T1MES, Cima.X)

- Fetched Zenodo 18749100 (`.seq` + ISMRMRD with trajectory; MD5). The `.seq` imports (5.1 M
  segments, 145,600 samples); the first three TRs are simulated (1024 spins across 32 mm:
  converged; B0 +-40 Hz, T1 and refocusing scale have no effect on the shape).
- Reconstruction (`spiral_tse_recon.py`): CG-SENSE per echo. First coil maps came from a
  Pipe-style DCF that does not converge with the NUFFT Gram operator: random maps, and a false
  "tube 3 anomaly" (Bloch 3.8 % vs mono 1.2 %) that I chased through B0, B1, T1 and refocusing-
  scale tests. Per-coil least-squares maps (deterministic) removed it.
- Result: Bloch on the exact `.seq` at nominal B1, T2 fitted per tube: shape residual 1.1–2.7 %
  vs mono-exponential 3.9–5.4 % (short-T2 tubes) and 1.6–2.3 % vs 1.7–2.0 % (long-T2); method floor
  0.2–0.6 % (digital twin). Global B1 fit = 1.00. Remaining < 1 % of echo 1 per echo: odd/even at
  late echoes, echo-1/2 step; candidates MT of agarose, eddy currents, multi-exponential T2.
- T1MES references are 1.4 T values; the 3 T T2s here are 35–50 % shorter (agarose exchange).
