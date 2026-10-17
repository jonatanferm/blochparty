# 2026-10-08 — CEST: benchmarks 8 (MAGMA, consistency) and 9 (iohexol, Tsat prediction)

- Benchmark 8 (5f6640b): blochparty = BMCTool on the library's APTw .seq to 8e-4; the library's
  L-arginine models miss their phantom (rms 0.05-0.12). New: offsets simulated in their RF frame
  (exact; 100 us steps; 7-8x faster).
- Benchmark 9: raw iohexol references reconstructed with an own twix reader (the prepared .mat
  was ambiguous); offsets batched as spins with the per-pulse phase steps moved into the gaps
  (exact to 3e-5, 58 offsets in 15 s); LM calibration at 1.5 s (rms 0.013 median); prediction of
  4 s: rms 0.022 vs 0.032 for "no change", change correlation 0.75, CEST build-up under-predicted
  1.3-6x (T1 from a single Tsat). Dead ends: a name clash (step) cost one 20-min fit; the
  annulus tube finder picked inter-tube gaps here (tubes taken from the +1 ppm image instead).
