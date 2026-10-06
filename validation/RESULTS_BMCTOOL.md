# Cross-validation: magnetization transfer vs. BMCTool

    uv run --with bmctool==1.0.3 python validation/compare_bmctool.py

Z-spectrum at 3 T: water T1/T2 1.0 s / 50 ms; bound pool f 0.1351, k 30 Hz, T1b 1 s, shifted to
−3 ppm (so the spectrum is asymmetric and any sign error in the RF frequency, the importer or
the shift would show); T2b 40 µs (Lorentzian) / 12 µs (super-Lorentzian); CW block pulse 1.5 µT
for 1 s per offset, then a spoiler. BMCTool 1.0.3 (pulseq-cest), read before running, solves
the Bloch–McConnell matrix exponentially per RF sample; blochparty splits relaxation and
exchange around the free pool's rotation on a 10 µs RF raster. Z = Mz / Mz(−300 ppm).

## Results

- **Lorentzian (identical lineshape definitions): max |ΔZ| = 5.8e-7** over 30 offsets, the
  bound pool's asymmetry included (MTR_asym at 3 ppm +0.0050 in both). Through the `.seq`
  file (`read_pulseq`, RF frequency from the phase ramp, spoilers as ideal spoils): max |ΔMz|
  2.1e-5 over 10 offsets, from sampling the frequency offset's phase ramp at 10 µs (BMCTool
  rotates its frame exactly).
- **Super-Lorentzian: max |ΔZ| = 0.12, near the pool**, from the lineshape convention alone:
  blochparty holds the super-Lorentzian at its 1 kHz value closer to resonance (Gloor's cut,
  `BoundPool.cutoff`), which over ±1 kHz of a pool at −3 ppm makes the spectrum symmetric about
  water; BMCTool cuts at ±1 ppm (its spline region, |Δω| < γB0 in rad/µs taken as rad/s) and
  integrates with a 101-point rectangle rule elsewhere (0.5–2 % above the exact integral at
  1–10 kHz). Far from the pool the difference is 1.4e-3.
- **With BMCTool's lineshape** (a replica, equal to BMCTool's function to 1.3e-15, patched in
  for this run only): max |ΔZ| = 6.4e-3, at −4 ppm only, exactly 1 ppm from the pool, where
  BMCTool's lineshape jumps between its spline and its outer formula and the two codes'
  rounding lands on opposite sides; every other offset agrees to ≤ 4e-7. So the engine agrees
  with BMCTool for both lineshapes; what differs is the near-resonance super-Lorentzian, a
  model choice (see `exchange.py` and `docs/LESSONS.md`).
- Found on the way: a block pulse without a frequency offset is one RF segment in
  `read_pulseq` (relaxation split around one 1 s rotation: errors of 0.6 in Mz).
  `read_pulseq(..., max_rf_step=)` cuts it.

## Z-spectra

| offset [ppm] | Lorentzian: BMCTool | blochparty | Δ | super-Lorentzian: BMCTool | blochparty | Δ |
|---|---|---|---|---|---|---|
| -300 | 1.00000 | 1.00000 | +0.0e+00 | 1.00000 | 1.00000 | +0.0e+00 |
| -50 | 0.88007 | 0.88007 | +9.3e-09 | 0.86225 | 0.86366 | +1.4e-03 |
| -40 | 0.84265 | 0.84265 | -5.8e-08 | 0.82603 | 0.82746 | +1.4e-03 |
| -30 | 0.79459 | 0.79459 | -1.6e-07 | 0.78211 | 0.78349 | +1.4e-03 |
| -20 | 0.73995 | 0.73995 | +1.8e-07 | 0.72571 | 0.72695 | +1.2e-03 |
| -10 | 0.68636 | 0.68636 | +1.0e-07 | 0.63479 | 0.64421 | +9.4e-03 |
| -9 | 0.67990 | 0.67990 | +1.6e-07 | 0.62012 | 0.64057 | +2.0e-02 |
| -8 | 0.67233 | 0.67233 | +2.4e-07 | 0.60297 | 0.63553 | +3.3e-02 |
| -7 | 0.66279 | 0.66279 | +4.1e-08 | 0.58243 | 0.62830 | +4.6e-02 |
| -6 | 0.64974 | 0.64974 | +3.1e-08 | 0.55652 | 0.61741 | +6.1e-02 |
| -5 | 0.63013 | 0.63013 | +6.5e-09 | 0.52170 | 0.60001 | +7.8e-02 |
| -4 | 0.59743 | 0.59743 | -6.2e-08 | 0.46996 | 0.56998 | +1.0e-01 |
| -3 | 0.53627 | 0.53627 | -2.6e-07 | 0.39002 | 0.51291 | +1.2e-01 |
| -2 | 0.40936 | 0.40936 | +5.7e-07 | 0.33717 | 0.39371 | +5.7e-02 |
| -1 | 0.17219 | 0.17219 | +4.1e-07 | 0.15953 | 0.16844 | +8.9e-03 |
| 0 | 0.00565 | 0.00565 | -2.4e-10 | 0.00558 | 0.00561 | +3.3e-05 |
| 1 | 0.17242 | 0.17242 | +4.1e-07 | 0.16367 | 0.16844 | +4.8e-03 |
| 2 | 0.41157 | 0.41157 | +5.8e-07 | 0.37819 | 0.39371 | +1.6e-02 |
| 3 | 0.54132 | 0.54131 | -2.7e-07 | 0.49824 | 0.51291 | +1.5e-02 |
| 4 | 0.60527 | 0.60527 | -6.6e-08 | 0.56212 | 0.56998 | +7.9e-03 |
| 5 | 0.64060 | 0.64060 | +6.6e-09 | 0.60066 | 0.60152 | +8.6e-04 |
| 6 | 0.66267 | 0.66267 | +3.5e-08 | 0.62684 | 0.62778 | +9.4e-04 |
| 7 | 0.67802 | 0.67802 | +4.8e-08 | 0.64632 | 0.64732 | +1.0e-03 |
| 8 | 0.68970 | 0.68970 | +3.0e-07 | 0.66182 | 0.66287 | +1.0e-03 |
| 9 | 0.69926 | 0.69926 | +2.0e-07 | 0.67476 | 0.67585 | +1.1e-03 |
| 10 | 0.70755 | 0.70755 | +1.3e-07 | 0.68596 | 0.68708 | +1.1e-03 |
| 20 | 0.77096 | 0.77096 | +1.9e-07 | 0.75953 | 0.76086 | +1.3e-03 |
| 30 | 0.82389 | 0.82389 | -1.4e-07 | 0.80880 | 0.81022 | +1.4e-03 |
| 40 | 0.86601 | 0.86601 | -4.8e-08 | 0.84820 | 0.84963 | +1.4e-03 |
| 50 | 0.89754 | 0.89754 | +9.5e-09 | 0.88079 | 0.88216 | +1.4e-03 |

## Run log

```
BMCTool super-Lorentzian replica vs BMCTool: max rel. difference 1.3e-15 (0-30 kHz)
lorentzian (T2b 40 us), Z = Mz / M0 at -300 ppm; BMCTool 0.0 s, blochparty 12.8 s (30 offsets as spins), .seq path 122 s
max |dZ| 5.8e-07; via .seq: max |dMz| 2.1e-05 (offsets -300, -20, -5, -3, -1, 0, 1, 3, 5, 20); MTR_asym at 3 ppm: BMCTool +0.0050, blochparty +0.0050
superlorentzian (T2b 12 us), Z = Mz / M0 at -300 ppm; BMCTool 0.0 s, blochparty 14.4 s (30 offsets as spins), .seq path 146 s
max |dZ| 1.2e-01; via .seq: max |dMz| 1.2e-01 (offsets -300, -20, -5, -3, -1, 0, 1, 3, 5, 20); MTR_asym at 3 ppm: BMCTool +0.1082, blochparty +0.0000
blochparty with BMCTool's super-Lorentzian (replica): max |dZ| 6.4e-03
```
