# 2026-10-07 — Receive phase of loop coils fixed (found while importing full-wave maps)

- Mapping mariepy's circular components onto blochparty showed that nothing in the repo fixed the
  rotation sense of coil phases. A Faraday's-law test (new, `tests/test_loops.py`) shows the
  receive weight is (Bx - iBy)/2; `loops.receive_sensitivities` returned its conjugate, mirroring
  coil phase patterns (magnitudes, RSS and B1+ unaffected). Fixed; defaults unaffected (opt-in).
- On the way, mariepy validated here: B1 maps equal Biot-Savart to 0.3 % at 0.01-0.1 T and depart
  by 14-16 % at 3 T (wavelength); interior E vs Mie for a conducting sphere at 3 T 5.2 % (5 mm) and
  2.5 % (2.5 mm) with the linear basis, 26 % / 10 % with the constant one; port resistance equals
  absorbed power to < 1 %. Noise resistance of a coaxial loop: constant basis wrong at low field
  (no omega^2 scaling), linear basis within 6-9 % of quasi-static at 0.1-1.5 T and +48 % at 3 T
  (converged): quasi-static noise covariance underestimates head-size losses at 3 T.
