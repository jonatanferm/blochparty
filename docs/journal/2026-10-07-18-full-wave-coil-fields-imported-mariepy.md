# 2026-10-07 — Full-wave coil fields imported (mariepy)

- `src/blochparty/em.py`: reads mariepy map files (plus / minus per channel, exp(+j w t), grid in
  x y z) into `FieldMaps`: B1+ = conj(minus)/2 (quadrature drive chosen by size), |B1+| / median;
  receive conj(plus)/2; nearest-inside fill at the skin; noise resistance reader.
  `tests/test_em.py`: rotating-field phase (three phases), quasi-static limit equals
  `loops.field_maps` to 1e-5, orientation and fill. Suite 477 passed.
- `validation/em/run_brainweb.py` (mariepy env, 1b0dc435, 14 min on CPU): BrainWeb 04 at 5 mm,
  3 T. Birdcage: co-rotating mode 1.09 vs 0.17 uT; |B1+| 0.90–1.37x median, centre 1.38x. Eight
  loops: 1.27–1.75 Ohm, centre-sampled E 2–3 % below exact losses (rescaled), noise correlation
  up to 0.40.
- `validation/em/brainweb_demo.py`: T1w GRE with birdcage B1+ and the eight-loop array with
  correlated noise (RSS, Roemer), 37 s; registration check -0.995 (vs -0.93 transposed).
- Limitation kept: `FieldMaps.b1` is real, so transmit phase (pTx) is dropped; roadmap item.
