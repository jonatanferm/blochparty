# 2026-10-13 — M6.1 J-coupled fat: the model, and why it is not the bright-fat story

- `blochparty.jcoupling` (density matrices of coupled spin fragments under single spin echoes or CPMG trains; ideal
  pulses, strong-coupling Hamiltonian) and a triglyceride fragment model whose composition reproduces the simulator's
  six-peak spectrum (methyl 8.7 %, 2.1 ppm 12.8 %, diallylic 0.4 %, 5.3 ppm 4.8 %: 1.9 double bonds, 0.2
  methylene-interrupted). Tests: no coupling or equivalent spins give no modulation; a weakly coupled AX pair's echo is
  |cos(pi J TE)| to 1 % (the strong-coupling correction, J / dnu); a 2 ms CPMG train keeps a 0.4 ppm pair above 0.95
  where one echo at 100 ms is below 0.8.
- Against measured 3 T fat (validation/jcoupling_fat.py): calibrated to CPMG 145 ms at 8 ms (Keene 2020), the model
  needs an intrinsic T2 of 266 ms and predicts single-echo fat at 222 ms where 48 ms (Jordan 2013) and 68 ms (de
  Bazelaire 2004) are measured; fat at TE 100 ms only 1.13 x brighter in a 10.26 ms train than in one echo (Henkelman
  1992: up to ~2). About half the fat signal is bulk methylene with same-shift neighbours, unmodulated in the model,
  and the coupled groups' modulation partly recovers by TE 160. So the in-vivo SE / CPMG fat difference needs more
  (the methylene chain's many-spin dynamics, or what else fast refocusing suppresses); no echo-spacing fat T2 is
  shipped to the simulator. For the realism question: at ~113 ms the J modulation is 0.687 with 8 ms spacing and
  0.663 with fastMRI's 10.26 ms, so J coupling would make that protocol's scalp fat ~3.5 % dimmer than the CPMG value,
  not the 12-15 % brighter the side / front scalp reads.
