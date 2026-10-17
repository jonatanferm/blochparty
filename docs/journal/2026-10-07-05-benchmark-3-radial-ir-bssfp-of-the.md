# 2026-10-07 — Benchmark 3: radial IR-bSSFP of the NIST T2 plane (Scholand et al. 2023)

- Downloaded Zenodo 7654462 (phantom k-space, vendor B1 map, Cartesian IR-SE / SE gold
  standards; MD5-checked) and the authors' ROIs (bloch-moba, pinned). Correction to the source
  catalogue: the RF pulse-duration variants are volunteer scans; the phantom has one IR-bSSFP.
- No `.seq`: rebuilt the sequence in blochparty from the paper, the authors' scripts and BART's
  simulator (HS inversion with BART's parameters, alpha/2 preparation, Hamming sinc, Skyra-like
  trapezoids); 10 us RF raster is 3e-4 from 1 us; GPU = core to 2e-5.
- Measurement side (`scholand_recon.py`, numpy + finufft): trajectory conventions recovered from
  BART's source and checked on images; anisotropic gradient delays from antiparallel spokes
  (0.64 / 0.78 samples, 0.015 rms); sign from data consistency (0.23 vs 0.78 residual); coil
  compression, ratio coil maps, rank-K subspace CG reconstruction; noise covariance from k-space
  edges. Gold standards refitted with complex coil combination: agree with the authors for
  spheres 1–10 (1–3 %); spheres 11–14 are not measurable with a 15 ms first echo (authors'
  735 ms for sphere 14 = magnitude noise floor). B1 map: low in short-T1 spheres; smooth fit used.
- Dead ends and corrections: (1) assumed the two ROI sets share an order, then a geometric
  mirror mapping — both wrong; the radial set is reversed (signals and the authors' table agree;
  geometry is ambiguous for a ring); (2) the first basis (non-selective signals only) biased the
  digital-phantom fits by up to +41 % T2: truncation; a union basis with slice-selective signals
  fixed it (spheres 2–8 within ±3 % T1, ±11 % T2); (3) stated the B1-map bias for long-T1
  spheres before the ROI order was fixed: it is the short-T1 ones.
- Result (full model, spheres 2, 3, 5, 6): T1 −4 to −10 %, T2 −1 to −14 % vs gold; without the
  slice profile T2 +35 to +55 %; B1 map ~10 % in T2; HS vs perfect inversion ≤ 2 %. Residuals
  4–9 % vs 1.4–2.3 % on the digital twin: an unexplained remainder (B0 unmeasured, coil maps,
  trajectory beyond linear delays, the rebuilt sequence). Spheres 9–14 are out of reach of the
  pipeline, so the short-T2 question from benchmark 2 stays open.
