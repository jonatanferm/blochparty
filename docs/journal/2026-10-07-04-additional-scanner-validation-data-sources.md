# 2026-10-07 — Additional scanner-validation data sources

- Searched primary dataset records, papers and repositories for diffusion and CEST phantom
  raw data and complementary validation of implemented physics. Added
  `validation/real/SOURCES.md` and linked it from the existing real-data README.
- Directly read PulseqDiffusion's 21.1 MB phantom MATLAB sample in memory: complex64 k-space
  `(96, 8, 3672)`, 8 channels, 3 slices/directions, 5 nonzero b-values, bmax 1000 s/mm²,
  partial Fourier 0.75. Its current sequence-generator settings differ from the scan, and
  the repository contains no `.seq`; acquisition provenance and independent phantom
  properties remain prerequisites for a forward-prediction benchmark.
- Verified iohexol CEST raw data + full `.seq` definitions, WASABI/IR-LL references and
  repeat scans in Zenodo version 17244380. Inspected the 403-entry ZIP directory using
  small range reads; did not download the 31.6 GB archive. The small MAGMA APTw/WASABI
  release is 53.6 MB of reconstructed DICOMs, useful for spectra but not raw k-space.
- Also catalogued tensor-valued diffusion waveforms/phantoms, a known-diameter taxon phantom
  (current XNAT access unverified), BSA/MnCl2 NMR for generalized Bloch, measured GIRFs,
  NIST IR-bSSFP with B1/reference scans, and a cross-vendor MPRAGE/EPI package. Distinguished
  synthetic data, derived maps and in-vivo-only raw data from measured phantom sources.
- Research/documentation only: no simulator changes or new benchmark results; no bulk data
  retained. Recommended beginning with the small diffusion/CEST sources and resolving their
  metadata gaps, then adding MT/GIRF checks and a held-out raw CEST benchmark.
- Second pass: added the radial and spiral-TSE Pulseq phantom pairs (`.seq` + raw), the Pulseq
  QA package (one sequence, four 3 T scanners), RRSG 2020 T1 data, the multi-site fat-water
  phantom, the QSM 2.0 head phantom and mariepy (full-wave EM maps). Re-ranked: Scholand
  7654462 is the NIST T2 plate with B1 map and pulse-duration variants, so it goes first (the
  short-T2 question). Added routes for what is not public (nonlinearity from grid phantoms,
  filter response from noise spectra, timing by fitting with held-out scans).
