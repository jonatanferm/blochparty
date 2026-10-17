# 2026-10-07 — Anatomy M3: QSM loader and public challenge comparison

- Continued M2 `7a87c19`. Added optional nibabel loading through `load_qsm2`, pinned local
  manifests, 16-label memberships, NIfTI affine permutation/flip handling, ppm→SI/mm→m
  conversion, explicit exterior susceptibility and acquisition field. Reject contradictory
  geometry/units, unknown labels, invalid rates, changed hashes and silent field scaling.
  No engine changes. Full-head files require a locally supplied authorized package.
- Corrected a source assumption: Radboud version 1 metadata declares restrictedAccess and
  RU-DI-HD-1.0, not CC BY. Asked for a local copy while continuing implementation. The user's
  Stage-1 bit.ly link led to a closed Google registration form. Found the challenge author's
  public SURFdrive share via Berkin Bilgic's lab page; read-only directory listings avoided
  downloading its 12.65 GB archive. Retrieved both chi/segmentation ground truths (~107 MB)
  and exploratory frequency/mask files; all volumes stay git-ignored.
- Public GT shapes are (164,205,205) x/y/z with 0.999024391-mm headers. Frequency/mask
  images use exactly 1 mm; do not silently assume equal grid alignment. The latter are
  estimates from simulated noisy multi-echo images and are not exact forward references.
  Public challenge maps omit background susceptibility and contain no R1/R2*/M0.
- Independently executed the two pinned public qsm-forward functions (commit
  `92149383b0f43f47d9a351ceb4f525c0c4ee8299`), unchanged, on both downloaded chi maps:
  3 T, z B0, 2× padding, 1,469,325 brain voxels. Sim1/Sim2 RMS 2.11e-16 / 3.76e-16 Hz,
  maxima 4.47e-15 / 1.38e-14 Hz, after subtracting constant reference differences
  −4.19232e-5 / −1.15128e-5 Hz. Synthetic water/air cases exposed D(0)=1/3 in the reference
  vs zero here (~375 Hz offsets), with <6.83e-13 Hz maximum residual after only demeaning.
  The Python reference is based on Marques 2021; it is not the original MATLAB package.
- Clarified M0 semantics: measured M0 is total voxel density, apportioned among signalling
  memberships; multiplying it by tissue/air fraction again loses mass. Coarsen rates with
  M0 weighting so air cannot lengthen T1. This preserves density volume but remains an
  approximation for multi-exponential relaxation. New synthetic fixtures check these
  invariants and geometry/provenance failures; both real GTs pass fetch/hash/load checks.
- M3 remains open for actual full-head R1/R2*/M0 ingestion and its original MATLAB forward
  check. Their unavailability is explicit in `validation/anatomy/QSM.md`; public-data
  checks do not stand in for those unexecuted steps. M4 and M5 have not started.
- Final validation: **462 passed, 4 opt-in crosschecks deselected** with host Apple GPU
  access in 184.47 s. Thirteen QSM tests are new. Before/after default outputs and
  gradients are bit-identical; both pinned public fetch/load checks pass.
