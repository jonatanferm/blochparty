# 2026-10-09 — M5: tissue schema, head sampler, dataset writer, all 20 BrainWeb subjects

The parts of M5 that need no outside access, done while the user was away.

- **Schema (P0).** Every signalling BrainWeb class now has cited MT and diffusion, so
  `to_phantom(mt=True, diffusion=True)` works on a whole head (it raised for everything outside
  WM/GM/CSF, and those had illustrative placeholders: f 0.16/0.07, exchange 40/s, T2b 12 us). At
  3 T: WM/GM bound fraction 0.1277 ± 0.0041 / 0.0673 ± 0.0015 (Yarnykh 2016, 8 subjects),
  bound-to-free rate 19.0 /s and T2b 9.7 us (Yarnykh 2012's whole-brain medians); muscle f 0.084 ±
  0.006 and T2b 5.9 ± 0.2 us (Sinclair 2010), its rate assumed (it did not converge in muscle);
  fat and CSF without a semi-solid pool (Chen 2010; CSF MTR 0.01). Diffusivity: WM 0.70, cortical
  GM 0.89 (Helenius 2002), CSF 3.04 (water at 37 C, Holz 2000), muscle 1.78 (Li 2014), blood 1.54
  (Funck 2018) um²/ms, fat 1.83e-5 mm²/s (an upper bound, Weidlich 2018). Skin, connective tissue
  and dura take muscle's values, marrow fat's, flagged; qMT measured at 3 T is flagged at 1.5/7 T.
  The k convention was the trap: Yarnykh's k is free -> bound and his R = k(1-f)/f is our
  `BoundPool.k` (LESSONS). Default `to_phantom` on subject 04 has the same SHA-256 before and
  after; core snapshot bit-identical. `DigitalPhantom` (anatomy, one field's table, `to_phantom`
  options, `meta`) gives `to_phantom` bit for bit.
- **Sampler (P1).** `sample_head(seed)`: one of the 20 subjects, a cubic B-spline displacement
  (control points N(0, (3 mm)²) every 40 mm: ~1 mm SD, Jacobian 0.83-1.21 on subject 04, halved
  until > 0.2), a per-subject table (`draw_tissues`: log-normal with each cited mean and SD; 1800
  standardised draws over 100 subjects have mean -0.03, SD 1.00), the M4 texture and a spin seed,
  all from one `SeedSequence`; `meta` holds every draw and is JSON. Checks: integer shifts exact,
  B-spline weights sum to 1 (1e-15), an affine field's Jacobian equals det(I + A) to 1e-12, equal
  seeds give equal heads. 4 s per head at 2 mm, 16 s at 1 mm with the full-head B0.
- **Dataset writer.** `io.write_dataset` writes ISMRMRD HDF5 through ismrmrd-python (new extra
  `io`): one acquisition per ADC run with its k-space (`sequence.kspace_trajectory`, the M4 test's
  rule moved into the library and vectorised; still equal to pypulseq's `calculate_kspace`),
  encode indices, centre sample, ACQ_IS_REVERSE for EPI's reversed lines; a `/provenance` group
  with the sequence arrays and tagged-JSON meta, the `.seq` file, the head's table and draws,
  coils, noise, seed, engine, git commit and versions. Round trip on GRE/TSE/EPI with 4 coils:
  signal equal to its complex64 cast, k-space assembled from ISMRMRD fields alone equal to
  `cartesian_kspace`, the rebuilt sequence equal and writing the same `.seq` text.
- **All 20 BrainWeb subjects** fetched on the mini (~1.4 GB each) and pinned. The raw files have no
  header and the fetcher writes subject 04's origin for all; the brain centroids agree to 2.6 mm,
  as a shared origin requires. WM 618 ± 42, GM 969 ± 55, CSF 385 ± 43 mL over subjects.
- Full suite on the Mac mini: 701 passed in 3 min 56 s.
- **Example** (`examples/sampled_heads.py`, Mac mini): seeds 0-9 as T1w GRE, T2w TSE and T2*w GRE at
  3 T with scanner-playable sequences, 8 coils and noise; 8.8-10.0 million spins per head, 44-58 s
  per image on Metal, 27 min for 30 ISMRMRD files (148 MB). The first run's GRE files had no `.seq`:
  a round TE leaves a block half a dwell off the 10 us raster (LESSONS); TE 2.505 / 20.005 ms fixed
  it and the run was repeated, so the figure comes from the committed script. `--replot` redraws
  it from the files through `read_dataset`.
- **Left for a decision** (ROADMAP M5, outside the brain): skin/connective/dura relaxation are
  muscle's, marrow's is the legacy fat prior, adipose tissue is pure lipid; changing any moves
  default `to_phantom` output. The realism number needs fastMRI access.
- Commits: 99ef90a (schema values), ecc1ddb (DigitalPhantom, sampler), ae55fae (writer,
  `kspace_trajectory`), 69b2301 (20 subjects), 585e857 (example).
