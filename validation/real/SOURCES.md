# Additional real-data validation sources

Research checked **2026-10-07**. This is a source catalogue, not a report of completed
benchmarks. The existing Tong et al. NIST IRSE/TSE data remain our baseline.

Distinguish **acquired complex k-space**, **raw NMR time-domain signals**, **reconstructed
measurement images**, and **derived maps / synthetic data**. All can be useful, but only the
first tests the complete imaging chain. A published fitted parameter is not automatically
independent ground truth for a forward simulation.

## Recommended shortlist

| Source | Verified data level | Size / access | What it could validate |
|---|---|---|---|
| [PulseqDiffusion](https://github.com/ritagnunes/PulseqDiffusion/tree/master/Example_Data) | Acquired complex phantom k-space in MATLAB | **21.1 MB** raw phantom file; public GitHub | Free diffusion attenuation, direction consistency, SE-EPI and receive/reconstruction effects |
| [Dynamic transitions CEST–Rex / T1](https://zenodo.org/records/17244380) | Siemens phantom `.dat` + matching `.seq`, calibration scans and repeat measurements | **31.6 GB** ZIP; CC BY 4.0 | CEST with transient recovery, inversion and imaging readout; B0/B1 corrections; held-out saturation times |
| [Pulseq-CEST MAGMA phantom](https://zenodo.org/records/15053086) | Reconstructed APTw / WASABI DICOM images | **53.6 MB** in two ZIPs; CC BY 4.0 | CEST/MT spectra and B0/B1 effects, initially without the full readout chain |
| [Tensor-valued diffusion phantoms](https://github.com/filip-szczepankiewicz/Szczepankiewicz_DIB_2019) | DICOM / NIfTI with actual diffusion gradient waveforms and detailed encoding metadata | Public GitHub, separable water/oil/HEX packages | Free/tensor diffusion under arbitrary waveforms and linear/planar/spherical encoding |
| [Generalized-Bloch NMR data](https://github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData) | NMR measurement files and acquisition parameters for BSA / MnCl2 | About **238 MB** uncompressed tracked files; MIT repository license | Short-pulse bound-pool dynamics, inversion recovery and the rate-model/generalized-Bloch difference |
| [Phantom GIRF measurements](https://zenodo.org/records/6376737) | Measured decay signals plus calculated GIRFs | **1.23 GB** total; **90.7 MB** GIRFs alone; CC BY 4.0 | Gradient impulse responses, eddy currents, drift/stability and hardware calibration |
| [NIST IR-bSSFP / B1 / reference SE data](https://zenodo.org/records/7654462) | BART complex arrays including explicit phantom k-space and separate B1/reference measurements | **61.4 MB** IR-bSSFP k-space + **18.7 MB** B1 file; CC BY 4.0 | Transient bSSFP, slice profile, B1 and independent T1/T2 checks |
| [Cross-vendor Pulseq workflow](https://zenodo.org/records/20725134) | Example raw-data conversion package plus sequence-generation and reconstruction tutorials | **7.85 GB** conversion ZIP; MIT record license | MPRAGE/EPI acquisition, vendor differences and raw-data import |

Sizes are decimal MB/GB; they describe selected files or archives, not required memory.
Only the diffusion MATLAB sample was read in full (in memory). For the large CEST archive,
the ZIP directory was inspected with small HTTP range reads; the 31.6 GB archive was not
downloaded. Other verification used primary papers, repository files and record metadata.

## 1. Diffusion: start with acquired Pulseq phantom k-space

[PulseqDiffusion](https://github.com/ritagnunes/PulseqDiffusion) by Nunes and colleagues is
the best small raw-data candidate found. At repository commit
`0517ae1656adf30f736bb3efe99552e7e421b9e4`:

- `Example_Data/kPhantom3dirs5b3slices.mat`: 21,095,722 bytes. Directly inspected: `k` is
  **complex64, shape (96, 8, 3672)**. This is acquired k-space, not an FFT of supplied images.
- `dim_struct` records five nonzero b-values, three directions, one b0, three slices,
  partial Fourier 0.75 and maximum b-value 1000 s/mm².
- `ImgPhantom3dirs5b3slices.mat` (3.34 MB) and `ADCPhantom3dirs5b3slices.mat` (0.637 MB)
  provide reconstruction/analysis comparisons; the ADC map is derived from these data.
- Python sequence generators and MATLAB reconstruction code are available. The repository
  license is AGPL-3.0; check the scope for data reuse rather than assuming a separate data license.

**Important gap:** no `.seq` files were present in the recursive repository tree. The current
single-SE generator has example settings (13 × 10 matrix, one nonzero b-value and full
Fourier) that differ from this acquisition. Do not regenerate its defaults and call that
the scanner's sequence. Recover the acquisition-era configuration, actual timing and phantom
properties before an absolute scanner-versus-simulator benchmark.

**Proposed first check:** reconstruct the supplied k-space using the documented ordering and
ghost correction, then examine ROI S(b)/S(0) across the three directions. For forward
prediction, obtain independent ADC/temperature information or explicitly calibrate on some
b-values and score others. This dataset alone does not establish cylinder radii or test
restricted diffusion.

## 2. CEST: a complete raw-data and sequence pair

Huemer et al., [Dynamic Transitions for Fast Joint Acquisition and Reconstruction of CEST–Rex
and T1](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.70198), uses an iohexol/Gadovist
tube phantom at 3 T. CEST saturation, inversion and stack-of-stars FLASH encode the transient
signal. The study also acquired WASABI, IR Look-Locker and conventional CEST references.

The paper's concept DOI [10.5281/zenodo.15768062](https://doi.org/10.5281/zenodo.15768062)
resolved during this search to **version 10.5281/zenodo.17244380**. Its `00_CEST_T1.zip`
is **31,553,548,401 bytes**, CC BY 4.0. An older indexed version, 16263825, is 12.3 GB;
do not confuse that size with the verified version.

The verified ZIP directory has 403 entries, including:

- `data/raw/phantom/`: Siemens `.dat` for the CEST experiment at saturation times 1, 1.5,
  2 and 4 s, WASABI, IR-LL and reference acquisitions.
- `data/raw/phantom_testretest/`: two sets of CEST / WASABI / IR-LL / reference `.dat` files.
- `sequences/phantom/`: corresponding `.seq` / `.mat` definitions, including
  `Seq_01_MPL_IR_LL_OC_15_100ms_1ms_4ppm_Phantom_Trec_0_IR.seq` and 10/20/40-pulse variants.
- `data/prepared/phantom/`: prepared HDF5 measurements and separate WASABI B0/B1 maps.
- Numerical phantom data and in-vivo data are in separate folders. Keep measured phantom
  data separate from synthetic data when constructing the benchmark.

**Recommendation:** use this for the eventual complete CEST imaging-chain benchmark.
Start with one phantom scan and its calibration/reference scans. Use independently
characterized tube properties where available, or label exchange-parameter fitting as
calibration; hold out saturation times and repeat scans for prediction. A central-k-space
time course is also a useful intermediate check before full image reconstruction.

## 3. CEST: a small spectrum-level benchmark

[MAGMA_2025 Phantom Data](https://zenodo.org/records/15053086) explicitly contains **DICOM**,
not scanner k-space:

- `PULSEQ_HYBRID_GRE_2_2_5_APTW_001_RR_0015.zip`: 27,471,352 bytes.
- `PULSEQ_HYBRID_GRE_2_2_5_WASABI_RR_0016.zip`: 26,088,134 bytes.

The [Pulseq-CEST Library paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12255581/)
describes the five-tube L-arginine phantom, including different concentrations and relaxation
conditions. Use the library's **`MAGMA_2025_PAPER` tag**, verified at commit
`0bd96f5a3ccbf30897417afee9eab32d15e3242b`, for figure scripts, preparation definitions and
matching phantom YAML files. The published release URL did not resolve as a GitHub release,
but the [tag tree](https://github.com/kherz/pulseq-cest-library/tree/MAGMA_2025_PAPER) exists.

**Recommendation:** easiest first CEST check: predict unsmoothed ROI Z-spectra, use WASABI
to characterize B0/B1, and compare normalized signal at the measured offsets. The library
defines the preparation; verify the hybrid GRE readout separately before claiming a complete
sequence replay. Model parameters distributed with the example need a provenance check
before being treated as independently measured constants.

## 4. Other diffusion sources

- **Arbitrary-waveform diffusion:** [Szczepankiewicz, Hoge and Westin 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6626882/)
  supplies water, oil and hexagonal liquid-crystal measurements with gradient waveforms,
  DICOM private-header tools, b-tensors and timing. This is particularly useful because
  b-values/directions alone do not define a restricted-diffusion experiment. These are
  reconstructed images, not scanner k-space. The HEX phantom is structurally more complex
  than a single uniform tensor; distinguish microscopic anisotropy from voxel-scale DTI.
- **Known restricted geometry:** [Fan et al. 2018](https://dash.harvard.edu/entities/publication/73120379-397a-6bd4-e053-0100007fdf3b)
  used hollow taxons with inner diameter **11.8 ± 1.2 µm**, multiple diffusion times and
  gradient strengths on the MGH CONNECTOM. NIfTI data, minimally corrected for susceptibility
  and eddy currents, were deposited in XNAT Central project **`dMRI_Phant_MGH`**. Strong
  candidate for cylindrical restriction, but current project/download access was not verified.
  Account requirements and waveform/timing completeness need checking before integration.
- **Small anisotropic/crossing-fibre dataset:** [Hansen and Jespersen, Dryad 9bc43](https://datadryad.org/dataset/doi:10.5061/dryad.9bc43)
  has a **23.16 MB** phantom ZIP, three orthogonal asparagus orientations, MATLAB diffusion
  signals, effective b-values, directions and DICOM/NIfTI. The repository calls these “raw
  dMRI data,” but scanner k-space is not established. Asparagus microstructure is not
  independently specified well enough for a simple cylinder ground-truth benchmark.
- **Acquired water/noise data:** [Zenodo 2483105](https://zenodo.org/records/2483105) has
  `acquired_water_phantom.zip` (**89.4 MB**) separately from synthetic `fiberfox_pmri` and
  `phantomas` packages. Useful noise-model lead; inspect the archive before asserting raw
  complex measurements or independent temperature-calibrated diffusion ground truth.

## 5. Highest-value additions beyond diffusion and CEST

1. **Pulsed MT / generalized Bloch:** [Assländer et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8810695/)
   provide MnCl2 control and 15% BSA measurements over inversion pulse durations, with
   native `data.2d`, acquisition files and analyzed recovery curves. The paper pins the data
   to `77950c4`; current `main` was `f038fff568f6fdd7ba354db28b5b91f4dbb26512`.
   This directly tests the short-intense-pulse limitation already identified in
   `generalized_bloch.py`. It is NMR validation, not imaging k-space.
2. **Measured GIRF / eddy currents:** [Wu et al., Zenodo 6376737](https://zenodo.org/records/6376737)
   has `Meas1.zip` (384.2 MB), `Meas2.zip` (755.6 MB), and `CalculatedGIRF.zip` (90.7 MB),
   with [processing code](https://github.com/BRAIN-TO/girfISMRM2022). Released decay signals
   average over receive coils; the supplied GIRFs were calculated from the unaveraged data.
   An additional [Python GIRF release, 15557750](https://zenodo.org/records/15557750)
   provides sequence/calculation code and Prisma 3 T / Magnetom 7 T examples (14.9 GB,
   BSD-3-Clause record license). Pin this version; the concept record can resolve to a
   different release and scanner. Calibrate on one measurement and predict another.
3. **Transient bSSFP with B1/reference scans:** [Scholand et al., version 7654462](https://zenodo.org/records/7654462)
   includes `data_05b_kspace.cfl/.hdr`, `data_05b_b1map.cfl/.hdr` and separate NIST reference
   SE/IR-SE acquisitions. [Reproduction scripts](https://github.com/mrirecon/bloch-moba)
   explain the BART arrays. This addresses the need for companion B1/reference measurements
   beyond our current NIST benchmark; verify the provenance of each reference/map array.
4. **Cross-vendor MPRAGE/EPI:** [Chen et al.](https://onlinelibrary.wiley.com/doi/10.1002/mrm.70548)
   link the verified [workflow package 20725134](https://zenodo.org/records/20725134).
   The 7.85 GB conversion tutorial has example raw data; the acquisition tutorial provides
   sequence generators. Check which phantom/vendor examples are actually released before
   treating all nine scanners in the paper as downloadable raw benchmarks.
5. **Susceptibility / T2\*:** [7 T phantom data, 5120627](https://zenodo.org/records/5120627)
   has a 31.9 GB `Data_7T_mix.zip`, masks and documentation for dia-/paramagnetic
   substructure. Potential test of subvoxel susceptibility/dephasing; “raw” in the title
   is not enough to establish scanner k-space without inspecting the archive.

## Sources that should not be mistaken for new raw phantom benchmarks

- [HarmonizedMRI/Pulseq-diffusion](https://github.com/HarmonizedMRI/Pulseq-diffusion) and its
  [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11055665/) share sequence/reconstruction
  code and report a physical calibration phantom, but this search did not establish a
  public download of that phantom's acquired k-space.
- [CEST–MRF protocol record 14211517](https://zenodo.org/records/14211517) and
  [molecular-mrf](https://github.com/momentum-laboratory/molecular-mrf) include acquired
  signal/image examples and sequences; the protocol uses “raw MRF” for DICOM/NumPy inputs.
  Inspect individual arrays before classifying them as k-space. Derived maps and trained
  networks are not independent physical ground truth.
- [Free-breathing DWI raw k-space, 10607734](https://zenodo.org/records/10607734) is real
  diffusion raw data, but [its README](https://github.com/ecat/fbdwi) identifies volunteer
  breathing experiments, not the controlled phantom sought here.

## 6. Further sources (second pass, 2026-10-07)

Checked from record pages / repository READMEs only; nothing downloaded.

| Source | Data level | Size / licence | Why |
|---|---|---|---|
| [Pulseq multi-slice radial 2D phantom, Zenodo 14228459](https://zenodo.org/records/14228459) | `.seq` (golden-angle, 402 spokes, 5 slices) + acquired ISMRMRD HDF5 with trajectory | 38.4 MB; CC BY 4.0 | Smallest complete sequence + raw-data pair; non-Cartesian timing, gradient delays / GIRF |
| [2D multi-echo spiral TSE of the T1MES phantom, Zenodo 18749100](https://zenodo.org/records/18749100) | `.seq` + Siemens `.dat` + ISMRMRD (with and without trajectory) | 35 MB `.dat`, 0.7 MB `.seq`; record 2.9 GB in all; CC BY 4.0 | Second echo-train benchmark after the Tong TSE, with spiral readouts (GIRF) and a phantom with published reference T1/T2 (T1MES, Captur et al. 2016; values not yet checked). A `.dat` usually carries the noise-adjust scan (unverified) |
| [Pulseq QA package, Zenodo 14217778](https://zenodo.org/records/14217778) | `QA_epi.seq`, `QA_T1.seq`; Siemens `.dat`, GE `.mat`, ISMRMRD; fBIRN phantom on Trio / Prisma.Fit / Cima.X / GE UHP, all 3 T | 25.4 GB (20.4 GB raw ZIP: range-read single members); CC BY 4.0 + MIT | Same sequence on four scanners: scanner-to-scanner differences the simulator should explain (EPI ghosts, distortion, GIRF) |
| [RRSG 2020 T1 challenge](https://rrsg2020.db.neurolibre.org) | IR images of the NIST phantom from many sites | – | Inter-site CoV ~6 %: the yardstick for "agrees with a scanner as well as scanners agree with each other" |
| [Multi-site fat-water phantom, Zenodo 48266](https://zenodo.org/record/48266) | Complex multi-echo images (3 central slices), 6 sites, 1.5 and 3 T, two protocols each; reference PDFF verified by Calimetrix | MATLAB; open | Multi-peak fat spectrum and chemical-shift encoding (images, not k-space) |
| [QSM challenge 2.0 head phantom](https://pmc.ncbi.nlm.nih.gov/articles/PMC8048665/), doi:10.34973/m20r-jt17 | Digital: chi (two models), R1, R2*, M0, 16-tissue segmentation, 0.64 mm and 1 mm; simulation code | CC BY 4.0; Radboud/Donders repository (account requirement unverified) | Input for B0-from-susceptibility (dipole convolution) and realistic anatomy; not validation data |
| [mariepy](https://github.com/pulserver/mariepy) (port of MARIE 3.0) | Tool: full-wave B1+/B1-, E fields, absorbed power and noise covariance for voxel bodies (BrainWeb) in body coils, loop arrays, surface coils; `.npz` output | GPL-3.0 (run as an external tool, do not vendor) | The full-wave EM maps the `FieldMaps` importer needs at 3 T and above |

Corrections to the ranking above, given what the simulator already has:

- **Scholand et al. 7654462 has the NIST Model 130 T2 plate** with one radial IR-bSSFP (1 ms
  pulses), a preconditioned-pulse TurboFLASH B1 map and SE / IR-SE gold standards; the RF
  pulse-duration variants and IR-FLASH are volunteer brain scans (corrected after download).
  Now benchmark 3 (`RESULTS.md`): it confirms the slice-profile physics on spheres 2–8 but cannot
  reach T2 <= 33 ms, so it does not settle the short-T2 question.
- **Free diffusion in a water bottle adds little** beyond the exact Stejskal-Tanner tests; the
  value of PulseqDiffusion is the imaging chain (eddy currents, EPI ghosts), which needs the
  acquisition `.seq`. R. G. Nunes is a co-author of the Tong data already used: ask for it.
- **Large ZIPs are not a download problem**: single members can be extracted with HTTP range
  reads (as the directory was), so the 31.6 GB CEST and 20 GB QA archives are usable per scan.

## 7. What is not openly available, and how to estimate it

| Missing | Route | Status of the result |
|---|---|---|
| Gradient nonlinearity coefficients (vendor-confidential) | Fit spherical harmonics to fiducial displacements in a grid / distortion phantom scanned with the vendor correction off (two readout polarities separate it from B0); or the spatial b-value bias in an ice-water DWI phantom (ADC known at 0 °C). Without a scan: synthetic coefficient sets within published magnitudes | Calibration of one scanner, or explicitly "plausible", never "vendor X" |
| ADC filter response | Spectrum of noise-only (noise-adjust) samples: the squared filter magnitude directly | Measurement |
| Played vs written timing, gradient delays | Fit global delays and GIRF parameters to calibration scans with the differentiable field model; check on held-out scans (radial / spiral pairs above) | Calibration + held-out prediction |
| Receive noise covariance, coil maps, g-factor | Noise-adjust scans give the covariance; ESPIRiT on any multi-coil raw data gives coil maps; pseudo-replica g-factor (Robson et al. 2008) on the same data | Measurement, for comparison with `loops.py` structure |
| B1+ of real anatomy at 3 T and above | mariepy on BrainWeb; measured phantom B1 maps (7654462) to check the method | Simulation, checked on phantoms |
| Flow with its sequence | Analytic (Poiseuille, first-moment phase) remains the reference; published PC-phantom accuracies as a sanity range | Analytic |
| J-coupled fat | Own density-matrix reference (Stables et al. 1999) + published SE vs TSE fat ratios; lipid shifts and J from the spectroscopy literature | Physics reference |

Rule: a parameter fitted to scans is calibration. Report the fit separately and score only
held-out scans or quantities.

## Proposed order of work

1. Scholand NIST T2-plate IR-bSSFP + B1 map (short-T2 question).
2. Assländer BSA / MnCl2 NMR (generalized Bloch, using the paper's fitted parameters).
3. The radial (14228459) and spiral TSE (18749100) Pulseq pairs (non-Cartesian timing, GIRF,
   noise-adjust spectrum for the ADC filter).
4. Measured GIRFs (6376737): calibrate on one, predict the other.
5. CEST: MAGMA Z-spectra, then one raw iohexol scan with held-out saturation times.
6. Diffusion once the PulseqDiffusion `.seq` is recovered; the taxon phantom for restriction.

mariepy and the QSM head phantom feed the next physics items (EM maps, susceptibility B0)
rather than validation.
