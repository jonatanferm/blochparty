# Lessons: Reference data, datasets, formats and other simulators

- **Check what a reference table's columns refer to, not just the numbers.** Jiang et al.
  2016 (PMC5408299) Table 1 gives T1 of the NiCl2 spheres *and T2 of the MnCl2 spheres*;
  benchmark 1 first used those T2 values for the NiCl2 spheres. Nobody tabulates the cross
  terms (T2 of NiCl2, T1 of MnCl2): derive them from the 3 T relaxivities (Martin et al. 2023):
  R2 = R2_w + (r2/r1)(R1 − R1_w). The derived MnCl2 T1s (2454 → 85 ms) land in the published
  80–2480 ms range they weren't fitted to. (`validation/real/nist.py`)

- **NIST phantom planes are mirror images**: T1-plane spheres 1–10 run clockwise on the outer
  ring (−36° steps), T2-plane spheres counter-clockwise; spheres 11–14 on the inner ring
  continue next to sphere 10. Identify spheres by geometry and confirm by decay rates, not by
  a T2 fit: the authors' fitted T2 map is ~2 s almost everywhere on that plane.

- **Multi-echo spin echo with imperfect refocusing**: echo 2 is ~20 % brighter than echo 1 in
  long-T2 spheres (stimulated-echo pathways), which is why T2 fits commonly drop the first
  echoes. (`validation/real/nist_tse.py`)

- **"Discard the first N excitations" may mean "use the other slice".** The NIST TSE file is a
  two-slice acquisition (x = +12 mm in pass 1, −12 mm in pass 2); the authors' recon keeps
  pass 2. A phantom on the first slice found in the file is never excited in the imaged pass:
  near-zero signal whose leftovers looked like growing, mis-centred echoes (61 % error that no
  spin density fixed). Enumerate all slice positions (`nist.slice_positions`) and choose the
  imaged one explicitly; sweep a pulse across x per pass when in doubt.

- **When the simulation plateaus away from reality as spins increase, it's a model or setup
  error, not convergence.** Diagnose by stripping down (one TR, spins only at the slice
  centre: textbook CPMG decay is the expected answer). Core and fused agreeing proves the
  kernels, not the inputs.

- **MATLAB-saved arrays reshape with `order="F"`** to reproduce MATLAB's `reshape`.

- **A sphere-radius estimate from sphere/fill contrast fails when contrast is weak** (T2 plane:
  0 mm and 9–10 mm vs 6–8 mm on the T1 plane); reuse the better-conditioned plane's geometry.

- **NIST phantom references**: T1 of the NiCl2 array at 3 T / 20 °C for version 1 (serial < 43)
  and 2 (≥ 43) are in the RRSG 2020 code (`rrsg2020/analysis`, `src/nist.py`); version 2 and
  T2 of the MnCl2 array are the phantom-manual values in Carr et al. 2021 (Table 2); 3 T
  relaxivities for the cross terms in Martin et al. 2023 (Table 1). All in
  `validation/real/nist.py`. The Tong et al. dataset records neither serial nor temperature;
  v2 fits it better (1.5 % vs 2.8 %).

- **To test a reference value, give the nuisance its own free amplitude** (benchmark 2,
  2026-10-06). The short-T2 spheres' misfit mixed two things: water blurring in (a flat tail
  whose level depends on geometry we only know roughly) and the decay itself. Fitting
  real ≈ a·sim(T2) + b·e^(−TE/T2_water) per sphere removes the geometry and leaves the T2
  question: the reference values held (scaling T2 by 1.25 made 13 of 14 spheres worse), and
  the remaining 6.5–8.7 % on spheres 11–14 sits in the first echoes. Before blaming the
  reference values, also check that a measured tail isn't noise: σ from the air (Rayleigh
  mean/std ≈ 1.9) put these tails 7–12σ above the floor.

- **Scholand IR-bSSFP data (Zenodo 7654462), facts not in any document** (benchmark 3,
  2026-10-07): BART trajectory `traj -x 384 -y 1 -s7 -G -t 1000 -D -r`: spoke n at n pi / (phi + 6)
  (tiny golden angle 7; `-D` does not double golden angles since BART v0.4); samples at
  (i - 192 + 1/2) / 2 base-grid units. Measured spokes sit at -d(theta) along the readout (the
  other sign triples the data residual). The receiver phase follows the alternating RF phase.
  The B1 map (304 x 192 asymmetric echo; k-space centre at sample 111 of 384) uses an 80 deg
  preparation (hard-coded in the authors' script).

- **The authors' two ROI sets are ordered differently**: the Cartesian set lists sphere 1 first,
  the radial set sphere 14 first, as their reference table does. Geometry cannot tell: the
  spheres form a near-symmetric ring, so a 123 deg rotation and a mirror both match the centres
  to 0.55 mm. The signals can: a 14 x 14 matrix of curve-vs-simulation residuals is diagonal
  only for the reversed order. First tried the mirror, and every result was nonsense.

- **Vendor preconditioned-TurboFLASH B1 maps read low in short-T1 compartments** (0.87 against a
  smooth 1.09 in a T1 = 0.13 s sphere): T1 recovery during the readout raises the prepared
  image. In a phantom, fit a smooth field to the long-T1 background (agrees with long-T1 spheres
  to 0.02).

- **A single-echo SE series with first TE 15 ms cannot measure T2 < 15 ms**, and the magnitude
  (RSS) fit then returns the noise floor: 735 ms for a 5 ms sphere in the published table.
  Combining coils with a complex reference and fitting the real part removes the floor (25 ± 11 ms:
  unbiased, but still uninformative).

- **Asslaender NMR data**: Magritek `data.2d` = 12 header bytes, int32 501 (complex), size1,
  size2, 1, 1, then complex float32, column-major; the authors trim 6 points at each end. Column 1
  is the *shortest* inversion time (3 ms), although their T2* comment says 5 s. TI carries a
  timing correction 12 x 22.8 us + (13 x 15.065 − 5) us and is counted from the pulse centre.

- **PTB spiral TSE (Zenodo 18749100)**: the ISMRMRD trajectory is in matrix-index units
  (+-80 for 160) and each readout spirals out from the k-space centre at sample 0 — the header's
  `center_sample` 45 is wrong. Each echo of a TR uses a different spiral rotation.

- **T1MES reference values are from a 1.4 T relaxometer** (Captur et al. 2016, Table 2); the
  shared tubes have no 3 T values, and agarose T2 depends on field and echo spacing (here 35–50 %
  below the references). Not an independent 3 T T2 reference.

- **Legacy simulator tissue values are not tissue values** (anatomy, 2026-10-07): BrainWeb's
  MRISIM file gives CSF T2 0.33 s (measured at 3 T: ~2 s) and T1 2.57 s (~4.4 s); used at 3 T it
  made T2-weighted CSF dark, the most visible error in a brain image. Flags marked it as an
  assumption, but nothing stopped it being the default — check defaults against images.

- **A measured T2* is not an intrinsic property once the simulation computes fields**: it
  includes the macroscopic dephasing at the measurement's voxel size and echo times. Separating it
  needs those acquisition parameters (AHEAD: MP2RAGEME, TE 3/11.5/19/27.5 ms); the round trip
  (simulate the acquisition, fit R2*, compare with the input) is the test, and a field strong
  enough to exceed R2* - R2 on its own means the inputs disagree (clip, do not error). Fixing it
  exposed the next issue at once: the occipital WM/GM T2* proxies make cortex dark at long TE.

- **Quasi-static coil checks cannot fix the rotation sense; Faraday's law can** (2026-10-07):
  for real (quasi-static) fields |Bx + iBy| = |Bx - iBy|, so magnitude maps, RSS images and a
  comparison against a full-wave solver at low field all pass with either sign, and a test along a
  single axis is phase-trivial. `loops.receive_sensitivities` returned conj(B1-) and so mirrored
  every coil's phase. A brute-force lab-frame Faraday calculation (precessing dipole, numerical
  -dPhi/dt, demodulation by e^{+i w0 t}) divided by the simulated coil x Mxy must be one constant
  across positions: 2 i w0 with (Bx - iBy)/2, a 137 % spread with the conjugate. With full-wave
  phasors (exp(+j w t)) the same derivation gives transmit conj(minus)/2, receive conj(plus)/2.

- **Full-wave map import** (2026-10-07, `em.py`, `validation/em/`): (1) mariepy leaves which
  circular component excites to the reader; for protons with B0 along +z it is `minus`
  (clockwise), conjugated into blochparty's rotating frame. A birdcage's two quadrature drives
  (1, +-j) differ by ~6x in mean |B1+| on a head (1.09 vs 0.17 uT per uT), so picking the wrong
  sense is obvious in magnitude, but a wrong conjugation is not: test against a rotating field
  with a non-zero phase. (2) The solver stores zeros outside the body; trilinear sampling at the
  skin then mixes in zero B1 / coil weight, so fill the outside with nearest-inside values. (3)
  Noise resistance from E sampled at voxel centres falls 2-3 % below mariepy's exact absorbed
  power (piecewise-linear basis) at 5 mm; rescale the matrix so its diagonal matches. (4) A
  registration check that survives near-symmetric maps: correlate the image ratio (B1 map on /
  off) with the sampled map and with its transpose and mirror (-0.995 vs -0.93 / -0.97).

- **Read the table, not the abstract** (2026-10-08): Shin et al. 2025's abstract gives "white
  matter and gray matter 55.9 ms and 51.5 ms"; Table 3 (columns GM, WM) and every per-subject row
  give GM > WM. Values for the table come from tables, cross-checked against a second source
  (Peters, ISMRM 2006, Table 1 in the PDF: GM 59.7 > WM 54.6 ms at 3 T). A tissue T2* also needs
  its source's voxel size and echo times: the macroscopic part it contains depends on them, and a
  longer echo train (3-66 vs 5-40 ms) clips R2' at gradients half as large.

- **When only the abstract is readable, test its numbers against physics before using them**
  (2026-10-10): Wansapura et al. 1999 (JMRI 9:531) gives T2 "for gray matter and white matter"
  as 80 and 110 ms, opposite to every other 3 T study (GM > WM, Stanisz 2005 99 / 69 ms), and
  also calls its 14-echo FLASH T2* values "T2" and an occipital GM value that is the lower one
  "significantly higher". `validation/anatomy/anatomy_brainweb.py` used the abstract order since M2
  (2026-10-07); its human-3 T T2-weighted GM/WM ratio was 0.69 (cortex darker than white matter,
  which no T2-weighted image shows) and is 1.56 with GM 110 / WM 80 ms. A predicted contrast
  ratio that inverts the textbook appearance is the quick check. The Wiley full text sits
  behind a bot check and ResearchGate returns 403, so the assignment rests on physics; the
  tissue database (`tissuedb` branch) flags the same rows.

- **Library phantom models are not ground truth**: the Pulseq-CEST L-arginine YAMLs (comments:
  "simplified", MT size "from comparison with measurement") miss their own phantom by rms
  0.05-0.12; the engine meanwhile matches BMCTool to 8e-4. Separate engine checks from model checks.

- **Prefer the raw data over a dataset's prepared arrays when the bookkeeping does not add up**
  (2026-10-08, benchmark 9): the iohexol archive's prepared Z-spectra had 57 columns for 58
  offsets, a zero normalisation column and the water minimum one column apart between files; a
  60-line twix reader and the .seq's own trajectory gave unambiguous spectra. MATLAB-saved maps
  and masks are column-major: their 2D arrays are the transpose of numpy images (registration
  found it; their masks then still did not match our tubes and were not used).

- **One saturation time does not pin T1 in a CEST fit**: calibrated at 1.5 s, the model predicted
  the 4 s spectra better than "no change" but under-predicted the build-up 1.3-6x; T1 needs an
  independent measurement (or several Tsat) before a Tsat prediction tests the engine.

- **Batch parameter sets as spins**: a Levenberg-Marquardt Jacobian by forward differences costs
  one simulation if every perturbed parameter set is a spin (here 8 tubes x 8 sets x 58 offsets);
  ~7 min per iteration on the CPU core with exchange pools.

- **Look-Locker subspace bases need more components than their singular values suggest**
  (2026-10-08): K = 4 (4th singular value 3 % of the 1st) left per-tube curve fits off by 4-13 %
  and T1 moving with the fit's start time; K = 6 gave 0.4-1.4 % and stable T1. Check a subspace
  reconstruction by the fit residual of the curves it produces, not by the spectrum alone. In a
  3D slab, use the central partition for flip-angle-corrected T1 (the edges see less flip).

- **qMT papers name their exchange rates differently; convert before tabulating** (M5 tissue
  table, 2026-10-09). Our `BoundPool.k` and `TissueProperties.exchange_rate` are bound -> free.
  Yarnykh's k is free -> bound and his R = k(1-f)/f, with f = M0b/(M0b+M0f) (Yarnykh 2012, Eq. 2
  and 7), so his R is our k: whole-brain median 19.0 /s at 3 T. Henkelman/Sled-Pike's R with
  M0f = 1 is also our k (kf = R·F). Dortch's SIR kmf is bound -> free; its PSR is M0b/M0f, not a
  fraction (Li et al. 2014 muscle PSR 0.097 is f = 0.088). Fat fractions follow the same split:
  "pool size ratio" is F, "MPF"/"bound pool fraction" is f = F/(1+F).

- **Some qMT parameters are not identifiable in vivo**: in human muscle at 3 T the exchange rate
  did not converge in any of 10 subjects (Sinclair et al. 2010), so the muscle rate in the table
  is the brain's, flagged as assumed. Lipid ADCs from DW imaging rise with slice thickness
  (motion): 2.78e-5 at 20 mm, 1.83e-5 mm²/s at 2 mm (Weidlich, ISMRM 2018); take the smallest
  value as an upper bound.

- **ismrmrd-python's header arrays ignore whole assignment** (ismrmrd 1.15, writing M5's dataset
  writer): `acq.read_dir = (1., 0., 0.)` leaves the ctypes array unchanged without an error; set
  `acq.read_dir[k]` element by element. Scalars (`acq.sample_time_us`, `acq.idx.kspace_encode_step_1`)
  assign normally. The header model (`ismrmrd.xsd`) requires `experimentalConditions`.

- **A flagged row should not set a default** (2026-10-11): the 3 T grey-matter T2 (96.5 ms) was the median of
  Lu 2005's CPMG (83 ms) and Wansapura 1999's abstract (110 ms, its GM/WM labels contradicting physics and
  swapped by the curator, flagged). Unresolved flagged rows now stay in the table but do not pool: grey 83,
  white 72 ms (Lu 2005; Krug 2022). On six IXI heads with partial-volume fractions the brain-interior
  quantiles at slices 1-3 went from 1.42 / 1.93 / 2.51 to 1.37 / 1.84 / 2.49 (real 1.31 / 1.74 / 2.58).

- **Grey matter T2 at 3 T rests on two numbers** (2026-10-11): the database pools Lu 2005 (CPMG, 83
  ms) and Wansapura 1999 (110 ms, its abstract's GM/WM labels swapped by the curator, flagged).
  A 32-echo 2D train inflates apparent T2 through stimulated echoes, and our engine plays those
  echoes itself, so it needs the intrinsic T2; the real images' GM/WM (1.50 against our 1.58)
  lean the same way. Needs sources robust to stimulated echoes (EMC, nonselective CPMG,
  single-echo series). Leads not yet in the database: Utkur 2025 (MRM, 1 volunteer: 3D GRASE GM
  75.7, WM 59.5 ms); a multi-slice CPMG study with stimulated-echo correction reporting cortical GM
  79-88 ms by lobe (source not traced).

## Other simulators

- **MRzero**: RF is instantaneous; numerics are float32 (limits agreement to ~1e-5–4e-4 on
  long sequences, mostly from merging states whose k agrees only to float32); default PDG
  pruning thresholds discard most signal of point-isochromat phantoms, and its pre-pass with
  mean T1/T2 prunes long-lived CSF states; ADC frequency offsets are ignored; it tips +z to −y
  (constant −i relative to us). (`validation/mrzero/RESULTS.md`)

- **KomaMRI (from source, 2026)**: ideal linear gradients only, uniform transmit, no noise
  model; diffusion "under construction", susceptibility commented out; strong on motion,
  flow and speed. Its README roadmap is out of date.

- **JEMRIS**: the hardware-realism leader on CPU, but its GPU port drops diffusion, exchange,
  pTx, MT, concomitant fields, nonlinear gradients and eddy currents.

- **BMCTool / pulseq-cest (1.0.3, read before running)**: per RF sample, a Padé matrix
  exponential of the full Bloch–McConnell matrix (exact for piecewise-constant RF). MT pool:
  Mz only, exchange on z only, `k` from the pool to water (k_ab = f·k), W = ω1²·G with
  G = π·g. Its sequence semantics are not an imaging simulator's: any block with only a z
  gradient is an ideal spoiler, an ADC block records Mz (and by default resets to
  equilibrium), gradients otherwise do nothing, and a frequency offset rotates the frame for the
  pulse's duration, with phase bookkeeping (`accum_phase`) between pulses. Shaped pulses are
  downsampled to `max_pulse_samples` (500 by default). With the Lorentzian lineshape it
  agrees with blochparty to 6e-7 in Z (`validation/bmctool/compare_bmctool.py`).

## Tissue-property database — 2026-10-10

- **Getting at full texts.** Through this environment's proxy: Wiley, RSNA, AJR, ScienceDirect and
  the Wayback Machine answer 403 / Cloudflare / captcha; PMC article pages start serving reCAPTCHA
  after ~20 quick requests. What works: PMC OAI-PMH
  (`https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:<digits>&metadataPrefix=pmc`)
  and NCBI `efetch db=pmc` return clean JATS XML for open-access papers AND NIH author manuscripts;
  Europe PMC `fullTextXML` returns HTTP 500 for author manuscripts. In NIH-manuscript JATS the
  tables sit in `<floats-group>`, outside `<body>`. ISMRM proceedings PDFs before ~2010 download
  from `cds.ismrm.org/protected/<yy>MProceedings/PDFfiles/<nnnnn>.pdf` without login. Some
  "tables" are images (Campbell-Washburn 2019 Table 1). Source: the eight row-extraction passes.

- **IT'IS proxies are copies, and the copies reveal them**: Eye Retina = Brain GM, Iris/Ciliary
  body/Diaphragm = Muscle, Choroid/Heart lumen/Placenta = Blood, Connective tissue's water content =
  WM's. A (mean, SD, n, min, max) tuple shared by several names is a proxy; the owner must be chosen
  per table (the Gabriel 1996 dielectric sets have other owners than the relaxation rows: Testis,
  Thyroid, Tendon, Nerve, Stomach, Colon, inflated Lung). `tools/tissuedb_import_itis.py` keeps 212
  blocks and logs 459 skipped. IT'IS "Breast Fat" has T1 0.9-1 s (breast tissue, not adipose), and
  the T1 of WM there falls from 1.0 s at 1.5 T to 0.87 s at 3 T because each field pools other studies.

- **Abstract labels lie**: Wansapura 1999 calls T2* "T2" and lists "GM and WM 80 and 110 ms" in the
  order opposite to physics (GM T2 > WM T2); Shin 2025 swaps WM/GM in the abstract (Table 3 is GM,
  WM); the ISMRM archive page of Peters et al. 2006 lists Hoogenraad first (the PDF has Peters).
  `validation/anatomy/anatomy_brainweb.py` used the abstract order until 2026-10-10 (now GM .110, WM .080 s;
  see the Wansapura entry under "Pulseq, data and reference values").

- **The CSF T1 the project quotes (4.4 s, via Spijkerman 2018 ref. 13) is 4.3 +- 0.2 s in Rooney
  2007's abstract** (across 3 subjects and all fields; Rooney did not scan at 3 T). Spijkerman's
  2.03 s 3 T CSF T2 is a median from the longest-TE (partial-volume) analysis. Caan 2019 scanned 8
  but analysed 6 subjects.

- **Method beats tissue in many cells**: myocardial T1 depends on MOLLI/ShMOLLI/SASHA, kidney T1 on
  the readout (>200 ms between IR SE-EPI and bFFE, Cox 2017), cortical-bone T1 by a factor ~2.5
  (in vivo UTE-IR vs ex vivo IR-CPMG), MRF T2 runs low in kidney and high in abdominal muscle, VFA
  T1 is 16-28 % above Look-Locker in the same subjects (Krug 2022), blood T2 depends on the CPMG
  spacing, tendon "T2*" on the echo set. The `method` column is part of the value.

- **MT conventions in practice**: Vanderbilt SIR kmf and Bayer's bSSFP kmf already are the
  bound -> free R; Yarnykh/CRI k is free -> bound (R = k(1-f)/f); PSR/F = M0b/M0f -> f = PSR/(1+PSR),
  SD scaled by 1/(1+PSR)^2; the Gloor 2008 bSSFP equation overestimates F by ~40 % (Bayer 2022).
  Stanisz 2006 prints blood T2B 280 +- 50 us next to T2 275 +- 50 ms: probably a typo, kept flagged.

- **QSM numbers carry their reference**: CSF, a deep-WM ROI set to -0.030 ppm, the whole-brain mean
  and phantom ground truth all occur; the database keeps them as `chi_qsm` with the reference in
  `method`, separate from absolute `chi`. Susceptometry papers report cgs: SI = 4 pi cgs.

- **Most blood-oxygenation calibrations are bovine blood** (Lu 2012, Qin 2011, Krishnamurthy 2014,
  Grgac 2013/2017), and Liu 2016's printed T2 coefficients do not reproduce its own 191 ms.

- **Pool studies as equals, and relaxation times by their reference method** (2026-10-10): n-weighted
  pooling let Ndengera 2022 (SyMRI, 54 subjects aged 83 on average) set WM T2 108 ms against 56-80
  ms elsewhere: pooled WM and GM T2 came out 86 ms each, and simulated T2w heads lost all GM/WM
  contrast (realism f2). Between-study differences (method, cohort, region) dwarf sampling error, so
  the random-effects weights are equal: median over sources. T1/T2 from spin-echo decay and
  inversion/saturation recovery first: WM T2 74.5, GM 110, muscle 30 ms (52 with MRF rows). Rows
  without a method cannot be classified and drop out of a tier that has reference rows (de
  Bazelaire's fat rows had an empty method; filled from the paper's protocol).

- **The bright scalp in T2w TSE was anatomy first, fat T2 second** (2026-10-10): real fastMRI T2w
  heads show a scalp peak 2.3x the brain interior 5-7 mm below the edge; the simulation had a flat
  1.6, and changing fat's T2 from 68 to 148 ms changed nothing, because BrainWeb labels the scalp
  "Muscle/Skin" (here `skin`, given muscle values) and the fat class covers 0-24 % of it. Splitting
  that class by depth (dermis 1.61 mm, subcutaneous 4.14 mm, Li 2024 ultrasound: `layer_scalp`) put
  the peak in place (1.26 at fat T2 68 ms), and the J-suppressed fat T2 lifted it to 1.86. Measure
  where a contrast lives (an edge-depth profile, tissue composition per depth band) before tuning a
  parameter. Still off: the calvarial diploe (fatty marrow) is 0.8 vs 1.6 in the real profile.

- **J coupling of the chain ends does not explain bright fat by itself** (2026-10-13, validation/jcoupling/jcoupling_fat.py): a
  density-matrix model of the coupled triglyceride fragments (methyl, carboxyl, allylic, diallylic, glycerol; J 7 Hz),
  calibrated to CPMG fat T2 145 ms at 8 ms spacing, predicts single-echo fat T2 222 ms where 48-68 ms is measured at
  3 T. Half the fat signal is bulk methylene whose neighbours share its shift (no observable J), and the coupled groups'
  modulation oscillates and partly recovers. Check a mechanism against both limits (here SE and CPMG) before turning it
  into a parameter.

- **A Bloch simulator's lipid T2 is the J-suppressed one** (Henkelman 1992; Keene 2020): single-echo
  SE and STEAM read fat T2 40-75 ms at 3 T because J coupling modulates the echo; a train of
  refocusing pulses spaced well below 1/(chemical-shift difference of the coupled groups) suppresses
  it (Allerhand), and CPMG at 8 ms spacing reads 140-159 ms. A Bloch model has no J coupling but
  plays the refocusing train, so it needs the CPMG value; with it, simulated single-echo SE fat is
  too bright (the missing physics, REALISM.md).

- **Fractions of a compartmentalised tissue must not be pooled as one tissue**: cortical-bone pd rows
  of bound and pore water add up, an average of them means nothing; they are separate ids
  (`bone.cortical.bound_water`, `.pore_water`, `.collagen`).
