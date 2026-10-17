# 2026-10-11 — Visual inspection rounds: what made the simulated heads look fake

- **Method**: the CNN AUC (~0.998) cannot rank fixes, so side-by-side sheets (same recon, noise floor
  and normalisation as score.py) plus a number per tell, and blinded 12-tile sheets (6 real, 6
  simulated, key read after the guesses). Rounds 1-2 on BrainWeb 04 locally, then 20 subjects.
- **Tells found and fixed (each with its number)**:
  - Grey matter too bright (GM/WM 1.91 vs real 1.49): GM T2 rested on one study (110 ms); added the
    Bojorquez 2017 compilation's primaries (Lu 2005 CPMG 83/75 ms, six IR T1 studies; 1f9741d):
    GM T2 96.5 ms; GM/WM over 54 slices 1.54.
  - Heads too large and round (264 vs 189 cm^2 at the ventricles, AP/LR 1.14 vs 1.31): BrainWeb is
    MNI-sized; ANSUR II head breadth/length draws, yaw (real SD 3.8 deg), centred FOV (ff00d4a,
    60603de): 211 cm^2, 1.28.
  - Salt-and-pepper grain: categorical spin draws, and random z through the 5 mm slice profile;
    proportional sampling + z quadrature (`jitter="inplane"`, quality "real").
  - Slice 0 too high (slit ventricles): fitted the real head-area profile and placed slice 0 at the
    foramen of Monro (10 mm below the largest ventricle section), 5 mm spacing.
  - Readout fold-over: 2x readout oversampling as the scanner (b20c148).
  - Skull band too dark (0.26 vs 0.35-0.41): BrainWeb marrow -> yellow marrow (Loevner 2002; 9494d56).
  - Brain tissue ~15 % bright relative to CSF and fat: myelin water as a second free-water species
    (WM 12.3 %, 9680845); body-coil B1+ from the full-wave map (042312f).
- **Dead end**: `open_sulci` (age-dependent sulcal widening, Kochunov 2005) made CSF specks on the
  1 mm grid; the first blinded test (r2) picked every simulated slice by that grain. Left out.
- **Blinded tests**: r2 12/12, r3 12/12 (still obviously fake). Remaining cues: scalp thinner and more
  regular than real, BrainWeb's regular young anatomy (no asymmetry, small sulcal CSF, no pathology),
  uniform white matter, and the contrast gap above.
- **r4** (20 subjects available locally, 6 heads x 9 slices, realistic preset of 9680845/042312f):
  CSF max/WM 4.95 (real 5.21), scalp/WM 2.62 (2.91), GM/WM 1.58 (1.49); blinded 12/12. Head outline
  RMS residual from its ellipse 2.73 % against real 3.32 %; a 9 mm deformation reaches 3.14 % but
  swirls the occipital cortex; not adopted.
- **Deep grey nuclei** (8ead2e0, 30afdd8): BrainWeb calls the caudate, putamen, pallidum and thalamus
  grey matter, so the basal ganglia shone like cortex; at 3 T their iron puts T2 near white
  matter's (Krug 2022: 57-76 ms against cortex 96.5). Labelled by MNI regions, their own tissues
  from the database (nearest-ancestor fallback for what they lack), drawn from their own stream.
  The stack landmark counted gm + wm as brain and jumped 28 mm when the nuclei left gm: fixed.
- **Where the remaining tell sits**: per-ray profiles aligned at each ray's bone minimum (not at a
  fixed depth: that had sent the marrow the wrong way) agree with real from 5 px inward; the excess
  is a bright subarachnoid rim in the first 2-4 px (frontal 0.62-0.64 against 0.39-0.44), with a
  brighter bone band (0.27-0.29 against 0.16-0.21) and a dimmer scalp (0.79 against 0.89 at p90,
  while CSF sets the p99 scale alone: the whole brain ~13 % bright). Variants on the mini (4 heads x
  3 slices each): receive ring at 12 cm or flat, GM T2 85 ms, red/trabecular marrow, the shell-kept
  size, thinning the subarachnoid layer by 1 or 2 mm. Only marrow (bone) and thinning (rim) moved
  their numbers; the full-wave 8-loop head array would make the periphery 1.4-2x brighter, which
  the real scans do not show (they behave as receive-normalised).
- **Shell-kept size** (7603d03): plain resizing thinned the skull and scalp with the head (0.84-0.90);
  `size="anthropometric_shell"` scales only the skull's contents. Skull front/side/back 6.8 -> 7.2,
  2.2 -> 3.0, 5.0 -> 5.5 mm.
- **Flow voids** (f180c83): moving spins through 5 mm, one echo train at 10.26 ms: blood keeps
  0.82 / 0.33 / 0.11 / 0.002 of its static signal at 113 ms for 0.5 / 2 / 5 / 10 cm/s; BrainWeb's
  vessel class (sinuses, large vessels) now leaves no signal; the real superior sagittal sinus is
  dark in every 0.57 mm reconstruction checked.
- **Marrow back to trabecular** (c10c7d2): reverts 9494d56 on the aligned evidence.
- **Ventricles with age** (340dde0, 29d6def): Yamada 2023 lateral-ventricle means as a ratio to 30 y
  (x1.93 at 70); a continuous wall push per side (septum kept), periventricular tissue compressed.
- **Refocusing angle** from each header (0d07df1); headers also give the echo spacing (10.26 ms,
  TE = echo 11) and the true slice thickness (`reconSpace` z 5 mm).
- **Blinded**: r5 (deep grey) 12/12; r6 (shell, flow void) 10/12, the first misses, both at the
  basal-ganglia level; still picked by the bright rim and speckled cortex higher up. r7 (r6 +
  1.5 mm of convexity CSF to inner table, a calibration) 10/12.
- **The "speckled cortex", measured**: CSF-bright components (> half the image's own ventricular
  CSF) in the band 8-25 px under the head edge, slices 2-6: real 8.1 per 1000 px, mean 24 px,
  bright fraction 0.206; r6 6.1, 48 px (the rim joins the sulci), 0.275; r7 8.6, 26 px, 0.232.
  The rim, not the sulci, made the cortex read as fake; the calibration removes 16 % of the CSF
  (245 -> 205 mL), so it stays labelled a calibration (`--sas-mm`), with the age-enlarged
  ventricles giving some back.
- **r8 / r9**: ventricles for ages 20-80, trabecular marrow, protocol refocusing (r8), plus 1.5 mm
  convexity thinning (r9; 4c98996 puts both in the preset). r9 slices 0-8: rim 0.91 / 0.71 / 0.65
  (real 0.75 / 0.52 / 0.57), cortical CSF components 8.5 / 1000 px, 19 px, 0.18 (real 8.1, 24,
  0.21). Blinded r9: 8/12, 10/12, 12/12 = 30/36 (I had seen its six heads by the third sheet).
- **Head shape by height**: simulated heads lost area faster above slice 0 than the real ones. Part
  was my measurement (`lab.head_mask` opened twice and ate the real thin lateral scalp: real
  left-right 143 -> 157 mm at slice 0 with one opening); the rest is tilt. Real routine slices sit
  ~10 deg nose-up to BrainWeb's AC-PC frame: pitch -10 +- 5 deg (1802b56, b7a4ca1) matches the area
  at slices 5 / 8 / 12 / 15 (0.89 / 0.77 / 0.50 / 0.21 vs 0.94 / 0.79 / 0.48 / 0.21).
- **Receive ring at the coil's surface** (1a6d19a): 12.5 cm instead of 18 cm; the scalp top decile
  is as bright as ventricular CSF in the real scans (0.93 of it) and was 0.68 of it in ours.
- **BrainWeb 38's marker** (7332ee8): a detached blob in front of the forehead showed as a dash and
  gave tiles away; detached tissue is dropped.
- Dead ends: membership sharpening (power 2; no better), GM T2 85 ms (no visible change), a 9 mm
  deformation (swirls). The "grain" metric flips sign between slice sets; not used.
- **r11 / r12** (12 new heads each): the 12.5 cm ring (r11) lowered the CNN AUC to 0.967 but the
  blinded test went to 12/12 (bright peripheral cortex, GM/WM 1.73, and a black-looking bone band);
  back to 18 cm with 2 mm thinning (r12): blinded 30/36, CNN 1.0, histogram AUC 0.86, but the
  cortical CSF broke into dots (6f1fe24, then 1.5 mm again: f73fc96).
- **Cleanups**: faint "skin" around BrainWeb 42's neck and fuzzy-classifier haze in the air are
  dropped with the marker (0982579); simulate_heads frees memory between heads (a465208; one
  process grew to 15 GB on the mini and swapped).
- **r13, the scoring protocol of f0-f4** (20 new heads, slices 0-12, realistic preset at f73fc96):
  CNN AUC 0.973 [0.916, 0.999] (f4: 0.998 [0.992, 1]), histogram AUC 0.774 [0.662, 0.879] (0.911),
  intensity W1 0.025 (0.041), quartiles 0.256 / 0.375 / 0.558 against real 0.242 / 0.344 / 0.539.
  Blinded on its first nine heads: 12/12, by the speckled cortex and bright posterior sulcal CSF at
  the high slices: BrainWeb's closed sulci (LESSONS), which no setting here reaches.
- **r14 (12 new heads, slices 1-7, GM T2 83 ms as an experiment)** against r13 on the same slices:
  CNN AUC 0.905 [0.78, 0.99] vs 0.936 [0.82, 0.998], histogram AUC 0.866 vs 0.898, GM/WM 1.54 vs
  1.58 (real 1.49); blinded 12/12 (two tiles by things stuck to the head: BrainWeb 18's marker,
  a thin line in a tilted 42; trimmed, ed15108). The database keeps 96.5 ms until sources robust
  to stimulated echoes are in (LESSONS lists leads).
- **Sulci**: three openers failed (white-matter-distance ridge, Hessian sheets, distance slab), and
  so did using BrainWeb's own faint CSF trace along the closed sulci (real, but too noisy as a
  ridge). The anatomy, not the physics, is what still gives the simulated slices away; the next
  step is a richer anatomy source (needs a download: the user's call) or surface-based sulci.
- Blinded tally this session: r4-r5 12/12; r6-r7 10/12; r9 30/36; r11 12/12; r12 30/36; r13 24/24;
  r14 12/12. Caveat: I have looked at the 21 real scans dozens of times and recognise some; a fair
  human test needs real scans not seen before.
- **Where the CNN sees it** (r13 against real, slices 0-12, the same cross-validated CNN on masked
  images): full 0.973 [0.916, 0.999]; interior beyond 8 px of the head edge 0.991 [0.97, 1.0];
  outer 8 px ring only 0.924 [0.813, 0.985]; centre beyond 25 px 0.831 [0.664, 0.964]. The cortex
  carries most of it, the deep brain least: the same verdict as the eye.
- **r15 (12 new heads, slices 1-7, + `--trace-sulci`, ~13 mL of sheets along BrainWeb's faint CSF
  trace)**: CNN AUC 1.0 [1.0, 1.0] (r13 on the same slices 0.936), histogram AUC 0.86, blinded 12/12.
  No better; the option stays experimental and off. The QSM Challenge 2.0 head (the one other local
  full-head segmentation) has crisp labels but a noisy ribbon and few sulcal lines, and is one person.
  The remaining tell is BrainWeb's cortex; the way on is a population of heads with resolved sulci
  (e.g. HCP or OASIS with FreeSurfer surfaces, IXI with a segmentation), which needs the user's
  decision on downloads and data-use terms.
- **Structure or texture?** The same CNN on blurred images (sigma 1.5 px) still scores 0.979, on
  high-pass residuals alone 0.985: two independent cues. The texture one is ours to fix: in flat
  brain the simulated slices carry 0.022 (Immerkaer sigma, of p99) of pixel-scale texture, the
  same anatomy rendered without spins 0.011, real scans 0.014 with their noise. It is the grain of
  random in-plane spin positions and falls as 1 / sqrt(spins) (4x2x2 0.026, 2x4x4 0.017, 3x4x4
  0.015, 4x4x4 0.0136); subcell centres band and a per-voxel shift makes it worse (0.037).
  In-brain thermal noise is right: real 0.0036 of p99 (each scan's coil covariance from its
  k-space corners, through the same recon), simulated 0.0031. Real coil noise is correlated
  (median |r| 0.09-0.16, up to 0.5) and unequal (2.5-4x), so the background is a few-channel chi
  (std/mean 0.30 raw) but the matching by the background mean still lands the brain's noise.
  Quality "real_fine" (4x4x4) added; r16 runs it.
