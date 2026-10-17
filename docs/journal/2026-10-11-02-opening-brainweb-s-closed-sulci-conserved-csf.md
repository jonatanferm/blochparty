# 2026-10-11 — Opening BrainWeb's closed sulci; conserved CSF; IXI as the next anatomy

- **Opened sulci** (`open_sulci` rewritten; `sample_head(age=)`): a closed sulcus is grey matter
  reached from two facing white-matter banks, found with the feature transform (neighbouring voxels
  whose nearest white points are >= 3 mm apart and within 60 deg of opposite: the lambda-medial axis)
  on BrainWeb's native 0.5 mm grid, kept where the sheet touches the CSF around the brain, outside the
  deep grey regions; CSF fills the age's width (Jin et al. 2018: 1.27 mm at 47 y, 1.49 at 67, with
  their between-subject SD) either side of the midplane, then the head is resampled (0e2f15f; peak
  memory 6.9 -> 3.8 GB by opening a three-class copy, 24e1db8). Unlike the four earlier openers it
  draws continuous sulcal lines; 0.5 mm bumps and 3 mm grooves open nothing (tests); 20 subjects
  get 117-210 mL at 1.3 mm (MNI size). Subject 04's CSF 254 -> 322 mL (25 y) / 381 mL (75 y, with
  ventricles of the same age; Yamada 2023 287 / 389). Opening brings BrainWeb's grey/white volume
  1.44 -> 1.23 and its pial-to-white distance 2.92 -> 2.35 mm. The ridge opener is gone.
- **Matched pair (local, quality real, 6 heads, slices 1-7)**: the same heads with (A) and without (B)
  opened sulci. Blinded on A: 10/12 (one sim at slice 6 taken for real, one real for sim; r13-r15
  were 12/12). Cortical-band CSF (slices 2-6, components per 1000 px / mean px / bright fraction):
  real 7.2 / 31 / 0.25 (10-90 %: 5.1-9.6 / 11-66 / 0.14-0.33); B 8.8 / 21 / 0.19; A 4.9 / 65 / 0.33,
  i.e. from the real lower half to its upper edge, with young draws (25-48 y). Brain power spectrum
  over real, by band 0.025-0.475 cycles/px: B 1.4-1.5 low, ~1.0 mid, 0.72-0.86 high (soft); A 1.6-1.95
  low and mid, 1.26-1.50 high. GM/WM 1.63 -> 1.59 (real 1.49); the rim under the skull brighter
  (front 0.77 -> 0.82, side 0.70 -> 0.79; real 0.71 / 0.61). Per head the excess follows the subject
  (BrainWeb 04 and 45 already bright, 42 matches real) more than the width.
- **Width x 0.6 (pair C, same heads)**: cortical-band CSF 6.7 / 42 / 0.28 (real 7.2 / 31 / 0.25), the
  spectrum's high bands ~1.0 of real (mid 1.36-1.64); blinded 12/12 nonetheless, by hazy lateral
  patches and a foggy, mottled cortex: no longer the sulci.
- **The fog is BrainWeb's diffuse CSF**: grey-dominant voxels hold 37.6 mL of CSF (3.9 % on average,
  subject 04), which at T2w lifts grey matter ~9 %, the gap between our GM/WM (1.59) and real (1.49).
  It is the blurred trace of the sulci now opened explicitly. Grey matter T2 (96.5 ms, the median of
  Lu 2005 83 ms and Wansapura 1999 110 ms) is the other suspect; the 3 T literature with
  slice-selective multi-echo fits runs to 147 ms (Mumuni 2015), so it needs stimulated-echo-robust
  sources (EMC: Radunsky 2021; Wiggermann 2020 3 T values only in a figure). Parked.
- **Brain-interior intensity quantiles settle the amount** (q70 / q85 over q15 ~ white matter, slices
  1-4; real 1.71 / 2.51): unopened 1.73 / 2.55, Jin x 0.6 1.91 / 2.83, Jin x 1.0 2.07 / 3.22,
  gathered + Jin 2.07 / 3.19 (slices 5-7: real 1.79 / 2.67, unopened 1.70 / 2.27). The real images
  hold about BrainWeb's amount of bright CSF, in lines rather than a rim and diffuse speckle. Hence
  **conserved sulci** (0988ea3, now `sample_head(age=)`'s default): each grey/CSF mixture outside the
  deep nuclei goes to its larger share (`gather_cortical_csf`), the sheets take the width that
  keeps the CSF volume (`open_sulci(volume=)`), widened by Jin's slope (0.011 mm/yr) beyond 30 y;
  `sulci="measured"` keeps Jin's widths. On the six heads: 1.67 / 2.42 and 1.78 / 2.36; cortical-band
  CSF 8.9 / 21 / 0.20 (young draws, 25-48 y, so thin lines). Subject 04 at 25 y: CSF unchanged
  (253.7 mL), 19.9 mL gathered.
- **Blinded, six local heads each (quality real)**: Jin widths 10/12, x 0.6 12/12, gathered + Jin
  12/12, conserved 10/12 (one miss was subject 42's skin wedge at the source grid's front edge, a
  cut of BrainWeb's field of view the pitch lifts into slice 3 in front of the forehead, which I read
  as a real skin marker). What still gives tiles away: a broad light band along the midline at
  slices 6-7 (subjects 04, 45: BrainWeb's posterior medial cortex is a ~15 mm grey slab with no
  white cores, which no opener can split), hazy lateral patches, dense fine lines at slice 7.
- **Checked and not the tell**: head aspect (left-right / front-back 0.81-0.82 at slices 1-7, real
  0.79-0.81), the dark band under the scalp (front 2.5 px vs real 2.0, back 3.5 vs 2.5; before the
  rim calibration it was narrower than real, so later changes widened it, by 0.5-1 px).
- **Anatomy source**: the remaining tells (midline grey slab, hazy patches, fine lines at slice 7) are
  BrainWeb's 1.5 T, 1 mm segmentation. With the user's go-ahead, IXI's 3 T subjects (Hammersmith,
  T1 0.94 x 0.94 x 1.2 mm, T2, ages; CC BY-SA 3.0) are being fetched (`validation/anatomy/fetch_ixi.py`,
  parallel byte ranges: one stream ran at 0.65 MB/s) to segment into BrainWeb's 12 classes on its
  MNI grid (affine registration to a T1-like image from BrainWeb's fractions works: 94 s, det 0.83).
- **r16 (real_fine, 4x4x4 spins, BrainWeb preset)**: two heads died silently on the swapping mini, 10
  heads from 8 subjects; the score's sim-vs-sim control came out 0.008 (anti-learning with 4 people
  per side), so its real-vs-sim AUC 0.07 means nothing (LESSONS). The other measures: in-brain grain
  0.0132 of p99 (real 0.0141, r13 0.0226), so the spin grain is gone; the finest spectral bands
  (0.33-0.48 cycles/px) now at 0.50-0.77 of real (r13 0.74-0.87); blinded 12/12 by the anatomy
  (dark gap under the skull, hazy lateral cortex, midline band).
- **IXI heads** (f86012d): `validation/anatomy/prepare_ixi.py` segments each 3 T subject into
  BrainWeb's 12 classes on its 1 mm MNI grid (affine to a T1-like BrainWeb image, EM on log T1 / log T2
  with BrainWeb priors and a cubic log-bias field; outside the skull, intensity rules: the EM swapped
  classes in the thin head layers). ~3 min, 13 MB per subject; age and sex kept from IXI.xls (ages
  20-82, median 48, 93 of 185 women). `load_ixi`, `sample_head(source="ixi", size="native")`,
  `simulate_heads --source ixi` (no rim calibration, no sulcal or ventricle edits; layered scalp as
  for BrainWeb). Rendered IXI anatomy shows long sulcal lines and a thin midline at the vertex where
  BrainWeb has a grey slab; first simulated IXI heads look like the real scans at slices 5-7. Known
  gap: the T2 covers MNI z <= ~92, so the scalp above is background (above the scored slices).
- **Blinded, six IXI heads (v1, quality real)**: 10/12, with several guesses; the cues were a uniform
  bright scalp ring (fixed 4.14 mm `layer_scalp` everywhere), orbits and frontal sinus at slice 2 (a
  -19.5 deg pitch draw, 2 SD), and a squiggle in front of the forehead (a skin marker in IXI277's own
  scan, attached to the scalp). One IXI tile passed as real.
- **IXI v2 + drawn scalp** (4fb9fed): scalp fat by unmixing the subject's bias-corrected T2 between soft
  tissue (0.9 x WM, temporalis) and its own pure fat (98th percentile of T1-bright head tissue; orbital
  fat 2.3-3.8 x WM in T2), the head one connected piece and, above MNI z 15 mm, each axial section
  opened by an 8 mm disc (the marker goes); `scalp="layered_drawn"` (preset) draws each head's dermis
  and subcutaneous thickness from Li 2024 (1.61 +- 0.23, 4.14 +- 1.10 mm). IXI's own fat alone made the
  scalp dimmer than fastMRI's (the real scalp is as bright as CSF), so the layers stay on top of it.
  26 subjects re-prepared (~3 min each); age trends on v1: GM/ICV 0.484 -> 0.437 from 25 to 75 y
  (r -0.78), CSF/ICV 0.194 -> 0.229 (r 0.69), ICV men 1465, women 1290 mL.
- **r18 stopped** after 6 of 20 v1 heads (230 s per slice on the swapping mini); **r19**: 16 IXI v2 heads,
  one subject each, spins 2x4x4, slices 1-7, then the score with its controls.
- **Blinded, six IXI v2 heads (local, quality real; subjects 12, 148, 202, 253, 334, 538, not looked at
  before)**: 8/12, mostly guesses (two simulated tiles taken for real, two real for simulated). The
  blinded tally on BrainWeb heads was 10-12/12 all day; one 12-tile sheet is weak evidence (8 of 12
  is not significantly above chance), so r19's 16 heads get several sheets and the CNN with its controls.
- **Open: IXI heads narrow a slice too fast.** On the six v2 heads, brain-interior quantiles at slices 5-7
  (q50 / q70 / q85 over q15) are 1.71 / 2.68 / 3.90 against real 1.31 / 1.79 / 2.67 (slices 1-4 closer:
  1.45 / 1.95 / 2.85 vs 1.30 / 1.71 / 2.51), and head area falls faster (slice 7 / slice 1: 0.78 vs
  real 0.86, BrainWeb 0.84). Pitch (-12 to 0 deg) barely moves the anatomy's area profile; a stack
  offset of 15 mm instead of 10 matches it (rms 0.020 vs 0.038 over slices 0-11), but anatomy argues
  for ~8.5 mm at IXI's measured height (Monro ~10 mm below the bodies in MNI, z scale ~0.85). The MNI
  frame is taller than real heads, so BrainWeb heads sized by breadth and length stayed tall and the
  -10 deg pitch was fitted on them. Kept the anatomical offset for r19; its per-slice AUC will say
  whether the top slices carry the difference. Sulcal band statistics now match (7.95 / 34 / 0.27 vs
  7.25 / 31 / 0.25); the back's dark band is wide (4.0 vs 2.5 px).
- **Blinded on r19's first eight heads (v2)**: 10/12; the cues that held were a lighter, hazier white
  matter and cortex (three tiles) and IXI206's scalp split into two arcs. IXI tally: 28/36 (ixiB 10, ixiE
  8, r19a 10): harder than BrainWeb's but above chance.
- **T1 and T2 move apart** (230c44d): IXI's two scans are separate and heads move between them; the best
  integer shift of T2's gradient magnitude onto T1's was 4+ mm in z (IXI206), ~2 mm (IXI012), ~1.4 mm
  (IXI334), 0 (IXI148). Misaligned, the joint EM blurs every tissue boundary (a likely part of the
  haze and the thick cortex: GM/WM volume median 1.43, 1.49 at 30 y) and the head mask doubles the
  scalp. A rigid T2-to-T1 correction (gradient-magnitude correlation, 2 then 1 mm) fixes IXI206's scalp;
  all 26 heads are being re-prepared (v3) for r20. Deep white matter is ~96-98 % white in both IXI and
  BrainWeb, so the haze is not white-matter impurity.
- **r19 (16 IXI v2 heads, one subject each, spins 2x4x4, slices 1-7)**: controls fine this time (real vs
  real 0.44, simulated vs simulated 0.59), so the score stands: CNN AUC 0.979 [0.93, 1.0] at every slice
  (0.976-0.988), histogram AUC 0.965 [0.93, 0.99]: the intensity distribution alone separates them.
  Quartiles of the image (of p99) 0.245 / 0.382 / 0.617 against real 0.216 / 0.299 / 0.491 (W1 0.065;
  BrainWeb runs 0.025): the IXI heads are brighter. IXI's grey-dominant voxels hold 5-7 % CSF (EM
  posteriors, not partial volume; BrainWeb 4.9 %), ~+14 % on grey matter; grey-matter T2 (96.5 ms vs
  Lu 2005's 83) another ~5 %; the high slices more (stack height, open). Variants on four v3 heads
  (baseline, GM T2 83 ms, diffuse CSF < 0.15 in grey given back) are running.
- **Why IXI heads are bright: variants on four v3 heads** (brain-interior q50 / q70 / q85 over q15; real
  slices 1, 3: 1.32 / 1.75 / 2.54, slices 5, 7: 1.32 / 1.79 / 2.66): baseline 1.37 / 1.80 / 2.52 and
  1.50 / 2.10 / 3.33 (the low slices already match; the excess is at the top); grey T2 83 ms 1.28 / 1.61 /
  2.35 and 1.38 / 1.98 / 3.30 (the truth between 83 and 96.5 ms); diffuse CSF < 0.15 in grey given back
  1.34 / 1.71 / 2.40 and 1.43 / 1.98 / 3.25; slice 0 15 mm below the ventricle bodies (three heads)
  1.39 / 1.73 / 2.45 and 1.49 / 1.96 / 3.02, the head's q75 at the top 0.56 -> 0.50 (real 0.50).
  Adopted the 15 mm offset for IXI heads as a calibration (27ae8f1); grey T2 and partial volume open.
- **Partial volume, tried and reverted**: in IXI012's grey/CSF boundary voxels the EM posterior holds 103 mL
  of CSF, linear unmixing of the T2 52 mL (the posterior is close to a step in intensity). Unmixing all
  three classes from two channels and the sum turned noise into white matter (GM/WM 1.48 -> 0.90);
  pairwise unmixing with a dead zone left the total unchanged (CSF reference pulled down by sulcal
  "pure" CSF); ventricle-core references took CSF to 173 mL (11 % of ICV), local references (10 mm) to
  200 mL; on one head the image's high-slice tail improved (q85 3.33 -> 3.09) but the low slices moved
  away from real. With no independent check of which CSF volume is right, v3 keeps the EM posteriors.
- **Blinded, six v3 IXI heads (T2 aligned, 15 mm offset)**: 10/12; both giveaways were slice-7 tiles that
  sat at the wrong height: IXI255 a tiny vertex section (its deep-CSF maximum fell on wide fissure and
  sulcal CSF at 47 mm instead of the ventricle bodies at ~20) and IXI598 a mottled section (a failed
  segmentation: registration lncc 0.112, CSF throughout the brain, 6 % in the white-matter core).
  Fixes (bd47ef2): heads with lncc < 0.15 are not drawn (IXI598, IXI586 rejected; 24 left; the others
  0.16-0.22), and the IXI stack searches the deep-CSF maximum within 12 mm of where MNI's ventricle
  bodies (0, -15, +20 mm) land after the pose (IXI255's slice 0: 28 -> 7 mm). Shape filters (a 2-3 mm
  opening, a central box) did not separate the vertex fissure from the ventricles. IXI tally: 38/48.
- **The mini is off-limits until 23:00 tonight** (user); r20 (16 IXI v3 heads, the 24 that pass QC, the
  landmark prior, 15 mm offset) starts then by a local job that syncs first.
- **The 15 mm offset was wrong** (fa46472): blinded on six heads with the landmark prior and 15 mm, 12/12; the
  simulated slices 1-3 showed basal ganglia, insula and temporal lobes where the real ones show the
  ventricle bodies (1-1.5 slices too low). The 15 mm had been fitted while IXI255's landmark sat 20+ mm
  high; with the prior the area profile prefers 12 mm (rms 0.017, 10 mm 0.029, 15 mm 0.023), but the
  anatomy the eye sees decides: back to 10 mm. IXI tally: 50/60.
- **Blinded, six heads at 10 mm with the landmark prior**: 10/12; the giveaways were slices 6-7 with dense fine
  lines and a hazy midline (three tiles) and one low slice 1 (IXI609). IXI tally: 60/72.
- **Partial volume and grey T2, on the same six heads and seeds** (brain-interior q50 / q70 / q85 over q15;
  real slices 1-3: 1.31 / 1.74 / 2.58, slices 5-7: 1.31 / 1.79 / 2.67): EM posteriors 1.50 / 2.02 / 2.95 and
  1.64 / 2.30 / 3.80; partial-volume unmixing (local references) 1.42 / 1.93 / 2.51 and 1.61 / 2.22 / 3.18;
  plus GM / WM T2 83 / 72 ms 1.37 / 1.84 / 2.49 and 1.55 / 2.12 / 3.25. Both adopted (a7eb14d): unmixing is
  prepare_ixi's default; tissuedb leaves flagged rows out of the pool (only Wansapura 1999's abstract, whose
  GM/WM labels were swapped by assumption), so GM T2 at 3 T is Lu 2005's 83 ms and WM 72 ms (this also
  changes the BrainWeb realism runs' table). The top slices stay ~18 % too bright at the median: IXI's
  cortex grows with height faster than the clinical scans' (open: population age, head shape, stack).
- 24 heads re-prepared with unmixing (v5) for r20, which starts on the mini after 23:00.
- **My blinded sheets are wearing out**: 12/12 on six v5 heads (unmixing, GM T2 83 ms), but these were the six
  subjects I had just looked at in three variants; I now recognise some IXI brains by shape (IXI334's bright
  posterior midline). The CNN (r20) and a fresh observer are the fair judges from here.
- **A quiz for a fresh eye**: `outputs/real_or_simulated.html` (local only: the real tiles are fastMRI), 12 real
  and 12 simulated slices at the final settings (v5 heads, GM T2 83 ms), shuffled, scored in the page; built
  by the scratch quiz.py, its key unseen by me. 24 v5 heads prepared (all pass QC); ~30 more subjects being
  prepared into prepared_more/ for a larger pool. r20 starts on the mini after 23:00.
- **r20 (16 IXI v5 heads, PV unmixing, GM/WM T2 83 / 72 ms, landmark prior, 10 mm offset; one head died on the
  mini, 15 scored)**: CNN AUC 0.959 [0.90, 1.0] (r19 0.979), histogram 0.927 [0.85, 0.98] (0.965), W1 0.042
  (0.065), image quartiles 0.230 / 0.339 / 0.573 vs real 0.216 / 0.299 / 0.491; controls 0.44 / 0.47. By slice
  0.93 (1), 0.93 (2), 0.97 (3-6), 0.98 (7): the low slices are the most realistic. By region (of p99,
  slices 1-3): the band 8-20 px in 0.33 vs real 0.26, the core 0.25 vs 0.21; at slices 5-7 mid and core
  match and the outer ring is brighter (0.41 vs 0.37). Bone-aligned profiles: the rim 2-4 px inside the
  skull is bright again (front 0.55-0.56 vs 0.39-0.46), now from IXI's own subarachnoid CSF, and the lateral
  scalp is too bright (0.80 vs 0.54): `layer_scalp` lays 4 mm of fat over the temporalis, where real scalp
  has little. Simulated heads are also ~7 % larger in area at slices 1-3 (8.2-8.3 vs 7.7 kpx).
