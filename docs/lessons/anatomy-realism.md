# Lessons: Anatomy and realism

- **The T2* policy can silently drop a measured R2* map**: with susceptibility on, the default
  (`t2star="auto"`) omits R2' to avoid double counting, so AHEAD's measured R2* produced no T2*
  contrast at all until R2' was supplied explicitly. Look at a long-TE image before trusting it.

- **AHEAD v3 states no units**; ppm and 1/s follow from WM R1 ~0.88 /s, R2* ~ 36 /s at 7 T
  (Caan 2019) and a positive pallidum. PMC served a CAPTCHA for the methods paper: do not try to
  get around it; use the publisher record and the values.

- **A sum-to-one check at 16 eps fails after cumulative-sum resampling in float64**: resampling
  complementary fractions by differencing cumulative sums along a 362-voxel axis leaves ~1e-14
  per class, so BrainWeb at 1.5 mm in float64 raised "fractions sum to more than one" (float32
  hid it: its tolerance is 2e-6). Renormalise by max(total, 1) after resampling.

- **fastMRI brain k-space is (slices, coils, readout x2, phase)** (realism score, 2026-10-10):
  the inverse FFT, then the central rows of the oversampled readout and the central columns of the
  phase direction (to the reconstruction matrix, 384 x 384 over 220 mm), reproduces the files'
  `reconstruction_rss` with correlation 1.0000000. The readout runs anterior-posterior (image
  rows), posterior is up (corrected 2026-10-12: the next entry). The 3 T AXT2 stacks (16 x 5 mm) run
  from the lateral ventricles (slice 0) to above the vertex (slice 15) in all 21 volumes.

- **fastMRI's brain images show the head posterior up; a mean-image correlation cannot tell** (2026-10-12):
  for two days the simulated heads were displayed anterior up against posterior-up real ones. The first
  check correlated mean images (r 0.94 as-is, 0.82 flipped), but an AP flip barely changes a mean oval;
  reading the anatomy by eye failed too (the flattened, dark-banded edge I took for the occiput on the
  pad is the forehead and frontal sinus). What settled it: a head-shape statistic that separated the
  two sets perfectly (the width of the real heads peaks 0.34-0.44 of their length from the top row, of
  IXI's 29 axial T2 heads, oriented by their NIfTI affines, 0.48-0.62 from the front; the values
  mirror each other), and an interior template from IXI (each slice cropped to its head's box,
  central ellipse only): real slices match it flipped in 91 % of 147. Check an orientation against
  data whose orientation is known, with a feature that is not symmetric (`validation/realism/orientation.py`).

- **Unbalanced person-wise folds bias a classifier against the held-out people**: a real-vs-real
  null test gave AUC 0.20 for a histogram logistic regression (CI excluding 0.5) when folds held
  out people without balancing classes; dealing each class's people round-robin into folds moved
  it toward 0.5 but not all the way (0.27). Calibrate small AUC differences with a permutation test.

- **Ask the classifier where it looks before tuning by eye** (2026-10-12): rerunning the realism CNN on images
  masked to bands of distance from the head's edge (the rest zeroed) said at once that the outer 6 px (scalp
  and skull) separated real from simulated (AUC 0.998) and the brain did not (0.79), after days spent on the
  cortex. Then a band's intensity quantiles track a fix in minutes, before a full run.

- **A segmentation label is a contrast claim** (2026-10-12): prepare_ixi called soft tissue deeper than 8 mm
  "muscle", and the simulator gave it muscle's T2 (30 ms, black in a T2 TSE), though the subject's own T2 read
  it at 0.5-0.8 x WM. Check what each label's tissue values predict for the source scan against the source
  scan itself (here: soft tissue unmixed between skin and muscle by its T2).

- **A cropped scan's k-space remembers its echo ordering** (2026-10-12): the realism score keeps the central 128 of
  the real scans' 396 phase-encode lines, so in a linear 16-echo TSE those lines come from ~5 echoes around the
  effective one; our 128-line simulation spread them over all 16 (TE 13-200 ms). Its k-space was lopsided (+ky / -ky
  0.6-0.8 at the edges, the real 0.95-1.01), the point spread complex along the phase axis, and the CNN lost most of
  its edge when both sets were low-passed across that axis. Simulate the acquired matrix's echo-to-line mapping
  (tse_2d(acquired_lines=)), and check a simulated k-space's symmetry against the real one's.

- **Count the resamplings**: an anatomy segmented from a scan resampled onto an atlas grid and then pulled back
  through the pose is interpolated twice; trilinear keeps ~0.6 of the power at an image's Nyquist each time (cubic
  B-spline ~0.9). Our heads had ~half the real fine-structure power and 10 % softer ventricle walls (2026-10-12).

- **An atlas can miss a structure that every real image shows**: neither BrainWeb nor the IXI segmentation has a
  falx (dura 0 at the midline), so the interhemispheric fissure was solid CSF, 20 % brighter at its centre than
  real (2026-10-12; anatomy.add_falx).

- **A segmentation inherits its source scan's artefacts** (2026-10-12): IXI's T2 displaces fat 1.63 mm posterior of the
  water tissue (a water-fat shift along its readout), so our IXI heads had fat at the skin and soft tissue against the
  skull at the back, the reverse of the real layering, and the realism CNN read it there. Check a segmented layer's
  order by direction around the head (fat's depth minus soft tissue's was -2.6 mm at the back, +0.8 at the front,
  -1.6 at the sides): an asymmetry along one axis is the scanner, not anatomy.

- **Measure the thing, not a proxy that mixes two layers** (2026-10-12): "skin to the darkest point of the skull"
  said our posterior scalp was 1.2 mm too thick and led to a compression model; skin to the skull's outer surface said
  the scalp was 1.6 mm too thin and the skull's dark band too wide.

- **A residual fitted per voxel carries the source images' noise, amplified** (2026-10-12): T2 from the ratio of two
  echoes (IXI's TE 8 / 100 ms) turns ~2 % image noise into ~4 % per voxel after 0.7 mm smoothing, and r25's first heads
  had 1.4-2.4 x the real fine texture in the hemispheres. Smooth the fitted map (1.5 mm: log SD 0.19 -> 0.05 on IXI012,
  the structure above 2 mm kept: same-person RMS 0.074 against 0.072), within each tissue so that boundaries the
  labels already carry stay sharp; check a fitted map's power spectrum against the real images' before simulating.

- **Fit a residual only where the model and the source images share geometry** (2026-10-12): outside the skull the
  phantom's fat sits where it belongs (the water-fat shift undone) but the subject's images still show it shifted, so
  the residual "corrected" the scalp layers against misplaced fat and dimmed the posterior scalp peak from 0.65 to
  0.43 of p99 (real 0.83). The residual is now fitted inside the skull only.

- **A correlation across a run's heads is not a parameter's effect** (2026-10-12): in r24 the posterior scalp-to-brain
  depth correlated with each head's drawn pitch (r -0.64, anterior +0.74) and suggested +8 deg instead of -10; the
  same 16 heads as proxy images at fixed pitches -10..+10 moved it by < 1 px (posterior 7.8-8.5, anterior 5.2-6.5).
  The subjects carried the variation. Sweep the parameter on the same heads before changing it.

- **Compare against the same anatomy version** (2026-10-12): r24's heads were v8 (IXI's fat still 1.63 mm posterior),
  the proxies v10 (shift undone); their 1.5 px posterior / anterior disagreement looked like a simulation effect.
  Simulating the v10 head with and without fat's chemical shift moved the edges by ~0.5 px (the expected 0.56 px:
  434 Hz over 128 samples at 10 us); v8 -> v10 moved the posterior dark minimum 5.0 -> 4.5 px (real 4.5).

- **Read the echo train off the real k-space before fitting one** (2026-10-12): in a linear-ordered TSE each echo fills
  a block of adjacent lines (one per shot) and the line energy steps between blocks, so the real AXT2 k-space shows its
  train directly: steps every 19 lines, 21 echoes, ky = 0 in echo 11, at the header's own echo spacing (10.26 ms;
  validation/realism/echo_train.py). Our fitted 16-echo train at TE / 9 = 12.6 ms had bigger steps every 25 lines,
  whose replicas of the head along the phase axis left a plateau of ~1 noise floor beside it (0.87 at 9 px; real
  train 0.28): the background there read 1.19-1.22 x the floor in every run since r21 against the real 1.09, and
  1.03 with the real train. The header had the spacing all along.

- **A residual absorbs whatever its forward model leaves out** (2026-10-12): the PD residual predicted IXI's TE 8 ms
  echo without magnetization transfer, so the subject's own multi-slice MT (white matter darkened most, CSF not at
  all) came out as "PD": CSF x1.21-1.28 and grey matter x1.05-1.09 in every subject, and the simulator added its own MT
  on top (r26: CSF p99 1.28 x r24's). A multiplier that is the same in every subject is the model's gap, not anatomy:
  keep only the within-tissue pattern of a quantity whose forward model is incomplete (re-centred per tissue), or model
  the missing mechanism. T2 from an echo ratio cancels MT and kept its tissue offsets.

- **Partial-volume unmixing needs a pure reference where it unmixes** (2026-10-13): IXI's CSF / grey matter shares were
  unmixed against CSF's intensity from the cores of CSF regions, which near the convexity means the ventricles' (no
  pure CSF within 10 mm). 1 mm sulcal voxels are never pure, so the labels carried 30-50 % more CSF-like voxels than the
  subject's own T2 at MNI z 30-70, and the simulated top slices too much CSF (6.38 x WM vs 5.25) and too little white
  matter. Check a segmentation's fractions against its source image by region (height, depth) before simulating.

- **BrainWeb's heads are MNI-sized** (2026-10-10): the 20 models live in stereotaxic space, so every
  head is template-sized and template-shaped: subject 04 is 174 x 207 mm (breadth x length; ANSUR II
  men 154 +- 5.5 x 200 +- 7.0 mm), its brain 1580 cm^3, and simulated heads at the ventricles were
  264 cm^2 against 189 cm^2 in fastMRI, rounder (AP/LR 1.14 vs 1.31) and all upright. Scale each
  head to an anthropometric draw (``sample_head(size="anthropometric")``) and turn it
  (``yaw_sd``); centre the field of view on it as an operator would.

- **Sub-voxel spins make a grain through a 2D slice's profile** (2026-10-10): with the tissue
  composition exact (``sampling="proportional"``; categorical draws add a salt-and-pepper of their
  own), noiseless images still had a pixel-scale grain. More spins in-plane (1x3x3, 1x4x4) did not
  help; more along z did (3x3x3), because random z positions through the steep 5 mm slice profile
  make every pixel's slice integral noisy. z at subcell centres (``jitter="inplane"``, a quadrature)
  fixes it; a full lattice instead makes horizontal banding. 4x2x2 in-plane-jittered is as smooth
  as 3x3x3 at a third of the time.

- **Image statistics can mislead where the eye does not** (2026-10-10): a high-pass "grain" metric
  said the real scans had more fine structure (0.26 vs 0.17), while the simulated ones looked
  grainier. Both were right: real images have crisp CSF lines, simulated ones had sampling grain
  and few sulcal lines. Look first, then build the number that measures what you saw.

- **Where the real slice stack sits** (2026-10-10): fastMRI ships 16 of the protocol's 34 slices
  without positions; the real slice 0 shows frontal horns, septum, caudate heads and thalami
  (foramen of Monro), ~10 mm below the lateral ventricles' largest cross-section. The head-area
  profile alone cannot separate offset from spacing (a valley of equal fits); anatomy picks the
  offset, and then contiguous 5 mm slices fit.

- **fastMRI headers hold more protocol than the manifest kept** (2026-10-11): `reconSpace` z is
  the slice thickness (5 mm; `encodedSpace` z says 7.5, a padded value, not a gap or a thickness),
  `sequenceParameters` has `echo_spacing` (10.26 ms at TE 113, so TE is echo 11; we played 12.6 ms
  and echo 9 of 16), and `flipAngle_deg` is the refocusing angle (180 in 11 of the 21 3 T AXT2,
  140-179 in the rest). No echo-train length anywhere.

- **Measure layers aligned at a landmark, not at a fixed depth** (2026-10-11): a band 5-7 px below
  the head edge said the simulated calvarium was too dark (0.26 vs 0.35-0.41), and BrainWeb's
  marrow became yellow (fatty) marrow. Aligned at each ray's bone minimum, the same images say the
  opposite (bone 0.27-0.29 vs real 0.16-0.21): the scalp and skull thicknesses differ, so a fixed
  depth samples different layers in real and simulated heads. 0.57 mm reconstructions of six real
  heads show a uniformly dark frontal bone, no bright diploe; the mapping went back to the
  unspecified (trabecular) marrow.

- **p99 normalisation couples the scalp to the brain** (2026-10-11): in the real scans the scalp's
  top decile (0.89) is as bright as ventricular CSF (0.87); in ours the scalp was 0.79 and CSF
  1.02, so CSF set the scale alone and the whole brain came out ~13 % brighter. A dim scalp makes
  a bright brain.

- **The bright rim under the skull is the tell** (2026-10-11): blinded r5 12/12 again, picked by a
  continuous bright CSF rim just inside the skull and speckled bright cortex. Aligned at the bone,
  real and simulated profiles agree from 5 px in; the excess is the first 2-4 px (0.62-0.64 vs
  0.39-0.44 frontal). Not the receive ring (12 cm: rim unchanged, white matter periphery worse),
  not grey-matter T2 (85 ms: no change), not marrow. Thinning BrainWeb's subarachnoid layer by
  1.5-2 mm matches, but removes 16-25 % of the CSF, while BrainWeb's total (245 mL after resizing,
  18.5 % of ICV) is already at or below young adults' (Yamada 2023: 287 mL; older 389 mL). Open.

- **BrainWeb labels the deep nuclei grey matter, and much of the pallidum and thalamus white**
  (2026-10-11): with one grey-matter T2 (96.5 ms) the basal ganglia shone like cortex; at 3 T iron
  puts them near white matter (Krug 2022: 57-76 ms). Labelled by MNI regions: caudate 3.3,
  putamen 4.9 mL per side; thalamus 3.3, pallidum 0.3 mL (the rest stays white matter).

- **A morphological opening can eat the real scalp** (2026-10-11): `lab.head_mask` opens the
  thresholded image twice; in real scans the thin, dim lateral scalp at the low slices does not
  survive it, so real heads measured 143 mm left-right at slice 0 (opened once: 157) and their area
  looked flat for 35 mm (it falls like ours, only slower). The simulated scalp survived, so the
  same function biased real against simulated. Check a mask on the images it is used on.

- **Routine axial slices are tilted to BrainWeb's AC-PC frame** (2026-10-11): sampled heads lost
  area faster with height than the real stacks (slice 8: 0.73 vs 0.79 of slice 0; slice 15: 0.15 vs
  0.21). A vertical stretch filled the top too much; tipping the head 10 deg nose up matched all
  of slices 5, 8, 12 and 15 (0.89 / 0.77 / 0.50 / 0.21 against 0.94 / 0.79 / 0.48 / 0.21).

- **BrainWeb's CSF sits in the wrong place for T2w realism** (2026-10-11): at the vertex levels most
  sulci are closed (two grey banks touching, no CSF) and a continuous CSF rim wraps the brain, at
  0.5 mm as at 1 mm; real 3 T scans show a thin outer layer and long sulcal lines. Its 1.5 T source
  scans could not resolve narrow sulcal CSF, so it went to grey matter. Thinning the rim (a
  calibration) removes one tell and exposes the other: the broken sulcal CSF reads as speckle (2 mm:
  10.3 components per 1000 px of 17 px; real 7.2 of 31). Opening the closed sulci failed twice: on
  the ridge of the white-matter distance (salt along the sulci, at 0.5 mm too) and on a Hessian
  sheet filter (2-3 mm blobs, 62 mL in a 50 mm block); and a third time as "grey matter >= 1.8 mm
  from white matter and >= 1.5 mm from CSF", which also hollows grey matter that is only thick
  where cut tangentially (53 mL; busier cortex at 1.72 mm). Needs surface-based sulci (pial
  surfaces) or another anatomy source. One lead: BrainWeb did keep a trace of the closed sulci.
  At 0.5 mm, grey matter dips to ~0.8-0.9 along their midlines, where CSF is 0.02-0.15 (90th
  percentile inside GM 0.08), in continuous lines. But ridges of that trace are as noisy as
  the others (dotted along nearly all grey matter at a 0.03 threshold), and concentrating it
  locally (power 3, 1 mm window) moves too little mass to show. A Hessian sheetness filter on it
  (0.75 mm, non-maximum suppression along the normal) finds thin, plausible pieces, but sparse
  (3-13 mL in a 50 mm block) and still dashed; at 1.72 mm the change is barely visible.

- **Closed sulci are the lambda-medial axis of the non-white region, not the ridge of its distance**
  (2026-10-11): ridges of the distance to white matter fire on every bump of the white surface
  (salt), Hessian sheets on blobs. The feature transform (each voxel's nearest white point) gives a
  robust test instead: two neighbouring voxels whose nearest white points lie >= 3 mm apart and face
  each other (within 60 deg of opposite) straddle a closed sulcus; 0.5 mm bumps make none. On
  BrainWeb 04 at 0.5 mm it draws continuous sulcal lines like real 3 T T2 (12 s). Two traps: the
  voxels beside the sheet must take their midplane distance from the two banks (distance to the
  sheet plus the sheet voxel's own offset under-filled a 30 deg sheet by 16 %); and with only "more
  than 90 deg apart" a 3 mm deep, 6 mm wide groove in the white surface opened too (and BrainWeb got
  173 instead of 137 mL at 1.3 mm). Widths: Jin et al. 2018 (PATH, n ~800) 1.27 mm at 47 y, 1.49 at
  67; Kochunov 2005 with the same definition has a six times steeper slope (0.71 mm per decade); the
  added CSF brings BrainWeb 04's 254 mL to 322 mL at 25 y and 381 mL at 75 y (Yamada 2023: 287 / 389).

- **Measured sulcal widths over-brighten fastMRI T2w; conserve BrainWeb's CSF instead** (2026-10-11):
  on six matched heads (slices 1-7), brain-interior intensity quantiles over the 15th (~white
  matter) at slices 1-4: real q70 / q85 1.71 / 2.51, unopened 1.73 / 2.55, Jin widths x 0.6 1.91 /
  2.83, x 1.0 2.07 / 3.22 (slices 5-7: real 1.79 / 2.67; unopened 1.70 / 2.27). The real images hold
  about BrainWeb's amount of bright CSF, only in lines instead of a rim and diffuse speckle: 3.9 % CSF
  spread through grey-dominant voxels (38 mL, subject 04) lifts T2w grey matter ~9 %. Gathering each
  grey/CSF mixture to its larger share and opening the sheets to the width that keeps the CSF volume
  (plus Jin's widening, 0.011 mm per year beyond 30) gives 1.67 / 2.42 and 1.78 / 2.36. Jin's widths
  are for five major sulci measured on 1 mm T1 (BrainVISA), not every sheet a medial axis finds.
  Opening also brings BrainWeb's grey/white volume (1.44, subject 04; 20-subject mean 1.63) to
  1.23 and its pial-to-white distance from 2.92 to 2.35 mm (adult cortex ~2.5).

- **Segmenting IXI heads: what the EM gets wrong** (2026-10-11): with BrainWeb priors aligned by an affine
  only, a 12-class EM on (log T1, log T2) places the brain well (the subject's intensities decide when
  the CSF/GM/WM priors are shared at 8 mm) but swaps class identities in the thin head layers ("marrow"
  came out dark, "skin" took the scalp fat): rules on bias-corrected intensities relative to the
  subject's white matter do better there. IXI's T1 (3D, sagittal) covers the head, its T2 (axial,
  112 x 1.2 mm) stops ~10 mm short of the vertex. Some scans carry skin markers that a 3 mm closing
  welds to the scalp; above the ears an 8 mm disc opening per axial section removes them. One HTTP
  stream from the IXI server ran at 0.65 MB/s, eight byte ranges at ~8 MB/s together.

- **The MNI frame is taller than real heads** (2026-10-11): IXI subjects registered to BrainWeb's MNI
  frame scale by ~0.85 vertically but ~0.9-1.0 across, so BrainWeb heads sized from ANSUR breadth and
  length (height as their geometric mean) stay tall, and parameters fitted on them (the -10 deg pitch,
  the 10 mm stack offset) absorbed that. IXI heads at their own size narrow a slice faster than the
  real stacks under the same settings.

- **EM posteriors are not partial volumes** (2026-10-11): a mixture model's posterior is close to a step in
  intensity, so a boundary voxel a little brighter than grey matter reads as mostly CSF; in IXI012's grey/CSF
  boundary voxels the posteriors hold 103 mL of CSF, linear unmixing of the T2 52 mL. Unmixing three classes
  from two channels and the sum is exactly determined and turns noise into spurious white matter (GM/WM
  1.48 -> 0.90); pairwise unmixing needs clean references (sulcal "pure" CSF is partial volume; ventricle
  cores are not) and local ones (residual receive bias). Without an independent CSF volume to check
  against, the posteriors stayed; the simulated images' high slices still carry too bright a tail.

- **Read the score's controls before its AUC** (2026-10-11): r16 (10 heads from 8 BrainWeb subjects,
  two heads lost mid-run on the swapping mini) scored real-vs-simulated CNN AUC 0.07 [0, 0.19] and
  histogram 0.06: inverted, not a success. Its simulated-vs-simulated control (4 subjects against 4)
  gave 0.008 where 0.5 is expected; person-split cross-validation over a handful of people anti-learns
  (each held-out person sits closer to the other class's training people). Score runs need ~15+
  distinct subjects, and a control far from 0.5 voids the headline number.

- **Sub-voxel spins leave a Monte-Carlo grain a CNN reads** (2026-10-11): random in-plane positions
  (and random subcells per tissue) are a point sampling of the anatomy; through the 1.72 mm PSF they
  add pixel-scale texture that falls as 1 / sqrt(spins): flat-brain Immerkaer sigma 0.026 at 5.4 M
  spins per slice, 0.0136 at 21 M, the anatomy itself 0.011, real 0.014. A CNN on high-pass
  residuals alone told real from simulated at AUC 0.985. Regular positions band (aliasing), a
  shared per-voxel shift blotches. Count the grain before trusting a texture comparison.

- **Open: why the real scalp is as bright as CSF** (2026-10-11): in the fastMRI 3 T T2 scans the
  scalp's top decile is 0.93 of ventricular CSF; the tissue values give fat / CSF ~0.63 at TR 6.5 s
  (fat T2 148 ms from CPMG, CSF partly saturated), and ours measure 0.68-0.73. A receive ring at
  the coil's surface (12.5 cm, 1/r) brightens the scalp but also the cortex (GM/WM 1.73 vs real
  1.50), and the full-wave 8-loop map makes the periphery 1.4-2x brighter than the real images
  allow; a blinded test went back to 12/12 with it. Candidates left: a steeper near field than 1/r
  with 3 T central brightening, fat's effective T2 in a 10 ms-spaced train (one 3 T study reports
  371 ms), ventricular CSF dimmed by pulsation.

- **A resized grid's margin is empty, not air** (2026-10-11): pull-back resampling fills points
  outside the source grid with zeros in every class, background included, so `1 - background`
  calls the margin tissue and `fill_holes` then swallows the air around the head. Use
  "sum of tissue classes > 0.5" (`_head_mask`).

- **A long simulate_heads run grows** (2026-10-11): one process over many realistic heads reached
  15 GB on the 16 GB mini (allocator caches across heads), which swapped with another user's 5 GB
  app and slowed slices 3-5x. It now frees memory between heads; for long batches, one process per
  head also works.
