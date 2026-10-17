# 2026-10-12 — Upside-down heads, the coil and the scalp

- **The simulated heads were upside down against the real ones** (6eccb75). Measuring head shape for the
  bright frontal rim, the real heads' width peaked 0.34-0.44 of their length from the top row and IXI's axial
  T2 heads (oriented by their NIfTI affines) 0.48-0.62 from the front, with no overlap between 21 and 29
  people, and the simulated heads matched IXI. No tilt closes that gap (even 40 deg nose-down left IXI heads
  at 0.44-0.47), so I checked orientation itself: an IXI interior template matches 91 % of 147 real slices
  flipped up-down (r 0.35 vs 0.22) and the simulated ones as displayed (0.32 vs 0.22). fastMRI shows the
  head posterior up; `to_real_orientation` showed ours anterior up since the first score (a mean-image check
  could not see an AP flip; I had read the forehead's dark band as the occiput on the pad). Fixed;
  `validation/realism/orientation.py` keeps the check. r20 rescored, same k-space: CNN AUC 0.998 [0.99, 1.0]
  (was 0.959), histogram 0.927 (orientation-free), controls 0.44 / 0.46. Every earlier front/back finding
  compared our front with the real back: the "bright frontal rim" of r20 is at our front against the real
  front's 0.31-0.33 (still too bright: 0.48-0.52), and the real back is bright.
- **The head lies on the coil** (5600381). With matched orientation the real images fall from back to front
  (white matter q25: front / back 0.89; outer ring 0.71) and ours did not (1.00, 0.82): a head resting on the
  posterior elements. `coil_offset` puts the receive ring's back a gap behind the occiput; re-weighting r20
  by the rings' RSS ratio bracketed the real numbers with a 14 cm ring and a 20-30 mm gap (preset: 25 mm).
  Simulated on IXI012: 0.83 / 0.70.
- **Scalp: IXI's own fat, unmixed against the median** (e40f17d). Next to the real heads, every simulated
  one had a uniform white scalp halo: the preset laid Li 2024's 4 mm vertex fat layer everywhere. Integrated
  scalp signal over white matter (back / side / front): fastMRI 15.1 / 10.3 / 9.8; IXI012 with the layer
  20.3 / 17.1 / 13.1, with its own segmented fat 8.6 / 6.7 / 6.1, because prepare_ixi's pure fat was the
  98th percentile of T1-bright tissue's T2 (3.09 x WM; median 1.41) and unmixed subcutaneous fat to a third.
  With the median: 14.6 / 9.5 / 6.9 (fat 120 -> 274 mL). IXI's raw T2 itself shows a dim scalp (5-6 over
  WM: the late echo of a dual-echo TSE), so it is no direct reference for fastMRI's fat. The preset keeps IXI
  heads' own scalp; BrainWeb heads keep the layer.
- Dead end: a temporalis rule in prepare_ixi (only 2 mm of skin where the soft tissue under the surface is
  thicker than 10 mm) fired across the upper head too, where the outer table is detected in patches; with the
  heads' own fat it is moot. The lateral scalp, with the own fat: 9.5 against 10.3.
- **Where the CNN looks** (scratch regions.py: score.py's test on images masked to a band of distance from the
  head's edge, the rest zeroed). r20, orientation fixed: the head's outer 0-6 px alone (scalp and skull) CNN
  0.998, histogram 0.984; 6-14 px 0.81 / 0.73; deeper than 12 px (brain) 0.79 / 0.79. The tell is the outside
  of the head, not the brain.
- **The dark band at the skull was muscle** (03d0513). The outer band's quantiles in our heads were too dark
  at the bottom (q10 0.056 against 0.136): the dark band at the skull ~4 px wide against ~1.5. The segmented
  bone is thin (IXI012 1.7-5.6 mm along rays); next to it lay 3-7 mm of "muscle", prepare_ixi's label for soft
  tissue deeper than 8 mm, which at T2 30 ms is black at TE 113 ms. That tissue reads 0.53-0.80 x WM in the
  subject's own T2 at every depth (skin-like 0.77, muscle 0.21 at IXI's TR 5725.8 / TE 100 ms by our tissue
  values), so soft tissue is now unmixed between skin and muscle by its T2. IXI012 (muscle 433 -> 290 mL):
  outer band 0.157 / 0.227 / 0.382 / 0.623 / 0.833 against the real 0.136 / 0.214 / 0.375 / 0.616 / 0.859
  (v6 0.056 / 0.130 / 0.227 / 0.470 / 0.764); the side and back profiles close to the real ones, IXI012's
  forehead short of fat.
- The v6 preparation (median fat only) was stopped half-way; 24 heads re-prepared as v7 (both changes) for r21:
  orientation, coil, own scalp fat, soft tissue by T2.
- **r21 (16 IXI v7 heads, the same 16 subjects as r20; orientation fixed, head on the coil's posterior elements,
  own scalp fat with the median reference, soft tissue by T2)**: intensity W1 0.017 (r20 0.042), quartiles 0.222 /
  0.314 / 0.506 vs real 0.216 / 0.299 / 0.491 (r20 0.230 / 0.339 / 0.573), histogram AUC 0.735 [0.59, 0.87] (r20
  0.927), but CNN AUC 1.0 at every slice; controls 0.44 / 0.40. Scalp signal over WM per head inside the real
  range in every sector (back 10-20 vs real 9-23, side 8.5-10.4 vs 7-17, front 7.4-9.4 vs 6-17). By band: the
  outer 6 px CNN 0.944 / histogram 0.89 (r20 0.998 / 0.984), 6-14 px 0.84 / 0.55, interior 0.74 / 0.70: no
  single region carries the cue any more.
- **Phase-encode ordering.** Real AXT2 k-space is symmetric in ky (+ky / -ky 0.95-1.01 from 10 to 190 lines out);
  ours was lopsided (0.6-0.8 at the edges): 128 lines over all 16 echoes (TE 13-200 ms), where the real scans
  encode 396 lines (all 21 headers) and the central 128 come from ~5 echoes. Low-passing both sets across the
  phase axis dropped r21's CNN to 0.92 (across the readout: 0.99). Signal beside the head (phase direction) was
  1.22 x the noise floor against the real 1.09. Two options in tse_2d (6a1a1bf, 88c4c20): reverse-centric
  (symmetric bands, edges first) and acquired_lines (the n lines on the echoes they have in the full linear scan;
  simulate_heads --ordering linear_full, 25 shots, ~2.5x the time). Reverse-centric on IXI012/013: beside the
  head 1.08 (real 1.09), but no more phase-axis detail.
- **Where the remaining fine-detail gap is**: real/sim spectral power along the phase axis rose to 4.6 at
  Nyquist in a central patch but only ~2-2.8 (both axes alike) in patches inside the hemispheres; the central
  excess is the interhemispheric fissure, which in ours is brighter and wider (peak 4.71 x WM, +-1 px 3.59, vs
  real 3.89, 2.68): solid CSF, because neither BrainWeb nor the IXI heads have a falx. White-matter texture
  matches (relative SD 0.087 vs 0.088, same band powers); ventricle walls are ~10 % softer than real (0.418 vs
  0.459 per px: the anatomy's resampling chain). add_falx (0a96dc7): the falx as a 0.91 mm dural sheet in the
  fissure (Walsh 2021), above the callosum's free edge (Kayalioglu 2004), 4-5 mL per head.
- Dead ends here: a GRAPPA hypothesis for the real phase-axis detail (the real k-space noise is flat across
  lines, even / odd 1.00) and physiological ghosting (beside the head the real images carry less, not more).
- **r22 (r21's 16 subjects and seeds + the falx + reverse-centric ordering)**: CNN AUC 0.9996, histogram 0.749, W1
  0.023, quartiles 0.225 / 0.328 / 0.521 (real 0.216 / 0.299 / 0.491); controls 0.44 / 0.41. The fissure now peaks at
  3.92 x WM (real 3.89; r21 4.71), the background beside the head 1.10 x the noise floor (real 1.09; r21 1.22), the outer
  band's quantiles 0.119 / 0.187 / 0.332 / 0.613 / 0.851 (real 0.136 / 0.214 / 0.375 / 0.616 / 0.859). By band, the
  outer 6 px dropped to CNN 0.83 (r21 0.94) but the interior rose to 0.87 (0.74), and low-passing across the phase
  axis no longer helped the classifier (0.998): reverse-centric's early echoes at the k-space edges brighten
  tissue texture and lift the median. Not the scanner's ordering either; r23 (the 396-line linear ordering) runs.
- **The stack climbs too fast** (801752f): at slices 5-7 our heads narrowed faster than the real ones (slice 7 area
  6064 vs 6730 px) and their interior held less white matter (share 0.32-0.37 vs 0.44-0.47; the grey / white
  mode ratio itself matches, 1.17), so the high slices showed extra cortex. Proxy images (each class's spin-echo
  signal, 5 mm slabs, 1.72 mm pixels; at 10 mm they reproduce r22's ventricle drop) put the best stack offset at
  12.5 mm: ventricle cross-section over slices 0-3 211 / 240 / 250 / 223 px against the real 181 / 220 / 272 / 218.
  The 15 mm offset rejected yesterday had been judged by eye on upside-down heads. At slices 4-5 the real ventricles
  stay larger (an older clinical population).
- **Cubic resampling** (82a79cb): deform(order=3) and prepare_ixi onto(order=3); 24 heads re-prepared as v8 (GM / WM /
  CSF volumes within 1 % of v7, skull +10 %, all pass QC); 47 more subjects being prepared as v8 for a larger pool.
- **r23 (the 396-line linear ordering) stopped after 3 heads**: on those heads against r22's, the phase-axis fine
  structure in a central patch came within 1.25-1.75 of real power at high k (reverse-centric 2.0-3.9), hemisphere
  texture slightly closer; the background beside the head 1.17 x the floor (reverse-centric 1.08, real 1.09: a halo of
  the short-T2 scalp tissue along the phase axis, not ghost copies). The 396-line ordering goes into r24.
- **The coil noise is coloured** (9b2abbd): the real scans' background is wider and more skewed than independent
  equal-sigma noise in 16 coils gives (SD / median 0.163, skew 0.41 vs 0.126 / 0.20): their coil SDs span 0.66-1.38
  of the mean and correlate (|r| 0.13-0.17). score.py now estimates each real scan's coil covariance from its image
  corners and colours the paired simulation's noise with it (background 0.158 / 0.35). r21 rescored: CNN AUC 1.0 ->
  0.971 [0.92, 1.0] (by slice 0.96-0.98), controls 0.42 / 0.43; histogram 0.735 -> 0.841 (CIs overlap). The quiz
  pages use the same noise.
- **r24 running** (v8 heads with cubic resampling and pull-back, falx, slice 0 12.5 mm below the ventricle bodies,
  396-line linear ordering; r21's subjects and seeds, ages 20-82, median 52): 14.7 M spins per slice; the mini held
  14 GB and swapped, so simulate_heads --serial (14716ce) builds one slice's phantom at a time (6 GB); ~135 s per slice.
- **r24's first two heads against r21's**: hemisphere texture in the top bands 1.10 / 1.01 of real power (r21 0.65 /
  0.49), ventricle walls 0.453 per px (real 0.459; r21 0.417), the fissure 3.91 / 2.68 at its centre / +-1 px (real
  3.89 / 2.68), phase-axis detail in the central patch ~1.0 of real (r21 up to 3.4). By eye the slices are now hard to
  tell apart; the white matter at the upper slices looks a little mottled.
- **What is left** (two heads): coarse texture 1.4 x real power at 0.3-0.45 of Nyquist; the background beside the head
  1.18 x the floor (real 1.09); the outer band dark at its q25-q50; and at slices 5-7 less of the interior near the
  white-matter mode (0.35 vs 0.44), with the central white matter darker than the peripheral (0.865 vs real 0.93).
  Not the receive coil (3D element rings move the ratio by < 0.01), not the slice geometry (proxy images: 0.39-0.42),
  not IXI's anatomy (its own T2 at those levels: share 0.49-0.59, centre / ring 1.03-1.08), not B1+ alone (the map's
  central overflip is stronger at the low slices, which match). The within-tissue texture (5 % T1/T2 at a 10 mm
  correlation length, an M4 assumption) sits at the scale of the coarse excess; ablations (--no-texture, --b1 uniform;
  01c39ba, 858bbf4) queued after r24. Proxy fit of the pitch at the 12.5 mm offset keeps -10 deg (area rms 0.017).
- **Same-person residual (step 1 of a fitted residual layer)**: IXI012's real T2 (bias-corrected, over its white-matter
  median, on the phantom's 1 mm grid) against the T2 its phantom predicts (each class's spin-echo signal at IXI's TR
  5.726 s / TE 100 ms with our 3 T tissue values, deep grey and falx as simulated; scratch same_person.py). IXI's
  noise: background SD 0.020 x WM. In the brain |log(real / predicted)| median 0.059, RMS 0.160, still 0.071 after
  2 mm smoothing: structure, not noise. Pure white matter: RMS 0.077 raw, 0.046 / 0.023 after 2 / 8 mm smoothing,
  its centre 1-5 % brighter than its periphery at MNI z 25-45 (periventricular and central white matter), the direction
  our high slices lack (centre / ring 0.86-0.90 vs fastMRI 0.93). By class (fraction > 0.9, real / predicted): GM 1.01,
  WM 1.00, CSF 0.92, putamen 1.27, thalamus 1.15, caudate 1.07, dura 1.59, skin 1.15, muscle 0.89, fat 0.67. Outside
  the skull |log ratio| median 0.14 (layer boundaries). Caveat for fitting: some of this is sequence physics, not
  tissue (fat 0.67: IXI's echo spacing does not J-suppress fat as our CPMG T2 assumes), so fat needs its own model and
  T2 / PD need IXI's PD echo to separate.
- **r24 (v8 heads with cubic resampling, cubic pull-back, falx, slice 0 12.5 mm below the ventricle bodies, 396-line
  linear ordering, coloured coil noise; r21's subjects and seeds)**: intensity W1 0.0155 (the closest yet), quartiles
  0.221 / 0.313 / 0.501 vs real 0.216 / 0.299 / 0.491, histogram AUC 0.826 [0.69, 0.93], but CNN AUC 0.9996 (r21
  rescored with the same noise: 0.971); controls 0.44 / 0.42. The targeted measures all moved toward the real ones
  and the classifier still separates the sets: it finds whatever systematic difference remains, so it says where to
  look next rather than how close we are.
- **Where the CNN looks (r24, occlusion of 16 px patches in held-out images per person-wise fold)**: for the simulated
  images the evidence sits at the back of the head (outer 0-6 px: logit drop 300 / 1000 per px with the patch mean,
  366 with only a 2 px blur, against 79-192 at the sides and front; rim 294; interior 144). The fine structure of the
  occipital scalp and skull gives them away.
- **IXI's fat was shifted** (966d8a0). At 0.57 mm the real posterior scalp has a dark dermis outside, fat 5 mm in, fat
  running onto a ~3 mm dark skull line (soft tissue 6.6 mm [5.6-7.6], skull outer surface to its darkest point ~2.0
  mm); ours had fat at the skin and soft tissue against the skull (soft tissue 5.0 mm, then ~4.8 mm to the darkest
  point). Fat's depth below the skin minus skin and muscle's in the prepared heads: -2.6 mm at the back, +0.8 at the
  front, -1.6 at the sides: IXI's T2 displaces fat 1.63 mm posterior (all 24 subjects, IQR 1.1-2.1), a water-fat
  shift along a front-back readout. Undone in prepare_ixi before fat, marrow and soft tissue are assigned. 24 heads
  re-prepared as v9. Dead end on the way: pressing the occipital scalp against the pad (compress_occiput), from an
  edge-to-darkest-bone measure that mixed scalp and skull; the soft tissue alone is thinner than real, not thicker.
- **Ablations at full quality** (IXI012, seed 200, slices 5-7): without texture the central / peripheral white matter
  stays 0.871 (with 0.869; real 0.931), with uniform B1+ 0.900; white-matter mode over the median 0.69 / 0.72 / 0.75
  (real 0.83). The B1+ map explains about half of the central darkening; the rest matches the same-person residual
  (IXI's central white matter 1-5 % brighter than one white-matter class allows), a tissue property.
- IXI's PD volumes fetched (c55897f; 185 Hammersmith subjects, 1.3 GB) for the fitted residual layer.
- **A residual layer fitted to each subject's own scans** (28991d4). prepare_ixi resamples IXI's PD (the dual-echo
  TSE's TE 8 ms echo) with the T2's transform and fits per voxel the T2 and PD multipliers that make the simulator's own
  tissue mixture (deep grey, falx as simulated) reproduce both echoes: T2 from the echo ratio (scale from ventricular
  CSF), PD from the first echo over white matter; water tissue only (not fat, not CSF's unmeasurable T2); 0.7 mm
  smoothing; a cubic trend fitted over pure white matter removed, which also removes the two-point fit's absolute level
  (IXI012's white matter reads 1.18 x our 72 ms: stimulated echoes) and its B1+ pattern (+-15 % base to top; a 20 mm
  high-pass instead left halos around the nuclei); clipped to x0.61-1.65. IXI012: deep grey PD x1.08-1.13, T2
  x0.98-1.09, grey matter PD x1.06, scalp soft tissue T2 x1.52 / PD x0.68. Same-person check (scanner trend removed
  from both): brain |log ratio| median 0.064 -> 0.032, 2 mm-smoothed RMS 0.080 -> 0.072. 24 heads re-prepared as v10
  (fat shift undone + residual maps) for r25.
- **r25 (v10 heads with the residual) stopped after two heads** (IXI012 seed 200, IXI445 seed 210, against r24's same
  heads): the brain moved the right way (central / peripheral white matter 0.873 -> 0.907, real 0.931; white-matter
  share at slices 5-7 0.38 / 0.31 / 0.34 -> 0.47 / 0.34 / 0.42, real 0.47 / 0.44 / 0.44; posterior lumpiness 0.071 ->
  0.054, real 0.042), but the hemispheres' fine texture rose to 1.4-2.4 x real (r24 ~1.0: the echo ratio's noise in the
  maps on top of the assumed texture) and the posterior scalp dimmed (peak 0.65 -> 0.43 of p99, real 0.83: the
  residual fitted the phantom's unshifted scalp fat against the subject's shifted images).
- **Residual v11** (9b8d0f8): fitted inside the skull only, smoothed 1.5 mm within each voxel's largest tissue (log SD
  on IXI012 0.19 -> 0.054 for T2, 0.23 -> 0.065 for PD; same-person 2 mm-smoothed RMS 0.074 vs v10's 0.072 and labels
  only 0.080), fractions bit-identical to v10; simulate_heads drops the assumed GRF texture when the residual is on (the
  fitted maps are the subject's own within-tissue variation). 24 heads re-prepared as v11 for r26.
- **The posterior layers, with r26 waiting for its heads.** At the back of the head, edge-aligned profiles (half the
  scalp peak, 1.72 mm pixels, slices 3-7) put r24's dark minimum at 6-6.5 px and the brain (0.4 of p99) at 9.5 px
  against the real 4.5 and 7.0, while the front read 4.0 against 6.5 and the sides matched (6.5 / 6.5). Not the
  labels: IXI012's own T2 on the MNI grid has the dark zone 7-10 mm deep and its labels reproduce it to ~1 mm. Not the
  pitch: across r24's heads the depths tracked the drawn pitch (r -0.64 / +0.74), but proxy images of the same 16
  heads at fixed pitches -10 to +10 deg (each class's spin-echo signal, 5 mm slabs, the receive ring's RSS; scratch
  pitchfit3.py) moved them < 1 px, and the head-area rms stays best at -10 (0.027 vs 0.033-0.039): pitch kept. Mostly
  the v8 anatomy: IXI012 simulated on v10 (fat shift undone) moves the posterior minimum 5.0 -> 4.5 px and the brain
  8.0 -> 7.5; with fat's chemical shift switched off (all peaks at 0 ppm, scratch sim_noshift.py) the edges move
  another ~0.5 px, the expected 0.56 px (the real scans' shift direction is not settled by one head). The skull band
  stays darker than real (IXI012 0.07, r24 0.12-0.16, real 0.19-0.27). The fastMRI headers (Biograph mMR, 2.89 T)
  give a 384 x 396 encoding, echo spacing 10.26 ms, no readout bandwidth.
- The v11 preparation thrashed the laptop (18 GB swap with the pitch fits running); the last 17 heads moved to the mini
  (~250 s each, two jobs).
- **The real echo train, from the k-space** (f3de52e): the AXT2 line energy (summed over coils and readout, log minus
  a 15-line running median, 21 volumes x slices 1-7) steps every 18.9 lines, at ky -161, -142, ..., -28, -9 and +10:
  each echo fills a block of 19 lines with ky = 0 in the middle of one, so 396 lines over 21 echoes (19 shots) and
  ky = 0 in echo 11; the headers' echo spacing (10.26 ms at TE 113, 9.56 at TE 105) gives TE / spacing = 11.0 for
  both. Ours was a fitted 16-echo train at TE / 9 = 12.6 ms (25 shots): 23 % longer spacing and 30 % fewer refocusing
  pulses per slice (the neighbours' trains drive MT). `simulate_heads --train header` plays the real one (opt-in,
  with --ordering linear_full); validation/realism/echo_train.py records the analysis; the manifests now carry
  echo_spacing_ms. The background beside the head along the phase axis still reads 1.19 x the noise floor on the v10
  IXI012 head (real 1.09), the readout direction 1.00 (real 1.02).
- **r26** = r24's 16 subjects and seeds on v11 heads with the residual (no assumed texture) and the real echo train;
  the laptop first simulates IXI012 (seed 200, slices 3-7) with the real train and no residual against the fitted
  train's run, to separate the two changes.
- **The real echo train removes the phase-axis halo** (IXI012, seed 200, slices 3-7, v10 fractions, no residual; only
  the train differs): noise-free signal beside the head along the phase axis 2.06 / 0.62 / 0.42 / 0.28 x the paired
  noise floor at 1 / 2 / 5 / 9 px (fitted train 2.34 / 1.40 / 1.09 / 0.87; the readout direction 0.46-0.11 for both);
  the noisy background beside the head 1.03 x the floor (fitted 1.19, every run since r21 1.18-1.22; real 1.09). The
  fitted train's 25-line blocks with bigger steps (12.6 ms spacing) put replicas of the head along the phase axis.
  Scalp profiles and the GM / WM mode ratio unchanged (1.17, real 1.17); the white-matter mode over the interior
  median 0.70 (fitted 0.72, real 0.83) — the slice 5-7 composition gap the residual addresses in r26.
- **r26 stopped after two heads** (seeds 200-201 against r24's): the phase-axis halo gone (beside the head 1.03 x the
  floor, r24 1.18, real 1.09), the front / back ring gradient 0.70 (r24 0.45, real 0.71), the front scalp over WM 8.83
  (7.85, real 9.79); but CSF too bright against the brain (interior p95 / p99 3.97 / 5.22 x the median, r24 3.14 /
  4.09, real 3.23 / 4.04; fissure peak 4.36, real 3.89), the outer band dimmer under the higher p99 (q75 / q90 0.518 /
  0.717, real 0.616 / 0.859) and hemisphere texture 1.3-2.0 x real. The train alone (IXI012 A/B, no residual) raised
  CSF's p99 by 7 % and texture by ~6 %; the rest was the PD residual: CSF x1.21-1.28 and grey matter x1.05-1.09 in all
  subjects, IXI's own MT read as PD (1d90f18: PD multipliers re-centred per tissue; T2's kept). 24 heads re-prepared
  as v12 (mini 18, laptop 6) for r27 = r26 on v12.
