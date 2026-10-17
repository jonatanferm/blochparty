# 2026-10-13 — Where the gap sits now (`validation/realism/gap.py`)

Diagnostics on r27's scored images (147 real from 21 scans, 112 simulated from 16 IXI heads, slices 1-7), to decide
the next realism steps; the three scratch analyses became `validation/realism/gap.py` (features, polar, structure,
controls).
- **The benchmark's own resolution** (`controls`): real vs real split by scanner model scores 0.52 / 0.56 (Skyra vs
  Prisma + mMR, two CNN seeds) and 0.46 / 0.42 (Prisma vs Skyra); by echo spacing 0.58 / 0.46 (one scan against
  twenty). The real set's scanners are not told apart, so 0.875 is signal. But the CNN's own seed moves the full
  number: seed 1 gives 0.927 (seed 0 0.875), and against the 11 Skyra scans alone the simulations score 0.99 / 0.94.
  With 21 against 16 people the interval is +-0.12 and the seed adds +-0.05: the next steps need a bigger benchmark
  (more heads, more scans, seeds averaged) before changes below ~0.1 can be read.
- **Per scan and per head**: 18 of 21 real scans score < 0.2 (three Skyra scans 0.93-0.96); the simulated heads are
  bimodal: ixi150 / 277 / 013 / 216 / 130 at 0.07-0.26, ixi255 / 334 / 511 / 167 / 538 / 146 / 205 at 0.89-1.00. The
  caught heads are mostly the older women (58-82 y) with the most CSF (share 0.18-0.27), but ixi277 (61 y) passes and
  real scans with as much CSF (0.22-0.24) score as real: it is how the CSF-rich heads look, not how much CSF they have.
- **Features** (AUC of one feature alone, > 0.5 larger in the simulations): the head centroid row 63.2 +- 1.9 against
  the real 60.4 +- 2.6 (AUC 0.83): our heads sit 2.8 px (4.8 mm) further anterior in the field of view (the real
  operators centre the FOV 6 mm behind the head's centroid); the interior p99 1.04 against 0.95 (0.76): in the real
  scans the scalp fat sets the p99 scale above the CSF, in ours the CSF is the brightest; posterior scalp 0.53 vs 0.47
  (0.68) and anterior scalp 0.36 vs 0.45 (0.28): a posterior / anterior ratio of 1.48 against the real 1.03, the
  receive ring behind the head. Area 0.95 x real (IXI volunteers' heads are smaller), scalp 0.41 vs 0.44, the band
  minimum 0.125 vs 0.140. Within the simulations the CNN score tracks the CSF share (Spearman +0.33) and nothing else
  above 0.25.
- **Polar map** (angle x radius / head radius, mean over images): sim / real 0.72-0.89 at the posterior 0.80-0.94
  (the deep dark band), 1.31 at the posterior edge (0.98-1.02), 0.83-0.93 at the sides' and front's 0.90-1.02 (the
  dim scalp), 1.08-1.14 at the anterior 0.80-0.90; inside 0.8 of the radius everything is within 2-10 %.
- **Structure**: interior power by band relative to the real mean: all simulations 1.16 / 1.28 / 1.25 / 1.13 / 1.03 /
  0.90 (0.05-0.15 ... 0.8-1.0 of Nyquist); the caught heads 1.36 / 1.40 / 1.36 / 1.22 / 1.10 / 0.96 at CSF share 0.196,
  where the real scans with CSF share > 0.17 (mean 0.235) carry 1.39 / 1.29 / 1.29 / 1.24 / 1.25 / 1.21. At a matched
  CSF share ours have ~15-20 % too much coarse structure (0.15-0.45) and ~10 % too little at the top band: the sulcal
  CSF of the older heads is rendered coarser and smoother than real. CSF-bright components under the edge: 14.3 per
  1000 px, mean 9.9 px, bright fraction 0.137 (real 13.9, 11.5, 0.152): count right, pieces smaller.
- The scratch `profile.py` in the mini's data directory shadowed the stdlib again (LESSONS, 2026-10-11); scripts now
  run from their own directory.
- **Not the receive ring (q33)**: the v14 heads with a 0.5 m ring centred on the head (`--coil-radius 0.5
  --coil-gap-mm 405`, a flat receive) keep the posterior / anterior scalp at 1.49 (posterior 0.51, anterior 0.34; real
  0.47 / 0.45), so the asymmetry is the anatomy (our anterior scalp is dim), and the brain reads 20 % too bright against
  the scalp's p99 (quartiles 0.232 / 0.288 / 0.438, white-matter mode 0.229; real 0.197 / 0.239 / 0.365, 0.194): the
  14 cm ring behind the head has been compensating a scalp ~20 % too dim against brain, or a brain too bright. CNN
  0.825 [0.65, 0.96], W1 0.034 (q29, the same heads with the ring: 0.809, 0.024).
- **The anterior scalp is the preparation**: along rays at MNI z 10-45 mm in eight v14 heads (scratch scalp_ap.py),
  the frontal scalp holds 0.9-2.1 mm of labelled fat (IXI334 4.2) in 3.7-7.9 mm of soft tissue, the occipital scalp
  2.5-4.3 mm in 5.3-10.3; the frontal bone band is 0.8-2.6 mm thick where the occipital is 1.8-6.6 (the forehead bone
  is ~6 mm in life). Too little frontal fat makes our anterior scalp 0.36 against the real 0.45; too little frontal
  bone leaves brain and scalp where the real dark band is (the polar map's 1.08-1.14 at the anterior 0.80-0.90).
  Both sit in prepare_ixi's unmixing at the front, to be checked against IXI's own T1 (fat bright, bone dark).
- **What the forehead is in IXI012's own scans** (raw T1 and T2, midline rays from the skin inward at three levels
  above the orbits): skin and fat 2-3 mm (T1 0.5-0.8 x WM), soft tissue 3-4 mm, then a 4-5 mm band bright in T1
  (0.5-0.7) and intermediate in T2 (0.4-0.5) between two dark tables, then CSF: the frontal diploe. Our rules call bone
  only what is dark in both (r1 < 0.35, r2 < 0.45), so the diploe becomes "skin" (7.0 mm of skin on the anterior rays
  against 5.9 at the back; skull 2.4 against 4.5), rendered at skin's 0.77 x WM where the real frontal bone is dark.
  At the occiput the diploe reads 0.2 in T1 and 0.4-0.56 in T2 and comes out half skull, half skin. The frontal
  scalp's fat itself is thin in the T1 (~2 mm), so the anterior scalp deficit is partly real anatomy (IXI's young
  volunteers against fastMRI's patients) and partly the shift-undo at the front. Next: a diploe rule (a band between
  two dark tables, T1-bright) labelled marrow, with marrow's T2w brightness set from the real scans' calvaria
  (frontal dark, occipital brighter: the aligned profiles).
- **Not the bound pool either (q34)**: q33's flat receive with MT off (`--no-mt`, new): white-matter mode 0.332 against
  0.229 with it, so the simulated incidental MT of the 16-slice train takes 31 % off the brain, inside the 30-45 %
  multi-slice MTR reported at 3 T (REALISM.md, [21]); the brain's 20 % excess over the scalp under a flat receive is the
  scalp fat's dimness against brain (the J-coupling route was negative; what is left: adipose water and PD, the
  diploe below), not missing MT. (CNN 0.897, histogram 0.993, W1 0.088: the ablation, not a candidate.)
- **The real calvarium at 0.57 mm** (scratch calv_real.py: 12 scans, slices 2-5, rays +-30 deg, in units of each
  slice's white-matter mode): posterior, from the edge: 0.82 at the skin, the scalp peak 5.0 at 5.7 mm, the bone
  valley 0.81 at 10.3 mm (one valley ~2 mm wide, no separate diploe maximum in the median), then 1.1 / 1.4 / 1.7 / 2.0
  at 11.5-15 mm (cortex and CSF); anterior: 1.09 at the skin, the scalp peak 3.1 at 2.3 mm, the valley only 1.19 at
  5.7 mm, then a slow rise 1.4-2.0 over 7-12 mm: the 5 mm slab cuts the sloping forehead obliquely and smears
  scalp, bone and brain together. Our classes at this protocol (spin-echo signal over WM): fat 3.21, marrow 0.73, skin /
  dura / connective 0.71, muscle 0.16, dermis / skull 0. Relabelling the frontal diploe skin -> marrow would change
  nothing (0.71 -> 0.73): the anterior excess at 0.80-0.90 of the radius is the band's composition and the slab
  geometry, to be tested on the simulated slices against these profiles (item 4 of M5b), not a one-line rule.
- **Found: the frontal bone was labelled skin under displaced fat** (prepare_ixi `--save-intensities`, IXI012's r1 / r2
  on the MNI grid, viewed): in IXI's T2 the scalp's fat is displaced 1.63 mm inward at the forehead and vertex (the
  water-fat shift along the front-back readout), so a 3-voxel band inside the frontal bone reads T1-dark (r1
  0.06-0.21: bone) and T2-bright (r2 1.4-1.8: the fat's ghost). The rule "bone = dark in both" called it skin, which
  the simulator renders at 0.71 x WM: the 7 mm of "skin" on the anterior rays, the 2.4 mm of frontal bone, and the
  anterior band's 1.08-1.14 x real. Fat is T1-bright, so a T1-dark, T2-bright voxel next to bone (within 3 mm; not
  in the eyes) is bone under displaced fat: the T2 factor is waived there. IXI012 (from the saved maps): frontal
  bone 2.4 -> 3.6 mm, occipital 4.5 -> 5.2 mm along the rays; 96 mL of such voxels in the head outside the brain, 58
  of them below MNI z 0 (face, sinuses). The 64-head v16 preparation was stopped (7 heads in) and restarts with the
  rule; the orbit's vitreous is found before the rule so it stays an eye.
- **Placement fixed (q35)**: with the field of view 4.8 mm anterior of the head's box (`--centre-offset-mm`, the preset's
  default now) the v15 heads' centroid row is 60.1 +- 1.2 against the real 60.4 +- 2.6 (that feature's AUC 0.83 -> 0.48);
  CNN 0.872 [0.70, 0.99], W1 0.0245 (q29 0.809 / 0.0236: within the quick CNN's noise). Still there, now the largest
  single features: posterior / anterior scalp 0.59 / 0.32 against 0.47 / 0.45 (AUC 0.76 / 0.20), the quartiles 10-15 %
  high (0.215 / 0.275 / 0.420 vs 0.197 / 0.239 / 0.365) with the scalp as the p99 scale.
- **v16** (the 64 selected subjects, the bone rule, prepared on the mini): IXI012's anterior rays now 3.4 mm of bone (2.2
  before) and 5.4 mm of soft tissue (6.2), posterior 4.8 (4.2); 64 mL of bone under displaced fat with the 3 mm guard.
- **Levels over the white-matter mode** (gap.py features, new: free of the p99 normalisation): real q50 / q75 / CSF
  (interior p99.5) / scalp (outer 6 px p90) = 1.24 / 1.89 / 5.27 / 4.40. With the ring (q35): 1.32 / 2.03 / 4.93 /
  4.07 (the brain +7 %, CSF -6 %, scalp -8 %); with the flat receive (q33): 1.25 / 1.92 / 4.78 / 3.49 (the brain
  right, CSF -9 %, scalp -21 %). So the real images are flat-receive images and the 14 cm ring has been buying the
  scalp's brightness with a brighter periphery of the brain. What is left under a flat receive: the scalp 21 % and
  CSF 9 % dim against white matter. A stronger incidental MT (ours 31 %; 37-40 % would put CSF / WM right and the
  scalp at -13 %) and the fat's own brightness are the candidates; the test that separates a residual peripheral
  receive gain from fat is the non-fat scalp tissue and the bone minimum at the scalp's radius (the real skin edge
  0.82-1.09 x WM at 0.57 mm). Next quick runs: flat receive with the bound-pool fraction scaled 1.3 (`--override`),
  and with fat's T2 at 170 ms, each against these four levels.
- **The real set grows**: the fastMRI link was not expired (a HEAD request is refused, a ranged GET works; LESSONS).
  `fetch_fastmri.py` streams the validation archive again into `~/data/fastmri_more` on the mini (symlinks to the 71
  files already kept, the AXT2 cap raised 15 -> 27 GB: ~40-60 more 3 T AXT2 volumes, 8 GB of disk kept free). The
  queued runs (r30, q36-q39) keep the frozen 21-scan set; r31 (64 v16 heads x 7 slices) scores against the larger one
  and r30 will be re-scored against it for the comparison. score.py's noise-covariance cache now grows with the set.
- **r30: CNN AUC 0.748 [0.594, 0.883]** (24 v15 heads x slices 1-7 at quick quality, 8 spins per voxel, the FOV offset;
  168 simulated against the 147 real slices; r27: 0.875 on 16 v12 heads at 64 spins), histogram 0.549 [chance], W1
  0.0181; controls real-vs-real 0.440, sim-vs-sim 0.527; by slice 0.72-0.77. gap.py on it: the placement (row AUC
  0.54) and every level over the white-matter mode are within noise of real (q50 1.24 vs 1.25, q75 1.93 vs 1.92, CSF
  5.24 vs 5.29, scalp 4.21 vs 4.37; the ring is still in, so the brain's +7 % periphery stays); the head area 0.94 x
  real (0.40); posterior / anterior scalp 0.54 / 0.38 vs 0.47 / 0.45 (0.67 / 0.34: the frontal bone rule is in v16,
  not here); structure by band 1.03 / 1.05 / 0.99 / 0.91 / 0.84 / 0.86 x real: the coarse excess is gone (refine_csf
  and the subjects), the fine bands 15 % short (the reader's "smoother" cue; q36 tests the residual's smoothing, and
  the 8-spin quality adds grain, not smoothness). 105 of the 168 simulated images still score > 0.8 with no feature
  here separating them from the 61 that pass: the cue left is finer than these measures.
- **q36, the residual smoothed 0.75 mm instead of 1.5** (the same 8 heads as q35, re-prepared as v15s): structure by
  band 1.02 / 1.10 / 1.10 / 0.99 / 0.89 / 0.92 x real against q35's 1.02 / 1.08 / 1.05 / 0.92 / 0.82 / 0.86: the fine
  bands up 8 %, the coarse unchanged; CNN 0.792 [0.59, 0.95] (q35 0.872; noise), W1 0.0253 (0.0245). Half the fine
  deficit; a 0.5 mm variant (q40) follows, with the v10 lesson in mind (the unsmoothed residual carried IXI's noise at
  1.4-2.4 x real fine texture). The 64-head v16 preparation keeps 1.5 mm; the default moves with the next preparation.
- **Flat receive on the v15 heads, three ways** (q39 alone, q37 with the bound-pool fractions x1.3, q38 with fat T2
  170 ms; levels over the white-matter mode, real q50 / q75 / CSF / scalp = 1.24 / 1.89 / 5.27 / 4.40): q39 1.30 / 1.98
  / 4.78 / 3.40; q37 1.32 / 2.09 / 5.26 / 3.78; q38 1.30 / 1.98 / 4.77 / 3.60. The stronger MT puts CSF / WM exactly on
  real but lifts q75 / WM 10 % (grey matter loses less to MT than white); fat's T2 moves the scalp 6 %; neither
  reaches it: the scalp stays 14-23 % dim against white matter under a flat receive, and the posterior / anterior
  ratio 1.44 (these heads lack the frontal bone rule). CNN 0.78 / 0.90 / 0.74 (noise), W1 0.040 / 0.029 / 0.036 (the ring's
  q35: 0.025; a flat receive without the scalp fixed is worse on W1). Against the real 0.57 mm layers (skin 0.8-1.1,
  the soft tissue under the fat 1.9, fat 5.0 x WM) our classes give skin 0.71, muscle 0.16, fat 3.2: the scalp's soft
  tissue (galea, loose connective tissue) is 2-3 x too dark and the fat 1.5 x; both are flagged proxies (muscle's
  relaxation for the soft tissue; CPMG fat T2). Next quick runs: skin's T2 90 ms with the MT x1.3 (q41), plus fat 170 ms
  (q42), against the four levels, the edge profile and W1; if they land, the values enter the table as calibrations
  on fastMRI's 0.57 mm profiles, flagged as such.
- **q40, the residual smoothed 0.5 mm**: bands 1.04 / 1.11 / 1.13 / 1.03 / 0.94 / 0.97 x real (q36 at 0.75 mm: ... 0.89 /
  0.92; q35 at 1.5: ... 0.82 / 0.86): the fine structure is within 3-6 % of real, the mid band 13 % over; W1 0.0276
  (q35 0.0245), CNN 0.876 (noise). 0.5 mm becomes prepare_ixi's default for the next preparation (v16, in progress,
  keeps 1.5; r31 runs on it).
- **q41 / q42, the scalp under a flat receive with MT x1.3** (skin T2 90 ms; + fat T2 170 ms): levels over WM
  q50 / q75 / CSF / scalp 1.34 / 2.10 / 5.27 / 3.98 and 1.33 / 2.10 / 5.26 / 4.24 (real 1.24 / 1.89 / 5.27 / 4.40):
  CSF exact, the scalp within 4 % with both changes, but grey matter 8-11 % high over white (5 % at MT x1 in q39, the
  rest from the stronger MT, which darkens WM more than GM); W1 0.038 / 0.033 against the ring's 0.025, the posterior
  scalp 0.58-0.59 vs 0.47 (anatomy; the v15 heads). So the flat receive needs the whole scalp calibration at once and a
  milder MT (x1.15 would put GM / WM within ~1 % and CSF / WM within ~4 %); decided after r31 (64 v16 heads, the ring)
  gives the scaled baseline, as r32 with the flat receive, MT x1.15, skin 90 ms, fat 170 ms, each value flagged as a
  calibration on fastMRI's levels.
- **The scalp at 0.57 mm, simulated against real** (IXI012 and IXI255 of v16 at `--n 384` under a flat receive, slices
  2-4; both sides smoothed 1.1 mm against the simulation's grain of ~2 spins per pixel, ray means from the edge in
  units of the white-matter mode; scratch calv_sim.py): posterior peak 2.05 at 4.6 mm against the real 4.47 at 5.7,
  valley 0.52 at 9.2 against 1.05 at 11.5, the brain beyond 1.7-1.8 against 2.1-2.3; sides 1.81 / 1.00 against 2.98
  / 0.93 (the valley 3 mm deeper in real); anterior 1.02 / 0.64 against 2.78 / 1.18. Everything in the outer two
  centimetres is 1.3-2.7 x brighter in the real scans while the interior's distribution matches a flat receive
  (q39); a 2 x on fat and the soft tissue is beyond their physics (fat's T2 would have to be ~650 ms), so this is
  the receive after all, in its normalised form: the array's sensitivity rises steeply in the last 1-2 cm and the
  scanner's normalisation divides by a smooth estimate that cannot follow it. The 14 cm ring (q35) has the rise but
  spread over the whole brain (+7 % at the periphery of the brain). Model it as the ring's sensitivities divided by
  their own spatially smoothed RSS (`--receive-normalise-mm`, next), and read the hi-res profiles again. The
  calibration rows for skin and fat are not written. Also seen: the real scalp is ~2 mm thicker (patients) and the
  real valley never as dark as ours (0.93-1.18 vs 0.52-1.00).
- **Normalised receive, first geometry (q43: ring 0.12 m, 15 mm behind the occiput, RSS smoothed over 30 mm)**: levels
  q50 / q75 / CSF / scalp over WM 1.29 / 1.97 / 4.77 / 3.77 (flat 1.30 / 1.98 / 4.78 / 3.40; the ring 1.32 / 2.03 /
  4.93 / 4.07; real 1.24 / 1.89 / 5.27 / 4.40): the interior stays flat and the scalp gains 11 %. At 384 px the
  posterior peak goes 2.05 -> 2.72 (real 4.47), the sides 1.81 -> 1.90 (2.98), the front 1.02 -> 1.37 (2.78): the
  right kind of change, a third of the size. A loop's field falls as (a^2 + d^2)^-3/2 (a its radius, d the distance),
  so the real excess of ~2 x over the outer 2 cm needs elements 1-2 cm from the scalp all round, which a circle
  around a 18 x 15 cm head cannot do: an elliptical ring next (q45 / q46), then the geometry fitted on the three
  0.57 mm profiles. The sides also show our temporalis region too dark (0.62-0.76 vs 0.93-1.0 x WM at 8-12 mm:
  muscle at T2 30 ms is near black at TE 113; the real temporalis reads ~1 x WM, partial volume with fat and bone).
- **Elliptical normalised receive (q45: semi-axes 0.11 m AP x 0.095 LR, 10 mm behind the occiput, RSS smoothed 30 mm;
  q46 the same smoothed 20 mm)**: q45 levels 1.31 / 1.99 / 4.84 / 4.10 over WM (real 1.24 / 1.89 / 5.27 / 4.40), W1
  0.0240 (the best so far; the ring 0.0245, flat 0.040), histogram AUC 0.545 (the lowest so far), CNN 0.793; q46
  3.77 at the scalp (a shorter smoothing follows the rise and removes it: the excess kept grows with the smoothing
  length). At 384 px with q45's receive: posterior peak 3.24 (flat 2.05, real 4.47), valley 0.69 (1.05), the brain
  beyond 1.95-2.45 (2.1-2.3); sides 2.17 (2.98); front 2.01 (2.78): the outer 2 cm uniformly 0.72 x real now, the
  interior right. q47 (0.105 x 0.09 m, 8 mm, smoothed 45 mm) next; the seven missing v16 heads are prepared (IXI598
  skipped: a non-finite map), r31 runs on the 57.
- **q47 (0.105 x 0.09 m, 8 mm, smoothed 45 mm)**: scalp 4.58 over WM (real 4.40), W1 0.0161 (the best), histogram
  0.712 (worse than q45's 0.545): the front elements now too close, the anterior interior over-boosted at 384 px
  (3.7-4.5 vs the real 2.9-3.1) while the posterior peak barely moves (3.27 vs q45's 3.24, real 4.47): with this
  model's 1 / d falloff a tighter ring brightens the front interior before it lifts the posterior scalp. A loop's
  own near field, a^2 / (a^2 + d^2)^(3/2), is steeper within a few radii of the element (`element_radius`, next,
  q48); r32 (the 64 v16 heads) is chained behind r31 with q45's receive.
- **q49 / q48 on the v16 heads** (q45's elliptical normalised receive; q48 the same with 4 cm loop elements instead
  of 1 / d): q49 levels 1.26 / 1.98 / 4.78 / 4.52 over WM (real 1.24 / 1.89 / 5.27 / 4.40), histogram AUC 0.432 (the
  first at chance with the scalp right), W1 0.031, CNN 0.847; with the bone rule the anterior scalp is now 0.44
  against the real 0.45 (AUC 0.44; it was 0.32-0.36), the posterior 0.56 vs 0.47 stays. q48: scalp 5.72 (+30 %),
  histogram 0.862; at 384 px its posterior peak lands (4.30 vs 4.47) but the brain beyond it reads 2.3-3.1 (real
  2.0-2.3) and the anterior interior 3.0-4.4 (2.9-3.1): the loop near field boosts the outer brain with the scalp,
  the 1 / d ring under-boosts the scalp alone. The anterior scalp peak stays short in both (2.0 vs 2.78): the
  frontal scalp's composition, not the coil. r32 runs with q45's receive (1 / d ellipse); the element model needs
  the real array's geometry (element size and spacing) rather than another guess.
- **The real set grown to 34 scans** (13 new 3 T AXT2 volumes from the stream; `~/data/fastmri_more`): r30 re-scored
  against it reads CNN 0.904 (0.748 against the 21), controls 0.54 / 0.55, W1 0.0138. The new scans' periphery is
  dimmer: real scalp over WM 4.06 against the old set's 4.37, posterior scalp 0.44 vs 0.47, while r30's ring gives
  4.21 — brighter than this real set. The real scans are not one receive: sites and coils differ, and some may not
  be intensity-normalised at all. The receive calibration must be judged per scanner model / coil count, not on the
  pooled set (scratch realscans.py lists them).
- **The 34 scans one by one** (scratch realscans.py on score_r30m): Biograph mMR 14 coils (5), Prisma 16 / 20 (7),
  Skyra 12 / 16 / 20 (20), TrioTim 4 coils at 120 deg (2); the scalp's bright decile over WM runs 2.1-6.0 (mMR
  3.2-4.6, Prisma 2.7-4.9, Skyra 3.1-6.0, TrioTim 2.1-2.8), posterior / anterior 0.66-2.7, CSF / WM 3.5-6.2; our
  runs sit at one scalp level (4.5 +- 0.4 in q49) and one ratio (1.45). The receive is not one thing across scans
  (coil, positioning, the site's normalisation), so the model draws it per head (`--receive-draw`: semi-axes x
  U(1, 1.25), gap U(5, 30) mm, normalisation length U(20, 60) mm), to be checked on the spread of the scalp level
  (q50) and then as r33 on the 64 heads behind r31 / r32.
- **q50, the drawn receive on 8 v16 heads**: scalp over WM 4.50 +- 0.79 against the real 4.40 +- 0.82 (q49 with one
  receive: 4.52 +- 0.34), posterior / anterior 0.78-2.16 per head (real 0.66-2.7), q50 / q75 over WM 1.24 / 1.94
  (1.24 / 1.89), histogram AUC 0.417, W1 0.036, CNN 0.907 (8 heads; noise, and two of these heads look wrong: ixi175
  with 9033 px of area and a CSF share of 0.012, ixi239 with 4844 px: the proxy-based selection admitted extremes
  that need a screen on the prepared heads (area, CSF share, lncc) before r33's score is read). r33 (64 heads, drawn
  receive) is chained behind r32 and r31b.
- **v16 screened** (the preparation metadata of the 63 heads): lncc 0.195 +- 0.014, CSF 179 +- 54 mL, GM 713 +- 130,
  WM 492 +- 66, skull 305 +- 58, fat 318 +- 101 mL. IXI175 is a failed preparation (CSF 26 mL, refine ratio 0.06,
  GM 1064, fat 880: the one q50 flagged) and IXI586 registered badly (lncc 0.137 < IXI_MIN_LNCC 0.15); both moved to
  prepared_v16_rejected before r31b and r33 list their heads (r32 had listed 64 and skips 175 with an error).
  IXI097 (lncc 0.154) and IXI048 (skull 473 mL) stay. The proxy selection needs these two checks built in
  (select_ixi after preparation: lncc and the CSF refine ratio), noted for M5b item 2.
