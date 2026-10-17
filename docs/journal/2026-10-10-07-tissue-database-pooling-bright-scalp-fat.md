# 2026-10-10 — Tissue-database pooling; bright scalp fat

- **Pooling** (cf0d85a): n-weighting let one synthetic-MRI study of 54 elderly subjects (WM T2 108 ms
  > GM 80 ms) set pooled WM and GM T2 to 86 ms each (f2 had no GM/WM contrast). Now each source
  counts once (median, robust spread: the random-effects limit), and T1/T2 come from reference
  methods first (spin-echo decay; inversion/saturation recovery), naming what is left out. 3 T:
  WM T2 74.5, GM 110, muscle 30 (was 52), WM T1 804, GM T1 1276 ms. f3 (f2 with this): CNN AUC
  0.9993, histogram AUC 0.944 (0.967), W1 0.039 (0.048), quartiles 0.230 / 0.346 / 0.552 against
  real 0.242 / 0.344 / 0.539.
- **Fat**: measured where the contrast lives first. Edge profile (mean intensity against depth from
  the head's edge over the interior median, slices 0-8): real scalp peak 2.28 at 3-4 px (1.72 mm),
  f1 a flat 1.53. Fat's T2 68 → 148 ms changed nothing there: BrainWeb labels the scalp
  "Muscle/Skin" (our `skin`, muscle values) and its fat class is 0-24 % of the outer 9 mm.
- **Physics, two parts**: (1) lipids are J-coupled; single-echo SE/STEAM read fat T2 40-75 ms at 3 T,
  a closely spaced refocusing train suppresses the modulation (Henkelman 1992: multi/single-echo
  ratio 1.5 in oil, 1.9-2.4 in fat at 20 ms spacing, ~1 beyond 100 ms) and CPMG at 8 ms reads
  140-159 ms (Keene 2020, four cohorts, EPG fits). A Bloch simulation has no J coupling but plays
  the train, so lipid T2 = the multi-echo-train value (8f6b8ba): pooled fat T2 148 ms. (2) the scalp
  is skin over a subcutaneous fat layer: `layer_scalp` moves BrainWeb's skin class 1.61-5.75 mm deep
  into fat (Li 2024, vertex ultrasound, 30 adults; 2d978be, opt-in `sample_head(scalp="layered")`).
  Laptop A/B, 3 heads: layered 1.26 with fat T2 68 ms, 1.86 with 148 ms.
- **f4** (f3 + layered scalp + fat T2 148 ms), 20 heads x 13 slices: CNN AUC 0.998 [0.992, 1],
  histogram AUC 0.911 [0.823, 0.976], W1 0.041 (quartiles 0.277 / 0.402 / 0.587). Edge profile peak
  1.84 at 3 px (real 2.28), but the band below it (calvarium) is 1.03-1.05 against real 1.56-1.68:
  adult diploe is fatty marrow, and our marrow class has vertebral/knee single-echo values (T2
  50 ms) and one T2 for its water and lipid parts. Next for fat: diploic marrow composition, lipid
  relaxation per species in mixed classes, J modulation (single-echo SE fat is now too bright).
  Also seen: the simulated heads are larger than the real ones and fold a faint scalp line in at
  the field of view's top edge.
- Tests: quick suite 760 passed (laptop). Commits: cf0d85a, 8f6b8ba, 2d978be, 89fa077.
