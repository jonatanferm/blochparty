# 2026-10-13 — Milestone: simulated heads that are not obviously fake

Where the realism work stands after r20-r29 (2026-10-12 / 13), against 21 fastMRI 3 T AXT2 scans (slices 1-7, the
same reconstruction, normalisation and coloured coil noise for both sides):
- **By eye: not distinguishable.** A blinded reader (the user) scored 7 / 24 on r27's quiz (chance 12; P(<= 7) =
  0.03). The one cue that had worked before, "the simulations look smoother", now points the wrong way.
- **By statistics:** full-run CNN AUC 1.0 (r20-r24) -> **0.875** [0.745, 0.976] (r27), the intensity histogram test at
  chance (0.572 [0.40, 0.73]); controls 0.44 / 0.46. Masked to the interior or to the outer 12 px the CNN reaches only
  0.77 / 0.80: no single region gives the simulations away any more.
- **What closed the gap** (each found by a targeted measure, fixed in the source, checked independently):
  the display orientation; the receive coil behind the head; IXI's water-fat shift in the segmented scalp; cubic
  resampling (half the real fine structure before); the falx; the slice stack's level; coloured coil noise; a
  residual layer fitted to each subject's own dual-echo scans (intracranial, smoothed within tissue, PD re-centred per
  tissue); the real echo train read off the k-space (21 echoes at 10.26 ms; the phase-axis halo gone); each voxel's CSF
  share from the subject's own T2 (convexity CSF had been 30-50 % too high), with the falx accounted for.
- **What is left, measured:** a posterior skull band ~0.5-1 px deeper and darker than real (minimum 0.13-0.17 vs
  0.19-0.27); the side / front scalp 12-15 % dimmer than real against white matter while the back matches (fastMRI's
  fat brighter than our CPMG fat T2 gives, IXI's dimmer: sequence-dependent fat, i.e. J coupling, ROADMAP M6.1); and
  population: fastMRI's clinical patients carry more deep CSF at slices 3-7 than a 20-82-year spread of IXI volunteers
  (median 554 vs 438 px; IXI subjects over 55: 566; age correlation 0.70), and one real scan has a large lesion.
- **Iteration**: validation/realism/quick.sh (c0fe213) answers in 10 min on the mini (8 heads x slices 2 / 4 / 6,
  8 spins per voxel, score --quick) against ~5.5 h for a full run; its CNN is noisy (+-0.15), its intensity and
  targeted measures are not. Calibration: q27 (r27's heads) 0.774, q29 (v14 heads) 0.809, q29r12 (12 cm coil ring)
  0.769, q29f200 (fat T2 200 ms, a test, not a database change) 0.851 with W1 0.0236 -> 0.0127.
- **Age-matched subjects, quick test (q30)**: the 11 prepared v14 subjects aged 50-82 instead of the 20-82 spread:
  CNN 0.894 [0.77, 0.98] (q29 0.809), W1 0.0207; quartiles 0.200 / 0.280 / 0.492 (real 0.199 / 0.280 / 0.460; the
  median now exact), but hemisphere texture back to 1.1-1.7 x real (older brains' sulcal CSF) and the posterior brain
  at 10 px (real 7, q29 8.5). The real population is not simply IXI's older half; age matching alone is not a win.
