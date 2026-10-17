# 2026-10-09 — M3: the digital-twin loop on NIST benchmarks 1 and 2

- **Reduced model for fitting** (`validation/real/nist_twin.py`, abd5675): one column of spins per
  sphere at its ROI centre (slab x one pixel along the readout), only the TR that acquires the
  k-space centre after its warm-up TRs, the two ADC samples either side of k = 0 per echo. 25k
  spins and 1.7e5-2.7e5 segments instead of 85k-174k spins and 4.3e6 segments per series: a
  gradient through the Metal adjoint costs seconds. Against the full simulation: 0.5-1.0 % (shape)
  on benchmark 2's spheres 1-10, <= 1.4 % on benchmark 1's long-T1 spheres 1-4. A 3 x 3 grid of
  columns at half the slab resolution was worse (resolution, not coverage).
- **Which inputs.** Scanned before fitting: per-axis gradient delays of 10 us move nothing on y and
  z and < 0.1 % on x; the roadmap's suspect for benchmark 2, a longer first refocusing interval,
  does nothing without off-resonance and < 0.1 % with 50 Hz (and the files have no gradient-free
  gap to stretch). So the twin's input is B1+ alone: five parameters, axially symmetric about the
  bore so one sagittal plane fixes them (test). Receive maps from the data (the roadmap's coil
  image / RSS) have unit norm and add nothing to an RSS image, so the absolute metric is reported
  with uniform receive and with a smooth receive profile from the water fill (which carries the
  water's transmit shading too).
- **Fits** (reduced model, L-BFGS through the adjoint, gradients checked against differences to
  1e-4..3e-3): on benchmark 1, 1.05 -> 0.50 % in either branch of |B1+ - 1| (field 0.93-1.09 on
  its spheres, 1.21 at isocentre falling off radially: centre brightening); on benchmark 2, 2.26
  -> 1.57-1.66 %, but its field is 0.70-0.91 or 0.99-1.26 on the T1 plane.
- **Full-model predictions** (Monte Carlo, 256 Sobol spins per pixel, 4 seeds, jackknife SEs, on the
  Mac mini). Benchmark 1 from the benchmark-2 fit: 13.57 ± 0.31 % against 1.77 ± 0.31 % nothing
  fitted (the TSE's B1+ is wrong on the other plane); paired ablations around it: water fill
  +0.16 ± 0.08, ideal slab +11.67 ± 0.32, concomitant fields -0.01 ± 0.00, the Prisma GIRF of
  benchmark 7 +0.10 ± 0.07. A refocusing-only amplitude factor on top of the benchmark-1 field
  converges to 1.003 on benchmark 2: the TSE fit's gain is not refocusing calibration but five
  smooth parameters absorbing sphere-specific residuals.
- **Temperature** as an input (Stupic et al., MRM 2021: at 3 T the MnCl2 spheres' T1 +2.7 %/C and
  T2 +1.6 %/C over 16-26 C; NiCl2 insensitive, <= 4 % non-monotonic, so benchmark 1 cannot see it):
  fitted on benchmark 2's reduced model, +2.6 C with B1+ = 1, spheres 1-10 2.26 -> 2.19 %; with
  benchmark 1's B1+ field 20 C stays best (2.15 %). Over the plausible range it moves spheres 1-10
  by < 0.1 point and 11-14 by ~1. What is left on that plane sits in spheres 5 (reference T2
  withheld by the manufacturer, Jiang 2016 instead) and 9-10 (short T2, reached by the water fill).
- **Found on the way.** `write_pulseq` labelled off-centre refocusing pulses as excitations (a phase
  ramp cancels |sum s dt|; fixed by `pulseq.flip_angle`, ede53ab). The Prisma GIRF needed 2.8 GB
  per input/output pair on a NIST sequence (fixed by blocks, 59fca2a).
- **Held-out prediction** (benchmark 2 from benchmark 1's B1+, full model, Mac mini, 90 min):
  1.83 ± 0.03 % against 2.20 ± 0.02 % nothing fitted (paired -0.38 ± 0.04), better on every sphere
  but 1 and 5, sphere 11 7.8 -> 3.8 %. In sample on benchmark 1: 1.77 ± 0.31 -> 1.13 ± 0.25 %
  (paired -0.64 ± 0.16). Ablations around the twin: water fill +0.25 ± 0.07 (benchmark 2) / +0.01
  (benchmark 1), ideal slab +6.8 / +24.5, concomitant fields -0.01 / 0.00, the Prisma GIRF of
  benchmark 7 -0.02 ± 0.03 / -0.03 ± 0.10: what is not in the model moves the prediction less than
  its error bar (condition 3). Spheres 12-14 stay at 13-17 %, sphere 5 at 3.6 %.
- **Absolute metric**: 7-8 % with one global scale and uniform receive, ±1 point with the twin;
  a water-fill receive profile helps where the transmit shading does not double up. Receive
  shading is the input these data cannot separate (needs a B1+ map or PD scan).
- Runs moved to the Mac mini midway (same seeds, nothing-fitted row identical to the laptop's:
  2.20 ± 0.02 %); one prediction ~40-90 min there against ~2.5 h on the laptop.
- Commits: abd5675 (reduced model, fits), 59fca2a (GIRF blocks), ede53ab (flip angle of
  phase-ramped pulses), then the write-up.
