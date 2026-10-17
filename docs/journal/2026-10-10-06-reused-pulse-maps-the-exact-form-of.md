# 2026-10-10 — Reused pulse maps (the exact form of PERFORMANCE 2.3); realism suite rerun

- **Where the time went after NUFFT readouts** (one realism slice, 16 coils, laptop, fast): single
  1.50 s (RF 0.34), single + MT 2.41 s (RF 0.90), 16 slices + MT 6.3 s (RF ~4.5; the far
  bound-only pulses 0.1 s, the neighbours' exact pulses triple the stepped RF). The MT kernel is
  slower everywhere (registers). Caching the lineshape per pulse and a series exchange step: 14 %,
  not kept.
- **Dead end — pulses as blocks**: a float64 prototype of 2.3's cheap form (exact rotation of K
  samples with relaxation, exchange and mean saturation around it) is 5e-3 to 1.3e-2 off per pulse
  (WM, GM, muscle), 2e-4 to 1.6e-3 at K = 2. Tables over (Δf, B1) would need the interaction-picture
  expansion and ~60-100 floats per node with the bound pool.
- **What worked** (cedc6ab, `simulate_metal(pulses="reuse")`): a static spin's affine map of a pulse
  is the same at every occurrence; a phase-shifted copy is the map conjugated by Rz exactly. Each
  distinct pulse is stepped once per spin (5 columns) and applied at every occurrence. 1e-7 from
  stepping (TSE, TSE + MT + 16 slices, RF-spoiled GRE + MT). Realism slice: 1.65x single, 2.3x with
  16 slices + MT (full quality 12.9 → 5.6 s; 20.8 s before NUFFT).
- **Suite on the mini** (new code: bound_only, NUFFT; f2 also reuse): pytest 777 passed (783 with the reuse tests). Realism,
  20 heads x 13 slices against 21 fastMRI 3 T T2w scans (CNN AUC, histogram AUC, intensity W1;
  real-vs-real W1 0.0066):
  - f0, single slice, no MT: CNN 0.990 [0.961, 1], histogram 0.965, W1 0.050 (quartiles 0.268 /
    0.360 / 0.548 vs real 0.242 / 0.344 / 0.539).
  - f1, 16 slices + MT: CNN 0.9995 [0.998, 1], histogram 0.968, W1 0.045 (0.213 / 0.304 / 0.502):
    the other slices' pulses and MT darken the slice past the real intensities.
  - f2, 16 slices + MT, `tissuedb.tissue_table(3.0)`: CNN 0.998 [0.993, 1], histogram 0.967, W1
    0.048 (0.222 / 0.308 / 0.501). Its images have no GM/WM contrast: the n-weighted pooled WM and
    GM T2 are both 86 ms (flagged earlier, one SyMRI study dominates).
  - Seen side by side (seed 0 against its paired scan): the real scans' scalp fat is a bright ring
    (TSE "bright fat": rapid refocusing suppresses J-coupling dephasing); the simulation's fat (T2
    68 ms, no J-coupling) is as dim as WM. GM/WM contrast is weaker than real in f0/f1.
- Per head on the mini (13 slices, full quality, ~25 s of it loading and deforming the head): f0 48 s
  (single slice), f1 94 s (NUFFT, stepped pulses), f2 77 s (NUFFT + reused pulses, partly sharing the
  GPU with scoring): ~4 s per multi-slice MT slice, 6-9 s before reuse.
- Commits: ebabdaa (bound_only), 6b8814c (NUFFT), be204b6, b63fb83, 1bd87ee (docs), cedc6ab (reuse),
  6516673 (docs).
