# 2026-10-07 — Anatomy M1: susceptibility fields and shimming

- Started from anatomy spec `a31f4a9` on `codex/anatomy-susceptibility`. Added
  `susceptibility_to_b0` (SI susceptibility → Hz, arbitrary B0 direction, anisotropic
  voxels, configurable padding and explicit surrounding material) and `shim` (order
  0–2 solid harmonics, matching `b0_shim`'s names and physical units). No engine changes.
- Independent closed forms, 3 T, dchi=9.4e-6: 10-mm sphere inside/outside RMS
  0.908/0.369 → 0.804/0.190 → 0.439/0.108 Hz at 32³/48³/64³, with 2× padding;
  far-field RMS 4.287 → 0.134 → 0.0216 Hz at 1×/2×/3× padding. Partial-volume
  sampling plus the standard kernel suffices; an additional anti-alias kernel is deferred.
- Long-cylinder inside/outside RMS at 0°, magic angle, 90°: 1.604/1.598,
  1.671/0.331, 1.766/0.830 Hz, using 4× transverse padding. Interior means
  398.624/0/-199.312 Hz vs infinite-cylinder 400.228/0/-200.114 Hz; finite ends
  and periodic copies remain. A sphere-adjacent ROI shims from 94.111 to 20.615 Hz RMS;
  harmonic inputs vanish below 1e-12 Hz and the residual is orthogonal to every fitted term.
- Corrections: the naive even-grid dipole kernel failed joint object/field rotation by
  ~2 Hz because of pairs of Nyquist axes. Average their signs independently, eliminating
  the offending cross terms (< 2e-12 Hz after correction). A long cylinder initially had
  ~6.1 Hz offset at 2× padding: increased transverse padding, not relaxed tolerances.
  For large float32 shim masks, avoid N*eps rank cutoffs and use the exact mean for order 0.
- Clarified the spec: Anatomy chi uses dimensionless SI (loaders convert ppm by 1e-6);
  padded absolute maps need an explicit air background. Rotation of the head relative to
  B0 requires field recomputation; only translation can carry the susceptibility field
  unchanged, and scanner shims must remain separate. M2–M5 and the QSM cross-check are open.
- Validation: 22 focused tests pass (including linearity, rotation, gradients, dtype,
  malformed inputs and a >8-million-voxel float32 map); before/after snapshots of existing
  default outputs and gradients are bit-identical. Full suite: 390 passed, 37 GPU tests
  skipped in the sandbox, 4 opt-in crosschecks deselected (413.99 s). All 37 GPU skips
  were then covered by 51 passing tests with host GPU access. The final focused rerun
  includes one additional large-map test: 428 distinct default tests covered overall.
