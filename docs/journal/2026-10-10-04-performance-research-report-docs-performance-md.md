# 2026-10-10 — Performance research report (`docs/PERFORMANCE.md`)

- **What**: a read of the engines, the benchmark history and the realism workload, with a cost
  model and estimates for the ways the mathematics can remove work; nothing implemented.
- **Measured on this laptop's 8-core M4 GPU** (realism TSE 128 at the 100 µs RF raster on a
  130k-spin sub-voxel Shepp-Logan, 14 coils): full 1.82 s, ADC flags off 0.97 s, raster-cut ramps
  and crushers merged exactly (36,335 → 19,726 segments, 4e-6 relative) 1.20 s, merged + ADC off
  0.46 s. GRE 3d-48, 8 coils: 1.24 s full, 0.36 s with ADC off. So the coil reduction is 45-70 %
  of the forward, the ramp pieces 25 % of the TSE, RF 10 % (dominant in the multi-slice variant).
  Per spin-segment: cached free 1e-10 s, cache-miss free 2e-10 s, RF 4.4e-10 s, ADC +4-8e-10 s.
- **Segment mix**: 3D GRE is 94-97 % ADC samples; the realism TSE is 46 % raster-cut free
  segments without ADC (154 free runs averaging 77 distinct gradient values each).
- **Ideas, ranked** (gains estimated unless marked): merge ramps exactly (1.5x measured);
  readouts as type-1 NUFFTs along the readout line with time-segmented T2 decay, spreading on the
  same simdgroup matrix products (readout cost ÷ n/(w·M) ≈ 7-14x; 3D GRE 6x); periodic sequences
  (RF-spoiled GRE has period 360/gcd(117, 360) = 40 TRs in the RF frame; affine steady state per
  spin; k-space by 3D NUFFT per residue: 3D GRE 128³ 192 s → 2-4 s); tabulated RF propagators in
  (Δf, B1) with first-order relaxation (multi-slice realism 3x); a reversible adjoint (5-7x →
  ~3x the forward); lattice quadrature instead of jittered spins along the crusher axis (4x on
  spins if the aliasing study holds); half-precision coil reduction, GIRF-friendly phase
  recurrence, interaction-picture free pools; CUDA on rented GPUs for scale.
- Combined estimate for the realism TSE slice: 10-15x (30-60x with the spin reduction); for the
  3D GRE example 50-100x. Suggested order and independent checks in the report.
