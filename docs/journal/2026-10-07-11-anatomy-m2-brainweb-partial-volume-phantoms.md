# 2026-10-07 — Anatomy M2: BrainWeb partial-volume phantoms

- Continued M1 `d513a43` on `codex/anatomy-susceptibility`; unrelated benchmark commit
  `a3a8a00` arrived during the work and is preserved. Added `Anatomy`, conservative
  overlap-integral resampling/crop/slab, pinned raw-short BrainWeb loading, tissue
  distributions with citations/assumption flags, per-tissue or categorical spin
  assembly, six fat peaks, optional MT/diffusion, and susceptibility/scanner map pairs.
  No engine, `phantom.py`, `maps.py` or real-scanner validation changes.
- Accessed the public download form without accounts or identity fields. Committed
  subject-04 and older SBD manifests only; downloaded/derived volumes stay ignored.
  Corrected the spec's geometry: fuzzy anatomy is 0.5-mm (362,434,362), while the
  newer T1 reference is a separate 1-mm grid. Initially tried raw bytes: documented
  unsigned values were actually signed/offset encoded. Switched to raw shorts, then
  found the actual byte order is little-endian, contrary to the legacy format page.
  Pin decompressed bytes; normalize only overfull membership totals, leaving deficits air.
- Separate tissue spins conserve per-tissue/per-voxel PD to <1e-12; resampling physical
  volume to <1e-6. Mixed-voxel GRE agrees with Ernst to <1e-8. Shifted continuous-box
  Fourier error 0.177089→0.00384746→0.000230445→0.0000143652 for 1/4/16/64 samples/cell.
  Sphere B0 at jittered spins: 0.464157 Hz RMS. GPU/core with B0+MT+diffusion: 3.23e-7.
  Rounded fat weights sum to 0.999: normalize in new assembly to preserve PD.
- T2* policy is explicit: auto omits uncalibrated R2′ when chi is enabled; combining
  observed table T2* and chi needs a supplied macroscopic calibration or intrinsic map.
  K=1 has only the zero Cauchy median and warns. Tissue SDs are metadata, not M4 draws.
  Ex-vivo/other-tissue priors are flagged; only WM/GM have verified 7 T T1/T2* entries.
  Missing requested MT/diffusion parameters and 7 T classes raise.
- Downloaded older T1/T2/PD comparisons: GM/WM relative differences −0.0144/−0.1253/+0.0275%;
  CSF/GM −1.549/+4.733/+2.201%. Newer subject-04 T1 differs −5.44% GM/WM, −21.31% CSF/GM;
  older priors are not established as its exact parameters. Published human 3 T relaxation
  values propagated through these sequences provide a sensitivity check, not measured
  in-vivo ratio validation. Results and limitations: `validation/anatomy/RESULTS.md`.
- Full 2-mm head B0 at 3 T: brain RMS 45.2976→19.1716 Hz after second-order shim;
  residual 1/50/99 percentiles −52.8329/0.5364/51.4486 Hz. Scalar tissue/air/bone
  assumptions, not measured-field agreement; QSM forward-field comparison stays M3.
- Full 1-mm head, categorical K=1: 5,328,321 spins; load 5.807 s, assembly 0.308 s,
  GPU RF/FID 0.686 s (7.77 M spins/s). Phantom 149.19 MB; peak process RSS 3.64 GB;
  end-of-run MPS allocation 1.08 GB (not peak, not additive to unified-memory RSS).
  Benchmark excludes susceptibility/T2′/MT/diffusion and is not full imaging throughput.
- Validation: 21 anatomy tests; full suite with host GPU access **449 passed, 4 opt-in
  crosschecks deselected** in 185.08 s. Before/after existing outputs and gradients are
  bit-identical. M3 QSM, M4 spatial variability and M5 example remain open.
