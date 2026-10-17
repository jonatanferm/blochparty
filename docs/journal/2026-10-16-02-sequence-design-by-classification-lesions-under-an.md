# 2026-10-16 — Sequence design by classification: lesions under an acquisition budget (exploratory)

- The user's question: optimise a sequence jointly with a network that reads its raw output to classify something,
  against a standard sequence read through an image. Built as `validation/design/lesion_slabs.py` (the data) and
  `validation/design/lesion_design.py` (the arms), on the mini's 62 v17 heads.
- **Data**: slabs of 5 mm at MNI z 10 / 20 / 30 mm, 64 x 64 over 220 mm (3.4 mm pixels), each head rendered twice
  through the lesion stream, with 1-2 lesions of 6-15 mm placed within 2 mm of each imaged level (`add_lesions z_mm`)
  and without, so pairs differ only by the lesion; the tissue table at 3 T, the subject's residual maps and the bound
  pool, at 2 mm anatomy voxels (145-165 k spins a slab); 372 slabs (133 with a lesion of 2 or more pixels in the
  slab), 50 training / 12 test subjects. 2.2 GB on the mini.
- **The constraint**: every arm acquires at most a fraction f of k-space (K = f x 64 lines) through the same 8-element
  receive ring (radius 0.14 m), with complex Gaussian noise at a fixed SNR (20, in units of each slab's mean sample of a
  reference GRE) on every acquisition. A: the standard T2w TSE (ETL 16, TE 113 ms, TR 6 s) undersampled uniformly to
  K lines with 4 centre lines, reconstructed by CG-SENSE with the exact coil maps (an upper bound for any ESPIRiT-style
  calibration; adjoint checked to 1e-7, a phantom recovered to 24 % at K = 13), then a small CNN on the image. B: the
  same lines, the CNN on the raw multi-coil samples. C: a spoiled GRE of K lines whose per-line flip angles, TE, TR
  and k-space positions (continuous, not on the Cartesian grid) are learned together with the CNN on its raw
  multi-coil samples, by Adam through the Metal adjoint (gradients reach all four groups; 0.07 s forward + 0.25 s
  backward per slab on the laptop at K = 13), 300 steps of batches of 8, the flips in (2, 90) deg, TE in (2, 30) ms,
  TR in (8, 120) ms by sigmoids. Reported: test AUC and accuracy per arm, the scan time (the TSE's kept shots; the
  GRE's K + 8 dummies x TR), and C's learned sequence. The sweep f = 0.2, 0.1, 0.05 runs on the mini.
- **First sweep at chance, and why**: at f = 0.2 every arm scored AUC 0.40-0.56 at the negative-class accuracy, and
  f = 0.1 / 0.05 crashed the CNN's pooling on 6 and 3 lines. Rendering the slabs showed the task was not learnable
  as posed: 6-15 mm lesions with the table's T2 (111 ms, West 2014's mixed MS lesions scaled from 1.5 T) came out as
  one or two pixels at 3.4 mm, ~20 % above their surroundings after partial volume and the residual maps, under the
  grain of one spin per 2 mm voxel; a mask check on the mirrored, transposed and flipped grids confirmed the mask's
  frame and that the lesion simply was not bright. Changed: lesions of 10-20 mm, four in-plane spins per voxel
  (~245 k a slab), and the task's lesion given T2 200 ms / PD 0.85 (a conspicuous T2-hyperintense lesion, recorded as
  assumed in every slab's meta; the table's value is unchanged), which puts the lesion at 1.2-1.7 x the brain's
  median (peak 1.6-2.2 x) in the full TSE; the label needs 4 or more lesion pixels. The CNN pools an axis only while
  it has room. The cache is rebuilt and the sweep rerun with the full sampling first as the ceiling.
