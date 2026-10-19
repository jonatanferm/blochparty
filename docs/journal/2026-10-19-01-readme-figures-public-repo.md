# 2026-10-19 — A README with figures, and the repository goes public

- **Why**: the user asked for the GitHub repository to be made public with a README that shows what the simulator
  does. The README's reference sections (engines, phantoms, sequences, Pulseq, field model, gradients, design,
  layout, conventions) stay as they were; a showcase goes in front of them.
- **Before going public, the history was scanned** (all 456 commits, every blob): no API keys, tokens, private keys
  or signed URLs. The Runpod key is read at run time from `~/.runpod/config.toml` (`validation/design/pod/launch.sh`),
  the fastMRI `.eml` with its signed links is git-ignored and never was committed, and no raw data (fastMRI,
  BrainWeb, IXI, the scanner sets) is in the history: the largest blobs are the 700 kB sampled-heads figure and the
  tissue table. The IT'IS values in `anatomy/tissuedb` are cited per row (`itis_v4_2`, `itis_water_v5_0`).
  The real fastMRI images in `outputs/scans_real_vs_sim.png` were left out of the README (its data agreement).
- **`tools/readme_figures.py`** makes every new figure in `docs/images/` with blochparty itself:
  - `hero.png` / `pipeline.png`: `sample_head(3)` (BrainWeb subject 4), 9 664 704 spins (2×2×2 per voxel, ±4 mm
    around the 5 mm slice), 160², the three scanner-playable protocols of `examples/sampled_heads.py`, 8 birdcage
    coils, noise at image SNR 60; Metal on the laptop, ~6 min for the three. The simulation is cached in
    `outputs/readme_head.npz` and the tissue maps in `outputs/readme_head_maps.npz`, so restyling only redraws.
  - `spin_echo.gif`: 36 isochromats over ±30 Hz, 90°ₓ / 180°ᵧ hard pulses (50 µs), TE 60 ms, T2 120 ms. The arrows
    are `SimResult.m` at each of 75 frames; the trace is the ADC against the closed form
    e^{−t/T2}·|mean e^{−i2πfτ}|: max difference 2.4e-3, the precession during the 50 µs pulses.
  - `optimize_flip[_dark].png`: `examples/optimize_contrast.py`'s problem (L-BFGS on the WM−GM contrast of a 32²
    spoiled GRE): 16.109° in 18 simulations against the analytic 16.093°. Light and dark versions in a `<picture>`.
  - Reused: `validation/real/nist_irse_curves.png` (1.29 % against the analytic formula's 19.88 %),
    `validation/anatomy/sampled_heads.png`, and `outputs/em_brainweb.png` (from `validation/em/brainweb_demo.py`),
    copied to `docs/images/`.
- **Two dead ends on the tissue maps**: averaging the spins in each pixel let one CSF spin (T2 ≈ 2 s) make a brain
  pixel's "T2" several hundred ms, and fat spins carry their −434 Hz chemical shift in `db0`, so the B0 map showed fat.
  A plain per-pixel median was worse (speckle everywhere): `to_phantom` gives every tissue in a voxel its own spins
  with PD = fraction·PD/K, so most spins belong to minor tissues. PD is the pixel's sum; T1, T2 and B0 are
  PD-weighted medians. The first crop (8 % on every side) cut off the occipital lobe; heads
  are taller than wide, so only the sides are trimmed now.
- The user asked whether matching the closed form so closely means the simulator is doing work for nothing. It
  doesn't: those figures are checks, picked because they have a known answer; the README now says so, and points to
  the NIST benchmark, where the scanner's own sequence has no closed form and the textbook formula is ~20 % off.
- Commits: 8eac716 (README, `tools/readme_figures.py`, `docs/images/`), this entry with the lesson.
