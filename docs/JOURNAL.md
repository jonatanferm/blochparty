# Project journal

A chronological record of what we did, why, and how it went — decisions, results with
numbers, dead ends and corrections — kept so the project can be written up later. Newest
entries at the bottom. Hard-won technical lessons are distilled separately in
[LESSONS.md](LESSONS.md); the plan lives in [../ROADMAP.md](../ROADMAP.md) and
[REALISM.md](REALISM.md).

---

## 2026-10-05 — Day 1: from idea to a fast, differentiable, verified simulator

### Scoping
- **Idea**: an MRI simulator — phantom + coils + sequence → k-space. Goals, in the owner's
  words: a fun project, simulated training data, and (agentic) sequence development;
  differentiability "speaks to" the owner.
- **Landscape check**: KomaMRI (Julia, GPU, Pulseq) is closest in spirit; MRzero (PyTorch) is
  the differentiable one; JEMRIS the hardware-realism veteran. Room to add something: a
  Python-native, differentiable simulator with coils/noise as first-class citizens and an
  agent-friendly API.
- **Language**: first recommended JAX (function transforms, fusion, forward mode). The owner
  asked about PyTorch's user base → switched to **PyTorch**: MRI deep-learning ecosystem
  (fastMRI, MONAI, MRzero, torchkbnufft), better Apple-GPU story, and agents write it best.
- **Verification as a principle** from the start: compare against analytic results, an ODE
  solver and an independent signal-equation model; later other simulators and real data.

### MVP (Phase 0)
- Core: isochromat Bloch with exact free precession/relaxation, Rodrigues RF rotations,
  Strang-split relaxation inside RF; sequences as flat piecewise-constant segments
  (continuous parts differentiable, structure plain Python); Shepp-Logan with 3 T tissues;
  analytic birdcage-like coils; 2D spoiled GRE; Cartesian recon.
- **21 tests**: SciPy ODE (rotations exact to 1e-9; measured 2nd-order splitting
  convergence), FID / spin echo / inversion recovery / Ernst closed forms, signal equation
  (1e-13), gradcheck, and gradient descent recovering the Ernst angle and the analytic
  WM/GM-contrast-optimal flip angle (16.109°, to 0.001°).
- Most first test failures were the *tests'* idealisations (T2 blur during readout,
  off-resonant excitation, finite pulse duration, float32 constants) — investigated, not
  loosened.
- Named **Bloch Party** (`blochparty`); moved to `~/projects/blochparty`; private GitHub repo.

### Performance and 3D (Phase 1, moved up front)
- Considered a C++/LibTorch rewrite (owner: performance, shipping without Python, "feels more
  serious"). Measured first: the PyTorch core was memory-bandwidth bound, so LibTorch would
  not help; the win is a fused kernel, which needs no LibTorch. The owner chose to stay in
  Python, with compiled kernels as extensions.
- 3D phantom/builder/recon + benchmark harness (`a430b05`). Baseline on the Apple M4: 3D 64³
  in 233 s; 128³ projected at ~4 h.
- **Closed-form free runs** (all RF-free segments of a TR in one step, any gradient
  waveform): 3.5–9.5× (`e899957`); 64³ → 60–68 s.
- MPS (Apple GPU) ran unchanged but no faster; `torch.compile` only 1.4×/2×.

### First parallel agents
- The owner offered $100 of cloud credits. Three workers were launched; the "remote" option
  silently fell back to **local worktrees** (discovered from paths; noted for the future).
- **EPG reference model** — exact against isochromats once spins outnumber dephasing orders.
- **Memory-bounded backward** (`simulate_chunked`) — 16× less saved memory at N/16.
- **Fused C++ kernel** (forward) — 11–16× over the PyTorch core; 3D 64³ in 5.4 s
  (`26a64b2`). A full **3D 128³** (565 k spins, 2.2 M segments, 8 coils) ran in **6.3 min**
  (`4ff288f`), against the ~4 h projection that morning.

### Cloud workers (owner-launched, claude.ai/code)
- **Pulseq import + MRzero cross-validation** (PR #1): GRE / SE / bSSFP / EPI agree to
  1e-6–4e-5; an exact solver as tie-breaker puts the residual on MRzero's float32 numerics.
- **Hand-written adjoint for the fused kernel** (PR #2, `9484880`): gradients match autograd to
  ~1e-14; ~8× faster than the PyTorch core with gradients. Fixed an Apple-clang `-Werror`
  build issue at merge. Backward costs ~7× forward on the M4 (3.8× on a Xeon).
- **Three parallel Phase 2/4 workers** (PRs #3–#5, integrated `c48a98b`): realistic sequences
  (ramps, slice selection, real spoiling, SE/TSE/bSSFP/EPI), realistic phantoms (sub-voxel
  spins, T2′, B0/B1⁺ maps, fat), and a design toolkit (DESPOT1 reproduces the 1/√2-of-Ernst
  optimum). Integration caught what no single worker could: two different γ constants and a
  150 vs 170 T/m/s slew-limit mismatch.

---

## 2026-10-06 — Day 2: realism, measured against real scanners

### Strategy
- The owner: the end goal is to be the best and "step out of the shadow" of other
  simulators. Researched what others model (JEMRIS, KomaMRI and MRzero from their source,
  MRiLab, POSSUM, CMRsim, MCMR, Pulseq-CEST, EM-coupled and vector-field simulators) →
  **inventory of ~40 effects** in five physical layers (`docs/REALISM.md`, `974f228`).
- Finding: every simulator owns one layer; none combines them fast and differentiably; **none
  is calibrated against real scanners**. Strategy: a public real-data realism benchmark and a
  differentiably calibrated digital twin.
- Cloud credits down to ~$38: one cloud brief at a time, local work in parallel.

### Field model (cloud, PR #6, `d025dc1`)
- Concomitant fields (leading order + exact |B|), gradient nonlinearity (gradunwarp/Siemens
  coefficients), GIRF/eddy currents, drift and respiration — all opt-in, both kernels,
  differentiable in hardware parameters. Verified on merge; on the real 3 T IR-SE sequence
  concomitant fields change sphere signals by 0.37 % at 3 T and 2.0 % at 0.55 T (the
  expected 1/B0 scaling).

### Benchmark 1: real 3 T NIST phantom, inversion-recovery spin echo (`7c99bd8`)
- Data: Tong et al. 2022 (Pulseq `.seq` + raw 20-coil data, ISMRM/NIST phantom, Prisma Fit).
- Method: simulate the scanner's own `.seq`; phantom T1 from CaliberMRI/NIST references
  (versions 1 and 2, serial unknown); **nothing fitted**; metric = per-sphere
  inversion-curve shape.
- Dead ends on the way: phantom at x = 0 while the slice is at **+57 mm** (RF frequency
  offset) → nonsense; too few spins across the slice → speckle and shallow nulls (14 %).
- **Result: 1.49 %** (v2 references; v1 2.76 %) vs **25.9 %** for the analytic IR formula the
  authors fitted (they needed a free offset). Converged in spin density.
- **Correction** (`0ae53a4`): the T2 first used for the NiCl2 spheres belonged to the MnCl2
  spheres (a misread table). No table gives these cross terms; derived them from 3 T
  relaxivities (Martin et al. 2023), validated by the derived MnCl2 T1s landing in the
  published range. Result moved 1.44 % → 1.49 %; recorded openly in RESULTS.md.

### Lessons and process (`3af579f`)
- Started `docs/LESSONS.md` (owner's request), `tools/snapshot.py` (bit-identity guard) and
  `CLAUDE.md` (working rules). Started this journal.

### Benchmark 2: real 3 T NIST phantom, 23-echo multi-echo spin echo
- Signature to reproduce: echo 2 ~20 % brighter than echo 1 in long-T2 spheres (stimulated
  echoes from imperfect refocusing) — the reason the authors dropped echoes 1–3.
- Sphere identity from geometry (the T2 plane mirrors the T1 plane), confirmed by decay rates
  — the authors' T2 map was unusable for this.
- **Dead end, instructive**: the simulation plateaued at ~61 % error with *growing* echoes at
  every spin density. Ruled out an RF-phase (CPMG) hypothesis, corrected a flawed k-space
  diagnostic of our own, then found the cause: the file images **two slices** (+12 / −12 mm);
  "discard the first 128 excitations" meant "use the second slice", and the phantom sat on the
  first.
- **Result** (converged, nothing fitted): **2.35 %** shape error over 23 echoes vs **10.47 %**
  for mono-exponential decay with the reference T2; sphere 2 echo2/echo1 1.173 vs 1.213 on
  the scanner.
- **First digital-twin step: B1⁺** (`nist_tse_b1.py`). The shape error is *highest* at nominal
  B1⁺ and falls both ways: echo shapes constrain only |B1⁺ − 1|. A global scale fitted on
  even spheres (1.144) does not improve the held-out odd spheres at converged density (2.09 %
  either way); smooth maps overfit; even a free B1⁺ per sphere only reaches 1.6 %. An honest
  negative result: B1⁺ is not the main remaining error.
- **Diagnosis of what is**: short-T2 spheres plateau at late echoes at ~15–30× the thermal noise
  floor — blurring from the unmodelled **fill fluid** (water, T2 ≈ 2 s) into ROIs that sit
  ~0.7 px inside the spheres. Second suspect: T2 reference uncertainty (sphere 5's T2 is one the
  manual withholds; MnCl2 T2 is temperature-sensitive).
- Next: model the fill fluid and sphere walls (both benchmarks).
