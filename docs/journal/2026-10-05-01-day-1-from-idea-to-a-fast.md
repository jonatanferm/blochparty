# 2026-10-05 — Day 1: from idea to a fast, differentiable, verified simulator

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
