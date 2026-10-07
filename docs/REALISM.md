# Physical realism: what other simulators do, what nobody does, and where we go

Research inventory, October 2026. Goal: the most physically realistic MRI simulator, judged
against **real scanners**, not against other simulators. Sources at the end; claims about other
simulators come from their papers, documentation or source code (noted where it matters).

## 1. The landscape

| simulator | method | strongest at | notably missing |
|---|---|---|---|
| **JEMRIS** [1,2] | isochromats, CVODE ODE solver (adaptive), CPU/MPI; GPU port 2025 | hardware realism: concomitant fields, nonlinear gradients, eddy currents, pTx/multi-coil, chemical shift, susceptibility off-resonance, motion/flow, diffusion, multi-pool | speed; not differentiable; the GPU port drops diffusion, exchange, pTx, MT, concomitant fields, nonlinear gradients and eddy currents [2] |
| **KomaMRI** [3,4] | isochromats, GPU, Magnus integrators, Pulseq | speed; motion and flow (paths, heartbeat, flow paths), birdcage/arbitrary receive maps | *from source code*: ideal linear gradients only (`LinearXYZ`), uniform transmit only, no noise model, diffusion "UNDER CONSTRUCTION", susceptibility commented out |
| **MRzero** [5] | phase distribution graphs (EPG generalisation), PyTorch, differentiable | speed per state, differentiability, voxel-shape model, T2′, free diffusion (exp(−D k² t) on states), rigid/per-voxel motion | instantaneous RF, float32 numerics (see `validation/RESULTS.md`), no hardware imperfections, no exchange |
| **MRiLab** [6] | multi-pool Bloch–McConnell on GPU | tissue model: N free + N bound pools with exchange (MT), chemical-shift spectra; B1⁺/B1⁻ from coil configurations, pTx | dated, MATLAB GUI, no motion, no hardware chain |
| **POSSUM** (FSL) [7] | voxel-wise Bloch with motion | spin history under motion, motion × B0 interaction, fMRI time series, eddy-current DWI | non-functional in FSL 6; EPI/fMRI-centred |
| **CMRsim** [8] | numerical + analytical Bloch, TensorFlow | cardiac motion, turbulent flow (CFD particles), cardiac DTI | no hardware chain, no exchange |
| **MCMRSimulator.jl** [9] | Monte Carlo random walks in geometric substrates | microstructure: diffusion, permeability, surface relaxation, local off-resonance from myelin/vessels, MT — all at once | not an imaging simulator (no realistic encoding chain, coils, hardware) |
| **Pulseq-CEST / BMCTool** [10] | Bloch–McConnell, Pulseq | CEST/MT preparation, standardised parameter sets and in-vivo data | preparation only, no imaging chain |
| **Penn State EM-Bloch** [11] | finite-difference Bloch + full EM fields | B1⁺, B1⁻ and **E fields** per coil → realistic noise and SAR, validated SNR against experiment | not open / not maintained as a general tool |
| **Vector-field framework** (Bschorr et al., 2025) [12] | matrix-based, MATLAB | full vector fields: B0 direction changes, concomitant (perpendicular) components, low field; claims no other simulator handles the full vector field | narrow scope |
| **Virtual Scanner** [13] | digital-twin platform | end-to-end with Pulseq, hardware-aware B0/B1⁻ | simplified physics |
| **blochparty (now)** | isochromats; closed-form free runs; fused C++ kernel with adjoint | fast *and* differentiable; sub-voxel spins, T2′, B0/B1⁺ maps, fat, real spoiling, ramps, slice selection; validated vs ODE, EPG, signal equation, MRzero; field model (concomitant, nonlinearity, GIRF/eddy, drift) differentiable in the hardware parameters | everything in §3 marked ✗ |

**What nobody does.** Each simulator owns one layer (hardware → JEMRIS, microstructure → MCMR,
exchange → Pulseq-CEST, flow → CMRsim, EM noise → Penn State). Nobody combines them, nobody is
fast and differentiable while doing so, and — the main gap — **nobody calibrates against real
scanners**. Simulators are validated against analytic cases and against each other; real data,
when used at all, is a qualitative figure. Being "best" therefore means two things: (a) the
union of physical layers in one fast, differentiable engine, and (b) a quantitative,
public realism benchmark against real raw data that we can measure progress on.

## 2. How each effect enters the physics

A useful decomposition — every effect lands in exactly one layer, so the engine needs five
extension points, not forty special cases:

1. **Field at the spin** `B(r, t)` (full vector): B0 map, gradient fields (incl. nonlinearity and
   concomitant terms), eddy currents / GIRF, vibration, drift, physiological fluctuation,
   transmit field B1⁺ (per channel, complex) after the RF chain.
2. **Spin system per isochromat**: one pool (Bloch) → coupled pools (Bloch–McConnell: myelin
   water, bound macromolecular pool with lineshape saturation, CEST pools), J-coupled fat,
   orientation-dependent relaxation.
3. **Sub-voxel and mesoscopic structure**: isochromat distribution within voxels, continuous
   anatomy, susceptibility microstructure (vessels, myelin) → T2′/R2 as *emergent*, diffusion.
4. **Motion and physiology**: spin positions as functions of time (rigid, deformable, flow
   pathlines), coupled to the field model (motion moves spins through B0 and gradient
   nonlinearity).
5. **Receive chain**: B1⁻ per coil, EM-consistent noise covariance, preamp/ADC (filters,
   oversampling, quantisation, gain), spikes and interference.

## 3. Inventory

Status: ✓ have, ◐ partial, ✗ missing. Priority = (magnitude × how often it matters) for the
project's goals (training data, sequence development), at 0.55–7 T.

### 3.1 Fields and hardware

| effect | magnitude / when it matters | who models it | us | how to model | how to verify |
|---|---|---|---|---|---|
| B0 inhomogeneity from anatomy (air–tissue χ) | 10s–100s Hz near sinuses/ear canals at 3 T; EPI distortion, dropout | JEMRIS, POSSUM, MRzero (map) | ◐ χ → B0, order 0–2 shimming, BrainWeb full-head source and separate scanner correction | padded Fourier dipole convolution; scalar tissue/air/bone priors | sphere inside/outside RMS 0.439/0.108 Hz; full-head brain RMS 45.30→19.17 Hz after shim at 3 T; QSM forward-field check pending M3 |
| **Gradient nonlinearity** | mm-scale distortion at FOV edge; b-value errors; vendors correct to 5th order, 9th–10th order leaves ~0.4 mm [14] | JEMRIS (CPU) | ✓ encoding positions (`GradientNonlinearity`, gradunwarp files); ✗ b-matrix | per-axis spherical-harmonic expansion of the encoding field; per-spin effective gradient (and b-matrix) | distortion of a grid phantom vs gradunwarp-style analytic maps; field-camera data |
| **Concomitant (Maxwell) fields** | ∝ G²/B0: signal errors up to 100 % at low field + strong gradients [15]; spiral/EPI/DWI, 0.55 T systems | JEMRIS (CPU), Bschorr [12] | ✓ leading order (closed form) and exact |B| (`Scanner(field_strength=)`) | evaluate |B| of the full vector field (B0 + G·r + concomitant terms from ∇·B = ∇×B = 0) instead of Bz only | Bernstein's analytic phase for PC/spiral; Bschorr's results |
| **Eddy currents / GIRF** | trajectory errors, EPI/spiral blurring and ghosts, bSSFP steady-state disruption [16] | JEMRIS (CPU, exponential), POSSUM (DWI) | ✓ `apply_girf`: GIRF taps / transfer functions, multi-exponential eddy currents, B0 and cross terms | linear time-invariant system per axis (GIRF convolution) incl. B0 and cross terms | measured GIRFs (thin-slice method); trajectory vs field-camera data |
| Gradient delays, amplifier bandwidth | k-space shifts, EPI ghosts | via GIRF | ✓ (GIRF delays; bandwidth via taps) | part of GIRF | – |
| Mechanical vibration | oscillating B0/phase in EPI; odd/even shifts, residual Nyquist ghosts [17] | nobody | ◐ (any measured GIRF incl. resonances; no mechanical model) | resonances in the GIRF (narrow-band poles), B0 modulation | published GIRF magnitude spectra with resonances |
| B0 drift (gradient/passive-shim heating) | median < 1 Hz / 5 min at 3 T, worse on some systems [18] | nobody | ◐ linear / exponential drift (`drift.py`); not yet driven by gradient duty | time-dependent global (and linear) offset driven by gradient duty | multi-site drift data [18] |
| Physiological B0 fluctuation (respiration) | 2–7 Hz in the brain at 7 T, plus small gradients [19] | nobody | ◐ sinusoidal uniform + linear terms (`drift.respiration`) | time-varying low-order spherical-harmonic field from a breathing model | field-probe literature [19] |
| **B1⁺ (transmit) inhomogeneity** | ±20–40 % at 3 T body, worse at 7 T (dielectric effects) | JEMRIS, MRiLab | ◐ analytic maps, `FieldMaps`, loop arrays | import EM-simulated complex B1⁺ per channel (pTx = sum over channels with drive weights) | measured B1 map ✓ in benchmark 3 (moves fitted T2 by ~10 %; the vendor map reads low in short-T1 compartments); EM ground truth |
| RF amplifier nonlinearity, droop, phase | slice-profile widening, flip-angle bias at low/high drive [20] | nobody | ✓ `transmit.amplifier`: Rapp AM/AM, Saleh-type AM/PM, droop with delivered energy (parameters for calibration, R7) | AM/AM, AM/PM transfer curve + low-pass on the RF waveform | published amplifier curves; flip-angle sweeps |
| RF coil ringdown, dead times; played vs. written timing | small, but decisive when T2 ≈ echo spacing: the prime suspect (with local B1⁺) for the 6.5–8.7 % misfit of the 5–15 ms T2 spheres in the NIST TSE benchmark, which sits in the first echoes (`validation/real/RESULTS.md`) | – | ◐ (dead times in `System`) | shift/stretch the first refocusing interval; a B1⁺ map at the phantom | needs a dataset with a B1⁺ map, or scanner timing logs |

**R1 status (October 2026).** Concomitant fields (leading order and exact |B|), gradient
nonlinearity (encoding positions), GIRF / eddy currents incl. B0 and cross terms, per-segment
uniform fields, linear / exponential drift and a periodic respiratory field are in both
simulators, forward and adjoint, opt-in and bit-identical to before when off. Verified against
Bernstein's analytic phase, a Maxwell-compensated echo, |B| in 50-digit arithmetic, gradunwarp's
basis, Laplace's equation, point-grid images, closed-form eddy responses, an FFT
transfer-function convolution, closed-form drift phases and finite differences (numbers in
`ROADMAP.md`, Phase 2b). Not yet: measured GIRFs / field-camera data, b-matrix with
nonlinearity, duty-driven drift, Bschorr et al. comparisons.

### 3.2 Spin system

| effect | magnitude / when it matters | who models it | us | how to model | how to verify |
|---|---|---|---|---|---|
| **Magnetization transfer (bound pool)** | incidental MT in multi-slice: MTR 33 % (1.5 T) → 46 % (3 T) in corpus callosum; up to 30 % signal loss and 20 % T1 bias in IR-EPI [21] | MRiLab, JEMRIS (CPU), MCMR, BMCTool | ✓ two-pool rate model (`exchange.py`, `Phantom.bound`) with super-Lorentzian/Lorentzian lineshape, saturation at each spin's offset from the RF (incidental MT); in the core, the fused CPU kernel (forward + adjoint) and on the Apple GPU. Beyond it: the rate model misses the bound pool's rotation by short intense pulses (T_RF ≲ 0.3 ms) [30]; the generalized Bloch model (`BoundPool(model="generalized")`, Green's-function modes, `memory` selectable) runs in the core, the fused kernel and on the GPU (`validation/RESULTS_GENERALIZED_BLOCH.md`) | Bloch–McConnell with a bound pool saturated by the RF lineshape (super-Lorentzian) and exchange; for short pulses the generalized Bloch model [30] | BMCTool ✓ (Lorentzian: 6e-7 in Z; `validation/compare_bmctool.py`); two-pool ODE and CW steady state ✓; **real NMR ✓** (BSA inversion with 22.8–912 µs pulses: generalized Bloch 0.42 % residual vs the authors' 0.41 %, rate model 1.3 %; benchmark 4); Pulseq-CEST library [10]; qMT literature |
| Multi-compartment exchange (myelin water) | bias of T2/T1 mapping; mcDESPOT/MWF | MRiLab, JEMRIS | ◐ core only: `Phantom.pools`, exact Bloch–McConnell engine (0.3 ms/segment) | same Bloch–McConnell engine, free pools with exchange | analytic two-pool exchange; published MWF |
| CEST pools | CEST imaging | Pulseq-CEST | ◐ core only (`Phantom.pools`); BMCTool's 5-pool model to 8.3e-7 | same engine | Pulseq-CEST library data [10] |
| **J-coupling (fat)** | "bright fat" in TSE vs SE; lipid echo modulation [22] | nobody among imaging simulators | ✗ | per-peak phase-evolution model for coupled lipid protons (density-matrix-lite), or empirical echo-spacing-dependent T2 | density-matrix simulation literature; SE vs TSE fat ratios |
| Multi-peak fat chemical shift | water–fat shift, in/opposed phase | JEMRIS, MRiLab (spectra) | ✓ (6 peaks) | – | ✓ |
| Orientation-dependent R2/R2* (magic angle, susceptibility anisotropy) | WM R2* varies with fibre angle to B0; tendons: T2 maximal at 54.7° [23] | nobody | ✗ | relaxation as a function of fibre direction (from DTI maps) and B0 direction | published orientation curves |
| Non-mono-exponential T2 | intrinsic multi-compartment | via pools | ✗ | pools | – |
| Radiation damping, distant dipolar fields | negligible clinically | – | – | skip | – |

### 3.3 Sub-voxel, mesoscopic and anatomical realism

| effect | magnitude / when it matters | who models it | us | how to model | how to verify |
|---|---|---|---|---|---|
| **Intravoxel dephasing / isochromat count** | too few spins → spurious echoes; up to ~400 per voxel in hard cases [24]; spins per voxel side must exceed the cycles of dephasing carrying signal (our `validation/RESULTS.md`) | all (MRzero: analytic voxel shape) | ✓ jittered sub-voxel spins | add an analytic voxel-shape option where it is exact; adaptive spin counts | EPG ✓; MRzero box voxels ✓ |
| **Continuous anatomy (inverse crime)** | rasterised phantoms simulated and reconstructed on the same grid give optimistic results [25] | nobody in the imaging simulators (Guerquin-Kern analytic phantoms are signal-equation only) | ◐ independent anatomy grid, conservative fractions and sub-voxel quadrature/jitter (M2); unknown subcell geometry is not recovered | refine anatomy independently of reconstruction; separate tissue spins | shifted-box analytic FT error 0.1771→1.44e-5 at 1→64 samples/cell; [results](../validation/anatomy/RESULTS.md) |
| Realistic anatomy and fuzzy tissue | partial volume, textures | MRzero (BrainWeb), MRXCAT/XCAT [26], JEMRIS+XCAT | ◐ BrainWeb loader, cited means/SD metadata and flagged proxies (M2); QSM and spatial variability pending M3/M4 | fuzzy mixtures retain each tissue’s relaxation; missing 7 T classes raise; MT/diffusion priors need calibration | older BrainWeb GM/WM within 0.13%, CSF/GM within 4.74%; newer T1 differences and clinical-validation limits in [results](../validation/anatomy/RESULTS.md) |
| Susceptibility microstructure (vessels, myelin) → T2′, BOLD | T2′ and BOLD are *emergent* from vessel geometry and diffusion; mono-exponential only in limits [27] | MCMR | ◐ (T2′ as Lorentzian) | per-tissue intravoxel frequency distributions and diffusion kernels tabulated from Monte Carlo (MCMR) runs | MCMR; static-dephasing theory |
| Diffusion: free / anisotropic / restricted | DWI; attenuation by imaging gradients at high resolution | JEMRIS, MRzero (free), MCMR (restricted), CMRsim | ✓ exact Brownian walkers (free, tensor; core + Metal, shared Philox walk), restricted in planes / cylinders / spheres with specular reflection; vs Stejskal–Tanner, EPG with diffusion, Tanner–Stejskal series and long-time limits | Bloch–Torrey per isochromat with b-matrix accumulation (free/tensor); restricted via MCMR-derived kernels | Stejskal–Tanner; tensor rotation; MCMR |
| Perfusion / IVIM, contrast agents (DCE/DSC) | time-varying T1/T2/χ | partial (motion frameworks) | ✗ | time-varying tissue properties | tracer-kinetic closed forms |

### 3.4 Motion and physiology

| effect | magnitude / when it matters | who models it | us | how to model | how to verify |
|---|---|---|---|---|---|
| Rigid / deformable motion, spin history | ghosts, blurring, spin-history signal changes | POSSUM, KomaMRI, MRzero, JEMRIS, CMRsim | ◐ rigid and periodic-displacement motion (`motion.py`), spin history automatic; scanner-frame B0 / B1⁺ / coils sampled where the spin is (`maps.py`); not yet with gradient nonlinearity / concomitant fields; no deformable fields beyond displacement × waveform | positions as functions of time, interpolated per segment (free-run closed form generalises with time-varying k·r) | analytic phase for constant velocity (first moment); POSSUM-style spin history |
| Flow (inflow, phase contrast, turbulence) | flow voids, TOF enhancement, PC velocity | KomaMRI, CMRsim (CFD), JEMRIS | ◐ per-spin velocity (plug / profiles): inflow and phase contrast validated; no turbulence or pulsatile profiles yet | spin pathlines with entry/exit; CFD particle import | first-moment phase; TOF steady-state closed form |
| Respiration / cardiac (body), CSF pulsation | ghosts, B0 fluctuation | CMRsim, MRXCAT | ✗ | motion fields + coupled B0 fluctuation | – |

### 3.5 Receive chain and noise

| effect | magnitude / when it matters | who models it | us | how to model | how to verify |
|---|---|---|---|---|---|
| **EM-consistent receive maps and noise covariance** | body-noise dominated at ≥ 1.5 T: noise covariance follows from conductivity-weighted overlaps of coil E fields; g-factor realism | Penn State EM-Bloch [11]; others use analytic maps + i.i.d. noise | ◐ quasi-static loop arrays (`loops.py`): Biot–Savart B1± by reciprocity, noise covariance from ω²∫σ A_i·A_j (Roemer), as `FieldMaps`; full-wave EM maps enter through `FieldMaps` (no format importers yet) | import B1⁻ and E per coil (open EM solvers: MARIE [11], openEMS), Ψᵢⱼ ∝ ∫σ Eᵢ·Eⱼ* dV; quasi-static Biot–Savart at low field | measured SNR / g-factor maps of phantoms |
| ADC: digital filtering, oversampling, dead time, quantisation, gain | passband ripple, edge roll-off, clipping | nobody | ✓ `receive.py`: oversampled readouts, centred decimation FIR with known response, quantisation and clipping, gain drift (dead time: ✗) | receive-chain filter on the oversampled signal | vendor-like filter responses |
| Spikes, RF interference (zipper), noise drift | artefacts for robustness training | nobody | ✓ `receive.py`: spikes, narrowband interference, gain drift | stochastic artefact models | – |

## 4. Strategy: out of the shadow

1. **Realism is measured against real raw data.** Build `validation/real/`: the ISMRM/NIST
   system phantom (spheres with reference T1/T2/PD) scanned with Pulseq sequences, raw data
   kept. Open datasets of exactly this kind already exist (raw data + `.seq` + simulated
   counterparts on 3 T Prisma / 1.5 T Aera [28]; OpenMRF at 0.55/1.5/3 T with the NIST phantom
   [29]). Metric: normalised raw-data error between simulation and scanner on held-out
   sequences, per effect switched on. Publish the table; it's the benchmark nobody has.
2. **Calibrated digital twin via differentiability.** Our unique capability: fit the scanner's
   own imperfections — GIRF, B0/B1⁺ maps, RF-amplifier curve, receive filter, noise
   covariance — from a few calibration scans by gradient descent through the simulator, then
   predict *other* sequences on that scanner. No other simulator can learn its hardware model
   end to end; MRzero is differentiable but has no hardware model to fit.
3. **One engine, five extension points** (§2), all inside the fused kernel's forward and
   adjoint, so each effect stays fast and differentiable. Effects are opt-in and default to
   today's results (the bit-identity discipline we already have).
4. **Borrow the niche leaders as references, not competitors**: BMCTool/Pulseq-CEST for
   exchange, MCMR for microstructure kernels, CMRsim/KomaMRI for flow, JEMRIS for hardware
   effects, MRzero for voxel shapes. Each gets a cross-check like `compare_mrzero.py`.
5. **Avoid the inverse crime by default** in training-data generation: continuous or
   super-resolved anatomy, reconstruction grids decoupled from the simulation grid.

## 5. Proposed order

| step | effects | why first | verified against |
|---|---|---|---|
| R1 field model ✓ (except real-data checks) | vector-field core: gradient nonlinearity (SH), concomitant fields, GIRF/eddy (incl. B0 terms), drift and physiological fluctuation | largest unmodelled hardware effects; JEMRIS-only today and absent from every fast simulator | analytic concomitant phase, SH distortion maps, measured GIRFs, Bschorr et al. |
| R2 spin system (**started**: bound pool in all engines, BMCTool ✓) | Bloch–McConnell: bound pool (MT, lineshape), myelin water, CEST | incidental MT is a 30–45 % WM effect in every multi-slice scan | BMCTool, Pulseq-CEST library, two-pool closed forms |
| R3 real-data benchmark | NIST-phantom Pulseq raw data (open + own), realism metric, effect ablations | turns "realistic" into a number | – (it is the reference). **Started**: benchmark 1 predicts real 3 T IR-SE data to 1.5 % with nothing fitted (`validation/real/RESULTS.md`) |
| R4 receive/transmit chain | EM-imported B1⁺/B1⁻/E, noise covariance from E fields, RF amplifier, ADC filter | noise and coil realism decide training-data usefulness | measured SNR/g-factor; EM ground truth |
| R5 anatomy | χ-map → B0, BrainWeb/XCAT/MIDA, continuous phantoms, orientation-dependent R2* | anatomy realism + inverse crime | Guerquin-Kern closed forms; published orientation curves |
| R6 motion, flow, diffusion | time-dependent positions in the free-run closed form; Bloch–Torrey; MCMR kernels | big but well covered elsewhere — we need parity, then speed | KomaMRI/CMRsim/MRzero/MCMR cross-checks, analytic moments |
| R7 digital twin | differentiable calibration of R1/R4 parameters from calibration scans | the capability nobody else has | held-out sequences on the same scanner (R3) |

J-coupling and vibration slot in when R2/R1 are in place.

## Sources

1. JEMRIS overview — [helmholtz.software/software/jemris](https://helmholtz.software/software/jemris)
2. GPU-accelerated JEMRIS, MAGMA 2025 — [PMC12443918](https://pmc.ncbi.nlm.nih.gov/articles/PMC12443918/)
3. KomaMRI.jl — [github.com/JuliaHealth/KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl) (source: `KomaMRIBase/src/datatypes/Phantom.jl`, `scanner/GradientCoils.jl`, `scanner/RFTransmitSystem.jl`, `KomaMRICore/src/other/DiffusionModel.jl`)
4. KomaMRI motion and flow — [ISMRM 2024 #4686](https://cds.ismrm.org/protected/24MProceedings/PDFfiles/4686_U2PNzrPsQ.html); [portaldelaciencia.uva.es](https://portaldelaciencia.uva.es/documentos/69a5bb1d8abc0d7df38913fd)
5. Phase distribution graphs / MRzero — [cris.fau.de/publications/321606251](https://cris.fau.de/publications/321606251); [arXiv 2606.15164](https://arxiv.org/pdf/2606.15164); [arXiv 2606.09233](https://arxiv.org/pdf/2606.09233) (MRzeroCore source: `simulation/main_pass.py`)
6. MRiLab — [PMC5322984](https://pmc.ncbi.nlm.nih.gov/articles/PMC5322984); [mrilab.sourceforge.net](https://mrilab.sourceforge.net/)
7. POSSUM — [FSL docs](https://fsl.fmrib.ox.ac.uk/fsl/docs/utilities/possum.html)
8. CMRsim — [research-collection.ethz.ch](https://www.research-collection.ethz.ch/handle/20.500.11850/656442)
9. MCMRSimulator.jl, Imaging Neuroscience 2026 — [ndcn.ox.ac.uk](https://www.ndcn.ox.ac.uk/publications/publication_modal/2383978)
10. Pulseq-CEST library — [PMC12255581](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12255581/); BMCTool — [pypi.org/project/BMCTool](https://pypi.org/project/BMCTool/0.5.0/)
11. Bloch-based MRI system simulator with realistic EM fields — [pure.psu.edu](https://pure.psu.edu/en/publications/bloch-based-mri-system-simulator-considering-realistic-electromag/); [ISMRM 2013 #2805](https://archive.ismrm.org/2013/2805.html)
12. Bschorr et al., MRI simulation and reconstruction framework for magnetic vector fields — [arXiv 2512.19470](https://arxiv.org/html/2512.19470v1)
13. Virtual Scanner — [arXiv 2403.17575](https://arxiv.org/abs/2403.17575v1)
14. Gradient nonlinearity calibration — [pr.ibs.re.kr/handle/8788114/3773](https://pr.ibs.re.kr/handle/8788114/3773); [PMC5360473](https://pmc.ncbi.nlm.nih.gov/articles/PMC5360473)
15. Concomitant fields at low field / strong gradients — [lup.lub.lu.se](https://lup.lub.lu.se/record/9b78e91d-5f59-47ac-89d1-660e8b951b92); [PMC5217757](https://pmc.ncbi.nlm.nih.gov/articles/PMC5217757)
16. GIRF-based eddy-current modelling — [PMC7154723](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7154723/)
17. Gradient mechanical resonances in EPI — [MRM Highlights: beat phenomena of oscillating readouts](https://mrm.ismrm.org/t/beat-phenomena-of-oscillating-readouts/1202)
18. Frequency drift across 99 3 T scanners, NeuroImage 2021 — [nottingham-repository](https://nottingham-repository.worktribe.com/index.php/output/47005782/frequency-drift-in-mr-spectroscopy-at-3t)
19. Respiration-induced B0 in the head — [pr.ibs.re.kr/handle/8788114/5895](https://pr.ibs.re.kr/handle/8788114/5895?mode=full); [PMC6239980](https://pmc.ncbi.nlm.nih.gov/articles/PMC6239980)
20. RF amplifier distortion and predistortion — [PMC3391187](https://pmc.ncbi.nlm.nih.gov/articles/PMC3391187); [ISMRM 2024 #1417](https://cds.ismrm.org/protected/24MProceedings/PDFfiles/1417_XVky2JROa.html)
21. Incidental MT in multi-slice brain MRI at 3 T — [kumel.medlib.dsmc.or.kr](https://kumel.medlib.dsmc.or.kr/handle/2015.oak/34156); [PMC2779549](https://pmc.ncbi.nlm.nih.gov/articles/PMC2779549)
22. J-coupling and bright fat in FSE — [mriquestions.com/bright-fat](https://mriquestions.com/bright-fat.html); [PMC6401018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6401018/)
23. Orientation-dependent R2* / magic angle — [PMC3638902](https://pmc.ncbi.nlm.nih.gov/articles/PMC3638902); [PMC6436728](https://pmc.ncbi.nlm.nih.gov/articles/PMC6436728/)
24. Isochromat counts and spurious echoes — [BlochSolver, JMR 2017](https://www.bk.tsukuba.ac.jp/~mrlab/PDF_labo_paper/2017_JMR_BlochSolver.pdf); [SIMRI](https://www.creatis.insa-lyon.fr/~yougz/references/beno05-simri.pdf)
25. Guerquin-Kern et al., realistic analytical phantoms, IEEE TMI 2012 — [bigwww.epfl.ch](https://bigwww.epfl.ch/publications/guerquinkern1201.html)
26. MRXCAT / XCAT and BrainWeb — [ISMRM 2020 #1871](https://cds.ismrm.org/protected/20MProceedings/PDFfiles/1871.html); [BrainWeb](https://www.mcgill.ca/bic/software/brainweb-mri-simulator)
27. Vessel-network Monte Carlo and static dephasing — [PMC3999331](https://pmc.ncbi.nlm.nih.gov/articles/PMC3999331); [Sci. Rep. 2017](https://link.springer.com/10.1038/s41598-017-15968-4)
28. Open-source MRI acquisition: data and documentation for two validated pulse sequences — [doaj.org](https://doaj.org/article/12c0f01b03a54996b3006b4b13c6bfcf)
29. OpenMRF (Pulseq MRF, NIST phantom, 0.55/1.5/3 T) — [arXiv 2604.22713](https://arxiv.org/pdf/2604.22713)
30. Generalized Bloch model for pulsed MT (Assländer et al.) — [arXiv 2107.11000](https://arxiv.org/abs/2107.11000)
