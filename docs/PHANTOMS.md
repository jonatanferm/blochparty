# Digital phantoms: generative, physically consistent, indistinguishable from real scans

Design notes, October 2026. Companion to [REALISM.md](REALISM.md), which covers the *engine*;
this document covers what goes *into* it. Status: ideas. What exists today is
`phantom_from_maps` (PD, T1, T2, T2′, B0, B1⁺, multi-peak fat; sub-voxel spins) and analytic
phantoms (Shepp–Logan, χ spheres/cylinders).

Two goals, the second one the bigger prize:

1. **Generative phantoms**: a parameterised model that samples a brain, a beating heart or a
   DCE breast whose simulated scans are hard or impossible to tell apart from real ones.
2. **Running the simulator backwards** (§5): from a *reconstructed* scan (DICOM — abundant) to
   believable *raw multi-coil k-space* (rare). If it works, every clinical archive becomes
   k-space training data.

## 1. The requirement

A phantom is not an image. It is a set of physical property fields, and the image comes out of
the simulator. Image generators already produce brain MRI that is hard to tell from real —
e.g. latent diffusion models trained on 31 740 UK Biobank T1w volumes, with 100 000 synthetic
brains released [1] — but they solve a weaker problem. Ours is:

> **The phantom must look real under _any_ sequence, including ones it was never tuned on.**

That is what makes a phantom more valuable than an image generator: one sample yields a T1w, a
FLAIR, a DWI, a bSSFP cine and the raw k-space of each, all physically consistent with each
other.

## 2. What a phantom has to contain

Every physical layer in REALISM.md §2 needs its inputs from the phantom.

| property | feeds (REALISM layer) | today |
|---|---|---|
| PD, T1, T2 | spin system | ✓ maps |
| T2′ / intravoxel frequency distribution | spin system, microstructure | ✓ Lorentzian per voxel |
| fat fraction, fat spectrum, fat T1/T2 | spin system | ✓ 6 peaks |
| bound-pool fraction, exchange rate, lineshape (MT); myelin water pools; CEST pools | spin system (Bloch–McConnell) | ✗ |
| susceptibility χ → B0 by dipole convolution | fields | ◐ analytic shapes; B0 maps |
| conductivity σ, permittivity ε → B1⁺, B1⁻, E (noise covariance) | fields, receive chain | ◐ analytic B1⁺ / coil maps |
| diffusion tensor, fibre orientation (→ orientation-dependent R2*) | microstructure | ✗ |
| vessel trees, perivascular spaces (→ emergent T2′, flow, BOLD) | microstructure, motion | ✗ |
| displacement fields over time (cardiac, respiratory, CSF, bulk) | motion | ✗ |
| velocity fields / pathlines (inflow, pulsatile flow) | motion | ✗ |
| tracer kinetics (Ktrans, ve, vp, AIF → T1(t), T2*(t)) | time-varying properties | ✗ |

**Representation.** Continuous, not voxels: meshes or label surfaces plus procedural texture,
or neural implicit fields. Sub-voxel spins sample it directly at any resolution, so the
simulation grid never equals the reconstruction grid and we avoid the inverse crime
(REALISM §3.3). Time enters as properties and positions that are functions of t.

## 3. A factorised generative model

One giant model would be untrainable and uncontrollable. Instead we factor the phantom into
pieces with separate data sources, separate knobs and separate failure modes:

| factor | latent | controls | data | model |
|---|---|---|---|---|
| anatomy | z_anatomy | age, sex, body size, ventricle volume, pathology | label maps (SynthSeg/FreeSurfer on large cohorts [2]), MIDA (153 structures, 0.5 mm, head and neck) [3], BrainWeb, XCAT | diffusion model over label maps or meshes; deformation fields on templates |
| tissue physics | z_tissue | field strength, age (myelination), per-tissue offsets | literature values; qMRI datasets; **weighted images through the simulator (§4)** | per-tissue joint distributions of (T1, T2, PD, χ, MT, …), spatially varying |
| texture | z_texture | granularity, vessel density | high-res anatomy; procedural rules | Gaussian random fields; vessel-tree growth; cortical folding from surfaces |
| dynamics | z_dynamics | heart rate, breathing pattern, contrast-injection timing | cine (CMRxRecon [4]), motion literature, kinetic models | shape-model motion, biophysical LV models (MRXCAT2.0 [5]), Tofts kinetics |
| derived fields | — | B0 shim, coil array | — | physics: χ → B0, (σ, ε) → B1⁺/B1⁻/E |
| pathology | z_lesion | lesion type, size, location, enhancement curve | segmentations, radiology literature | insertion models conditioned on anatomy |

Parameterisation comes for free: set the knobs you care about ("60-year-old, enlarged
ventricles, 2 cm enhancing lesion, HR 75") and sample the rest.

**Procedural v0** — useful long before any learning: take labels from BrainWeb/MIDA, deform
them randomly (SynthSeg-style domain randomisation [2], but drawing *physical* parameters
instead of random intensities), assign tissue properties from literature distributions with
inter-subject variance and correlation, add random-field texture, compute B0 from χ.

## 4. Learning tissue physics *through* the simulator

### 4.1 The data asymmetry

| data | how much exists | what it tells us |
|---|---|---|
| quantitative maps (MRF, QALAS, mcDESPOT, MP2RAGE T1) | small, low resolution, method-specific biases | properties directly, but biased |
| raw multi-coil k-space | moderate: fastMRI, M4Raw [6], CMRxRecon [4], SKM-TEA [7], fastMRI Breast [8] | everything the scanner measured |
| reconstructed weighted images (DICOM) with protocol headers | enormous: every hospital archive, UK Biobank, OASIS, IXI, ADNI | properties *as seen through a known sequence* |

The abundant data is weighted images, and the weighting is something we can simulate exactly.
Because blochparty is differentiable, it can serve as the **measurement operator** of a
generative model:

```
generator → property maps → blochparty(seq from DICOM header, coils, noise) → recon
                                                                                │
                                         real scan acquired with that seq ──► compare
```

This is AmbientGAN [9] (learn a distribution from measurements only, with the measurement
process inside the training loop), with a Bloch simulator as the measurement process.
Ambient diffusion has already been trained on undersampled multi-coil MRI measurements [10];
here the operator is nonlinear and physical rather than a Fourier mask.

### 4.2 Two stages, not one adversarial loop

1. **Amortised inversion.** A network maps a study's weighted images (+ headers) to property
   maps and is trained self-supervised: re-simulating every acquired contrast from its output
   must reproduce the input. Prior art does exactly this with signal equations: Moya-Sáez et
   al. trained on BrainWeb-synthesised T1w/T2w pairs [11]; van Lune et al. trained
   self-supervised with Bloch-based signal models on 4 121 clinical sessions from four 3 T
   scanners, getting T1/T2/PD consistent with literature and invariant across scanners
   (CoV ≤ 1.1 %) [12]. What we add:
   - the *actual* protocol simulated at sequence level (TSE echo trains with their
     refocusing angles and view ordering, IR preparation, incidental MT, B1⁺, slice
     profiles), not a closed-form signal equation;
   - more parameters (T2′, χ, MT pool, fat, B1⁺/B1⁻ as nuisances);
   - consistency at the raw-data level where raw data exists.
2. **Prior.** Train the generative model (§3) on thousands of inferred map sets. Far more
   stable than adversarial training through a simulator, and each stage is testable on
   its own.

### 4.3 Degeneracies — the main risk

The generator will happily put realism into the wrong parameter.

| confounded pair | in which data | mitigation |
|---|---|---|
| PD vs receive sensitivity B1⁻ | any single contrast | B1⁻ as explicit smooth nuisance field, constrained by EM physics of the body outline |
| T1 vs B1⁺ error | T1w GRE, MP-RAGE | B1⁺ as nuisance from EM; multi-flip-angle data where available |
| T2 vs T2′ | GRE-only studies | T2′ prior from χ/vessel model; SE contrasts when present |
| T2 vs MT and stimulated echoes | multi-slice TSE | sequence-level simulation (this is where signal equations break) |
| χ vs B0 shim | phase data | shim as low-order SH nuisance |
| tissue vs scanner imperfections | everything | scanner model calibrated separately on phantom data (REALISM R3/R7) |

Anchors: literature value distributions as priors, the few qMRI datasets as held-out
checks (SKM-TEA has quantitative T2 with its raw data [7]).

### 4.4 Sequence knowledge

DICOM headers give TR, TE, TI, flip angle, echo-train length, bandwidth, matrix, partial
Fourier, acceleration, field strength, vendor and coil. They map onto our builders. What they
do not give — vendor refocusing trains, view ordering, k-space filters, normalisation — becomes
a small set of **latent sequence parameters per vendor protocol family**, calibrated once on
paired raw/DICOM data (§5.4) and then reused.

## 5. Running it backwards: believable raw k-space from reconstructed scans

### 5.1 Why

Deep learning on k-space is starved for data: raw data is rarely kept, while reconstructed
images exist by the million. Today's workarounds:

- **FFT of the DICOM.** Shimron et al. showed this is an *implicit data crime*: processing
  pipelines hidden in public data make reconstruction algorithms look up to 48 % better than
  they are [13].
- **Image-domain synthesis.** Learned phase from a conditional GAN + coil maps taken from
  ESPIRiT of other subjects' raw data + FFT; recon networks trained on it approach those
  trained on real raw data [14].

Both remain image-domain: the k-space is the Fourier transform of *some* image. But **the
k-space of a TSE is not the Fourier transform of any image.** Echo-train T2 modulation across
k_y, the IR transient, off-resonance accumulating during the readout, EPI phase errors and
motion between shots live only in k-space, and a reconstruction network trained on real data
learns to deal with exactly these. Only a sequence-level simulator can put them back.

### 5.2 What a DICOM has thrown away, and how a lift restores it

| lost in the DICOM | what real raw data has | how the lift restores it |
|---|---|---|
| phase | B0 phase (from χ), transmit/receive phase per coil, flow and motion phase | χ → B0 by dipole convolution; EM-consistent coil phases; flow from vessel model |
| coil-resolved data | N coils with physically consistent B1⁻ maps | EM simulation on the body outline (or a learned EM surrogate) |
| noise | complex Gaussian per coil, correlation Ψ from E fields; ADC filter roll-off | EM-consistent Ψ; receive-chain model (REALISM §3.5) |
| resolution beyond the matrix | continuous object → realistic Gibbs and high-k energy | super-resolved continuous phantom (sampled from the posterior, §5.3) |
| readout oversampling | 2× in most Cartesian data | simulate the oversampled readout |
| partial Fourier, zero-filling, interpolation, apodisation, distortion correction, intensity normalisation | none of these (they happen in recon) | learned/known vendor recon model, *inverted*; latent per protocol family (§4.4) |
| magnitude-only, coil combination (RSS noise floor), quantisation | complex, uncombined | discarded, the phantom carries the signal instead |
| acquisition physics in k-space | echo-train modulation, IR transient, off-resonance during readout, view ordering + motion | **the Bloch simulation itself** |

### 5.3 The pipeline

```
DICOM series + header
      │  (a) posterior sample: super-resolved property maps + nuisances
      │      (B0/χ, coil outline, motion), conditioned on the image and the protocol
      ▼
  phantom twin(s) ── (b) blochparty: sequence from header (+ vendor latents),
      │                EM coils, EM noise, hardware chain (REALISM R1/R4)
      ▼
multi-coil raw k-space (ISMRMRD), any acceleration / coil array / noise level
```

Two levels of ambition:

- **Contrast lift.** One series → maps that are *effective for that sequence* (enough to
  re-acquire it with other undersampling, coils, noise, motion). Needs only (a) with a
  single-contrast posterior. Already enough for reconstruction training.
- **Study lift.** A multi-contrast study → a full twin (§4) that can be re-scanned with *other*
  sequences, at other field strengths. The patient digital twin.

The lift is one-to-many: a DICOM does not determine its raw data. We sample several twins per
image. This is honest about the uncertainty and works as physically grounded augmentation.

### 5.4 Validation with paired data — the benchmark that makes this real

Several public datasets ship raw data *and* reconstructions of the same acquisitions:
fastMRI (multi-coil knee and brain), SKM-TEA (raw + scanner-generated DICOM, GE 3 T) [7],
fastMRI Breast (radial DCE raw + DICOM, 3 T, 16-channel breast coil) [8], M4Raw (0.3 T, T1w/T2w/
FLAIR of the same subjects, repetitions) [6], CMRxRecon (cine, mapping, flow, tagging) [4].

Protocol: hide the raw data, lift the reconstruction, compare against the hidden raw data.

1. **Distributional match in k-space**: noise covariance; coil-map smoothness and phase
   versus measured; phase power spectrum; radial energy profile beyond the DICOM resolution;
   k_y modulation in TSE; classifier two-sample test [15] on raw k-space.
2. **Downstream (the one that counts)**: train reconstruction networks on (i) real raw,
   (ii) lifted, (iii) FFT of DICOM, (iv) GAN phase + ESPIRiT coils [14]; test all on held-out
   real raw. Success: (ii) ≈ (i), and (i) + lifted large DICOM archives > (i) alone.
3. **Data-crime check**: measure the over-optimism of (ii) the way Shimron et al. measure it
   for (iii) [13].

Evaluation is always against real raw data, never against our own simulations, so the method
cannot grade itself.

### 5.5 Why blochparty is the right tool

Sequence-level physics (so k-space carries the acquisition, not just the image);
differentiability (so the lift is amortised inference trained by gradient descent through the
simulator, and vendor latents can be calibrated); speed (so posterior sampling of many twins
per image is affordable); and the hardware chain from REALISM (so coils and noise are
EM-consistent instead of borrowed).

## 6. Measuring "indistinguishable"

- **Classifier two-sample test** [15]: train real vs simulated; AUC → 0.5. Compare *raw*
  k-space through the *same* recon as the real data, or the classifier learns DICOM scaling
  and recon pipelines rather than anatomy.
- **Sim-to-real**: train segmentation/reconstruction on simulated data, test on real.
- **Reader studies**: a small visual Turing test with radiologists at milestones.

What gives synthetic scans away is rarely grey/white contrast. It is vessels (flow voids,
inflow, pulsation ghosts), scalp fat and chemical shift at boundaries, eye motion, sinus
dropout, perivascular spaces, white-matter texture, dura and skull layers, and noise without
coil-combination correlations. BrainWeb looks synthetic mostly because of what lies *outside*
the brain.

## 7. Three anatomies

**Brain**: the place to start. Most labels (SynthSeg [2]), MIDA [3] and BrainWeb for the head,
most multi-contrast and raw data (fastMRI, M4Raw). Vessel trees and χ make B0 and T2′
*emerge* from anatomy instead of being painted on.

**Beating heart**: mostly dynamics. LV/RV shape model with motion over the cycle (MRXCAT2.0
couples XCAT with a statistical shape model and biophysical LV simulation, texturised tissue,
bSSFP [5]; XCAT licensing is an obstacle, so prefer open shape/motion models learned from
cine), blood as pathlines through the chambers, respiratory coupling. CMRxRecon [4] is the
raw-data reference. Depends on REALISM R6 (motion and flow in the engine).

**DCE breast**: mostly kinetics and fat. Fibroglandular structure from the VICTRE procedural
generator (open source, ten tissue types incl. ducts, ligaments, vessels) [16], Tofts kinetics
per tissue and lesion with an AIF (prior art: an anthropomorphic breast DCE digital reference
object [17]), background parenchymal enhancement, fat dominated (our 6-peak model; J-coupling
matters, REALISM §3.2). fastMRI Breast [8] (radial raw + DICOM, 275 labelled cases) is the
realism and lift reference.

## 8. Risks and open questions

- **Degeneracies** (§4.3): the main scientific risk; ablate each nuisance model.
- **Circularity**: inversion, prior and lift all use the simulator; only real raw data may
  judge them (§5.4).
- **Vendor black boxes**: recon filters and sequence details are unknown; how few latents per
  protocol family suffice is an empirical question.
- **Do we need full Bloch?** For many contrasts a signal equation may do; ablate per
  contrast, keep Bloch where k-space physics matters (TSE, IR, EPI, bSSFP).
- **Compute**: 3D super-resolved phantoms × many spins per voxel × many twins per image.
  Needs the fast paths (PDG/signal-equation backends, ROADMAP parking lot) for the bulk.
- **Privacy and memorisation**: twins lifted from patient DICOM inherit identifiability
  (faces!); generative models trained on patient data can memorise. Deface, and test
  nearest-neighbour memorisation before releasing models or data.
- **Licensing**: XCAT, ViP and parts of MIDA have restrictive licences; prefer open sources
  for anything we redistribute.

## 9. Proposed order

| step | what | verified against |
|---|---|---|
| P0 schema | continuous, multi-property, time-varying phantom format extending `phantom_from_maps`; ISMRMRD/HDF5 export with provenance | round trips; bit-identity with today's maps |
| P1 procedural v0 | BrainWeb/MIDA labels + random deformations + literature property distributions + texture + χ → B0 | literature tissue values; analytic χ cases ✓ |
| P2 realism benchmark | two-sample test and sim-to-real on fastMRI brain; per-effect ablations | real raw data |
| P3 contrast lift | DICOM → raw k-space for one contrast; paired validation (§5.4) on fastMRI, SKM-TEA, M4Raw | hidden real raw; recon-training downstream test |
| P4 learned anatomy | diffusion model over labels/meshes conditioned on age, sex, pathology | label statistics; memorisation tests |
| P5 learned tissue physics | amortised inversion (§4.2) then prior; study lift | held-out qMRI (SKM-TEA T2); re-simulation of held-out contrasts |
| P6 heart, breast | dynamics and kinetics, after REALISM R6 | CMRxRecon; fastMRI Breast |

P3 comes early on purpose: it needs only single-contrast inversion and the receive chain
(REALISM R4), and it answers the most valuable question first.

## Sources

1. Pinaya et al., Brain imaging generation with latent diffusion models, MICCAI DGM4MICCAI 2022 — [arXiv 2209.07162](https://arxiv.org/abs/2209.07162)
2. Billot et al., SynthSeg, Medical Image Analysis 83:102789 (2023) — [arXiv 2107.09559](https://arxiv.org/abs/2107.09559); [github.com/BBillot/SynthSeg](https://github.com/BBillot/SynthSeg)
3. Iacono et al., MIDA, PLoS ONE 2015 — [PMC4406723](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4406723/); [itis.swiss MIDA](https://itis.swiss/virtual-population/regional-human-models/mida-model)
4. CMRxRecon, Sci Data 2024 — [doi 10.1038/s41597-024-03525-4](https://link.springer.com/10.1038/s41597-024-03525-4); CMRxRecon2024 — [arXiv 2406.19043](https://arxiv.org/abs/2406.19043)
5. MRXCAT2.0, JCMR 2023 — [PMC10116689](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10116689/)
6. M4Raw, Sci Data 10:264 (2023) — [PMC10172399](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10172399/); [zenodo 8056074](https://zenodo.org/records/8056074)
7. Desai et al., SKM-TEA, NeurIPS Datasets & Benchmarks 2021 — [arXiv 2203.06823](https://arxiv.org/abs/2203.06823)
8. fastMRI Breast, Radiology: AI — [arXiv 2406.05270](https://arxiv.org/abs/2406.05270); [pubs.rsna.org/doi/10.1148/ryai.240345](https://pubs.rsna.org/doi/epdf/10.1148/ryai.240345)
9. Bora, Price, Dimakis, AmbientGAN, ICLR 2018 — [mlanthology.org](https://mlanthology.org/iclr/2018/bora2018iclr-ambientgan)
10. Ambient diffusion posterior sampling (training on Fourier-subsampled multi-coil MRI), ICLR 2025 — [arXiv 2403.08728](https://arxiv.org/abs/2403.08728)
11. Moya-Sáez et al., synthetic MRI from two routine sequences, trained on synthetic data, CMPB 2021 — [uvadoc.uva.es](https://uvadoc.uva.es/handle/10324/48450)
12. van Lune et al., Quantitative mapping from conventional MRI using self-supervised physics-guided deep learning, 2026 — [arXiv 2601.05063](https://arxiv.org/abs/2601.05063)
13. Shimron, Tamir, Wang, Lustig, Implicit data crimes, PNAS 2022 — [PMC9060447](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9060447/)
14. Deveshwar et al., Synthesizing complex-valued multicoil MRI data from magnitude-only images, Bioengineering 2023 — [PMC10045391](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10045391/)
15. Lopez-Paz, Oquab, Revisiting classifier two-sample tests, ICLR 2017 — [arXiv 1610.06545](https://arxiv.org/abs/1610.06545)
16. VICTRE breast phantom (FDA, open source) — [breastphantom.readthedocs.io](https://breastphantom.readthedocs.io/); [FDA regulatory science tool](https://cdrh-rst.fda.gov/node/1601)
17. Anthropomorphic digital reference object for breast DCE MRI — [PMC9031444](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9031444/)
