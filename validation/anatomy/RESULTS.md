# Anatomy M2 results — 2026-10-07

M2 adds BrainWeb loading and tissue spin assembly without engine changes.
Full suite with Apple GPU access: **449 passed, 4 opt-in crosschecks deselected**
(185.08 s). Existing default output/gradient snapshots are **bit-identical**.

## Independent synthetic checks

`tests/test_anatomy.py` contains 21 tests including parameterized cases.

| Check | Result |
|---|---|
| Per-tissue and per-voxel PD conservation | absolute error < 1e-12, float64; K=1, 8 and (1,2,3) |
| Resampled physical tissue volume | relative error < 1e-6; three nontrivial spacings, float32/64 |
| Mixed-tissue spoiled GRE vs Ernst closed form | maximum complex error < 1e-8 |
| Shifted continuous box vs analytic Fourier transform | maximum normalized error 0.177089, 0.00384746, 0.000230445, 0.0000143652 at 1,4,16,64 x samples/cell |
| Sphere susceptibility at jittered spin positions | RMS 0.464157 Hz vs independent `b0_sphere`, 3 T |
| Anatomical GPU vs core, B0 + MT + diffusion | maximum relative signal error 3.23e-7 |
| 4096 Cauchy quantiles vs exponential T2* FID | maximum complex error < 0.005 |

The box anatomy grid is offset 0.3 mm from a nominal reconstruction grid. Subcell
quadrature converges to its continuous, piecewise-constant density. This does not
prove that fuzzy memberships recover unknown subcell boundary geometry. Additional
checks cover categorical air/tissue sampling, seeds, fat mass, MT pool conversion,
checksums, coordinate updates, separate scanner shims and T2* double counting.

## Downloaded reference contrasts

Run `uv run python validation/anatomy_brainweb.py` after fetching with `--legacy`.
Manifests pin decompressed source bytes, with all volumes kept outside Git.
Reference ROIs use memberships > 95%: 449,752 WM, 509,944 GM, 202,279 CSF voxels
for the older SBD. Report median image intensities, divided into tissue ratios.

The [published tissue file](https://brainweb.bic.mni.mcgill.ca/brainweb/tissue_mr_parameters.txt)
supplies the older 1.5 T model. Its [ICBM protocol templates](https://brainweb.bic.mni.mcgill.ca/cgi/bw/submit_request)
specify SFLASH T1 (TR/TE=18/10 ms, flip=30°), and dual spin echo
(TR=3300 ms, TE=35/120 ms; early echo PD, late echo T2). Downloaded aliases use
1-mm voxels, zero noise, zero RF nonuniformity. Bloch signals use ideal pulses,
steady-state initialization for GRE and 24 repeated TRs for DSE, including both
longitudinal inversions. Uniform fields and linearity allow tissue basis signals
without a full spatial reconstruction.

| Protocol | Ratio | Bloch | BrainWeb image | Relative difference |
|---|---|---:|---:|---:|
| T1 | GM/WM | 0.742607 | 0.742714 | −0.0144% |
| T1 | CSF/GM | 0.401751 | 0.408072 | −1.549% |
| T2 | GM/WM | 1.431303 | 1.433099 | −0.1253% |
| T2 | CSF/GM | 2.488377 | 2.375921 | +4.733% |
| PD | GM/WM | 1.183405 | 1.183080 | +0.0275% |
| PD | CSF/GM | 1.157072 | 1.132151 | +2.201% |

The [20-model release](https://brainweb.bic.mni.mcgill.ca/brainweb/anatomic_normal_20.html)
provides **only T1**, with SFLASH TR/TE=22/9.2 ms and flip=30°. Subject 04 is
checked separately, interpolating its different image FOV/grid at anatomy centres:

| Ratio | Bloch with older tissue priors | Subject-04 image | Relative difference |
|---|---:|---:|---:|
| GM/WM | 0.753111 | 0.796414 | −5.44% |
| CSF/GM | 0.411395 | 0.522794 | −21.31% |

These are cross-simulator plausibility checks, **not scanner ground truth**.
The larger subject-04 differences remain unexplained; the older tissue file has
not been established as the exact parameter set behind the newer release. Pure
ROI thresholds also do not eliminate every partial-volume/interpolation effect.
No tolerance was tuned to hide those differences.

### In-vivo comparison limit

There is no matched clinical image-ratio validation in M2. As a sensitivity check,
the script propagates the human 3 T WM/GM mean T1/T2 values from
[Wansapura et al. 1999](https://pubmed.ncbi.nlm.nih.gov/10232510/) through the same
protocols, retaining the stated PD/T2* priors. Predicted GM/WM ratios are 0.709695
(T1), 0.688959 (T2), 0.920534 (PD). These are **predictions from measured relaxation
values**, not observed in-vivo ratios; field strength and population differ from
the older BrainWeb model. They cannot establish 1.5 T clinical accuracy. Matching
real acquisitions and calibrating the tissue priors remain separate validation work.

## Full-head susceptibility and shim

Subject 04 at 2 mm, 3 T, complete head source, 2× padding, brain membership > 0.5:
RMS falls from **45.2976 Hz** to **19.1716 Hz** after second-order shimming. Residual
1st/50th/99th percentiles are −52.8329 / 0.5364 / 51.4486 Hz. Tissue and scanner
corrections remain separate maps. This uses scalar water-like soft tissue, air,
and a −2 ppm bone proxy; it is not a comparison to a measured head field. QSM
Challenge forward-field validation belongs to M3.

## Full 3D GPU benchmark

Run `uv run python validation/anatomy_brainweb.py --benchmark-only` with Apple GPU
access. Local macOS host, Python 3.12.13. Full 1-mm subject-04 anatomy, categorical
K=1, seed=0, all head tissues, six fat peaks; 1-ms 30° pulse split into 50 segments
followed by 32 FID samples at 0.2-ms dwell (83 sequence segments total). Shader
warm-up is excluded and GPU synchronization brackets the timer.

| Measurement | Value |
|---|---:|
| Spins | 5,328,321 |
| Read/normalize/resample | 5.807 s |
| Spin assembly | 0.308 s |
| GPU simulation | 0.686 s |
| Spins per second for this sequence | 7.77 million |
| Phantom tensor storage | 149,192,988 bytes |
| Process peak RSS | 3,640,066,048 bytes |
| MPS driver allocation after simulation | 1,082,621,952 bytes |
| Finite output signal | yes |

This measures basic volumetric RF/FID throughput, **not a complete imaging scan**.
B0, microscopic T2′, MT and diffusion are off in this benchmark; their joint GPU
agreement is checked separately above. K=1 is not a spatial/partial-volume
convergence result. Peak RSS includes loading and CPU intermediates; the driver
allocation is an end-of-run reading, not measured peak GPU memory. Apple unified
memory means these numbers must not be added as independent allocations. Multiple
spins per tissue and full-head field construction cost more.

## Tissue provenance and remaining assumptions

Code records means, reported SDs, source/row and assumption flags. Unknown SD is
`None`. Key sources are [Stanisz 2005, Table 1](https://doi.org/10.1002/mrm.20605)
(ex-vivo tissue proxies at 1.5/3 T), Wansapura above (regional 3 T T2*), and
[Caan 2019, Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC6590660/)
(human 7 T WM/GM T1/T2*). Full 7 T tissue coverage, calibrated MT/diffusion,
microstructure/iron susceptibility and spatial variability are not provided.
See [README.md](README.md) for the explicit T2* subtraction policy and usage.
