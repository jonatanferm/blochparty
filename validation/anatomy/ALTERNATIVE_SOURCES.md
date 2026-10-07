# Accessible alternatives to the complete QSM Challenge phantom

Checked 2026-10-07. **AHEAD v3 is the best verified replacement for the missing
brain parameter maps.** It supplies co-registered susceptibility, R1, R2* and
relative proton density. It can support a separate brain-template simulation
path, but does not replace the original whole-head phantom or its independent
simulation ground truth. M3's original acceptance criteria remain open.

## Recommended: AHEAD quantitative MRI templates, v3

- [Publisher record](https://uvaauas.figshare.com/articles/dataset/AHEAD_quantitative_MRI_templates_MNI2009b_/19646364),
  DOI [10.21942/uva.19646364.v3](https://doi.org/10.21942/uva.19646364.v3).
- Authors: P.-L. Bazin, B. Forstmann, A. Alkemade, S. Miletić, S. Isherwood.
- Public direct downloads, no account or agreement required; **CC BY 4.0**.
- Population medians from 105 participants acquired at 7 T, in MNI2009b space.
  The record also provides interquartile ranges; these are population summaries,
  not individual subjects or a spatial covariance model.
- Downloaded the four median maps, **250,309,250 bytes** compressed in total.
  Each matches its publisher-provided MD5. Compressed and decompressed SHA-256
  hashes are pinned in [ahead_source.json](ahead_source.json).
- All four actual NIfTI files share shape **394×466×378** in stored x/y/z,
  **0.5 mm** spacing and exactly equal affines. Storage orientation is **RPS**:
  the y axis must be flipped when canonicalizing to RAS. Header spatial units
  are mm. Affine diagonal is (0.5, −0.5, 0.5), translation (−98, 98.5, −72) mm.
- Every array value is finite; rates and relative density are nonnegative.

| Map filename component | Meaning | Observed raw range | Nonzero voxels |
|---|---|---:|---:|
| `med_qsmap` | Reconstructed susceptibility | −0.161053…0.118474 | 15,322,642 |
| `med_r1map` | Longitudinal relaxation rate | 0…1.936060 | 16,950,701 |
| `med_r2map` | **R2***, despite the shortened filename | 0…151.554398 | 16,950,701 |
| `med_pdmap` | Relative proton density | 0…1.506546 | 16,950,701 |

These are raw numerical ranges, not a calibration certificate. The expected
QSM/rate conventions are ppm and s⁻¹, respectively; establish those explicitly
from the reconstruction provenance before wiring a public loader. The NIfTI
spatial/time unit fields do not themselves specify voxel intensity units.

Reproduce the actual-file checks:

```sh
uv sync --extra anatomy
uv run python validation/anatomy/fetch_ahead.py
# Verify the existing downloads without contacting the server:
uv run python validation/anatomy/fetch_ahead.py --offline
```

Files remain ignored under `validation/anatomy/data/ahead/`. The fetcher verifies
hashes, common grids, finite values and nonnegative rates/density. It deliberately
does not create a `qsm2.json` manifest or label this atlas as a QSM Challenge release.

### What this replacement can and cannot establish

AHEAD is suitable for testing real-map ingestion, coordinate handling, conservative
resampling, relaxation and susceptibility-driven **brain-only** simulations.
Its newer [acquisition/reconstruction paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12319765/)
describes relative PD estimation from MP2RAGEME and distinguishes R2* from R2.
That paper covers the wider AHEAD release, including separate high-resolution
slabs; those slabs are not the population template downloaded here.

Implementation must preserve relative density as relative density, use the 7 T
acquisition field, and explicitly define the valid tissue mask. QSM can genuinely
be zero inside tissue, so `chi != 0` is not a valid brain-mask rule. The record
does not include the challenge's 16-label segmentation, full-head air/bone map,
intrinsic R2, or a ground-truth field produced by its simulator. A skull-stripped
QSM exterior of zero must not be interpreted as physical air at 9.2 ppm.

Recommended validation strategy: use AHEAD for measured-map ingestion and brain
simulations, retain BrainWeb for whole-head geometry, and retain the public QSM
challenge ground truths plus the independent forward calculation for field
verification. Do not splice maps from these different anatomies voxel by voxel,
or claim that this combination executes the unavailable original MATLAB code.
An AHEAD loader and its physics checks are a next implementation step, not
something completed by this source-discovery work.

## Other sources inspected

| Source | Actual available files / access | Fit for this task |
|---|---|---|
| [NeuroPoly susceptibility-separation phantom](https://github.com/neuropoly/Susceptibility-Separation-Phantom) and [OSF data](https://osf.io/9xwhz/) | Public GitHub tree at `d009ab67660fd6e5368d9f9811e04e19eca1636a`; recursively listed public OSF files. Chi+/chi−, anisotropic chi−, masks, simulated T2, simulated local field and reconstruction results. GitHub code is MIT; OSF metadata has no assigned dataset licence. | Useful additional susceptibility/T2 reference, but **no R1 or M0 maps** in either inventory. Generating everything still depends on the original package. The README's broad claim that maps are available is insufficient by itself. |
| [Jochmann/Schweser digital brain models](https://zenodo.org/records/14969113) | Public CC BY 4.0 archive, 1,115,055,633 bytes. Inspected ZIP directory and README with small HTTP range requests. Chi, segmentation, multi-echo magnitude/phase, field and microstructure maps. | Author-published alternative source of challenge-derived data, with/without microstructure. **No R1/R2*/M0 source maps** in the archive. Potential image/field validation set. |
| [BigBrain-MR](https://zenodo.org/records/7432527) | Public CC BY-NC-SA 4.0, with separate 800/400/100 µm maps. Inspected record, `data_info.txt` and author usage example. T1, R2*, QSM, 20-region labels, TE=0 magnitude and other image/coil maps. | Strong anatomical simulation alternative, but TE=0 GRE magnitude is **not equilibrium M0**: it retains acquisition weighting. Noncommercial/share-alike licence also differs from AHEAD. |
| [Lüsebrink human phantom](https://www.nature.com/articles/s41597-021-00923-w) | Public multi-contrast in-vivo images, reconstructed QSM, structural images and processing outputs. | Valuable imaging reference, but T1/T2/PD-weighted images are not automatically quantitative T1/T2/PD maps. Less direct for this loader. |

No restricted-source login, account creation, agreement acceptance or author
contact was performed. This search found a practical alternative, not a verified
unrestricted mirror of the original complete package.
