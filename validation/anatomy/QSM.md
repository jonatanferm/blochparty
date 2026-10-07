# QSM Challenge 2.0 — M3

The loader and public challenge forward checks are implemented. Validation with
the **full-head R1/R2*/M0 source package and original MATLAB simulator remains
pending**: the public challenge release does not contain those files.

An accessible alternative was subsequently verified: AHEAD v3 supplies aligned
7 T brain susceptibility, R1, R2* and relative-PD maps under CC BY 4.0.
See [alternative sources](ALTERNATIVE_SOURCES.md) for actual-file checks,
reproducible downloads and the limits of substituting it for the full-head model.

## Two distinct releases

The [Radboud collection](https://data.ru.nl/collections/di/dccn/DSC_3015069.02_542)
(DOI 10.34973/m20r-jt17, version 1) advertises restricted access and the
RU-DI-HD-1.0 data-use agreement in its public metadata, checked 2026-10-07.
The spec's earlier CC BY assumption was incorrect. No login or agreement
acceptance is automated here. This is the source of the complete head maps.

[Berkin Bilgic's lab software page](https://www.martinos.org/~berkin/software.html)
separately links a public [challenge data/code share](https://surfdrive.surf.nl/index.php/s/uTvrxJ5NELbnGa5).
It provides Sim1/Sim2 susceptibility ground truths and segmentation, plus
challenge images, estimated frequency maps and evaluation scripts. It does not
provide R1/R2*/M0 or the original phantom-generation package. Access to this
share requires no account or form submission; an explicit redistribution licence
was not found. Commit only code and checksum manifests, not downloaded data.

The user-supplied `2019-qsm-challenge-stage1-download` short link resolves to a
closed registration form, which refers participants to their Stage 2 email and
contains no replacement download URL.

## Reproduce the public comparison

```sh
uv sync --extra anatomy
uv run python validation/anatomy/fetch_qsm_challenge.py --model 1
uv run python validation/anatomy/fetch_qsm_challenge.py --model 2
uv run python validation/anatomy_qsm.py --fetch-reference
uv run python validation/anatomy_qsm.py --root validation/anatomy/data/qsm_challenge1 --resolution native
uv run python validation/anatomy_qsm.py --root validation/anatomy/data/qsm_challenge2 --resolution native
```

`qsm_challenge1.json` and `qsm_challenge2.json` pin decompressed NIfTI bytes
captured on 2026-10-07. The independent reference is
[qsm-forward](https://github.com/astewartau/qsm-forward/tree/92149383b0f43f47d9a351ceb4f525c0c4ee8299),
pinned by commit and SHA-256 in `qsm_reference.json`. This public Python project
is based on Marques 2021; it is **not** the original restricted MATLAB package.
The script loads only its two forward functions, unchanged, from verified local
source. Reference code stays in ignored data storage and is not vendored here.

Both calculations use the same susceptibility array, 2× padding, isotropic voxels,
z-directed B0 and 3 T. The reference fills its padding from the last array corner;
we match that exact encoded value after checking it agrees with the declared
exterior. Tiny nonzero corner values are retained, not rounded away.

The reference sets its zero-frequency kernel coefficient to 1/3; ours sets it to
zero. Consequently their fields differ by a constant. Subtract only the mean
difference over the brain ROI to compare relative fields; **no slope, spatial
fit or shim is fitted**. The reference's transverse anisotropic/oblique axis
conventions are not validated by this comparison. Those directions remain covered
by M1's analytic/rotation tests for our implementation.

| Input | Brain voxels | Subtracted difference (Hz) | RMS (Hz) | Maximum error (Hz) |
|---|---:|---:|---:|---:|
| Synthetic 17×19×21, air 9.2 ppm | 2,145 | −376.224204 | 1.77e-13 | 6.82e-13 |
| Synthetic 18×20×22, air 9.2 ppm | 2,688 | −375.091881 | 1.69e-13 | 5.12e-13 |
| Public Sim1 ground truth | 1,469,325 | −4.19232e-5 | 2.11e-16 | 4.47e-15 |
| Public Sim2 ground truth | 1,469,325 | −1.15128e-5 | 3.76e-16 | 1.38e-14 |

The downloaded GT grid is (164,205,205) in source x/y/z, with header spacing
**0.999024391 mm**, despite being described as nominally 1 mm. We preserve that
header, converting to z/y/x internally. Chi ranges approximately −0.0434 to
0.2168 ppm for Sim1. Brain ROI is segmentation labels 1…11 plus 16. The public
ground truth suppresses background susceptibility, so its exterior is **zero**,
not the full-head model's air susceptibility. Keep those source types separate.

Agreement here validates a forward calculation on actual challenge anatomy;
it does not establish scanner accuracy or reproduce the challenge's full image
simulation. Its supplied frequency map is estimated from noisy, downsampled
multi-echo images rather than a direct convolution of the distributed GT. Its
header also uses exactly 1 mm instead of the GT's spacing. It is deliberately
not used as an exact forward-convolution reference.
[Challenge design paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12142683/).

## Load the public ground truth

```python
import blochparty as bp

brain = bp.load_qsm2("validation/anatomy/data/qsm_challenge1", resolution="native")
tissue_field, scanner_shim = bp.anatomy_field_maps(brain, field_T=3.)
```

`kind="challenge"` manifests require chi and segmentation only. The resulting
`Anatomy` keeps named memberships for all present labels and a derived brain mask.
It has **no measured relaxation or M0 maps**. A full Bloch phantom therefore
needs an explicit tissue table, including deep-grey labels and calcification;
no substitute tissue values are silently invented.

## Prepare a complete authorized source package

After obtaining the full-head package under its access terms, pass its extracted
`data` directory containing `chimodel/`, `maps/`, and `masks/`:

```sh
uv run python validation/anatomy/prepare_qsm2.py /path/to/data \
    --chi chimodel/ChiModelMIX.nii --background-ppm 9.2 --field-T 7
```

Confirm the susceptibility exterior/reference in that selected model before using
the example's 9.2 ppm. A different model can be selected with `--chi`; use a
separate root/manifest rather than overwriting a reviewed pin. Preparation records
hashes of local files; it does not certify their authenticity as publisher data.
If format validation fails, inspect the reported error and the recorded manifest.

Required keys are `chi`, `M0`, `R1`, `R2star`, `segmentation`, `brain_mask`.
The manifest explicitly declares spatial units, chi units (ppm), background and
acquisition field. Loading checks hashes, grid alignment, valid labels/rates and
finite data. NIfTI permutations/flips are reoriented to positive RAS axes; oblique
or sheared grids raise and require a deliberate common-grid preprocessing step.
Unknown header units use the manifest's explicit unit; contradictory units raise.

`resolution="native"` keeps the acquired grid. `"0.64mm"`, `"1mm"` or a spacing
tuple in metres requests conservative cell-overlap averaging, **not** the
challenge generator's k-space cropping. M0 is total voxel water density. Its
integral is conserved; assembly distributes it among signal-producing tissue
memberships, avoiding a second partial-volume attenuation. Relaxation maps are
averaged with M0 weights so air does not dilute rates. Coarsening heterogeneous
rates is still an approximation to a multi-exponential signal; use native maps
and sub-voxel spins for convergence studies.

Full-source R1/R2* maps were acquired at 7 T. `map_field_T` prevents their silent
use at another field; susceptibility-only calculations may use any field.
The source has no measured R2: choose explicit T2 priors for Bloch assembly.
R2* stays separate, with M2's calibration guard before combining observed decay
and macroscopic susceptibility. Background reference and map acquisition field
survive crop/slab/resampling.

## Local checks and unfinished validation

Thirteen synthetic NIfTI tests cover affine permutations/flips, mm→m and ppm→SI,
checksum/path/grid failures, unknown labels, invalid rates/units, background
reference invariance, conservative resampling, source-field guards, and M0 mass.
The public fetcher and loader have also been exercised against both real GT files.
Full suite with host Apple GPU access: **462 passed, 4 opt-in crosschecks deselected**
(184.47 s). Existing default output/gradient snapshots remain bit-identical.

Still needed to close the original M3 specification: an authorized full-head source
package to verify R1/R2*/M0 ingestion on its actual files, and its original MATLAB
forward routine for that final cross-check. The public Python comparison above is
an additional independent check, not a claim to have executed unavailable code.
