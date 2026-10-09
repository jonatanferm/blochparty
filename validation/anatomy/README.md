# BrainWeb anatomy (M2)

`blochparty.anatomy` loads fuzzy tissue memberships, conservatively resamples them,
and assembles separate tissue spins. No simulator engine or existing phantom default
changes. Numerical checks, reference contrasts and measured throughput are in
[RESULTS.md](RESULTS.md).

QSM source loading and public challenge checks are documented separately in
[QSM.md](QSM.md).

## Obtain the data

From the repository root:

```sh
uv run python validation/anatomy/fetch_brainweb.py --legacy
uv run python validation/anatomy_brainweb.py
uv run python validation/anatomy_brainweb.py --benchmark-only
```

The benchmark needs an Apple GPU. The contrast script runs on CPU. Both write JSON
under the ignored `data/` directory. Unit tests use small synthetic fixtures and
never download data.

Access was checked on 2026-10-07: the [20-model download page](https://brainweb.bic.mni.mcgill.ca/brainweb/anatomic_normal_20.html)
offers public files without an account or acceptance step. Its download form asks
for citations and optional name/institution/email; the fetcher leaves these blank.
No explicit redistribution licence was found. Do not redistribute downloaded or
derived volumes; only manifests and code are committed here. Cite BrainWeb's
listed publications when using the data.

The default snapshot is subject 04, plus optional older SBD T1/T2/PD references.
The committed manifests pin SHA-256 of **decompressed bytes**: gzip headers can
vary between requests. A mismatch stops loading/fetching; it never refreshes a pin.
`--record --subject N --manifest PATH` deliberately records a new snapshot for review.
Valid subject IDs are 4, 5, 6, 18, 20, 38, and 41–54. Only subject 04 has been
downloaded and validated here. Check the source header geometry before using a
new subject snapshot; the fetcher currently uses the captured subject-04 geometry.

For manual download, choose raw short + gzip on each tissue's public form, place
the files directly inside your chosen root, and supply `subjectNN.json` following
the pinned manifest. `load_brainweb(N, root)` does no network access. All twelve
memberships are required, including background. The extra `t1w` entry is for
validation and is not needed by the loader.

## Verified format and coordinates

The fuzzy models have shape **(362, 434, 362)** at **0.5 mm**, in z/y/x order.
For subject 04 the centre of voxel (0,0,0) is (-90.25,-126.25,-72.25) mm in
x/y/z order. Positive array axes follow positive MINC physical axes; no transpose
or flip is applied. The simulated T1 image uses a different grid: (181,256,256),
1 mm, origin (-127.75,-145.75,-72.25) mm.

Captured raw-short memberships are **little-endian uint16, 0…4095**, despite the
legacy format page describing big-endian shorts. Byte counts, decoded range and
checksums are verified. Raw-byte downloads were offset-encoded signed bytes,
despite being described as unsigned; they must not be interpreted as ordinary
0…255 membership maps. The pinned loader uses raw shorts.

All classes are divided by `max(sum_of_classes, 4095)` before resampling: this
removes overfull totals while retaining missing mass as air. Normalizing every
positive total to one would turn tiny background memberships into tissue.
Resampling uses cell-overlap integrals, preserves physical tissue volumes, holds
the lower outside edge fixed and pads the upper edge with air when necessary.

## Use

```python
import blochparty as bp

head = bp.load_brainweb(4, "validation/anatomy/data", voxel=(.002,)*3)
# Restrict spins to a slab, but retain the full head as the B0 source.
phantom = bp.to_phantom(
    head, field_T=3., fov_region=(slice(40, 44), slice(None), slice(None)),
    spins_per_voxel=8, seed=0,
)
```

`spins_per_voxel` is a perfect cube or `(kz,ky,kx)`. Default `sampling="per_tissue"`
uses K spins for every nonzero tissue in each voxel, retaining each tissue's own
T1/T2. PD weights conserve fraction × tissue PD exactly. `sampling="categorical"`
draws one tissue per subcell, including the chance of air: cheaper, unbiased, but
individual tissue volumes fluctuate. Fat is split across six chemical peaks with
normalized weights. Spatial jitter is stratified; `jitter=None` uses subcell
centres. Seeds do not change material constants, although categorical identities
and B0 sampled at new positions can change.

PD is normalized per anatomy voxel, like `phantom_from_maps`. Multiply signals
by voxel volume to compare absolute signal at different anatomy resolutions.
`crop` takes z/y/x slices; `slab(z0, thickness)` takes metres and snaps outward to
cell boundaries. Cropping the anatomy itself discards susceptibility sources.
Use `fov_region` on the full head for a thin-slab simulation with full-head B0.

If a measured `maps["M0"]` is supplied, it represents total voxel water density;
assembly apportions that density among signal-producing tissue memberships.
When resampling measured rates with M0, use M0-weighted averages to avoid diluting
rates with empty air. Such averaging is still a compartment approximation.

Fractions specify cell-average densities, not an unknown boundary's position
inside that cell. Decouple the anatomy grid from reconstruction and refine anatomy
to resolve boundaries; jitter alone cannot recover information absent from the map.

## Tissue priors and T2*

`DEFAULT_TISSUES[field_T][name]` contains `TissueProperties`, whose parameters are
`Distribution(mean, sd, citation, assumed)`. Unknown SD is `None`, not zero. SDs are
metadata; spatial variability is M4 and is explicitly rejected for now.

- 1.5/3 T T1/T2: Stanisz 2005 specimen measurements, marked as animal/ex-vivo proxies
  (blood is human). CSF, fat and PD include legacy BrainWeb priors; skin, connective
  tissue and dura use explicit muscle proxies. These are not calibrated human defaults.
- 3 T WM/GM T2*: whole-brain Shin 2025 measurements (FreeSurfer GM/WM, 4 subjects, with SDs;
  their protocol is `DEFAULT_R2STAR_ACQUISITION`); 7 T WM/GM T1/T2*: Caan 2019,
  with measured SDs. Other 7 T classes are absent, and 7 T T2 is explicitly a 3 T
  proxy. Missing values raise; there is no silent field interpolation.
- Chi uses absolute SI: air +0.36 ppm, water-like soft tissue −9.05 ppm, bone a
  −2 ppm contrast relative to water. There is no fitted iron or myelin model.
- MT and isotropic diffusion (opt-in, `to_phantom(mt=True, diffusion=True)`) are cited for every
  signalling BrainWeb class since 2026-10-09 (M5). Bound fraction f = M0b/(M0b+M0f), of the
  total pool; assembly converts it to the engine's bound/free ratio. Bound T1 equals free T1;
  the rate model and a super-Lorentzian line are used.

  | class | f | exchange (bound → free) | T2b | diffusivity [m²/s] |
  |---|---|---|---|---|
  | WM | 0.1277 ± 0.0041 (Yarnykh 2016) | 19.0 /s (Yarnykh 2012, R) | 9.7 µs (Yarnykh 2012) | 0.70 ± 0.03e-9 (Helenius 2002) |
  | GM | 0.0673 ± 0.0015 (Yarnykh 2016) | 19.0 /s | 9.7 µs | 0.89 ± 0.04e-9 (cortex, Helenius 2002) |
  | CSF | 0 (MTR 0.01, Smith 2005) | | | 3.04e-9 (water at 37 °C, Holz 2000) |
  | muscle | 0.084 ± 0.006 (Sinclair 2010) | 19 /s, assumed (did not converge in muscle) | 5.9 ± 0.2 µs (Sinclair 2010) | 1.78 ± 0.08e-9 (Li 2014) |
  | fat | 0 (no exchange with triglyceride, Chen 2010) | | | 1.83e-11, an upper bound (Weidlich 2018) |
  | vessels | 0, assumed | | | 1.54 ± 0.12e-9 (blood, Funck 2018) |
  | skin, connective, dura | muscle, flagged proxy | | | muscle, flagged |
  | marrow | fat, flagged proxy | | | fat, flagged |

  The qMT values are 3 T measurements; at 1.5 T and 7 T they are flagged as carried over
  (Stanisz 2005 found MT parameters at 1.5 T and 3 T similar). Diffusivity does not depend on B0.
  Exchange-rate conventions differ between papers; see docs/LESSONS.md.

CSF and subcutaneous fat use measured values since 2026-10-07 (CSF T1 4.4 s, Rooney 2007;
3 T CSF T2 2.03 s, Spijkerman 2018; 3 T fat T1/T2 382/68 ms, de Bazelaire 2004; 1.5 T fat from the
IT'IS database), replacing the legacy simulator priors (CSF T2 0.33 s) that made T2-weighted CSF
dark. The 1.5 T CSF T2 is the 3 T value, flagged.

`BRAINWEB_TISSUES` separately preserves the older simulator's 1.5 T relaxation
priors for reference comparisons. Existing `phantom.TISSUES` is untouched.

A measured T2* (map or table) contains R2, microscopic R2' (vessels, myelin, iron) and the
dephasing that the voxel-scale field causes, which the simulation computes itself from chi. The
default `t2star="auto"` (= `"separate"` with chi on) estimates that last part from the simulated
field's gradient at the voxel size and echo times of the R2* measurement
(`estimate_macroscopic_r2prime`; `r2star_acquisition`, default 1 mm and 5-40 ms, an assumption for
table values; `AHEAD_R2STAR_ACQUISITION` for AHEAD) and puts only R2' = max(R2* - R2 - macro, 0) on
the spins. Without chi, R2' = R2* - R2. `t2star="table"` with chi still accepts a supplied
`macroscopic_r2prime`, and `maps["intrinsic_r2prime"]` an already separated rate.
`t2star="off"` disables that broadening. Cauchy quantiles need many spins to resolve
an FID; K=1 has only the zero median and emits a warning for finite requested T2′.

`anatomy_field_maps(head)` returns `(susceptibility, scanner_shim)` as separate
`FieldMaps`. The ordinary `to_phantom` result combines them for stationary anatomy.
For motion, sample only the first into tissue `db0` and pass the second as scanner
maps; preserve any chemical/intrinsic offsets. Translation carries the tissue
field; rotation relative to B0 requires recomputing it. A thin planar mask cannot
fit all 3D second-order harmonics: compute the full head first or explicitly choose
a lower shim order.

## Spatial variability (M4)

`to_phantom(..., variability={"t1": .05, "t2": .05, "pd": .02, "length": .01, "seed": 0})`
multiplies each tissue's T1, T2 and PD by its own mean-1 log-normal field (relative SD as
given, Gaussian correlation exp(-r^2 / 2L^2)), drawn on a coarse grid (L / 4) over the whole
anatomy and interpolated to voxel centres, so crops and slabs share one realisation; its seed
is independent of the spin-position seed. It is texture within a tissue, not the population
spread recorded in `Distribution.sd`.

## AHEAD 7 T templates

`load_ahead(root)` reads the four pinned AHEAD v3 maps fetched by `fetch_ahead.py`: one
`"brain"` class (R1 > 0 and PD > 0), measured `M0` (relative PD), `R1`, `R2star`, and relative
QSM as `chi` (no air or bone: `chi_background = 0`), `map_field_T = 7`. There is no R2 map:
supply a 7 T table entry `"brain"` with T2, and, because the default T2* policy omits R2'
when susceptibility is on, supply `maps["intrinsic_r2prime"]` (e.g. R2* - R2) to keep the
measured R2* contrast (`validation/anatomy_ahead.py` does both, as stated assumptions).
