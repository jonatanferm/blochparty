# Spec: realistic anatomy and susceptibility-driven B0

Status: M1, M2, M4, M5 complete; M3: QSM loader and public-data checks implemented, the
restricted full-head source not available, AHEAD 7 T maps loaded as the open alternative (2026-10-07).
M2 results and remaining clinical-validation limits: [RESULTS](../../validation/anatomy/RESULTS.md).
M3 progress/access requirements: [QSM notes](../../validation/anatomy/QSM.md).
Verified alternative for brain parameter maps: [AHEAD v3 and source comparison](../../validation/anatomy/ALTERNATIVE_SOURCES.md).
Coordination rules at the end.
Read first: `CLAUDE.md` (working rules), `docs/REALISM.md` (rows "Realistic anatomy and fuzzy
tissue", "Continuous anatomy (inverse crime)", "B0 inhomogeneity from anatomy"),
`src/blochparty/phantom.py` (`Phantom`, `phantom_from_maps`, `TISSUES`, `b0_sphere`,
`b0_cylinder`, `b0_shim`), `src/blochparty/maps.py` (`FieldMaps`), `docs/LESSONS.md`.

## 1. Why

Before this work, the project phantoms were geometric (Shepp-Logan ellipses, NIST spheres,
tubes) and rasterised on the grid they were reconstructed on. Real images differ in three ways this item closes:

1. **Anatomy**: brain geometry with partial volume between tissues (fuzzy membership), not
   piecewise-constant labels.
2. **Tissue-property realism**: literature values per tissue *and field strength*, with
   within-tissue variability, instead of one constant per label.
3. **B0 from anatomy**: the air–tissue susceptibility difference (~9.4 ppm) gives tens to
   hundreds of Hz near sinuses and ear canals at 3 T (EPI distortion, dropout, T2*). The
   simulator already takes any B0 map (`Phantom.db0`, `FieldMaps.b0`); what is missing is the
   map from a susceptibility distribution.

## 2. Scope

In scope (in this order; each milestone is a separate, tested, committed step):

- **M1 `susceptibility.py`**: chi map -> B0 field map by the Fourier dipole kernel, plus
  low-order shimming. Pure function of arrays; no data needed.
- **M2 `anatomy.py`, BrainWeb**: loader for the BrainWeb 20 normal anatomical models (fuzzy
  tissue volumes), a tissue-property table, and assembly into a `Phantom` with sub-voxel spins.
- **M3 QSM Challenge 2.0 head phantom**: loader for its chi, R1, R2*, M0 maps and segmentation,
  as a second anatomy source and the susceptibility input for M1.
- **M4 within-tissue variability**: smooth, seeded random fields modulating properties.
- **M5 example and docs**: `examples/anatomy_brain.py` (a 2D slice and a 3D GPU run),
  REALISM rows, LESSONS, JOURNAL.

M1 API: `susceptibility_to_b0(chi, voxel, field_T=3.0, *, b0_dir=(0, 0, 1),
padding=2.0, background=0.0)` and `shim(df, mask, order=2, *, voxel=(1, 1, 1), origin=None)`.
Both are exported from `blochparty`. `voxel` is (dz, dy, dx) in m; `origin` is
(x, y, z) in m, defaulting to a grid centred on zero. Shim coefficients use the exact
`b0_shim` names/units, describe the subtracted field, and require a full-rank mask.
Tests report sphere inside/outside RMS 0.439/0.108 Hz at 64³, cylinder RMS < 1.77 Hz
at all three orientations, and harmonic removal < 1e-12 Hz (3 T, dchi=9.4e-6).

Out of scope: body / cardiac anatomy (XCAT and MIDA are licence-restricted), anatomical motion
(the existing `motion.py` applies to any phantom), EM coil maps (separate item), perfusion /
contrast agents, J-coupling, new kernel physics. **No changes to the engines should be needed**:
everything maps onto existing `Phantom` fields. If one seems needed, stop and write it down.

## 3. Data sources (verify access and licence before writing a fetcher)

| Source | What | Access / licence (to verify) | Use |
|---|---|---|---|
| BrainWeb, 20 normal anatomical models (Aubert-Broche et al., NeuroImage 2006; brainweb.bic.mni.mcgill.ca) | fuzzy membership volumes per tissue class (background, CSF, GM, WM, fat, muscle, muscle/skin, skull, vessels, around-fat, dura, bone marrow), 0.5 mm isotropic, (362,434,362) z/y/x; T1 reference is a separate 1-mm grid | public form verified 2026-10-07; optional identity fields, citation request, no explicit redistribution licence found; do not redistribute | M2 primary anatomy |
| QSM Challenge 2.0 head phantom (Marques et al., MRM 2021, doi:10.1002/mrm.28716; data doi:10.34973/m20r-jt17) | chi (two models), R1, R2*, M0, 16-label segmentation, 0.64 mm and 1 mm, plus simulation code | Radboud metadata verified 2026-10-07: restrictedAccess, RU-DI-HD-1.0 (not CC BY); obtain an authorized local copy. Author-linked public challenge share separately provides chi + segmentation only | M3 full-head loader; public Sim1/Sim2 forward checks completed, complete source/MATLAB validation pending |
| MIDA, XCAT, Virtual Population | whole-head / body models | licences required | not used |

Rules: data goes under a git-ignored directory (e.g. `validation/anatomy/data/` or a user cache);
fetch scripts verify checksums and pin versions (see `validation/real/fetch_scholand.py` for the
pattern); never commit data or derived volumes. If a source needs an account or click-through,
the fetcher must explain how to obtain the file and then read it from a path, not automate the
login. The user has given standing permission to download from reputable sources.

## 4. M1: susceptibility -> B0 (`src/blochparty/susceptibility.py`)

Physics (SI chi, dimensionless; field in Hz at the Larmor frequency):

    dB_z(k) / B0 = (1/3 - (k . b)^2 / |k|^2) chi(k),     b = unit vector of B0 in the object frame
    df(r) = GAMMA_BAR * B0 * dB_z(r) / B0                [Hz]

- The 1/3 term is the Lorentz-sphere correction (the field a spin inside the material feels);
  this must agree with the existing closed forms `phantom.b0_sphere` / `b0_cylinder` (zero
  inside a sphere; inside an infinite cylinder at angle theta to B0, dchi/6 (3 cos^2 theta - 1)).
- k = 0: set the kernel to 0 (only relative fields are defined); document it.
- Zero-padding (default 2x per axis, configurable) against wrap-around; voxel size and B0
  direction (`b0_dir`, default (0, 0, 1) in (x, y, z)) as parameters; arrays in the project's
  (z, y, x) order.
- Subtract an explicit surrounding susceptibility (`background`, SI, default zero) before
  zero-padding: absolute head maps need air outside, not a vacuum boundary. k=0 sets the
  mean over the padded domain, not the returned crop. Even-grid Nyquist cross terms are
  averaged over both signs of each Nyquist axis to preserve joint rotation covariance.
- Optional anti-aliasing: the kernel sampled on a finer grid then averaged, or the
  "continuous kernel" variant; measure which one the analytic tests need.
  M1 measurements meet the above accuracy with partial-volume input and the standard
  kernel; an extra anti-alias kernel is deferred unless data validation shows a need.
- `shim(df, mask, order=2)`: least-squares fit of the real spherical harmonics up to
  `order` (reuse the term definitions of `phantom.b0_shim` for 0th-2nd order) over a mask,
  returning the shimmed map and the coefficients. Scanners shim to 2nd order.
- Output: a Hz map usable as `Phantom.db0` (tissue-attached) or `FieldMaps.b0` (scanner frame).
  Explain in the docstring when to use which (moving subjects: the susceptibility field moves
  with the tissue, the shim field stays with the scanner; `maps.py` already separates them).
  This tissue-attached approximation is exact for rigid translation of the source;
  rotation relative to B0 changes the induced field and requires recomputation using the
  new object-frame `b0_dir`. Do not attach the scanner shim to the tissue.
- Typical values (cite primary sources in code): water -9.05 ppm, air +0.36 ppm, so air-tissue
  ~ +9.4 ppm; cortical bone ~ -2 ppm relative to water; deep grey matter iron +0.05-0.2 ppm.

Required checks (`tests/phantoms/test_susceptibility.py`):

1. Sphere in a large box: inside and outside against `b0_sphere`, error falls with padding and
   resolution; report the converged numbers in the test docstring.
2. Infinite cylinder (long finite cylinder, interior far from the ends) at 0, 54.7 and 90 deg:
   inside value dchi/6 (3 cos^2 - 1) and the outside 2D dipole, against `b0_cylinder`.
3. Linearity and the B0-direction rotation (rotating the object and `b0_dir` together gives the
   rotated field).
4. Shim: a field made of the spherical harmonics up to order 2 is removed to machine precision;
   a sphere's field is reduced and the residual is orthogonal to the basis on the mask.
5. M3 cross-check: our field from the QSM 2.0 chi map against the field their published code
   produces from the same map (same padding / orientation conventions; report the difference).

## 5. M2: anatomy and tissue properties (`src/blochparty/anatomy/`, a single `anatomy.py` until 2026-10-17)

### 5.1 Data model

```python
@dataclass
class Anatomy:
    fractions: dict[str, Tensor]      # tissue -> (nz, ny, nx) membership in [0, 1]
    voxel: tuple[float, float, float] # (dz, dy, dx) [m]
    origin: tuple[float, float, float]# position of voxel (0, 0, 0) centre, (x, y, z) [m]
    chi: Tensor | None = None         # susceptibility in SI, same grid; reference explicit below
    maps: dict[str, Tensor] = field(default_factory=dict)  # optional measured maps (R1, R2*, M0, ...)
    chi_background: float = AIR_CHI   # exterior in the same SI reference as chi
    map_field_T: float | None = None  # acquisition field of measured relaxation maps
```

Fractions need not sum to 1 (background); document how the remainder is treated.
Convert ppm source data by 1e-6 on loading, keeping the Anatomy and M1 units identical. Loaders:
`load_brainweb(subject: int, root, *, voxel=None, dtype=torch.float32)`;
`load_qsm2(root, resolution="1mm", *, dtype=torch.float32)` reads a reviewed local manifest.
M3 supports axis-aligned NIfTI permutations/flips; oblique/sheared grids raise. The optional
`anatomy` extra supplies nibabel. Public challenge manifests contain chi/segmentation only;
full-head manifests also require R1/R2star/M0/brain_mask. Unknown maps are not invented. Utilities:
`crop`, `slab(z0, thickness)`, `resample(voxel)` (fraction-preserving, i.e. area/volume
weighted, not nearest neighbour), `reorient` if a source uses another axis convention (state the
conventions explicitly in the docstring; ours are (z, y, x) arrays, positions in (x, y, z) m).

### 5.2 Tissue table

A data structure (and a default instance) giving, per tissue and field strength (1.5 T, 3 T,
7 T where sources exist): PD (relative water content), T1, T2, T2* (or T2'), chi, and optionally
MT (`BoundPool` f, k, T2b), diffusivity (mean, and FA for white matter if tensors are used),
fat fraction. Each entry is a distribution (mean, SD across subjects or within tissue), not just a
mean (`sd=None` explicitly records an unreported SD), and **every number carries a citation** (author, year, table) in the code next to it; where
a value is assumed, say so. Starting points to verify, not to copy blindly:

- T1/T2 at 1.5 and 3 T: Stanisz et al., MRM 2005 (many tissues, both fields, MT parameters too);
  Wansapura et al., JMRI 1999 (brain, 3 T); Bojorquez et al., MRI 2017 (T1 meta-analysis).
- T2* at 3 T and 7 T: Peters et al., MRI 2007; Wharton & Bowtell (WM orientation dependence;
  record but do not model).
- MT in brain: Gloor et al., MRM 2008 (bSSFP qMT: WM bound fraction ~0.16, GM ~0.07); Stanisz 2005.
- Diffusion: MD of WM ~0.7e-9, GM ~0.8e-9, CSF ~3.0e-9 m^2/s at body temperature (cite).
- Susceptibility: QSM consensus / QSM 2.0 paper values for deep GM, WM, CSF reference.
- Keep the existing `phantom.TISSUES` untouched (Shepp-Logan relies on it); the new table is a
  separate, richer object.

### 5.3 Assembly into a `Phantom`

`to_phantom(anatomy, table, field_T=3.0, *, fov_region=None, spins_per_voxel=..., jitter="stratified",
seed=0, mt=False, diffusion=False, chi_to_b0=True, shim_order=2, variability=None) -> Phantom`

- **Partial volume, no inverse crime**: a voxel with several tissues contributes spins of each
  tissue, weighted by its fraction (each spin keeps its tissue's T1/T2/...; PD weights carry the
  fractions). Positions are jittered sub-voxel points (`jitter="stratified"`), so the phantom is
  not on the reconstruction grid. Offer both "spins per tissue per voxel" and "sample a tissue
  per spin with probability = fraction" and document the bias/variance trade-off.
- T2*: from the table's T2* (or the QSM 2.0 R2* map) as T2' = 1 / (1/T2* - 1/T2), applied with
  the existing `t2prime` mechanism of `phantom_from_maps` (quantile Cauchy offsets, many spins).
  The susceptibility field from M1 is the *macroscopic* part and must not be double counted:
  observed T2* may include intravoxel macroscopic gradients, depending on the measurement
  and correction method. M2 defaults to `t2star="auto"`: use table broadening only without
  susceptibility. With susceptibility, omit uncalibrated R2′ (a lower-bound model, not full
  T2*) or supply `maps["intrinsic_r2prime"]`. Explicit `t2star="table"` plus susceptibility
  requires `macroscopic_r2prime` calibration in 1/s; no universal subtraction is assumed.
- B0: `chi -> df` via M1 on the anatomy grid (air outside the head must be in the chi map; the
  BrainWeb background is air), then shimmed (2nd order over the brain mask by default), sampled
  at each spin's position (trilinear, as `maps.sample`). Return it in `Phantom.db0` and also
  offer a `FieldMaps` builder for moving-subject use.
- MT / diffusion: optional, from the table (`BoundPool` per spin with `model="rate"` by default;
  isotropic diffusivity per spin; tensors only if a source provides orientations).
- Memory and speed: a full 3D BrainWeb head at 1 mm is ~3-6 M voxels inside the head; with
  partial-volume spins and K sub-voxel spins this reaches tens of millions. Provide `fov_region`
  / `slab` so 2D slices and thin slabs are cheap, measure a full 3D run on the GPU
  (`blochparty.metal.simulate_metal`) and report spins/s and memory in the docstring.

### 5.4 Required checks (`tests/phantoms/test_anatomy.py`; use a small synthetic anatomy in unit tests,
real data only in an opt-in marked test or a validation script)

1. Conservation: the sum of spin PD per tissue equals sum(fraction x PD) over voxels to 1e-12;
   resampling preserves tissue volumes to 1e-6 relative.
2. Signal equation: a spoiled-GRE steady state of a two-tissue partial-volume voxel equals the
   fraction-weighted sum of the closed-form signals (`reference.py` or the Ernst equation),
   per voxel, with ideal spoiling and one spin per tissue.
3. No inverse crime: a sharp tissue boundary at a sub-voxel offset gives the k-space of the
   continuous object (analytic box / sphere FT) to within the spin-sampling error, measured as a
   function of spins per voxel; contrast with the on-grid phantom. A membership volume
   specifies cell-average density, not an unknown subcell boundary: resolve boundaries on
   an independent, finer anatomy grid. Jitter alone cannot recover missing geometry.
4. B0 path: a chi map with a single sphere in the anatomy reproduces `b0_sphere` at spin positions.
5. Determinism: same seed -> identical phantom; different seeds -> different spatial jitter,
   with fixed material properties. Categorical tissue draws and B0 sampled at changed
   positions also change; quantile broadening uses its own fixed random stream.
6. GPU = core on a small anatomical phantom (relative error ~1e-5, as `tests/engines/test_gpu.py`).

Validation script (`validation/anatomy_brainweb.py`, [RESULTS](../../validation/anatomy/RESULTS.md)):
compare tissue contrast ratios (GM/WM, CSF/GM) against downloaded references. The newer
20-model set publishes only SFLASH T1 (22/9.2 ms, 30°); the older SBD publishes SFLASH T1
(18/10 ms, 30°) and DSE T2/PD (TR 3300 ms, echoes 35/120 ms). These are separate comparisons.
Older GM/WM differences are <0.13%, CSF/GM <4.74%; newer subject-04 T1 differs by −5.44% and
−21.31%, respectively, using the older tissue priors. Do not claim those priors are the exact
newer simulator parameters. This is cross-simulator plausibility, not scanner ground truth.
M2 propagates published human 3 T relaxation values through these protocols as a sensitivity
check; a comparison to observed, protocol-matched in-vivo ratios remains unvalidated.

M2 delivered API and limits:

- `Anatomy`, `Distribution`, `TissueProperties`, `DEFAULT_TISSUES`, `BRAINWEB_TISSUES`,
  `load_brainweb`, `to_phantom`, `anatomy_field_maps` are exported. See the
  [usage and data-format notes](../../validation/anatomy/README.md).
- Pinned subject-04 raw-short data: little-endian 0…4095, verified from delivered files,
  despite the website's legacy endian description. Checksum is over decompressed bytes.
  Normalize only overfull class totals; deficits stay air. Data is git-ignored.
- Separate per-tissue spins conserve PD to 1e-12; categorical sampling conserves in
  expectation. Resampling conserves physical volume to 1e-6. The box Fourier error falls
  0.1771 → 1.44e-5 over 1→64 x samples/cell; sphere B0 RMS at spin positions is 0.464 Hz;
  GPU/core relative signal error with B0, MT and diffusion is 3.23e-7.
- 1.5/3 T priors include flagged ex-vivo and other-tissue proxies. Verified 7 T T1/T2*
  entries cover WM/GM only, with an explicit 3 T T2 proxy; missing classes raise. Optional
  MT/diffusion parameters are illustrative assumptions, not calibrated measurements.
  Unreported SDs remain unknown; M4 spatial sampling is not implemented.
- Full 1-mm head: 5.33 million spins, RF/FID GPU 0.686 s (7.77 million spins/s), phantom
  149 MB, peak process RSS 3.64 GB, end-of-run MPS driver allocation 1.08 GB. Benchmark
  excludes B0/T2′/MT/diffusion; it does not establish full imaging speed or convergence.
- Full 2-mm head B0 at 3 T: brain RMS 45.30 → 19.17 Hz after second-order shim with
  scalar tissue/air/bone priors. A measured field comparison is still pending M3.
- 449 tests pass with Apple GPU access; 4 opt-in crosschecks deselected. Existing output
  and gradient snapshots are bit-identical. No engine, `phantom.py` or `maps.py` edits.

## M3 implementation status (2026-10-07)

- `load_qsm2` preserves source geometry, 16-label memberships, susceptibility reference and
  measured-map acquisition field. Conservative resampling preserves M0 volume; M0-weighted
  rates do not dilute relaxation with air. Coarsened rates remain a compartment approximation.
- Public Sim1/Sim2 data was obtained from the SURFdrive share linked by an author's lab;
  decompressed checksums are pinned. Source header spacing is 0.999024391 mm, distinct from
  the exactly 1-mm estimated frequency images. The full-head package remains access-restricted.
- Both public chi maps agree with pinned independent `qsm-forward` code: brain RMS
  2.11e-16 / 3.76e-16 Hz after subtracting only a constant reference difference, 3 T, 2× padding.
  That code is a public Python implementation based on the paper, not the unavailable original
  MATLAB package. Synthetic air/exterior cases agree to <6.83e-13 Hz maximum.
- Full-source R1/R2*/M0 ingestion is unit-tested with synthetic NIfTI files, but not yet tested
  on the restricted original files. The original MATLAB cross-check also remains pending;
  M3 is therefore not marked complete. Reproduction and access details are in
  [QSM.md](../../validation/anatomy/QSM.md).

## 6. M4: within-tissue variability

`variability = {"t1": 0.05, "t2": 0.05, "pd": 0.02, "length": 0.01}`: multiplicative log-normal
fields with the given relative SD and correlation length [m], drawn as smooth Gaussian random
fields (FFT filtering of white noise, seeded, `torch.Generator`), per tissue. Check: the
realised SD and correlation length within 10 % of the request on a large grid; default off
(bit-identical outputs without it).

## 7. Project rules that apply (from CLAUDE.md)

- Nothing is done without an independent check (above). Report numbers, not "agrees".
- Defaults bit-identical: everything is new and opt-in. Before and after any change to an
  existing module run `uv run python tools/snapshot.py save <file>` and `compare`.
- New physics goes into the core and Metal; this item should need none (stop and ask if it does).
- Record lessons in `docs/LESSONS.md`, append a dated entry to `docs/JOURNAL.md` per milestone,
  update the three REALISM rows and `ROADMAP.md`.
- Conventions: Pulseq units; arrays (z, y, x); positions (x, y, z) in m; `GAMMA_BAR` from
  `constants.py`; float64 in references, float32 allowed on the GPU.
- `uv run pytest` must pass (GPU tests skip without MPS); add tests, never loosen
  existing ones.

## 8. Coordination (two agents on one repo)

- **You own**: `src/blochparty/susceptibility.py`, `src/blochparty/anatomy/`,
  `tests/phantoms/test_susceptibility.py`, `tests/phantoms/test_anatomy.py`, `examples/anatomy_*.py`,
  `validation/anatomy*`, `docs/specs/ANATOMY.md` (update its status line), and exports in
  `src/blochparty/__init__.py`.
- **Shared, edit with small diffs**: `docs/REALISM.md`, `docs/LESSONS.md`, `docs/JOURNAL.md`,
  `ROADMAP.md`, `pyproject.toml` (new optional extras only, e.g. `nibabel` for NIfTI; keep the
  core dependency-free of them).
- **Do not edit**: `validation/real/` (another agent is working there), the engines
  (`bloch.py`, `metal/`), `phantom.py` beyond a re-export, `maps.py` beyond helpers
  you need (ask first).
- Work on a branch or worktree; commit per milestone with a message describing the check and
  its numbers; rebase on `main` before pushing; never force-push `main`. End commit messages with
  the attribution line used in this repository.
- When something in this spec is wrong or ambiguous, fix the spec in the same commit and say so.
