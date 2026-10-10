# Tissue-property database

A sourced table of MR properties of human tissues: good enough, not perfect. Every number keeps its paper,
table or page, field strength and conditions, so it can be checked and replaced. It can produce the
simulator's tissue tables (`tissue_table`), but `anatomy.DEFAULT_TISSUES` is not built from it:
switching is a decision (see the comparison in `docs/JOURNAL.md`, 2026-10-10).

```python
from blochparty import tissuedb
tissuedb.query("brain.wm", "t1", 3.0, species="human", state="in_vivo")   # list of Measurement rows
tissuedb.pooled("brain.wm", "t1", 3.0)          # anatomy.Distribution(mean, sd, citation, assumed)
table = tissuedb.tissue_table(3.0)              # {BrainWeb class: TissueProperties}, a candidate
to_phantom(anatomy, {3.0: table}, 3.0)
```

## Files

| file | content |
|---|---|
| `measurements.csv` | one row per published value (long format) |
| `sources.csv` | `key, authors, year, title, venue, doi, url, kind, note`; every key has a DOI or URL; kind = article / abstract / review / database |
| `tissues.csv` | `id, name, parent, brainweb_class, description`: the controlled vocabulary |
| `itis_import_log.csv` | written by `tools/tissuedb_import_itis.py`: every IT'IS row and block, taken or skipped and why, with the SHA-256 of the downloads |

### measurements.csv

`tissue, region, property, field_T, value, sd, sd_kind, n, unit, species, state, temperature_C, method, source, location, note`

- **tissue**: an id from `tissues.csv`; **region** the paper's own label ("putamen", "gastrocnemius").
- **property / unit** (SI only; the original number and unit go in `note`):

  | property | unit | meaning |
  |---|---|---|
  | `t1`, `t2`, `t2star` | s | relaxation times (field_T required) |
  | `bound_t2` | s | T2 of the semi-solid pool (super-Lorentzian unless noted) |
  | `bound_fraction` | 1 | M0b / (M0b + M0f); a pool size ratio PSR = M0b/M0f gives f = PSR / (1 + PSR) |
  | `exchange_rate` | 1/s | bound -> free (= `anatomy` `BoundPool.k`): Henkelman / Sled-Pike R with M0f = 1, Yarnykh's R = k(1-f)/f (his k is free -> bound) |
  | `md` | m^2/s | ADC or DTI mean diffusivity (method says which) |
  | `fa` | 1 | fractional anisotropy |
  | `pd` | 1 | MR-visible proton density relative to pure water |
  | `water_content_mass` | 1 | water mass fraction (not `pd`) |
  | `fat_fraction` | 1 | proton-density fat fraction |
  | `chi` | ppm | absolute SI volume susceptibility (water -9.05, air +0.36) |
  | `chi_qsm` | ppm | QSM value relative to the paper's reference (`method` says "QSM ref=...") |
  | `permittivity_rel`, `conductivity` | 1, S/m | at the Larmor frequency of field_T |

- **field_T**: B0 [T]; required for t1, t2, t2star, permittivity and conductivity, given whenever known.
- **value / sd**: mean (median if `note` says so) and SD; an empty SD means unreported, never zero. Ranges go in `note`.
- **sd_kind**: between_subject, within_subject, specimen, across_studies, se, unknown. **n**: subjects, specimens or studies.
- **species**: human, bovine, ...; **state**: in_vivo, ex_vivo, in_vitro, compilation (a database or review), physical (a pure substance).
- **method**, **location** ("Table 2", "abstract", "Results"), **note**: what is needed to read the value; "secondary: X" when the number was quoted by another paper, "derived:" when computed from published coefficients (a fit, R2* -> T2*, PSR -> f).

### tissues.csv

Dotted ids (`brain.gm.deep.putamen`), with `parent` the next level up. `brainweb_class` names the one
tissue that stands for each BrainWeb class: background = air, csf, gm = brain.gm, wm = brain.wm,
fat = fat.subcutaneous, muscle = muscle.skeletal, skin, skull = bone.cortical, vessels = blood,
connective, dura, marrow = bone_marrow (pooled over red and yellow marrow when it has no rows of its own).
Compartments that add up rather than average (cortical bone's bound water, pore water and collagen
protons) have their own ids (`bone.cortical.bound_water`, ...), so they are never pooled together.

## Pooling (`pooled`)

1. Rows of the tissue and property; tiers: human in vivo at the requested field (assumed = False), then
   human compilation, human ex vivo, human in vitro, physical, animal in vivo / ex vivo / in vitro
   (assumed = True). T1, T2, T2* and dielectric values never come from another field; MT parameters
   may (assumed = True); diffusion, PD, fat fraction, susceptibility and water content do not depend on field.
2. Within the best tier, one value per source: the mean of its rows (its regions), its SD the RMS of
   their SDs (an SE becomes SD x sqrt(n)), its weight its largest n (1 if unknown).
3. Mean = n-weighted mean over sources; SD = sqrt(weighted mean within-source SD^2 + weighted
   between-source variance); None for a single source without an SD.

`tissue_table(field_T)` pools every TissueProperties field for each class's tissue, then its
descendants (assumed), then falls back to the current `DEFAULT_TISSUES[field_T]` entry; T2* above T2
is clipped to T2; a bound pool without an exchange rate and bound T2 is omitted, and a signalling class
without MT or diffusion data gets bound_fraction 0 and diffusivity 0 (all marked assumed), so
`to_phantom(mt=True, diffusion=True)` runs. `tools/tissuedb_report.py` prints the row counts and the
candidate next to the current default. Rows describing different subjects of one cohort in two papers (Peters 2006 / 2007)
count as two sources: the pooling cannot see shared subjects.

## Adding a row

1. Read the number from the page source (PMC/Europe PMC XML tables, the PDF text), not from a summary;
   abstracts via PubMed efetch, with location "abstract".
2. Convert to SI, keep the original in `note`; say "secondary" or "derived" when it is.
3. Add the source to `sources.csv` (DOI without the https://doi.org/ prefix) and any new tissue id to `tissues.csv`.
4. `uv run pytest tests/test_tissuedb.py`: schema, unit magnitudes, T2 < T1 and T2* <= T2, T1 rising
   with field within a source, human in-vivo sources within a factor of two, and the round trip to
   `DEFAULT_TISSUES`.

IT'IS rows are regenerated by `uv run python tools/tissuedb_import_itis.py` (it replaces only rows with
sources `itis_v4_2` and `itis_water_v5_0`); edit its mapping, not the rows.
