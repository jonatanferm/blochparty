# 2026-10-10 — Tissue-property database (`blochparty.tissuedb`, branch `tissuedb`)

- **What**: a sourced table of MR tissue properties, package data in `src/blochparty/tissuedb/`:
  `measurements.csv` (1859 rows, one published value each, SI, with source, table/page, field,
  species, state, temperature, method, n and SD kind), `sources.csv` (158, each with DOI or URL),
  `tissues.csv` (126 dotted ids, one per BrainWeb class marked). `query`, `pooled` (human in vivo
  at the field first; every fallback marked assumed; T1/T2/T2* never borrow another field) and
  `tissue_table(field)`, a candidate {class: TissueProperties} that falls back to DEFAULT_TISSUES
  per field. DEFAULT_TISSUES is unchanged (snapshot bit-identical; pickle-equal before/after).
- **Why**: the tissue table is the largest remaining unknown outside the brain (ROADMAP M5 left
  skin/connective/dura/marrow/skull as proxies) and the values in anatomy.py had no machine-readable
  provenance. Switching the default to the candidate is the user's decision.
- **Rows**: the values anatomy.py uses (round trip tested per value); IT'IS v4.2 relaxation
  (T1/T2 at 1.5/3 T) and Gabriel Cole-Cole dielectric properties at 64/128/298 MHz, and v5.0 water
  content (`tools/tissuedb_import_itis.py`, reproducible, 212 blocks kept, 459 skipped as proxies
  or empty, logged with download SHA-256); then eight literature passes (open PMC tables, classic
  abstracts, body relaxometry incl. 0.55 T, head gaps: eye, cortical bone UTE, skin, glands; deep
  GM and QSM; blood vs oxygenation and fat fractions; MT, DTI and PD; organs incl. myocardium and
  5/7 T abdomen). Property x field coverage: T1 463 rows, T2 281, T2* 137, MT 185, MD/FA 165,
  PD 65, PDFF 72, QSM 120, dielectric 300, water content 68.
- **Checks** (`tests/test_tissuedb.py`, 55 tests): schema (keys, units, fractions, field, no
  duplicates), unit magnitudes per property, pooled T2 < T1 and T2* <= T2, T1 rising with field
  within a source, human in-vivo sources within a factor of two (fractions < 0.2: within 0.1), the
  round trip to DEFAULT_TISSUES, Li 2014's 8 muscle MDs reproduce 1.78 +- 0.08 um^2/ms, the
  Cole-Cole evaluation against the IFAC/Gabriel values at 128 MHz (GM 73.5 / 0.59 S/m, WM 52.5 /
  0.34, CSF 84.0 / 2.14, muscle 63.5 / 0.72), and tissue_table(1.5 / 3) through
  to_phantom(mt=True, diffusion=True). Eight documented exceptions, method- or region-driven (MRF
  vs multi-echo T2 in kidney cortex/medulla and muscle, UTE echo times in lung, atlas-ROI vs
  age-fit cortical FA, disc compartments, Peters' CSF-contaminated WM T2*, Ma 2019 lens T1);
  compilations are left out of the T1-rises-with-field check (IT'IS pools other studies per field). A spot check of 8 random agent rows against the PMC XML: all 8 values match.
- **What would change at 3 T** if the candidate became the default (full table in the PR): WM T1
  1084 -> 889 ms and GM 1820 -> 1344 ms (human in vivo replaces ex vivo bovine), WM PD 0.77 -> 0.70,
  fat PDFF 1 -> 0.89, marrow PDFF 1 -> 0.39 (vertebral red marrow pooled with yellow), skull gains
  signal (PD 0.23, T1 237 ms, T2* 0.58 ms), CSF T2* 2.03 s -> 0.20 s (Shin 2025, flow/diffusion
  included), muscle MD 1.78 -> 1.22 um^2/ms (pharyngeal constrictor pooled with thigh). Weak cells:
  WM T2 86 ms is pulled up by SyMRI (flagged), blood MT comes from Stanisz 2006 with a suspect T2B.
- **Corrections found on the way**: Rooney's CSF T1 is 4.3 +- 0.2 s (the code quotes 4.4 via
  Spijkerman); Caan 2019 n = 6 analysed of 8; Jordan JMRI 2013 (the chi citation) is a breast-B0
  paper quoting Schenck 1996; `validation/anatomy_brainweb.py` takes Wansapura's T2 in the
  abstract's (swapped) order. Lessons in LESSONS.md.
- **Dead ends**: Stanisz 2005, Gold 2004, Bojorquez 2017, Sled & Pike 2001, Gloor 2008 and de
  Bazelaire's full text are behind Cloudflare/captcha here; Stanisz's 3 T values were confirmed from
  the group's ISMRM 2006 abstract, the 1.5 T values remain unverified. No skin/scalp T1/T2 at 1.5/3 T
  from a primary source, and no dura, connective-tissue or mucosa relaxometry at all.
- Commits (branch `tissuedb`): ba847f8 (package, IT'IS import, checks), 655ffd4, 67b68d8 (deep GM),
  6050eec (head gaps, blood/fat), 247abc7 (organs), and the MT/diffusion/PD rows with this entry.
