# 2026-10-09 — fastMRI fetch started; tissue database begun (research only)

- fastMRI brain: `validation/realism/fetch_fastmri.py` streams `brain_multicoil_val_batch_0.tar.xz`
  (curl -> tarfile, xz on the fly) and keeps AXT2 (15 GB), AXT1/AXT1PRE (4 + 4) and AXFLAIR (7) on
  the Mac mini (`~/data/fastmri_brain`, outside the repo) with a manifest of field strength,
  protocol, TR/TE and matrix per volume. Tested on a synthetic archive; running at ~5 MB/s (network
  bound, ~2 h). The signed link lives in a mode-600 curl config on the mini, never on a command
  line; `*.eml` is git-ignored. Links expire 2026-11-06.
- Tissue database (user's request): sources located, nothing written yet. IT'IS v4.2 relaxation
  times (T1/T2 at 1.5/3 T, ~30 tissues with own data; others are IT'IS proxies) and v5.0 water
  content (mass %, >100 tissues) come as JSON from the IT'IS site; open full texts with tables:
  Jordan 2013 (MSK 3/7 T, PMC3310288), Edden 2010 (breast 3 T, PMC3154635), Ma 2013 (MRF, PMC3602925),
  Rane 2013 (7 T blood, PMC3561475); abstracts carry Rooney 2007 (T1 power laws, CSF 4.3 s), Lu 2004
  (blood T1 vs Hct at 3 T), Wansapura 1999 (its abstract swaps T2 and T2* labels). Next: schema,
  loader and tests (`src/blochparty/tissuedb/`), then rows.
