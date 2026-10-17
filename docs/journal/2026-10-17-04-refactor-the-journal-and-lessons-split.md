# 2026-10-17 — Refactor: the journal and the lessons split (plan phase 9, documents)

- The refactoring plan (`docs/REFACTOR_PLAN.md`) is being carried out phase by phase on `main`; the user asked for all
  ten phases and an audit for tautological tests along the way.
- **Journal**: the 2729-line `docs/JOURNAL.md` split into 64 files `docs/journal/<date>-<nn>-<slug>.md` (nn: the
  entry's place in its day, so files sort chronologically), each the entry as it was with its `## ` heading as
  `# `; one relative link rewritten for the deeper folder. Checked: the 2590 non-blank lines of the entries are
  identical, in order, before and after. `docs/JOURNAL.md` is now the intro and one line per entry.
- **Lessons**: the 1395-line `docs/LESSONS.md` split into eight topic files under `docs/lessons/`: physics (59), gpu
  (10), performance (27), pulseq (22), reference-data (49), anatomy-realism (42), design (4), process (46). The
  "Pulseq, data and reference values" section held three topics and was sorted by keyword with 21 hand overrides by
  title; the four dated sections folded into their topics as subsections; GPU lessons from the physics and process
  sections moved to `gpu.md`. Checked: the same 259 lessons, text identical. `docs/LESSONS.md` is now an index with
  counts.
- `tests/docs/test_docs.py` keeps the indexes honest (index lines and files agree, entries in date order, file names
  and headings, lesson counts); checked to fail on an unindexed entry and a miscounted topic.
- `CLAUDE.md` and `ROADMAP.md` name the new places. Also fixed on the way: `test_phantom_engines.py` called
  `test_metal_grad.compare` without the engine argument the GPU fixture gave it (b5cb509); the quick suite's
  baseline run caught it (`ec16ddc`).
