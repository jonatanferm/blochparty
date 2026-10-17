# 2026-10-17 — Refactor: the sequences package (plan phase 7)

- **Layout** (`02d6fac` moves, `0f4e3a8` split, `b3b26a3` meta, `a336632` docs): `sequence.py` keeps `Sequence`,
  `kspace_trajectory` and now `flip_angle`; `sequences/` holds `builder` (System, the events and pulses,
  SequenceBuilder: the lower half of `sequence.py`, verbatim), `library` (was `builders.py`), `pulseq/read.py` and
  `pulseq/write.py`, `girf`, `drift`, `transmit` and the new `meta`. `blochparty.builders`, `.pulseq`, `.girf`,
  `.drift` and `.transmit` forward to the new modules. `flip_angle` went to `sequence.py` rather than the builder
  (the plan's place), because `kspace_trajectory` uses it and the builder imports `sequence.py`; that removed the
  last sequence <-> pulseq lazy imports.
- **One `_subdivide`**: pulseq's copy returned the points with the extra points appended, the builder's only the
  extra points. The reader now calls the builder's and concatenates the points first at its three call sites, so
  every array is the same element for element; pulseq's own `_T_TOL` was the same 1e-11 as sequence's.
- **`sequences/meta.py`**: the 31 keys written into `Sequence.meta` (by the builder, `merge_free`, `read_pulseq` and
  the library), each with its meaning. `tests/sequences/test_meta.py` scans the package's source for the six ways
  it writes a key, asserts every written key is listed and every listed key written, and checks the scan on a
  file with each form; dropping one key from the table fails it. The plan's per-key constants were left out:
  nothing would use them.
- **A move that broke something only a round trip shows**: `io.read_dataset` rebuilds the sequence's events by
  class name and looked only in `blochparty.sequence`; after the move they came back as dicts. `test_io`'s round
  trips caught it; it looks in the builder first now.
- **Checks**: core snapshot bit-identical (its cases are built by the library); tests/sequences and tests/signal:
  127 passed.
- **Full suite on the mini** (phases 3, 5 and 7 together, after all three were committed): 847 passed, 180 skipped,
  0 failed, 13 min.
