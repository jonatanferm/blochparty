# 2026-10-08 — Roadmap rewritten as a plan; old roadmap archived

- Review of the project state: 110 commits in four days, 484 tests green (3 min 22 s), nine
  real-data benchmarks. The roadmap had become a log: most of its length was verified-numbers
  paragraphs, its open items mixed stale Phase 1 performance ideas (superseded by the Metal
  kernel) with real gaps, and it listed three goals as equals while nearly all work went into
  realism. Two structural points were not written down anywhere: the engine policy ("Metal is
  the engine") leaves the project without GPU gradients, so the differentiable-calibration
  claim in REALISM.md has never been exercised on real data (benchmark 9 fitted by finite
  differences in the core); and anatomy generation, the second pillar, has loaders and texture
  but no sampler.
- `ROADMAP.md` → `docs/ROADMAP_ARCHIVE.md` unchanged (the citable verified record; README and
  REALISM.md now point there for it). New `ROADMAP.md`: the vision (core first on this Mac,
  anatomy generation second, training data and agentic design on top), a testable definition of
  "the core is satisfactory" (Metal parity, Metal adjoint, physics no longer moves the NIST
  benchmark table once the unknown inputs are fitted), working rules for an implementing agent,
  and milestones M0–M8 with design notes, file pointers and verification for each: consolidate
  (engine matrix + raise tests), Metal parity (free pools, motion with the field model), Metal
  adjoint (two-level checkpoints, deterministic per-segment reductions, retire the fused kernel
  after), the NIST digital-twin loop (fit B1+/receive/geometry/timing on one plane, predict the
  other, ablation table against jackknife bars), Pulseq writer with block provenance, anatomy
  sampler + dataset writer + two-sample test, the physics backlog reordered by what training
  data hits first (J-coupled fat, deformable motion, complex B1+/pTx, contrast kinetics,
  orientation-dependent R2*), then training-data/agentic hooks and ports.
- Explicitly not planned: model-based recon for benchmark 3, PDG backend, XCAT, GUI.
