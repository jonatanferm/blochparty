# Lessons learned

A running notebook of things we found out the hard way: physics and numerics that bite, quirks of formats, data
and other tools, performance facts, and process. Add to it whenever something costs more than a few minutes to
figure out, in the topic file it belongs to (newest at the top of a section); each lesson says where it came from
so it can be re-checked. `tests/docs/test_docs.py` checks this index lists every topic file.

- [Physics and numerics](lessons/physics.md) (59): relaxation, rotations, exchange, diffusion, float32, estimators
- [GPU engines](lessons/gpu.md) (15): Metal and CUDA kernels, the port, GPU processes
- [Performance](lessons/performance.md) (27): what makes the simulators fast or slow
- [Pulseq and sequences](lessons/pulseq.md) (22): Pulseq files, pypulseq, rasters, GIRF and gradient delays
- [Reference data, datasets, formats and other simulators](lessons/reference-data.md) (49): NIST, published values, the tissue database, MRzero, KomaMRI, BMCTool
- [Anatomy and realism](lessons/anatomy-realism.md) (42): BrainWeb, IXI, AHEAD, fastMRI, segmentation, the realism benchmark
- [Sequence design](lessons/design.md) (9): learning sequences, classifiers under an acquisition budget
- [Process and tooling](lessons/process.md) (48): shells, machines, downloads, tests, git
