# Sequence design by lesion classification

`lesion_slabs.py` renders slab phantoms from prepared IXI heads with and without white-matter lesions; `lesion_design.py` trains a classifier on what each arm acquires (a TSE with CG-SENSE, the TSE's adjoint, a fixed GRE, and a GRE whose flips, TE, TR and line positions are learned with the classifier through the GPU adjoint) under a k-space budget. Results are in the journal (2026-10-16 and 2026-10-17).

    uv run python validation/design/lesion_slabs.py --prepared validation/anatomy/data/ixi/prepared_v17 --out ~/data/slabs
    uv run python validation/design/lesion_design.py --slabs ~/data/slabs --out ~/data/design_f0.5 --fraction 0.5 --sampling gauss
