# The realism benchmark

Can a classifier tell real fastMRI 3 T T2w slices from simulated ones (`score.py`), where the gap sits (`gap.py`), and how much a reconstruction network trained on simulated data transfers to real k-space (`transfer.py`). `simulate_heads.py` simulates the heads with the ablation options; `generate.py` is the command line of `blochparty.generate` (spec + seed -> HDF5 samples with their truth). The real data: `fetch_fastmri.py` (kept outside the repository); `orientation.py` and `echo_train.py` read conventions from it. `quick.sh` is a ~10-minute iteration on the mini. Results: [docs/REALISM.md](../../docs/REALISM.md) and the journal.

    uv run python validation/realism/generate.py --spec spec.json --seeds 0-9 --out ~/data/gen
    validation/realism/quick.sh <tag> [prepared dir] [simulate_heads args]
