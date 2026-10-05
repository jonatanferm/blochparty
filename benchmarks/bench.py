"""Fixed simulator workloads, timed. Use before and after every performance change.

    uv run python benchmarks/bench.py                  # default cases
    uv run python benchmarks/bench.py --cases 3d-64    # pick cases
    uv run python benchmarks/bench.py --record         # also append to history.jsonl

Throughput is reported as spin-segments/s (work the simulator does) and as
spin-samples/s (the irreducible spins x ADC-samples cost of any Bloch simulator).
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import blochparty as bp

N_DUMMY = 200
CASES = {
    "2d-96": (2, 96),
    "3d-32": (3, 32),
    "3d-48": (3, 48),
    "3d-64": (3, 64),
}
DEFAULT = ["2d-96", "3d-32", "3d-48"]
HISTORY = Path(__file__).with_name("history.jsonl")


def run_case(name: str, dtype, device: str) -> dict:
    ndim, n = CASES[name]
    phantom = bp.shepp_logan(n, ndim=ndim, dtype=torch.float64).to(dtype=dtype, device=device)
    coils = bp.birdcage_sensitivities(phantom.pos, n_coils=8)
    seq = bp.gre_cartesian((n,) * ndim, n_dummy=N_DUMMY, dtype=dtype)
    t0 = time.perf_counter()
    signal = bp.simulate(phantom, seq, coils).signal
    if device != "cpu":
        torch.mps.synchronize() if device == "mps" else torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    assert torch.isfinite(signal).all()
    return {
        "case": name,
        "spins": len(phantom),
        "segments": len(seq),
        "adc": seq.n_adc,
        "seconds": elapsed,
        "spin_segments_per_s": len(phantom) * len(seq) / elapsed,
        "spin_samples_per_s": len(phantom) * seq.n_adc / elapsed,
    }


def projected_128(throughput: float) -> float:
    """Seconds for a 128^3 GRE at a given spin-segments/s rate."""
    n = 128
    spins = int((bp.shepp_logan_maps(n, ndim=3)["pd"] > 0).sum())
    segments = 2 * N_DUMMY + n * n * (n + 4)  # rf + prephaser + (n + 1) readout + fill per TR
    return spins * segments / throughput


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).parent, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               cwd=Path(__file__).parent).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", nargs="+", default=DEFAULT, choices=list(CASES))
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--record", action="store_true", help="append results to history.jsonl")
    args = p.parse_args(argv)
    dtype = getattr(torch, args.dtype)

    print(f"blochparty {git_commit()} | torch {torch.__version__} | {platform.processor() or platform.machine()}"
          f" | {args.device} {args.dtype} | {torch.get_num_threads()} threads")
    print(f"{'case':>7} {'spins':>9} {'segments':>9} {'time [s]':>9} {'spin-seg/s':>11} {'spin-smp/s':>11}")
    results = []
    for name in args.cases:
        r = run_case(name, dtype, args.device)
        results.append(r)
        print(f"{r['case']:>7} {r['spins']:>9} {r['segments']:>9} {r['seconds']:>9.2f}"
              f" {r['spin_segments_per_s']:>11.2e} {r['spin_samples_per_s']:>11.2e}", flush=True)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20 if sys.platform == "darwin" else 1 << 10)
    best = max(results, key=lambda r: r["spins"])
    proj = projected_128(best["spin_segments_per_s"])
    print(f"peak RSS {peak_mb:.0f} MB | projected 3D 128^3 at {best['case']} throughput: {proj / 60:.0f} min")

    if args.record:
        entry = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": git_commit(),
                 "torch": torch.__version__, "machine": platform.machine(), "device": args.device,
                 "dtype": args.dtype, "threads": torch.get_num_threads(), "peak_rss_mb": round(peak_mb),
                 "projected_128_min": round(proj / 60, 1), "results": results}
        with HISTORY.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"recorded to {HISTORY.name}")


if __name__ == "__main__":
    main()
