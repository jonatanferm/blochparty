"""The generation driver (ROADMAP M7): a spec and a seed -> one simulated sample with its truth, written with provenance.

    for s in $(seq 0 299); do uv run python validation/realism/generate.py --spec spec.json --seeds $s-$s --out ~/data/gen_t2w; done

One head per process: a process that ran 60 heads on the mini began returning all-zero signals from the GPU after
~30 slices (2026-10-14); the driver refuses to write such a sample, so a per-head loop can simply retry.

A sample is one slice of multi-coil k-space as a scanner would store it (ISMRMRD HDF5 via ``blochparty.io.write_dataset``:
the signal in sequence order, the trajectory, the sequence, the phantom's draws), plus a ``/truth`` group a scanner never
has: the noise-free k-space and image on the acquisition grid, the noisy k-space the sample "acquired", the tissue
fractions and the residual maps on that grid, the lesion mask, and ``meta.json`` (every draw: anatomy, protocol,
scanner, quality). One JSONL manifest line per sample. The same spec, seed and engine version regenerate a sample.

The spec, the anatomy, the protocol, the simulation, the noise and the files are ``blochparty.generate``
(``generate.spec`` documents the spec's keys); this script adds the real scans: their protocols from the fastMRI
manifest (default ``realism.FASTMRI``), their background noise floors and coil covariances (``score.real_set``,
``score.noise_covariances``), and the anatomy data in this repository.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blochparty import generate  # noqa: E402
import realism as R  # noqa: E402
import score as SC  # noqa: E402
import simulate_heads as SH  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, default=None, help="JSON spec (blochparty.generate.spec); defaults without it")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path.home() / "data" / "realism_cache")
    ap.add_argument("--fastmri", type=Path, default=None, help="real scans (floors, coil covariances); default from the spec's manifest")
    ap.add_argument("--set", action="append", default=[], help="override spec entries: section.key=json, e.g. protocol.n=64")
    ap.add_argument("--serial", action="store_true", help="build each slice's phantom after the previous one is simulated (half the memory)")
    args = ap.parse_args()
    spec = {"protocol": {"manifest": str(R.FASTMRI / "manifest.json")}}  # the real scans, unless the spec says otherwise
    for sec, values in (json.loads(args.spec.read_text()) if args.spec else {}).items():
        spec.setdefault(sec, {}).update(values)
    for item in args.set:
        key, _, value = item.partition("=")
        sec, k = key.split(".")
        spec.setdefault(sec, {})[k] = json.loads(value)
    spec = generate.merged(spec)
    manifest = None if spec["protocol"]["manifest"] is None else Path(spec["protocol"]["manifest"]).expanduser()
    from blochparty import tissuedb
    table = {3.0: tissuedb.tissue_table(3.0)}
    if manifest is not None and manifest.exists():
        root = args.fastmri or manifest.parent
        protocols = generate.protocols.real_protocols(manifest)
        real, floors = SC.real_set(args.cache, root, range(1, 8))
        covs = SC.noise_covariances(args.cache, root) if spec["scanner"]["noise"] == "paired" else None
        del real
    else:  # no real scans at hand: one fastMRI-like protocol, the typical background floor, white noise
        protocols, floors, covs = generate.FALLBACK_PROTOCOLS, {generate.FALLBACK_PROTOCOLS[0]["file"]: 0.017}, None
        print("no fastMRI manifest: the fallback protocol, white noise at a floor of 0.017", flush=True)
    (args.out).mkdir(parents=True, exist_ok=True)
    (args.out / "spec.json").write_text(json.dumps(spec, indent=1))
    for seed in generate.span(args.seeds):
        for s in generate.samples(spec, seed, root=SH.ROOT, table=table, protocols=protocols, floors=floors, covs=covs,
                                  serial=args.serial):
            path = generate.write(s, args.out)
            with open(args.out / "manifest.jsonl", "a") as m:
                m.write(json.dumps(s["record"]) + "\n")
            print(f"seed {seed} slice {s['meta']['slice']}: {s['meta']['spins']:,} spins, {s['meta']['seconds']} s "
                  f"({s['meta']['engine']}) -> {path.name}", flush=True)


if __name__ == "__main__":
    main()
