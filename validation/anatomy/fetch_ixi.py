"""Fetch the IXI dataset's 3 T subjects (Hammersmith Hospital, Philips Intera 3 T): T1 and T2 volumes and
the demographics sheet, from Imperial College's IXI page (brain-development.org/ixi-dataset, CC BY-SA 3.0:
acknowledge "IXI dataset, brain-development.org"). Each tar (4.84 and 3.85 GB) is fetched in parallel byte
ranges (one stream ran at 0.65 MB/s, eight ranges at ~2 MB/s each) to a temporary file, only the HH files
are extracted (~1/3), and the tar is deleted. The first fetch pins every kept file's SHA-256 in
<root>/ixi/manifest.json; later runs verify against it and never refresh pins silently.

    uv run python validation/anatomy/fetch_ixi.py [--root validation/anatomy/data] [--offline]
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
BASE = "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/"
TARS = {"T1": ("IXI-T1.tar", 4840816640), "T2": ("IXI-T2.tar", 3853445120)}
SHEET = ("IXI.xls", 210432)
SITE = "-HH-"  # Hammersmith Hospital, the 3 T scanner


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(root, offline=False):
    out = root / "ixi"
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is None and offline:
        raise FileNotFoundError(manifest_path)
    for contrast, (name, size) in TARS.items():
        dest = out / contrast
        if manifest is None:
            dest.mkdir(parents=True, exist_ok=True)
            tar = out / (name + ".part")
            parts, n = [], 8
            edges = [size * i // n for i in range(n + 1)]
            procs = []
            for i in range(n):
                part = out / f"{name}.{i}"
                parts.append(part)
                procs.append(subprocess.Popen(["curl", "-fsSL", "--retry", "3", "-r", f"{edges[i]}-{edges[i + 1] - 1}",
                                               BASE + name, "-o", str(part)]))
            if any(p.wait() != 0 for p in procs):
                raise RuntimeError(f"download failed: {name}")
            with tar.open("wb") as whole:
                for part in parts:
                    whole.write(part.read_bytes())
                    part.unlink()
            if tar.stat().st_size != size:
                raise ValueError(f"{name}: {tar.stat().st_size} bytes, expected {size}")
            subprocess.run(["tar", "-x", "-f", str(tar), "-C", str(dest), "--include", f"*{SITE}*"], check=True)
            tar.unlink()
    sheet = out / SHEET[0]
    if manifest is None:
        subprocess.run(["curl", "-fsSL", "--retry", "2", BASE + SHEET[0], "-o", str(sheet)], check=True)
        files = {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                 for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"}
        manifest = {"schema": 1, "source": "IXI dataset, brain-development.org/ixi-dataset (CC BY-SA 3.0)",
                    "urls": {k: BASE + v[0] for k, v in TARS.items()} | {"sheet": BASE + SHEET[0]},
                    "archive_bytes": {k: v[1] for k, v in TARS.items()}, "site_filter": SITE,
                    "fetched": datetime.date.today().isoformat(), "files": files}
        manifest_path.write_text(json.dumps(manifest, indent=1))
    for rel, entry in manifest["files"].items():
        path = out / rel
        if not path.exists() or sha256(path) != entry["sha256"]:
            raise ValueError(f"missing or changed: {path}; do not silently refresh pins")
    counts = {c: sum(1 for k in manifest["files"] if k.startswith(c + "/")) for c in TARS}
    print(f"IXI {SITE.strip('-')}: {counts} files verified in {out}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=HERE / "data")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    fetch(args.root, args.offline)
