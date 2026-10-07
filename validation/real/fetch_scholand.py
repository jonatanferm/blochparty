"""Download the data of benchmark 3 (radial IR-bSSFP of the NIST T2 plane) into
validation/real/data/scholand/ (~560 MB), checking Zenodo's MD5 sums.

Scholand N, Wang X, Roeloffs V, Rosenzweig S, Uecker M. Quantitative MRI by nonlinear inversion
of the Bloch equations. MRM 2023;90:520-538. Data: Zenodo 10.5281/zenodo.7654462 (CC BY 4.0).
Sphere ROIs: the authors' github.com/mrirecon/bloch-moba (CC BY 4.0) at a pinned commit.

    uv run python validation/real/fetch_scholand.py
"""

import hashlib
import json
import subprocess
from pathlib import Path

RECORD = "https://zenodo.org/api/records/7654462"
FILES = ("data_05b_kspace", "data_05b_b1map", "data_GSM_t1", "data_GSM_t2")
MOBA = "https://raw.githubusercontent.com/mrirecon/bloch-moba/df9bc5b336b7dbcde341d29ab2b0888b81901beb"
ROIS = {"rois_radial": "05b_IR-bSSFP_NIST/vertices", "rois_cartesian": "00_gold_standard_nist/vertices"}
OUT = Path(__file__).with_name("data") / "scholand"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    record = json.loads(subprocess.run(["curl", "-sfL", RECORD], capture_output=True, check=True).stdout)
    files = {f["key"]: f for f in record["files"]}
    n = 0
    for name in FILES:
        for ext in ("hdr", "cfl"):
            f = files[f"{name}.{ext}"]
            dest = OUT / f["key"]
            want = f["checksum"].removeprefix("md5:")
            if dest.exists() and md5(dest) == want:
                continue
            subprocess.run(["curl", "-sfL", f["links"]["self"], "-o", str(dest)], check=True)
            if md5(dest) != want:
                raise RuntimeError(f"checksum mismatch for {dest}")
            n += 1
    for sub, path in ROIS.items():
        (OUT / sub).mkdir(exist_ok=True)
        for i in range(14):
            for ext in ("hdr", "cfl"):
                dest = OUT / sub / f"v{i:03d}.{ext}"
                if not dest.exists():
                    subprocess.run(["curl", "-sfL", f"{MOBA}/{path}/v{i:03d}.{ext}", "-o", str(dest)], check=True)
                    n += 1
    print(f"fetched {n} files into {OUT}")


if __name__ == "__main__":
    main()
