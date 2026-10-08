"""Download the phantom GIRF measurements of Wu et al. (ISMRM 2022; Zenodo 10.5281/zenodo.6376737,
CC BY 4.0; 1.23 GB: Meas1.zip, Meas2.zip, CalculatedGIRF.zip) into validation/real/data/girf/, checking
Zenodo's MD5 sums, and unzip them. Processing reference: github.com/BRAIN-TO/girfISMRM2022 (92bc337).

    uv run python validation/real/fetch_girf.py
"""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

RECORD = "https://zenodo.org/api/records/6376737"
OUT = Path(__file__).with_name("data") / "girf"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    record = json.loads(subprocess.run(["curl", "-sfL", RECORD], capture_output=True, check=True).stdout)
    for f in record["files"]:
        dest = OUT / f["key"]
        want = f["checksum"].removeprefix("md5:")
        if not (dest.exists() and md5(dest) == want):
            subprocess.run(["curl", "-sfL", f["links"]["self"], "-o", str(dest)], check=True)
            if md5(dest) != want:
                raise RuntimeError(f"checksum mismatch for {dest}")
        with zipfile.ZipFile(dest) as z:
            if not all((OUT / n).exists() for n in z.namelist()):
                z.extractall(OUT)
        print(f"{f['key']}: ok", flush=True)


if __name__ == "__main__":
    main()
