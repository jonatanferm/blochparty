"""Download the Pulseq multi-slice radial 2D phantom pair (Zenodo 10.5281/zenodo.14228459, CC BY 4.0;
38.4 MB: the .seq and the acquired ISMRMRD HDF5 with trajectory) into validation/real/data/radial/,
checking Zenodo's MD5 sums.

    uv run python validation/real/fetch_radial.py
"""

import hashlib
import json
import subprocess
from pathlib import Path

RECORD = "https://zenodo.org/api/records/14228459"
OUT = Path(__file__).with_name("data") / "radial"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    record = json.loads(subprocess.run(["curl", "-sfL", RECORD], capture_output=True, check=True).stdout)
    n = 0
    for f in record["files"]:
        dest = OUT / f["key"]
        want = f["checksum"].removeprefix("md5:")
        if dest.exists() and md5(dest) == want:
            continue
        subprocess.run(["curl", "-sfL", f["links"]["self"], "-o", str(dest)], check=True)
        if md5(dest) != want:
            raise RuntimeError(f"checksum mismatch for {dest}")
        n += 1
    print(f"fetched {n} files into {OUT}")


if __name__ == "__main__":
    main()
