"""Download the Pulseq-CEST MAGMA 2025 phantom DICOMs (Zenodo 10.5281/zenodo.15053086, CC BY 4.0; 53.6 MB:
APTw and WASABI hybrid-GRE series) into validation/real/data/magma/, check Zenodo's MD5 sums and unzip.
Preparation sequences: github.com/kherz/pulseq-cest-library, tag MAGMA_2025_PAPER (0bd96f5).

    uv run python validation/real/fetch_magma.py
"""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

RECORD = "https://zenodo.org/api/records/15053086"
LIBRARY = "https://raw.githubusercontent.com/kherz/pulseq-cest-library/0bd96f5a3ccbf30897417afee9eab32d15e3242b"
SEQS = (
    "seq-library/APTw_3T_001_2uT_36SincGauss_DC90_2s_braintumor/APTw_3T_001_2uT_36SincGauss_DC90_2s_braintumor.seq",
    "seq-library/WASABI_3T_001_3p7uT_1block_5ms/WASABI_3T_001_3p7uT_1block_5ms.seq",
)
OUT = Path(__file__).with_name("data") / "magma"


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
            z.extractall(OUT / dest.stem)
    for s in SEQS:
        dest = OUT / Path(s).name
        if not dest.exists():
            subprocess.run(["curl", "-sfL", f"{LIBRARY}/{s}", "-o", str(dest)], check=True)
    print(f"done: {sorted(p.name for p in OUT.iterdir())}")


if __name__ == "__main__":
    main()
