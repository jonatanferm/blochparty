"""Fetch selected members of the iohexol CEST phantom archive (Huemer et al., "Dynamic transitions
for fast joint estimation of CEST-Rex and T1", MRM 2025; Zenodo 10.5281/zenodo.17244380, CC BY 4.0)
into validation/real/data/iohexol/, without downloading the 31.6 GB ZIP: HTTP range reads of the
ZIP directory and of each member (CRC-checked by zipfile on extraction).

Members: the two conventional CEST reference scans (Siemens raw, 3 T MAGNETOM Vida, saturation
1.5 s and 4 s), the 15-pulse reference .seq, the WASABI B0 / B1 maps and ROI masks, and the
authors' scripts for reference.

    uv run python validation/real/fetch_iohexol.py
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path

RECORD = 17244380
KEY = "00_CEST_T1.zip"
OUT = Path(__file__).with_name("data") / "iohexol"
MEMBERS = (
    "data/raw/phantom/meas_MID00127_FID56966_Ref_Tsat_1p5.dat",
    "data/raw/phantom/meas_MID00131_FID56970_Ref_Tsat_4.dat",
    "sequences/phantom/Ref_00_3D_res128_alpha_4_OC_15_100ms_1ms_4ppm_Phantom_Trec_3p5.seq",
    "data/prepared/phantom/meas_MID00122_FID56961_WASABI_3D_dB0_map_NN.mat",
    "data/prepared/phantom/meas_MID00122_FID56961_WASABI_3D_B1_map_NN.mat",
    "data/prepared/phantom/mask.mat",
    "scripts/Script_05_reco_CEST_Ref_4Pool.m",
    "scripts/Script_06_reco_WASABI.m",
    "readme",
)


class _Ranged(io.RawIOBase):
    """A seekable read-only view of a URL by HTTP range requests."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def main():
    rec = json.loads(urllib.request.urlopen(f"https://zenodo.org/api/records/{RECORD}").read())
    url = next(f for f in rec["files"] if f["key"] == KEY)["links"]["self"]
    z = zipfile.ZipFile(io.BufferedReader(_Ranged(url), buffer_size=1 << 20))
    for m in MEMBERS:
        info = z.getinfo("00_CEST_T1/" + m)
        dest = OUT / Path(m).name
        if dest.exists() and dest.stat().st_size == info.file_size:
            continue
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with z.open(info) as src, open(tmp, "wb") as dst:  # zipfile checks the CRC at the end
            while chunk := src.read(1 << 24):
                dst.write(chunk)
        tmp.rename(dest)
        print(f"{dest.name}: {info.file_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
