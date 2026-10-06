"""Download the real-scanner data used by the realism benchmark into validation/real/data/.

Tong G, Gaspar AS, Qian E, Ravi KS, Vaughan JT, Nunes RG, Geethanath S. Open-source magnetic
resonance imaging acquisition: data and documentation for two validated pulse sequences.
Data in Brief 2022;42:108105. Data: Mendeley Data, doi:10.17632/8458pz722c.5 (CC BY 4.0).

Only the quantitative ISMRM/NIST-phantom acquisitions of the main site (3 T Siemens Prisma Fit)
are fetched: IRSE T1 mapping and multi-echo TSE T2 mapping, ~120 MB.

    uv run python validation/real/fetch_data.py
"""

import json
import subprocess
from pathlib import Path

API = "https://data.mendeley.com/public-api/datasets/8458pz722c"
VERSION = 5
WANTED = ("developer_main_site/IRSE_T1_mapping/", "developer_main_site/TSE_T2_mapping/")
OUT = Path(__file__).with_name("data")


def get_json(url: str):
    # curl rather than urllib: the API rejects urllib's default client
    out = subprocess.run(["curl", "-sfL", "-H", "Accept: application/json", url], capture_output=True, check=True)
    return json.loads(out.stdout)


def main():
    folders = {f["id"]: f for f in get_json(f"{API}/folders/{VERSION}")}

    def path(fid):
        parts = []
        while fid:
            parts.append(folders[fid]["name"])
            fid = folders[fid].get("parent_id")
        return "/".join(reversed(parts))

    n = 0
    for fid in folders:
        folder = path(fid) + "/"
        if not folder.startswith(WANTED):
            continue
        for f in get_json(f"{API}/files?folder_id={fid}&version={VERSION}"):
            dest = OUT / folder / f["filename"]
            if dest.exists() and dest.stat().st_size == f["content_details"].get("size", -1):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["curl", "-sfL", f["content_details"]["download_url"], "-o", str(dest)], check=True)
            n += 1
    print(f"fetched {n} files into {OUT}")


if __name__ == "__main__":
    main()
