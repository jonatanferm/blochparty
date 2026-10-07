"""Fetch pinned public challenge ground truth, not the restricted full-head package.

This SURFdrive share is linked by Berkin Bilgic's lab software page. No login,
personal information or form submission is needed. No explicit redistribution
licence is asserted: keep downloaded data ignored and cite the challenge papers.
"""
import argparse
import json
from pathlib import Path
import subprocess

from blochparty.anatomy import _file_sha256, load_qsm2

HERE = Path(__file__).resolve().parent


def fetch(model,root=None):
    manifest = json.loads((HERE/f"qsm_challenge{model}.json").read_text())
    root = Path(root) if root is not None else HERE/"data"/f"qsm_challenge{model}"
    root.mkdir(parents=True,exist_ok=True)
    for entry in manifest["files"].values():
        path = root/entry["path"]
        if not path.exists():
            part = path.with_suffix(".part.gz")
            subprocess.run(["curl","-fsSL","--retry","2","--max-time","180",entry["url"],"-o",str(part)],check=True)
            if _file_sha256(part) != entry["sha256"]:
                raise ValueError(f"download checksum mismatch: {part}")
            part.replace(path)
        if _file_sha256(path) != entry["sha256"]:
            raise ValueError(f"checksum mismatch: {path}; do not silently refresh pins")
        print(f"Verified {path.name}",flush=True)
    manifest_path = root/"qsm2.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("existing manifest differs; choose a separate root")
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    anatomy = load_qsm2(root,resolution="native")
    print(f"Model {model}: shape={anatomy.shape}, voxel={anatomy.voxel}; chi and segmentation only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=int,choices=(1,2),default=1)
    parser.add_argument("--root",type=Path)
    args = parser.parse_args()
    fetch(args.model,args.root)
