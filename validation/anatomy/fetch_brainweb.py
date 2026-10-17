"""Fetch the public BrainWeb 20-model data, verified against a pinned raw-short manifest.

Run: uv run python validation/anatomy/fetch_brainweb.py
No account, licence acceptance, or personal information is submitted. The public
form asks for citations, with optional name/institution/email fields (left blank).
No explicit redistribution licence was found: downloaded/derived volumes stay in
the ignored data directory. See README.md for access and manual-download details.
--record deliberately creates a new local snapshot; review its manifest before use.
"""
import argparse
import gzip
import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path
from urllib.parse import urlencode

HERE = Path(__file__).resolve().parent
ENDPOINT = "https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1"
ALIASES = dict(background="bck", csf="csf", gm="gry", wm="wht", fat="fat",
               muscle="mus", skin="m-s", skull="skl", vessels="ves",
               connective="fat2", dura="dura", marrow="mrw")
SUBJECTS = (4, 5, 6, 18, 20, 38, *range(41, 55))


def fetch_legacy(root, record=False):
    """Pin the older single-subject fuzzy brain masks and T1/T2/PD references."""
    path = HERE/"legacy.json"
    manifest = dict(schema=1,snapshot="2026-10-07",shape=[181,217,181],dtype="<u2",scale=4095,
                    origin=[-.09,-.126,-.072],voxel=[.001]*3,files={}) if record else json.loads(path.read_text())
    aliases = {k:f"phantom_1.0mm_normal_{v}" for k,v in dict(wm="wht",gm="gry",csf="csf").items()}
    aliases |= {k:f"{k} ICBM normal 1mm pn0 rf0" for k in ("T1","T2","PD")}
    for key,alias in aliases.items():
        dest = root/f"legacy_{key}.raws.gz"
        if not dest.exists():
            part = dest.with_suffix(".part")
            body = urlencode(dict(do_download_alias=alias,format_value="raw_short",zip_value="gnuzip",
                                  download_for_real="[Start download!]"))
            subprocess.run(["curl","-fsSL","--retry","2","--max-time","180",ENDPOINT,
                            "--data",body,"-o",str(part)],check=True)
            part.replace(dest)
        raw = gzip.open(dest,"rb").read()
        if len(raw) != 2*181*217*181:
            raise ValueError(f"wrong legacy byte count: {dest}")
        digest = hashlib.sha256(raw).hexdigest()
        if record:
            manifest["files"][key] = dict(path=dest.name,sha256=digest,alias=alias)
        elif manifest["files"][key]["sha256"] != digest:
            raise ValueError(f"checksum mismatch: {dest}")
        print(f"verified legacy {key}: {digest}",flush=True)
    if record:
        path.write_text(json.dumps(manifest,indent=2)+"\n")
    (root/"legacy.json").write_text(json.dumps(manifest,indent=2)+"\n")


def fetch(subject, root, manifest_path, record=False):
    if subject not in SUBJECTS:
        raise ValueError(f"subject must be one of {SUBJECTS}")
    if record:
        manifest = dict(schema=1, subject=subject, snapshot=date.today().isoformat(),
                        source="https://brainweb.bic.mni.mcgill.ca/brainweb/anatomic_normal_20.html",
                        shape=[362, 434, 362], voxel=[0.0005]*3, dtype="<u2", scale=4095,
                        origin=[-0.09025, -0.12625, -0.07225], files={})
    else:
        manifest = json.loads(manifest_path.read_text())
        if manifest["subject"] != subject:
            raise ValueError("manifest belongs to another subject; use its matching snapshot")
    root.mkdir(parents=True, exist_ok=True)
    for tissue, alias in (ALIASES | {"t1w": "t1w"}).items():
        name = f"subject{subject:02d}_{alias}"
        path = root / f"{name}.raws.gz"
        if not path.exists():
            part = path.with_suffix(".part")
            subprocess.run(["curl", "-fsSL", "--retry", "2", "--max-time", "180", ENDPOINT,
                            "--data", f"do_download_alias={name}&format_value=raw_short&zip_value=gnuzip"
                            "&download_for_real=%5BStart+download%21%5D", "-o", str(part)], check=True)
            part.replace(path)
        raw = gzip.open(path, "rb").read()
        shape = (181, 256, 256) if tissue == "t1w" else tuple(manifest["shape"])
        if len(raw) != 2*shape[0]*shape[1]*shape[2]:
            raise ValueError(f"wrong byte count: {path}")
        digest = hashlib.sha256(raw).hexdigest()
        entry = dict(path=path.name, sha256=digest, alias=name)
        if record:
            manifest["files"][tissue] = entry
        elif digest != manifest["files"][tissue]["sha256"]:
            raise ValueError(f"checksum mismatch: {path}; do not silently refresh the snapshot")
        print(f"verified {name}: {digest}", flush=True)
    if record:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (root / f"subject{subject:02d}.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, default=4)
    parser.add_argument("--root", type=Path, default=HERE / "data")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--legacy", action="store_true", help="also fetch the older T1/T2/PD reference set")
    args = parser.parse_args()
    fetch(args.subject, args.root, args.manifest or HERE / f"subject{args.subject:02d}.json", args.record)
    if args.legacy:
        fetch_legacy(args.root,args.record)
