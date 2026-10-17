"""Stream a fastMRI brain archive and keep a capped subset of contrasts (ROADMAP M5, the realism
number).

The archives (`brain_multicoil_*.tar.xz`, ~90 GB each) do not fit next to everything else, so
the download is read as a stream (curl -> Python's tarfile, xz on the fly) and only members whose
name carries a wanted contrast are written, each contrast up to its own cap; the stream stops
once every cap is reached (or free disk falls below ``--min-free``). Each kept volume is opened
and described in ``manifest.json``: bytes, SHA-256, field strength, scanner, protocol, TR/TE/TI,
flip angle, matrix, coils, slices. A file already present with the recorded size is skipped, so
an interrupted run can be restarted.

The download URL is a signed, expiring link from the fastMRI access email: pass it in a curl
config file (``url = "..."``, mode 600) so it never appears on a command line, and never commit
it. Data stay outside the repository (no redistribution; fastMRI data-sharing agreement).

    uv run python validation/realism/fetch_fastmri.py --curl-config ~/.fastmri_val0.curl \\
        --out ~/data/fastmri_brain --cap AXT2=15 --cap AXT1=4 --cap AXT1PRE=4 --cap AXFLAIR=7

Since 2026-10-14 the kept form is compact: ``--compact 128 --min-field 2`` stores each 3 T volume's central 128 x 128
coil k-space per slice (~40 MB) and deletes the raw file; the cap still counts the raw bytes taken from the archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np

GB = 1e9


def contrast(name: str) -> str | None:
    """'file_brain_AXT2_200_2000019.h5' -> 'AXT2'."""
    stem = Path(name).name
    if not (stem.startswith("file_brain_") and stem.endswith(".h5")):
        return None
    return stem.removeprefix("file_brain_").split("_")[0]


def describe(path: Path) -> dict:
    """What the realism test needs to know about one volume, from its ISMRMRD header."""
    import h5py
    import ismrmrd

    with h5py.File(path, "r") as f:
        shape = list(f["kspace"].shape)  # (slices, coils, ky, kx)
        xml = f["ismrmrd_header"][()]
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
    h = ismrmrd.xsd.CreateFromDocument(xml)
    sys, seq, enc = h.acquisitionSystemInformation, h.sequenceParameters, h.encoding[0]
    first = lambda v: float(v[0]) if v else None
    return {"kspace_shape": shape, "field_T": sys.systemFieldStrength_T, "model": sys.systemModel,
            "vendor": sys.systemVendor, "protocol": h.measurementInformation.protocolName
            if h.measurementInformation else None, "TR_ms": first(seq.TR), "TE_ms": first(seq.TE),
            "TI_ms": first(seq.TI), "flip_deg": first(seq.flipAngle_deg), "echo_spacing_ms": first(seq.echo_spacing),
            "encoded": [enc.encodedSpace.matrixSize.x, enc.encodedSpace.matrixSize.y],
            "recon": [enc.reconSpace.matrixSize.x, enc.reconSpace.matrixSize.y],
            "fov_mm": [enc.encodedSpace.fieldOfView_mm.x, enc.encodedSpace.fieldOfView_mm.y,
                       enc.encodedSpace.fieldOfView_mm.z], "attrs": attrs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curl-config", type=Path, required=True, help="curl config holding the signed url")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cap", action="append", default=[], help="CONTRAST=GB, e.g. AXT2=15 (repeatable)")
    ap.add_argument("--min-free", type=float, default=30.0, help="stop when free disk falls below this [GB]")
    ap.add_argument("--compact", type=int, default=None,
                    help="keep each volume as its central N x N coil k-space per slice (realism.coil_kspace) in a .npz "
                         "and delete the raw file: ~40 MB a scan instead of ~500 (the realism score and the transfer "
                         "test use nothing else)")
    ap.add_argument("--min-field", type=float, default=None, help="drop volumes below this field strength [T] after reading their header")
    args = ap.parse_args()
    if args.compact:
        import realism as R
    caps = {k: float(v) * GB for k, v in (c.split("=") for c in args.cap)}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    kept = {k: sum(e["bytes"] for e in manifest.values() if e["contrast"] == k) for k in caps}
    seen = {}
    curl = subprocess.Popen(["curl", "-fsS", "--retry", "3", "-K", str(args.curl_config)], stdout=subprocess.PIPE)
    t0, streamed = time.time(), 0
    try:
        with tarfile.open(fileobj=curl.stdout, mode="r|xz") as tar:
            for member in tar:
                streamed += member.size
                kind = contrast(member.name)
                seen[kind] = seen.get(kind, 0) + 1
                if kind not in caps or kept[kind] >= caps[kind] or not member.isfile():
                    if all(kept[k] >= caps[k] for k in caps):
                        break
                    continue
                if shutil.disk_usage(args.out).free < args.min_free * GB + member.size:
                    print("stopping: free disk below", args.min_free, "GB", flush=True)
                    break
                name = Path(member.name).name
                dest = args.out / name
                if name in manifest and ((dest.exists() and dest.stat().st_size == manifest[name]["bytes"])
                                         or ("compact" in manifest[name] and (args.out / manifest[name]["compact"]).exists())):
                    continue
                digest = hashlib.sha256()
                part = dest.with_suffix(".part")
                with tar.extractfile(member) as src, open(part, "wb") as out:
                    while chunk := src.read(1 << 24):
                        digest.update(chunk)
                        out.write(chunk)
                part.replace(dest)
                entry = {"contrast": kind, "bytes": member.size, "sha256": digest.hexdigest()}
                try:
                    entry |= describe(dest)
                except Exception as err:  # keep the file, record why it could not be read
                    entry["error"] = f"{type(err).__name__}: {err}"
                if args.min_field is not None and "error" not in entry and (entry.get("field_T") or 0) < args.min_field:
                    dest.unlink()
                    print(f"{name}: {entry.get('field_T')} T, dropped", flush=True)
                    continue
                if args.compact and "error" not in entry:
                    low, info = R.coil_kspace(dest, args.compact)
                    compact = dest.with_suffix(".npz")
                    np.savez_compressed(compact, kspace=low, info=json.dumps(info | {"compact": args.compact}))
                    dest.unlink()
                    entry["compact"] = compact.name
                    entry["compact_bytes"] = compact.stat().st_size
                manifest[name] = entry
                kept[kind] += member.size
                manifest_path.write_text(json.dumps(manifest, indent=1))
                rate = streamed / GB / max(time.time() - t0, 1e-9) * 60
                print(f"{name}: {member.size / GB:.2f} GB, {entry.get('field_T')} T; kept "
                      + ", ".join(f"{k} {kept[k] / GB:.1f}/{caps[k] / GB:.0f}" for k in caps)
                      + f"; streamed {streamed / GB:.1f} GB ({rate:.1f} GB/min)", flush=True)
    finally:
        curl.kill()
        curl.stdout.close()
        curl.wait()
        for part in args.out.glob("*.part"):
            part.unlink()
    print("members seen by contrast:", seen, flush=True)
    print("done:", {k: round(kept[k] / GB, 2) for k in caps}, "GB", flush=True)


if __name__ == "__main__":
    main()
