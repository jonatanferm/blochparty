"""QSM forward comparison against pinned, independently published Python code.

uv run python validation/anatomy/anatomy_qsm.py --fetch-reference
uv run python validation/anatomy/anatomy_qsm.py --root validation/anatomy/data/qsm2

Without --root, only synthetic checks run; no real-head validation is claimed.
This public qsm-forward implementation is a separate reference based on the paper,
not the authors' original restricted MATLAB package. The two selected functions
are executed unchanged from the hash-verified local source, with NumPy only.
"""
import argparse
import ast
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch

from blochparty import GAMMA_BAR, load_qsm2, susceptibility_to_b0

HERE = Path(__file__).parent
REFERENCE = HERE/"data"/"qsm-forward-reference.py"


def reference_function(fetch=False):
    manifest = json.loads((HERE/"qsm_reference.json").read_text())
    if not REFERENCE.exists() and fetch:
        raw = urllib.request.urlopen(manifest["url"],timeout=60).read()
        if hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
            raise ValueError("downloaded reference checksum mismatch")
        REFERENCE.parent.mkdir(parents=True,exist_ok=True)
        REFERENCE.write_bytes(raw)
    raw = REFERENCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
        raise ValueError("reference checksum mismatch")
    tree = ast.parse(raw)
    tree.body = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in manifest["functions"]]
    if len(tree.body) != 2:
        raise ValueError("reference functions missing")
    namespace = {"np":np}
    exec(compile(tree,str(REFERENCE),"exec"),namespace)
    return namespace["generate_field"]


def compare(chi, voxel, background, mask, reference, field_T=3.):
    # qsm-forward uses x/y/z array order and ppm. Its z-oriented isotropic
    # kernel is comparable to ours; transverse anisotropic/oblique conventions
    # need a separate audit and are deliberately excluded from this check.
    if not np.allclose(voxel,voxel[0],rtol=0,atol=1e-12):
        raise ValueError("this reference comparison requires isotropic voxels")
    # Its padding is filled from the last corner: require that to be exterior.
    corner = float(chi[-1,-1,-1])
    if abs(corner-background) > 1e-12:
        raise ValueError("reference pads with corner chi; corner must equal declared background")
    source = chi.numpy().transpose(2,1,0)*1e6
    ref = reference(source,voxel_size=np.array(voxel[::-1])*1e3,B0_dir=[0,0,1])
    ref = torch.from_numpy(ref.transpose(2,1,0).copy())*(GAMMA_BAR*field_T*1e-6)
    # Match the encoded corner exactly (e.g. float32 9.2 ppm differs slightly
    # from decimal 9.2), otherwise padding conventions introduce a small field.
    actual = susceptibility_to_b0(chi,voxel,field_T,padding=2,background=corner)
    # Reference sets D(0)=1/3; ours sets D(0)=0. Remove only an ROI constant,
    # not any spatial fit, to compare physically observable relative fields.
    raw_difference = actual-ref
    offset = raw_difference[mask].mean()
    difference = raw_difference-offset
    selected = difference[mask]
    return dict(roi_voxels=int(mask.sum()),offset_Hz=float(offset),
                rms_Hz=float(selected.square().mean().sqrt()),max_Hz=float(selected.abs().max()),
                full_volume_rms_Hz=float(difference.square().mean().sqrt()),field_T=field_T,
                padding_background_ppm=corner*1e6)


def run(root=None,fetch=False,resolution="1mm"):
    reference = reference_function(fetch)
    results = {"reference":"qsm-forward 92149383b0f43f47d9a351ceb4f525c0c4ee8299",
               "original_MATLAB_crosscheck":"pending authorized original package"}
    generator = torch.Generator().manual_seed(44)
    for shape in ((17,19,21),(18,20,22)):
        source = torch.full(shape,9.2e-6,dtype=torch.float64)
        source[3:-3,3:-3,3:-3] = torch.randn(tuple(n-6 for n in shape),generator=generator,dtype=torch.float64)*.1e-6
        mask = source != 9.2e-6
        result = compare(source,(.001,)*3,9.2e-6,mask,reference)
        if result["max_Hz"] > 1e-9:
            raise AssertionError(result)
        results[f"synthetic_{shape}"] = result
    if root is not None:
        a = load_qsm2(root,resolution=resolution,dtype=torch.float64)
        results["head"] = compare(a.chi,a.voxel,a.chi_background,a.maps["brain_mask"] > .5,reference)
        results["head"]["resolution"] = resolution
        if results["head"]["max_Hz"] > 1e-8:
            raise AssertionError(results["head"])
    else:
        results["head"] = "not run; authorized data unavailable"
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-reference",action="store_true",help="download only the pinned public source code")
    parser.add_argument("--root",type=Path,help="local head maps and reviewed qsm2.json")
    parser.add_argument("--resolution",choices=("native","0.64mm","1mm"),default="1mm")
    args = parser.parse_args()
    results = run(args.root,args.fetch_reference,args.resolution)
    print(json.dumps(results,indent=2))
    output = HERE/"data"/"qsm_results.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(results,indent=2)+"\n")
