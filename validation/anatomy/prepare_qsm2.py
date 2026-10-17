"""Record checksums for an authorized local QSM Challenge 2.0 source directory.

No network access or login automation. This records provenance, not independent
proof that arbitrary local files are the official release. Review qsm2.json before
using it. The access-restricted collection is doi:10.34973/m20r-jt17, version 1.
"""
import argparse
import json
from datetime import date
from pathlib import Path

from blochparty.anatomy import _file_sha256, load_qsm2


def prepare(root, chi, background_ppm, field_T=7., spatial_unit="mm"):
    root = Path(root).resolve()
    paths = dict(chi=chi, M0="maps/M0.nii.gz", R1="maps/R1.nii.gz",
                 R2star="maps/R2star.nii.gz", segmentation="masks/SegmentedModel.nii.gz",
                 brain_mask="masks/BrainMask.nii.gz")
    manifest = dict(schema=1, source="https://doi.org/10.34973/m20r-jt17",
                    release="DSC_3015069.02_542_v1", snapshot=date.today().isoformat(),
                    provenance="user-supplied local files; hashes recorded, not publisher-verified",
                    chi_unit="ppm", chi_background_ppm=background_ppm,
                    spatial_unit=spatial_unit, map_field_T=field_T, files={})
    for key, relative in paths.items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("paths must stay inside the source directory")
        manifest["files"][key] = dict(path=str(path.relative_to(root)),sha256=_file_sha256(path))
    path = root/"qsm2.json"
    # Never overwrite a reviewed pin, even if the supplied data has changed.
    with path.open("x") as stream:
        stream.write(json.dumps(manifest,indent=2)+"\n")
    anatomy = load_qsm2(root,resolution="native")
    print(f"Recorded {path}; validated grid {anatomy.shape}, voxel={anatomy.voxel}, origin={anatomy.origin}")
    print("Review provenance and susceptibility reference before simulation.")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path,help="authorized extracted data directory containing maps, masks, chimodel")
    parser.add_argument("--chi",default="chimodel/ChiModelMIX.nii",help="relative path of selected susceptibility model")
    parser.add_argument("--background-ppm",type=float,required=True,help="surrounding air value in the selected chi reference")
    parser.add_argument("--field-T",type=float,default=7.,help="acquisition field of R1/R2star maps, not the simulation field")
    parser.add_argument("--spatial-unit",choices=("mm","meter","micron"),default="mm")
    args = parser.parse_args()
    prepare(args.root,args.chi,args.background_ppm,args.field_T,args.spatial_unit)
