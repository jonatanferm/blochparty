"""Interactive 3D view of a BrainWeb head as blochparty sees it: 50 % iso-surfaces of the tissue
fractions (display only; the simulation uses the fractions), with the simulated 3 T slab (T2w
TSE, T1w GRE, B0 after shimming) placed where it sits in the head. Writes a self-contained
HTML file (three.js from a CDN) into outputs/, which is git-ignored: BrainWeb has no
redistribution licence, so keep the result local.

    uv run --with pillow python examples/anatomy_viewer.py   # then open outputs/anatomy_viewer.html
"""

import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from skimage import measure

import blochparty as bp

HERE = Path(__file__).resolve().parent


def build(ROOT, z0=60):
    t = time.time()
    t=time.time()
    head=bp.load_brainweb(4,ROOT,voxel=(1.5e-3,)*3, dtype=torch.float64)
    f={k:v.numpy() for k,v in head.fractions.items()}
    vox=np.array(head.voxel)*1e3; org=np.array(head.origin)*1e3  # (dz,dy,dx) mm ; (x,y,z) mm
    print("load",time.time()-t, head.shape)
    surfaces={
     "skin":      (1-f["background"], 0.5, 1.0),
     "skull":     (f["skull"]+f["marrow"], 0.5, 0.8),
     "csf":       (f["csf"], 0.5, 0.6),
     "grey":      (f["gm"]+f["wm"], 0.5, 0.8),
     "white":     (f["wm"], 0.5, 0.6),
     "vessels":   (f["vessels"], 0.4, 0.0),
    }
    out={}
    for name,(vol,level,sig) in surfaces.items():
        v=ndimage.gaussian_filter(vol,sig) if sig else vol
        verts,faces,_,_=measure.marching_cubes(v.astype(np.float32),level=level,step_size=1,allow_degenerate=False)
        # (z,y,x) index -> world mm (x,y,z)
        xyz=np.stack([org[0]+verts[:,2]*vox[2], org[1]+verts[:,1]*vox[1], org[2]+verts[:,0]*vox[0]],1)
        out[name]=(xyz,faces)
        print(name, len(xyz), "verts", len(faces), "tris")
    centre=out["skin"][0].mean(0); allv=np.concatenate([v for v,_ in out.values()])
    lo,hi=allv.min(0),allv.max(0)
    meshes={}
    for name,(xyz,faces) in out.items():
        q=np.round((xyz-lo)/(hi-lo)*65535).astype("<u2")
        meshes[name]={"pos":base64.b64encode(q.tobytes()).decode(),"idx":base64.b64encode(faces.astype("<u4").tobytes()).decode(),"n":len(xyz)}
    # simulated slab (same as the example): T2w TSE + B0 at the slab, on the image grid
    head1=bp.load_brainweb(4,ROOT,voxel=(1e-3,)*3)
    fm=bp.anatomy_field_maps(head1,field_T=3.0)
    ph=bp.to_phantom(head1,field_T=3.0,fov_region=(slice(z0,z0+2),slice(None),slice(None)),spins_per_voxel=(2,2,2),field_maps=fm,
                     variability={"t1":.05,"t2":.05,"pd":.02,"length":.01})
    c=ph.pos.mean(0).numpy()*1e3; ph.pos=ph.pos-ph.pos.mean(0)
    n,fov=192,0.24
    imgs={}
    for key,seq in (("t2w",bp.tse_2d(n,fov,etl=8,esp=12e-3,tr=4.0,n_dummy=1)),("t1w",bp.gre_2d(n,fov,flip_deg=30.0,tr=20e-3,te=4.6e-3,n_dummy=60))):
        s=bp.simulate(ph.to(torch.float32),seq.to(torch.float32),engine="gpu").signal.to(torch.complex128)
        im=bp.cartesian_recon(s,seq)[0].abs().numpy(); imgs[key]=im
    # grid convention: image pixel (i, j) at y = (i - n/2 + 1/2?) -> use grid_positions
    gp=bp.grid_positions((n,n),fov).numpy()  # (n,n,3) (x,y,z) m
    print("grid x range", gp[0,0], gp[-1,-1])
    # B0 on the same grid (tissue + shim), at the slab mid-plane
    pts=torch.tensor(np.stack([gp[...,0].ravel()+c[0]*1e-3, gp[...,1].ravel()+c[1]*1e-3, np.full(n*n,c[2]*1e-3)],1),dtype=torch.float64)
    from blochparty.maps import sample
    b0=sum(sample(m.b0[None].double(),torch.tensor(m.origin,dtype=torch.float64),torch.tensor(m.spacing,dtype=torch.float64),pts)[:,0] for m in fm).numpy().reshape(n,n)
    inside=(1-head1.fractions["background"]).numpy()
    mask=sample(torch.tensor(inside)[None].double(),torch.tensor(head1.origin,dtype=torch.float64),torch.tensor(head1.voxel[::-1],dtype=torch.float64),pts)[:,0].numpy().reshape(n,n)>.5
    def png(a):
        import io

        from PIL import Image
        buf=io.BytesIO(); Image.fromarray(a).save(buf,format="PNG"); return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()
    def gray(im):
        v=np.clip(im/np.percentile(im,99.5),0,1); a=(v*255).astype(np.uint8); return np.dstack([a,a,a,np.where(im>0.04*im.max(),255,0).astype(np.uint8)])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    b=np.clip(b0/150,-1,1); rgba=(cm.RdBu_r((b+1)/2)*255).astype(np.uint8); rgba[...,3]=np.where(mask,255,0)
    images={"t2w":png(gray(imgs["t2w"])),"t1w":png(gray(imgs["t1w"])),"b0":png(rgba)}
    data={"lo":lo.tolist(),"hi":hi.tolist(),"centre":centre.tolist(),"meshes":meshes,
          "slab":{"centre_mm":c.tolist(),"fov_mm":fov*1e3,"n":n,"z_mm":float(c[2]),"images":images,
                  "b0_range_hz":[float(b0[mask].min()),float(b0[mask].max())],"spins":len(ph)},
          "grid_first":gp[0,0].tolist(),"grid_last":gp[-1,-1].tolist()}
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=HERE.parent / "validation" / "anatomy" / "data")
    ap.add_argument("--z", type=int, default=60)
    ap.add_argument("--out", type=Path, default=HERE.parent / "outputs" / "anatomy_viewer.html")
    args = ap.parse_args()
    data = build(str(args.root), args.z)
    html = (HERE / "anatomy_viewer.html").read_text().replace("/*DATA*/null", json.dumps(data))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"wrote {args.out} ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
