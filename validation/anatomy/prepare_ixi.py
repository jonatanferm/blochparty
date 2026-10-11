"""Segment IXI 3 T heads into BrainWeb's 12 tissue classes on BrainWeb's 1 mm MNI grid (for `load_ixi`).

Per subject: the T1 is registered (affine, local NCC, 4 then 2 mm) to a T1-like image made from BrainWeb
subject 04's fractions, T1 and T2 are resampled onto BrainWeb's grid (they share world coordinates: one
session), and an EM with one Gaussian per class in (log T1, log T2) and a cubic-polynomial log-bias field
per channel assigns the classes. Priors: BrainWeb 04's head classes smoothed 2 mm; the intracranial total
smoothed 2 mm and split among CSF / grey / white by their 8 mm proportions, so the subject's intensities,
not BrainWeb's sulci, place the cortex. Outside the skull the head classes come from rules on the
bias-corrected intensities relative to the subject's white matter (the EM swapped them in the thin layers):
fat bright in T1 and T2 (inside the bone shell: marrow), bone dark in both, soft tissue between: within 8 mm
of the surface it is BrainWeb's "muscle/skin" (skin), which `layer_scalp` splits into dermis and
subcutaneous fat as for BrainWeb's heads; deeper it is muscle (temporalis, neck). Posteriors are stored as fractions (uint8 / 255) with the affine (BrainWeb mm -> IXI world
mm), age and sex (IXI.xls; nothing else of it is kept) and the tissue volumes in the subject's own size.
Known gap: the T2 covers ~134 mm (MNI z <= ~92), so the scalp and skull above it are left as background
(the brain ends at z ~75; the scored slices lie below ~50).

    uv run --with xlrd python validation/anatomy/prepare_ixi.py [--subjects 12,13 | --count 40] [--root ...]
"""
import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as Fn

import blochparty as bp

HERE = Path(__file__).resolve().parent
CLASSES = ("background", "csf", "gm", "wm", "fat", "muscle", "skin", "skull", "vessels", "connective", "dura", "marrow")
BRAIN = ("csf", "gm", "wm")
# T1-weighted intensities for the registration target (relative; only the pattern matters for local NCC)
T1W = {"background": 0., "csf": .25, "gm": .62, "wm": 1., "fat": 1.15, "muscle": .45, "skin": .7, "skull": .08,
       "vessels": .5, "connective": .55, "dura": .3, "marrow": .85}
ATLAS = 4


def demographics(root):
    """{IXI id: {"sex": "m"/"f", "age": years}} from IXI.xls (cached as demographics.json)."""
    cache = root / "demographics.json"
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    import xlrd  # one-off: uv run --with xlrd

    sheet = xlrd.open_workbook(str(root / "IXI.xls")).sheet_by_index(0)
    head = [str(c) for c in sheet.row_values(0)]
    i_id, i_sex, i_age = head.index("IXI_ID"), head.index("SEX_ID (1=m, 2=f)"), head.index("AGE")
    out = {}
    for r in range(1, sheet.nrows):
        row = sheet.row_values(r)
        if row[i_age] != "":
            out[int(row[i_id])] = {"sex": "m" if int(row[i_sex]) == 1 else "f", "age": round(float(row[i_age]), 2)}
    cache.write_text(json.dumps(out, indent=0))
    return out


def load_ras(path):
    im = nib.as_closest_canonical(nib.load(path))
    v = im.get_fdata(dtype=np.float32)
    return v / np.percentile(v[v > np.percentile(v, 50)], 99), im.affine


def sample(vol, aff_vol, coords_mm):
    """trilinear sample of vol (x, y, z; world affine aff_vol) at world coordinates (..., 3) mm"""
    inv = torch.linalg.inv(aff_vol)
    vox = coords_mm @ inv[:3, :3].T + inv[:3, 3]
    n = torch.tensor(vol.shape, dtype=coords_mm.dtype)
    g = 2 * vox / (n - 1) - 1
    v = torch.as_tensor(vol)[None, None].permute(0, 1, 4, 3, 2)
    return Fn.grid_sample(v, g[None], align_corners=True, padding_mode="zeros")[0, 0]


def lncc(a, b, w=7):
    k = torch.ones(1, 1, w, w, w) / w ** 3
    f = lambda t: Fn.conv3d(t[None, None], k, padding=w // 2)[0, 0]
    ma, mb = f(a), f(b)
    va, vb, cab = f(a * a) - ma ** 2, f(b * b) - mb ** 2, f(a * b) - ma * mb
    return -(cab ** 2 / (va * vb + 1e-4)).mean()


def register(fixed, aff_f, moving, aff_m, levels=(4., 2.), iters=(150, 100)):
    """affine M (fixed world mm -> moving world mm) maximising local NCC"""
    com = lambda v, a: a[:3, :3] @ np.array(ndimage.center_of_mass(v > 0.1)) + a[:3, 3]
    A = torch.eye(3, requires_grad=True)
    t = torch.tensor(com(moving, aff_m) - com(fixed, aff_f), dtype=torch.float32, requires_grad=True)
    aff_m_t = torch.tensor(aff_m, dtype=torch.float32)
    for step, (lv, it) in enumerate(zip(levels, iters)):
        s = lv / aff_f[0, 0]
        fz = ndimage.zoom(ndimage.gaussian_filter(fixed, s / 2), 1 / s, order=1)
        grid = np.stack(np.meshgrid(*[np.arange(n) * lv for n in fz.shape], indexing="ij"), -1) + aff_f[:3, 3]
        grid_t = torch.tensor(grid, dtype=torch.float32).permute(2, 1, 0, 3)
        fixed_t = torch.tensor(fz).permute(2, 1, 0)
        msm = ndimage.gaussian_filter(moving, lv / 2)
        oa, ot = torch.optim.Adam([A], lr=2e-3 if step else 4e-3), torch.optim.Adam([t], lr=0.5)
        for _ in range(it):
            oa.zero_grad(); ot.zero_grad()
            loss = lncc(fixed_t, sample(msm, aff_m_t, grid_t @ A.T + t))
            loss.backward(); oa.step(); ot.step()
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = A.detach().numpy(), t.detach().numpy()
    return M, -float(loss)


def onto(vol, aff_vol, M, shape_zyx, origin_mm):
    """vol sampled on BrainWeb's 1 mm grid (z, y, x) through M"""
    nz, ny, nx = shape_zyx
    g = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1) + np.asarray(origin_mm)
    pts = torch.tensor(g, dtype=torch.float32).permute(2, 1, 0, 3) @ torch.tensor(M[:3, :3], dtype=torch.float32).T \
        + torch.tensor(M[:3, 3], dtype=torch.float32)
    return sample(vol, torch.tensor(aff_vol, dtype=torch.float32), pts).numpy()


def em(y, prior, basis, iters=30):
    """EM: y (C, N) log intensities, prior (K, N), basis (N, T) for the log-bias field; posteriors (K, N)"""
    C, N = y.shape
    K = prior.shape[0]
    bias = np.zeros_like(y)
    post = prior.copy()
    for it in range(iters):
        yc = y - bias
        w = post.sum(1) + 1e-6
        mu = (post @ yc.T) / w[:, None]
        cov = np.empty((K, C, C))
        for k in range(K):
            d = yc - mu[k][:, None]
            cov[k] = (post[k] * d) @ d.T / w[k] + 1e-4 * np.eye(C)
        logp = np.empty((K, N), np.float32)
        for k in range(K):
            d = yc - mu[k][:, None]
            inv = np.linalg.inv(cov[k])
            logp[k] = np.log(prior[k] + 1e-12) - 0.5 * np.einsum("in,ij,jn->n", d, inv, d) - 0.5 * np.log(np.linalg.det(cov[k]))
        logp -= logp.max(0)
        post = np.exp(logp)
        post /= post.sum(0)
        if it % 3 == 2:  # bias: weighted least squares of each channel's residual over the tissue classes
            for c in range(C):
                prec = np.array([np.linalg.inv(cov[k])[c, c] for k in range(K)])
                wt = (post[1:] * prec[1:, None]).sum(0)
                res = (post[1:] * prec[1:, None] * (y[c] - mu[1:, c][:, None])).sum(0) / (wt + 1e-9)
                sw = np.sqrt(wt)
                coef, *_ = np.linalg.lstsq(basis * sw[:, None], res * sw, rcond=None)
                bias[c] = basis @ coef
    return post, mu, bias


def prepare(ixi_id, root, atlas, fixed2):
    t0 = time.time()
    t1p = next((root / "T1").glob(f"IXI{ixi_id:03d}-HH-*-T1.nii.gz"))
    t2p = next((root / "T2").glob(f"IXI{ixi_id:03d}-HH-*-T2.nii.gz"))
    t1, a1 = load_ras(t1p)
    t2, a2 = load_ras(t2p)
    M, score = register(*fixed2, t1, a1)
    shape, origin = atlas.shape, np.array(atlas.origin) * 1e3
    chans = [onto(t1, a1, M, shape, origin), onto(t2, a2, M, shape, origin)]
    valid = (onto(np.ones_like(t1), a1, M, shape, origin) > 0.99) & (onto(np.ones_like(t2), a2, M, shape, origin) > 0.99)
    fr = {k: atlas.fractions[k].numpy().astype(np.float32) for k in CLASSES}
    intra = ndimage.gaussian_filter(sum(fr[k] for k in BRAIN), 2.0)
    share = {k: ndimage.gaussian_filter(fr[k], 8.0) + 0.02 for k in BRAIN}
    tot = sum(share.values())
    prior = np.stack([intra * share[k] / tot if k in BRAIN else ndimage.gaussian_filter(fr[k], 2.0) for k in CLASSES]) + 1e-3
    prior /= prior.sum(0)
    idx = np.nonzero(valid)
    y = np.stack([np.log(np.maximum(c[idx], 1e-3)) for c in chans])
    zz, yy, xx = (np.linspace(-1, 1, n, dtype=np.float32) for n in shape)
    terms = [(i, j, k) for i in range(4) for j in range(4 - i) for k in range(4 - i - j)]
    basis = np.stack([zz[idx[0]] ** i * yy[idx[1]] ** j * xx[idx[2]] ** k for i, j, k in terms], 1)
    post, mu, bias = em(y, prior[:, idx[0], idx[1], idx[2]], basis)
    frac = np.zeros((len(CLASSES),) + tuple(shape), np.float32)
    frac[:, idx[0], idx[1], idx[2]] = post
    frac[0][~valid] = 1
    # outside the skull, rules on the bias-corrected intensities relative to the subject's white matter (the
    # EM's head classes swap identities where BrainWeb's affine-aligned priors miss the thin layers): fat bright
    # in both (inside the bone shell: marrow), bone dark in both, soft tissue between (skin to 8 mm, then muscle)
    corr = []
    for c in range(2):
        v = np.zeros(shape, np.float32)
        v[idx] = np.exp(y[c] - bias[c])
        corr.append(v)
    wmref = [np.median(v[frac[3] > .9]) for v in corr]
    r1, r2 = corr[0] / wmref[0], corr[1] / wmref[1]
    sig = lambda x: 1 / (1 + np.exp(-x))
    intra = ndimage.binary_fill_holes(ndimage.binary_closing(frac[[1, 2, 3, 8, 10]].sum(0) > .5, iterations=2))
    head = ndimage.binary_fill_holes(ndimage.binary_closing((r1 > .25) | (r2 > .25) | intra, iterations=3))
    p_fat = sig((r1 - .8) / .08) * sig((r2 - 1.2) / .15)
    p_bone = sig((.35 - r1) / .06) * sig((.45 - r2) / .08)
    p_soft = np.clip(1 - p_fat - p_bone, 0, 1)
    shell = ndimage.binary_closing(p_bone > .5, iterations=2) & ~intra
    depth = ndimage.distance_transform_edt(head)
    out_mask = head & ~intra & valid
    i = {c: CLASSES.index(c) for c in CLASSES}
    keep_in = ("csf", "gm", "wm", "vessels", "dura")
    for c in CLASSES:
        if c not in keep_in:
            frac[i[c]][out_mask] = 0
    for c in keep_in:
        frac[i[c]][out_mask] = 0
    frac[i["marrow"]][out_mask] = (p_fat * shell)[out_mask]
    frac[i["fat"]][out_mask] = (p_fat * ~shell)[out_mask]
    frac[i["skull"]][out_mask] = p_bone[out_mask]
    frac[i["skin"]][out_mask] = (p_soft * (depth <= 8))[out_mask]  # BrainWeb's "muscle/skin": layer_scalp splits it
    frac[i["muscle"]][out_mask] = (p_soft * (depth > 8))[out_mask]
    frac[i["background"]][~head] = 1
    for c in CLASSES:
        if c != "background":
            frac[i[c]][~head] = 0
    frac /= frac.sum(0, keepdims=True)
    scale = abs(np.linalg.det(M[:3, :3])) * 1e-3  # mL per MNI voxel in the subject's own size
    vol = {c: round(float(frac[i].sum()) * scale, 1) for i, c in enumerate(CLASSES) if c != "background"}
    return frac, M, score, vol, {"t1": t1p.name, "t2": t2p.name}, round(time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=HERE / "data" / "ixi")
    ap.add_argument("--subjects", default=None, help="comma-separated IXI ids")
    ap.add_argument("--count", type=int, default=None, help="this many subjects spread over the ages")
    args = ap.parse_args()
    demo = demographics(args.root)
    ids = sorted(int(p.name[3:6]) for p in (args.root / "T1").glob("*-HH-*-T1.nii.gz"))
    ids = [i for i in ids if i in demo and any((args.root / "T2").glob(f"IXI{i:03d}-HH-*"))]
    if args.subjects:
        ids = [int(s) for s in args.subjects.split(",")]
    elif args.count:
        order = sorted(ids, key=lambda i: demo[i]["age"])
        ids = sorted(order[int(j)] for j in np.linspace(0, len(order) - 1, args.count).round())
    out = args.root / "prepared"
    out.mkdir(exist_ok=True)
    atlas = bp.load_brainweb(ATLAS, HERE / "data", voxel=(1e-3,) * 3)
    a2 = bp.load_brainweb(ATLAS, HERE / "data", voxel=(2e-3,) * 3)
    img = sum(T1W[k] * v for k, v in a2.fractions.items()).numpy().astype(np.float32).transpose(2, 1, 0)
    aff = np.diag([2., 2., 2., 1.])
    aff[:3, 3] = np.array(a2.origin) * 1e3
    for i in ids:
        dest = out / f"IXI{i:03d}.npz"
        if dest.exists():
            continue
        frac, M, score, vol, files, secs = prepare(i, args.root, atlas, (img, aff))
        meta = {"ixi_id": i, **demo[i], "atlas": f"brainweb{ATLAS:02d}", "classes": CLASSES, "affine_mm": M.tolist(),
                "lncc": round(score, 3), "volumes_ml": vol, "files": files, "origin": list(atlas.origin),
                "voxel": list(atlas.voxel), "shape": list(atlas.shape)}
        np.savez_compressed(dest, fractions=np.round(frac * 255).astype(np.uint8), meta=json.dumps(meta))
        print(f"IXI{i:03d} age {demo[i]['age']:.0f} {demo[i]['sex']}: GM {vol['gm']:.0f} WM {vol['wm']:.0f} CSF {vol['csf']:.0f} mL,"
              f" lncc {score:.3f}, {secs} s", flush=True)
