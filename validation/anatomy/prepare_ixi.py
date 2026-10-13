"""Segment IXI 3 T heads into BrainWeb's 12 tissue classes on BrainWeb's 1 mm MNI grid (for `load_ixi`).

Per subject: the T1 is registered (affine, local NCC, 4 then 2 mm) to a T1-like image made from BrainWeb
subject 04's fractions, T1 and T2 are resampled onto BrainWeb's grid (they share world coordinates: one
session, though the head moves between them: the T2 is rigidly aligned to the T1 by the correlation of their
gradient magnitudes), and an EM with one Gaussian per class in (log T1, log T2) and a cubic-polynomial log-bias field
per channel assigns the classes. Priors: BrainWeb 04's head classes smoothed 2 mm; the intracranial total
smoothed 2 mm and split among CSF / grey / white by their 8 mm proportions, so the subject's intensities,
not BrainWeb's sulci, place the cortex; inside the brain CSF / grey / white partial volumes are then unmixed from
the bias-corrected T1 and T2 against local pure intensities (``unmix_partial_volume``; the EM posterior is close
to a step in intensity and doubled the CSF in boundary voxels), and every CSF-containing voxel's CSF share is set
from the T2 against the subject's ventricular CSF (``refine_csf``). Outside the skull the head classes come from rules on the
bias-corrected intensities relative to the subject's white matter (the EM swapped them in the thin layers):
bone dark in both; fat by unmixing T2 between soft tissue (0.9 x white matter) and the subject's own pure
fat (the median T2 of T1-bright head tissue), with a bright T1 (inside the bone shell: marrow); soft tissue the rest,
unmixed by its T2 between skin and muscle (``soft_references``). The head is its largest connected piece. Posteriors are stored as fractions (uint8 / 255) with the affine (BrainWeb mm -> IXI world
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
EM_CLASSES = ("background", "csf", "gm", "wm", "fat", "muscle", "skin", "skull", "vessels", "connective", "dura", "marrow")
CLASSES = EM_CLASSES + ("eye",)  # the eye (vitreous) from the rules outside the skull: BrainWeb has no eye prior
# the orbits in MNI space [mm]: |x| 10-55, y 30-85, z -60 to -10 (generous boxes around both globes)
ORBIT_MM = ((10., 55.), (30., 85.), (-60., -10.))
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


def onto(vol, aff_vol, M, shape_zyx, origin_mm, order=1):
    """vol sampled on BrainWeb's 1 mm grid (z, y, x) through M; ``order`` 1 trilinear, 3 cubic B-spline (keeps the
    fine structure: trilinear keeps ~0.6 of the power at the realism images' Nyquist, the spline ~0.9)"""
    nz, ny, nx = shape_zyx
    g = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1) + np.asarray(origin_mm)
    pts = torch.tensor(g, dtype=torch.float32).permute(2, 1, 0, 3) @ torch.tensor(M[:3, :3], dtype=torch.float32).T \
        + torch.tensor(M[:3, 3], dtype=torch.float32)
    if order == 1:
        return sample(vol, torch.tensor(aff_vol, dtype=torch.float32), pts).numpy()
    inv = np.linalg.inv(np.asarray(aff_vol, dtype=np.float64))
    vox = pts.numpy().astype(np.float64) @ inv[:3, :3].T + inv[:3, 3]  # (z, y, x, 3): indices into vol's (x, y, z)
    return ndimage.map_coordinates(np.asarray(vol, dtype=np.float32), np.moveaxis(vox, -1, 0).astype(np.float32),
                                   order=order, mode="constant", cval=0.0).astype(np.float32)


def _rot(angles):
    a, b, c = angles
    one, zero = torch.ones(()), torch.zeros(())
    rx = torch.stack([torch.stack([one, zero, zero]), torch.stack([zero, torch.cos(a), -torch.sin(a)]), torch.stack([zero, torch.sin(a), torch.cos(a)])])
    ry = torch.stack([torch.stack([torch.cos(b), zero, torch.sin(b)]), torch.stack([zero, one, zero]), torch.stack([-torch.sin(b), zero, torch.cos(b)])])
    rz = torch.stack([torch.stack([torch.cos(c), -torch.sin(c), zero]), torch.stack([torch.sin(c), torch.cos(c), zero]), torch.stack([zero, zero, one])])
    return rz @ ry @ rx


def align_t2(t1_grid, t2, a2, M, shape_zyx, origin_mm, iters=(120, 60)):
    """Rigid correction R (the subject's world mm -> world mm, about the head's centre) so that T2 sampled at
    R(M x) matches T1 on the grid: IXI's T1 and T2 are separate scans and the head moves between them
    (up to 4+ mm). Cost: correlation of their gradient magnitudes inside the head (contrast-independent)."""
    def gradmag(v):
        gz, gy, gx = v[2:, 1:-1, 1:-1] - v[:-2, 1:-1, 1:-1], v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1], v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]
        return torch.sqrt(gz ** 2 + gy ** 2 + gx ** 2 + 1e-8)
    angles = torch.zeros(3, requires_grad=True)
    shift = torch.zeros(3, requires_grad=True)
    nz, ny, nx = shape_zyx
    aff2 = torch.tensor(a2, dtype=torch.float32)
    for step, (stride, it) in enumerate(zip((2, 1), iters)):
        g = np.stack(np.meshgrid(np.arange(0, nx, stride), np.arange(0, ny, stride), np.arange(0, nz, stride), indexing="ij"), -1) \
            + np.asarray(origin_mm)
        pts = torch.tensor(g, dtype=torch.float32).permute(2, 1, 0, 3) @ torch.tensor(M[:3, :3], dtype=torch.float32).T \
            + torch.tensor(M[:3, 3], dtype=torch.float32)
        centre = pts.reshape(-1, 3).mean(0)
        ref = torch.tensor(ndimage.gaussian_filter(t1_grid, stride)[::stride, ::stride, ::stride].copy())
        g1 = gradmag(ref)
        mask = (ref[1:-1, 1:-1, 1:-1] > .1)
        g1 = (g1 - g1[mask].mean()) / g1[mask].std()
        sm = ndimage.gaussian_filter(t2, stride)
        opt = torch.optim.Adam([{"params": [angles], "lr": 2e-3}, {"params": [shift], "lr": 0.3}])
        for _ in range(it):
            opt.zero_grad()
            moved = (pts - centre) @ _rot(angles).T + centre + shift
            g2 = gradmag(sample(sm, aff2, moved))
            g2 = (g2 - g2[mask].mean()) / g2[mask].std()
            loss = -(g1 * g2)[mask].mean()
            loss.backward()
            opt.step()
    R = np.eye(4)
    rot = _rot(angles.detach()).numpy()
    c = centre.numpy()
    R[:3, :3], R[:3, 3] = rot, c - rot @ c + shift.detach().numpy()
    return R, -float(loss.detach())


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


def unmix_partial_volume(post, y, bias, idx, shape):
    """Inside the brain each voxel mixes its two likeliest of CSF / grey / white: its bias-corrected (T1, T2)
    projected on the line between their local pure intensities (Gaussian-weighted means of confident voxels
    within ~10 mm, CSF from the cores of CSF regions; the global median where they are few), a 5 % dead zone at
    each end; the pair shares the voxel's brain share. The EM posterior is close to a step in intensity: IXI012's
    grey/CSF boundary voxels hold 103 mL of CSF by posterior, 52 mL by unmixing the T2. (Solving all three
    classes from two channels and the sum turned noise into white matter: grey/white 1.48 -> 0.90.)"""
    lin = np.exp(y - bias)
    k3 = [CLASSES.index(c) for c in BRAIN]
    p3 = post[k3]
    share = p3.sum(0)
    pure = [p3[j] > .97 * share for j in range(3)]
    core = np.zeros(tuple(shape), bool)
    core[idx] = pure[0] & (share > .9)
    pure[0] = ndimage.binary_erosion(core, iterations=1)[idx]
    mu_loc = np.empty((3, 2, idx[0].size), np.float32)
    for j in range(3):
        w = np.zeros(tuple(shape), np.float32)
        w[idx] = pure[j] & (share > .9)
        ws = ndimage.gaussian_filter(w, 10.)[idx]
        for c in range(2):
            v = np.zeros(tuple(shape), np.float32)
            v[idx] = lin[c] * w[idx]
            glob = np.median(lin[c][w[idx] > 0])
            mu_loc[j, c] = np.where(ws > .02, ndimage.gaussian_filter(v, 10.)[idx] / np.maximum(ws, 1e-6), glob)
    inside = share > .5
    order = np.argsort(-p3[:, inside], axis=0)
    ja, jb = order[0], order[1]
    sel = np.nonzero(inside)[0]
    v = lin[:, inside].T
    ma, mb = mu_loc[ja, :, sel], mu_loc[jb, :, sel]
    axis = mb - ma
    t = ((v - ma) * axis).sum(1) / (axis ** 2).sum(1)
    tb = np.clip((t - .05) / .9, 0, 1)
    fb = tb * share[inside]
    f = np.zeros((3, inside.sum()))
    n = np.arange(inside.sum())
    f[jb, n] = fb
    f[ja, n] = share[inside] - fb
    out = post.copy()
    for j, k in enumerate(k3):
        out[k, inside] = f[j]
    return out


# IXI's T2 places fat 1.63 mm posterior of the water tissue around it (median over the 24 run subjects, IQR 1.09-2.14,
# all 24 the same way): in the scalp at MNI z 10-50, fat's mean depth below the skin minus skin and muscle's is -2.6 mm
# at the back, +0.8 mm at the front and -1.6 mm at the sides (the natural layering), half the front-back difference
# being the shift: a water-fat shift along a front-back readout (434 Hz at 3 T over ~330 Hz per 1.25 mm pixel). Left
# in, our heads had fat at the skin and soft tissue against the skull at the back (fastMRI's 0.57 mm images: the
# reverse) and fat pressed into the bone at the front. Removed by moving the fat map anterior before it is assigned.
IXI_FAT_SHIFT_MM = 1.63
IXI_T2 = {"tr": 5.726, "te": 0.100}  # Hammersmith's T2: TR 5725.8 ms, TE 100 ms, ETL 16 (brain-development.org/scanner-philips-medical-systems-intera-3t)


def soft_references():
    """T2-weighted brightness of skin and muscle relative to white matter in IXI's T2, from the tissue values the
    simulator uses at 3 T (spin echo: PD (1 - exp(-TR/T1)) exp(-TE/T2)); about 0.77 and 0.21."""
    from blochparty import tissuedb
    table = tissuedb.tissue_table(3.0)
    tr, te = IXI_T2["tr"], IXI_T2["te"]
    signal = {k: table[k].pd.mean * (1 - np.exp(-tr / table[k].t1.mean)) * np.exp(-te / table[k].t2.mean)
              for k in ("skin", "muscle", "wm")}
    return signal["skin"] / signal["wm"], signal["muscle"] / signal["wm"]


IXI_PD_TE = 0.008  # the dual-echo TSE's first echo (brain-development.org/scanner-philips-medical-systems-intera-3t)


def orbit_vitreous(r1, r2, p_bone, head, intra, origin_mm, voxel):
    """The eyes' vitreous fraction outside the skull. It is bright in T2 (T2 ~0.5 s) and dark in T1 (T1 ~4 s) like CSF,
    so the soft-tissue rules called it skin (dark in the simulation). In the orbits (``ORBIT_MM``, MNI mm; ``origin_mm``
    (x, y, z), arrays (z, y, x)): sigmoids of T2 above 2 x (width 0.3) and T1 below 0.55 x (width 0.05) white matter (vitreous T1 0.29-0.33 x in IXI012: 0.99), kept in the largest
    connected piece over 0.5 on each side, grown by two voxels (the globe; not sinus fluid or orbital fat).
    r1, r2: bias-corrected T1 / T2 over white matter. IXI012: two globes ~24 x 21 x 27 mm at MNI (-35, 56, -39) and
    (33, 55, -39), 9.0 mL of vitreous at its own size (2026-10-13)."""
    sig = lambda x: 1 / (1 + np.exp(-x))
    zmm, ymm, xmm = (origin_mm[2 - a] + np.arange(r1.shape[a]) * voxel[a] * 1e3 for a in range(3))
    orbit = ((np.abs(xmm)[None, None, :] >= ORBIT_MM[0][0]) & (np.abs(xmm)[None, None, :] <= ORBIT_MM[0][1])
             & (ymm[None, :, None] >= ORBIT_MM[1][0]) & (ymm[None, :, None] <= ORBIT_MM[1][1])
             & (zmm[:, None, None] >= ORBIT_MM[2][0]) & (zmm[:, None, None] <= ORBIT_MM[2][1]))
    p_eye = sig((r2 - 2.) / .3) * sig((.55 - r1) / .05) * (1 - p_bone) * orbit * head * ~(np.asarray(intra) > .5)
    globes = np.zeros(r1.shape, bool)
    for side in (xmm > 0, xmm < 0):
        lab, n = ndimage.label((p_eye > .5) & side[None, None, :])
        if n:
            globes |= lab == np.argmax(np.bincount(lab[lab > 0]))  # bincount is indexed by label
    return p_eye * ndimage.binary_dilation(globes, iterations=2)


def refine_csf(frac, t2w, valid, origin=None, voxel=None):
    """Each voxel's CSF fraction from the subject's own T2 (TE 100 ms, where CSF is ~4 x white matter): the observed
    signal (over the white-matter median) between the subject's pure CSF (deep ventricles, eroded twice) and the
    voxel's other tissues at the simulator's IXI-T2 values (their mix renormalised; grey matter where they are under
    5 %), clipped to [0, 1]; the other tissues keep their proportions. Only voxels the segmentation gave CSF. The
    unmixing calibrates CSF on the ventricles' T2, but the 1 mm convexity voxels it calls pure are not: IXI012's
    labels predicted 30-50 % more CSF-like voxels (> 2.5 x WM) than its T2 shows at MNI z 30-70, its "pure" CSF there
    read 0.86-0.88 of the prediction against 1.04 low down (2026-10-13). With ``origin`` / ``voxel`` [m] the falx that
    ``sample_head`` will add (``add_falx``: CSF -> dura in the fissure) is part of the model: the subject's T2 already
    shows its own falx, so the CSF left beside it is solved for and the falx's share added back to the stored CSF (v13
    without this: IXI012's midline fissure 1.26 x too dark once the falx was added). Returns the fractions and a
    summary."""
    from blochparty import tissuedb
    table = tissuedb.tissue_table(3.0)
    tr, te = IXI_T2["tr"], IXI_T2["te"]
    i = {c: CLASSES.index(c) for c in CLASSES}
    sig = np.array([table[c].pd.mean * (1 - np.exp(-tr / table[c].t1.mean)) * np.exp(-te / table[c].t2.mean)
                    if c in table and c != "background" else 0. for c in CLASSES], np.float32)
    sig /= sig[i["wm"]]
    s = t2w / np.median(t2w[(frac[i["wm"]] > .9) & valid])
    deep = ndimage.binary_erosion(frac[i["csf"]] > .95, iterations=2) & valid
    s_csf = float(np.median(s[deep]))
    sel = valid & (frac[i["csf"]] > .01) & ~deep
    f = frac[:, sel]
    total = 1 - f[i["background"]]
    rest = f.copy(); rest[i["csf"]] = 0; rest[i["background"]] = 0
    rs = rest.sum(0)
    s_rest = np.where(rs > .05, (rest * sig[:, None]).sum(0) / np.maximum(rs, 1e-6), sig[i["gm"]])
    before = float(frac[i["csf"]].sum())
    out_frac = frac.copy()
    falx = np.zeros(frac.shape[1:], np.float32)
    # the falx depends on the CSF it is drawn from: three rounds of (falx on the current fractions, refine the
    # original ones beside it) settle it (IXI012: 4.08 mL from the unrefined CSF, 4.90 after one round)
    for _ in range(3 if origin is not None else 1):
        if origin is not None:
            an = bp.Anatomy({c: torch.from_numpy(out_frac[k]) for k, c in enumerate(CLASSES)}, voxel, origin)
            falx = np.clip(bp.add_falx(an).fractions["dura"].numpy() - out_frac[i["dura"]], 0, None)
        d = np.minimum(falx[sel], total)
        # s = c s_csf + d s_dura + (total - c - d) s_rest: the CSF c beside the falx's dura d; stored c + d
        c = np.clip((s[sel] - d * sig[i["dura"]] - (total - d) * s_rest) / np.maximum(s_csf - s_rest, 1e-3), 0, total - d)
        csf_new = c + d
        scale = np.where(rs > .05, (total - csf_new) / np.maximum(rs, 1e-6), 0.)
        out = f * np.where(np.arange(len(CLASSES))[:, None] == i["background"], 1., scale[None])
        out[i["csf"]] = csf_new
        out[i["gm"]] += np.where(rs > .05, 0., total - csf_new)
        out = np.clip(out, 0, 1); out /= out.sum(0, keepdims=True)  # float rounding (e.g. -1e-8 where all is CSF)
        out_frac = frac.copy(); out_frac[:, sel] = out
    frac = out_frac
    return frac, {"csf_over_wm_deep": round(s_csf, 3), "falx_ml": round(float(falx.sum()) * 1e-3, 2), "csf_table_over_wm": round(float(sig[i["csf"]]), 3),
                  "csf_voxels_refined": int(sel.sum()), "csf_ratio_after_before": round(float(frac[i["csf"]].sum()) / before, 3)}


RESIDUAL_SMOOTH_MM = 0.5  # --residual-smooth-mm: the residual maps' within-tissue smoothing; 1.5 from v11 to v16, 0.5 since 2026-10-13 (q36 / q40: the fine structure bands 0.82 / 0.86 -> 0.94 / 0.97 x real)


def residual_maps(frac, t2w, pdw, valid, origin, voxel, smooth_mm=None):
    """Per-voxel multipliers on the simulator's T2 and PD that make its tissue mixture reproduce this subject's own
    dual-echo TSE (TE 8 / 100 ms, TR 5.726 s): the echo ratio's log L = ln(S_8 / S_100) ~ (100 - 8) ms / T2; L is put
    on its absolute scale by deep ventricular CSF (predicted 0.048 with T2 1.91 s), t2_scale = L_predicted / L_measured
    where water tissue fills the voxel and L > 0.2 (CSF's T2 is too long for 92 ms to measure), over its median in pure
    white matter (the residual is the spatial pattern; the database keeps absolute T2s), pd_scale = the measured
    TE-8 signal over the predicted one (both over white matter) after that T2 change, re-centred on each tissue's median
    (the echo carries the subject's sequence's MT, which the simulator models itself); elsewhere (fat, bone, air, CSF for
    T2) 1. The prediction uses the simulator's own tissue values on the anatomy as it simulates it (deep grey nuclei,
    falx). Inside the skull only. Log-scales smoothed 1.5 mm within their masks and within each voxel's largest tissue
    (the echo ratio amplifies the images' noise: 0.7 mm left ~4 % per voxel and r25's first heads 1.4-1.7 x the real
    fine texture), less a cubic trend fitted over pure white matter (the
    scanner's B1+ and bias; this also puts white matter at 1), clipped to +-0.5. t2w, pdw: bias-corrected echoes."""
    from blochparty import tissuedb
    table = tissuedb.tissue_table(3.0)
    an = bp.Anatomy({c: torch.from_numpy(frac[k]) for k, c in enumerate(CLASSES)}, voxel, origin)
    an = bp.add_falx(bp.label_deep_grey(an))
    tr, tes = IXI_T2["tr"], (IXI_PD_TE, IXI_T2["te"])
    names = [k for k in an.fractions if k != "background" and k in table and k != "vessels"]
    f = {k: an.fractions[k].numpy() for k in names}

    def predicted(t2_scale=1.0):
        out = []
        for te in tes:
            s = 0
            for k in names:
                p = table[k]
                fat = p.fat_fraction is not None and p.fat_fraction.mean > .5
                t2 = p.t2.mean * (1.0 if fat else t2_scale)
                s = s + f[k] * p.pd.mean * (1 - np.exp(-tr / p.t1.mean)) * np.exp(-te / t2)
            out.append(s)
        return out
    s8p, s100p = predicted()
    water = sum(f[k] for k in names if not (table[k].fat_fraction and table[k].fat_fraction.mean > .5) and k not in ("skull", "marrow", "dura"))
    csf = f["csf"]
    deep_csf = ndimage.binary_erosion(csf > .95, iterations=2) & valid
    # inside the skull only: the scalp's layers are thin, their fat still shifted in the subject's images (the
    # phantom's is not), so a scalp residual compares misaligned tissue (r25's first heads: posterior scalp peak 0.43
    # of p99 against r24's 0.65 and the real 0.83)
    intracranial = sum(f[c] for c in f if c in ("csf", "gm", "wm", "caudate", "putamen", "pallidum", "thalamus")) > .5
    ok = valid & (t2w > 1e-3) & (pdw > 1e-3)
    lr = np.zeros_like(t2w); lr[ok] = np.log(pdw[ok] / t2w[ok])
    k = np.log(np.exp((tes[1] - tes[0]) / table["csf"].t2.mean)) - np.median(lr[deep_csf]) if deep_csf.sum() > 50 else 0.0
    L = lr + k
    Lp = np.log(np.maximum(s8p, 1e-9) / np.maximum(s100p, 1e-9))
    t2mask = ok & (water > .7) & (L > .2) & (csf < .3) & intracranial
    log_t2 = np.zeros_like(L); log_t2[t2mask] = np.log(Lp[t2mask] / L[t2mask])

    # each voxel to its largest tissue, smoothed among voxels of the same tissue: the noise falls, the boundaries
    # between tissues (a nucleus's edge, the cortex) stay where the anatomy puts them
    dominant = np.argmax(np.stack([f[k] for k in names]), 0)

    def smooth(v, m, sigma=RESIDUAL_SMOOTH_MM if smooth_mm is None else smooth_mm):
        out = np.zeros_like(v)
        for i in np.unique(dominant[m]):
            mi = m & (dominant == i)
            num = ndimage.gaussian_filter(v * mi, sigma); den = ndimage.gaussian_filter(mi.astype(np.float32), sigma)
            out[mi] = num[mi] / np.maximum(den[mi], 1e-6)
        return out

    zz, yy, xx = np.meshgrid(*(np.linspace(-1, 1, n, dtype=np.float32) for n in frac.shape[1:]), indexing="ij")
    terms = [(i, j, l) for i in range(4) for j in range(4 - i) for l in range(4 - i - j)]
    wm_pure = f["wm"] > .9

    def local(v, m):  # minus a cubic trend fitted over pure white matter (uniform tissue): the scanner's B1+ / bias
        fit = wm_pure & m
        A = np.stack([zz[fit] ** i * yy[fit] ** j * xx[fit] ** l for i, j, l in terms], 1)
        coef = np.linalg.lstsq(A, v[fit], rcond=None)[0]
        trend = sum(c * zz ** i * yy ** j * xx ** l for c, (i, j, l) in zip(coef, terms))
        return np.where(m, v - trend, 0.0)
    log_t2 = smooth(log_t2, t2mask)
    # relative to white matter: the absolute level of a two-point TSE T2 is the sequence's (stimulated echoes lengthen
    # it: IXI012's white matter reads 1.18 x our 72 ms), the tissue database keeps the absolute values; and local: a
    # cubic trend fitted over pure white matter (+-15 % from the base of the brain to the top in IXI012: B1+) is removed
    wm_offset = float(np.median(log_t2[wm_pure & t2mask]))
    log_t2 = np.where(t2mask, np.clip(local(log_t2, t2mask), -.5, .5), 0.0)
    t2_scale = np.exp(log_t2)
    s8q, _ = predicted(t2_scale)
    wm = f["wm"] > .9
    pdmask = ok & (water > .7) & intracranial
    ratio = np.zeros_like(L)
    ratio[pdmask] = (pdw[pdmask] / np.median(pdw[wm & ok])) / (s8q[pdmask] / np.median(s8q[wm & ok]))
    log_pd = np.clip(local(smooth(np.log(np.maximum(ratio, 1e-6)) * pdmask, pdmask), pdmask), -.5, .5)
    # each tissue's median PD multiplier 1, only the pattern within a tissue kept: the subject's short-TE echo carries its
    # own multi-slice sequence's magnetization transfer (white matter darkest, grey matter less, CSF none), which a
    # prediction without MT reads as PD (v11: CSF x1.21-1.28, grey matter x1.05-1.09 in every subject) and the simulator
    # then applies its own MT on top. T2 from the echo ratio is MT-free (both echoes scale alike): its tissue offsets stay
    pd_offset = {}
    for i in np.unique(dominant[pdmask]):
        mi = pdmask & (dominant == i)
        pure = mi & (f[names[i]] > .9)
        pd_offset[names[i]] = float(np.median(log_pd[pure if pure.sum() > 100 else mi]))
        log_pd[mi] -= pd_offset[names[i]]
    diag = {"csf_offset": round(float(k), 4), "wm_t2_apparent_over_table": round(float(np.exp(wm_offset)), 3),
            "pd_tissue_offset_removed": {c: round(float(np.exp(v)), 3) for c, v in pd_offset.items() if c in
                                         ("csf", "gm", "wm", "caudate", "putamen", "pallidum", "thalamus")},
            "t2_scale_median": {}, "pd_scale_median": {}}
    for c in ("wm", "gm", "skin", "muscle", "caudate", "putamen", "thalamus"):
        if c in f:
            m = (f[c] > .9) & t2mask
            if m.sum() > 100:
                diag["t2_scale_median"][c] = round(float(np.exp(np.median(log_t2[m]))), 3)
                diag["pd_scale_median"][c] = round(float(np.exp(np.median(log_pd[m & pdmask]))), 3)
    return t2_scale.astype(np.float32), np.exp(log_pd).astype(np.float32), diag


def prepare(ixi_id, root, atlas, fixed2, partial_volume=True, residual_smooth_mm=None, save_intensities=False):
    t0 = time.time()
    t1p = next((root / "T1").glob(f"IXI{ixi_id:03d}-HH-*-T1.nii.gz"))
    t2p = next((root / "T2").glob(f"IXI{ixi_id:03d}-HH-*-T2.nii.gz"))
    t1, a1 = load_ras(t1p)
    t2, a2 = load_ras(t2p)
    M, score = register(*fixed2, t1, a1)
    shape, origin = atlas.shape, np.array(atlas.origin) * 1e3
    c1 = onto(t1, a1, M, shape, origin, order=3)
    R2, t2score = align_t2(c1, t2, a2, M, shape, origin)
    M2 = R2 @ M
    chans = [c1, onto(t2, a2, M2, shape, origin, order=3)]
    pdp = next((root / "PD").glob(t2p.name.replace("-T2.nii.gz", "-PD.nii.gz")), None) if (root / "PD").exists() else None
    pd_onto = None
    if pdp is not None:
        pdv, apd = load_ras(pdp)
        pd_onto = onto(pdv, apd, M2, shape, origin, order=3)  # the T2's acquisition: same geometry, same motion
    valid = (onto(np.ones_like(t1), a1, M, shape, origin) > 0.99) & (onto(np.ones_like(t2), a2, M2, shape, origin) > 0.99)
    fr = {k: atlas.fractions[k].numpy().astype(np.float32) for k in EM_CLASSES}
    intra = ndimage.gaussian_filter(sum(fr[k] for k in BRAIN), 2.0)
    share = {k: ndimage.gaussian_filter(fr[k], 8.0) + 0.02 for k in BRAIN}
    tot = sum(share.values())
    prior = np.stack([intra * share[k] / tot if k in BRAIN else ndimage.gaussian_filter(fr[k], 2.0) for k in EM_CLASSES]) + 1e-3
    prior /= prior.sum(0)
    idx = np.nonzero(valid)
    y = np.stack([np.log(np.maximum(c[idx], 1e-3)) for c in chans])
    zz, yy, xx = (np.linspace(-1, 1, n, dtype=np.float32) for n in shape)
    terms = [(i, j, k) for i in range(4) for j in range(4 - i) for k in range(4 - i - j)]
    basis = np.stack([zz[idx[0]] ** i * yy[idx[1]] ** j * xx[idx[2]] ** k for i, j, k in terms], 1)
    post, mu, bias = em(y, prior[:, idx[0], idx[1], idx[2]], basis)
    if partial_volume:
        post = unmix_partial_volume(post, y, bias, idx, shape)
    frac = np.zeros((len(CLASSES),) + tuple(shape), np.float32)
    frac[:len(EM_CLASSES), idx[0], idx[1], idx[2]] = post
    frac[0][~valid] = 1
    # outside the skull, rules on the bias-corrected intensities relative to the subject's white matter (the
    # EM's head classes swap identities where BrainWeb's affine-aligned priors miss the thin layers): fat bright
    # by intensity: bone dark in both, fat bright in T2 and T1 (inside the bone shell: marrow), soft tissue between
    corr = []
    for c in range(2):
        v = np.zeros(shape, np.float32)
        v[idx] = np.exp(y[c] - bias[c])
        corr.append(v)
    wmref = [np.median(v[frac[3] > .9]) for v in corr]
    r1, r2 = corr[0] / wmref[0], corr[1] / wmref[1]
    sig = lambda x: 1 / (1 + np.exp(-x))
    intra = ndimage.binary_fill_holes(ndimage.binary_closing(frac[[1, 2, 3, 8, 10]].sum(0) > .5, iterations=2))
    raw = ndimage.binary_closing((r1 > .25) | (r2 > .25) | intra, iterations=3)
    # one piece, and nothing hanging on by a thread (skin markers, ghosts): the largest component of the mask
    # opened by ~2 mm, grown back inside the mask
    lab, _ = ndimage.label(ndimage.binary_opening(raw, iterations=2))
    core = lab == np.argmax(np.bincount(lab[lab > 0]))
    head = ndimage.binary_fill_holes(ndimage.binary_dilation(core, iterations=3) & raw)
    # above the ears each axial section of the head is a convex oval: what an 8 mm disc opening removes there
    # is an object on the skin (markers), not anatomy
    zmm = origin[2] + np.arange(shape[0]) * atlas.voxel[0] * 1e3
    disc = np.hypot(*np.mgrid[-8:9, -8:9]) <= 8
    for kz in np.nonzero(zmm > 15)[0]:
        if head[kz].any():
            opened = ndimage.binary_opening(head[kz], structure=disc)
            head[kz] = ndimage.binary_fill_holes(ndimage.binary_dilation(opened, iterations=2) & head[kz])
    p_bone = sig((.35 - r1) / .06) * sig((.45 - r2) / .08)
    p_eye = orbit_vitreous(r1, r2, p_bone, head, intra, origin, atlas.voxel)
    # 2026-10-13: the T2's water-fat shift (IXI_FAT_SHIFT_MM, below) lays the scalp's fat over the bone's T2, so the
    # skull reads T1-dark but T2-bright and the rule above called it soft tissue (a 3-voxel "skin" band inside the
    # frontal bone, bone 2.4 mm thick at the forehead against ~6 in life; the simulated anterior band 1.08-1.14 x
    # real). Fat is T1-bright, so a T1-dark, T2-bright voxel next to bone is bone under displaced fat: the T2 factor
    # is waived there (not in the eyes, which are T1-dark and T2-bright too). IXI012: frontal bone 2.4 -> 3.6 mm.
    ghost = (sig((.30 - r1) / .05) * sig((r2 - .9) / .1) * ndimage.binary_dilation(p_bone > .5, iterations=3)
             * (1 - np.clip(p_eye / max(p_eye.max(), 1e-6), 0, 1)))
    p_bone = sig((.35 - r1) / .06) * np.clip(sig((.45 - r2) / .08) + ghost, 0, 1)
    ghost_ml = float(((ghost > .5) & head & ~intra).sum()) * np.prod(atlas.voxel) * 1e6
    # fat by unmixing T2 between soft tissue (~0.9 x WM, temporalis) and the subject's own pure fat: the median T2
    # of head tissue brighter than WM in T1 (fat, marrow). The 98th percentile of T2 among T1-bright tissue (to
    # 2026-10-12; IXI012 3.09 x WM against a median of 1.41) unmixed subcutaneous fat to about a third: its own fat
    # then gave a simulated scalp 35-40 % short of fastMRI's in signal over WM, and the 4 mm layer laid on top
    # (layer_scalp) 35-65 % over
    pure = float(np.median(r2[head & ~intra & (r1 > 1.)]))
    p_fat = np.clip((r2 - .9) / (pure - .9), 0, 1) * sig((r1 - .55) / .08) * (1 - p_bone)
    # undo the T2's water-fat shift: its fat sits IXI_FAT_SHIFT_MM posterior of the water tissue (front-back readout)
    p_fat = np.clip(ndimage.shift(p_fat, (0, IXI_FAT_SHIFT_MM / (atlas.voxel[1] * 1e3), 0), order=1, mode="nearest"), 0, 1) * (1 - p_bone)
    p_fat = p_fat * (1 - p_eye)
    p_soft = np.clip(1 - p_fat - p_bone - p_eye, 0, 1)
    shell = ndimage.binary_closing(p_bone > .5, iterations=2) & ~intra
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
    frac[i["eye"]][out_mask] = p_eye[out_mask]
    # soft tissue unmixed by its T2 between skin and muscle, at the brightness the simulator's own tissue values give
    # them in IXI's T2 (soft_references). Until 2026-10-12 soft tissue deeper than 8 mm was muscle: IXI012's soft
    # tissue at MNI z 5-50 reads 0.56-0.70 x WM in T2 at every depth, so a block of near-black muscle (T2 30 ms)
    # under the scalp made the simulated dark band at the skull ~4 px wide against the real ~1.5
    r_skin, r_muscle = soft_references()
    m = np.clip((r_skin - r2) / (r_skin - r_muscle), 0, 1)
    frac[i["skin"]][out_mask] = (p_soft * (1 - m))[out_mask]
    frac[i["muscle"]][out_mask] = (p_soft * m)[out_mask]
    frac[i["background"]][~head] = 1
    for c in CLASSES:
        if c != "background":
            frac[i[c]][~head] = 0
    frac /= frac.sum(0, keepdims=True)
    maps, diag = {}, None
    t2w = np.zeros(shape, np.float32); t2w[idx] = np.exp(y[1] - bias[1])  # bias-corrected T2
    frac, csf_diag = refine_csf(frac, t2w, valid, tuple(np.asarray(origin) * 1e-3), atlas.voxel)
    if pd_onto is not None:  # both echoes corrected by the T2's bias field (one coil, one acquisition)
        pdw = np.zeros(shape, np.float32); pdw[idx] = np.maximum(pd_onto[idx], 1e-6) * np.exp(-bias[1])
        t2s, pds, diag = residual_maps(frac, t2w, pdw, valid, tuple(np.asarray(origin) * 1e-3), atlas.voxel, residual_smooth_mm)
        diag["smooth_mm"] = RESIDUAL_SMOOTH_MM if residual_smooth_mm is None else residual_smooth_mm
        maps = {"t2_scale": t2s, "pd_scale": pds}
    scale = abs(np.linalg.det(M[:3, :3])) * 1e-3  # mL per MNI voxel in the subject's own size
    vol = {c: round(float(frac[i].sum()) * scale, 1) for i, c in enumerate(CLASSES) if c != "background"}
    t2_motion = {"shift_mm": np.round(R2[:3, 3], 2).tolist(), "rotation_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(R2[:3, :3]) - 1) / 2, -1, 1)))), 2),
                 "gradient_r": round(t2score, 3)}
    files = {"t1": t1p.name, "t2": t2p.name, "t2_to_t1": t2_motion, "csf_refined": csf_diag, "bone_under_shifted_fat_ml": round(ghost_ml, 1)} | ({"pd": pdp.name, "residual": diag} if diag else {})
    if save_intensities:  # for developing label rules offline: the bias-corrected T1 and T2 over white matter
        maps = maps | {"r1": r1.astype(np.float16), "r2": r2.astype(np.float16)}
    return frac, M, score, vol, files, round(time.time() - t0), maps


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=HERE / "data" / "ixi")
    ap.add_argument("--subjects", default=None, help="comma-separated IXI ids")
    ap.add_argument("--count", type=int, default=None, help="this many subjects spread over the ages")
    ap.add_argument("--posteriors", action="store_true", help="keep the EM posteriors as fractions (no partial-volume unmixing)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default root/prepared)")
    ap.add_argument("--save-intensities", action="store_true",
                    help="also store the bias-corrected T1 / T2 over white matter (r1, r2) in the npz (diagnostics)")
    ap.add_argument("--residual-smooth-mm", type=float, default=None,
                    help=f"the residual maps' within-tissue smoothing [mm] (default {RESIDUAL_SMOOTH_MM})")
    args = ap.parse_args()
    demo = demographics(args.root)
    ids = sorted(int(p.name[3:6]) for p in (args.root / "T1").glob("*-HH-*-T1.nii.gz"))
    ids = [i for i in ids if i in demo and any((args.root / "T2").glob(f"IXI{i:03d}-HH-*"))]
    if args.subjects:
        ids = [int(s) for s in args.subjects.split(",")]
    elif args.count:
        order = sorted(ids, key=lambda i: demo[i]["age"])
        ids = sorted(order[int(j)] for j in np.linspace(0, len(order) - 1, args.count).round())
    out = args.out or args.root / "prepared"
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
        try:
            frac, M, score, vol, files, secs, maps = prepare(i, args.root, atlas, (img, aff), not args.posteriors, args.residual_smooth_mm, args.save_intensities)
        except Exception as e:  # one subject's failure (2026-10-13: a non-finite map on one of 64) must not end the batch
            print(f"IXI{i:03d} failed: {type(e).__name__}: {e}", flush=True)
            continue
        meta = {"ixi_id": i, **demo[i], "atlas": f"brainweb{ATLAS:02d}", "classes": CLASSES, "affine_mm": M.tolist(),
                "lncc": round(score, 3), "volumes_ml": vol, "files": files, "origin": list(atlas.origin),
                "voxel": list(atlas.voxel), "shape": list(atlas.shape)}
        np.savez_compressed(dest, fractions=np.round(frac * 255).astype(np.uint8), meta=json.dumps(meta),
                            **{k: v.astype(np.float16) for k, v in maps.items()})
        print(f"IXI{i:03d} age {demo[i]['age']:.0f} {demo[i]['sex']}: GM {vol['gm']:.0f} WM {vol['wm']:.0f} CSF {vol['csf']:.0f} mL,"
              f" lncc {score:.3f}, {secs} s", flush=True)
