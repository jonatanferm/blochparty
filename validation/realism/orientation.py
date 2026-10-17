"""Which way up the fastMRI images are, checked against IXI's axial T2 volumes (orientation from their NIfTI
affines, consistent with the T1s registered to MNI by validation/anatomy/prepare_ixi.py).

1. Template: IXI slices 45-75 mm below the head top, each cropped to its head's bounding box, resampled to
   64 x 64 and standardised inside a central ellipse (70 % of the box, so the outline barely counts), are
   averaged anterior up. Each real (and simulated) slice is correlated with that template and with it
   flipped up-down.
2. Shape: where along the head's length (0 = top row) its width peaks, per subject (median over slices).

    uv run python validation/realism/orientation.py [--sim DIR] [--cache ~/data/realism_cache]

2026-10-12: real slices 1-7 matched the flipped template in 91 % of 147 (r 0.35 vs 0.22) and peaked at
0.34-0.44 of their length from the top row against IXI's 0.48-0.62 measured from the front: fastMRI shows
the head posterior up. The simulated slices (to_real_orientation before the fix) preferred the template
as-is, i.e. were shown anterior up.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

import realism as R  # noqa: E402
import score as S  # noqa: E402
import simulate_heads as SH  # noqa: E402

IXI = Path(__file__).resolve().parents[1] / "anatomy" / "data" / "ixi" / "T2"
N = 64
_yy, _xx = np.mgrid[:N, :N]
ELLIPSE = ((_yy - N / 2 + .5) ** 2 + (_xx - N / 2 + .5) ** 2) <= (N / 2 * .7) ** 2


def head_mask(img):
    """The head: above 12 % of the 99th percentile, opened once, holes filled, the largest piece."""
    m = ndimage.binary_fill_holes(ndimage.binary_opening(img > .12 * np.percentile(img, 99), iterations=1))
    lab, _ = ndimage.label(m)
    return lab == np.argmax(np.bincount(lab.ravel())[1:]) + 1


def standardised(img):
    m = head_mask(img)
    r, c = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    box = img[r.min():r.max() + 1, c.min():c.max() + 1]
    v = ndimage.zoom(box, (N / box.shape[0], N / box.shape[1]), order=1)[:N, :N][ELLIPSE]
    return (v - v.mean()) / v.std()


def width_peak(img):
    """Fraction of the head's length from the top row at which its (smoothed) width peaks."""
    m = head_mask(img)
    rows = np.nonzero(m.any(1))[0]
    w = ndimage.uniform_filter1d(m.sum(1).astype(float), 5)[rows.min():rows.max() + 1]
    return np.argmax(w) / (len(w) - 1)


def ixi_slices(count=30):
    """IXI axial T2 slices 45-75 mm below the head top, anterior up, at the volume's in-plane spacing."""
    import nibabel as nib
    out = []
    for f in sorted(IXI.glob("*-HH-*.nii.gz"))[:count]:
        img = nib.as_closest_canonical(nib.load(f))
        v, z = img.get_fdata(dtype=np.float32), img.header.get_zooms()
        if z[2] > 2:  # a few volumes are 5 mm thick
            continue
        inside = v > .1 * np.percentile(v, 99)
        top = np.nonzero(inside.sum((0, 1)) * z[0] * z[1] > 2000)[0].max()
        for depth in (75, 65, 55, 45):  # (x, y) RAS -> rows anterior first, isotropic
            sl = v[:, :, int(round(top - depth / z[2]))].T[::-1]
            out.append((f.name[:6], ndimage.zoom(sl, (z[1], z[0]), order=1)))
    return out


def report(name, items, template):
    """items: [(person, image)] as displayed. Correlations with the template as-is (anterior up) and flipped."""
    flat = np.zeros((N, N))
    flat[ELLIPSE] = template
    flipped = flat[::-1][ELLIPSE]
    r = np.array([[np.corrcoef(standardised(im), t)[0, 1] for t in (template, flipped)] for _, im in items])
    peaks = {}
    for person, im in items:
        peaks.setdefault(person, []).append(width_peak(im))
    med = sorted(float(np.median(v)) for v in peaks.values())
    up = "posterior" if np.mean(r[:, 1] > r[:, 0]) > .5 else "anterior"
    print(f"{name}: {len(r)} slices, r {r[:, 0].mean():.3f} anterior up / {r[:, 1].mean():.3f} posterior up "
          f"(posterior up better in {np.mean(r[:, 1] > r[:, 0]) * 100:.0f} %): {up} up; width peak "
          f"{med[0]:.2f}-{med[-1]:.2f} of the length from the top row (median {np.median(med):.2f}, {len(med)} people)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=Path.home() / "data" / "realism_cache")
    ap.add_argument("--fastmri", type=Path, default=R.FASTMRI)
    ap.add_argument("--sim", type=Path, default=None, help="simulate_heads output to check as score.py displays it")
    ap.add_argument("--slices", default="1-7")
    args = ap.parse_args()
    lo, hi = (int(a) for a in args.slices.split("-"))
    ixi = ixi_slices()
    template = np.mean([standardised(im) for _, im in ixi], 0)
    report("IXI (anterior up by the affine)", ixi, template)
    real, _ = S.real_set(args.cache, args.fastmri, slices=range(lo, hi + 1))
    report("fastMRI", [(p, im) for p, s, im in real], template)
    if args.sim:
        sims = []
        for f in sorted(args.sim.glob("sim_*.npz")):
            d = np.load(f)
            meta = json.loads(str(d["meta"]))
            sims += [(f.stem, R.normalise(SH.to_real_orientation(R.recon(d[f"slice{s:02d}"].astype(np.complex128)))))
                     for s in range(lo, hi + 1) if f"slice{s:02d}" in d]
        report(f"simulated ({args.sim.name}, {meta.get('source', 'brainweb')})", sims, template)


if __name__ == "__main__":
    main()
