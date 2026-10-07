"""AHEAD v3 7 T templates through ``load_ahead``: units, geometry and a measured-map simulation.

Checks (printed; recorded in validation/anatomy/RESULTS.md):

* units: the R1 histogram's white-matter mode and the R2* there against Caan et al. 2019
  (doi:10.1002/hbm.24490, Table 1, MP2RAGEME at 7 T: WM T1 1.19 s, T2* 26.7 ms; GM T1 2.0 s,
  T2* 32.4 ms). Same acquisition family as AHEAD, so this is a units/consistency check, not an
  independent measurement; QSM in the globus pallidus (MNI ~ (+-20, -3, 0) mm) should be a few
  tenths of a ppm positive (iron), which fixes ppm against ppb.
* geometry: the brain centroid in MNI2009b coordinates and the MNI origin inside the brain.
* a 7 T spoiled GRE of an axial slab through the basal ganglia from the measured R1 / R2* /
  relative PD maps, with the QSM-derived field (brain only, relative); T2 an explicit assumption and
  R2' = R2* - R2 (32 Cauchy-quantile spins per voxel).

    uv run python validation/anatomy/fetch_ahead.py && uv run python validation/anatomy_ahead.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import blochparty as bp
from blochparty.anatomy import Distribution, TissueProperties

ROOT = Path(__file__).resolve().parent / "anatomy" / "data" / "ahead"


def world(a: bp.Anatomy):
    zz, yy, xx = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) for n in a.shape), indexing="ij")
    return [(g * v + o) * 1e3 for g, v, o in ((xx, a.voxel[2], a.origin[0]), (yy, a.voxel[1], a.origin[1]),
                                              (zz, a.voxel[0], a.origin[2]))]  # mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=Path("outputs/anatomy_ahead.png"))
    args = ap.parse_args()
    a = bp.load_ahead(args.root)
    m = a.maps["brain_mask"] > 0.5
    r1, r2s, chi = a.maps["R1"][m].numpy(), a.maps["R2star"][m].numpy(), a.chi[m].numpy() * 1e6
    hist, edges = np.histogram(r1, bins=200, range=(0.2, 1.2))
    centres = 0.5 * (edges[1:] + edges[:-1])
    wm_mode = centres[np.argmax(np.where(centres > 0.7, hist, 0))]
    gm_mode = centres[np.argmax(np.where((centres > 0.4) & (centres < 0.65), hist, 0))]
    wm = np.abs(r1 - wm_mode) < 0.02
    gm = np.abs(r1 - gm_mode) < 0.02
    print(f"brain voxels {int(m.sum()):,} (1 mm); R1 modes: WM {wm_mode:.3f} /s (T1 {1 / wm_mode:.2f} s; Caan 1.19 s), "
          f"GM {gm_mode:.3f} /s (T1 {1 / gm_mode:.2f} s; Caan 2.0 s)")
    print(f"R2* median at those R1: WM {np.median(r2s[wm]):.1f} /s (T2* {1e3 / np.median(r2s[wm]):.1f} ms; Caan 26.7), "
          f"GM {np.median(r2s[gm]):.1f} /s (T2* {1e3 / np.median(r2s[gm]):.1f} ms; Caan 32.4)")
    x, y, z = world(a)
    gp = m & (x.abs() > 10) & (x.abs() < 26) & (y > -14) & (y < 6) & (z.abs() < 6)  # pallidum / putamen box
    print(f"QSM: brain 1/50/99 % {np.percentile(chi, [1, 50, 99]).round(3)} ppm; basal-ganglia box 99th percentile "
          f"{float(np.percentile(a.chi[gp].numpy() * 1e6, 99)):.3f} ppm ({int(gp.sum())} voxels)")
    w = m.double()
    c = [float((w * g).sum() / w.sum()) for g in (x, y, z)]
    i = [int(round((0 - o) / v)) for o, v in zip(a.origin, reversed(a.voxel))]
    print(f"brain centroid (MNI mm) ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}); MNI origin inside the brain: "
          f"{bool(m[i[2], i[1], i[0]])}")

    # 7 T GRE of a 2 mm slab at z = 0 (basal ganglia), measured R1 / R2* / PD, assumed T2
    note = "Assumption: AHEAD has no R2 map; brain T2 50 ms at 7 T for the demonstration"
    table = {7.0: {"brain": TissueProperties(pd=Distribution(1.0, None, "Assumption: overridden by the PD map", True),
                                            t1=Distribution(1.5, None, "Assumption: overridden by the R1 map", True),
                                            t2=Distribution(0.05, None, note, True),
                                            chi=Distribution(0.0, None, "Assumption: QSM map used", True))}}
    k0 = int(round((0 - a.origin[2]) / a.voxel[0]))
    # R2' = R2* - R2 from the measured R2* map and the assumed T2 (clamped at 0 where R2* < R2,
    # e.g. CSF). Assumption: the brain-only QSM field adds little macroscopic dephasing on top.
    a.maps["intrinsic_r2prime"] = torch.where(m, (a.maps["R2star"] - 1 / 0.05).clamp_min(0), 0.0)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
    ax[0].imshow(a.chi[k0].numpy()[::-1] * 1e6, cmap="gray", vmin=-0.1, vmax=0.15)
    ax[0].set_title("AHEAD QSM [ppm], z = 0 (MNI)")
    for j, (te, label) in enumerate(((5e-3, "TE 5 ms"), (20e-3, "TE 20 ms"))):
        ph = bp.to_phantom(a, table, 7.0, fov_region=(slice(k0, k0 + 2), slice(None), slice(None)),
                           spins_per_voxel=(2, 4, 4), shim_order=2)
        ph.pos = ph.pos - ph.pos.mean(0)
        seq = bp.gre_2d(192, 0.24, flip_deg=15.0, tr=30e-3, te=te, n_dummy=40)
        try:
            from blochparty import metal

            s = metal.simulate_metal(ph.to(torch.float32), seq.to(torch.float32)).signal.to(torch.complex128)
        except ImportError:
            s = bp.simulate(ph, seq).signal
        img = bp.reconstruct(s, seq)[0].abs().numpy()
        ax[j + 1].imshow(img[::-1], cmap="gray", vmax=np.percentile(img, 99.5))
        ax[j + 1].set_title(f"7 T spoiled GRE {label}, from measured R1/R2*/PD + QSM field")
    for a_ in ax:
        a_.axis("off")
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=80)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
