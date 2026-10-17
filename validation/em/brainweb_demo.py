"""A BrainWeb head at 3 T with full-wave coil fields: birdcage B1+ and an eight-loop receive array
with its correlated thermal noise, against the same head with ideal fields.

Needs the maps of ``validation/em/run_brainweb.py`` (body.npz, head8.npz, head8_noise.npz in
validation/anatomy/data/em) and BrainWeb (validation/anatomy/fetch_brainweb.py). BrainWeb's terms
do not allow redistribution: keep the output local.

    uv run python validation/em/brainweb_demo.py [--snr 40] [--out outputs/em_brainweb.png]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import blochparty as bp
from blochparty import em
from blochparty.maps import FieldMaps
from blochparty.receive import thermal_noise

ROOT = Path(__file__).resolve().parents[1] / "anatomy" / "data"


def simulate(phantom, seq, maps=None):
    if bp.engines.available("gpu"):
        return bp.simulate(phantom.to(torch.float32), seq.to(torch.float32), maps=maps, engine="gpu").signal.to(
            torch.complex128
        )
    return bp.simulate(phantom, seq, maps=maps).signal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=Path, default=ROOT / "em")
    ap.add_argument("--height", type=float, default=0.02, help="slab centre z [m] (the loops' height)")
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--snr", type=float, default=40.0, help="white-matter SNR of the ideal single-channel image")
    ap.add_argument("--out", type=Path, default=Path("outputs/em_brainweb.png"))
    args = ap.parse_args()
    t0 = time.perf_counter()
    body, head8 = em.read_mariepy_maps(args.maps / "body.npz"), em.read_mariepy_maps(args.maps / "head8.npz")
    resistance = em.read_noise_resistance(args.maps / "head8_noise.npz")
    fm = em.field_maps(body, head8)  # quadrature birdcage, |B1+| relative to its median over the head
    head = bp.load_brainweb(4, ROOT, voxel=(1e-3,) * 3)
    k = int(round((args.height - head.origin[2]) / 1e-3))
    ph = bp.to_phantom(
        head,
        field_T=3.0,
        fov_region=(slice(k - 1, k + 1), slice(None), slice(None)),
        spins_per_voxel=(2, 2, 2),
        variability={"t1": 0.05, "t2": 0.05, "pd": 0.02, "length": 0.01},
    )
    # centre the slab in the field of view, moving the scanner-frame maps with it
    shift = ph.pos.mean(0)
    ph.pos = ph.pos - shift
    fm.origin = tuple(float(o - s) for o, s in zip(fm.origin, shift.tolist()))
    print(f"{len(ph):,} spins at z = {float(shift[2]) * 1e3:.0f} mm ({time.perf_counter() - t0:.1f} s)")
    seq = bp.gre_2d(args.n, 0.24, flip_deg=30.0, tr=20e-3, te=4.6e-3, n_dummy=60)
    ideal = bp.cartesian_recon(simulate(ph, seq), seq)[0]
    b1_only = bp.cartesian_recon(simulate(ph, seq, maps=FieldMaps(fm.origin, fm.spacing, b1=fm.b1)), seq)[0]
    sig = simulate(ph, seq, maps=fm)
    print(f"three simulations done ({time.perf_counter() - t0:.1f} s)")

    # noise: thermal, correlated as the sample-loss resistance; sigma set by the ideal image's WM SNR
    psi = resistance / resistance.diagonal().real.mean()
    g = torch.Generator().manual_seed(0)
    unit = bp.cartesian_recon(thermal_noise(torch.zeros(1, sig.shape[1], dtype=sig.dtype), 1.0, generator=g), seq)[0]
    img_ideal = ideal.abs()
    level = float(torch.quantile(img_ideal[img_ideal > 0.2 * img_ideal.max()], 0.75))  # WM is the brightest at T1w
    # the ideal coil has the birdcage-normalised sensitivity 1; the array's channels are scaled to max 1
    sigma = level / args.snr / float(unit.std())
    noisy = bp.cartesian_recon(thermal_noise(sig, sigma, cov=psi, generator=g), seq)  # (C, n, n)
    clean = bp.cartesian_recon(sig, seq)
    rss = clean.abs().pow(2).sum(0).sqrt()
    sens = clean / rss.clamp(min=1e-12 * float(rss.max()))  # prescan-normalised sensitivities
    pinv = torch.linalg.inv(psi.to(noisy.dtype))
    num = torch.einsum("cyx,cd,dyx->yx", sens.conj(), pinv, noisy)
    den = torch.einsum("cyx,cd,dyx->yx", sens.conj(), pinv, sens).real
    roemer = (num / den.clamp(min=1e-12)).abs()
    rss_noisy = noisy.abs().pow(2).sum(0).sqrt()
    corr = psi.abs() / torch.sqrt(psi.diagonal().real[:, None] * psi.diagonal().real[None])
    off = corr[~torch.eye(len(corr), dtype=torch.bool)]
    print(f"noise correlation |rho|: mean {float(off.mean()):.2f}, max {float(off.max()):.2f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from blochparty.maps import sample

    n, fov = args.n, 0.24
    x = (torch.arange(n, dtype=torch.float64) - n / 2) * fov / n
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    pts = torch.stack([xx, yy, torch.zeros_like(xx)], -1)
    o, sp = torch.as_tensor(fm.origin, dtype=torch.float64), torch.as_tensor(fm.spacing, dtype=torch.float64)
    b1_map = sample(fm.b1[None].double(), o, sp, pts)[..., 0]
    support = img_ideal > 0.05 * img_ideal.max()
    b1_vals = b1_map[support]
    print(
        f"|B1+| / median over the slab: {float(b1_vals.min()):.2f} to {float(b1_vals.max()):.2f}, "
        f"cv {float(b1_vals.std() / b1_vals.mean()):.3f}"
    )
    # registration: where B1+ is high the 30 deg T1w GRE (above the Ernst angle) loses signal in
    # proportion; the ratio image must follow the sampled map, not its mirror or transpose
    ratio = (b1_only.abs() / img_ideal.clamp(min=1e-12))[support]
    for name, m in (("map", b1_map), ("transposed", b1_map.T), ("flipped y", b1_map.flip(0))):
        r = torch.corrcoef(torch.stack([ratio, m[support]]))[0, 1]
        print(f"corr(birdcage / ideal image, {name}) = {float(r):+.3f}")
    panels = {
        "ideal B1 and coil": img_ideal,
        "full-wave birdcage B1+, ideal coil": b1_only.abs(),
        "relative |B1+| (birdcage, quadrature)": torch.where(support, b1_map, torch.nan),
        "8 loops, correlated noise, RSS": rss_noisy,
        "8 loops, Roemer (noise-whitened)": roemer,
    }
    fig, ax = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6))
    for a, (name, img) in zip(ax, panels.items()):
        im = img.numpy()
        if "relative" in name:
            h = a.imshow(im[::-1], cmap="viridis", vmin=0.7, vmax=1.5)
            fig.colorbar(h, ax=a, fraction=0.046)
        else:
            a.imshow(im[::-1], cmap="gray", vmax=np.nanpercentile(im, 99.5))
        a.set_title(name, fontsize=9)
        a.axis("off")
    fig.suptitle(
        f"BrainWeb 04 at 3 T (127.7 MHz), T1w spoiled GRE, slab z = {float(shift[2]) * 1e3:.0f} mm: "
        "mariepy full-wave fields (5 mm)",
        fontsize=10,
    )
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=80)
    print(f"wrote {args.out} ({time.perf_counter() - t0:.1f} s total)")


if __name__ == "__main__":
    main()
