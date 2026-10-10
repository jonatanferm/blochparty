"""Simulated counterparts of the fastMRI 3 T T2-weighted brain scans (realism score, ROADMAP M5).

Each seed is a ``sample_head`` head imaged with the real protocol it is paired with (TR, TE and coil
count of one real volume, in turn): 2D TSE, ETL 16 with linear ordering (effective TE = 9 echo
spacings), 5 mm slices, 220 mm field of view, 128 x 128, RF on a 10 us raster. The slice stack is
placed by the same landmarks as the real stacks: slice 0 where the lateral ventricles are largest
(CSF more than 15 mm inside the brain) and the brain top at slice 13.5, so the spacing follows the
head (5.1 mm for BrainWeb subject 04, against 5 mm slices in the real data).

The real scans read out anterior-posterior; the phantom is turned by -90 deg about z, (x, y) ->
(y, -x), so the simulated readout (x) runs anterior-posterior too and chemical shift and artefacts
point the same way. ``to_real_orientation`` turns the image back (anterior up, patient right on
the image left, as fastMRI shows it).

Noise-free k-space (coils, 128, 128; k = 0 at index 64) is saved per slice; noise is added later,
calibrated to the real images' noise floor, so the noise model can change without simulating again.

Quality (``--quality``), measured on one 64 x 64 slice against 27 spins per tissue and voxel
(whose own seed-to-seed RMS difference is 0.0073 of the normalised image):

* ``full``: 8 spins per voxel drawing their tissue (``sampling="categorical"``), a +-3 mm slab,
  RF on a 100 us raster: 1.8 M spins, RMS 0.014;
* ``fast``: 4 such spins per voxel: 0.9 M spins, RMS 0.022;
* ``legacy``: what the first runs used, 8 spins per tissue and voxel, +-4 mm, 10 us: 9.4 M spins,
  RMS 0.013, about 14 times the GPU time of ``full``.

The RF raster costs nothing in accuracy (100 us against 10 us: RMS 0.001) and the +-3 mm slab
nothing against +-4 mm; ``--spins`` overrides the spin count.

Variants (for ablations, same seeds = same heads): ``--multislice`` plays the other 15 slices'
pulses in the real interleaved order (odd slices, then even; ``tse_2d(slice_order=...)``), so the
imaged slice gets their incidental magnetization transfer and cross-talk; ``--mt`` gives the spins
their bound pool (``to_phantom(mt=True)``); ``--table tissuedb`` draws the tissues from
``tissuedb.tissue_table(3.0)`` instead of ``anatomy.DEFAULT_TISSUES``.

    uv run python validation/realism/simulate_heads.py --seeds 0-19 --out ~/data/realism_sim
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

import blochparty as bp
from blochparty.sequence import System

ROOT = Path(__file__).resolve().parents[2] / "validation" / "anatomy" / "data"
FOV, THICKNESS, SLICES = 0.22, 5e-3, 16


def real_protocols(manifest: Path) -> list[dict]:
    m = json.loads(manifest.read_text())
    return [{"file": k, "TR": v["TR_ms"] * 1e-3, "TE": v["TE_ms"] * 1e-3, "coils": v["kspace_shape"][1]}
            for k, v in sorted(m.items()) if v["contrast"] == "AXT2" and (v.get("field_T") or 0) > 2]


def stack(anatomy) -> np.ndarray:
    """z [m] of slices 0..15: slice 0 at the largest lateral-ventricle cross-section, the brain
    top at slice 13.5 (how the real stacks sit: ventricles in slice 0, a scalp cap in slice 15)."""
    csf = anatomy.fractions["csf"].numpy()
    brain = (anatomy.fractions["wm"] + anatomy.fractions["gm"]).numpy()
    z = anatomy.origin[2] + np.arange(anatomy.shape[0]) * anatomy.voxel[0]
    inside = ndimage.binary_erosion((brain + csf) > 0.5, iterations=round(15e-3 / anatomy.voxel[1]))
    deep = (csf * inside).reshape(len(z), -1).sum(1)
    top = z[np.nonzero(brain.reshape(len(z), -1).sum(1) * np.prod(anatomy.voxel[1:]) > 50e-6)[0].max()]
    z0 = z[np.argmax(deep)]
    return z0 + np.arange(SLICES) * (top - z0) / 13.5


def turn(ph: bp.Phantom) -> bp.Phantom:
    """(x, y, z) -> (y, -x, z): anterior-posterior becomes the readout axis."""
    pos = torch.stack([ph.pos[:, 1], -ph.pos[:, 0], ph.pos[:, 2]], -1)
    ph.pos = pos
    return ph


def to_real_orientation(img: np.ndarray) -> np.ndarray:
    """Simulated (y', x') image -> fastMRI display: rows anterior to posterior, columns patient right
    to left."""
    return img.T[::-1, :]


def interleaved(count=SLICES) -> list[int]:
    """Siemens' default interleaved order: slices 1, 3, 5, ... then 2, 4, ... (0-based 0, 2, ..., 1, 3, ...)."""
    return list(range(0, count, 2)) + list(range(1, count, 2))


QUALITY = {  # spins per voxel (z, y, x), sampling, slab half-width [voxels], RF raster [s]
    "full": ((2, 2, 2), "categorical", 3, 1e-4),
    "fast": ((1, 2, 2), "categorical", 3, 1e-4),
    "legacy": ((2, 2, 2), "per_tissue", 4, 1e-5),
}


def sequence(n: int, tr: float, te: float, slice_order=None, rf_raster=1e-4, spacing=None) -> bp.Sequence:
    """The paired protocol. With ``slice_order``: the neighbours as their real pulses, the rest on
    the bound pool only (0.71 % against every pulse exact, ``tse_2d``)."""
    multi = {} if slice_order is None else dict(bound_only_beyond=1.5 * spacing)
    return bp.tse_2d(n, FOV, etl=16, esp=round(te / 9, 5), tr=tr, n_dummy=1, slice_thickness=THICKNESS,
                     spoiling="gradient", ramps=True, system=System(rf_raster=rf_raster), slice_order=slice_order,
                     **multi)


def simulate(ph, seq, coils):
    from blochparty import metal

    if metal.available():
        out = metal.simulate_metal(ph.to(torch.float32), seq.to(torch.float32), coils.to(torch.complex64),
                                   readouts="nufft", pulses="reuse")  # PERFORMANCE 2.1 and 2.3 (exact)
        return out.signal.to(torch.complex128), "metal"
    return bp.simulate(ph, seq, coils).signal, "core"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--slices", default="0-12")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--subject", type=int, default=None)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--manifest", type=Path, default=Path.home() / "data" / "fastmri_brain" / "manifest.json")
    ap.add_argument("--out", type=Path, default=Path.home() / "data" / "realism_sim")
    ap.add_argument("--multislice", action="store_true", help="play the other 15 slices' pulses")
    ap.add_argument("--mt", action="store_true", help="bound pool on (to_phantom(mt=True))")
    ap.add_argument("--table", choices=("default", "tissuedb"), default="default")
    ap.add_argument("--quality", choices=tuple(QUALITY), default="full")
    ap.add_argument("--spins", default=None, help="override: spins per voxel along z,y,x")
    args = ap.parse_args()

    def span(s):  # "a-b" or "a-b:step"
        rng, _, step = s.partition(":")
        return range(int(rng.split("-")[0]), int(rng.split("-")[-1]) + 1, int(step or 1))

    spins, sampling, half, rf_raster = QUALITY[args.quality]
    if args.spins:
        spins = tuple(int(c) for c in args.spins.split(","))
    table = bp.DEFAULT_TISSUES
    if args.table == "tissuedb":
        from blochparty import tissuedb

        table = {3.0: tissuedb.tissue_table(3.0)}
    protocols = real_protocols(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in span(args.seeds):
        dest = args.out / f"sim_{seed:03d}.npz"
        if dest.exists():
            continue
        t0 = time.perf_counter()
        proto = protocols[seed % len(protocols)]
        head = bp.sample_head(seed, subject=args.subject, root=args.root, voxel=(1e-3,) * 3, table=table)
        fields = head.field_maps()
        zs = stack(head.anatomy)
        kspace, timing = {}, []

        def prepare(i):
            """Slice i's phantom, coils and merged sequence (CPU), built while the GPU runs slice i - 1."""
            order = [float(zs[j] - zs[i]) for j in interleaved()] if args.multislice else None
            seq = sequence(args.n, proto["TR"], proto["TE"], order, rf_raster, float(zs[1] - zs[0]))
            k = round((zs[i] - head.anatomy.origin[2]) / head.anatomy.voxel[0])
            region = (slice(max(k - half, 0), k + half + 1), slice(None), slice(None))
            ph = head.to_phantom(fov_region=region, field_maps=fields, spins_per_voxel=spins, sampling=sampling,
                                 mt=args.mt)
            ph.pos = ph.pos - torch.tensor([0.0, 0.0, zs[i]], dtype=ph.pos.dtype)
            ph = turn(ph)
            return seq, ph, bp.birdcage_sensitivities(ph.pos, proto["coils"])

        slices = list(span(args.slices))
        with ThreadPoolExecutor(max_workers=1) as pool:  # PERFORMANCE 2.7: CPU and GPU overlap
            pending = pool.submit(prepare, slices[0])
            for n, i in enumerate(slices):
                t = time.perf_counter()
                seq, ph, coils = pending.result()
                if n + 1 < len(slices):
                    pending = pool.submit(prepare, slices[n + 1])
                signal, engine = simulate(ph, seq.merge_free(), coils)  # exact for static spins (PERFORMANCE 2.4)
                kspace[f"slice{i:02d}"] = bp.cartesian_kspace(signal, seq).numpy().astype(np.complex64)
                timing.append(time.perf_counter() - t)
                print(f"seed {seed} slice {i}: {len(ph):,} spins, {timing[-1]:.1f} s ({engine})", flush=True)
        meta = {"seed": seed, "subject": head.meta["subject"], "protocol": proto, "z_mm": (zs * 1e3).round(2).tolist(),
                "multislice": args.multislice, "mt": args.mt, "table": args.table, "quality": args.quality,
                "spins": list(spins), "sampling": sampling, "slab_voxels": 2 * half + 1, "rf_raster": rf_raster,
                "jacobian": head.meta["deformation"]["jacobian"], "n": args.n, "readouts": "nufft", "pulses": "reuse",
                "seconds": round(time.perf_counter() - t0)}
        np.savez(dest, meta=json.dumps(meta), **kspace)
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
