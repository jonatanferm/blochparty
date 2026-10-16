"""The generation driver (ROADMAP M7): a spec and a seed -> one simulated sample with its truth, written with provenance.

    for s in $(seq 0 299); do uv run python validation/realism/generate.py --spec spec.json --seeds $s-$s --out ~/data/gen_t2w; done

One head per process: a process that ran 60 heads on the mini began returning all-zero signals from the GPU after
~30 slices (2026-10-14); the driver refuses to write such a sample, so a per-head loop can simply retry.

A sample is one slice of multi-coil k-space as a scanner would store it (ISMRMRD HDF5 via ``blochparty.io.write_dataset``:
the signal in sequence order, the trajectory, the sequence, the phantom's draws), plus a ``/truth`` group a scanner never
has: the noise-free k-space and image on the acquisition grid, the noisy k-space the sample "acquired", the tissue
fractions and the residual maps on that grid, the lesion mask, and ``meta.json`` (every draw: anatomy, protocol,
scanner, quality). One JSONL manifest line per sample. The same spec, seed and engine version regenerate a sample.

The spec (JSON) says what the brain and the scan are; the seed fills in what it leaves open:

    {"anatomy": {"source": "ixi", "prepared": "validation/anatomy/data/ixi/prepared_v16", "subject": null,
                 "lesions": {"count": 3, "size_mm": [3, 10], "region": "periventricular"}},
     "protocol": {"family": "fastmri_t2w", "manifest": "~/data/fastmri_more/manifest.json", "n": 128, "slices": "2-6:2"},
     "scanner": {"receive": "draw", "b1": "birdcage", "noise": "paired"},
     "quality": {"spins": [2, 3, 3]}}

anatomy: an IXI head (``subject`` null: drawn from the prepared directory, or from ``subjects`` if the spec lists a pool), the realism preset's anatomy options
(deep grey, falx, the fitted residual, native size, drawn yaw / pitch) and ``lesions`` (``anatomy.add_lesions``
keywords; null for none; "count": [lo, hi] draws the count per head, 0 meaning none). protocol: ``fastmri_t2w`` pairs the sample with one real scan's header (TR, TE, refocusing
angle, coil count, echo spacing: ``simulate_heads.real_protocols``) and plays its 21-echo train over 16 slices; ``n``
the matrix on the 220 mm field of view; ``slices`` the stack indices (a-b:step) or {"count": k} for k drawn levels.
scanner: ``receive`` "draw" (the elliptical normalised array drawn per head, simulate_heads 2026-10-13; ``receive_scale``
the semi-axes' scale range, re-centred on the 65-scan set's scalp level on 2026-10-15) or "ring"
(the 14 cm ring); ``b1`` "birdcage" or "uniform"; ``noise`` "paired" (the paired scan's background floor and coil
covariance, as score.py adds it) or "none". quality: spins per voxel (z, y, x); 2 x 3 x 3 keeps the sub-voxel grain at
the real scans' level (journal 2026-10-14).

The anatomy and scan code is simulate_heads' (imported); its command-line preset is reproduced here as data.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blochparty as bp  # noqa: E402
from blochparty.io import write_dataset  # noqa: E402
import realism as R  # noqa: E402
import score as SC  # noqa: E402
import simulate_heads as SH  # noqa: E402

DEFAULT_SPEC = {
    "anatomy": {"source": "ixi", "prepared": None, "subject": None, "lesions": None, "deform_mm": 3.0},
    "protocol": {"family": "fastmri_t2w", "manifest": str(R.FASTMRI / "manifest.json"),
                 "n": 128, "slices": {"count": 3}},
    "scanner": {"receive": "draw", "b1": "birdcage", "noise": "paired", "receive_scale": [1.1, 1.45],
                "coil_radius": 0.11, "coil_radius_lr": 0.095, "coil_gap_mm": 10., "receive_normalise_mm": 30.},
    "quality": {"spins": [2, 3, 3], "sampling": "proportional", "jitter": "inplane", "slab_voxels": 3, "rf_raster": 1e-4},
}


def merged(spec: dict) -> dict:
    out = {k: dict(v) for k, v in DEFAULT_SPEC.items()}
    for k, v in (spec or {}).items():
        out.setdefault(k, {}).update(v)
    return out


def head_for(spec: dict, seed: int, table) -> bp.DigitalPhantom:
    """The realism preset's anatomy (simulate_heads --preset realistic --source ixi --residual) for this seed."""
    a = spec["anatomy"]
    prepared = None if a.get("prepared") is None else Path(a["prepared"]).expanduser()
    subject = a.get("subject")
    if subject is None and a.get("subjects"):  # a pool of subjects (e.g. a held-out set for evaluation), drawn by the seed
        pool = list(a["subjects"])
        subject = int(pool[int(np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(17)[16]).integers(len(pool)))])
    head = bp.sample_head(seed, subject=subject, root=SH.ROOT, voxel=(1e-3,) * 3, table=table, scalp="brainweb",
                          ixi_prepared=prepared, size="native", yaw_sd=3.8, age=None, deep_grey=True,
                          magnitude=a.get("deform_mm", 3.0) * 1e-3, ventricle_age=None, pitch=(-10., 5.), source=a["source"],
                          falx=True, interpolation=3, residual=True, dermis=False, lesions=a.get("lesions"), texture={})
    SH.drop_detached(head)
    SH.flow_void(head)
    return head


def truth_on_grid(head, zs_i: float, thickness: float, n: int, fov: float, cx: float, cy: float) -> dict:
    """The anatomy's fractions and residual maps averaged over the slice slab and resampled onto the acquisition grid,
    in the image's display orientation (rows posterior -> anterior, columns the patient's left -> right, i.e.
    anatomy -x: simulate_heads.turn and to_real_orientation, checked numerically). Returns {class: (n, n)} plus the residual maps if the head carries them."""
    from scipy import ndimage

    an = head.anatomy
    z = an.origin[2] + np.arange(an.shape[0]) * an.voxel[0]
    k = np.nonzero(np.abs(z - zs_i) <= thickness / 2 + 1e-9)[0]
    if len(k) == 0:
        k = np.array([int(np.argmin(np.abs(z - zs_i)))])
    # the image's pixel (row, column) in the anatomy frame: rows run along +y (posterior at the top, anterior at the
    # bottom as fastMRI displays), columns along -x; both checked against the simulated image (below)
    px = fov / n
    u = (np.arange(n) - n / 2 + .5) * px  # image coordinates [m] from the field-of-view centre
    rows, cols = np.meshgrid(u, u, indexing="ij")
    # display rows: posterior (top) -> anterior (bottom) = anatomy +y upward in the array (rows increase toward anterior)
    ya = cy + rows
    xa = cx - cols  # columns run from the patient's left (+x in the anatomy frame) to the right: checked against the
    # simulated image (the flipped grid correlated 0.74 with a T2w proxy of the fractions, the unflipped 0.68; smoke test)
    iy = (ya - an.origin[1]) / an.voxel[1]
    ix = (xa - an.origin[0]) / an.voxel[2]
    out = {}
    for name, frac in an.fractions.items():
        slab = frac.numpy()[k].mean(0)
        out[name] = ndimage.map_coordinates(slab, [iy.ravel(), ix.ravel()], order=1, mode="constant", cval=0.).reshape(n, n)
    for name in ("t2_scale", "pd_scale"):
        m = getattr(an, "maps", {}).get(name) if getattr(an, "maps", None) else None
        if m is not None:
            slab = np.asarray(m)[k].mean(0)
            out[name] = ndimage.map_coordinates(slab, [iy.ravel(), ix.ravel()], order=1, mode="constant", cval=1.).reshape(n, n)
    return out


def add_noise(k: np.ndarray, floor: float, cov, rng) -> tuple[np.ndarray, float]:
    """score.sim_set's recipe: complex Gaussian noise at the paired scan's background floor (relative to the clean
    image's p99), coloured by its coil covariance when the coil count matches. Returns (noisy k-space, sigma)."""
    c = k.shape[0]
    chol = None if cov is None or cov.shape[0] != c else np.linalg.cholesky(cov * c / np.real(np.trace(cov)))
    unit = SC.unit_floor(c, n=k.shape[-1], chol=chol)
    clean = R.recon(k)
    sigma = floor * np.percentile(clean, 99) / unit
    noise = rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape)
    if chol is not None:
        noise = np.tensordot(chol, noise, 1)
    return k + sigma * noise, float(sigma)


def generate(spec: dict, seed: int, out: Path, *, table, protocols: list[dict], floors: dict, covs: dict | None,
             serial: bool = False) -> list[dict]:
    """One head for ``seed``: its slices simulated, noised, written as ``out/sample_<seed>_<slice>.h5``; returns the
    manifest records."""
    spec = merged(spec)
    a, p, s, q = spec["anatomy"], spec["protocol"], spec["scanner"], spec["quality"]
    streams = np.random.SeedSequence(int(seed)).spawn(4)
    rng_proto, rng_noise, rng_slices = (np.random.default_rng(x) for x in streams[:3])
    proto = protocols[int(rng_proto.integers(len(protocols)))]
    lesions = a.get("lesions")
    if lesions and isinstance(lesions.get("count"), (list, tuple)):  # "count": [lo, hi]: drawn per head, 0 = no lesions
        lo, hi = lesions["count"]
        count = int(np.random.default_rng(streams[3]).integers(lo, hi + 1))
        spec["anatomy"]["lesions"] = (dict(lesions, count=count) if count > 0 else None)
    head = head_for(spec, seed, table)
    fields = head.field_maps()
    zs = SH.stack(head.anatomy, "protocol", SH.MONRO_BELOW_BODIES + 2.5e-3, head.meta["ventricle_landmark"][0])
    cx, cy = SH.centre(head.anatomy, zs)
    cy += 4.8e-3  # the real operators' placement (journal 2026-10-13)
    # the receive
    coil_radius, coil_radius_lr, gap_mm, norm_mm = s["coil_radius"], s["coil_radius_lr"], s["coil_gap_mm"], s["receive_normalise_mm"]
    draw = None
    if s["receive"] == "draw":
        rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(15)[14])
        lo, hi = s.get("receive_scale", (1.1, 1.45))
        draw = {"radius_scale": float(rng.uniform(lo, hi)), "gap_mm": float(rng.uniform(5., 30.)),
                "normalise_mm": float(rng.uniform(20., 60.))}
        coil_radius, coil_radius_lr = coil_radius * draw["radius_scale"], coil_radius_lr * draw["radius_scale"]
        gap_mm, norm_mm = draw["gap_mm"], draw["normalise_mm"]
    elif s["receive"] == "ring":
        coil_radius, coil_radius_lr, gap_mm, norm_mm = 0.14, None, 25., None
    ring = SH.coil_offset(head.anatomy, zs, cy, coil_radius, gap_mm * 1e-3)
    # the slices
    if isinstance(p["slices"], dict):
        count = int(p["slices"]["count"])
        slices = sorted(rng_slices.choice(np.arange(1, 8), size=min(count, 7), replace=False).tolist())
    else:
        slices = _span(p["slices"])
    spins = tuple(q["spins"])
    half = q["slab_voxels"]
    n = int(p["n"])

    def prepare(i):
        order = [float(zs[j] - zs[i]) for j in SH.interleaved()]
        seq = SH.sequence(n, proto["TR"], proto["TE"], order, q["rf_raster"], float(zs[1] - zs[0]), 2,
                          proto["refocus_deg"], "linear_full", etl=SH.REAL_ETL, esp=proto["esp"])
        k = round((zs[i] - head.anatomy.origin[2]) / head.anatomy.voxel[0])
        region = (slice(max(k - half, 0), k + half + 1), slice(None), slice(None))
        ph = head.to_phantom(fov_region=region, field_maps=fields, spins_per_voxel=spins, sampling=q["sampling"],
                             jitter=q["jitter"], mt=True)
        if s["b1"] == "birdcage":
            ph.b1 = SH.b1_plus(head, ph.pos).to(ph.pos.dtype)
        ph.pos = ph.pos - torch.tensor([cx, cy, zs[i]], dtype=ph.pos.dtype)
        ph = SH.turn(ph)
        ring_pos = ph.pos - torch.tensor([ring, 0.0, 0.0], dtype=ph.pos.dtype) if ring else ph.pos
        sens = bp.birdcage_sensitivities(ring_pos, proto["coils"], radius=coil_radius, radius_y=coil_radius_lr)
        if norm_mm:
            sens = SH.normalised_sensitivities(sens, ph.pos, norm_mm)
        return seq, ph, sens

    import h5py
    out.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=1) as pool:  # the next slice's phantom built while the GPU runs this one
        pending = None if serial else pool.submit(prepare, slices[0])
        for j, i in enumerate(slices):
            t0 = time.perf_counter()
            seq, ph, coils = prepare(i) if serial else pending.result()
            if j + 1 < len(slices) and not serial:
                pending = pool.submit(prepare, slices[j + 1])
            signal, engine = SH.simulate(ph, seq.merge_free(), coils)
            k_clean = SH.kspace_on_fov(signal, seq).astype(np.complex64)
            if not np.isfinite(k_clean).all() or np.abs(k_clean).max() == 0:
                # 2026-10-14: a long-running process on the mini started returning all-zero signals from the GPU after
                # ~30 slices (41 of 181 samples, not reproducible head by head); never write such a sample
                raise RuntimeError(f"the engine returned {'no' if np.abs(k_clean).max() == 0 else 'a non-finite'} signal for seed {seed} "
                                   f"slice {i} ({engine}): run one head per process and retry")
            floor = floors.get(proto["file"], float(np.median(list(floors.values()))))
            cov = None if covs is None or s["noise"] != "paired" else covs.get(proto["file"])
            k_noisy, sigma = (add_noise(k_clean.astype(np.complex128), floor, cov, rng_noise) if s["noise"] != "none"
                              else (k_clean.astype(np.complex128), 0.))
            image = SH.to_real_orientation(R.recon(k_clean.astype(np.complex128))).astype(np.float32)
            truth = truth_on_grid(head, float(zs[i]), SH.THICKNESS, n, SH.FOV, cx, cy)
            path = out / f"sample_{seed:06d}_{i:02d}.h5"
            meta = {"seed": int(seed), "slice": int(i), "z_mm": round(float(zs[i]) * 1e3, 2), "protocol": proto, "engine": engine,
                    "spec": spec, "head": {k: v for k, v in head.meta.items() if k not in ("draws", "streams")},
                    "draws": head.meta.get("draws"), "receive": {"radius": coil_radius, "radius_lr": coil_radius_lr, "gap_mm": gap_mm,
                                                                  "normalise_mm": norm_mm, "offset_mm": round(ring * 1e3, 1), "draw": draw},
                    "centre_m": [cx, cy], "noise": {"floor": floor, "sigma": sigma, "coloured": cov is not None},
                    "spins": len(ph), "seconds": round(time.perf_counter() - t0, 1)}  # write_dataset records the git commit and versions
            write_dataset(path, signal, seq, phantom=head, coils={"model": "birdcage", "n": proto["coils"], **meta["receive"]},
                                noise=meta["noise"], seed=int(seed), engine=engine)
            with h5py.File(path, "a") as f:
                g = f.create_group("truth")
                g.create_dataset("kspace_clean", data=k_clean)
                g.create_dataset("kspace", data=k_noisy.astype(np.complex64))
                g.create_dataset("image_clean", data=image)
                classes = sorted(c for c in truth if c not in ("t2_scale", "pd_scale"))
                g.create_dataset("fractions", data=np.stack([truth[c] for c in classes]).astype(np.float32))
                g.attrs["classes"] = json.dumps(classes)
                for name in ("t2_scale", "pd_scale"):
                    if name in truth:
                        g.create_dataset(name, data=truth[name].astype(np.float32))
                g.create_dataset("lesion", data=(truth["lesion"] if "lesion" in truth else np.zeros((n, n))).astype(np.float32))
                g.attrs["meta.json"] = json.dumps(meta, default=str)
                g.attrs["orientation"] = "rows posterior -> anterior, columns the patient's left -> right (the fastMRI display)"
            rec = {"file": path.name, "seed": int(seed), "slice": int(i), "subject": head.meta.get("subject"),
                   "protocol": proto["file"], "coils": proto["coils"], "lesions": len((head.meta.get("lesions") or {}).get("placed", [])),
                   "spins": len(ph), "seconds": meta["seconds"], "engine": engine}
            records.append(rec)
            with open(out / "manifest.jsonl", "a") as m:
                m.write(json.dumps(rec) + "\n")
            print(f"seed {seed} slice {i}: {len(ph):,} spins, {meta['seconds']} s ({engine}) -> {path.name}", flush=True)
            del seq, ph, coils, signal
    return records


FALLBACK_PROTOCOLS = [{"file": "fastmri-like (no manifest)", "TR": 6.19, "TE": 0.113, "coils": 16, "refocus_deg": 180., "esp": 10.26e-3}]


def _span(s):
    a, _, step = str(s).partition(":")
    lo, _, hi = a.partition("-")
    return list(range(int(lo), int(hi or lo) + 1, int(step or 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, default=None, help="JSON spec (module docstring); defaults without it")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path.home() / "data" / "realism_cache")
    ap.add_argument("--fastmri", type=Path, default=None, help="real scans (floors, coil covariances); default from the spec's manifest")
    ap.add_argument("--set", action="append", default=[], help="override spec entries: section.key=json, e.g. protocol.n=64")
    ap.add_argument("--serial", action="store_true", help="build each slice's phantom after the previous one is simulated (half the memory)")
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text()) if args.spec else {}
    for item in args.set:
        key, _, value = item.partition("=")
        sec, k = key.split(".")
        spec.setdefault(sec, {})[k] = json.loads(value)
    spec = merged(spec)
    manifest = None if spec["protocol"]["manifest"] is None else Path(spec["protocol"]["manifest"]).expanduser()
    from blochparty import tissuedb
    table = {3.0: tissuedb.tissue_table(3.0)}
    if manifest is not None and manifest.exists():
        root = args.fastmri or manifest.parent
        protocols = SH.real_protocols(manifest)
        real, floors = SC.real_set(args.cache, root, range(1, 8))
        covs = SC.noise_covariances(args.cache, root) if spec["scanner"]["noise"] == "paired" else None
        del real
    else:  # no real scans at hand: one fastMRI-like protocol, the typical background floor, white noise
        protocols, floors, covs = FALLBACK_PROTOCOLS, {FALLBACK_PROTOCOLS[0]["file"]: 0.017}, None
        print("no fastMRI manifest: the fallback protocol, white noise at a floor of 0.017", flush=True)
    (args.out).mkdir(parents=True, exist_ok=True)
    (args.out / "spec.json").write_text(json.dumps(spec, indent=1))
    for seed in _span(args.seeds):
        generate(spec, seed, args.out, table=table, protocols=protocols, floors=floors, covs=covs, serial=args.serial)


if __name__ == "__main__":
    main()
