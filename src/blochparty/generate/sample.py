"""Spec + seed -> simulated, noised samples with their truth on the acquisition grid (``samples``), and their files
(``write``). A sample is one slice of multi-coil k-space as a scanner would store it, plus what a scanner never
has: the noise-free k-space and image, the tissue fractions and residual maps, the lesion mask, every draw."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import blochparty as bp

from ..coils import coil_offset, normalised_sensitivities
from ..io import write_dataset
from .heads import MONRO_BELOW_BODIES, b1_plus, centre, head_for, stack, turn
from .protocols import FOV, REAL_ETL, THICKNESS, interleaved, sequence
from .spec import merged
from .spec import span as _span


def centred_fft(x, axes):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=axes), axes=axes, norm="ortho"), axes=axes)


def centred_ifft(k, axes):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(k, axes=axes), axes=axes, norm="ortho"), axes=axes)


def crop_centre(x, shape, axes=(-2, -1)):
    sl = [slice(None)] * x.ndim
    for ax, n in zip(axes, shape):
        start = (x.shape[ax] - n) // 2
        sl[ax] = slice(start, start + n)
    return x[tuple(sl)]


def rss_image(kspace, n=None):
    """(C, n, n) centred k-space (k = 0 at index n/2) -> (n, n) RSS magnitude image."""
    if n is not None and kspace.shape[-1] != n:
        kspace = crop_centre(kspace, (n, n))
    return np.sqrt((np.abs(centred_ifft(kspace, (-2, -1))) ** 2).sum(0))


def unit_floor(coils, n=128, rng=None, chol=None):
    """Background median of the RSS image of unit-sigma complex noise in ``coils`` channels, coloured by ``chol`` (the
    Cholesky factor of a coil covariance scaled to mean variance 1) if given."""
    rng = rng or np.random.default_rng(0)
    k = (rng.standard_normal((coils, n, n)) + 1j * rng.standard_normal((coils, n, n)))
    if chol is not None:
        k = np.tensordot(chol, k, 1)
    return float(np.median(rss_image(k)))


def to_real_orientation(img: np.ndarray) -> np.ndarray:
    """Simulated (y', x') image -> fastMRI display: rows posterior to anterior (fastMRI's stored images
    show the head posterior up: validation/realism/orientation.py), columns from the patient's right
    (+x) to left (left-right is not established against the real data)."""
    return img.T


def kspace_on_fov(signal, seq) -> np.ndarray:
    """(C, n, n) k-space on the field of view: an oversampled readout is cropped to the central n
    pixels after the inverse FFT, as the real reconstruction crops the scanner's 2x readout."""
    k = bp.cartesian_kspace(signal, seq).numpy()
    n, m = k.shape[-2], k.shape[-1]
    if m != n:  # centred, orthonormal transforms as centred_ifft / centred_fft
        img = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(k, axes=-1), axis=-1, norm="ortho"), axes=-1)
        img = img[..., (m - n) // 2:(m - n) // 2 + n]
        k = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(img, axes=-1), axis=-1, norm="ortho"), axes=-1)
    return k


def simulate(ph, seq, coils):
    if bp.engines.available("gpu"):
        out = bp.simulate(ph.to(torch.float32), seq.to(torch.float32), coils.to(torch.complex64), engine="gpu",
                          readouts="nufft", pulses="reuse")  # PERFORMANCE 2.1 and 2.3 (exact)
        return out.signal.to(torch.complex128), bp.engines.resolve("gpu")
    return bp.simulate(ph, seq, coils).signal, "core"


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
    """The realism benchmark's recipe (score.sim_set): complex Gaussian noise at the paired scan's background floor (relative to the clean
    image's p99), coloured by its coil covariance when the coil count matches. Returns (noisy k-space, sigma)."""
    c = k.shape[0]
    chol = None if cov is None or cov.shape[0] != c else np.linalg.cholesky(cov * c / np.real(np.trace(cov)))
    unit = unit_floor(c, n=k.shape[-1], chol=chol)
    clean = rss_image(k)
    sigma = floor * np.percentile(clean, 99) / unit
    noise = rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape)
    if chol is not None:
        noise = np.tensordot(chol, noise, 1)
    return k + sigma * noise, float(sigma)


def samples(spec: dict, seed: int, *, root, table, protocols: list[dict], floors: dict, covs: dict | None,
            serial: bool = False):
    """One head for ``seed``, and one sample per slice of it, simulated and noised (an iterator of dicts; ``write``
    stores one). ``root``: the anatomy data; ``table``: the tissue table (``tissuedb.tissue_table``); ``protocols``:
    the paired scans' headers (``protocols.real_protocols``, or ``spec.FALLBACK_PROTOCOLS``); ``floors`` and ``covs``:
    their background noise floors and coil covariances by file. The next slice's phantom is built while the GPU
    runs this one (``serial``: one after the other, half the memory)."""
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
    head = head_for(spec, seed, table, root)
    fields = head.field_maps()
    zs = stack(head.anatomy, "protocol", MONRO_BELOW_BODIES + 2.5e-3, head.meta["ventricle_landmark"][0])
    cx, cy = centre(head.anatomy, zs)
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
    ring = coil_offset(head.anatomy, zs, cy, coil_radius, gap_mm * 1e-3)
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
        order = [float(zs[j] - zs[i]) for j in interleaved()]
        seq = sequence(n, proto["TR"], proto["TE"], order, q["rf_raster"], float(zs[1] - zs[0]), 2,
                          proto["refocus_deg"], "linear_full", etl=REAL_ETL, esp=proto["esp"])
        k = round((zs[i] - head.anatomy.origin[2]) / head.anatomy.voxel[0])
        region = (slice(max(k - half, 0), k + half + 1), slice(None), slice(None))
        ph = head.to_phantom(fov_region=region, field_maps=fields, spins_per_voxel=spins, sampling=q["sampling"],
                             jitter=q["jitter"], mt=True)
        if s["b1"] == "birdcage":
            ph.b1 = b1_plus(head, ph.pos, root).to(ph.pos.dtype)
        ph.pos = ph.pos - torch.tensor([cx, cy, zs[i]], dtype=ph.pos.dtype)
        ph = turn(ph)
        ring_pos = ph.pos - torch.tensor([ring, 0.0, 0.0], dtype=ph.pos.dtype) if ring else ph.pos
        sens = bp.birdcage_sensitivities(ring_pos, proto["coils"], radius=coil_radius, radius_y=coil_radius_lr)
        if norm_mm:
            sens = normalised_sensitivities(sens, ph.pos, norm_mm)
        return seq, ph, sens

    with ThreadPoolExecutor(max_workers=1) as pool:  # the next slice's phantom built while the GPU runs this one
        pending = None if serial else pool.submit(prepare, slices[0])
        for j, i in enumerate(slices):
            t0 = time.perf_counter()
            seq, ph, coils = prepare(i) if serial else pending.result()
            if j + 1 < len(slices) and not serial:
                pending = pool.submit(prepare, slices[j + 1])
            signal, engine = simulate(ph, seq.merge_free(), coils)
            k_clean = kspace_on_fov(signal, seq).astype(np.complex64)
            if not np.isfinite(k_clean).all() or np.abs(k_clean).max() == 0:
                # 2026-10-14: a long-running process on the mini started returning all-zero signals from the GPU after
                # ~30 slices (41 of 181 samples, not reproducible head by head); never write such a sample
                raise RuntimeError(f"the engine returned {'no' if np.abs(k_clean).max() == 0 else 'a non-finite'} signal for seed {seed} "
                                   f"slice {i} ({engine}): run one head per process and retry")
            floor = floors.get(proto["file"], float(np.median(list(floors.values()))))
            cov = None if covs is None or s["noise"] != "paired" else covs.get(proto["file"])
            k_noisy, sigma = (add_noise(k_clean.astype(np.complex128), floor, cov, rng_noise) if s["noise"] != "none"
                              else (k_clean.astype(np.complex128), 0.))
            image = to_real_orientation(rss_image(k_clean.astype(np.complex128))).astype(np.float32)
            truth = truth_on_grid(head, float(zs[i]), THICKNESS, n, FOV, cx, cy)
            meta = {"seed": int(seed), "slice": int(i), "z_mm": round(float(zs[i]) * 1e3, 2), "protocol": proto, "engine": engine,
                    "spec": spec, "head": {k: v for k, v in head.meta.items() if k not in ("draws", "streams")},
                    "draws": head.meta.get("draws"), "receive": {"radius": coil_radius, "radius_lr": coil_radius_lr, "gap_mm": gap_mm,
                                                                  "normalise_mm": norm_mm, "offset_mm": round(ring * 1e3, 1), "draw": draw},
                    "centre_m": [cx, cy], "noise": {"floor": floor, "sigma": sigma, "coloured": cov is not None},
                    "spins": len(ph), "seconds": round(time.perf_counter() - t0, 1)}  # write_dataset records the git commit and versions
            rec = {"file": f"sample_{seed:06d}_{i:02d}.h5", "seed": int(seed), "slice": int(i), "subject": head.meta.get("subject"),
                   "protocol": proto["file"], "coils": proto["coils"], "lesions": len((head.meta.get("lesions") or {}).get("placed", [])),
                   "spins": len(ph), "seconds": meta["seconds"], "engine": engine}
            yield {"name": rec["file"], "signal": signal, "seq": seq, "head": head, "k_clean": k_clean,
                   "k_noisy": k_noisy, "image": image, "truth": truth, "meta": meta, "record": rec}
            del seq, ph, coils, signal


def write(s: dict, out: Path) -> Path:
    """Store one sample of ``samples`` as ``out / s["name"]``: the raw data as a scanner would (ISMRMRD HDF5 through
    ``io.write_dataset``, with the sequence and the phantom's draws) and a ``/truth`` group (noise-free and noisy
    k-space, the image, the tissue fractions and residual maps on the grid, the lesion mask, ``meta.json``)."""
    import h5py

    out.mkdir(parents=True, exist_ok=True)
    path = out / s["name"]
    signal, seq, head, meta, truth = s["signal"], s["seq"], s["head"], s["meta"], s["truth"]
    k_clean, k_noisy, image, n = s["k_clean"], s["k_noisy"], s["image"], s["k_clean"].shape[-1]
    proto, seed = meta["protocol"], meta["seed"]
    write_dataset(path, signal, seq, phantom=head, coils={"model": "birdcage", "n": proto["coils"], **meta["receive"]},
                  noise=meta["noise"], seed=int(seed), engine=meta["engine"])
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
    return path
