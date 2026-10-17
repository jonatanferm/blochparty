"""Realism score (ROADMAP M5, P2): real fastMRI brain scans and simulated heads through one
reconstruction, then a classifier two-sample test.

Shared reconstruction (``recon``): coil images on a square field of view ``fov`` (readout
oversampling removed), then only the central ``n`` x ``n`` of k-space is kept, so real data sample
exactly the k-space points a simulated ``n`` x ``n`` Cartesian acquisition over ``fov`` does
(k = (i - n/2) / fov); root-sum-of-squares over coils; scaled by ``normalise``. Real and simulated
images never meet a different pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blochparty.generate.sample import centred_fft, centred_ifft, crop_centre  # noqa: E402,F401 (moved there)
from blochparty.generate.sample import rss_image as recon  # noqa: E402

FASTMRI = (
    Path.home() / "data" / "fastmri_c"
)  # the compact real set (fetch_fastmri --compact); fastmri_brain until 2026-10-14


def lowres_kspace(coil_images, n):
    """(C, N, N) coil images on the final field of view -> (C, n, n) central k-space."""
    return crop_centre(centred_fft(coil_images, (-2, -1)), (n, n))


def normalise(img):
    """Scale so the 99th percentile of the head is 1 (the same function for real and simulated)."""
    return img / np.percentile(img, 99)


def manifest_key(path) -> str:
    """The manifest's key for a real scan: its fastMRI file name, also for the compact ``.npz`` form."""
    path = Path(path)
    return path.name if path.suffix == ".h5" else path.stem + ".h5"


def coil_kspace(path, n=128):
    """A real scan -> ((slices, coils, n, n) central coil k-space on the reconstruction field of view, info).

    From a raw fastMRI file (``.h5``: (slices, coils, readout x2 oversampled, phase); the image after the 2D inverse
    FFT is cropped to the reconstruction field of view, the fastMRI convention, then reduced to the central n x n of
    k-space) or from its compact form (``.npz`` written by ``fetch_fastmri.py --compact``: that k-space and the
    header's facts as JSON; ~40 MB a scan instead of ~500, since 2026-10-14). Everything downstream (the realism
    score, the noise covariances, the transfer test) uses only this."""
    path = Path(path)
    if path.suffix == ".npz":
        d = np.load(path)
        k = d["kspace"]
        if k.shape[-1] < n:
            raise ValueError(f"{path.name} holds {k.shape[-1]} x {k.shape[-1]} k-space, {n} asked")
        return crop_centre(k, (n, n)).astype(np.complex64), json.loads(str(d["info"]))
    import h5py
    import ismrmrd

    with h5py.File(path, "r") as f:
        k = f["kspace"][()]
        hdr = ismrmrd.xsd.CreateFromDocument(f["ismrmrd_header"][()])
    enc = hdr.encoding[0]
    rx, ry = enc.reconSpace.matrixSize.x, enc.reconSpace.matrixSize.y
    imgs = crop_centre(centred_ifft(k, (-2, -1)), (rx, ry))  # (S, C, rx, ry) on the recon FOV
    low = np.stack([lowres_kspace(s, n) for s in imgs]).astype(np.complex64)
    seq = hdr.sequenceParameters
    info = {
        "file": path.name,
        "fov_mm": [enc.reconSpace.fieldOfView_mm.x, enc.reconSpace.fieldOfView_mm.y, enc.reconSpace.fieldOfView_mm.z],
        "recon": [rx, ry],
        "coils": int(k.shape[1]),
        "slices": int(k.shape[0]),
        "TR_ms": float(seq.TR[0]),
        "TE_ms": float(seq.TE[0]),
        "field_T": hdr.acquisitionSystemInformation.systemFieldStrength_T,
        "model": hdr.acquisitionSystemInformation.systemModel,
    }
    return low, info


def fastmri_volume(path, n=128):
    """A real scan (raw or compact) -> (slices, n, n) images through ``recon``, and its info (``coil_kspace``)."""
    low, info = coil_kspace(path, n)
    return np.stack([recon(s) for s in low]), info


def manifest(contrast="AXT2", min_field=2.0, root=FASTMRI):
    """The real scans of ``contrast`` at ``min_field`` or above, as paths: the compact ``.npz`` where the manifest
    records one, else the raw ``.h5``."""
    m = json.loads((root / "manifest.json").read_text())
    return sorted(
        root / v.get("compact", k)
        for k, v in m.items()
        if v["contrast"] == contrast and (v.get("field_T") or 0) >= min_field and "error" not in v
    )
