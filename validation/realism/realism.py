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

FASTMRI = Path.home() / "data" / "fastmri_brain"


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


def lowres_kspace(coil_images, n):
    """(C, N, N) coil images on the final field of view -> (C, n, n) central k-space."""
    return crop_centre(centred_fft(coil_images, (-2, -1)), (n, n))


def recon(kspace, n=None):
    """(C, n, n) centred k-space (k = 0 at index n/2) -> (n, n) RSS magnitude image."""
    if n is not None and kspace.shape[-1] != n:
        kspace = crop_centre(kspace, (n, n))
    return np.sqrt((np.abs(centred_ifft(kspace, (-2, -1))) ** 2).sum(0))


def normalise(img):
    """Scale so the 99th percentile of the head is 1 (the same function for real and simulated)."""
    return img / np.percentile(img, 99)


def fastmri_volume(path, n=128):
    """A fastMRI multicoil brain file -> (slices, n, n) images through ``recon``, and its header.

    The stored k-space is (slices, coils, readout x2 oversampled, phase); the image after the
    2D inverse FFT is cropped to the reconstruction field of view (the fastMRI convention: the
    central rows of the oversampled readout and the central columns of the phase direction),
    then reduced to the central n x n of k-space."""
    import h5py
    import ismrmrd

    with h5py.File(path, "r") as f:
        k = f["kspace"][()]
        hdr = ismrmrd.xsd.CreateFromDocument(f["ismrmrd_header"][()])
    enc = hdr.encoding[0]
    rx, ry = enc.reconSpace.matrixSize.x, enc.reconSpace.matrixSize.y
    imgs = centred_ifft(k, (-2, -1))                       # (S, C, 2*rx, ny)
    imgs = crop_centre(imgs, (rx, ry))                      # (S, C, rx, ry) on the recon FOV
    out = np.stack([recon(lowres_kspace(s, n)) for s in imgs])
    seq = hdr.sequenceParameters
    info = {"file": Path(path).name, "fov_mm": [enc.reconSpace.fieldOfView_mm.x, enc.reconSpace.fieldOfView_mm.y,
            enc.reconSpace.fieldOfView_mm.z], "recon": [rx, ry], "coils": int(k.shape[1]), "slices": int(k.shape[0]),
            "TR_ms": float(seq.TR[0]), "TE_ms": float(seq.TE[0]), "field_T": hdr.acquisitionSystemInformation.systemFieldStrength_T,
            "model": hdr.acquisitionSystemInformation.systemModel}
    return out, info


def manifest(contrast="AXT2", min_field=2.0, root=FASTMRI):
    m = json.loads((root / "manifest.json").read_text())
    return sorted(root / k for k, v in m.items() if v["contrast"] == contrast and (v.get("field_T") or 0) >= min_field
                  and "error" not in v)
