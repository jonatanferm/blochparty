"""Shared pieces of the NIST-phantom realism benchmarks: reference relaxation values, the
authors' reconstruction, sphere geometry, the digital phantom, RF coarsening and metrics.

Reference values (all 3 T, 20 °C unless noted):

- T1 of the NiCl2 (T1-plane) spheres: CaliberMRI/NIST, phantom version 1 (serial < 43) and 2
  (serial >= 43), from the RRSG 2020 analysis code (github.com/rrsg2020/analysis, src/nist.py).
  Version 2 also equals the phantom-manual values quoted in Carr et al., J Appl Clin Med Phys
  2021 (doi:10.1002/acm2.13432), Table 2.
- T2 of the MnCl2 (T2-plane) spheres: phantom manual via Carr et al. 2021, Table 2; spheres 1
  and 5 are withheld there on the manufacturer's advice, so Jiang et al., MRM 2016
  (PMC5408299, Table 1, temperature unstated) fills those two.
- The cross terms, which no table gives (T2 of NiCl2 spheres, T1 of MnCl2 spheres), follow from
  the 3 T relaxivities of the commercial phantom at 20 °C (Martin et al., MAGMA 2023,
  doi:10.1007/s10334-023-01086-y, Table 1): with R = 1/T and water rates R_w,
  R2(NiCl2) = R2_w + (r2/r1)(R1 - R1_w), R1(MnCl2) = R1_w + (r1/r2)(R2 - R2_w).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import torch
from scipy import ndimage

import blochparty as bp

N, FOV, SLICE = 128, 0.250, 6e-3  # both benchmark sequences
PIXEL = FOV / N
CENTRE = 63.5  # pixel index of position 0 in the authors' reconstruction (measured by simulating a point)

T1_NICL2 = {  # [ms]
    "v1": [1989, 1454, 984.1, 706, 496, 351.5, 247.13, 175.3, 125.9, 89.0, 62.7, 44.53, 30.84, 21.719],
    "v2": [1883.97, 1330.16, 987.27, 690.08, 484.97, 341.58, 240.86, 174.95, 121.08, 85.75, 60.21, 42.89,
           30.40, 21.44],
}
T2_MNCL2 = [581.0, 379.5, 267.3, 175.1, 143.0, 88.9, 63.4, 44.2, 29.9, 19.4, 14.7, 10.5, 7.3, 5.3]  # [ms]
T2_MNCL2_SOURCE = ["Jiang 2016"] + ["manual"] * 3 + ["Jiang 2016"] + ["manual"] * 9

RELAXIVITY_3T = {"NiCl2": (0.70, 1.00), "MnCl2": (7.38, 121.5)}  # (r1, r2) [1/(mM s)]
WATER_T1, WATER_T2 = 3.0, 2.0  # [s] deionised water, 3 T, ~20 °C (approximate; matters only for the longest T's)


def t2_of_nicl2(t1_ms) -> list[float]:
    """T2 [ms] of NiCl2 spheres from their T1, via the relaxivity ratio."""
    r1, r2 = RELAXIVITY_3T["NiCl2"]
    return [1e3 / (1 / WATER_T2 + r2 / r1 * (1e3 / t1 - 1 / WATER_T1)) for t1 in t1_ms]


def t1_of_mncl2(t2_ms) -> list[float]:
    """T1 [ms] of MnCl2 spheres from their T2, via the relaxivity ratio."""
    r1, r2 = RELAXIVITY_3T["MnCl2"]
    return [1e3 / (1 / WATER_T1 + r1 / r2 * (1e3 / t2 - 1 / WATER_T2)) for t2 in t2_ms]


def recon(data: np.ndarray) -> np.ndarray:
    """The authors' reconstruction: per-coil centred 2D iFFT of (samples, coils, lines), RSS."""
    im = np.fft.fftshift(np.fft.ifft2(np.transpose(data, (1, 0, 2))), axes=(-2, -1))
    return np.sqrt((np.abs(im) ** 2).sum(0))


def sphere_radius(images: np.ndarray, roi: np.ndarray) -> float:
    """Physical sphere radius [m] from the real images (geometry only): the half-maximum of the
    sphere-vs-surroundings contrast in the radial profile, at the image where it is largest."""
    r0, c0 = ndimage.center_of_mass(roi)
    rr, cc = np.indices(roi.shape)
    dist = np.hypot(rr - r0, cc - c0)
    best = None
    for im in images:
        prof = np.array([im[(dist >= d - 0.5) & (dist < d + 0.5)].mean() for d in np.arange(0, 12)])
        inside, outside = prof[:2].mean(), prof[9:].mean()
        if best is None or abs(inside - outside) > best[0]:
            best = (abs(inside - outside), prof, inside, outside)
    _, prof, inside, outside = best
    half = (inside + outside) / 2
    sign = 1 if inside > outside else -1
    k = int(np.argmax(sign * (prof - half) < 0))  # first radius past half-max
    d = k - 1 + (sign * (prof[k - 1] - half)) / (sign * (prof[k - 1] - prof[k]))
    return float(d * PIXEL)


def slice_position(seq_file) -> float:
    """Slice centre along x [m]: the excitation's frequency offset over its slice-select gradient
    (Pulseq positions slices by frequency; with blochparty's default "physical" reading a
    frequency offset f excites at +f/G). Works for trapezoids and extended trapezoids."""
    import pypulseq as pp

    s = pp.Sequence()
    s.read(str(seq_file))
    for i in range(1, len(s.block_events) + 1):
        b = s.get_block(i)
        if b.rf is None or b.gx is None:
            continue
        amp = getattr(b.gx, "amplitude", None)
        if not amp and getattr(b.gx, "waveform", None) is not None:
            amp = float(np.max(np.abs(b.gx.waveform)))
        if amp:
            return float(b.rf.freq_offset / amp)
    raise ValueError("no slice-selective pulse with a gradient on x")


def phantom(rois, radii, t1_ms, t2_ms, x0, *, n_slice=48, sub=2, sub_read=None, seed=0,
            dtype=torch.float32) -> tuple[bp.Phantom, torch.Tensor]:
    """Spheres as 3D balls centred on the slice at x0. In-plane spins at ``sub_read`` (readout,
    y; default ``sub``) x ``sub`` (phase, z) per pixel with stratified jitter; slice positions
    stratified over 1.5 slice thicknesses either side of x0 (the slice axis is x for both .seq
    files). Returns the phantom and each spin's sphere index."""
    sub_read = sub if sub_read is None else sub_read
    g = torch.Generator().manual_seed(seed)
    half = 1.5 * SLICE
    xs = (torch.arange(n_slice, dtype=torch.float64) + torch.rand(n_slice, generator=g, dtype=torch.float64)) \
        / n_slice * 2 * half - half
    pos, sph, t1, t2 = [], [], [], []
    for k, (roi, rad) in enumerate(zip(rois, radii)):
        r0, c0 = ndimage.center_of_mass(roi)
        y0, z0 = (r0 - CENTRE) * PIXEL, (c0 - CENTRE) * PIXEL
        m = math.ceil(rad / PIXEL) + 1
        grid_y = (torch.arange(-m * sub_read, m * sub_read, dtype=torch.float64) + 0.5) * (PIXEL / sub_read)
        grid_z = (torch.arange(-m * sub, m * sub, dtype=torch.float64) + 0.5) * (PIXEL / sub)
        gy, gz = torch.meshgrid(grid_y, grid_z, indexing="ij")
        gy = gy + (torch.rand(gy.shape, generator=g, dtype=torch.float64) - 0.5) * (PIXEL / sub_read)
        gz = gz + (torch.rand(gz.shape, generator=g, dtype=torch.float64) - 0.5) * (PIXEL / sub)
        for x in xs:
            inside = gy**2 + gz**2 <= rad**2 - x**2
            n = int(inside.sum())
            if n == 0:
                continue
            pos.append(torch.stack([torch.full((n,), x0 + float(x), dtype=torch.float64), gy[inside] + y0,
                                    gz[inside] + z0], -1))
            sph.append(torch.full((n,), k))
            t1.append(torch.full((n,), t1_ms[k] * 1e-3, dtype=torch.float64))
            t2.append(torch.full((n,), t2_ms[k] * 1e-3, dtype=torch.float64))
    pos, sph, t1, t2 = (torch.cat(v) for v in (pos, sph, t1, t2))
    pd = torch.full_like(t1, PIXEL**2 / (sub * sub_read) * (2 * half / n_slice) / (PIXEL**2 * SLICE))
    ph = bp.Phantom(pos=pos, pd=pd, t1=t1, t2=t2, db0=torch.zeros_like(t1)).to(dtype=dtype)
    return ph, sph


def coarsen_rf(seq: bp.Sequence, k: int) -> bp.Sequence:
    """Merge every k consecutive RF segments of a pulse into one with the same duration and
    the same RF and gradient integrals (averages weighted by dt). The real files sample RF on a
    1 us raster, which makes RF ~80 % of all segments; the error this introduces is measured in
    the convergence studies."""
    if k == 1:
        return seq
    groups, s, n = [], 0, len(seq)
    while s < n:
        e = s + 1
        if seq.is_rf[s]:
            while e < n and e - s < k and seq.is_rf[e] and not seq.adc[e - 1] and not seq.spoil[e - 1]:
                e += 1
        groups.append((s, e))
        s = e
    seg = torch.repeat_interleave(torch.arange(len(groups)), torch.tensor([b - a for a, b in groups]))
    dt = torch.zeros(len(groups), dtype=seq.dt.dtype).index_add_(0, seg, seq.dt)
    avg = lambda v: torch.zeros(len(groups), v.shape[1], dtype=v.dtype).index_add_(0, seg, v * seq.dt[:, None]) / dt[:, None]
    last = torch.tensor([b - 1 for _, b in groups])
    return replace(seq, dt=dt, rf=avg(seq.rf), grad=avg(seq.grad), adc_phase=seq.adc_phase[last],
                   is_rf=[seq.is_rf[a] for a, _ in groups], adc=[seq.adc[b - 1] for _, b in groups],
                   spoil=[seq.spoil[b - 1] for _, b in groups], meta=dict(seq.meta))


def roi_means(images: np.ndarray, rois) -> np.ndarray:
    """(n_images, n_rois) ROI-mean magnitudes."""
    return np.array([[im[r].mean() for r in rois] for im in images])


def shape_error(sim: np.ndarray, real: np.ndarray, idx=None) -> np.ndarray:
    """Per-sphere relative RMS error after one least-squares scale per sphere, over images ``idx``."""
    idx = slice(None) if idx is None else idx
    s, r = sim[idx], real[idx]
    a = (s * r).sum(0) / (s * s).sum(0)
    return np.sqrt(((a * s - r) ** 2).sum(0) / (r * r).sum(0))
