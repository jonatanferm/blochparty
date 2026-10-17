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
from typing import NamedTuple

import numpy as np
import torch
from scipy import ndimage

import blochparty as bp
from blochparty.montecarlo import jackknife, mc_magnitude, mc_power  # noqa: F401 (the scripts import them from here)
from blochparty.montecarlo import sobol_points as _sobol_points

N, FOV, SLICE = 128, 0.250, 6e-3  # both benchmark sequences
PIXEL = FOV / N
CENTRE = 63.5  # pixel index of position 0 in the authors' reconstruction (measured by simulating a point)

T1_NICL2 = {  # [ms]
    "v1": [1989, 1454, 984.1, 706, 496, 351.5, 247.13, 175.3, 125.9, 89.0, 62.7, 44.53, 30.84, 21.719],
    "v2": [1883.97, 1330.16, 987.27, 690.08, 484.97, 341.58, 240.86, 174.95, 121.08, 85.75, 60.21, 42.89, 30.40, 21.44],
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


def slice_positions(seq_file) -> list[float]:
    """Distinct slice centres along x [m], in order of first use: each slice-selective pulse's
    frequency offset over its slice-select gradient (Pulseq positions slices by frequency; with
    blochparty's default "physical" reading a frequency offset f excites at +f/G). Works for
    trapezoids and extended trapezoids. Multi-slice files return several positions."""
    import pypulseq as pp

    s = pp.Sequence()
    s.read(str(seq_file))
    out = []
    for i in range(1, len(s.block_events) + 1):
        b = s.get_block(i)
        if b.rf is None or b.gx is None:
            continue
        amp = getattr(b.gx, "amplitude", None)
        if not amp and getattr(b.gx, "waveform", None) is not None:
            amp = float(np.max(np.abs(b.gx.waveform)))
        if amp:
            x = float(b.rf.freq_offset / amp)
            if all(abs(x - y) > 1e-6 for y in out):
                out.append(x)
    if not out:
        raise ValueError("no slice-selective pulse with a gradient on x")
    return out


def slice_position(seq_file) -> float:
    """The slice centre of a single-slice file (raises for multi-slice files)."""
    xs = slice_positions(seq_file)
    if len(xs) != 1:
        raise ValueError(f"{len(xs)} slices at {[round(x * 1e3, 2) for x in xs]} mm; use slice_positions")
    return xs[0]


def phantom(
    rois, radii, t1_ms, t2_ms, x0, *, n_slice=48, sub=2, sub_read=None, seed=0, dtype=torch.float32
) -> tuple[bp.Phantom, torch.Tensor]:
    """Spheres as 3D balls centred on the slice at x0. In-plane spins at ``sub_read`` (readout,
    y; default ``sub``) x ``sub`` (phase, z) per pixel with stratified jitter; slice positions
    stratified over 1.5 slice thicknesses either side of x0 (the slice axis is x for both .seq
    files). Returns the phantom and each spin's sphere index."""
    sub_read = sub if sub_read is None else sub_read
    g = torch.Generator().manual_seed(seed)
    half = 1.5 * SLICE
    xs = (
        torch.arange(n_slice, dtype=torch.float64) + torch.rand(n_slice, generator=g, dtype=torch.float64)
    ) / n_slice * 2 * half - half
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
            pos.append(
                torch.stack(
                    [torch.full((n,), x0 + float(x), dtype=torch.float64), gy[inside] + y0, gz[inside] + z0], -1
                )
            )
            sph.append(torch.full((n,), k))
            t1.append(torch.full((n,), t1_ms[k] * 1e-3, dtype=torch.float64))
            t2.append(torch.full((n,), t2_ms[k] * 1e-3, dtype=torch.float64))
    pos, sph, t1, t2 = (torch.cat(v) for v in (pos, sph, t1, t2))
    pd = torch.full_like(t1, PIXEL**2 / (sub * sub_read) * (2 * half / n_slice) / (PIXEL**2 * SLICE))
    ph = bp.Phantom(pos=pos, pd=pd, t1=t1, t2=t2, db0=torch.zeros_like(t1)).to(dtype=dtype)
    return ph, sph


def fill_mask(images: np.ndarray, rois, r_out: float) -> np.ndarray:
    """(N, N) pixels of the water fill, from the images (geometry only): inside the phantom
    outline (holes filled), bright in at least one image like the water around the spheres
    (dark plastic -- insert, fiducial shells -- is excluded), and outside every sphere's outer
    wall radius ``r_out`` [m]."""
    peak = images.max(0)
    inside = ndimage.binary_fill_holes(peak > 0.2 * np.percentile(peak, 99))
    rr, cc = np.indices(peak.shape)
    near = np.zeros_like(inside)
    for roi in rois:
        r0, c0 = ndimage.center_of_mass(roi)
        d = np.hypot(rr - r0, cc - c0) * PIXEL
        near |= d < r_out + 2 * PIXEL
    ring = inside & ~near & ndimage.binary_dilation(near, iterations=4)  # water right around the spheres
    bright = peak > 0.5 * np.median(peak[ring])
    return inside & bright


def fill_phantom(
    rois, mask, r_in, wall, x0, *, margin, t1_ms, t2_ms, n_slice=48, sub=2, sub_read=None, seed=1, dtype=torch.float32
) -> bp.Phantom:
    """Water around the spheres: spins on the same slice sampling and sub-pixel grid as
    ``phantom``, in ``mask`` pixels, outside each sphere's ball of radius r_in + wall (r_in: one
    value or one per sphere; the wall gives no signal), and within ``margin`` [m] of a sphere's
    outer wall -- far water only reaches the sphere ROIs through weak ringing, so the margin is a
    convergence parameter."""
    sub_read = sub if sub_read is None else sub_read
    g = torch.Generator().manual_seed(seed)
    half = 1.5 * SLICE
    xs = (
        torch.arange(n_slice, dtype=torch.float64) + torch.rand(n_slice, generator=g, dtype=torch.float64)
    ) / n_slice * 2 * half - half
    r_out = torch.as_tensor(r_in, dtype=torch.float64).expand(len(rois)) + wall  # (14,)
    centres = torch.tensor(
        [[(r - CENTRE) * PIXEL, (c - CENTRE) * PIXEL] for r, c in (ndimage.center_of_mass(m) for m in rois)],
        dtype=torch.float64,
    )
    rr, cc = np.nonzero(mask)
    py = torch.as_tensor((rr - CENTRE) * PIXEL, dtype=torch.float64)
    pz = torch.as_tensor((cc - CENTRE) * PIXEL, dtype=torch.float64)
    oy = (torch.arange(sub_read, dtype=torch.float64) + 0.5) / sub_read - 0.5
    oz = (torch.arange(sub, dtype=torch.float64) + 0.5) / sub - 0.5
    oy, oz = torch.meshgrid(oy, oz, indexing="ij")
    y = (py[:, None] + PIXEL * oy.flatten()[None]).flatten()
    z = (pz[:, None] + PIXEL * oz.flatten()[None]).flatten()
    y = y + (torch.rand(y.shape, generator=g, dtype=torch.float64) - 0.5) * (PIXEL / sub_read)
    z = z + (torch.rand(z.shape, generator=g, dtype=torch.float64) - 0.5) * (PIXEL / sub)
    d = torch.cdist(torch.stack([y, z], -1), centres)  # (n, 14) in-plane distance to every sphere centre
    keep = (d - r_out).min(1).values <= margin
    y, z, d = y[keep], z[keep], d[keep]
    pos = []
    for x in xs:
        excl = (r_out**2 - float(x) ** 2).clamp_min(0).sqrt()  # each ball's cross-section at this slice offset
        ok = (d > excl).all(1)
        n = int(ok.sum())
        pos.append(torch.stack([torch.full((n,), x0 + float(x), dtype=torch.float64), y[ok], z[ok]], -1))
    pos = torch.cat(pos)
    n = len(pos)
    pd = torch.full((n,), PIXEL**2 / (sub * sub_read) * (2 * half / n_slice) / (PIXEL**2 * SLICE), dtype=torch.float64)
    return bp.Phantom(
        pos=pos,
        pd=pd,
        t1=torch.full((n,), t1_ms * 1e-3, dtype=torch.float64),
        t2=torch.full((n,), t2_ms * 1e-3, dtype=torch.float64),
        db0=torch.zeros(n, dtype=torch.float64),
    ).to(dtype=dtype)


# --- Monte Carlo sampling and estimation ---------------------------------------------------
#
# Spins are Monte Carlo samples of the continuous object. The samplers below place n spins per
# pixel (across the slab of +-1.5 slice thicknesses) with a scrambled Sobol set, randomised per
# pixel by a Cranley-Patterson shift: unbiased, and lower-discrepancy than random or stratified
# placement. Independent seeds give independent estimates of the complex image, whose spread
# gives error bars and the power debiasing of the magnitude (see ``mc_power``, ``mc_errors``).


def _sphere_centres(rois) -> np.ndarray:
    return np.array(
        [[(r - CENTRE) * PIXEL, (c - CENTRE) * PIXEL] for r, c in (ndimage.center_of_mass(m) for m in rois)]
    )


def phantom_qmc(rois, radii, t1_ms, t2_ms, x0, *, n=256, seed=0, dtype=torch.float32) -> bp.Phantom:
    """Spheres as 3D balls (centred on the slice at x0), n Sobol spins per pixel of each sphere's
    bounding box, kept inside the ball."""
    half = 1.5 * SLICE
    pos, t1, t2 = [], [], []
    for k, (c, rad) in enumerate(zip(_sphere_centres(rois), radii)):
        m = math.ceil(rad / PIXEL) + 1
        cy, cz = (np.round(c / PIXEL + CENTRE) - CENTRE) * PIXEL  # pixel centre nearest the sphere centre
        gy, gz = np.meshgrid(cy + PIXEL * np.arange(-m, m + 1), cz + PIXEL * np.arange(-m, m + 1), indexing="ij")
        pix = np.stack([gy.ravel(), gz.ravel()], -1)
        u = _sobol_points(n, len(pix), seed * 1000 + k)
        p = np.stack(
            [
                x0 + (2 * u[..., 0] - 1) * half,
                pix[:, None, 0] + (u[..., 1] - 0.5) * PIXEL,
                pix[:, None, 1] + (u[..., 2] - 0.5) * PIXEL,
            ],
            -1,
        ).reshape(-1, 3)
        inside = (p[:, 0] - x0) ** 2 + (p[:, 1] - c[0]) ** 2 + (p[:, 2] - c[1]) ** 2 <= rad**2
        pos.append(p[inside])
        t1.append(np.full(inside.sum(), t1_ms[k] * 1e-3))
        t2.append(np.full(inside.sum(), t2_ms[k] * 1e-3))
    pos, t1, t2 = (torch.as_tensor(np.concatenate(v), dtype=torch.float64) for v in (pos, t1, t2))
    pd = torch.full_like(t1, (2 * half / SLICE) / n)
    return bp.Phantom(pos=pos, pd=pd, t1=t1, t2=t2, db0=torch.zeros_like(t1)).to(dtype=dtype)


def fill_phantom_qmc(
    rois, mask, r_in, wall, x0, *, margin, t1_ms, t2_ms, n=256, dense=None, sparse=0.25, seed=0, dtype=torch.float32
) -> bp.Phantom:
    """Water around the spheres (see ``fill_phantom``), n Sobol spins per pixel. Importance
    sampling: pixels more than ``dense`` [m] from every sphere wall get ``sparse`` * n spins (a
    power-of-two fraction keeps the Sobol prefix balanced) with weights scaled up to match, so the
    estimate stays unbiased while spins go where the ROIs are sensitive."""
    if not 0 < sparse <= 1:
        raise ValueError(f"sparse must be in (0, 1], got {sparse}")
    half = 1.5 * SLICE
    centres = _sphere_centres(rois)
    r_out = np.broadcast_to(np.asarray(r_in, dtype=float), (len(rois),)) + wall
    rr, cc = np.nonzero(mask)
    pix = np.stack([(rr - CENTRE) * PIXEL, (cc - CENTRE) * PIXEL], -1)
    gap = (np.linalg.norm(pix[:, None] - centres[None], axis=-1) - r_out[None]).min(1)  # pixel centre to nearest wall
    pix, gap = pix[gap <= margin + PIXEL], gap[gap <= margin + PIXEL]
    dense = margin if dense is None else dense
    n_sparse = max(1, int(round(n * sparse)))
    pos, w = [], []
    for sel, n_pix in ((gap <= dense, n), (gap > dense, n_sparse)):
        if not sel.any():
            continue
        u = _sobol_points(n_pix, int(sel.sum()), seed * 1000 + 999)  # = the n_pix prefix of the n-point set
        p = np.stack(
            [
                x0 + (2 * u[..., 0] - 1) * half,
                pix[sel][:, None, 0] + (u[..., 1] - 0.5) * PIXEL,
                pix[sel][:, None, 1] + (u[..., 2] - 0.5) * PIXEL,
            ],
            -1,
        ).reshape(-1, 3)
        d = np.linalg.norm(p[:, None, 1:] - centres[None], axis=-1)  # (spins, 14)
        excl = np.sqrt(np.clip(r_out[None] ** 2 - (p[:, :1] - x0) ** 2, 0, None))
        ok = (d > excl).all(1) & ((d - r_out[None]).min(1) <= margin)
        pos.append(p[ok])
        w.append(np.full(ok.sum(), (2 * half / SLICE) / n_pix))
    pos, pd = (torch.as_tensor(np.concatenate(v), dtype=torch.float64) for v in (pos, w))
    k = len(pos)
    return bp.Phantom(
        pos=pos,
        pd=pd,
        t1=torch.full((k,), t1_ms * 1e-3, dtype=torch.float64),
        t2=torch.full((k,), t2_ms * 1e-3, dtype=torch.float64),
        db0=torch.zeros(k, dtype=torch.float64),
    ).to(dtype=dtype)


def recon_complex(data: np.ndarray) -> np.ndarray:
    """Single-coil complex image of (samples, 1, lines) data, same transform as ``recon``."""
    return np.fft.fftshift(np.fft.ifft2(data[:, 0, :]), axes=(-2, -1))


class MCErrors(NamedTuple):
    err: np.ndarray  # (14,) per-sphere shape errors of the power-debiased magnitude
    se: np.ndarray  # (14,) their jackknife SEs
    mean10: float  # mean error over spheres 1-10 ...
    mean10_se: float  # ... and its jackknife SE (covariance between spheres included)
    mean10_bc: float  # ... jackknife bias-corrected: MC noise in the curves inflates an RMS error
    curves: np.ndarray  # (n_images, 14) ROI-mean curves of all seeds
    clipped: np.ndarray  # (14,) fraction of each ROI's pixels (over all images) whose power was < 0


def _mean10(imgs, rois, real, idx=None) -> float:
    return float(shape_error(roi_means(mc_magnitude(imgs)[0], rois), real, idx)[:10].mean())


def mc_errors(imgs: np.ndarray, rois, real: np.ndarray, idx=None) -> MCErrors:
    """Shape errors of K >= 3 seeds' complex images (K, n_images, N, N) against the scanner's
    ROI means, with jackknife SEs over seeds, per sphere and for the spheres-1-10 mean."""

    def curves(x):
        return roi_means(mc_magnitude(x)[0], rois)

    err, se = jackknife(lambda x: shape_error(curves(x), real, idx), imgs)
    m, m_se = jackknife(lambda x: _mean10(x, rois, real, idx), imgs)
    m_bc, _ = jackknife(lambda x: _mean10(x, rois, real, idx), imgs, bias_corrected=True)
    clipped = roi_means((mc_power(imgs)[0] < 0).astype(float), rois).mean(0)
    return MCErrors(err, se, float(m), float(m_se), float(m_bc), curves(imgs), clipped)


def mc_paired_difference(imgs_a: np.ndarray, imgs_b: np.ndarray, rois, real: np.ndarray, idx=None):
    """Spheres-1-10 mean error of b minus that of a, with the paired jackknife SE (the same seed
    left out of both). Meaningful when a and b share their seeds (common random numbers): the
    covariance then cancels most of the MC noise of the difference. Returns (difference, se)."""
    if imgs_a.shape != imgs_b.shape:
        raise ValueError(f"paired runs need the same seeds and images, got {imgs_a.shape} and {imgs_b.shape}")
    d, se = jackknife(
        lambda x: _mean10(x[:, 1], rois, real, idx) - _mean10(x[:, 0], rois, real, idx), np.stack([imgs_a, imgs_b], 1)
    )
    return float(d), float(se)


def print_mc_table(rows, rois, real: np.ndarray, plain_rows=()):
    """Markdown table of (name, err) rows without MC error and (name, MCErrors, seed images) rows,
    then the clipped-pixel fractions and the paired differences between all MC rows."""
    print("\n| model | spheres 1–10 | bias-corrected | per sphere 1..14 [%] (± jackknife SE) |\n|---|---|---|---|")
    for name, err in plain_rows:
        print(f"| {name} | {100 * err[:10].mean():.2f} % | | {' '.join(f'{100 * e:.1f}' for e in err)} |")
    for name, r, _ in rows:
        print(
            f"| {name} | {100 * r.mean10:.2f} ± {100 * r.mean10_se:.2f} % | {100 * r.mean10_bc:.2f} % | "
            f"{' '.join(f'{100 * e:.1f}±{100 * d:.1f}' for e, d in zip(r.err, r.se))} |"
        )
    print("\nPixels with negative debiased power (clipped to 0), % of each ROI over all images:")
    for name, r, _ in rows:
        print(f"  {name}: {' '.join(f'{100 * c:.1f}' for c in r.clipped)}")
    if len(rows) > 1:
        print("\nPaired differences of the spheres-1–10 mean (same seeds; ± paired jackknife SE):")
        for i, (name_a, _, imgs_a) in enumerate(rows):
            for name_b, _, imgs_b in rows[i + 1 :]:
                if imgs_a.shape == imgs_b.shape:
                    d, se = mc_paired_difference(imgs_a, imgs_b, rois, real)
                    print(f"  {name_b} − {name_a}: {100 * d:+.2f} ± {100 * se:.2f} %")


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
    avg = lambda v: (
        torch.zeros(len(groups), v.shape[1], dtype=v.dtype).index_add_(0, seg, v * seq.dt[:, None]) / dt[:, None]
    )
    last = torch.tensor([b - 1 for _, b in groups])
    return replace(
        seq,
        dt=dt,
        rf=avg(seq.rf),
        grad=avg(seq.grad),
        adc_phase=seq.adc_phase[last],
        is_rf=[seq.is_rf[a] for a, _ in groups],
        adc=[seq.adc[b - 1] for _, b in groups],
        spoil=[seq.spoil[b - 1] for _, b in groups],
        meta=dict(seq.meta),
    )


def roi_means(images: np.ndarray, rois) -> np.ndarray:
    """(n_images, n_rois) ROI-mean magnitudes."""
    return np.array([[im[r].mean() for r in rois] for im in images])


def absolute_error(sim: np.ndarray, real: np.ndarray, spheres=slice(0, 10)) -> np.ndarray:
    """Per-sphere relative RMS error after ONE least-squares scale for all images and the spheres
    ``spheres`` (receiver gain and PD units are global; what remains is relaxation weighting, B1+,
    receive shading and PD differences between spheres). Returns the (n_rois,) errors, every sphere
    scored with that one scale."""
    s, r = sim[:, spheres], real[:, spheres]
    a = (s * r).sum() / (s * s).sum()
    return np.sqrt(((a * sim - real) ** 2).sum(0) / (real * real).sum(0))


def receive_profile(image: np.ndarray, water: np.ndarray, degree: int = 2) -> np.ndarray:
    """Smooth receive-magnitude shading (N, N), mean 1 over ``water``: a 2D polynomial of
    ``degree`` fitted to the magnitude ``image`` over the water-fill pixels (uniform water, one
    relaxation weighting). Estimated from the data like the authors' coil combination; it carries
    the water's transmit (B1+) shading too, which these data cannot separate from receive."""
    rr, cc = np.indices(image.shape)
    u, v = (rr - CENTRE) / N, (cc - CENTRE) / N
    terms = [u**i * v**j for i in range(degree + 1) for j in range(degree + 1 - i)]
    A = np.stack([t[water] for t in terms], 1)
    coef, *_ = np.linalg.lstsq(A, image[water], rcond=None)
    prof = sum(c * t for c, t in zip(coef, terms))
    return prof / prof[water].mean()


def shape_error(sim: np.ndarray, real: np.ndarray, idx=None) -> np.ndarray:
    """Per-sphere relative RMS error after one least-squares scale per sphere, over images ``idx``."""
    idx = slice(None) if idx is None else idx
    s, r = sim[idx], real[idx]
    a = (s * r).sum(0) / (s * s).sum(0)
    return np.sqrt(((a * s - r) ** 2).sum(0) / (r * r).sum(0))
