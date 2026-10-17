"""Image-quality metrics: tissue means, contrast, CNR, SNR (and per unit time), the image noise a k-space noise level
gives, magnitude images, the point-spread function and its width; a classifier's AUC and its bootstrap interval."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import Tensor

from ..coils import add_noise
from ..phantom import Phantom
from ..recon import reconstruct, rss
from ..sequence import Sequence
from .hardware import AXES

# --- image-quality metrics ------------------------------------------------------------------


def tissue_mean(image: Tensor, mask: Tensor) -> Tensor:
    """Mean magnitude of ``image`` over ``mask``."""
    return image.abs()[mask].mean()


def contrast(image: Tensor, mask_a: Tensor, mask_b: Tensor) -> Tensor:
    """Mean magnitude in ``mask_a`` minus that in ``mask_b``."""
    return tissue_mean(image, mask_a) - tissue_mean(image, mask_b)


def cnr(image: Tensor, mask_a: Tensor, mask_b: Tensor, sigma) -> Tensor:
    """Contrast-to-noise ratio, with ``sigma`` the image-domain noise standard deviation
    (given, or from ``image_noise_sigma``)."""
    return contrast(image, mask_a, mask_b) / sigma


def snr(image: Tensor, mask: Tensor, sigma) -> Tensor:
    return tissue_mean(image, mask) / sigma


def snr_efficiency(image: Tensor, mask: Tensor, sigma, duration) -> Tensor:
    """SNR per square root of scan time [1/sqrt(s)]."""
    return snr(image, mask, sigma) / torch.sqrt(torch.as_tensor(duration))


def image_noise_sigma(
    signal: Tensor,
    recon: Callable[[Tensor], Tensor],
    kspace_sigma: float,
    noise_cov: Tensor | None = None,
    mask: Tensor | None = None,
    n_replicas: int = 16,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Image-domain noise standard deviation by the pseudo-replica method: ``recon`` of
    (C, n_adc) ``signal`` plus k-space noise of per-channel ``kspace_sigma`` (and channel
    covariance ``noise_cov``, as ``coils.add_noise``), minus the noise-free image; the
    per-pixel standard deviation over replicas, RMS-averaged over ``mask``. Works for any
    reconstruction, including nonlinear ones such as RSS, at the actual signal level."""
    with torch.no_grad():
        clean = recon(signal)
        diffs = torch.stack(
            [recon(add_noise(signal, kspace_sigma, noise_cov, generator)) - clean for _ in range(n_replicas)]
        )
        if diffs.is_complex():
            diffs = diffs.real  # one component of complex noise, the same sigma as the magnitude at high SNR
        var = diffs.var(dim=0)
        return torch.sqrt(var[mask].mean() if mask is not None else var.mean())


def magnitude_image(signal: Tensor, seq: Sequence) -> Tensor:
    """Cartesian reconstruction: magnitude for one coil, root-sum-of-squares for several."""
    images = reconstruct(signal, seq)
    return images[0].abs() if images.shape[0] == 1 else rss(images)


def auc(y, s) -> float:
    """Mann-Whitney AUC of scores ``s`` for labels ``y`` (1 positive, 0 negative; ties count half)."""
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def bootstrap(y, s, groups, b=2000, seed=0) -> list[float]:
    """95 % interval of the AUC, resampling ``groups`` (people, not slices: every row of a group shares its label)."""
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    rng = np.random.default_rng(seed)
    people = np.array(sorted(set(groups)))
    rows = {p: np.nonzero(groups == p)[0] for p in people}
    pos = [p for p in people if y[rows[p][0]] == 1]
    neg = [p for p in people if y[rows[p][0]] == 0]
    out = []
    for _ in range(b):
        idx = np.concatenate(
            [rows[p] for p in rng.choice(pos, len(pos))] + [rows[p] for p in rng.choice(neg, len(neg))]
        )
        out.append(auc(y[idx], s[idx]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


# --- point-spread function ------------------------------------------------------------------


def _default_simulate():
    from ..engines.core import simulate

    return simulate


def point_spread_function(
    seq: Sequence, *, t1=1.0, t2=0.1, pd=1.0, db0=0.0, pos=(0.0, 0.0, 0.0), oversample: int = 4, simulate=None
) -> Tensor:
    """Magnitude PSF of a Cartesian ``seq``: simulate one spin at ``pos``, zero-pad k-space
    by ``oversample`` on every axis, reconstruct; normalised to its peak. Shape
    ``oversample * kspace_shape``; pixel ``oversample * n/2`` (per axis) is the grid centre."""
    simulate = simulate or _default_simulate()
    dtype = seq.dt.dtype
    f = lambda v: torch.as_tensor(v, dtype=dtype).reshape(1)
    spin = Phantom(pos=torch.as_tensor(pos, dtype=dtype).reshape(1, 3), pd=f(pd), t1=f(t1), t2=f(t2), db0=f(db0))
    k = simulate(spin, seq).signal[0].reshape(seq.meta["kspace_shape"])
    pads = []
    for n in reversed(k.shape):  # F.pad wants the last axis first
        before = (oversample * n) // 2 - n // 2
        pads += [before, oversample * n - n - before]
    big = torch.nn.functional.pad(torch.view_as_real(k), [0, 0] + pads)
    img = _ifft_centred(torch.view_as_complex(big.contiguous())).abs()
    return img / img.max()


def _ifft_centred(k: Tensor) -> Tensor:
    dims = tuple(range(k.ndim))
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(k, dim=dims), dim=dims), dim=dims)


def psf_metrics(psf: Tensor, oversample: int = 4) -> dict[str, Tensor]:
    """Per-axis FWHM [original pixels] and peak sidelobe level (largest value outside the
    main lobe, relative to the peak) of the profiles through the PSF's maximum. The FWHM
    interpolates linearly between samples, so it is differentiable in the PSF values.
    Axes are named as in the array order (z, y, x)."""
    peak = torch.unravel_index(torch.argmax(psf.detach()), psf.shape)
    out = {}
    names = AXES[: psf.ndim][::-1]
    for ax in range(psf.ndim):
        index = list(peak)
        index[ax] = slice(None)
        p = psf[tuple(index)]
        p = p / p[peak[ax]]
        c = int(peak[ax])
        half = []
        for step in (1, -1):  # walk out to the half-maximum crossing, interpolate linearly
            i = c
            while 0 <= i + step < len(p) and p[i + step] >= 0.5:
                i += step
            if not 0 <= i + step < len(p):
                half.append(p.new_tensor(float(abs(i - c))))
            else:
                half.append(abs(i - c) + (p[i] - 0.5) / (p[i] - p[i + step]))
        lobe = []  # main lobe: out to the first minimum on each side
        for step in (1, -1):
            i = c
            while 0 <= i + step < len(p) and p[i + step] <= p[i]:
                i += step
            lobe.append(i)
        lo, hi = min(lobe), max(lobe)
        outside = torch.cat([p[:lo], p[hi + 1 :]])
        out[f"fwhm_{names[ax]}"] = (half[0] + half[1]) / oversample
        out[f"sidelobe_{names[ax]}"] = outside.max() if outside.numel() else p.new_zeros(())
    return out
