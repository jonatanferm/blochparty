"""Scalar susceptibility -> Lorentz-corrected spin frequency, and low-order shimming.

The Fourier kernel is 1/3 - (k.b)^2/|k|^2 (Salomir et al., Concepts Magn Reson B 2003,
doi:10.1002/cmr.b.10083; Marques & Bowtell, Concepts Magn Reson B 2005,
doi:10.1002/cmr.b.20034). Arrays are (z, y, x), vectors are (x, y, z).
Susceptibility is dimensionless SI, *not ppm*: multiply ppm maps by 1e-6.

For scale: water at 37 C is -9.05e-6 and air +0.36e-6, a 9.41 ppm
contrast (Jordan et al., JMRI 2013, doi:10.1002/jmri.23762, Theory).
Bone is of order -2 ppm relative to water (bovine cortical bone controls:
-2.47 +/- 0.23 ppm, "Evaluation of Gadolinium Deposition in Cortical Bone
Using ... 3D UTE-QSM", Table 1, https://pmc.ncbi.nlm.nih.gov/articles/10726698/).
Deep-grey-matter contrast +0.05 to +0.2 ppm is an illustrative modelling range
("Quantitative Susceptibility Mapping Using the Multiple Dipole-inversion
Combination with k-space Segmentation Method", Numerical simulation,
https://pmc.ncbi.nlm.nih.gov/articles/PMC5743526/), not a universal in-vivo range.
These are illustrative values, not a tissue-property table.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .constants import GAMMA_BAR


def _volume(value: Tensor, name: str) -> None:
    if value.ndim != 3 or min(value.shape) == 0:
        raise ValueError(f"{name} must be a nonempty (z, y, x) volume")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must have dtype float32 or float64")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def _voxel(voxel) -> tuple[float, float, float]:
    voxel = tuple(float(v) for v in voxel)
    if len(voxel) != 3 or any(not math.isfinite(v) or v <= 0 for v in voxel):
        raise ValueError("voxel must contain three positive finite spacings (dz, dy, dx) in m")
    return voxel


def susceptibility_to_b0(
    chi: Tensor,
    voxel: tuple[float, float, float],
    field_T: float = 3.0,
    *,
    b0_dir=(0.0, 0.0, 1.0),
    padding: float | tuple[float, float, float] = 2.0,
    background: float = 0.0,
) -> Tensor:
    """Return the susceptibility-induced frequency [Hz], on the input grid.

    ``chi`` is a float32/float64 (nz, ny, nx) tensor of scalar SI susceptibility;
    ``voxel`` is (dz, dy, dx) in m. ``b0_dir`` is the field direction (x, y, z)
    in the object's frame and is normalized here. Dtype, device and autograd
    through chi are preserved (the device must support PyTorch FFTs).

    ``padding`` is the total size factor, >= 1, per axis (scalar or z/y/x
    tuple). Subtract ``background`` before zero-padding and centre-crop after
    the FFT. Thus material outside the supplied volume has susceptibility
    ``background``: use 0 for contrast maps, +0.36e-6 for an absolute head map
    surrounded by air. Cropping anatomy before convolution loses sources.
    Factor 1 is periodic; larger factors reduce interaction with periodic
    copies, but cannot fix poorly resolved tissue boundaries. Partial-volume
    input and resolution convergence are preferable to smoothing the kernel.
    The k=0 kernel is zero: only relative fields are defined. This sets the
    mean on the *padded* domain, not necessarily the returned crop, to zero.

    Sample this map into ``Phantom.db0`` for a tissue-attached susceptibility
    field, or use ``FieldMaps.b0`` for a stationary field in scanner space.
    With motion, keep the fitted shim in scanner space and susceptibility in
    tissue space; do not attach their combined, shimmed map to moving tissue.
    Rigid translation preserves the susceptibility field; rotation relative
    to B0 requires recomputing it with the new ``b0_dir``. This is a static,
    scalar-susceptibility model, not a dynamic field solver.
    """
    _volume(chi, "chi")
    voxel = _voxel(voxel)
    if not math.isfinite(field_T) or field_T < 0:
        raise ValueError("field_T must be finite and nonnegative")
    if not math.isfinite(background):
        raise ValueError("background must be finite")
    factors = (padding,) * 3 if isinstance(padding, (int, float)) else tuple(padding)
    if len(factors) != 3 or any(not math.isfinite(p) or p < 1 for p in factors):
        raise ValueError("padding must be >= 1, either a scalar or a (z, y, x) triple")
    direction = torch.as_tensor(b0_dir, dtype=chi.dtype, device=chi.device)
    if direction.shape != (3,) or not torch.isfinite(direction).all() or direction.norm() == 0:
        raise ValueError("b0_dir must be a finite nonzero (x, y, z) vector")
    direction = direction / direction.norm()
    shape = tuple(math.ceil(n * p) for n, p in zip(chi.shape, factors))
    before = tuple((m - n) // 2 for m, n in zip(shape, chi.shape))
    pads = tuple(v for n, m, start in reversed(list(zip(chi.shape, shape, before)))
                 for v in (start, m - n - start))
    source = F.pad(chi - background, pads)
    freq = [torch.fft.fftfreq(n, d=h, dtype=chi.dtype, device=chi.device)
            for n, h in zip(shape, voxel)]
    kz, ky, kx = freq[0][:, None, None], freq[1][None, :, None], freq[2][None, None, :]
    k2 = kx.square() + ky.square() + kz.square()
    # At an even grid's Nyquist bin, +k and -k denote the same samples.
    # Average over BOTH signs independently for each such axis: diagonal
    # terms survive, cross terms vanish. Merely taking ifft(...).real does
    # not fix pairs of Nyquist axes and breaks joint object/field rotations.
    cross_freq = [k.clone() for k in freq]
    for n, k in zip(shape, cross_freq):
        if n % 2 == 0:
            k[n // 2] = 0
    cz, cy, cx = (cross_freq[0][:, None, None], cross_freq[1][None, :, None],
                  cross_freq[2][None, None, :])
    bx, by, bz = direction.unbind()
    kb2 = ((bx*kx).square() + (by*ky).square() + (bz*kz).square()
           + 2 * (bx*by*cx*cy + bx*bz*cx*cz + by*bz*cy*cz))
    kernel = 1 / 3 - kb2 / torch.where(k2 > 0, k2, 1.0)
    kernel[0, 0, 0] = 0
    field = torch.fft.ifftn(torch.fft.fftn(source) * kernel).real
    crop = tuple(slice(start, start + n) for start, n in zip(before, chi.shape))
    return field[crop] * (GAMMA_BAR * field_T)


_TERMS = ("offset", "x", "y", "z", "z2", "zx", "zy", "x2y2", "xy")


def _basis(x: Tensor, y: Tensor, z: Tensor, order: int) -> list[Tensor]:
    terms = [torch.ones_like(x)]
    if order >= 1:
        terms += [x, y, z]
    if order == 2:
        terms += [z.square() - (x.square() + y.square()) / 2,
                  z * x, z * y, x.square() - y.square(), x * y]
    return terms


def shim(
    df: Tensor,
    mask: Tensor,
    order: int = 2,
    *,
    voxel: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Subtract a least-squares solid-harmonic fit; return (residual, coefficients).

    ``df`` [Hz] and boolean ``mask`` are matching (z, y, x) volumes. Fit over
    the mask, subtract over the whole volume. ``order`` is 0, 1 or 2, using
    precisely ``phantom.b0_shim``'s terms and coefficient names. Coefficients
    describe the fitted field; the scanner correction has the opposite sign.
    Units are Hz, Hz/m and Hz/m^2 for orders 0, 1 and 2, respectively.

    ``voxel`` is (dz, dy, dx) in m; ``origin`` is the (x, y, z) position of
    voxel (0, 0, 0). Default origin centres the grid on zero (between the middle
    voxels for even sizes). Default spacing is one metre: supply real spacing
    and origin to reuse the returned coefficients in ``b0_shim``. Empty or
    rank-deficient masks raise ValueError (e.g. use order 0 for a single voxel).
    Columns are scaled before solving to condition physical metre coordinates.
    The result remains differentiable with respect to ``df``.
    """
    _volume(df, "df")
    voxel = _voxel(voxel)
    if order not in (0, 1, 2):
        raise ValueError("shim order must be 0, 1 or 2")
    if mask.shape != df.shape or mask.dtype != torch.bool or mask.device != df.device:
        raise ValueError("mask must be boolean with the same shape and device as df")
    if origin is None:
        origin = tuple(-(n - 1) * h / 2 for n, h in reversed(list(zip(df.shape, voxel))))
    if len(origin) != 3 or any(not math.isfinite(o) for o in origin):
        raise ValueError("origin must be a finite (x, y, z) triple in m")
    if order == 0:
        if not mask.any():
            raise ValueError("shim mask has too few voxels for the requested order")
        # This least-squares solution is the mean; avoid a huge one-column QR
        # whose accumulation loses precision for million-voxel float32 maps.
        offset = df[mask].mean()
        return df - offset, {"offset": offset}
    axes = [torch.arange(n, dtype=df.dtype, device=df.device) * h + o
            for n, h, o in zip(df.shape, voxel, reversed(origin))]
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    fit = torch.stack(_basis(x[mask], y[mask], z[mask], order), dim=-1)
    if fit.shape[0] < fit.shape[1]:
        raise ValueError("shim mask has too few voxels for the requested order")
    scales = fit.norm(dim=0)
    if (scales == 0).any():
        raise ValueError("shim mask is rank deficient for the requested order")
    design = fit / scales
    # The default N*eps cutoff becomes order one for million-voxel float32
    # masks. Only nine scaled columns are fitted, irrespective of mask size.
    rcond = design.shape[1] * torch.finfo(df.dtype).eps
    if torch.linalg.matrix_rank(design, rtol=rcond) < design.shape[1]:
        raise ValueError("shim mask is rank deficient for the requested order")
    solution = torch.linalg.lstsq(design, df[mask], rcond=rcond).solution / scales
    residual = df.clone()
    for c, term in zip(solution, _basis(x, y, z, order)):
        residual = residual - c * term
    return residual, dict(zip(_TERMS, solution.unbind()))
