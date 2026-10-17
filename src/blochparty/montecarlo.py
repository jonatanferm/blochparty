"""Monte Carlo estimators for simulations repeated over independent random seeds (spin positions, walks): the
debiased power and magnitude of K complex-image estimates, the leave-one-seed-out jackknife of any statistic, and
scrambled Sobol points for quasi-Monte-Carlo spin placement. From the NIST benchmarks (validation/real/nist.py),
which use them for their error bars; their review is docs/reviews/2026-10-08-monte-carlo.md."""

from __future__ import annotations

import numpy as np


def sobol_points(n: int, n_pixels: int, seed: int) -> np.ndarray:
    """(n_pixels, n, 3) points in [0, 1)^3: one scrambled Sobol set, shifted per pixel. n must be
    a power of two (Sobol's balance properties hold for those prefixes only)."""
    from scipy.stats import qmc

    if n < 1 or n & (n - 1):
        raise ValueError(f"Sobol point counts must be powers of two, got {n}")
    base = qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(n.bit_length() - 1)
    shift = np.random.default_rng(seed + 1).random((n_pixels, 1, 3))
    return (base[None] + shift) % 1.0


def mc_power(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unbiased estimate of |true|^2 from K >= 2 independent complex-image estimates (K, ...):
    |mean|^2 - s2_hat, with s2_hat the seed spread's variance of the mean (E|mean|^2 = |true|^2 +
    s2 for any i.i.d. finite-variance MC noise). Not clipped: it is negative where the noise
    exceeds the signal, e.g. at an inversion null. Returns (power, s2_hat)."""
    k = images.shape[0]
    if k < 2:
        raise ValueError(f"need at least 2 independent estimates, got {k}")
    mean = images.mean(0)
    s2 = (np.abs(images - mean) ** 2).sum(0) / (k - 1) / k
    return np.abs(mean) ** 2 - s2, s2


def mc_magnitude(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Power-debiased magnitude sqrt(max(mc_power, 0)) of K >= 2 complex-image estimates; returns
    (magnitude, s2_hat). Only the power is unbiased: the clip and square root bias the magnitude
    again, upward where the signal is below the MC noise (a floor at a null) and downward just
    above it. (The NIST benchmarks' ``mc_errors`` reports how many pixels are clipped.)"""
    power, s2 = mc_power(images)
    return np.sqrt(np.clip(power, 0, None)), s2


def jackknife(stat, samples: np.ndarray, bias_corrected: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """``stat`` of all K seeds (the plug-in estimate; with ``bias_corrected`` the jackknife's
    K full - (K - 1) mean(leave-one-out), which removes a bias ~ 1/K) and its leave-one-seed-out
    standard error, over the leading (seed) axis. K >= 2, and ``stat`` must accept K - 1 seeds
    (``mc_magnitude`` needs 2, so K >= 3). Jackknife the final scalar itself (e.g. a mean over
    spheres) so covariance between its parts is included."""
    k = samples.shape[0]
    if k < 2:
        raise ValueError(f"the jackknife needs at least 2 seeds, got {k}")
    full = np.asarray(stat(samples))
    loo = np.array([stat(np.delete(samples, i, axis=0)) for i in range(k)])
    se = np.sqrt((k - 1) / k * ((loo - loo.mean(0)) ** 2).sum(0))
    return (k * full - (k - 1) * loo.mean(0) if bias_corrected else full), se
