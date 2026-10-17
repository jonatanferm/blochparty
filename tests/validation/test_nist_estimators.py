"""The Monte Carlo estimators of the NIST benchmarks (validation/real/nist.py): the debiased power
is exactly the unbiased pair statistic, the magnitude built on it is still biased at a null, the
jackknife of an aggregate keeps the covariance its parts' SEs drop, pairing cancels shared MC
noise, and invalid sample counts fail."""

import numpy as np
import pytest

from nist import _sobol_points, fill_phantom_qmc, jackknife, mc_errors, mc_magnitude, mc_paired_difference, mc_power


def noisy(amplitude, k, shape, rng):
    return amplitude + rng.normal(size=(k, *shape)) + 1j * rng.normal(size=(k, *shape))


def test_power_is_the_unbiased_pair_statistic():
    rng = np.random.default_rng(0)
    x = noisy(0.7 - 0.2j, 5, (3, 4), rng)
    pairs = (np.abs(x.sum(0)) ** 2 - (np.abs(x) ** 2).sum(0)) / (5 * 4)  # sum_{r != s} Re(X_r X_s*) / K(K-1)
    np.testing.assert_allclose(mc_power(x)[0], pairs, rtol=1e-12, atol=1e-12)
    many = noisy(0.0, 4, (200_000,), rng)  # at a null: E[power] = 0, but the magnitude keeps a floor
    power = mc_power(many)[0]
    assert abs(power.mean()) < 5 * power.std() / np.sqrt(power.size)
    assert (power < 0).mean() > 0.5
    assert mc_magnitude(many)[0].mean() > 0.2


def test_magnitude_unchanged():
    x = noisy(1.5, 4, (16,), np.random.default_rng(1))
    mean = x.mean(0)
    s2 = (np.abs(x - mean) ** 2).sum(0) / 3 / 4
    np.testing.assert_array_equal(mc_magnitude(x)[0], np.sqrt(np.clip(np.abs(mean) ** 2 - s2, 0, None)))


def test_aggregate_jackknife_keeps_covariance():
    samples = np.repeat(np.array([1.0, 2.0, 3.0, 4.0])[:, None], 10, axis=1)  # ten perfectly correlated parts
    _, part_se = jackknife(lambda x: x.mean(0), samples)
    _, agg_se = jackknife(lambda x: x.mean(0).mean(), samples)
    np.testing.assert_allclose(agg_se, part_se[0])
    np.testing.assert_allclose(agg_se / (np.linalg.norm(part_se) / 10), np.sqrt(10))


def test_jackknife_bias_correction_removes_1_over_k_bias():
    x = np.random.default_rng(3).normal(1.0, 2.0, size=(7, 5))  # mean^2 is biased by var / K ...
    bc, _ = jackknife(lambda v: v.mean(0) ** 2, x, bias_corrected=True)
    np.testing.assert_allclose(bc, x.mean(0) ** 2 - x.var(0, ddof=1) / 7)  # ... which the jackknife removes exactly


def test_paired_difference_cancels_shared_noise():
    rng = np.random.default_rng(2)
    n, k = 8, 6
    rois = [np.zeros((n, n), bool) for _ in range(3)]
    for i, r in enumerate(rois):
        r[2 * i:2 * i + 2, 2:6] = True
    decay = np.exp(-np.arange(5) / 3.0)[:, None, None]
    a = decay * (1 + 0.3 * rng.normal(size=(k, 5, n, n)))  # seed noise shared by both models
    b = a * 1.02 ** np.arange(5)[:, None, None]  # model b: a slightly slower decay, same seeds
    real = np.exp(-np.arange(5) / 3.3)[:, None] * np.ones((1, 3))
    ra, rb = mc_errors(a, rois, real), mc_errors(b, rois, real)
    d, se = mc_paired_difference(a, b, rois, real)
    np.testing.assert_allclose(d, rb.mean10 - ra.mean10)
    assert se < 0.2 * np.hypot(ra.mean10_se, rb.mean10_se)
    assert mc_paired_difference(a, a, rois, real) == (0.0, 0.0)
    assert ra.clipped.shape == (3,) and (ra.clipped == 0).all()


def test_sobol_prefix_and_guards():
    np.testing.assert_array_equal(_sobol_points(64, 5, 3), _sobol_points(256, 5, 3)[:, :64])
    for bad in (0, 100):
        with pytest.raises(ValueError, match="powers of two"):
            _sobol_points(bad, 2, 0)
    for k in (0, 1):
        with pytest.raises(ValueError):
            mc_power(np.ones((k, 3), complex))
    with pytest.raises(ValueError):
        jackknife(lambda x: x.mean(), np.ones((1, 3)))
    with pytest.raises(ValueError, match="sparse"):
        fill_phantom_qmc([], None, 0.01, 0.001, 0.0, margin=0.01, t1_ms=1, t2_ms=1, sparse=1.5)
