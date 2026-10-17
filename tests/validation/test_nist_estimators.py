"""The NIST benchmarks' own Monte Carlo statistics (validation/real/nist.py): pairing two models on the same seeds
cancels their shared noise, and the QMC filling refuses a sparse fraction outside [0, 1]. The generic estimators
are blochparty.montecarlo (tests/signal/test_montecarlo.py)."""

import numpy as np
import pytest
from nist import fill_phantom_qmc, mc_errors, mc_paired_difference


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



def test_qmc_filling_refuses_a_bad_sparse_fraction():
    with pytest.raises(ValueError, match="sparse"):
        fill_phantom_qmc([], None, 0.01, 0.001, 0.0, margin=0.01, t1_ms=1, t2_ms=1, sparse=1.5)
