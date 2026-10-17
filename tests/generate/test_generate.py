"""blochparty.generate: the spec and its checks, the preset's orientation, receive placement and protocol, and the
noise unit the samples are scaled by. A full sample (an IXI head on the GPU) is checked by the sample-identity run in
the journal (2026-10-17), not here."""

import numpy as np
import pytest
import torch

import blochparty as bp
from blochparty import generate
from blochparty.generate.protocols import REAL_ETL, interleaved, sequence
from blochparty.generate.heads import turn
from blochparty.generate.sample import kspace_on_fov, rss_image, to_real_orientation, unit_floor


def test_spec_merges_over_the_defaults_and_refuses_unknown_keys():
    spec = generate.merged({"protocol": {"n": 64}, "anatomy": {"subjects": [12, 13]}})
    assert spec["protocol"]["n"] == 64 and spec["protocol"]["family"] == "fastmri_t2w"  # the rest from the defaults
    assert spec["anatomy"]["subjects"] == [12, 13]
    assert generate.merged(spec) == spec  # idempotent
    with pytest.raises(ValueError, match="unknown spec key"):
        generate.merged({"protocol": {"slice": 3}})
    with pytest.raises(ValueError, match="unknown spec section"):
        generate.merged({"scanners": {}})
    d = generate.default_spec()
    d["quality"]["spins"] = [1, 1, 1]
    assert generate.DEFAULT_SPEC["quality"]["spins"] == [2, 3, 3]  # a copy
    assert generate.span("2-6:2") == [2, 4, 6] and generate.span("5") == [5]
    assert interleaved(6) == [0, 2, 4, 1, 3, 5]


def test_unit_floor_is_the_median_of_a_chi_distribution():
    """The RSS of C coils of unit-variance complex noise is chi-distributed with 2C degrees of freedom over sqrt(2)
    per component: its median is sqrt(chi2_2C median) with each component's variance 1."""
    from scipy import stats

    for c in (1, 8):
        expected = float(np.sqrt(stats.chi2.median(2 * c)))  # orthonormal ifft keeps the per-pixel variance 2 (re + im)
        assert unit_floor(c, n=256) == pytest.approx(expected, rel=0.01)


def test_cropping_an_oversampled_readout_keeps_the_field_of_view():
    """kspace_on_fov on a 2x oversampled readout equals the k-space of the image cropped to the central n columns."""
    n = 16
    rng = np.random.default_rng(0)
    img2 = rng.standard_normal((2, n, 2 * n)) + 1j * rng.standard_normal((2, n, 2 * n))
    k2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img2, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))

    class Seq:  # a stand-in whose cartesian_kspace is k2: kspace_on_fov only reshapes the signal by the sequence
        meta = {"kspace_shape": (n, 2 * n)}
    signal = torch.from_numpy(k2.reshape(2, -1))
    out = kspace_on_fov(signal, Seq())
    crop = img2[..., n // 2: n // 2 + n]
    ref = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(crop, axes=-1), axis=-1, norm="ortho"), axes=-1)
    ref = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(ref, axes=-2), axis=-2, norm="ortho"), axes=-2)
    assert np.allclose(out, ref, atol=1e-10)
    assert np.allclose(rss_image(out), np.sqrt((np.abs(crop) ** 2).sum(0)), atol=1e-10)


def test_display_is_posterior_up():
    """turn() makes the phantom's x the anterior direction (x' = y); a reconstructed (y', x') image with a mark at
    the most anterior x' shows it on the bottom row, as fastMRI shows the head (validation/realism/orientation.py)."""
    pos = torch.tensor([[0.0, 0.05, 0.0]])  # 5 cm anterior
    ph = turn(bp.Phantom(pos=pos, pd=torch.ones(1), t1=torch.ones(1), t2=torch.ones(1), db0=torch.zeros(1)))
    assert ph.pos[0, 0].item() == pytest.approx(0.05)
    img = np.zeros((8, 8))
    img[3, 7] = 1  # row y', column x' at the anterior end
    shown = to_real_orientation(img)
    assert np.unravel_index(shown.argmax(), shown.shape) == (7, 3)


def test_header_train_matches_the_real_echo_blocks():
    """The real AXT2 k-space energy steps between the echoes' blocks of 19 lines at ky -161, -142, ..., -47, -28, -9 and
    +10 (validation/realism/echo_train.py over 21 scans): 396 lines over 21 echoes, ky = 0 in echo 11. The header train
    (REAL_ETL at the header's 10.26 ms) puts the 128 central lines' block boundaries there and TE at 11 echo spacings."""
    seq = sequence(128, 6.19, 0.113, ordering="linear_full", etl=REAL_ETL, esp=0.01026)
    echo = np.asarray(seq.meta["line_echo"])
    ky = np.arange(128) - 64
    assert echo[ky == 0][0] == 10  # 0-based: echo 11
    assert set(ky[1:][np.diff(echo) != 0].tolist()) == {-47, -28, -9, 10, 29, 48}
    assert seq.meta["te"] == pytest.approx(11 * 0.01026)
