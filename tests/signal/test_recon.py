"""Undersampled reconstruction (recon.py): the orthonormal FFT pair, the line masks, CG-SENSE; and the AUC and its
bootstrap (design/metrics.py). Each is checked against a property it must have, not against its own formula."""

import numpy as np
import torch

import blochparty as bp
from blochparty.design import auc, bootstrap
from blochparty.recon import (cg_sense, fft2c, ifft2c_ortho, random_mask, uniform_mask, variable_density_mask)

c128 = torch.complex128


def _rand(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(shape, generator=g, dtype=torch.float64), torch.randn(shape, generator=g, dtype=torch.float64))


def test_orthonormal_pair_is_unitary_and_matches_the_centred_fft():
    x = _rand((3, 16, 12), 0)
    k = fft2c(x)
    assert torch.allclose(ifft2c_ortho(k), x, atol=1e-12)
    assert abs(float(k.norm() / x.norm()) - 1) < 1e-12  # Parseval: orthonormal
    # the same transform as the package's centred inverse, up to its 1/N normalisation
    assert torch.allclose(bp.ifft2c(k) * np.sqrt(16 * 12), ifft2c_ortho(k), atol=1e-12)


def test_sense_encoding_and_its_adjoint():
    """<A x, y> = <x, A^H y> for the masked multi-coil encoding cg_sense inverts."""
    n, c = 16, 4
    maps, x, y = _rand((c, n, n), 1), _rand((n, n), 2), _rand((c, n, n), 3)
    mask = uniform_mask(n, 8)
    m = mask[None, :, None]
    ax = fft2c(maps * x[None]) * m
    ahy = (maps.conj() * ifft2c_ortho(y * m)).sum(0)
    lhs, rhs = (ax.conj() * y).sum(), (x.conj() * ahy).sum()
    assert abs(complex(lhs - rhs)) < 1e-10 * abs(complex(lhs))


def test_cg_sense_unfolds_twofold_undersampling():
    """A disc seen by eight smooth coils at R = 2 (every other line plus the centre): CG-SENSE with the exact maps
    recovers it to under 1 %, ten times closer than the zero-filled coil-combined image (19 % off)."""
    n = 32
    u = (torch.arange(n, dtype=torch.float64) - n / 2) / n
    yy, xx = torch.meshgrid(u, u, indexing="ij")
    img = ((xx ** 2 + yy ** 2) < 0.3 ** 2).to(c128) * (1 + 0.5 * xx)
    ang = torch.arange(8, dtype=torch.float64) * 2 * np.pi / 8
    maps = torch.stack([torch.exp(-((xx - 0.6 * torch.cos(a)) ** 2 + (yy - 0.6 * torch.sin(a)) ** 2) / 0.2)
                        * torch.exp(1j * a) for a in ang]).to(c128)
    mask = uniform_mask(n, n // 2 + 2, centre=4)
    k = fft2c(maps * img[None]) * mask[None, :, None]
    rec = cg_sense(k, maps, mask, iters=100, lam=1e-6)
    err = float((rec - img).norm() / img.norm())
    zero_filled = (maps.conj() * ifft2c_ortho(k)).sum(0) / (maps.abs() ** 2).sum(0)
    assert err < 1e-2, err
    assert float((zero_filled - img).norm() / img.norm()) > 10 * err


def test_masks_take_the_lines_they_promise():
    for n, lines, centre in ((64, 20, 4), (128, 64, 8), (32, 4, 4)):
        for m in (uniform_mask(n, lines, centre), variable_density_mask(n, lines, centre, seed=3)):
            assert int(m.sum()) == lines
            lo = n // 2 - centre // 2
            assert bool(m[lo:lo + centre].all())
    a, b = variable_density_mask(64, 20, seed=1), variable_density_mask(64, 20, seed=1)
    assert torch.equal(a, b) and not torch.equal(a, variable_density_mask(64, 20, seed=2))
    # variable density: the central quarter is sampled more densely than the outer half
    hits = torch.stack([variable_density_mask(128, 40, seed=s) for s in range(50)]).double().mean(0)
    assert float(hits[48:80].mean()) > 2 * float(torch.cat([hits[:32], hits[96:]]).mean())
    r = random_mask(128, 4, 0.08, np.random.default_rng(0))
    assert r.sum() == 128 // 4 and r[64 - 5:64 - 5 + 10].all()


def test_auc_and_bootstrap():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    rng = np.random.default_rng(0)  # against the rank (Mann-Whitney U) formula on random scores without ties
    yy = rng.integers(0, 2, 200)
    s = rng.standard_normal(200) + yy
    ranks = np.argsort(np.argsort(s)) + 1
    n1, n0 = int(yy.sum()), int((1 - yy).sum())
    assert abs(auc(yy, s) - (ranks[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) < 1e-12
    groups = np.arange(200) // 2  # pairs of rows per "person"; each pair shares a label
    yy = np.repeat(rng.integers(0, 2, 100), 2)
    s = rng.standard_normal(200) + yy
    lo, hi = bootstrap(yy, s, groups, b=500)
    assert lo < auc(yy, s) < hi and hi - lo < 0.3
