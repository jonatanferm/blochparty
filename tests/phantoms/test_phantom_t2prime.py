"""T2': a Lorentzian distribution of static frequency offsets inside each voxel
(``phantom_from_maps(..., t2prime=...)``). The voxel's FID decays with
1/T2* = 1/T2 + 1/T2'; a spin echo refocuses the offsets and decays with T2 alone.

The Fourier transform of a Lorentzian of half-width G is exp(-2 pi G |t|), so
G = 1 / (2 pi T2'). K equal-weight spins reproduce it only approximately: the tail
quantiles sit at frequencies so high that their single spins dephase within the first
samples, leaving ~1/sqrt(K) of uncancelled signal (deterministic quantiles are ~3x better
than random draws). Measured with K deterministic quantiles: max |FID / FID(0) - exp(-t/T2')|
is <= 0.6 / sqrt(K) for t <= 3 T2' and <= 2 / sqrt(K) out to t = K T2' / 4 (K = 256 to 4096).
"""

import math

import pytest
import torch
from scipy import stats

import blochparty as bp
from blochparty import SequenceBuilder, simulate

one = lambda v: torch.full((1, 1), v, dtype=torch.float64)
SIMS = {"core": simulate}


def voxel(k, t2prime, t1=1e9, t2=1e9, sampling="quantile", seed=0):
    return bp.phantom_from_maps(one(1.0), one(t1), one(t2), 1e-3, spins_per_voxel=(1, k), t2prime=one(t2prime),
                                t2prime_sampling=sampling, seed=seed)


def fid(phantom, n, dwell, sim=simulate):
    """Normalised FID after a 10 ns 90 deg pulse, at t = (i + 1/2) dwell + 5 ns from its centre."""
    seq = SequenceBuilder().rf(10e-9, flip_deg=90.0).readout(n, dwell).build()
    t = (torch.arange(n, dtype=torch.float64) + 0.5) * dwell + 5e-9
    return sim(phantom, seq).signal[0] / phantom.pd.sum(), t


def test_offsets_are_lorentzian():
    t2p, k = 0.02, 64
    g = 1 / (2 * math.pi * t2p)
    ph = voxel(k, t2p)
    u = (torch.arange(k, dtype=torch.float64).numpy() + 0.5) / k
    expected = torch.tensor(stats.cauchy.ppf(u, scale=g), dtype=torch.float64)
    assert torch.allclose(ph.db0.sort().values, expected, rtol=1e-12, atol=1e-9)
    # random draws: a seeded sample of the same distribution
    rnd = voxel(4000, t2p, sampling="random", seed=3).db0
    assert stats.kstest(rnd.numpy(), stats.cauchy(scale=g).cdf).pvalue > 0.01
    assert torch.equal(rnd, voxel(4000, t2p, sampling="random", seed=3).db0)
    # one spin per voxel: offset 0; T2' = inf: none
    assert voxel(1, t2p).db0.item() == 0.0
    assert torch.equal(voxel(16, math.inf).db0, torch.zeros(16, dtype=torch.float64))


def test_map_per_voxel():
    # a T2' map scales each voxel's own (shuffled) set of quantiles, on top of its dB0
    n, k = 4, 9
    m = bp.shepp_logan_maps(n)
    pd = torch.ones(n, n, dtype=torch.float64)
    t2p = torch.linspace(0.01, 0.1, n * n, dtype=torch.float64).reshape(n, n)
    db0 = torch.arange(n * n, dtype=torch.float64).reshape(n, n)
    ph = bp.phantom_from_maps(pd, m["t1"] + 1, m["t2"] + 1, 0.1, db0, spins_per_voxel=k, t2prime=t2p)
    q = torch.tensor(stats.cauchy.ppf((torch.arange(k, dtype=torch.float64).numpy() + 0.5) / k))
    off = (ph.db0.reshape(n * n, k) - db0.reshape(-1, 1)) * (2 * math.pi * t2p.reshape(-1, 1))
    assert torch.allclose(off.sort(dim=1).values, q.expand(n * n, k), atol=1e-12)
    assert len({tuple(o.argsort().tolist()) for o in off}) > 1  # not the same order in every voxel


@pytest.mark.parametrize("sim", SIMS)
def test_fid_decays_with_t2_star(sim):
    t2, t2p = 0.08, 0.02
    t2s = 1 / (1 / t2 + 1 / t2p)
    errs = {}
    for k in (256, 1024, 4096):
        s, t = fid(voxel(k, t2p, t1=1.0, t2=t2), 120, 3 * t2p / 120, SIMS[sim])
        errs[k] = float((s - 1j * torch.exp(-t / t2s)).abs().max())
        assert errs[k] <= 0.6 / math.sqrt(k), (k, errs[k])
    assert errs[256] > errs[1024] > errs[4096]
    # without the spread the FID would decay with T2 alone
    s, t = fid(voxel(1, t2p, t1=1.0, t2=t2), 120, 3 * t2p / 120)
    assert torch.allclose(s, 1j * torch.exp(-t / t2), atol=1e-6)


def test_fid_window_scales_with_k():
    # T2 off, so only T2' acts; out to K T2' / 4 the error stays <= 2 / sqrt(K)
    t2p = 0.01
    for k in (256, 1024):
        s, t = fid(voxel(k, t2p), 400, k * t2p / 4 / 400)
        err = float((s - 1j * torch.exp(-t / t2p)).abs().max())
        assert err <= 2 / math.sqrt(k), (k, err)
    # the deterministic quantiles beat random draws of the same size (3 T2' window)
    def err(phantom):
        s, t = fid(phantom, 100, 3 * t2p / 100)
        return float((s - 1j * torch.exp(-t / t2p)).abs().max())

    k = 1024
    assert err(voxel(k, t2p)) < min(err(voxel(k, t2p, sampling="random", seed=seed)) for seed in range(4))


@pytest.mark.parametrize("sim", SIMS)
def test_spin_echo_refocuses_t2prime(sim):
    t2, t2p, k = 0.06, 0.01, 1024
    t2s = 1 / (1 / t2 + 1 / t2p)
    ph = voxel(k, t2p, t1=1.0, t2=t2)
    for te in (10e-3, 20e-3, 40e-3):
        # 90x at 5 ns, 180y centred at te/2, sample at te (pulses short against the largest offsets, ~3 kHz)
        seq = (SequenceBuilder().rf(10e-9, flip_deg=90.0).delay(te / 2 - 20e-9)
               .rf(20e-9, flip_deg=180.0, phase=math.pi / 2).delay(te / 2 - 15e-9 - 0.5e-9).readout(1, 1e-9).build())
        echo = SIMS[sim](ph, seq).signal[0, 0] / ph.pd.sum()
        assert abs(abs(echo) - math.exp(-te / t2)) < 2e-5, (te, abs(echo))
        # the FID at the same time has decayed with T2* (to the 1 / sqrt(K) floor)
        s, t = fid(ph, 1, 2 * te)  # one sample at te
        assert abs(abs(s[0]) - math.exp(-t[0].item() / t2s)) < 0.6 / math.sqrt(k)
        assert abs(echo) > 2 * abs(s[0])
