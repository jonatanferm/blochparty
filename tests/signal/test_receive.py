"""The receive chain (``receive.py``): oversampled readouts, the decimation filter, noise,
interference, spikes, drift and quantisation, against what each must do by construction:
the filter's designed frequency response with no phase delay, the noise power through the
filter, the quantisation noise step^2 / 12 and clipping, an interference tone at the image
column its frequency predicts."""


import numpy as np
import pytest
import torch

import blochparty as bp
from blochparty.coils import coil_offset, normalised_sensitivities
from blochparty.receive import ReceiveChain, decimate, fir, oversample, quantise, response, spikes, thermal_noise

f64 = torch.float64


def tones(freqs):
    n = len(freqs)
    return bp.Phantom(pos=torch.zeros(n, 3, dtype=f64), pd=torch.ones(n, dtype=f64), t1=torch.full((n,), 1e6, dtype=f64),
                      t2=torch.full((n,), 1e6, dtype=f64), db0=torch.as_tensor(freqs, dtype=f64))


def test_decimation_has_the_designed_response_and_no_delay():
    dwell, n = 10e-6, 256
    freqs = np.array([0.0, 10e3, 30e3, 40e3, 45e3, 48e3])
    ph = tones(freqs)
    seq = bp.SequenceBuilder().rf(1e-6, flip_deg=90.0).readout(n, dwell).build()
    coils = torch.eye(len(freqs), dtype=torch.complex128)  # one channel per spin
    plain = bp.simulate(ph, seq, coils).signal
    for os in (2, 3, 4):
        so, info = oversample(seq, os)
        assert so.n_adc == n * os and info.readouts == [(0, n)]
        ratio = (decimate(bp.simulate(ph, so, coils).signal, info) / plain)[:, 64:192]
        assert np.allclose(ratio.abs().mean(1).numpy(), response(os, freqs, dwell), atol=1e-5)
        assert float(ratio.angle().abs().max()) < 1e-6  # centred: no group delay


def test_noise_power_through_the_filter():
    os, n = 2, 4096
    info = oversample(bp.SequenceBuilder().rf(1e-6, flip_deg=90.0).readout(n, 10e-6).build(), os)[1]
    g = torch.Generator().manual_seed(0)
    noisy = thermal_noise(torch.zeros(4, n * os, dtype=torch.complex128), 1.0, generator=g)
    out = decimate(noisy, info)[:, 64:-64]
    expected = float((fir(os) ** 2).sum())  # the variance of a filtered white sample
    assert abs(float(out.abs().pow(2).mean()) - expected) < 0.03 * expected


def test_quantisation_noise_and_clipping():
    g = torch.Generator().manual_seed(1)
    x = torch.complex(torch.rand(100000, generator=g, dtype=f64) - 0.5, torch.rand(100000, generator=g, dtype=f64) - 0.5)
    bits, fs = 10, 1.0
    q = quantise(x[None], bits, fs)[0]
    step = fs / 2 ** (bits - 1)
    err = q - x
    assert abs(float(err.real.pow(2).mean()) - step**2 / 12) < 0.05 * step**2 / 12
    big = quantise(torch.tensor([[3.0 + 0j, -3.0 + 0j]]), bits, fs)[0]
    assert float(big[0].real) == (2 ** (bits - 1) - 1) * step and float(big[1].real) == -fs


def test_interference_lands_in_the_column_of_its_frequency():
    # a tone at f in a readout of bandwidth 1 / dwell lands in the image column f / (BW / N) from
    # the centre (a coherent tone: one point along phase encoding; real zippers are incoherent lines)
    n = 64
    ph = bp.shepp_logan(n)
    seq = bp.gre_2d(n, n_dummy=0)
    t_adc = np.cumsum(seq.dt.numpy())[np.array(seq.adc)]
    dwell = float(t_adc[1] - t_adc[0])
    f = 7 / (n * dwell)  # 7 columns off centre
    clean = bp.reconstruct(ReceiveChain(os=2).acquire(lambda s: bp.simulate(ph, s), seq), seq)[0].abs()
    dirty = bp.reconstruct(ReceiveChain(os=2, tone=(f, 50.0)).acquire(lambda s: bp.simulate(ph, s), seq), seq)[0].abs()
    diff = (dirty - clean).abs().numpy()
    col = int(np.argmax(diff.max(0)))
    assert col in (n // 2 + 7, n // 2 - 7), col  # the sign follows the receiver convention
    assert diff[:, col].sum() > 0.9 * diff.sum()  # nearly all of it (the filter edges leak a little)


def test_spikes_hit_isolated_samples():
    g = torch.Generator().manual_seed(2)
    s = spikes(torch.zeros(2, 1000, dtype=torch.complex128), 5, 10.0, generator=g)
    hit = (s.abs() > 0).any(0)
    assert 1 <= int(hit.sum()) <= 5 and torch.allclose(s.abs()[:, hit], torch.full_like(s.abs()[:, hit], 10.0))

def test_coil_offset_puts_the_ring_behind_the_occiput():
    """A box head from y = -90 to +90 mm: the ring's back ``gap`` behind the head's back."""
    n, h = (40, 46, 40), 4e-3
    head = torch.zeros(n)
    y = -0.092 + np.arange(n[1]) * h  # voxel centres
    head[:, (y >= -0.09) & (y <= 0.09), 5:35] = 1
    an = bp.Anatomy({"skin": head, "background": 1 - head}, voxel=(h,) * 3, origin=(-0.08, -0.092, -0.08))
    back = y[(y >= -0.09)].min()
    off = coil_offset(an, np.array([0.0]), 0.0, radius=0.14, gap=0.025)
    assert off == pytest.approx(back - 0.025 + 0.14)
    assert coil_offset(an, np.array([0.0]), 0.01, radius=0.14, gap=0.025) == pytest.approx(off - 0.01)



def test_normalised_sensitivities_flatten_a_smooth_profile():
    """The scanner's prescan normalisation divides by a smooth estimate of the array's RSS sensitivity: a profile that
    varies slowly against the smoothing comes out flat inside the object (the edges keep some excess, by design)."""
    g = torch.Generator().manual_seed(0)
    pos = (torch.rand(40000, 3, generator=g, dtype=torch.float64) - 0.5) * torch.tensor([0.16, 0.16, 0.0], dtype=torch.float64)
    pos = pos[(pos[:, :2] ** 2).sum(1) < 0.08 ** 2]
    sens = torch.stack([torch.exp(pos[:, 0] / 0.1), 0.5 * torch.exp(-pos[:, 1] / 0.1)], 1).to(torch.complex128)
    out = normalised_sensitivities(sens, pos, smooth_mm=30.0)
    rss = out.abs().pow(2).sum(1).sqrt()
    inner = (pos[:, :2] ** 2).sum(1) < 0.04 ** 2
    raw = sens.abs().pow(2).sum(1).sqrt()[inner]
    assert float(raw.max() / raw.min()) > 1.8  # the profile spans almost 2x over the inner disc
    assert float(rss[inner].max() / rss[inner].min()) < 1.15
    assert torch.allclose(out / out[:, :1], sens / sens[:, :1])  # every coil divided by the same estimate
