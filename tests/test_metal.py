"""Metal (Apple GPU) forward kernel vs. the float64 PyTorch core. Metal is float32-only, so
tolerances are those of the CPU fused kernel in float32 on the same cases."""

import math

import pytest
import torch

import blochparty as bp
from test_bloch_vs_ode import random_case

metal = pytest.importorskip("blochparty.metal")
pytestmark = pytest.mark.skipif(not metal.available(), reason="no Apple GPU (MPS)")


def rel(a, b):
    return float((a.to(b.dtype) - b).abs().max() / b.abs().max())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_rf_gradients_off_resonance(seed):
    ph, seq = random_case(seed)
    assert rel(metal.simulate_metal(ph, seq).m, bp.simulate(ph, seq).m) < 1e-5


@pytest.mark.parametrize("mma", [False, True])
def test_gre_2d_coils_b1_b0_offset_rf_spoiling(mma):
    ph = bp.shepp_logan(24)
    n = len(ph)
    ph = bp.Phantom(ph.pos, ph.pd, ph.t1, ph.t2, torch.linspace(-50, 50, n, dtype=torch.float64),
                    b1=torch.linspace(0.8, 1.2, n, dtype=torch.float64))
    seq = bp.gre_2d(24, n_dummy=10, spoiling="rf", ramps=True)
    seq.b0_offset = 3.0 * torch.sin(torch.arange(len(seq), dtype=torch.float64) * 1e-3)  # a slowly varying field
    coils = bp.birdcage_sensitivities(ph.pos, 8)
    ref = bp.simulate(ph, seq, coils)
    out = metal.simulate_metal(ph, seq, coils, mma=mma)
    assert rel(out.signal, ref.signal) < 1e-4  # long RF-spoiled train: float32, as the CPU kernel in float32
    assert rel(out.m, ref.m) < 1e-3


@pytest.mark.parametrize("n_coils, mma", [(32, False), (1, True), (13, True), (50, True)])
def test_many_coils_and_buffer_flushes(n_coils, mma):
    # direct: 32 coils shrink the sample buffer to 15 samples per flush. mma: blocks of 8 samples,
    # a partial last coil tile (13), and two passes through the buffer (50 coils > 48)
    ph = bp.shepp_logan(8, ndim=3)
    seq = bp.gre_3d(8, n_dummy=3)
    coils = torch.randn(len(ph), n_coils, dtype=torch.complex128, generator=torch.Generator().manual_seed(0))
    out = metal.simulate_metal(ph, seq, coils, mma=mma).signal
    assert rel(out, bp.simulate(ph, seq, coils).signal) < 1e-4


def test_spins_not_multiple_of_group_and_m_init():
    g = torch.Generator().manual_seed(3)
    n = 1000  # not a multiple of 256
    ph = bp.Phantom(pos=(torch.rand(n, 3, generator=g, dtype=torch.float64) - 0.5) * 0.1,
                    pd=torch.rand(n, generator=g, dtype=torch.float64), t1=torch.full((n,), 0.8, dtype=torch.float64),
                    t2=torch.full((n,), 0.07, dtype=torch.float64), db0=torch.zeros(n, dtype=torch.float64))
    m0 = torch.randn(n, 3, generator=g, dtype=torch.float64)
    seq = bp.gre_2d(16, n_dummy=2)
    ref = bp.simulate(ph, seq, m_init=m0)
    for mma in (False, True):
        out = metal.simulate_metal(ph, seq, m_init=m0, mma=mma)
        assert rel(out.signal, ref.signal) < 1e-5 and rel(out.m, ref.m) < 1e-5


def test_hard_pulse_flip_angle_and_fid():
    ph = bp.Phantom(pos=torch.zeros(1, 3, dtype=torch.float64), pd=torch.ones(1, dtype=torch.float64),
                    t1=torch.full((1,), 1e9, dtype=torch.float64), t2=torch.full((1,), 0.05, dtype=torch.float64),
                    db0=torch.full((1,), 37.0, dtype=torch.float64))
    seq = bp.SequenceBuilder().rf(10e-9, flip_deg=90.0).readout(43, 1e-3).build()
    s = metal.simulate_metal(ph, seq, mma=True).signal[0]  # 43 samples: 5 blocks of 8 and a partial one
    t = (torch.arange(43, dtype=torch.float64) + 0.5) * 1e-3 + 5e-9
    expected = 1j * torch.exp(-t / 0.05) * torch.exp(-2j * math.pi * 37.0 * t)
    assert (s.to(torch.complex128) - expected).abs().max() < 1e-5


@pytest.mark.parametrize("concomitant", [None, "leading", "exact"])
def test_scanner_field_model(concomitant):
    # gradient nonlinearity (encoding positions) and concomitant fields at 0.55 T, where they are large
    from test_fields_nonlinearity import distortion

    from blochparty.fields import Scanner

    ph = bp.shepp_logan(24)
    seq = bp.gre_2d(24, n_dummy=4)
    if concomitant is None:
        scanner = Scanner(nonlinearity=distortion())
    else:
        scanner = Scanner(field_strength=0.55, concomitant=concomitant, nonlinearity=distortion())
    ref = bp.simulate(ph, seq, scanner=scanner)
    plain = bp.simulate(ph, seq)
    out = metal.simulate_metal(ph, seq, scanner=scanner)
    assert rel(out.signal, ref.signal) < 1e-4 and rel(out.m, ref.m) < 1e-3
    assert rel(plain.signal, ref.signal) > 1e-2  # the field model matters in this case


@pytest.mark.parametrize("mma", [False, True])
@pytest.mark.parametrize("shape", ["lorentzian", "superlorentzian"])
def test_magnetization_transfer(shape, mma):
    from dataclasses import replace

    from test_exchange import bm_case
    from test_fused import with_readout

    ph, seq = bm_case(0, shape)
    ph = replace(ph, b1=torch.linspace(0.8, 1.2, len(ph), dtype=torch.float64))
    seq = with_readout(seq)
    coils = bp.birdcage_sensitivities(ph.pos, 4)
    m0 = torch.randn(len(ph), 4, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    for m_init in (None, m0):
        ref = bp.simulate(ph, seq, coils, m_init=m_init)
        out = metal.simulate_metal(ph, seq, coils, m_init=m_init, mma=mma)
        assert rel(out.signal, ref.signal) < 1e-4 and rel(out.m, ref.m) < 1e-4
        assert float((out.m_bound.double() - ref.m_bound).abs().max() / ref.m.abs().max()) < 1e-4


@pytest.mark.parametrize("mma", [False, True])
@pytest.mark.parametrize("shape", ["lorentzian", "superlorentzian"])
def test_generalized_bloch_bound_pool(shape, mma):
    from dataclasses import replace

    from test_exchange import bm_case
    from test_fused import with_readout

    ph, seq = bm_case(1, shape)
    ph = replace(ph, b1=torch.linspace(0.8, 1.2, len(ph), dtype=torch.float64),
                 bound=replace(ph.bound, model="generalized"))
    seq = with_readout(seq)
    coils = bp.birdcage_sensitivities(ph.pos, 4)
    ref = bp.simulate(ph, seq, coils)
    out = metal.simulate_metal(ph, seq, coils, mma=mma)
    assert rel(out.signal, ref.signal) < 1e-4 and rel(out.m, ref.m) < 1e-4
    assert float((out.m_bound.double() - ref.m_bound).abs().max() / ref.m.abs().max()) < 1e-4


@pytest.mark.parametrize("mma", [False, True])
def test_diffusion_same_walk_as_core(mma):
    # the same Philox walk per spin: the engines agree spin by spin, not only statistically
    import math

    from test_diffusion import pgse

    g = torch.Generator().manual_seed(2)
    n = 3000
    th = 0.4
    rot = torch.tensor([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]], dtype=torch.float64)
    dt = rot @ torch.diag(torch.tensor([0.5e-9, 1.5e-9, 2.5e-9], dtype=torch.float64)) @ rot.T
    ph = bp.Phantom(pos=(torch.rand(n, 3, generator=g, dtype=torch.float64) - 0.5) * 0.02,
                    pd=torch.ones(n, dtype=torch.float64), t1=torch.full((n,), 1.0, dtype=torch.float64),
                    t2=torch.full((n,), 0.08, dtype=torch.float64), db0=(torch.rand(n, generator=g, dtype=torch.float64) - 0.5) * 50,
                    diffusion=dt.expand(n, 3, 3).contiguous())
    seq = pgse((1.2e6, 3e5, -2e5), readout=32)
    coils = bp.birdcage_sensitivities(ph.pos, 4)
    ref = bp.simulate(ph, seq, coils, seed=9)
    out = metal.simulate_metal(ph, seq, coils, seed=9, mma=mma)
    assert rel(out.signal, ref.signal) < 1e-4 and rel(out.m, ref.m) < 1e-3
    other = metal.simulate_metal(ph, seq, coils, seed=10, mma=mma)
    assert rel(other.signal, ref.signal) > 1e-2  # a different walk


@pytest.mark.parametrize("memory", [100.0, 300.0])
def test_generalized_bloch_memory(memory):
    # the Green's-function memory reaches the kernel (the two memories differ by ~1e-3 in Mzb)
    one = lambda v: torch.full((4,), v, dtype=torch.float64)
    ph = bp.Phantom(pos=torch.zeros(4, 3, dtype=torch.float64), pd=one(0.9), t1=one(1.0), t2=one(0.05),
                    db0=torch.tensor([0.0, 50.0, -30.0, 10.0], dtype=torch.float64),
                    bound=bp.BoundPool(f=one(0.1), k=one(60.0), t1=one(1.0), t2=one(12e-6), model="generalized",
                                       memory=memory))
    seq = (bp.SequenceBuilder().rf(1e-3, flip_deg=180.0, n_steps=100).delay(5e-3)
           .rf(0.5e-3, flip_deg=90.0, n_steps=50).readout(32, 10e-6).build())
    a, b = bp.simulate(ph, seq), metal.simulate_metal(ph, seq)
    assert float((a.signal - b.signal).abs().max() / a.signal.abs().max()) < 5e-5
    assert float((a.m_bound - b.m_bound.to(a.m_bound.dtype)).abs().max()) < 5e-6


def _pools_case(n_pools, with_bound, seed=0):
    """pool_case's random RF / gradient / off-resonance sequence with 1-3 free pools, B1+ and a readout."""
    from dataclasses import replace

    from test_fused import with_readout
    from test_pools import pool_case

    ph, seq = pool_case(seed, with_bound)
    n = len(ph)
    g = torch.Generator().manual_seed(seed + 11)
    rnd = lambda: torch.rand(n, generator=g, dtype=torch.float64)
    mt = bp.ExchangePool(f=0.08 + 0.04 * rnd(), k=30 + 20 * rnd(), t1=0.8 + 0.2 * rnd(), t2=0.03 + 0.02 * rnd(),
                         dw=-200 + 50 * rnd())
    pools = (*ph.pools, mt)[:n_pools]
    return replace(ph, pools=pools, b1=torch.linspace(0.8, 1.2, n, dtype=torch.float64)), with_readout(seq)


@pytest.mark.parametrize("mma", [False, True])
@pytest.mark.parametrize("with_bound", [False, True])
@pytest.mark.parametrize("n_pools", [1, 2, 3])
def test_free_pools(n_pools, with_bound, mma):
    # the Bloch-McConnell kernel (_bm1.._bm3) against the core's matrix exponentials, from equilibrium
    # and from a random start (water, and the bound pool's Mz)
    ph, seq = _pools_case(n_pools, with_bound)
    coils = bp.birdcage_sensitivities(ph.pos, 4)
    m0 = torch.randn(len(ph), 4 if with_bound else 3, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    for m_init in (None, m0):
        ref = bp.simulate(ph, seq, coils, m_init=m_init)
        out = metal.simulate_metal(ph, seq, coils, m_init=m_init, mma=mma)
        scale = float(ref.m.abs().max())
        assert rel(out.signal, ref.signal) < 1e-5 and rel(out.m, ref.m) < 1e-5
        assert out.m_pools.shape == ref.m_pools.shape == (len(ph), n_pools, 3)
        assert float((out.m_pools.double() - ref.m_pools).abs().max()) < 1e-5 * scale
        if with_bound:
            assert float((out.m_bound.double() - ref.m_bound).abs().max()) < 1e-5 * scale
        else:
            assert out.m_bound is None


def test_free_pools_cest_spectrum():
    # a CEST Z-spectrum: offsets as spins in the RF frame (dB0 = -offset), a train of Gaussian
    # pulses with spoiled gaps, then a readout; the M0 offset at -300 ppm takes ~100 Taylor sub-steps
    # per RF sample (the others ~2), and the spins are sorted by that cost inside simulate_metal
    import math

    f0 = 127.7  # MHz
    offsets = torch.tensor([-300.0, *torch.linspace(-5, 5, 41).tolist()], dtype=torch.float64) * f0
    n = len(offsets)
    one = torch.ones(n, dtype=torch.float64)
    ph = bp.Phantom(pos=torch.zeros(n, 3, dtype=torch.float64), pd=one, t1=1.2 * one, t2=0.08 * one, db0=-offsets,
                    pools=(bp.ExchangePool(0.002 * one, 300 * one, one, 0.01 * one, 3.5 * f0 * one),
                           bp.ExchangePool(0.004 * one, 2000 * one, one, 0.01 * one, 1.0 * f0 * one)),
                    bound=bp.BoundPool(f=0.1 * one, k=30 * one, t1=one, t2=12e-6 * one, dw=-2.6 * f0 * one))
    t = (torch.arange(200, dtype=torch.float64) + 0.5) * 100e-6
    env = 120.0 * torch.exp(-0.5 * ((t - 0.01) / 0.004) ** 2)  # [Hz], ~2.8 uT peak
    b = bp.SequenceBuilder()
    for k in range(5):
        b.rf_waveform(torch.stack([env, 0 * env], -1), 100e-6).delay(5e-3, grad=(2000.0, 0.0, 0.0))
    b.spoil().rf(100e-6, flip_deg=90.0).readout(8, 10e-6)
    seq = b.build()
    ref = bp.simulate(ph, seq)
    out = metal.simulate_metal(ph, seq)
    assert float((out.m.double() - ref.m).abs().max()) < 1e-5  # Z of every offset
    assert rel(out.signal, ref.signal) < 1e-5
    assert float(ref.m[0, 2]) > 0.95 and float(ref.m[1:, 2].min()) < 0.5  # M0 barely saturated, the water dip deep
    assert math.isfinite(float(out.m_bound.abs().max()))


def test_free_pools_scanner_field_model():
    from dataclasses import replace

    from test_fields_nonlinearity import distortion

    from blochparty.fields import Scanner

    ph, seq = _pools_case(2, True)
    seq = replace(seq, grad=seq.grad * 30)  # ~3 mT/m: concomitant fields of a few Hz at 0.55 T
    scanner = Scanner(field_strength=0.55, nonlinearity=distortion())
    ref, plain = bp.simulate(ph, seq, scanner=scanner), bp.simulate(ph, seq)
    out = metal.simulate_metal(ph, seq, scanner=scanner)
    assert rel(out.signal, ref.signal) < 1e-5 and rel(out.m, ref.m) < 1e-5
    assert rel(plain.signal, ref.signal) > 1e-3  # the field model matters in this case


def test_five_pools_and_the_limit():
    from dataclasses import replace

    ph, seq = _pools_case(3, True)
    five = replace(ph, pools=ph.pools + tuple(replace(q, dw=q.dw + 300.0) for q in ph.pools[:2]))
    ref, out = bp.simulate(five, seq), metal.simulate_metal(five, seq)
    assert rel(out.signal, ref.signal) < 1e-5 and rel(out.m, ref.m) < 1e-5
    assert float((out.m_pools.double() - ref.m_pools).abs().max()) < 1e-5 * float(ref.m.abs().max())
    with pytest.raises(NotImplementedError, match="at most 5 free pools"):
        metal.simulate_metal(replace(five, pools=five.pools + ph.pools[:1]), seq)


@pytest.mark.parametrize("model", [None, "rate", "generalized"])
def test_long_rf_trains_do_not_drift(model):
    # CW saturation as 10 us RF segments: rotating all of m by a small angle every segment rounded it
    # every segment, and Mz drifted from the float64 core by ~1e-3 per 1e4 segments (1e-2 after 1e5)
    # until the increments were added by two_sum
    n = 32
    one = torch.ones(n, dtype=torch.float64)
    ph = bp.Phantom(torch.zeros(n, 3, dtype=torch.float64), one, one, 0.05 * one,
                    -torch.linspace(-2000, 2000, n, dtype=torch.float64))
    if model is not None:
        from dataclasses import replace

        ph = replace(ph, bound=bp.BoundPool(f=0.135 * one, k=30 * one, t1=one, t2=12e-6 * one, dw=-383.0 * one,
                                            model=model))
    n_seg = 3000 if model == "generalized" else 10_000
    seq = bp.SequenceBuilder().rf(n_seg * 10e-6, amplitude=42.577 * 1.5, n_steps=n_seg).readout(4, 10e-6).build()
    ref, out = bp.simulate(ph, seq), metal.simulate_metal(ph, seq)
    assert float((out.m.double() - ref.m).abs().max()) < 1e-5
    # the signal sums spins dephased by their offsets (|signal| ~ 0.2 of n): per spin, as m
    assert float((out.signal.to(ref.signal.dtype) - ref.signal).abs().max()) < 1e-5 * n
    if model is not None:
        assert float((out.m_bound.double() - ref.m_bound).abs().max()) < 1e-5
