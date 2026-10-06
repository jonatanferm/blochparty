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


def test_gre_2d_coils_b1_b0_offset_rf_spoiling():
    ph = bp.shepp_logan(24)
    n = len(ph)
    ph = bp.Phantom(ph.pos, ph.pd, ph.t1, ph.t2, torch.linspace(-50, 50, n, dtype=torch.float64),
                    b1=torch.linspace(0.8, 1.2, n, dtype=torch.float64))
    seq = bp.gre_2d(24, n_dummy=10, spoiling="rf", ramps=True)
    seq.b0_offset = 3.0 * torch.sin(torch.arange(len(seq), dtype=torch.float64) * 1e-3)  # a slowly varying field
    coils = bp.birdcage_sensitivities(ph.pos, 8)
    ref = bp.simulate(ph, seq, coils)
    out = metal.simulate_metal(ph, seq, coils)
    assert rel(out.signal, ref.signal) < 1e-4  # long RF-spoiled train: float32, as the CPU kernel in float32
    assert rel(out.m, ref.m) < 1e-3


def test_many_coils_and_buffer_flushes():
    # 32 coils shrink the sample buffer to 15 samples per flush; 3D GRE readouts cross many flushes
    ph = bp.shepp_logan(8, ndim=3)
    seq = bp.gre_3d(8, n_dummy=3)
    coils = torch.randn(len(ph), 32, dtype=torch.complex128, generator=torch.Generator().manual_seed(0))
    assert rel(metal.simulate_metal(ph, seq, coils).signal, bp.simulate(ph, seq, coils).signal) < 1e-4


def test_spins_not_multiple_of_group_and_m_init():
    g = torch.Generator().manual_seed(3)
    n = 1000  # not a multiple of 256
    ph = bp.Phantom(pos=(torch.rand(n, 3, generator=g, dtype=torch.float64) - 0.5) * 0.1,
                    pd=torch.rand(n, generator=g, dtype=torch.float64), t1=torch.full((n,), 0.8, dtype=torch.float64),
                    t2=torch.full((n,), 0.07, dtype=torch.float64), db0=torch.zeros(n, dtype=torch.float64))
    m0 = torch.randn(n, 3, generator=g, dtype=torch.float64)
    seq = bp.gre_2d(16, n_dummy=2)
    ref = bp.simulate(ph, seq, m_init=m0)
    out = metal.simulate_metal(ph, seq, m_init=m0)
    assert rel(out.signal, ref.signal) < 1e-5 and rel(out.m, ref.m) < 1e-5


def test_hard_pulse_flip_angle_and_fid():
    ph = bp.Phantom(pos=torch.zeros(1, 3, dtype=torch.float64), pd=torch.ones(1, dtype=torch.float64),
                    t1=torch.full((1,), 1e9, dtype=torch.float64), t2=torch.full((1,), 0.05, dtype=torch.float64),
                    db0=torch.full((1,), 37.0, dtype=torch.float64))
    seq = bp.SequenceBuilder().rf(10e-9, flip_deg=90.0).readout(40, 1e-3).build()
    s = metal.simulate_metal(ph, seq).signal[0]
    t = (torch.arange(40, dtype=torch.float64) + 0.5) * 1e-3 + 5e-9
    expected = 1j * torch.exp(-t / 0.05) * torch.exp(-2j * math.pi * 37.0 * t)
    assert (s.to(torch.complex128) - expected).abs().max() < 1e-5


def test_scanner_model_not_supported():
    ph, seq = random_case(0)
    with pytest.raises(NotImplementedError):
        metal.simulate_metal(ph, seq, scanner=object())
