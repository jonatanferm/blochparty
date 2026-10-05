"""Fused C++ kernel vs. the pure-PyTorch simulator it must reproduce.

float64: the same math step by step, so agreement is at rounding level (<= 1e-12).
float32: compared with the PyTorch simulator in float64 on the same (float32-rounded)
inputs. Both float32 implementations carry ~1e-5 of their own rounding error in
steady state (it accumulates over ~T1/TR repetitions), so float32 against float32
mostly measures PyTorch's error; that comparison gets a looser bound.
"""

from dataclasses import replace

import pytest
import torch

import blochparty as bp
from blochparty import Phantom, Sequence, SequenceBuilder, simulate
from test_bloch_vs_ode import random_case

fused = pytest.importorskip("blochparty.fused")


@pytest.fixture(scope="module", autouse=True)
def extension():
    try:
        fused._ext()
    except Exception as e:  # no compiler / ninja: nothing to test
        pytest.skip(f"fused extension unavailable: {e}")


def rel(a, b) -> float:
    if b.numel() == 0:
        return 0.0
    scale = b.abs().max()
    return float((a - b).abs().max() / scale) if scale > 0 else float((a - b).abs().max())


def cast(phantom, seq, coils, dtype):
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    return phantom.to(dtype=dtype), seq.to(dtype=dtype), None if coils is None else coils.to(cdtype)


def check(phantom, seq, coils=None, dtype=torch.float64, tol=None, **kw):
    """Fused vs. simulate; returns (signal, m) relative errors against float64."""
    ph, sq, cl = cast(phantom, seq, coils, dtype)
    out = fused.simulate_fused(ph, sq, cl, **kw)
    ref = simulate(*cast(ph, sq, cl, torch.float64))
    assert out.signal.shape == ref.signal.shape == (1 if coils is None else coils.shape[1], seq.n_adc)
    assert out.m.shape == (len(phantom), 3) and out.m.dtype == dtype
    assert out.signal.dtype == (torch.complex64 if dtype == torch.float32 else torch.complex128)
    errs = rel(out.signal.to(torch.complex128), ref.signal), rel(out.m.double(), ref.m)
    tol = tol or (1e-12 if dtype == torch.float64 else 1e-5)
    assert max(errs) <= tol, errs
    if dtype == torch.float32:  # against the PyTorch simulator in float32 itself
        same = simulate(ph, sq, cl)
        assert rel(out.signal, same.signal) <= 1e-4 and rel(out.m, same.m) <= 1e-4
    return errs


def with_readout(seq: Sequence, n=16) -> Sequence:
    """Append a readout (gradient, receiver phase) so a sequence also produces signal."""
    ro = SequenceBuilder().readout(n, 10e-6, grad=(800.0, -300.0, 150.0), phase=0.7).build()
    cat = lambda f: torch.cat([getattr(seq, f), getattr(ro, f)])
    return replace(seq, dt=cat("dt"), rf=cat("rf"), grad=cat("grad"), adc_phase=cat("adc_phase"),
                   is_rf=seq.is_rf + ro.is_rf, adc=seq.adc + ro.adc, spoil=seq.spoil + ro.spoil)


def head_3d(n=12, n_coils=4):
    phantom = bp.shepp_logan(n, ndim=3)
    g = torch.Generator().manual_seed(1)
    phantom = replace(phantom, db0=(torch.rand(len(phantom), generator=g, dtype=torch.float64) - 0.5) * 100)
    return phantom, bp.birdcage_sensitivities(phantom.pos, n_coils=n_coils)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_rf_gradients_off_resonance(seed, dtype):
    phantom, seq = random_case(seed)
    assert seq.n_adc == 0  # empty-ADC path: (1, 0) signal
    check(phantom, seq, dtype=dtype)
    check(phantom, seq.subdivide_rf(4), dtype=dtype)
    coils = bp.birdcage_sensitivities(phantom.pos, n_coils=3)
    check(phantom, with_readout(seq), coils, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_gre_2d_birdcage(dtype):
    phantom = bp.shepp_logan(32)
    coils = bp.birdcage_sensitivities(phantom.pos, n_coils=8)
    seq = bp.gre_2d(32, flip_deg=20.0, rf_phase=0.4, n_dummy=100)
    check(phantom, seq, coils, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_gre_3d_coils_off_resonance(dtype):
    phantom, coils = head_3d(n_coils=8)
    check(phantom, bp.gre_3d(12, n_dummy=50), coils, dtype=dtype)


@pytest.mark.parametrize("block", fused.BLOCK_SIZES)
@pytest.mark.parametrize("n_threads", [1, 3])
def test_blocks_and_threads(block, n_threads):
    # 6 coils: one 4-coil pass plus the single-coil remainder; spin count not a multiple of any block
    phantom, coils = head_3d(n=8, n_coils=6)
    assert all(len(phantom) % b for b in fused.BLOCK_SIZES)
    check(phantom, bp.gre_3d(8, n_dummy=5), coils, block=block, n_threads=n_threads)


def test_single_spin_and_initial_magnetization():
    one = Phantom(pos=torch.tensor([[0.01, -0.02, 0.005]], dtype=torch.float64),
                  pd=torch.tensor([0.8], dtype=torch.float64), t1=torch.tensor([0.9], dtype=torch.float64),
                  t2=torch.tensor([0.08], dtype=torch.float64), db0=torch.tensor([12.0], dtype=torch.float64))
    seq = bp.gre_2d(8, n_dummy=3)
    check(one, seq)
    check(one, seq, bp.birdcage_sensitivities(one.pos, n_coils=2))
    m0 = torch.tensor([[0.3, -0.2, 0.5]], dtype=torch.float64)
    out = fused.simulate_fused(one, seq, m_init=m0)
    ref = simulate(one, seq, m_init=m0)
    assert rel(out.signal, ref.signal) < 1e-12 and rel(out.m, ref.m) < 1e-12


def test_empty_sequence_and_no_grad():
    phantom, seq = random_case(0)
    empty = replace(seq, dt=seq.dt[:0], rf=seq.rf[:0], grad=seq.grad[:0], adc_phase=seq.adc_phase[:0],
                    is_rf=[], adc=[], spoil=[])
    out = fused.simulate_fused(phantom, empty)
    assert out.signal.shape == (1, 0)
    assert torch.equal(out.m, torch.stack([torch.zeros_like(phantom.pd)] * 2 + [phantom.pd], dim=-1))
    # differentiable (tests/test_fused_grad.py), except under no_grad
    flip = torch.tensor(15.0, dtype=torch.float64, requires_grad=True)
    out = fused.simulate_fused(phantom, bp.gre_2d(4, flip_deg=flip))
    assert out.signal.requires_grad and out.m.requires_grad
    with torch.no_grad():
        out = fused.simulate_fused(phantom, bp.gre_2d(4, flip_deg=flip))
    assert not out.signal.requires_grad and not out.m.requires_grad
