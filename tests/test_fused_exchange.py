"""Fused kernel with a bound pool (magnetization transfer) vs. the PyTorch core: forward to
rounding, and the hand-written adjoint vs. autograd through ``simulate`` for every input,
the pool parameters and the RF (through its instantaneous frequency) included."""

from dataclasses import replace

import pytest
import torch

import blochparty as bp
from blochparty import Phantom, simulate
from test_exchange import bm_case
from test_fused import rel, with_readout

fused = pytest.importorskip("blochparty.fused")


@pytest.fixture(scope="module", autouse=True)
def extension():
    try:
        fused._ext()
    except Exception as e:  # no compiler / ninja: nothing to test
        pytest.skip(f"fused extension unavailable: {e}")


def case(seed, shape, b1=True):
    ph, seq = bm_case(seed, shape)
    if b1:
        g = torch.Generator().manual_seed(seed + 10)
        ph = replace(ph, b1=0.8 + 0.4 * torch.rand(len(ph), generator=g, dtype=torch.float64))
    return ph, with_readout(seq)


@pytest.mark.parametrize("dtype, tol", [(torch.float64, 1e-12), (torch.float32, 2e-5)])
@pytest.mark.parametrize("shape", ["lorentzian", "superlorentzian"])
@pytest.mark.parametrize("seed", [0, 1])
def test_forward_matches_core(seed, shape, dtype, tol):
    ph, seq = case(seed, shape)
    coils = bp.birdcage_sensitivities(ph.pos, n_coils=3)
    ref = simulate(ph, seq, coils)
    out = fused.simulate_fused(ph.to(dtype=dtype), seq.to(dtype=dtype), coils)
    assert rel(out.signal, ref.signal) < tol
    assert rel(out.m, ref.m) < tol
    # the bound pool can be saturated far below its f PD; judge its rounding on the PD scale
    assert float((out.m_bound.double() - ref.m_bound).abs().max() / ref.m.abs().max()) < tol


SPIN = ("pos", "pd", "t1", "t2", "db0", "b1")
POOL = ("f", "k", "t1", "t2", "dw")
SEQ = ("dt", "rf", "grad", "adc_phase")


def gradients(sim, ph, seq, coils, m0, **kw):
    req = lambda t: t.detach().clone().requires_grad_()
    spin = {f: req(getattr(ph, f)) for f in SPIN}
    pool = {f: req(getattr(ph.bound, f).expand(len(ph))) for f in POOL}
    sq = {f: req(getattr(seq, f)) for f in SEQ}
    leaves = {**spin, **{"bound." + k: v for k, v in pool.items()}, **sq, "coils": req(coils), "m_init": req(m0)}
    p = Phantom(**{k: spin[k] for k in SPIN}, bound=replace(ph.bound, **pool))
    r = sim(p, replace(seq, **sq), leaves["coils"], m_init=leaves["m_init"], **kw)
    g = torch.Generator().manual_seed(3)
    w_sig = torch.randn(r.signal.shape, dtype=torch.complex128, generator=g).to(r.signal.dtype)
    w_m = torch.randn(r.m.shape, dtype=torch.float64, generator=g).to(r.m.dtype)
    w_b = torch.randn(r.m_bound.shape, dtype=torch.float64, generator=g).to(r.m.dtype)
    loss = ((r.signal * w_sig).real.sum() + (r.signal.abs() ** 2).sum() + (r.m * w_m).sum() + (r.m_bound * w_b).sum()
            + (r.m_bound ** 2).sum())
    return dict(zip(leaves, torch.autograd.grad(loss, list(leaves.values()))))


@pytest.mark.parametrize("shape", ["lorentzian", "superlorentzian"])
@pytest.mark.parametrize("seed", [0, 1])
def test_adjoint_matches_autograd(seed, shape):
    ph, seq = case(seed, shape)
    coils = bp.birdcage_sensitivities(ph.pos, n_coils=3)
    g = torch.Generator().manual_seed(seed)
    m0 = torch.randn(len(ph), 4, dtype=torch.float64, generator=g)
    ref = gradients(simulate, ph, seq, coils, m0)
    for every in (None, 7):  # the checkpoint interval must not matter
        out = gradients(fused.simulate_fused, ph, seq, coils, m0, checkpoint_every=every)
        errs = {k: rel(out[k], ref[k]) for k in ref}
        assert max(errs.values()) < 1e-10, errs


def test_gradcheck_pool_parameters():
    # finite differences, independently of the PyTorch simulator
    ph, seq = case(0, "superlorentzian")
    n = 2
    ph = Phantom(pos=ph.pos[:n], pd=ph.pd[:n], t1=ph.t1[:n], t2=ph.t2[:n], db0=ph.db0[:n], b1=ph.b1[:n],
                 bound=ph.bound[:n])

    def fn(f, k, t1b, t2b_us, rf):
        p = replace(ph, bound=replace(ph.bound, f=f, k=k, t1=t1b, t2=t2b_us * 1e-6))
        r = fused.simulate_fused(p, replace(seq, rf=rf), n_threads=1)
        return r.m, r.m_bound, torch.view_as_real(r.signal)

    args = [ph.bound.f.clone().requires_grad_(), ph.bound.k.clone().requires_grad_(),
            ph.bound.t1.clone().requires_grad_(), (ph.bound.t2 * 1e6).clone().requires_grad_(),
            seq.rf.clone().requires_grad_()]
    assert torch.autograd.gradcheck(fn, args, eps=1e-7, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("dtype, tol", [(torch.float64, 1e-12), (torch.float32, 2e-5)])
@pytest.mark.parametrize("shape", ["lorentzian", "superlorentzian"])
def test_generalized_forward_matches_core(shape, dtype, tol):
    ph, seq = case(1, shape)
    ph = replace(ph, bound=replace(ph.bound, model="generalized"))
    coils = bp.birdcage_sensitivities(ph.pos, n_coils=3)
    ref = simulate(ph, seq, coils)
    out = fused.simulate_fused(ph.to(dtype=dtype), seq.to(dtype=dtype), coils)
    assert rel(out.signal, ref.signal) < tol and rel(out.m, ref.m) < tol
    assert float((out.m_bound.double() - ref.m_bound).abs().max() / ref.m.abs().max()) < tol


@pytest.mark.parametrize("shape", ["lorentzian", "superlorentzian"])
def test_generalized_adjoint_matches_autograd(shape):
    ph, seq = case(1, shape)
    ph = replace(ph, bound=replace(ph.bound, model="generalized"))
    coils = bp.birdcage_sensitivities(ph.pos, n_coils=3)
    m0 = torch.randn(len(ph), 4, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    ref = gradients(simulate, ph, seq, coils, m0)
    for every in (None, 7):
        out = gradients(fused.simulate_fused, ph, seq, coils, m0, checkpoint_every=every)
        errs = {k: rel(out[k], ref[k]) for k in ref}
        assert max(errs.values()) < 1e-10, errs
