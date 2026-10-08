"""Gradients through the Metal kernel (``simulate_metal``'s adjoint, ROADMAP M2) against autograd through
the float64 core. The adjoint runs in float32: the tolerances below are those measured with ~10x margin
(per input: 1e-7 to 3e-6 on short random sequences, 2e-5 on a 1600-segment RF-spoiled GRE, up to 1e-4
with the bound pool, where the fused kernel's float32 adjoint is up to 1.5e-3 off)."""

from dataclasses import replace

import pytest
import torch

import blochparty as bp
from test_bloch_vs_ode import random_case
from test_fused import with_readout

metal = pytest.importorskip("blochparty.metal")
pytestmark = pytest.mark.skipif(not metal.available(), reason="no Apple GPU (MPS)")
f64 = torch.float64


def rel(a, b):
    return float((a.to(b.dtype) - b).abs().max() / b.abs().max())


def loss(r, which="both"):
    """Linear in the outputs with fixed random weights (float64, so float32 engines get the same)."""
    g = torch.Generator().manual_seed(1)
    w = lambda x: torch.randn(x.shape, generator=g, dtype=f64)
    total = 0
    if which in ("both", "signal"):
        total = total + (r.signal.real.double() * w(r.signal.real)).sum() + (r.signal.imag.double() * w(r.signal.imag)).sum()
    if which in ("both", "m"):
        total = total + (r.m.double() * w(r.m)).sum()
        if r.m_bound is not None:
            total = total + (r.m_bound.double() * w(r.m_bound)).sum()
    return total


def gradients(sim, phantom, seq, coils=None, m_init=None, scanner=None, which="both", **kw):
    """Gradients of ``loss`` w.r.t. every input tensor (phantom, bound pool, sequence, coils, m_init)."""
    leaves = {}

    def leaf(name, t):
        t = t.detach().clone().requires_grad_()
        leaves[name] = t
        return t

    ph = replace(phantom, **{k: leaf(k, getattr(phantom, k)) for k in ("pos", "pd", "t1", "t2", "db0")})
    if phantom.b1 is not None:
        ph = replace(ph, b1=leaf("b1", phantom.b1))
    if phantom.bound is not None:
        b = phantom.bound.expand(len(phantom), phantom.pd)
        ph = replace(ph, bound=replace(b, **{k: leaf("bound." + k, getattr(b, k)) for k in ("f", "k", "t1", "t2", "dw")}))
    sq = replace(seq, **{k: leaf(k, getattr(seq, k)) for k in ("dt", "rf", "grad", "adc_phase")})
    if seq.b0_offset is not None:
        sq = replace(sq, b0_offset=leaf("b0_offset", seq.b0_offset))
    c = None if coils is None else leaf("coils", coils)
    m0 = None if m_init is None else leaf("m_init", m_init)
    r = sim(ph, sq, c, m_init=m0, scanner=scanner, **kw)
    grads = torch.autograd.grad(loss(r, which), list(leaves.values()), allow_unused=True)
    return {k: g for k, g in zip(leaves, grads)}


def compare(phantom, seq, tol, **kw):
    ref, out = gradients(bp.simulate, phantom, seq, **kw), gradients(metal.simulate_metal, phantom, seq, **kw)
    errors = {k: rel(out[k], ref[k]) for k in ref if ref[k] is not None and ref[k].abs().max() > 0}
    assert max(errors.values()) < tol, errors
    return errors


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_rf_gradients_off_resonance(seed):
    ph, seq = random_case(seed)
    seq = with_readout(seq)
    seq = replace(seq, b0_offset=0.5 * torch.sin(torch.arange(len(seq), dtype=f64)))
    ph = replace(ph, b1=torch.linspace(0.8, 1.2, len(ph), dtype=f64))
    m0 = torch.randn(len(ph), 3, dtype=f64, generator=torch.Generator().manual_seed(5))
    compare(ph, seq, 3e-5, coils=bp.birdcage_sensitivities(ph.pos, 3), m_init=m0)


@pytest.mark.parametrize("which", ["signal", "m"])
def test_one_output_only(which):
    ph, seq = random_case(1)
    compare(ph, with_readout(seq), 3e-5, coils=bp.birdcage_sensitivities(ph.pos, 2), which=which)


@pytest.mark.parametrize("n_coils", [1, 3, 8, 20])
def test_gre_2d_coils(n_coils):
    # coil gradients in registers up to 16 coils (padded: 3 -> 4), from memory above (20)
    ph = bp.shepp_logan(16)
    seq = bp.gre_2d(16, n_dummy=5, spoiling="rf", ramps=True)
    coils = torch.randn(len(ph), n_coils, dtype=torch.complex128, generator=torch.Generator().manual_seed(0))
    compare(ph, seq, 2e-4, coils=coils)


@pytest.mark.parametrize("shape", ["lorentzian", "superlorentzian"])
def test_rate_model_bound_pool(shape):
    from test_exchange import bm_case

    ph, seq = bm_case(0, shape)
    ph = replace(ph, b1=torch.linspace(0.8, 1.2, len(ph), dtype=f64))
    m0 = torch.rand(len(ph), 4, dtype=f64, generator=torch.Generator().manual_seed(2))
    compare(ph, with_readout(seq), 1e-3, coils=bp.birdcage_sensitivities(ph.pos, 2), m_init=m0)


@pytest.mark.parametrize("concomitant", ["leading", "exact"])
def test_scanner_field_model(concomitant):
    # through the encoding positions and concomitant coefficients to the hardware parameters
    from test_fields_nonlinearity import distortion

    from blochparty.fields import Scanner

    ph, seq = random_case(1)
    seq = with_readout(seq)
    seq = replace(seq, grad=seq.grad * 30)
    out = {}
    for name, sim in (("core", bp.simulate), ("metal", metal.simulate_metal)):
        nl = distortion()
        alpha = nl.alpha.clone().requires_grad_()
        b0 = torch.tensor(0.55, dtype=f64, requires_grad=True)
        scanner = Scanner(field_strength=b0, concomitant=concomitant, nonlinearity=replace(nl, alpha=alpha))
        r = sim(ph, seq, bp.birdcage_sensitivities(ph.pos, 2), scanner=scanner)
        out[name] = torch.autograd.grad(loss(r), [alpha, b0])
    assert rel(out["metal"][0], out["core"][0]) < 2e-4 and rel(out["metal"][1], out["core"][1]) < 1e-4


def test_static_field_maps():
    zz, yy, xx = torch.meshgrid(torch.linspace(-0.1, 0.1, 4, dtype=f64), torch.linspace(-0.3, 0.3, 9, dtype=f64),
                                torch.linspace(-0.3, 0.3, 9, dtype=f64), indexing="ij")
    out = {}
    for name, sim in (("core", bp.simulate), ("metal", metal.simulate_metal)):
        b0 = (40 * xx**2 - 20 * yy).requires_grad_()
        maps = bp.FieldMaps((-0.3, -0.3, -0.1), (0.075, 0.075, 0.2 / 3), b0=b0, b1=1 + 0.3 * xx)
        ph, seq = random_case(2)
        r = sim(ph, with_readout(seq), maps=maps)
        out[name] = torch.autograd.grad(loss(r), [b0])[0]
    assert rel(out["metal"], out["core"]) < 3e-5


def test_checkpoint_interval_and_runs_do_not_change_gradients():
    # per-spin sums flushed every 64 segments, per-segment reductions in a fixed order: bit for bit
    ph = bp.shepp_logan(16)
    seq = bp.gre_2d(16, n_dummy=5, spoiling="rf", ramps=True)
    coils = bp.birdcage_sensitivities(ph.pos, 4)
    runs = [gradients(metal.simulate_metal, ph, seq, coils=coils, checkpoint_every=k) for k in (64, 128, 640, 64)]
    for r in runs[1:]:
        assert all(torch.equal(r[k], runs[0][k]) for k in r)


def test_ernst_angle_by_gradient_descent():
    # as test_gradients.test_gradient_descent_finds_ernst_angle, through the GPU adjoint
    import math

    t1, t2, tr, te = 0.1, 0.05, 5e-3, 2e-3
    spin = bp.Phantom(pos=torch.zeros(1, 3, dtype=f64), pd=torch.ones(1, dtype=f64), t1=torch.full((1,), t1, dtype=f64),
                      t2=torch.full((1,), t2, dtype=f64), db0=torch.zeros(1, dtype=f64))
    flip = torch.tensor(60.0, dtype=f64, requires_grad=True)

    def closure():
        opt.zero_grad()
        b = bp.SequenceBuilder()
        for _ in range(150):
            b.rf(1e-6, flip_deg=flip).delay(te).spoil().delay(tr - te - 1e-6)
        b.rf(1e-6, flip_deg=flip).delay(te).readout(1, 1e-9)
        loss_ = -metal.simulate_metal(spin, b.build()).signal.abs().sum()
        loss_.backward()
        return loss_

    opt = torch.optim.LBFGS([flip], lr=1, max_iter=50, line_search_fn="strong_wolfe")
    opt.step(closure)
    assert abs(abs(float(flip.detach())) - math.degrees(math.acos(math.exp(-tr / t1)))) < 0.01


def test_unsupported_inputs_raise():
    ph, seq = random_case(0)
    pd = ph.pd.clone().requires_grad_()
    n = len(ph)
    for extra in ({"diffusion": torch.full((n,), 2e-9, dtype=f64)},
                  {"motion": bp.Motion(velocity=torch.zeros(n, 3, dtype=f64))},
                  {"pools": (bp.ExchangePool(f=0.01, k=100.0, t1=1.0, t2=0.01, dw=100.0),)},
                  {"bound": bp.BoundPool(f=0.1, k=30.0, t1=1.0, t2=12e-6, model="generalized")}):
        with pytest.raises(NotImplementedError):
            metal.simulate_metal(replace(ph, pd=pd, **extra), seq)
    with pytest.raises(ValueError, match="multiple of 64"):
        metal.simulate_metal(replace(ph, pd=pd), seq, checkpoint_every=100)
