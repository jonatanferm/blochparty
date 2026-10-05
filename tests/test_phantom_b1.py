"""B1+ (transmit) maps: ``Phantom.b1`` scales the RF of every pulse, per spin, in the
PyTorch core, the signal equation, ``simulate_chunked`` and the fused kernel (forward and
adjoint); ``coils.birdcage_b1`` is an analytic transmit field."""

import math
from dataclasses import replace

import pytest
import torch
from test_bloch_vs_ode import random_case
from test_fused import head_3d, rel, with_readout

import blochparty as bp
from blochparty import Phantom, SequenceBuilder, simulate
from blochparty.chunked import simulate_chunked
from blochparty.coils import birdcage_b1
from blochparty.reference import signal_equation, spoiled_gre_signal

fused = pytest.importorskip("blochparty.fused")
SIMS = {"core": simulate, "fused": fused.simulate_fused, "chunked": lambda *a, **k: simulate_chunked(*a, spin_chunk=7, **k)}


@pytest.fixture(scope="module", autouse=True)
def extension():
    try:
        fused._ext()
    except Exception as e:  # no compiler / ninja: nothing to test
        pytest.skip(f"fused extension unavailable: {e}")


def random_b1(n, seed=0, lo=0.5, hi=1.4):
    g = torch.Generator().manual_seed(seed)
    return lo + (hi - lo) * torch.rand(n, dtype=torch.float64, generator=g)


@pytest.mark.parametrize("sim", SIMS)
def test_flip_angle_scales_exactly(sim):
    # Mz = cos(b1 alpha), Mxy = i sin(b1 alpha) after one pulse without relaxation, for any phase
    b1 = torch.tensor([0.0, 0.3, 0.5, 1.0, 1.25, 2.0, 3.7], dtype=torch.float64)
    n = len(b1)
    z = torch.zeros(n, dtype=torch.float64)
    ph = Phantom(pos=torch.zeros(n, 3, dtype=torch.float64), pd=z + 1, t1=z + 1e9, t2=z + 1e9, db0=z, b1=b1)
    for flip, phase in [(90.0, 0.0), (37.0, 0.8), (180.0, -1.1)]:
        seq = SequenceBuilder().rf(1e-6, flip_deg=flip, phase=phase).build()
        m = SIMS[sim](ph, seq).m
        alpha = b1 * math.radians(flip)
        assert torch.allclose(m[:, 2], torch.cos(alpha), atol=1e-12)
        mxy = torch.complex(m[:, 0], m[:, 1])
        assert torch.allclose(mxy, 1j * complex(math.cos(phase), math.sin(phase)) * torch.sin(alpha), atol=1e-12)


def test_signal_equation_uses_b1():
    ph = bp.shepp_logan(8)
    ph = replace(ph, b1=random_b1(len(ph)))
    seq = bp.gre_2d(8, flip_deg=40.0, rf_duration=1e-7, dwell=1e-6, n_dummy=10)
    ref = signal_equation(ph, seq)
    assert rel(simulate(ph, seq).signal, ref) < 1e-5
    assert rel(signal_equation(replace(ph, b1=None), seq), ref) > 0.05


@pytest.mark.parametrize("sim", ["core", "fused"])
def test_spoiled_gre_image_matches_ernst_at_b1_flip(sim):
    # 2D GRE with a centre-brightened birdcage B1+ map: every pixel is the ideal spoiled-GRE
    # steady state at its own flip angle b1 * alpha
    n, fov, flip, tr = 32, 0.22, 25.0, 10e-3
    maps = bp.shepp_logan_maps(n)
    b1_map = birdcage_b1(bp.grid_positions(n, fov).reshape(-1, 3), centre_brightening=0.8, radius=0.1).reshape(n, n)
    ph = bp.phantom_from_maps(maps["pd"], maps["t1"], maps["t2"], fov, b1=b1_map)
    assert ph.b1.min() < 0.7 and ph.b1.max() == 1.0
    seq = bp.gre_2d(n, fov, flip_deg=flip, tr=tr, dwell=1e-6, rf_duration=1e-6, n_dummy=300)
    img = bp.ifft2c(bp.kspace_grid(SIMS[sim](ph, seq).signal, seq))[0]
    mask = maps["pd"] > 0
    expected = 1j * spoiled_gre_signal(maps["pd"], maps["t1"], maps["t2"], tr, seq.meta["te"],
                                       b1_map * math.radians(flip))
    expected = torch.where(mask, expected, 0)
    assert (img - expected).abs().max() < 1e-4 * expected.abs().max()
    # with the nominal flip everywhere the image would be clearly different
    nominal = torch.where(mask, 1j * spoiled_gre_signal(maps["pd"], maps["t1"], maps["t2"], tr, seq.meta["te"],
                                                         math.radians(flip)), 0)
    assert (img - nominal).abs().max() > 0.05 * expected.abs().max()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_fused_forward_matches_core(dtype):
    phantom, seq = random_case(1)
    ph = replace(phantom, b1=random_b1(len(phantom), 1))
    coils = bp.birdcage_sensitivities(ph.pos, n_coils=3)
    seq = with_readout(seq.subdivide_rf(2))
    cd = torch.complex64 if dtype == torch.float32 else torch.complex128
    out = fused.simulate_fused(ph.to(dtype=dtype), seq.to(dtype=dtype), coils.to(cd))
    ref = simulate(ph.to(dtype=dtype).to(dtype=torch.float64), seq.to(dtype=dtype).to(dtype=torch.float64),
                   coils.to(cd).to(torch.complex128))
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    assert rel(out.signal.to(torch.complex128), ref.signal) <= tol and rel(out.m.double(), ref.m) <= tol
    # 3D GRE with B1+, off-resonance and 8 coils, several block sizes and threads
    ph3, c3 = head_3d(n=8, n_coils=8)
    ph3 = replace(ph3, b1=birdcage_b1(ph3.pos, 0.5, radius=0.08))
    seq3 = bp.gre_3d(8, n_dummy=5)
    ref = simulate(ph3, seq3, c3)
    for block, threads in [(16, 1), (64, 3), (256, 2)]:
        out = fused.simulate_fused(ph3, seq3, c3, block=block, n_threads=threads)
        assert rel(out.signal, ref.signal) <= 1e-12 and rel(out.m, ref.m) <= 1e-12


def test_unit_b1_changes_nothing():
    # b1 = 1 multiplies the RF exactly by one: same bits as no B1+ map (one thread: fixed sum order)
    phantom, seq = random_case(2)
    seq = with_readout(seq)
    coils = bp.birdcage_sensitivities(phantom.pos, n_coils=2)
    ones = replace(phantom, b1=torch.ones(len(phantom), dtype=torch.float64))
    for sim in (simulate, lambda *a: fused.simulate_fused(*a, n_threads=1)):
        a, b = sim(phantom, seq, coils), sim(ones, seq, coils)
        assert torch.equal(a.signal, b.signal) and torch.equal(a.m, b.m)


# --- gradients ----------------------------------------------------------------------------


def grads(sim, ph, seq, coils, names=("b1", "rf", "dt", "t1", "db0", "pos")):
    leaves = {f: getattr(ph, f).detach().clone().requires_grad_() for f in names if hasattr(ph, f)}
    sq_leaves = {f: getattr(seq, f).detach().clone().requires_grad_() for f in names if hasattr(seq, f)}
    r = sim(replace(ph, **leaves), replace(seq, **sq_leaves), coils)
    g = torch.Generator().manual_seed(9)
    w = torch.randn(r.signal.shape, dtype=torch.complex128, generator=g)
    loss = (r.signal * w).real.sum() + (r.signal.abs() ** 2).sum() + (r.m**2).sum() + r.m.sum()
    named = leaves | sq_leaves
    return dict(zip(named, torch.autograd.grad(loss, list(named.values()))))


@pytest.mark.parametrize("seed", [0, 1])
def test_b1_gradient_fused_matches_autograd(seed):
    phantom, seq = random_case(seed)
    ph = replace(phantom, b1=random_b1(len(phantom), seed))
    coils = bp.birdcage_sensitivities(ph.pos, n_coils=3)
    for sq in (with_readout(seq), with_readout(seq.subdivide_rf(3))):
        ref = grads(simulate, ph, sq, coils)
        for kw in ({}, {"block": 16, "n_threads": 3, "checkpoint_every": 5}):
            out = grads(lambda *a: fused.simulate_fused(*a, **kw), ph, sq, coils)
            errs = {k: rel(out[k], ref[k]) for k in ref}
            assert ref["b1"].abs().min() > 0 and max(errs.values()) <= 1e-10, errs
    # 2D GRE, 8 coils, B1+ map; and the chunked backward
    ph = bp.shepp_logan(12)
    ph = replace(ph, b1=birdcage_b1(ph.pos, 0.6, radius=0.1))
    coils = bp.birdcage_sensitivities(ph.pos, n_coils=8)
    sq = bp.gre_2d(12, flip_deg=30.0, rf_phase=0.3, n_dummy=10)
    ref = grads(simulate, ph, sq, coils)
    for sim in (fused.simulate_fused, SIMS["chunked"]):
        out = grads(sim, ph, sq, coils)
        errs = {k: rel(out[k], ref[k]) for k in ref}
        assert max(errs.values()) <= 1e-10, errs


def test_b1_gradcheck():
    # finite differences, through the PyTorch core and through the fused adjoint
    phantom, seq = random_case(0)
    phantom = Phantom(*(getattr(phantom, f)[:5] for f in ("pos", "pd", "t1", "t2", "db0")))
    seq = with_readout(seq)
    coils = bp.birdcage_sensitivities(phantom.pos, n_coils=2)
    b1 = random_b1(len(phantom), 4).requires_grad_()
    scale = torch.tensor(1.1, dtype=torch.float64, requires_grad=True)  # of every RF pulse
    g = torch.Generator().manual_seed(1)
    w = torch.randn(2, len(coils[0]), seq.n_adc, dtype=torch.complex128, generator=g)
    for sim in (simulate, fused.simulate_fused):
        def f(b1, scale):  # a few random projections of signal and m keep it quick
            r = sim(replace(phantom, b1=b1), replace(seq, rf=seq.rf * scale), coils)
            return torch.stack([(r.signal * w[0]).real.sum(), (r.signal * w[1]).imag.sum(), r.m.sum()]), r.m[:, 2]

        assert torch.autograd.gradcheck(f, [b1, scale], eps=1e-6, atol=1e-6, rtol=1e-4)


def test_b1_map_gradient_through_helper():
    # gradients reach the parameters of the B1+ model through the phantom
    cb = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    ph = bp.shepp_logan(8)
    seq = bp.gre_2d(8, flip_deg=60.0, n_dummy=5)

    def f(cb):
        return torch.view_as_real(fused.simulate_fused(replace(ph, b1=birdcage_b1(ph.pos, cb, radius=0.09)),
                                                       seq).signal)

    assert torch.autograd.gradcheck(f, [cb], eps=1e-6, atol=1e-6, rtol=1e-4)


# --- the analytic transmit field ---------------------------------------------------------


def test_birdcage_b1_field():
    g = torch.Generator().manual_seed(0)
    pos = (torch.rand(500, 3, generator=g, dtype=torch.float64) - 0.5) * 0.2
    assert torch.equal(birdcage_b1(pos), torch.ones(500, dtype=torch.float64))  # ideal birdcage: uniform
    radius, cb = 0.1, 0.6
    b1 = birdcage_b1(pos, cb, radius=radius)
    rho = pos[:, :2].norm(dim=1)
    # 1 on the axis, 1 / (1 + cb) at `radius`, radially symmetric, independent of z, decreasing
    edge = torch.tensor([[radius, 0, 0.3], [0, -radius, -0.1], [radius / 2**0.5, radius / 2**0.5, 0]],
                        dtype=torch.float64)
    assert torch.allclose(birdcage_b1(edge, cb, radius=radius), torch.full((3,), 1 / (1 + cb), dtype=torch.float64),
                          atol=1e-14)
    assert birdcage_b1(torch.zeros(1, 3, dtype=torch.float64), cb, radius=radius).item() == 1.0
    order = rho.argsort()
    assert torch.all(b1[order].diff() <= 0)
    # a CP field in a uniform dielectric solves the 2D Helmholtz equation: lap(B) + k^2 B = 0
    x0 = torch.tensor([[0.03, 0.02, 0.0]], dtype=torch.float64)
    h = 1e-4
    lap = sum(birdcage_b1(x0 + s * h * e, cb, radius=radius) for e in torch.eye(3, dtype=torch.float64)[:2]
              for s in (1, -1)) - 4 * birdcage_b1(x0, cb, radius=radius)
    lap = lap / h**2
    k2 = -lap / birdcage_b1(x0, cb, radius=radius)
    # k from the definition J0(k radius) = 1 / (1 + cb), via scipy's independent J0
    from scipy.optimize import brentq
    from scipy.special import j0
    k = brentq(lambda x: j0(x) - 1 / (1 + cb), 0, 2.4048) / radius
    assert abs(float(k2) / k**2 - 1) < 1e-5
