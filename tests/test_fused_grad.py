"""Fused kernel's hand-written adjoint vs. autograd through the PyTorch simulator.

float64: every input's gradient must match ``simulate``'s autograd to <= 1e-10 relative
(max-abs error over max-abs reference, per tensor); they agree to ~1e-14 in practice.
gradcheck covers the same code against finite differences, independently of PyTorch's
simulator. The checkpoint interval, block size and thread count must not change the
gradients (bit for bit with one thread).
"""

import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import blochparty as bp
from blochparty import Phantom, SequenceBuilder, simulate
from test_bloch_vs_ode import random_case
from test_free_runs import busy_sequence, random_phantom
from test_fused import head_3d, rel, with_readout
from test_gradients import few_spins, small_sequence

fused = pytest.importorskip("blochparty.fused")
ROOT = Path(__file__).parents[1]

SPIN = ("pos", "pd", "t1", "t2", "db0")
SEQ = ("dt", "rf", "grad", "adc_phase")


@pytest.fixture(scope="module", autouse=True)
def extension():
    try:
        fused._ext()
    except Exception as e:  # no compiler / ninja: nothing to test
        pytest.skip(f"fused extension unavailable: {e}")


def leaves(phantom, seq, coils=None, m_init=None, which=SPIN + SEQ + ("coils", "m_init")):
    """Copies of every differentiable input; those named in ``which`` require grad."""
    req = lambda name, t: None if t is None else t.detach().clone().requires_grad_(name in which)
    ph = Phantom(*(req(f, getattr(phantom, f)) for f in SPIN))
    sq = replace(seq, **{f: req(f, getattr(seq, f)) for f in SEQ})
    named = {f: getattr(ph, f) for f in SPIN} | {f: getattr(sq, f) for f in SEQ}
    named |= {"coils": req("coils", coils), "m_init": req("m_init", m_init)}
    return ph, sq, named["coils"], named["m_init"], {k: v for k, v in named.items() if v is not None}


def weights(r, seed=3):
    g = torch.Generator().manual_seed(seed)
    w_sig = torch.randn(r.signal.shape, dtype=torch.complex128, generator=g).to(r.signal.dtype)
    w_m = torch.randn(r.m.shape, dtype=torch.float64, generator=g).to(r.m.dtype)
    return w_sig, w_m


def loss_both(r):
    """Uses both outputs, linearly and quadratically."""
    w_sig, w_m = weights(r)
    return (r.signal * w_sig).real.sum() + (r.signal.abs() ** 2).sum() + (r.m * w_m).sum() + (r.m**2).sum()


def loss_signal(r):
    return (r.signal * weights(r)[0]).real.sum()


def loss_m(r):
    return (r.m * weights(r)[1]).sum()


def gradients(sim, phantom, seq, coils=None, m_init=None, loss=loss_both, which=SPIN + SEQ + ("coils", "m_init"),
              **kw):
    ph, sq, cl, m0, named = leaves(phantom, seq, coils, m_init, which)
    r = sim(ph, sq, cl, m_init=m0, **kw)
    wanted = {k: v for k, v in named.items() if v.requires_grad}
    grads = torch.autograd.grad(loss(r), list(wanted.values()), allow_unused=True)
    return dict(zip(wanted, grads))


def compare(phantom, seq, coils=None, m_init=None, loss=loss_both, tol=1e-10, **kw):
    """Fused adjoint vs. autograd through ``simulate``; returns the relative errors."""
    ref = gradients(simulate, phantom, seq, coils, m_init, loss)
    out = gradients(fused.simulate_fused, phantom, seq, coils, m_init, loss, **kw)
    assert out.keys() == ref.keys()
    errs = {}
    for k in ref:
        assert (out[k] is None) == (ref[k] is None), k  # None where the loss does not depend on it
        if ref[k] is None:
            continue
        assert out[k].shape == ref[k].shape and out[k].dtype == ref[k].dtype, k
        errs[k] = rel(out[k], ref[k]) if ref[k].abs().max() > 0 else float(out[k].abs().max())
    assert max(errs.values()) <= tol, errs
    return errs


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_rf_gradients_off_resonance(seed):
    phantom, seq = random_case(seed)
    coils = bp.birdcage_sensitivities(phantom.pos, n_coils=3)
    compare(phantom, with_readout(seq), coils)
    compare(phantom, with_readout(seq.subdivide_rf(3)), coils)
    g = torch.Generator().manual_seed(seed)
    m0 = torch.randn(len(phantom), 3, dtype=torch.float64, generator=g)
    compare(phantom, with_readout(seq), coils, m_init=m0)


def test_gre_2d_eight_coils():
    phantom = bp.shepp_logan(16)
    coils = bp.birdcage_sensitivities(phantom.pos, n_coils=8)
    compare(phantom, bp.gre_2d(16, flip_deg=20.0, rf_phase=0.4, n_dummy=20), coils)


def test_gre_3d_off_resonance():
    phantom, coils = head_3d(n=8, n_coils=5)
    assert phantom.db0.abs().min() > 0
    compare(phantom, bp.gre_3d(8, n_dummy=10), coils)


def test_spoils_and_adc_right_after_rf():
    phantom = random_phantom(n=20)
    seq = busy_sequence(800.0, 1500.0)  # ends with an RF pulse directly followed by a readout
    assert seq.is_rf[-6] and seq.adc[-5] and any(seq.spoil)
    coils = torch.randn(len(phantom), 3, dtype=torch.complex128, generator=torch.Generator().manual_seed(2))
    compare(phantom, seq, coils)
    # sampling at the end of an RF segment, and an RF segment that spoils
    adc, spoil = list(seq.adc), list(seq.spoil)
    adc[0] = spoil[0] = adc[-6] = True
    compare(phantom, replace(seq, adc=adc, spoil=spoil), coils)


@pytest.mark.parametrize("loss", [loss_signal, loss_m])
def test_one_output_only(loss):
    # the other output's incoming gradient is None (not zeros)
    phantom, seq = random_case(1)
    compare(phantom, with_readout(seq), bp.birdcage_sensitivities(phantom.pos, n_coils=2), loss=loss)


def test_gradcheck_sequence_and_tissue_parameters():
    def f(flip, tr, gx, t1, t2, psi):
        seq = small_sequence(flip, tr, gx)
        seq = replace(seq, adc_phase=seq.adc_phase + psi)
        r = fused.simulate_fused(few_spins(t1, t2), seq)
        return torch.view_as_real(r.signal), r.m

    args = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (30.0, 8e-3, 900.0, 0.7, 0.05, 0.3)]
    assert torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-6, rtol=1e-4)


def test_gradcheck_spins_coils_and_initial_magnetization():
    spins = few_spins(torch.tensor(0.7, dtype=torch.float64), torch.tensor(0.05, dtype=torch.float64))
    seq = small_sequence(30.0, 8e-3, 900.0)
    g = torch.Generator().manual_seed(0)
    coils = torch.randn(len(spins), 2, dtype=torch.complex128, generator=g, requires_grad=True)
    m0 = torch.randn(len(spins), 3, dtype=torch.float64, generator=g, requires_grad=True)
    pos, db0, pd = (t.clone().requires_grad_() for t in (spins.pos, spins.db0, spins.pd))

    def f(coils, m0, pos, db0, pd):
        r = fused.simulate_fused(replace(spins, pos=pos, db0=db0, pd=pd), seq, coils, m_init=m0)
        return torch.view_as_real(r.signal), r.m

    assert torch.autograd.gradcheck(f, [coils, m0, pos, db0, pd], eps=1e-6, atol=1e-6, rtol=1e-4)
    # with m_init=None the PD gradient also flows through the equilibrium start
    assert torch.autograd.gradcheck(lambda pd: torch.view_as_real(
        fused.simulate_fused(replace(spins, pd=pd), seq, coils.detach()).signal), [pd], eps=1e-6, atol=1e-6)


def test_checkpoint_interval_does_not_matter():
    phantom, seq = random_case(2)
    seq = with_readout(seq)
    coils = bp.birdcage_sensitivities(phantom.pos, n_coils=3)
    S = len(seq)
    runs = [gradients(fused.simulate_fused, phantom, seq, coils, checkpoint_every=k, n_threads=1)
            for k in (None, 1, 7, S - 1, S, S + 5)]
    for other in runs[1:]:
        for k in runs[0]:
            assert torch.equal(runs[0][k], other[k]), k


@pytest.mark.parametrize("block", fused.BLOCK_SIZES)
def test_blocks_and_threads(block):
    phantom, coils = head_3d(n=8, n_coils=6)  # spin count not a multiple of any block
    assert all(len(phantom) % b for b in fused.BLOCK_SIZES)
    seq = bp.gre_3d(8, n_dummy=3)
    one = gradients(fused.simulate_fused, phantom, seq, coils, block=block, n_threads=1)
    many = gradients(fused.simulate_fused, phantom, seq, coils, block=block, n_threads=3)
    for k in one:  # threads only change the order of the sums over spins
        assert rel(many[k], one[k]) <= 1e-13, k
    compare(phantom, seq, coils, block=block, n_threads=3)


def test_single_spin_and_empty_sequence():
    one = Phantom(pos=torch.tensor([[0.01, -0.02, 0.005]], dtype=torch.float64),
                  pd=torch.tensor([0.8], dtype=torch.float64), t1=torch.tensor([0.9], dtype=torch.float64),
                  t2=torch.tensor([0.08], dtype=torch.float64), db0=torch.tensor([12.0], dtype=torch.float64))
    compare(one, bp.gre_2d(8, n_dummy=3), bp.birdcage_sensitivities(one.pos, n_coils=2))
    seq = small_sequence(30.0, 8e-3, 900.0)
    empty = replace(seq, dt=seq.dt[:0], rf=seq.rf[:0], grad=seq.grad[:0], adc_phase=seq.adc_phase[:0],
                    is_rf=[], adc=[], spoil=[])
    m0 = torch.tensor([[0.3, -0.2, 0.5]], dtype=torch.float64)
    g = gradients(fused.simulate_fused, one, empty, m_init=m0, loss=loss_m)
    assert torch.equal(g["m_init"], weights(fused.simulate_fused(one, empty, m_init=m0))[1])
    assert not any(g[k] is not None and g[k].any() for k in g if k != "m_init")


def test_only_requested_gradients():
    phantom, seq = random_case(0)
    seq = with_readout(seq)
    coils = bp.birdcage_sensitivities(phantom.pos, n_coils=2)
    for which in (("t1",), ("coils",), ("rf", "m_init"), ("adc_phase",)):
        ph, sq, cl, m0, named = leaves(phantom, seq, coils, phantom.pos * 0 + 0.1, which)
        loss_both(fused.simulate_fused(ph, sq, cl, m_init=m0)).backward()
        for k, t in named.items():
            assert (t.grad is not None) == (k in which), (which, k)
        ref = gradients(simulate, phantom, seq, coils, phantom.pos * 0 + 0.1, which=which)
        for k in which:
            assert rel(named[k].grad, ref[k]) <= 1e-10


def test_gradients_reach_builders_and_maps():
    # flip angle through gre_2d, T1 map through phantom_from_maps, as in the README
    maps = bp.shepp_logan_maps(12)

    def run(sim):
        t1 = maps["t1"].clone().requires_grad_()
        flip = torch.tensor(15.0, dtype=torch.float64, requires_grad=True)
        phantom = bp.phantom_from_maps(maps["pd"], t1, maps["t2"], fov=0.22)
        kspace = sim(phantom, bp.gre_2d(12, flip_deg=flip, n_dummy=10)).signal
        kspace.abs().sum().backward()
        return flip.grad, t1.grad

    (f1, t1), (f0, t0) = run(fused.simulate_fused), run(simulate)
    assert rel(f1, f0) <= 1e-10 and rel(t1, t0) <= 1e-10
    assert t1[maps["pd"] == 0].abs().max() == 0 and t1.abs().max() > 0


def test_forward_unchanged_by_grad_mode():
    phantom, coils = head_3d(n=8, n_coils=3)
    seq = bp.gre_3d(8, n_dummy=3)
    plain = fused.simulate_fused(phantom, seq, coils, n_threads=1)
    assert not plain.signal.requires_grad
    ph, sq, cl, _, _ = leaves(phantom, seq, coils)
    tracked = fused.simulate_fused(ph, sq, cl, n_threads=1)
    assert tracked.signal.requires_grad and tracked.m.requires_grad
    assert torch.equal(tracked.signal.detach(), plain.signal) and torch.equal(tracked.m.detach(), plain.m)
    with torch.no_grad():
        assert not fused.simulate_fused(ph, sq, cl).signal.requires_grad
    with pytest.raises(ValueError):
        fused.simulate_fused(phantom, seq, coils, checkpoint_every=0)


def test_float32_gradients():
    # float32 rounding is amplified like the forward's (see test_fused.py); bound both paths
    phantom = bp.shepp_logan(16)
    coils = bp.birdcage_sensitivities(phantom.pos, n_coils=4)
    seq = bp.gre_2d(16, flip_deg=20.0, n_dummy=20).to(dtype=torch.float32)
    ph32 = phantom.to(dtype=torch.float32)
    ref = gradients(simulate, ph32.to(dtype=torch.float64), seq.to(dtype=torch.float64), coils)
    f32 = gradients(fused.simulate_fused, ph32, seq, coils.to(torch.complex64))
    t32 = gradients(simulate, ph32, seq, coils.to(torch.complex64))
    for k in ref:
        assert f32[k].dtype == t32[k].dtype
        assert rel(f32[k].to(ref[k].dtype), ref[k]) <= 2e-5, k
        assert rel(t32[k].to(ref[k].dtype), ref[k]) <= 2e-5, k


@pytest.mark.slow
def test_optimize_contrast_example():
    # the example asserts that gradient descent through the fused adjoint lands within
    # 0.05 deg of the analytic optimum
    out = subprocess.run([sys.executable, str(ROOT / "examples" / "optimize_contrast.py"), "--fused"],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "analytic optimum: 16.109 deg" in out.stdout


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("c++") is None, reason="no C++ compiler")
def test_kernel_builds_standalone(tmp_path):
    # kernel.h needs neither PyTorch nor more than C++17; finite differences in plain C++
    src = tmp_path / "main.cpp"
    src.write_text(STANDALONE)
    exe = tmp_path / "main"
    flags = ["-std=c++17", "-O2", "-fopenmp-simd", "-Wall", "-Wextra", "-Werror", "-pthread"]
    build = subprocess.run(["c++", *flags, f"-I{ROOT / 'src/blochparty/fused'}", str(src), "-o", str(exe)],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


STANDALONE = r"""
#include "kernel.h"
#include <cmath>
#include <cstdio>
#include <vector>
using namespace blochparty::fused;

// L = sum of the signal's real parts + sum of final m; d/dt1 of spin 1 by the adjoint and
// by central differences.
double loss(std::vector<double> t1, std::vector<double>* g_t1) {
    const int n = 3, ns = 6;
    std::vector<double> pos = {0.01, 0, 0, -0.02, 0.01, 0, 0, 0, 0.03}, pd = {1, 0.8, 0.6}, t2 = {0.1, 0.05, 0.2};
    std::vector<double> db0 = {5, -10, 0}, sens = {1, 0, 0.5, 0.5, 0.2, -0.3};
    std::vector<double> dt = {1e-4, 1e-3, 1e-5, 1e-5, 2e-3, 1e-4}, rf = {300, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200};
    std::vector<double> grad(3 * ns, 500.0), psi = {0, 0, 0.2, 0.3, 0, 0};
    std::vector<uint8_t> fl = {kRF, 0, kADC, kADC, kSpoil, kRF | kADC};
    std::vector<double> m = {0, 0, 1, 0, 0, 0.8, 0, 0, 0.6}, sig(3 * 2), ck(3 * n * 3);
    Spins<double> sp{n, 1, pos.data(), pd.data(), t1.data(), t2.data(), db0.data(), sens.data()};
    Segments<double> seq{ns, dt.data(), rf.data(), grad.data(), psi.data(), fl.data()};
    simulate<double>(sp, seq, m.data(), sig.data(), 2, 16, ck.data(), 2);
    double l = 0;
    for (double s : sig) l += s;  // re and im alike
    for (double v : m) l += v;
    if (g_t1) {
        std::vector<double> gs(sig.size(), 1.0), gm(m.size(), 1.0);
        Grads<double> g;
        g.t1 = g_t1->data();
        adjoint<double>(sp, seq, ck.data(), 2, sig.data(), gs.data(), gm.data(), g, 2, 16);
    }
    return l;
}

int main() {
    std::vector<double> t1 = {0.9, 0.5, 1.2}, g(3);
    loss(t1, &g);
    for (int i = 0; i < 3; ++i) {
        auto a = t1, b = t1;
        a[i] += 1e-4, b[i] -= 1e-4;
        const double fd = (loss(a, nullptr) - loss(b, nullptr)) / 2e-4;
        std::printf("dL/dt1[%d]: adjoint %.12g, finite difference %.12g\n", i, g[i], fd);
        if (!(std::abs(fd - g[i]) <= 5e-7 * std::abs(fd))) return 1;  // FD rounding: ~1e-7
    }
    return 0;
}
"""
