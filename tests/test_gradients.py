"""Differentiability: autograd vs. finite differences, checkpointing, and a known optimum."""

import math

import torch

import mrsim
from mrsim import Phantom, SequenceBuilder, simulate
from mrsim.reference import ernst_angle


def few_spins(t1, t2, n=3):
    g = torch.Generator().manual_seed(0)
    return Phantom(
        pos=(torch.rand(n, 3, generator=g, dtype=torch.float64) - 0.5) * 0.1,
        pd=torch.ones(n, dtype=torch.float64),
        t1=t1.expand(n) if t1.ndim == 0 else t1,
        t2=t2.expand(n) if t2.ndim == 0 else t2,
        db0=torch.linspace(-30, 30, n, dtype=torch.float64),
    )


def small_sequence(flip, tr, gx):
    b = SequenceBuilder()
    for _ in range(3):
        b.rf(50e-6, flip_deg=flip, phase=0.3)
        b.delay(1e-3, grad=(gx, 0.0, 0.0))
        b.readout(4, 20e-6, grad=(-gx, 500.0, 0.0))
        b.delay(tr - 1.13e-3).spoil()
    return b.build()


def test_gradcheck_sequence_and_tissue_parameters():
    def f(flip, tr, gx, t1, t2):
        sig = simulate(few_spins(t1, t2), small_sequence(flip, tr, gx)).signal
        return torch.view_as_real(sig)

    args = [torch.tensor(v, dtype=torch.float64, requires_grad=True)
            for v in (30.0, 8e-3, 900.0, 0.7, 0.05)]
    assert torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-6, rtol=1e-4)


def test_gradient_wrt_coil_sensitivities():
    spins = few_spins(torch.tensor(0.7, dtype=torch.float64), torch.tensor(0.05, dtype=torch.float64))
    seq = small_sequence(30.0, 8e-3, 900.0)
    coils = torch.randn(len(spins), 2, dtype=torch.complex128, requires_grad=True)
    assert torch.autograd.gradcheck(lambda c: torch.view_as_real(simulate(spins, seq, c).signal), [coils])


def test_checkpointing_gives_identical_gradients():
    phantom = mrsim.shepp_logan(16)
    flip = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
    seq = mrsim.gre_2d(16, flip_deg=flip, n_dummy=5)
    grads = []
    for every in (None, 7):
        loss = simulate(phantom, seq, checkpoint_every=every).signal.abs().sum()
        (g,) = torch.autograd.grad(loss, flip, retain_graph=True)
        grads.append(g)
    assert torch.allclose(*grads, rtol=1e-12)


def test_gradient_descent_finds_ernst_angle():
    t1, t2, tr, te = 0.1, 0.05, 5e-3, 2e-3
    spin = Phantom(pos=torch.zeros(1, 3, dtype=torch.float64), pd=torch.ones(1, dtype=torch.float64),
                   t1=torch.full((1,), t1, dtype=torch.float64), t2=torch.full((1,), t2, dtype=torch.float64),
                   db0=torch.zeros(1, dtype=torch.float64))
    flip = torch.tensor(60.0, dtype=torch.float64, requires_grad=True)

    def steady_state_signal():
        b = SequenceBuilder()
        for _ in range(150):  # (E1 cos a)^150 < 1e-6: steady state for any flip here
            b.rf(1e-6, flip_deg=flip).delay(te).spoil().delay(tr - te - 1e-6)
        b.rf(1e-6, flip_deg=flip).delay(te).readout(1, 1e-9)
        return simulate(spin, b.build()).signal.abs().sum()

    opt = torch.optim.LBFGS([flip], lr=1, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -steady_state_signal()
        loss.backward()
        return loss

    opt.step(closure)
    expected = math.degrees(ernst_angle(tr, t1))
    assert abs(flip.item() - expected) < 0.05, (flip.item(), expected)
