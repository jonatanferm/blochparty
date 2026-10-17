"""Closed-form free runs vs. stepping segment by segment.

The fast path evaluates every stretch of RF-free segments at once; it must agree
with the segment-wise path (itself checked against the ODE solver) to rounding
error, for signals, magnetization and gradients.
"""

import pytest
import torch

import blochparty as bp
from blochparty import Phantom, simulate

from cases import busy_sequence, random_phantom


def run(merge, amp, gx, t2, coils, **kw):
    ph = random_phantom()
    ph = Phantom(ph.pos, ph.pd, ph.t1, t2, ph.db0)
    return simulate(ph, busy_sequence(amp, gx), coils, merge_free_runs=merge, **kw)


def params():
    ph = random_phantom()
    g = torch.Generator().manual_seed(1)
    coils = torch.randn(len(ph), 3, dtype=torch.complex128, generator=g)
    return [
        torch.tensor(800.0, dtype=torch.float64, requires_grad=True),
        torch.tensor(1500.0, dtype=torch.float64, requires_grad=True),
        ph.t2.clone().requires_grad_(),
        coils.requires_grad_(),
    ]


@pytest.mark.parametrize("kw", [{}, {"checkpoint_every": 7}])
def test_merged_matches_segmentwise(kw, monkeypatch):
    monkeypatch.setattr(bp.bloch, "_MAX_BLOCK", 64)  # force several sample blocks per run
    results = {}
    for merge in (True, False):
        p = params()
        r = run(merge, *p, **kw)
        loss = (r.signal.abs() ** 2).sum() + r.m.sum()
        results[merge] = (r.signal, r.m, torch.autograd.grad(loss, p))
    (s1, m1, g1), (s0, m0, g0) = results[True], results[False]
    assert s1.shape == s0.shape == (3, 4 * 13 + 3)
    assert (s1 - s0).abs().max() <= 1e-12 * s0.abs().max()
    assert (m1 - m0).abs().max() <= 1e-12
    for a, b in zip(g1, g0):
        assert (a - b).abs().max() <= 1e-10 * b.abs().max()


def test_gradcheck_through_free_runs():
    ph = random_phantom(n=4)

    def f(amp, gx):
        return torch.view_as_real(simulate(ph, busy_sequence(amp, gx)).signal)

    args = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (800.0, 1500.0)]
    assert torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-7, rtol=1e-4)


def test_run_boundaries():
    seq = busy_sequence(800.0, 1500.0)
    steps = bp.bloch._steps(seq, merge=True)
    # each TR is one RF step followed by one free run that ends at the spoiled segment
    assert len(steps) == 2 * 5
    for i, (a, b) in enumerate(steps):
        if i % 2 == 0:
            assert b - a == 1 and seq.is_rf[a]
        else:
            assert not any(seq.is_rf[a:b]) and seq.spoil[b - 1] and not any(seq.spoil[a : b - 1])
    assert len(bp.bloch._steps(seq, merge=False)) == len(seq)
