"""Spin-chunked backward: identical results and gradients to plain ``simulate``, and
saved-tensor memory that scales with the chunk size instead of the spin count."""

import pytest
import torch

import blochparty as bp
from blochparty import Phantom, SequenceBuilder, simulate
from blochparty.chunked import simulate_chunked

from helpers import rel

N_MAP, FOV = 10, 0.22  # 51 spins: chunks of 17 divide it, 16 does not


def builder_sequence(flip, tr, gx, n_tr=3):
    b = SequenceBuilder()
    for _ in range(2):  # dummies
        b.rf(50e-6, flip_deg=flip, phase=0.3).delay(tr - 50e-6).spoil()
    for i in range(n_tr):
        b.rf(50e-6, flip_deg=flip, phase=0.3)
        b.delay(1e-3, grad=(gx, 200.0 * (i - 1), 0.0))
        b.readout(6, 20e-6, grad=(-gx, 0.0, 0.0), phase=0.1 * i)
        b.delay(tr - 1.17e-3).spoil()
    return b.build()


def make_inputs():
    """Leaf tensors for everything differentiable, and a builder that turns them
    into (phantom, seq, coils) the way user code would."""
    maps = bp.shepp_logan_maps(N_MAP)
    g = torch.Generator().manual_seed(1)
    leaves = {
        "flip": torch.tensor(25.0, dtype=torch.float64),
        "tr": torch.tensor(6e-3, dtype=torch.float64),
        "gx": torch.tensor(800.0, dtype=torch.float64),
        "pd": maps["pd"].clone(),
        "t1": maps["t1"].clone(),
        "t2": maps["t2"].clone(),
        "db0": 20.0 * torch.randn(maps["pd"].shape, generator=g, dtype=torch.float64),
    }
    for t in leaves.values():
        t.requires_grad_()

    def build():
        phantom = bp.phantom_from_maps(leaves["pd"], leaves["t1"], leaves["t2"], FOV, db0=leaves["db0"])
        return phantom, builder_sequence(leaves["flip"], leaves["tr"], leaves["gx"])

    phantom, _ = build()
    coils = bp.birdcage_sensitivities(phantom.pos, n_coils=3).requires_grad_()
    leaves["coils"] = coils
    return leaves, build


def loss_fn(res, weight):
    # Uses both outputs, so dL/dm reaches the backward alongside dL/d(signal).
    return (res.signal.abs() ** 2).sum() + (weight * res.m).sum()


@pytest.mark.parametrize("spin_chunk", [1, 16, 17, 51, 1000])
def test_forward_matches_simulate(spin_chunk):
    leaves, build = make_inputs()
    phantom, seq = build()
    ref = simulate(phantom, seq, leaves["coils"])
    out = simulate_chunked(phantom, seq, leaves["coils"], spin_chunk=spin_chunk)
    assert out.signal.shape == ref.signal.shape and out.m.shape == ref.m.shape
    assert rel(out.signal, ref.signal) <= 1e-12
    assert rel(out.m, ref.m) <= 1e-12


@pytest.mark.parametrize("spin_chunk,checkpoint_every", [(16, None), (17, None), (51, None), (1000, None),
                                                         (16, 7), (17, 50)])
def test_gradients_match_simulate(spin_chunk, checkpoint_every):
    leaves, build = make_inputs()
    weight = torch.randn(len(build()[0]), 3, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    names = list(leaves)

    def grads(sim, **kw):
        phantom, seq = build()
        loss = loss_fn(sim(phantom, seq, leaves["coils"], **kw), weight)
        return dict(zip(names, torch.autograd.grad(loss, [leaves[k] for k in names])))

    ref = grads(simulate, checkpoint_every=checkpoint_every)
    out = grads(simulate_chunked, spin_chunk=spin_chunk, checkpoint_every=checkpoint_every)
    for k in names:
        assert ref[k].abs().max() > 0, k  # every parameter actually matters
        assert rel(out[k], ref[k]) <= 1e-10, (k, rel(out[k], ref[k]))


@pytest.mark.slow
def test_gradcheck_tiny():
    n = 5
    g = torch.Generator().manual_seed(0)
    pos = (torch.rand(n, 3, generator=g, dtype=torch.float64) - 0.5) * 0.1

    def f(flip, tr, gx, t1, db0, coils):
        phantom = Phantom(pos=pos, pd=torch.ones(n, dtype=torch.float64), t1=t1,
                          t2=torch.full((n,), 0.05, dtype=torch.float64), db0=db0)
        res = simulate_chunked(phantom, builder_sequence(flip, tr, gx, n_tr=2), coils, spin_chunk=2)
        return torch.view_as_real(res.signal), res.m

    args = [torch.tensor(30.0, dtype=torch.float64), torch.tensor(8e-3, dtype=torch.float64),
            torch.tensor(900.0, dtype=torch.float64), torch.linspace(0.5, 1.0, n, dtype=torch.float64),
            torch.linspace(-30.0, 30.0, n, dtype=torch.float64),
            torch.randn(n, 2, generator=g, dtype=torch.complex128)]
    args = [a.requires_grad_() for a in args]
    assert torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-6, rtol=1e-4)


class SavedTensorMeter:
    """Peak bytes held by autograd's saved tensors, counting each storage once while
    any saved reference to it is alive."""

    def __init__(self):
        self.live, self.peak, self.refs = 0, 0, {}

    def pack(self, t):
        storage = t.untyped_storage()
        key = storage.data_ptr()
        if key not in self.refs:
            self.refs[key] = [0, storage.nbytes()]
            self.live += storage.nbytes()
            self.peak = max(self.peak, self.live)
        self.refs[key][0] += 1
        return _Held(self, key, t)

    @staticmethod
    def unpack(held):
        return held.t


class _Held:
    def __init__(self, meter, key, t):
        self.meter, self.key, self.t = meter, key, t

    def __del__(self):  # runs when autograd frees the saved tensor
        ref = self.meter.refs[self.key]
        ref[0] -= 1
        if ref[0] == 0:
            self.meter.live -= ref[1]
            del self.meter.refs[self.key]


def test_saved_tensor_memory_scales_with_chunk():
    n = 4096
    g = torch.Generator().manual_seed(0)
    t1 = torch.full((n,), 0.8, dtype=torch.float64, requires_grad=True)
    phantom = Phantom(pos=(torch.rand(n, 3, generator=g, dtype=torch.float64) - 0.5) * 0.2,
                      pd=torch.ones(n, dtype=torch.float64), t1=t1,
                      t2=torch.full((n,), 0.06, dtype=torch.float64), db0=torch.zeros(n, dtype=torch.float64))
    flip = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
    seq = builder_sequence(flip, torch.tensor(6e-3, dtype=torch.float64), 900.0, n_tr=16)
    coils = bp.birdcage_sensitivities(phantom.pos, n_coils=4).requires_grad_()

    def peak(sim, **kw):
        meter = SavedTensorMeter()
        with torch.autograd.graph.saved_tensors_hooks(meter.pack, meter.unpack):
            res = sim(phantom, seq, coils, **kw)
            loss = res.signal.abs().sum() + res.m.sum()
            gr = torch.autograd.grad(loss, [flip, t1, coils], retain_graph=True)
        return meter.peak, gr

    plain, g_ref = peak(simulate)
    chunked = {c: peak(simulate_chunked, spin_chunk=c) for c in (n, 1024, 256)}
    for _, gr in chunked.values():
        assert all(rel(a, b) <= 1e-10 for a, b in zip(gr, g_ref))
    # The Function keeps its O(N) inputs (phantom, coils, sequence) for the recompute;
    # on top of that it holds one chunk's graph at a time.
    inputs = (seq.dt, seq.rf, seq.grad, seq.adc_phase, phantom.pos, phantom.pd, t1, phantom.t2,
              phantom.db0, coils)
    floor = sum(t.untyped_storage().nbytes() for t in inputs)
    mib = lambda b: f"{b / 2**20:.2f} MiB"
    print(f"\nsaved-tensor peak, {n} spins x {len(seq)} segments: plain {mib(plain)}; "
          + ", ".join(f"chunk {c}: {mib(p)}" for c, (p, _) in chunked.items())
          + f" (inputs {mib(floor)}); plain / chunk 256 = {plain / chunked[256][0]:.1f}x, "
          + f"excluding inputs {plain / (chunked[256][0] - floor):.1f}x")
    assert 0.9 < chunked[n][0] / plain < 1.1  # one chunk of N costs what plain autograd does
    # including the fixed inputs (with the closed-form core, plain stores ~22 MiB here)
    assert plain / chunked[256][0] > 10  # chunk N/16
    assert 15 < plain / (chunked[256][0] - floor) < 17
    assert 3.8 < (chunked[1024][0] - floor) / (chunked[256][0] - floor) < 4.2
