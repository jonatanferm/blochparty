"""NUFFT readouts on the GPU (``simulate_metal(readouts="nufft")``, docs/PERFORMANCE.md 2.1): every
readout as a type-1 NUFFT gives the same signal as the per-sample path and the core (RF-spoiled GRE,
EPI with alternating readouts, TSE with a bound pool and 16 slices' pulses), also when threadgroups
fall back to the per-sample path or dispatches would split a readout, and leaves the final
magnetization unchanged."""
import pytest
import torch

import blochparty as bp
from blochparty.exchange import BoundPool

from blochparty import metal  # the GPU engines' shared driver (helpers such as _readout_runs)
from helpers import GPU, gpu_engines


@pytest.fixture(params=gpu_engines())
def gpu(request):
    return GPU(request.param)


f32, f64 = torch.float32, torch.float64


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


def phantom(mt=False):
    ph = bp.shepp_logan(48, chemical_shift=True)
    g = torch.Generator().manual_seed(0)
    ph = bp.Phantom(ph.pos, ph.pd, ph.t1, ph.t2, ph.db0 + (torch.rand(len(ph), generator=g, dtype=f64) - 0.5) * 40)
    if mt:
        n = len(ph)
        ph.bound = BoundPool(torch.full((n,), 0.15, dtype=f64), torch.full((n,), 19.0, dtype=f64), ph.t1,
                             torch.full((n,), 9.7e-6, dtype=f64), model="rate")
    return ph


CASES = {
    "GRE, RF spoiling": lambda: bp.gre_2d(48, ramps=True, spoiling="rf", slice_thickness=5e-3, n_dummy=2),
    "EPI, alternating readouts": lambda: bp.epi_2d(48, dwell=5e-6, slice_thickness=5e-3),
    "TSE, bound pool, 16 slices": lambda: bp.tse_2d(48, etl=8, tr=0.8, slice_thickness=5e-3, spoiling="gradient",
                                                    n_dummy=1, slice_order=[0.005 * (j - 6) for j in
                                                                            list(range(0, 16, 2)) + list(range(1, 16, 2))],
                                                    bound_only_beyond=0.0075),
}


@pytest.mark.parametrize("name", list(CASES))
def test_same_signal_as_per_sample_and_core(gpu, name):
    seq = CASES[name]().merge_free()
    ph = phantom(mt="bound" in name)
    coils = bp.birdcage_sensitivities(ph.pos, 8)
    direct = gpu.simulate(ph.to(f32), seq.to(f32), coils.to(torch.complex64))
    fast = gpu.simulate(ph.to(f32), seq.to(f32), coils.to(torch.complex64), readouts="nufft")
    assert metal._readout_runs(seq)  # the case does take the NUFFT path
    print(f"{name}: NUFFT vs per-sample {rel(fast.signal, direct.signal):.1e}")
    assert rel(fast.signal, direct.signal) < 1e-4  # measured 1.2e-6 (TSE) to 2.1e-5 (GRE)
    # one jump over each readout instead of n float32 steps: the end state moves at float32's level
    assert rel(fast.m, direct.m) < 1e-3  # measured 1.5e-5 to 2.5e-4
    core = bp.simulate(ph, seq, coils)
    assert rel(fast.signal.to(core.signal.dtype), core.signal) < 2e-4
    assert rel(fast.m.double(), core.m) < 2e-3  # per-sample vs core: 2e-4 to 9e-4


def test_fallback_and_dispatch_boundaries(gpu, monkeypatch):
    seq = CASES["GRE, RF spoiling"]().merge_free()
    ph, coils = phantom(), bp.birdcage_sensitivities(phantom().pos, 4).to(torch.complex64)
    ref = gpu.simulate(ph.to(f32), seq.to(f32), coils).signal
    monkeypatch.setattr(metal, "NUFFT_WMAX", 7)  # most threadgroups keep the per-sample path
    assert rel(gpu.simulate(ph.to(f32), seq.to(f32), coils, readouts="nufft").signal, ref) < 1e-4
    monkeypatch.undo()
    out = gpu.simulate(ph.to(f32), seq.to(f32), coils, readouts="nufft", chunk=301)  # chunks inside readouts
    assert rel(out.signal, ref) < 1e-4


def test_refuses_what_it_does_not_cover(gpu):
    seq = CASES["GRE, RF spoiling"]()
    ph = phantom()
    ph.diffusion = torch.full((len(ph),), 1e-9, dtype=f64)
    with pytest.raises(ValueError, match="static spins without diffusion"):
        gpu.simulate(ph.to(f32), seq.to(f32), readouts="nufft")
    with pytest.raises(ValueError, match="readouts must be"):
        gpu.simulate(phantom().to(f32), seq.to(f32), readouts="fft")
