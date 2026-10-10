"""Reused pulses on the GPU (``simulate_metal(pulses="reuse")``): a pulse that recurs up to a global RF
phase is stepped once per spin into its affine map and applied at every occurrence turned about z by
its phase. Same signal, magnetization and bound pool as stepping every pulse (TSE, TSE with a bound
pool and 16 slices' pulses, RF-spoiled GRE with a bound pool), also with NUFFT readouts, dispatches
that would split a pulse, and a memory cap that leaves some pulses stepped."""
import pytest
import torch

import blochparty as bp
from blochparty.exchange import BoundPool

metal = pytest.importorskip("blochparty.metal")
if not metal.available():
    pytest.skip("no Apple GPU", allow_module_level=True)

f32, f64 = torch.float32, torch.float64


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


def phantom(mt=False):
    ph = bp.shepp_logan(48, chemical_shift=True)
    g = torch.Generator().manual_seed(0)
    ph = bp.Phantom(ph.pos, ph.pd, ph.t1, ph.t2, ph.db0 + (torch.rand(len(ph), generator=g, dtype=f64) - 0.5) * 40)
    ph.b1 = 0.8 + 0.4 * torch.rand(len(ph), generator=g, dtype=f64)
    if mt:
        n = len(ph)
        ph.bound = BoundPool(torch.full((n,), 0.15, dtype=f64), torch.full((n,), 19.0, dtype=f64), ph.t1,
                             torch.full((n,), 9.7e-6, dtype=f64), model="rate")
    return ph


ORDER = [0.005 * (j - 6) for j in list(range(0, 16, 2)) + list(range(1, 16, 2))]
CASES = {  # (sequence, bound pool, distinct reused pulses)
    "TSE": (lambda: bp.tse_2d(48, etl=8, tr=0.8, slice_thickness=5e-3, spoiling="gradient", n_dummy=1), False, 2),
    "TSE, bound pool, 16 slices": (lambda: bp.tse_2d(48, etl=8, tr=0.8, slice_thickness=5e-3, spoiling="gradient",
                                                     n_dummy=1, slice_order=ORDER, bound_only_beyond=0.0075), True, 6),
    "GRE, RF spoiling, bound pool": (lambda: bp.gre_2d(48, ramps=True, spoiling="rf", slice_thickness=5e-3, n_dummy=2),
                                     True, 1),
}


@pytest.mark.parametrize("name", list(CASES))
def test_same_result_as_stepping_every_pulse(name):
    make, mt, distinct = CASES[name]
    seq = make().merge_free()
    groups = metal._reused_pulses(seq, torch.tensor(seq.is_rf, dtype=torch.int32), None)
    assert len(groups) == distinct  # the case does reuse pulses
    ph = phantom(mt)
    coils = bp.birdcage_sensitivities(ph.pos, 8).to(torch.complex64)
    p32, s32 = ph.to(f32), seq.to(f32)
    step = metal.simulate_metal(p32, s32, coils)
    reuse = metal.simulate_metal(p32, s32, coils, pulses="reuse")
    print(f"{name}: reuse vs step {rel(reuse.signal, step.signal):.1e}, m {rel(reuse.m, step.m):.1e}")
    assert rel(reuse.signal, step.signal) < 1e-5  # measured 9e-8 to 3e-7
    assert rel(reuse.m, step.m) < 2e-5  # 6e-7 to 3e-6
    if mt:
        assert rel(reuse.m_bound, step.m_bound) < 2e-5
    both = metal.simulate_metal(p32, s32, coils, pulses="reuse", readouts="nufft")
    assert rel(both.signal, step.signal) < 1e-4  # the NUFFT's own level
    core = bp.simulate(ph, seq, coils.to(torch.complex128))
    assert rel(reuse.signal.to(core.signal.dtype), core.signal) < 2e-4


def test_phases_and_lengths_found():
    """RF spoiling: one pulse, every TR at its own phase (quadratic increments of 117 deg)."""
    seq = bp.gre_2d(16, spoiling="rf", slice_thickness=5e-3, n_dummy=2)
    ((a, k, occ),) = metal._reused_pulses(seq, torch.tensor(seq.is_rf, dtype=torch.int32), None)
    assert len(occ) == 18 and k == sum(seq.is_rf[a:a + k])
    rf = torch.complex(seq.rf[:, 0].double(), seq.rf[:, 1].double())
    for s, ph in occ:
        assert torch.allclose(rf[s:s + k], rf[a:a + k] * torch.exp(torch.tensor(1j * ph)), rtol=0, atol=1e-6 * rf.abs().max())  # the grouping tolerance


def test_dispatch_boundaries_and_memory_cap(monkeypatch):
    seq = CASES["TSE, bound pool, 16 slices"][0]().merge_free()
    ph = phantom(True)
    p32, s32 = ph.to(f32), seq.to(f32)
    ref = metal.simulate_metal(p32, s32, pulses="reuse")
    out = metal.simulate_metal(p32, s32, pulses="reuse", chunk=7)  # chunk ends inside pulses
    assert rel(out.signal, ref.signal) < 1e-5 and rel(out.m, ref.m) < 1e-5
    K, threads = 20, 256 * -(-len(ph) // 256)
    monkeypatch.setattr(metal, "PULSE_MEMORY", 4 * K * threads * 2)  # two pulses reused, the rest stepped
    out = metal.simulate_metal(p32, s32, pulses="reuse")
    assert rel(out.signal, ref.signal) < 1e-5 and rel(out.m, ref.m) < 1e-5


def test_refuses_what_it_does_not_cover():
    seq = CASES["TSE"][0]()
    ph = phantom()
    ph.diffusion = torch.full((len(ph),), 1e-9, dtype=f64)
    with pytest.raises(ValueError, match="static spins without diffusion"):
        metal.simulate_metal(ph.to(f32), seq.to(f32), pulses="reuse")
    with pytest.raises(ValueError, match="pulses must be"):
        metal.simulate_metal(phantom().to(f32), seq.to(f32), pulses="cached")
