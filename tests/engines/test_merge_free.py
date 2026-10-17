"""``Sequence.merge_free`` (docs/PERFORMANCE.md 2.4): every stretch of RF- and ADC-free segments as
one segment of the same duration, gradient moment and b0_offset integral gives the same signal
for static spins (core in float64, Metal in float32), keeps RF and ADC segments bit for bit, and
passes gradients to the original segments."""
import pytest
import torch

import blochparty as bp
from blochparty.exchange import BoundPool
from helpers import rel

f64 = torch.float64


def phantom(n=400, mt=True):
    g = torch.Generator().manual_seed(0)
    pos = (torch.rand(n, 3, generator=g, dtype=f64) - 0.5) * torch.tensor([0.2, 0.2, 0.008], dtype=f64)
    ph = bp.Phantom(pos, torch.rand(n, generator=g, dtype=f64) + 0.5, torch.full((n,), 1.0, dtype=f64),
                    torch.full((n,), 0.08, dtype=f64), (torch.rand(n, generator=g, dtype=f64) - 0.5) * 60)
    if mt:
        ph.bound = BoundPool(torch.full((n,), 0.15, dtype=f64), torch.full((n,), 19.0, dtype=f64), ph.t1,
                             torch.full((n,), 9.7e-6, dtype=f64), model="rate")
    return ph


CASES = {
    "TSE, crushers, gradient spoiling": lambda: bp.tse_2d(16, etl=4, tr=0.5, slice_thickness=5e-3, spoiling="gradient",
                                                          n_dummy=1),
    "TSE, 5 slices (bound-only far pulses)": lambda: bp.tse_2d(
        16, etl=4, tr=0.5, slice_thickness=5e-3, spoiling="gradient", n_dummy=1,
        slice_order=[-0.01, 0.0, 0.01, -0.005, 0.005], bound_only_beyond=0.0075),
    "GRE, RF spoiling, ramps": lambda: bp.gre_2d(16, ramps=True, spoiling="rf", slice_thickness=5e-3, n_dummy=2),
    "EPI (ADC on ramps)": lambda: bp.epi_2d(16, dwell=5e-6, slice_thickness=5e-3),
}


@pytest.mark.parametrize("name", list(CASES))
def test_same_signal_fewer_segments(name):
    seq = CASES[name]()
    merged = seq.merge_free()
    assert len(merged) < len(seq) and merged.n_adc == seq.n_adc and abs(float(merged.duration - seq.duration)) < 1e-12
    keep = [i for i in range(len(seq)) if seq.is_rf[i] or seq.adc[i]]
    kept = [i for i in range(len(merged)) if merged.is_rf[i] or merged.adc[i]]
    assert torch.equal(seq.rf[keep], merged.rf[kept]) and torch.equal(seq.grad[keep], merged.grad[kept])
    assert torch.equal(seq.adc_phase[keep], merged.adc_phase[kept]) and torch.equal(seq.dt[keep], merged.dt[kept])
    ph = phantom(mt="slices" in name)
    a, b = bp.simulate(ph, seq), bp.simulate(ph, merged)
    assert rel(b.signal, a.signal) < 1e-11 and rel(b.m, a.m) < 1e-11


@pytest.mark.parametrize("name", list(CASES))
def test_same_signal_fewer_segments_gpu(gpu, name):
    seq = CASES[name]()
    merged = seq.merge_free()
    ph = phantom(mt="slices" in name)
    a = bp.simulate(ph, seq)
    p32 = ph.to(torch.float32)
    x = gpu.simulate(p32, seq.to(torch.float32)).signal.to(a.signal.dtype)
    y = gpu.simulate(p32, merged.to(torch.float32)).signal.to(a.signal.dtype)
    assert rel(y, x) < 2e-5  # merged vs unmerged on the GPU: measured 2e-6
    assert rel(y, a.signal) < 1e-4


def test_gradients_reach_the_original_segments():
    """Static spins see a free stretch only through its total moment, so the gradient with respect to every original
    segment's gradient is the same through the merged sequence (where the merged segment's moment carries its parts)
    as through the original one."""
    from dataclasses import replace

    seq = bp.tse_2d(8, etl=4, tr=0.3, slice_thickness=5e-3, spoiling="gradient")
    ph = phantom(60, mt=False)

    def grads(merge):
        grad = seq.grad.clone().requires_grad_(True)
        s = replace(seq, grad=grad)
        bp.simulate(ph, s.merge_free() if merge else s).signal.abs().square().sum().backward()
        return grad.grad

    merged, plain = grads(True), grads(False)
    free = [i for i in range(len(seq)) if not seq.is_rf[i] and not seq.adc[i]]
    assert merged[free].abs().sum() > 0.1 * merged.abs().sum()  # the merged segments' parts carry real gradient
    assert rel(merged, plain) < 1e-9


def test_nothing_to_merge_returns_the_sequence():
    seq = bp.SequenceBuilder().rf(1e-3, flip_deg=30, n_steps=4).readout(8, 1e-5).build()
    assert seq.merge_free() is seq
