"""Multi-slice TSE as one slice sees it (``tse_2d(slice_order=...)``): the other slices' pulses on a
coarse raster, moved to their slices by ``b0_offset``, give the same signal as the exact
phase-ramped pulses (incidental MT and cross-talk), cost a fraction, leave the single-slice
sequence untouched, and Metal agrees with the core."""
import pytest
import torch

import blochparty as bp
from blochparty.exchange import BoundPool

f64 = torch.float64
ORDER = [-0.005, 0.0, 0.005]  # interleaved neighbours 5 mm away: cross-talk and MT
KW = dict(n=8, etl=4, tr=0.6, slice_thickness=5e-3, spoiling="gradient", n_dummy=1, esp=12.5e-3)


def wm_column(n=11, mt=True):
    z = torch.linspace(-4.5e-3, 4.5e-3, n, dtype=f64)
    pos = torch.stack([torch.linspace(-0.04, 0.04, n, dtype=f64), torch.zeros_like(z), z], -1)
    ph = bp.Phantom(pos, torch.ones(n, dtype=f64), torch.full((n,), 1.084, dtype=f64),
                    torch.full((n,), 0.069, dtype=f64), torch.linspace(-15, 15, n, dtype=f64))
    if mt:  # white matter at 3 T, anatomy.DEFAULT_TISSUES: f 0.1277 of the total, k 19 /s, T2b 9.7 us
        ph.bound = BoundPool(torch.full((n,), 0.1277 / 0.8723, dtype=f64), torch.full((n,), 19.0, dtype=f64), ph.t1,
                             torch.full((n,), 9.7e-6, dtype=f64), model="rate")
    return ph


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


def test_single_slice_is_untouched_and_timing_kept():
    one = bp.tse_2d(**KW)
    assert one.b0_offset is None and one.rf_freq is None and "slice_order" not in one.meta
    multi = bp.tse_2d(**KW, slice_order=ORDER)
    assert abs(float(multi.duration - one.duration)) < 1e-12 and multi.n_adc == one.n_adc
    assert multi.meta["kspace_index"].tolist() == one.meta["kspace_index"].tolist()
    shifted = multi.b0_offset != 0
    assert bool(torch.tensor(multi.is_rf)[shifted].all())  # offsets only during the other slices' RF
    g = multi.grad[shifted, 2]
    ratio = (-multi.b0_offset[shifted] / g).abs()  # = |z_j| of the slice each pulse belongs to
    assert torch.allclose(ratio, torch.full_like(ratio, 0.005))
    with pytest.raises(ValueError, match="exactly one 0.0"):
        bp.tse_2d(**KW, slice_order=[-0.005, 0.005])


@pytest.mark.slow
def test_fast_other_slices_equal_exact_pulses():
    ph = wm_column()
    single = bp.simulate(ph, bp.tse_2d(**KW)).signal
    exact = bp.simulate(ph, bp.tse_2d(**KW, slice_order=ORDER, other_rf_raster=None)).signal
    fast = bp.simulate(ph, bp.tse_2d(**KW, slice_order=ORDER)).signal
    loss = 1 - float(exact.abs().max() / single.abs().max())
    print(f"two neighbours: {loss:.1%} white-matter loss; fast vs exact {rel(fast, exact):.2e}")
    assert loss > 0.05  # the neighbours' pulses do cost signal
    assert rel(fast, exact) < 0.015  # measured 0.8 % with 6 neighbours, TR 1 s (validation/realism)
    # without the bound pool only cross-talk (direct saturation) is left, and it is smaller
    ph0 = wm_column(mt=False)
    loss0 = 1 - float(bp.simulate(ph0, bp.tse_2d(**KW, slice_order=ORDER)).signal.abs().max()
                      / bp.simulate(ph0, bp.tse_2d(**KW)).signal.abs().max())
    assert 0 <= loss0 < loss


def test_gpu_equals_core():
    metal = pytest.importorskip("blochparty.metal")
    if not metal.available():
        pytest.skip("no Apple GPU")
    ph, seq = wm_column(), bp.tse_2d(**KW, slice_order=ORDER)
    ref = bp.simulate(ph, seq).signal
    out = metal.simulate_metal(ph.to(torch.float32), seq.to(torch.float32)).signal.to(ref.dtype)
    assert rel(out, ref) < 1e-4  # measured 3e-6
