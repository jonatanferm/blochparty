"""Multi-slice TSE as one slice sees it (``tse_2d(slice_order=...)``): the other slices' pulses as
their real phase-ramped pulses nearby and as bound-pool-only segments far away
(``bound_only_beyond``) give the same signal as every pulse exact, at a fraction of the segments;
the single-slice sequence is untouched; Metal agrees with the core."""
import pytest
import torch

import blochparty as bp
from blochparty.exchange import BoundPool
from blochparty.pulseq import write_pulseq
from blochparty.sequence import System

f64 = torch.float64
ORDER = [-0.015, -0.005, 0.005, 0.015, 0.0, -0.01, 0.01]  # interleaved, 5 mm apart, this slice 5th
KW = dict(n=16, etl=4, tr=1.0, slice_thickness=5e-3, spoiling="gradient", n_dummy=2, esp=12.5e-3)


def wm_column(n=29, half=3.5e-3, mt=True):
    """White matter (anatomy.DEFAULT_TISSUES at 3 T: bound fraction 0.1277, k 19 /s, T2b 9.7 us) along
    z over this slice and into its neighbours, spread in x and in off-resonance."""
    z = torch.linspace(-half, half, n, dtype=f64)
    pos = torch.stack([torch.linspace(-0.04, 0.04, n, dtype=f64), torch.zeros_like(z), z], -1)
    ph = bp.Phantom(pos, torch.ones(n, dtype=f64), torch.full((n,), 1.084, dtype=f64),
                    torch.full((n,), 0.069, dtype=f64), torch.linspace(-15, 15, n, dtype=f64))
    if mt:
        ph.bound = BoundPool(torch.full((n,), 0.1277 / 0.8723, dtype=f64), torch.full((n,), 19.0, dtype=f64), ph.t1,
                             torch.full((n,), 9.7e-6, dtype=f64), model="rate")
    return ph


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


def test_single_slice_untouched_timing_kept_and_bound_only_on_far_pulses():
    one = bp.tse_2d(**KW)
    assert one.bound_only is None and one.rf_freq is None and "slice_order" not in one.meta
    multi = bp.tse_2d(**KW, slice_order=ORDER, bound_only_beyond=0.0125)
    assert abs(float(multi.duration - one.duration)) < 1e-12 and multi.n_adc == one.n_adc
    assert multi.meta["kspace_index"].tolist() == one.meta["kspace_index"].tolist() and multi.b0_offset is None
    far = torch.tensor(multi.bound_only)
    assert bool(torch.tensor(multi.is_rf)[far].all())
    # one segment per far pulse: 2 far slices x 5 pulses x 6 TRs (2 dummies, 4 shots)
    assert int(far.sum()) == 2 * 5 * 6
    z_j = multi.rf_freq[far] / multi.grad[far, 2]  # where each far pulse sits
    assert torch.allclose(z_j.abs(), torch.full_like(z_j, 0.015))
    with pytest.raises(ValueError, match="exactly one 0.0"):
        bp.tse_2d(**KW, slice_order=[-0.005, 0.005])
    with pytest.raises(ValueError, match="cannot resolve"):  # a 6 kHz ramp on a 100 us raster aliases
        bp.tse_2d(**KW, slice_order=ORDER, system=System(rf_raster=1e-4))
    with pytest.raises(ValueError, match="bound-pool-only"):
        write_pulseq(multi, "/dev/null")
    with pytest.raises(NotImplementedError, match="bound_only"):
        bp.simulate_chunked(wm_column(5), multi)


@pytest.mark.slow
def test_realism_setting_equals_every_pulse_exact():
    """The setting of validation/realism/simulate_heads.py: 16 slices 4.7 mm apart (interleaved),
    TR 6.5 s, ETL 16; all RF on 100 us, the neighbours as their real phase-ramped pulses, the rest
    on the bound pool only; against every pulse exact on 10 us. Spins every 0.05 mm along z, so the spoilers dephase what they would in a voxel (a
    sparse column keeps coherences a voxel would lose, and its error is several times larger)."""
    ph = wm_column(141)
    sp, i = 4.7e-3, 6
    order = [round(j * sp - i * sp, 6) for j in list(range(0, 16, 2)) + list(range(1, 16, 2))]
    kw = dict(n=16, etl=16, esp=12.56e-3, tr=6.5, slice_thickness=5e-3, spoiling="gradient", n_dummy=1)
    single = bp.simulate(ph, bp.tse_2d(**kw, system=System(rf_raster=1e-4))).signal
    exact = bp.simulate(ph, bp.tse_2d(**kw, slice_order=order, system=System(rf_raster=1e-5))).signal
    fast = bp.simulate(ph, bp.tse_2d(**kw, slice_order=order, system=System(rf_raster=1e-4),
                                     bound_only_beyond=1.5 * sp)).signal
    loss = 1 - float(exact.abs().max() / single.abs().max())
    print(f"16 slices: {loss:.1%} white-matter loss; realism setting vs exact {rel(fast, exact):.1e}")
    assert loss > 0.2  # the other slices' pulses cost signal (MT and cross-talk)
    assert rel(fast, exact) < 0.015  # measured 0.71 % (n = 16)


def test_gpu_equals_core():
    metal = pytest.importorskip("blochparty.metal")
    if not metal.available():
        pytest.skip("no Apple GPU")
    for ph in (wm_column(11), wm_column(11, mt=False)):  # without a bound pool the far pulses are free segments
        seq = bp.tse_2d(**KW, slice_order=ORDER, bound_only_beyond=0.0075, system=System(rf_raster=1e-5))
        ref = bp.simulate(ph, seq)
        out = metal.simulate_metal(ph.to(torch.float32), seq.to(torch.float32))
        assert rel(out.signal.to(ref.signal.dtype), ref.signal) < 1e-4  # no frame offset: float32's level
        assert rel(out.m.double(), ref.m) < 1e-4
