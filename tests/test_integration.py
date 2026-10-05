"""Features from different modules working together: realistic sequences (ramps, slice
selection, RF spoiling) on realistic phantoms (sub-voxel spins, T2', fat, B1+ maps),
through both simulators, checked by the design toolkit."""

import pytest
import torch

import blochparty as bp
from blochparty import design, fused

N = 12
needs_fused = pytest.mark.skipif(not fused.available(), reason="fused extension unavailable")


@pytest.fixture(scope="module")
def phantom():
    maps = bp.shepp_logan_maps(N)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, N), torch.linspace(-1, 1, N), indexing="ij")
    b1 = (1.0 + 0.2 * (1 - xx**2 - yy**2)).to(torch.float64)
    fat = torch.where(maps["label"] == bp.phantom.LABELS.index("fat"), 0.8, 0.0).to(torch.float64)
    t2p = torch.where(maps["pd"] > 0, 0.05, 1.0).to(torch.float64)
    return bp.phantom_from_maps(maps["pd"], maps["t1"], maps["t2"], 0.22, spins_per_voxel=4,
                                jitter="stratified", t2prime=t2p, b1=b1, fat_fraction=fat)


def realistic_gre(flip=20.0):
    return bp.gre_2d(N, flip_deg=flip, tr=12e-3, n_dummy=4, spoiling="rf", slice_thickness=5e-3,
                     ramps=True)


@needs_fused
def test_fused_matches_core_on_realistic_inputs(phantom):
    flip = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
    b1 = phantom.b1.clone().requires_grad_()
    coils = bp.birdcage_sensitivities(phantom.pos, n_coils=4)
    out = {}
    for name, sim in (("core", bp.simulate), ("fused", fused.simulate_fused)):
        ph = bp.Phantom(phantom.pos, phantom.pd, phantom.t1, phantom.t2, phantom.db0, b1=b1)
        r = sim(ph, realistic_gre(flip), coils)
        out[name] = (r.signal.detach(), *torch.autograd.grad((r.signal.abs() ** 2).sum(), [flip, b1]))
    (s0, gf0, gb0), (s1, gf1, gb1) = out["core"], out["fused"]
    assert (s1 - s0).abs().max() <= 1e-10 * s0.abs().max()
    assert (gf1 - gf0).abs() <= 1e-9 * gf0.abs()
    assert (gb1 - gb0).abs().max() <= 1e-9 * gb0.abs().max()


def test_builder_limits_pass_design_checks():
    # a sequence built within System() limits must not violate the design toolkit's defaults
    seq = realistic_gre()
    report = design.design_report(seq)
    assert not report["violations"], report["violations"]


def test_ideal_gradients_are_reported():
    # instantaneous gradients have unbounded slew; the report must name them
    report = design.design_report(bp.gre_2d(N, n_dummy=2))
    assert any(v.startswith("slew") for v in report["violations"])
