"""Sub-voxel isochromats (``phantom_from_maps(..., spins_per_voxel=K, jitter=...)``): layout,
k-space against the signal equation for the same spins, and convergence toward the
extended phase graph (which integrates a voxel exactly) under gradient spoiling."""

import math

import pytest
import torch

import blochparty as bp
from blochparty import SequenceBuilder
from blochparty.epg import epg_spoiled_gre
from blochparty.reference import signal_equation

f64 = lambda v: torch.tensor(v, dtype=torch.float64)
one = lambda v: torch.full((1, 1), v, dtype=torch.float64)


def maps(n=8):
    m = bp.shepp_logan_maps(n)
    return m["pd"], m["t1"], m["t2"]


def test_default_is_one_spin_per_voxel_centre():
    pd, t1, t2 = maps()
    a = bp.phantom_from_maps(pd, t1, t2, 0.2)
    b = bp.phantom_from_maps(pd, t1, t2, 0.2, spins_per_voxel=1, jitter=None, seed=5)
    mask = pd > 0
    assert torch.equal(a.pos, bp.grid_positions(8, 0.2)[mask]) and torch.equal(a.pd, pd[mask])
    for f in ("pos", "pd", "t1", "t2", "db0"):
        assert torch.equal(getattr(a, f), getattr(b, f)), f
    assert a.b1 is None


@pytest.mark.parametrize("ndim, spv, counts", [(2, 9, (3, 3)), (2, (2, 5), (2, 5)), (3, 8, (2, 2, 2)),
                                               (3, (1, 1, 4), (1, 1, 4))])
@pytest.mark.parametrize("jitter", [None, "stratified"])
def test_layout_and_pd(ndim, spv, counts, jitter):
    n, fov = 6, 0.12
    m = bp.shepp_logan_maps(n, ndim=ndim)
    ph = bp.phantom_from_maps(m["pd"], m["t1"], m["t2"], fov, spins_per_voxel=spv, jitter=jitter, seed=3)
    k, mask = math.prod(counts), m["pd"] > 0
    n_vox = int(mask.sum())
    assert len(ph) == n_vox * k
    centres = bp.grid_positions(m["pd"].shape, fov)[mask]
    vox = fov / n
    # spins of voxel v are v*k ... v*k + k - 1, carry PD / K and stay inside their voxel
    frac = ((ph.pos.reshape(n_vox, k, 3) - centres[:, None]) / vox)[..., :ndim]  # (x, y[, z]) in voxels
    assert frac.abs().max() <= 0.5
    if ndim == 2:
        assert torch.equal(ph.pos[:, 2], torch.zeros(len(ph), dtype=torch.float64))
    assert torch.allclose(ph.pd.reshape(n_vox, k).sum(1), m["pd"][mask], rtol=1e-15)
    assert torch.equal(ph.t1.reshape(n_vox, k), m["t1"][mask][:, None].expand(n_vox, k))
    # exactly one spin per sub-cell; at its centre without jitter
    sides = torch.tensor(counts[::-1], dtype=torch.float64)  # (x, y[, z])
    cell = (frac + 0.5) * sides
    idx = cell.floor()
    flat = (idx * torch.cumprod(torch.cat([torch.ones(1, dtype=torch.float64), sides[:-1]]), 0)).sum(-1)
    assert torch.equal(flat.sort(dim=1).values, torch.arange(k, dtype=torch.float64).expand(n_vox, k))
    if jitter is None:
        assert torch.allclose(cell - idx, torch.full_like(cell, 0.5), atol=1e-12)
    else:  # uniform within the sub-cell: mean 1/2, variance 1/12
        u = cell - idx
        assert abs(u.mean() - 0.5) < 0.05 and abs(u.var() - 1 / 12) < 0.02


def test_seed_and_errors():
    pd, t1, t2 = maps()
    run = lambda seed: bp.phantom_from_maps(pd, t1, t2, 0.2, spins_per_voxel=4, jitter="stratified", seed=seed)
    assert torch.equal(run(1).pos, run(1).pos) and not torch.equal(run(1).pos, run(2).pos)
    with pytest.raises(ValueError, match="perfect"):
        bp.phantom_from_maps(pd, t1, t2, 0.2, spins_per_voxel=8)
    with pytest.raises(ValueError, match="positive counts"):
        bp.phantom_from_maps(pd, t1, t2, 0.2, spins_per_voxel=(2, 2, 2))
    with pytest.raises(ValueError, match="jitter"):
        bp.phantom_from_maps(pd, t1, t2, 0.2, spins_per_voxel=4, jitter="random")


@pytest.mark.parametrize("sim", ["core", "fused"])
def test_kspace_matches_signal_equation(sim):
    # 2D GRE (ideal spoiling), 3x3 jittered spins per voxel, 4 coils evaluated at the spins
    n, fov = 16, 0.22
    pd, t1, t2 = maps(n)
    ph = bp.phantom_from_maps(pd, t1, t2, fov, spins_per_voxel=9, jitter="stratified", seed=0)
    coils = bp.birdcage_sensitivities(ph.pos, n_coils=4)
    seq = bp.gre_2d(n, fov, flip_deg=20.0, dwell=1e-6, rf_duration=1e-6, n_dummy=50)
    if sim == "fused":
        fused = pytest.importorskip("blochparty.fused")
        out = fused.simulate_fused(ph, seq, coils).signal
    else:
        out = bp.simulate(ph, seq, coils).signal
    ref = signal_equation(ph, seq, coils)
    err = float((out - ref).abs().max() / ref.abs().max())
    # no gradient or off-resonance during the pulses, so the Strang-split pulse (relax, rotate,
    # relax) is exactly the signal equation's instantaneous rotation at the pulse centre
    assert err < 1e-12, err
    # and it is not the one-spin-per-voxel k-space: the sub-voxel spins change the signal
    centre = signal_equation(bp.phantom_from_maps(pd, t1, t2, fov), seq,
                             bp.birdcage_sensitivities(bp.phantom_from_maps(pd, t1, t2, fov).pos, n_coils=4))
    assert float((centre - ref).abs().max() / ref.abs().max()) > 1e-2


# --- gradient spoiling: K isochromats per voxel vs. EPG -----------------------------------
L = 1e-3  # voxel length [m]


def gradient_spoiled_gre(tr, te, flip, n, cycles=1):
    """Hard pulses, echo at TE, then a spoiler that winds ``cycles`` turns of phase across
    the voxel (EPG order +cycles) every TR; no RF spoiling, no ideal spoiling."""
    b = SequenceBuilder()
    rest = tr - te - 1e-6
    for _ in range(n):
        b.rf(1e-6, flip_deg=flip)
        b.delay(te - 1e-6).readout(1, 1e-6)
        b.delay(rest, grad=(cycles / (L * rest), 0.0, 0.0))
    return b.build()


def test_gradient_spoiling_converges_to_epg():
    t1, t2, tr, te, flip, n = 0.5, 0.05, 10e-3, 3e-3, 30.0, 150
    epg = epg_spoiled_gre(t1, t2, tr, te, flip, n, None)  # pulse centre at t=0, echo at te
    seq = gradient_spoiled_gre(tr, te, flip, n)

    def err(k, jitter, seed=0):
        ph = bp.phantom_from_maps(one(1.0), one(t1), one(t2), L, spins_per_voxel=(1, k), jitter=jitter, seed=seed)
        return float((bp.simulate(ph, seq).signal[0] - epg).abs().max())

    # the spoiled steady state is far from the ideally spoiled (Ernst) one: this tests real dephasing
    ernst = epg_spoiled_gre(t1, t2, tr, te, flip, n, None, ideal_spoil=True)
    assert abs(abs(epg[-1]) / abs(ernst[-1]) - 1) > 0.1
    # Regular sub-grid: K spins alias EPG order j onto j mod K, so the error falls as fast
    # as the population of orders >= K, and is round-off once K exceeds every order
    # reached (one per TR, n = 150).
    regular = {k: err(k, None) for k in (2, 8, 32, 64, 128, 151)}
    assert regular[2] > 0.1 and regular[2] > regular[8] > regular[32] > regular[64] > regular[128]
    assert regular[151] < 1e-13
    # Stratified jitter: no exact aliasing structure, so the error falls steadily
    # instead (~K^-1.3 measured; mean over 3 seeds)
    jit = {k: sum(err(k, "stratified", s) for s in range(3)) / 3 for k in (4, 16, 64, 256)}
    assert jit[4] > 2.5 * jit[16] > 2.5**2 * jit[64] > 2.5**3 * jit[256]
    assert jit[256] < 1e-3


def test_maps_stay_differentiable():
    pd, t1, t2 = maps()
    t1 = t1.clone().requires_grad_()
    ph = bp.phantom_from_maps(pd, t1, t2, 0.2, spins_per_voxel=4, jitter="stratified")
    seq = bp.gre_2d(8, 0.2, n_dummy=5)
    bp.simulate(ph, seq).signal.abs().sum().backward()
    mask = pd > 0
    assert t1.grad[mask].abs().min() > 0 and t1.grad[~mask].abs().max() == 0
