"""Realistic phantoms through every simulator: sub-voxel spins + T2' + fat + B1+ + B0 maps,
fused kernel and ``simulate_chunked`` against the PyTorch core, forward and gradients (to the
spins, the sequence, the coils and back to the maps they were built from)."""

from dataclasses import replace

import pytest
import torch
from test_fused import rel
from test_fused_grad import compare

import blochparty as bp
from blochparty import simulate
from blochparty.chunked import simulate_chunked
from blochparty.coils import birdcage_b1
from blochparty.phantom import b0_shim, b0_sphere

fused = pytest.importorskip("blochparty.fused")


@pytest.fixture(scope="module", autouse=True)
def extension():
    try:
        fused._ext()
    except Exception as e:  # no compiler / ninja: nothing to test
        pytest.skip(f"fused extension unavailable: {e}")


def realistic_maps(n, ndim=2):
    m = bp.shepp_logan_maps(n, ndim=ndim)
    grid = bp.grid_positions(m["pd"].shape, 0.22).reshape(-1, 3)
    shape = m["pd"].shape
    db0 = (b0_shim(grid, z=30.0, y=-60.0, z2=400.0) + b0_sphere(grid, (0.02, 0.03, 0.0), 0.012, 9.4e-6)).reshape(shape)
    return dict(pd=m["pd"], t1=m["t1"], t2=m["t2"], db0=db0,
                t2prime=torch.full(shape, 0.04, dtype=torch.float64),
                fat_fraction=0.1 + 0.8 * (m["label"] == bp.phantom.LABELS.index("fat")).double(),
                b1=birdcage_b1(grid, 0.4, radius=0.1).reshape(shape))


def realistic(n, ndim=2, spv=4, **maps):
    m = realistic_maps(n, ndim) | maps
    return bp.phantom_from_maps(m["pd"], m["t1"], m["t2"], 0.22, m["db0"], spins_per_voxel=spv, jitter="stratified",
                                t2prime=m["t2prime"], fat_fraction=m["fat_fraction"], b1=m["b1"], seed=1)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_forward_2d_and_3d(dtype):
    cd = torch.complex64 if dtype == torch.float32 else torch.complex128
    for ph, seq in [(realistic(16), bp.gre_2d(16, flip_deg=20.0, rf_phase=0.3, n_dummy=30)),
                    (realistic(8, ndim=3, spv=8), bp.gre_3d(8, flip_deg=20.0, n_dummy=10))]:
        assert ph.b1 is not None and ph.db0.abs().max() > 100
        coils = bp.birdcage_sensitivities(ph.pos, n_coils=6)
        out = fused.simulate_fused(ph.to(dtype=dtype), seq.to(dtype=dtype), coils.to(cd))
        ref = simulate(ph.to(dtype=dtype).to(dtype=torch.float64), seq.to(dtype=dtype).to(dtype=torch.float64),
                       coils.to(cd).to(torch.complex128))
        tol = 1e-12 if dtype == torch.float64 else 1e-5
        errs = rel(out.signal.to(torch.complex128), ref.signal), rel(out.m.double(), ref.m)
        assert max(errs) <= tol, errs
        if dtype == torch.float64:  # and the chunked simulator
            ch = simulate_chunked(ph, seq, coils, spin_chunk=97)
            assert rel(ch.signal, ref.signal) <= 1e-12 and rel(ch.m, ref.m) <= 1e-12


def test_spin_and_sequence_gradients():
    # every input, b1 included, against autograd through the PyTorch core (<= 1e-10)
    ph = realistic(8)
    coils = bp.birdcage_sensitivities(ph.pos, n_coils=3)
    seq = bp.gre_2d(8, flip_deg=30.0, rf_phase=0.4, n_dummy=8)
    compare(ph, seq, coils)

    def leaves_with_b1(sim):
        b1 = ph.b1.clone().requires_grad_()
        r = sim(replace(ph, b1=b1), seq, coils)
        return torch.autograd.grad((r.signal.abs() ** 2).sum() + r.m.sum(), b1)[0]

    ref = leaves_with_b1(simulate)
    assert ref.abs().min() > 0
    for sim in (fused.simulate_fused, lambda *a: simulate_chunked(*a, spin_chunk=50)):
        assert rel(leaves_with_b1(sim), ref) <= 1e-10


def test_gradients_reach_the_maps():
    # through phantom_from_maps: PD, T1, T2', fat fraction, fat T2, B1+ and B0 maps
    seq = bp.gre_2d(8, flip_deg=25.0, n_dummy=8)
    base = realistic_maps(8)
    names = ("pd", "t1", "t2prime", "fat_fraction", "b1", "db0")

    def grads(sim):
        leaves = {k: base[k].clone().requires_grad_() for k in names}
        fat_t2 = torch.tensor(0.09, dtype=torch.float64, requires_grad=True)
        m = base | leaves
        ph = bp.phantom_from_maps(m["pd"], m["t1"], m["t2"], 0.22, m["db0"], spins_per_voxel=4, jitter="stratified",
                                  t2prime=m["t2prime"], fat_fraction=m["fat_fraction"], b1=m["b1"], fat_t2=fat_t2)
        r = sim(ph, seq, bp.birdcage_sensitivities(ph.pos, n_coils=2))
        out = torch.autograd.grad((r.signal.abs() ** 2).sum(), [*leaves.values(), fat_t2])
        return dict(zip([*names, "fat_t2"], out))

    ref, out = grads(simulate), grads(fused.simulate_fused)
    mask = base["pd"] > 0
    for k in ref:
        assert ref[k].abs().max() > 0 and rel(out[k], ref[k]) <= 1e-10, k
    fat = (base["fat_fraction"] > 0) & mask
    assert ref["fat_fraction"][fat].abs().min() > 0 and ref["t2prime"][mask].abs().min() > 0
