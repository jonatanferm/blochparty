"""blochparty.simulate(..., engine=): each name reaches its engine with every argument (each argument is chosen so
that dropping it changes the result), one engine's options are refused by the others, and "gpu" is the GPU here."""

from dataclasses import replace

import pytest
import torch

import blochparty as bp
from blochparty import engines
from blochparty.bloch import simulate as core
from blochparty.chunked import simulate_chunked
from blochparty.fields import Scanner

from cases import random_phantom
from helpers import rel

f64 = torch.float64


def case():
    """Every argument that the entry point forwards, each with a visible effect on the signal."""
    ph = random_phantom(40, seed=3)
    g = torch.Generator().manual_seed(7)
    n = len(ph)
    coils = torch.complex(torch.rand(n, 3, generator=g, dtype=f64), torch.rand(n, 3, generator=g, dtype=f64))
    m_init = torch.rand(n, 3, generator=g, dtype=f64) * ph.pd[:, None]
    scanner = Scanner(field_strength=0.05)  # low field: large concomitant fields
    seq = bp.gre_2d(8, n_dummy=0, dtype=f64)
    return ph, seq, dict(coils=coils, m_init=m_init, scanner=scanner)


def test_core_is_the_core_with_every_argument():
    ph, seq, kw = case()
    walk = replace(ph, diffusion=torch.full((len(ph),), 2e-6, dtype=f64))
    ref = core(walk, seq, **kw, seed=5, merge_free_runs=False)
    out = bp.simulate(walk, seq, **kw, seed=5, merge_free_runs=False, engine="core")
    assert torch.equal(out.signal, ref.signal) and torch.equal(out.m, ref.m)
    assert torch.equal(bp.simulate(walk, seq, **kw, seed=5).signal, core(walk, seq, **kw, seed=5).signal)
    for name, changed in (("coils", dict(kw, coils=None)), ("m_init", dict(kw, m_init=None)),
                          ("scanner", dict(kw, scanner=None))):
        assert rel(core(walk, seq, **changed, seed=5).signal[:, :1], ref.signal[:, :1]) > 1e-3, name
    assert rel(core(walk, seq, **kw, seed=6).signal, ref.signal) > 1e-6, "seed"


def test_chunked_is_the_chunked_core():
    ph, seq, _ = case()
    coils = case()[2]["coils"]
    ref = simulate_chunked(ph, seq, coils, spin_chunk=7)
    out = bp.simulate(ph, seq, coils, engine="chunked", spin_chunk=7)
    assert torch.equal(out.signal, ref.signal) and torch.equal(out.m, ref.m)


def test_gpu_engine_by_name(gpu):
    ph, seq, kw = case()
    p32, s32 = ph.to(torch.float32), seq.to(torch.float32)
    kw32 = dict(kw, coils=kw["coils"].to(torch.complex64), m_init=kw["m_init"].float())
    ref = gpu.simulate(p32, s32, **kw32, mma=False)
    out = bp.simulate(p32, s32, **kw32, engine=gpu.name, mma=False)
    assert rel(out.signal, ref.signal) < 1e-6 and rel(out.m, ref.m) < 1e-6  # float atomics: not bit for bit
    assert out.signal.dtype == torch.complex64
    for name, changed in (("coils", dict(kw32, coils=None)), ("m_init", dict(kw32, m_init=None)),
                          ("scanner", dict(kw32, scanner=None))):
        assert rel(gpu.simulate(p32, s32, **changed, mma=False).signal[:, :1], ref.signal[:, :1]) > 1e-3, name
    core_ref = core(ph, seq, **kw)
    assert rel(out.signal.to(core_ref.signal.dtype), core_ref.signal) < 1e-4


def test_gpu_is_cuda_where_there_is_one_else_metal():
    if not engines.available("gpu"):
        with pytest.raises(RuntimeError, match="no GPU engine"):
            engines.resolve("gpu")
        return
    assert engines.resolve("gpu") == ("cuda" if engines.available("cuda") else "metal")


def test_options_belong_to_their_engine():
    ph, seq, _ = case()
    with pytest.raises(TypeError, match=r"readouts \(an option of engine='gpu'\)"):
        bp.simulate(ph, seq, readouts="nufft")
    with pytest.raises(TypeError, match=r"spin_chunk \(an option of engine='chunked'\)"):
        bp.simulate(ph, seq, spin_chunk=8)
    with pytest.raises(TypeError, match="colis"):
        bp.simulate(ph, seq, colis=None)
    with pytest.raises(ValueError, match="engine must be one of"):
        bp.simulate(ph, seq, engine="fused")
    with pytest.raises(ValueError, match="merge_free_runs"):
        bp.simulate(ph, seq, engine="chunked", merge_free_runs=False)
    with pytest.raises(ValueError, match="random walks"):
        bp.simulate(ph, seq, engine="chunked", seed=3)
    with pytest.raises(ValueError, match="engine must be one of"):
        engines.available("fused")
    assert engines.available("core") and engines.available("chunked")
