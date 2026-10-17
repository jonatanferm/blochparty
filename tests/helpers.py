"""Small helpers shared by the test modules (formerly in test_fused.py)."""

from dataclasses import replace

import torch

import blochparty as bp
from blochparty import Sequence, SequenceBuilder


def rel(a, b, scale=None) -> float:
    """max |a - b| / scale (default max |b|), ``a`` cast to ``b``'s dtype first (a float32 engine against the float64
    core); 0 for empty tensors, the plain maximum difference when the scale is zero. The suite's one definition."""
    if b.numel() == 0:
        return 0.0
    if a.dtype != b.dtype:
        a = a.to(b.dtype)
    diff = (a - b).abs().max()
    scale = b.abs().max() if scale is None else scale
    return float(diff / scale) if scale > 0 else float(diff)


def with_readout(seq: Sequence, n=16) -> Sequence:
    """Append a readout (gradient, receiver phase) so a sequence also produces signal."""
    ro = SequenceBuilder().readout(n, 10e-6, grad=(800.0, -300.0, 150.0), phase=0.7).build()
    cat = lambda f: torch.cat([getattr(seq, f), getattr(ro, f)])
    return replace(
        seq,
        dt=cat("dt"),
        rf=cat("rf"),
        grad=cat("grad"),
        adc_phase=cat("adc_phase"),
        is_rf=seq.is_rf + ro.is_rf,
        adc=seq.adc + ro.adc,
        spoil=seq.spoil + ro.spoil,
    )


def head_3d(n=12, n_coils=4):
    phantom = bp.shepp_logan(n, ndim=3)
    g = torch.Generator().manual_seed(1)
    phantom = replace(phantom, db0=(torch.rand(len(phantom), generator=g, dtype=torch.float64) - 0.5) * 100)
    return phantom, bp.birdcage_sensitivities(phantom.pos, n_coils=n_coils)


def gpu_engines():
    """pytest params "metal" (Apple GPU) and "cuda" (NVIDIA), each skipped where it is not available: the
    GPU engines share their Python driver, so their tests are one set run on each."""
    import importlib

    import pytest

    params = []
    for name, reason in (("metal", "no Apple GPU (MPS)"), ("cuda", "no CUDA device with NVRTC")):
        try:
            ok = importlib.import_module(f"blochparty.{name}").available()
        except Exception:
            ok = False
        params.append(pytest.param(name, marks=pytest.mark.skipif(not ok, reason=reason), id=name))
    return params


class GPU:
    """A GPU engine by name: ``simulate`` is ``simulate_metal`` or ``simulate_cuda``."""

    def __init__(self, name: str):
        import importlib

        self.name = name
        self.module = importlib.import_module(f"blochparty.{name}")
        self.simulate = getattr(self.module, f"simulate_{name}")


# ---- gradients of a GPU engine against the core's autograd (test_gpu_grad.py, test_phantom_engines.py)


def weighted_loss(r, which="both"):
    """Linear in the outputs with fixed random weights (float64, so float32 engines get the same)."""
    g = torch.Generator().manual_seed(1)
    w = lambda x: torch.randn(x.shape, generator=g, dtype=torch.float64)
    total = 0
    if which in ("both", "signal"):
        total = (
            total
            + (r.signal.real.double() * w(r.signal.real)).sum()
            + (r.signal.imag.double() * w(r.signal.imag)).sum()
        )
    if which in ("both", "m"):
        total = total + (r.m.double() * w(r.m)).sum()
        if r.m_bound is not None:
            total = total + (r.m_bound.double() * w(r.m_bound)).sum()
    return total


def input_gradients(sim, phantom, seq, coils=None, m_init=None, scanner=None, which="both", **kw):
    """Gradients of ``weighted_loss`` w.r.t. every input tensor (phantom, bound pool, sequence, coils, m_init)."""
    leaves = {}

    def leaf(name, t):
        t = t.detach().clone().requires_grad_()
        leaves[name] = t
        return t

    ph = replace(phantom, **{k: leaf(k, getattr(phantom, k)) for k in ("pos", "pd", "t1", "t2", "db0")})
    if phantom.b1 is not None:
        ph = replace(ph, b1=leaf("b1", phantom.b1))
    if phantom.bound is not None:
        b = phantom.bound.expand(len(phantom), phantom.pd)
        ph = replace(
            ph, bound=replace(b, **{k: leaf("bound." + k, getattr(b, k)) for k in ("f", "k", "t1", "t2", "dw")})
        )
    sq = replace(seq, **{k: leaf(k, getattr(seq, k)) for k in ("dt", "rf", "grad", "adc_phase")})
    if seq.b0_offset is not None:
        sq = replace(sq, b0_offset=leaf("b0_offset", seq.b0_offset))
    c = None if coils is None else leaf("coils", coils)
    m0 = None if m_init is None else leaf("m_init", m_init)
    r = sim(ph, sq, c, m_init=m0, scanner=scanner, **kw)
    grads = torch.autograd.grad(weighted_loss(r, which), list(leaves.values()), allow_unused=True)
    return {k: g for k, g in zip(leaves, grads)}


def compare_gradients(gpu, phantom, seq, tol, **kw):
    """A GPU engine's gradients of ``weighted_loss`` against the float64 core's autograd, every input; the
    largest relative error must stay below ``tol``."""
    ref, out = input_gradients(bp.simulate, phantom, seq, **kw), input_gradients(gpu.simulate, phantom, seq, **kw)
    errors = {k: rel(out[k], ref[k]) for k in ref if ref[k] is not None and ref[k].abs().max() > 0}
    assert max(errors.values()) < tol, errors
    return errors
