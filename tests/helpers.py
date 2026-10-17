"""Small helpers shared by the test modules (formerly in test_fused.py)."""

from dataclasses import replace

import torch

import blochparty as bp
from blochparty import Sequence, SequenceBuilder


def rel(a, b) -> float:
    if b.numel() == 0:
        return 0.0
    scale = b.abs().max()
    return float((a - b).abs().max() / scale) if scale > 0 else float((a - b).abs().max())


def with_readout(seq: Sequence, n=16) -> Sequence:
    """Append a readout (gradient, receiver phase) so a sequence also produces signal."""
    ro = SequenceBuilder().readout(n, 10e-6, grad=(800.0, -300.0, 150.0), phase=0.7).build()
    cat = lambda f: torch.cat([getattr(seq, f), getattr(ro, f)])
    return replace(seq, dt=cat("dt"), rf=cat("rf"), grad=cat("grad"), adc_phase=cat("adc_phase"),
                   is_rf=seq.is_rf + ro.is_rf, adc=seq.adc + ro.adc, spoil=seq.spoil + ro.spoil)


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
