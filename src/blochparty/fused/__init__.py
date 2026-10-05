"""Fused CPU kernel: ``simulate_fused`` is ``blochparty.simulate`` without autograd, fast.

The pure-PyTorch simulator runs ~30 tensor ops over all spins per segment, so it is
memory-bandwidth bound. Spins are independent, so the C++ kernel (``kernel.h``) instead
takes a block of spins, keeps its magnetization in L1 and steps it through every
segment, with blocks spread over threads. The math is that of ``bloch.py``: in float64
the two agree to ~1e-13; in float32 the kernel is closer to the float64 result than
the PyTorch simulator in float32 is (see ``tests/test_fused.py``).

Forward only: inputs are detached and no gradients flow. Use ``blochparty.simulate``
for optimization; this is for producing data. CPU, float32 / float64.

The extension is compiled on first use with ``torch.utils.cpp_extension.load`` (needs
a C++20 compiler, as PyTorch's headers do, and ``ninja``) and cached by PyTorch
(``TORCH_EXTENSIONS_DIR``, default ``~/.cache/torch_extensions``). Extra compiler flags
can be passed in ``BLOCHPARTY_FUSED_CFLAGS``. ``kernel.h`` itself is C++17.
"""

from __future__ import annotations

import functools
import hashlib
import os
import platform
from pathlib import Path

import torch
from torch import Tensor

from ..bloch import SimResult
from ..phantom import Phantom
from ..sequence import Sequence

_HERE = Path(__file__).parent
BLOCK_SIZES = (16, 32, 64, 128, 256)
_RF, _ADC, _SPOIL = 1, 2, 4  # segment flags, as in kernel.h


@functools.cache
def _ext():
    from torch.utils.cpp_extension import load

    # PyTorch only hashes the .cpp for its rebuild check; the define makes kernel.h count too
    digest = hashlib.sha1((_HERE / "kernel.h").read_bytes()).hexdigest()[:12]
    flags = ["-O3", "-fopenmp-simd", f"-DBLOCHPARTY_KERNEL_HASH={digest}"]
    if platform.machine() in ("x86_64", "AMD64"):
        flags.append("-march=native")  # AVX2 / AVX-512 instead of the SSE2 baseline
    flags += os.environ.get("BLOCHPARTY_FUSED_CFLAGS", "").split()
    return load(name="blochparty_fused", sources=[str(_HERE / "bindings.cpp")],
                extra_include_paths=[str(_HERE)], extra_cflags=flags, verbose=False)


def available() -> bool:
    """Whether the extension builds (or is already built) on this machine."""
    try:
        _ext()
        return True
    except Exception:
        return False


def default_block(n_spins: int, n_threads: int, dtype=torch.float32) -> int:
    """Largest block that still fits L1 (1 KB per lane array in float32, 0.5 KB in float64,
    best on an Apple M4) while leaving >= 4 blocks per thread for load balancing."""
    block = 256 if dtype == torch.float32 else 64
    while block > BLOCK_SIZES[0] and -(-n_spins // block) < 4 * n_threads:
        block //= 2
    return block


def simulate_fused(
    phantom: Phantom,
    seq: Sequence,
    coils: Tensor | None = None,
    *,
    m_init: Tensor | None = None,
    n_threads: int | None = None,
    block: int | None = None,
) -> SimResult:
    """Run ``seq`` on ``phantom``: same arguments and result as ``blochparty.simulate``,
    but forward only (inputs are detached; the result has no autograd graph).

    n_threads: worker threads (default: all logical CPUs).
    block: spins per block, one of ``BLOCK_SIZES`` (default: ``default_block``).
    """
    pd = phantom.pd.detach()
    dtype = pd.dtype
    if dtype not in (torch.float32, torch.float64):
        raise TypeError(f"simulate_fused supports float32 and float64, got {dtype}")
    if pd.device.type != "cpu":
        raise ValueError("simulate_fused runs on the CPU")
    n_threads = n_threads or os.cpu_count() or 1
    real = lambda t: t.detach().to(dtype=dtype, device="cpu").contiguous()
    n = len(phantom)
    if coils is None:
        sens = torch.zeros(n, 1, 2, dtype=dtype)
        sens[..., 0] = 1.0
    else:
        coils = coils.detach()
        sens = torch.stack([real(coils.real), real(coils.imag)], dim=-1)
    if m_init is None:
        m = torch.stack([torch.zeros_like(pd), torch.zeros_like(pd), pd], dim=-1)
    else:
        m = real(m_init).clone()
    flag = lambda bits, value: torch.tensor(bits, dtype=torch.uint8).reshape(len(seq)) * value
    flags = flag(seq.is_rf, _RF) | flag(seq.adc, _ADC) | flag(seq.spoil, _SPOIL)
    sig = _ext().simulate(
        real(phantom.pos), real(pd), real(phantom.t1), real(phantom.t2), real(phantom.db0), sens.contiguous(),
        real(seq.dt), real(seq.rf), real(seq.grad), real(seq.adc_phase), flags, m,
        n_threads, block or default_block(n, n_threads, dtype),
    )  # (n_adc, C, 2)
    assert sig.shape[0] == seq.n_adc
    return SimResult(torch.view_as_complex(sig).T, m)


__all__ = ["BLOCK_SIZES", "available", "default_block", "simulate_fused"]
