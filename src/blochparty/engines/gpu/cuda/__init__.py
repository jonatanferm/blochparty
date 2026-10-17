"""Bloch simulation on NVIDIA GPUs (CUDA), forward and adjoint: the Metal engine ported.

The physics is the Metal engine's (``blochparty.metal``), not a copy of it: the device functions of its shader
(``engines/gpu/shaders/*.metal``: steps, relaxation, exchange, lineshapes, walks, the field model, the reverse
sweep) are turned into CUDA C++ by a few textual rules (address-space qualifiers dropped, Metal's vector types and
precise math mapped onto a small compatibility layer, ``prelude.cuh``) and compiled as they are. The kernels are
written for CUDA (``*.cu.in``), because what differs between the two GPUs lives in them: a warp is a simdgroup of 32,
``__shfl_xor_sync`` sums replace ``simd_sum``, shared memory and ``atomicAdd`` replace threadgroup memory and Metal's
atomics. The coil sum of an ADC sample replaces Metal's 8 x 8 simdgroup matrix products (``bloch_chunk_mma``) by a
reduce-scatter across the warp: each lane forms its spin's products with eight coils (16 values), and five
butterfly steps (16 shuffles) leave every coil's sum in one lane, in full float32 (tensor cores would round the
products to TF32). Everything else, the per-spin state, the two-sum low parts, the NUFFT readouts, reused pulses,
the checkpointed adjoint with its in-order segment reduction, follows the Metal kernels line by line.

The Python driver is the one Metal uses (``engines.gpu.driver``, ``engines.gpu.adjoint``) with this module's
backend: the kernels take the same buffers in the same order, so the two engines cannot drift apart in what they
compute from a phantom and a sequence. ``simulate_cuda`` has ``simulate_metal``'s arguments and results (float32, on
the CPU).

Kernels are compiled at run time by NVRTC (the ``nvidia-cuda-nvrtc`` wheel PyTorch depends on, or a CUDA
toolkit), one kernel per program on its first use, for the current device's architecture, and cached on disk
(``BLOCHPARTY_CUDA_CACHE``, default ``~/.cache/blochparty/cuda``).

Here: the backend object and ``simulate_cuda``; ``translate.py`` builds the programs, ``rtc.py`` compiles and
launches them. ``blochparty.cuda`` is this module.
"""

from __future__ import annotations

import torch

from ....bloch import SimResult
from ....phantom import Phantom
from ....sequence import Sequence
from .. import _simulate
from .rtc import _Driver
from .translate import kernel_source

MAX_COILS = 480  # the block buffer holds at least one sample of 2 x coils floats per warp (TG_FLOATS / MAX_SG)

_DRIVER: _Driver | None = None


def _driver() -> _Driver:
    global _DRIVER
    if _DRIVER is None:
        _DRIVER = _Driver()
    return _DRIVER


class _Kernels:
    """metal._lib()'s interface: ``lib.<kernel name>(*buffers, threads=, group_size=)``."""

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        kernel_source(name)  # raises AttributeError for names that are no kernel

        def launch(*args, threads: int, group_size: int = 256):
            _driver().launch(name, args, threads, group_size)

        return launch


class _Cuda:
    """The driver's backend (``metal._Metal``'s interface) on the current CUDA device."""

    name = "cuda"
    max_coils = MAX_COILS

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", torch.cuda.current_device())

    def lib(self) -> _Kernels:
        return _Kernels()

    def synchronize(self) -> None:
        torch.cuda.synchronize()

CUDA = _Cuda()


def available() -> bool:
    """A CUDA device, NVRTC and the driver library are there (kernels compile on first use)."""
    try:
        return torch.cuda.is_available() and _driver() is not None
    except Exception:
        return False


def simulate_cuda(phantom: Phantom, seq: Sequence, coils: torch.Tensor | None = None, *, m_init: torch.Tensor | None = None,
                  chunk: int = 32768, group: int = 256, scanner=None, mma: bool | None = None, seed: int = 0, maps=None,
                  checkpoint_every: int | None = None, readouts: str = "direct", nufft_width: int = 6,
                  nufft_terms: int = 3, pulses: str = "step") -> SimResult:
    """Forward simulation on an NVIDIA GPU, with ``simulate_metal``'s arguments, results (float32, on the CPU),
    gradients (the same hand-written adjoint) and limits; see its docstring. ``mma`` picks the sensitivity layout
    and kernel as on Metal (``bloch_chunk_mma``: any coil count up to ``MAX_COILS``, NUFFT readouts, reused
    pulses; ``bloch_chunk``: at most 32 coils); both reduce the coils by warp shuffles here."""
    return _simulate(CUDA, phantom, seq, coils, m_init, chunk, group, scanner, mma, seed, maps, checkpoint_every,
                     readouts, nufft_width, nufft_terms, pulses)
