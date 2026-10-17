"""The GPU engines: one Python driver over two backends, Metal (``engines.gpu.metal``, Apple GPUs) and CUDA
(``engines.gpu.cuda``, NVIDIA GPUs).

- ``driver.py``: buffers, dispatch, NUFFT readouts, reused pulses (``_forward``); ``adjoint.py``: the
  hand-written reverse sweep (``_GPUSimulate``). ``_simulate`` below picks between them.
- ``shaders/*.metal``: the device code both backends share, Metal Shading Language. Metal compiles it as it is;
  CUDA through ``cuda/translate.py``'s text rules (address spaces dropped, vector types and precise math mapped
  onto ``cuda/prelude.cuh``, scalar math to its float forms).
- ``metal/*.metal.in``, ``cuda/*.cu.in``: each backend's kernels, templated by ``@NAME@``-style placeholders.
  New physics goes into the shared files once; a kernel-side change goes into both backends' kernels; the
  engine matrix (docs/ENGINES.md) checks both against the core.

A backend is an object with ``name``, ``device`` (where buffers live), ``lib()`` (the kernels, called as
``kernel(*buffers, threads=, group_size=)``), ``synchronize()`` and ``max_coils`` (None: no limit); the driver
knows nothing else about it.
"""

from __future__ import annotations

import dataclasses
from importlib.resources import files

import torch
from torch import Tensor

from ...result import SimResult
from ...phantom import Phantom
from ...sequence import Sequence
from .adjoint import _simulate_grad
from .driver import _forward


def source(name: str) -> str:
    """A device-code file of this package (``"shaders/common.metal"``, ``"cuda/prelude.cuh"``, ...) as text."""
    return files(__name__).joinpath(name).read_text(encoding="utf-8")


def _requires_grad(*objs) -> bool:
    """Whether any tensor in ``objs`` (tensors, dataclasses such as Phantom, Sequence, Scanner,
    FieldMaps and their pools, tuples) requires grad."""
    for o in objs:
        if isinstance(o, Tensor):
            if o.requires_grad:
                return True
        elif dataclasses.is_dataclass(o) and not isinstance(o, type):
            if _requires_grad(*(getattr(o, f.name) for f in dataclasses.fields(o))):
                return True
        elif isinstance(o, tuple) and _requires_grad(*o):
            return True
    return False


def _simulate(backend, phantom: Phantom, seq: Sequence, coils, m_init, chunk, group, scanner, mma, seed, maps,
              checkpoint_every, readouts, nufft_width, nufft_terms, pulses) -> SimResult:
    """simulate_metal on ``backend`` (``metal.METAL`` or ``cuda.CUDA``)."""
    if readouts not in ("direct", "nufft"):
        raise ValueError(f"readouts must be 'direct' or 'nufft', not {readouts!r}")
    if pulses not in ("step", "reuse"):
        raise ValueError(f"pulses must be 'step' or 'reuse', not {pulses!r}")
    if torch.is_grad_enabled() and _requires_grad(phantom, seq, coils, m_init, scanner, maps):
        return _simulate_grad(phantom, seq, coils, m_init, scanner, maps, mma, checkpoint_every, backend)
    with torch.no_grad():
        return _forward(phantom, seq, coils, m_init, chunk, group, scanner, mma, seed, maps,
                        nufft=(nufft_width, nufft_terms) if readouts == "nufft" else None, reuse=pulses == "reuse",
                        backend=backend)
