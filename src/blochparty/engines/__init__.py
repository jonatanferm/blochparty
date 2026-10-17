"""How a simulation is run: ``simulate(..., engine=)`` (``blochparty.simulate``) on the PyTorch core, the chunked core or a GPU engine (Metal, CUDA).

Engines: the core (``core.py``, the PyTorch reference and the default), the core with a spin-chunked backward
(``chunked.py``), and the GPU engines (``engines.gpu``: one driver, Metal on Apple GPUs and CUDA on NVIDIA GPUs).
What each engine supports, and how closely it agrees with the core: ``docs/ENGINES.md``. The core, the chunked
engine, the shared preparation and the capability table move into this package in phase 6 of
``docs/REFACTOR_PLAN.md``.
"""

from __future__ import annotations

from torch import Tensor

from ..fields import Scanner
from ..maps import FieldMaps
from ..phantom import Phantom
from ..result import SimResult
from ..sequence import Sequence
from .chunked import simulate_chunked as _chunked
from .core import simulate as _core

ENGINES = ("core", "chunked", "gpu", "metal", "cuda")
GPU_OPTIONS = ("chunk", "group", "mma", "readouts", "nufft_width", "nufft_terms", "pulses")
CHUNKED_OPTIONS = ("spin_chunk",)


def available(engine: str = "gpu") -> bool:
    """Whether ``engine`` runs here: the core and the chunked engine always; ``"metal"`` with an Apple GPU (MPS);
    ``"cuda"`` with an NVIDIA GPU and NVRTC; ``"gpu"`` with either."""
    if engine in ("core", "chunked"):
        return True
    if engine == "gpu":
        return available("cuda") or available("metal")
    if engine == "metal":
        from .gpu import metal

        return metal.available()
    if engine == "cuda":
        from .gpu import cuda

        return cuda.available()
    raise ValueError(f"engine must be one of {ENGINES}, not {engine!r}")


def resolve(engine: str) -> str:
    """The engine that ``engine`` names here: ``"gpu"`` is ``"cuda"`` where an NVIDIA GPU is, else ``"metal"``
    (RuntimeError when neither is); the others are themselves."""
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, not {engine!r}")
    if engine != "gpu":
        return engine
    for name in ("cuda", "metal"):
        if available(name):
            return name
    raise RuntimeError("engine='gpu': no GPU engine here (Metal needs an Apple GPU with MPS; CUDA an NVIDIA GPU "
                       "with NVRTC)")


def simulate(phantom: Phantom, seq: Sequence, coils: Tensor | None = None, *, m_init: Tensor | None = None,
             checkpoint_every: int | None = None, merge_free_runs: bool = True, scanner: Scanner | None = None,
             seed: int = 0, maps: FieldMaps | None = None, engine: str = "core", **options) -> SimResult:
    """Run ``seq`` on ``phantom``.

    coils: (N, C) complex receive sensitivities at the spin positions (default: one uniform coil).
    m_init: (N, 3) starting magnetization (default: equilibrium, Mz = PD); with a bound pool
        (``Phantom.bound``) also (N, 4), the last column its Mz (default f PD).
    checkpoint_every: the core recomputes the forward pass in chunks of about this many segments during
        backward instead of storing it, trading compute for memory; the GPU engines store the state every this
        many segments for their adjoint (a multiple of 64; default about sqrt(segments)).
    merge_free_runs: the core evaluates stretches of RF-free segments in closed form (fast). False steps segment
        by segment; the results agree to rounding error. The core's option only.
    scanner: hardware field model (``fields.Scanner``: concomitant fields, gradient nonlinearity); None is the
        ideal scanner. ``seq.b0_offset`` (None: zeros) is always applied.
    seed: the random walk of diffusing spins (``Phantom.diffusion``): spin i's path is a function of (seed, i)
        only (``rng``), so engines agree spin by spin.
    maps: scanner-frame B0, B1+ and coil-sensitivity maps (``maps.FieldMaps``) sampled where each spin is (every
        segment for moving spins); mapped coils replace ``coils``.
    engine: ``"core"`` (default) is the PyTorch reference: the phantom's dtype and device, autograd.
        ``"chunked"`` is the core with a backward that needs memory for ``spin_chunk`` spins (default 4096) at a
        time. ``"metal"`` and ``"cuda"`` are the GPU engines, and ``"gpu"`` is CUDA where an NVIDIA GPU is, else
        Metal: float32 (inputs are cast; results come back float32 on the CPU), gradients through a hand-written
        adjoint, and a clear error for what they do not do (no diffusion gradients, no free pools with the
        generalized bound pool, ...: ``docs/ENGINES.md``). ``engines.available(engine)`` says what runs here.
    options: the GPU engines' own keywords (``mma``, ``readouts``, ``nufft_width``, ``nufft_terms``, ``pulses``,
        ``chunk``, ``group``; ``blochparty.metal.simulate_metal`` documents them) or the chunked engine's
        ``spin_chunk``. Another engine's options raise TypeError.

    With ``Phantom.pools`` (free exchanging pools: CEST, myelin water) the core steps every segment by one exact
    matrix exponential of the Bloch-McConnell system instead (slower; checkpoint_every and merge_free_runs do not
    apply), with everything above but the generalized bound pool.
    """
    name = resolve(engine)
    allowed = GPU_OPTIONS if name in ("metal", "cuda") else CHUNKED_OPTIONS if name == "chunked" else ()
    if unknown := sorted(set(options) - set(allowed)):
        others = [e for e, o in (("gpu", GPU_OPTIONS), ("chunked", CHUNKED_OPTIONS)) if set(unknown) <= set(o)]
        hint = f" (an option of engine={others[0]!r})" if others else ""
        raise TypeError(f"simulate(engine={engine!r}) got unexpected keyword argument(s) {', '.join(unknown)}{hint}")
    if name == "core":
        return _core(phantom, seq, coils, m_init=m_init, checkpoint_every=checkpoint_every,
                     merge_free_runs=merge_free_runs, scanner=scanner, seed=seed, maps=maps)
    if not merge_free_runs:
        raise ValueError(f"merge_free_runs=False is the core's segment-by-segment stepping, not engine={engine!r}'s")
    if name == "chunked":
        if seed != 0:
            raise ValueError("engine='chunked' has no random walks (seed): it does not simulate diffusion")
        return _chunked(phantom, seq, coils, m_init=m_init, checkpoint_every=checkpoint_every, scanner=scanner,
                        maps=maps, **options)
    if name == "metal":
        from .gpu.metal import simulate_metal as run
    else:
        from .gpu.cuda import simulate_cuda as run
    return run(phantom, seq, coils, m_init=m_init, checkpoint_every=checkpoint_every, scanner=scanner, seed=seed,
               maps=maps, **options)
