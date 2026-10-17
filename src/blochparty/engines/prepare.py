"""One preparation of a simulation's inputs for every engine.

``prepare`` makes the checks each engine used to make on its own (one message each) and derives what the engines
compute alike from the phantom, the sequence, the coils, the maps and ``m_init``, in the phantom's dtype and on its
device: the field model's per-spin inputs, whether spins move under it, the bound pool per spin and the starting
magnetization. Each engine builds its own state from the result: the core its per-spin tuple, the GPU driver its
float32 buffers (sampling static maps and factorising diffusion tensors in float64 itself, where a float32 phantom
needs the precision), the chunked engine nothing beyond the capability check (``capabilities``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..exchange import BoundPool
from ..fields import Scanner, SpinFields, spin_fields
from ..maps import FieldMaps
from ..phantom import Phantom
from ..sequence import Sequence
from .capabilities import features


@dataclass(frozen=True)
class Prepared:
    """What every engine derives from a case before stepping it (``prepare``)."""

    n: int
    fields: SpinFields  # encoding positions and concomitant coefficients at the true positions
    moving: bool  # the spins diffuse or move
    field_model: tuple | None  # (nonlinearity, f0 or None, z_asymmetry): moving spins under a scanner field model
    start: Tensor  # where walkers and movers start: their true positions under a field model, else encoding ones
    bound: BoundPool | None  # the bound pool, expanded per spin
    generalized: bool  # the bound pool is the generalized Bloch model
    m0: tuple  # (mx, my, mz, mb): equilibrium or m_init; mb is the bound pool's Mz (default f PD), None without one
    features: frozenset  # what the case uses, by capabilities.FEATURES name


def prepare(phantom: Phantom, seq: Sequence, coils: Tensor | None = None, m_init: Tensor | None = None,
            scanner: Scanner | None = None, maps: FieldMaps | None = None) -> Prepared:
    """The checks and shared derivations of a case (module docstring). Raises ValueError for inputs no engine
    accepts: coil sensitivities from both ``coils`` and ``maps``, an ``m_init`` bound-pool column without a bound
    pool, restricted diffusion with a diffusion tensor."""
    pd = phantom.pd
    n = len(phantom)
    if maps is not None and maps.coils is not None and coils is not None:
        raise ValueError("coil sensitivities come from either `coils` or `maps`, not both")
    if phantom.diffusion is not None and phantom.restriction is not None and phantom.diffusion.dim() != 1:
        raise ValueError("restricted diffusion needs an isotropic diffusivity (N,)")
    fields = spin_fields(scanner, phantom.pos)
    bound = phantom.bound.expand(n, pd) if phantom.bound is not None else None
    gen = bound is not None and bound.model == "generalized"
    moving = phantom.diffusion is not None or phantom.motion is not None
    # moving spins under the field model walk and move in true coordinates; every segment maps them to
    # encoding positions and concomitant coefficients. Static spins: spin_fields once.
    field = None
    if moving and scanner is not None and (fields.mode or scanner.nonlinearity is not None):
        f0 = None if not fields.mode else torch.as_tensor(scanner.larmor, dtype=pd.dtype, device=pd.device)
        field = (scanner.nonlinearity, f0, scanner.z_asymmetry)
    start = phantom.pos if field is not None else fields.pos
    mb = None
    if m_init is None:
        mx, my, mz = torch.zeros_like(pd), torch.zeros_like(pd), pd
    elif m_init.shape[-1] == 4:
        mx, my, mz, mb = m_init.unbind(-1)
    else:
        mx, my, mz = m_init.unbind(-1)
    if bound is not None and mb is None:
        mb = bound.f * pd
    elif bound is None and mb is not None:
        raise ValueError("m_init has a bound-pool column but the phantom has no bound pool")
    return Prepared(n, fields, moving, field, start, bound, gen, (mx, my, mz, mb),
                    features(phantom, seq, m_init, scanner, maps))
