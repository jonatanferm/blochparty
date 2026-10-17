"""What a simulation returns, whichever engine ran it."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class SimResult:
    signal: Tensor  # (C, n_adc) complex, in ADC order
    m: Tensor  # (N, 3) magnetization after the last segment
    m_bound: Tensor | None = None  # (N,) bound-pool Mz after the last segment (with ``Phantom.bound``)
    m_pools: Tensor | None = None  # (N, P, 3) free exchanging pools after the last segment (``Phantom.pools``)
