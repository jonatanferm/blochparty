"""Moving spins: flow, periodic deformation and rigid motion.

A spin's position at time t is

    r(t) = A(t) (r0 + u(t) + v t + d w(t)) + b(t)

with r0 its position at t = 0, u(t) its diffusion walk (if any, in the tissue's own frame),
v its velocity (``velocity``, (N, 3) m/s: flow), d its displacement amplitude (``displacement``,
(N, 3) m) under a scalar waveform w(t) (``waveform``: respiration, pulsation), and a rigid
transform (``rigid``: t -> (A (T, 3, 3), b (T, 3)); head motion). Every segment sees the spin
at its effective position, the formula at the segment's mid-time: exact for the phase of a
constant velocity, second order in the segment length for the rest.

Tissue properties (PD, T1, T2, dB0, coil sensitivities) move with the spin here; scanner-frame
field maps sampled at the moving position are a separate step. Not supported together with
motion: gradient nonlinearity and concomitant fields (their per-spin coefficients depend on
the position).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import Tensor


@dataclass
class Motion:
    velocity: Tensor | None = None  # (N, 3) [m/s]
    displacement: Tensor | None = None  # (N, 3) [m], times waveform(t)
    waveform: Callable[[np.ndarray], np.ndarray] | None = None  # t (T,) [s] -> (T,)
    rigid: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None  # t -> A (T, 3, 3), b (T, 3)

    def __post_init__(self):
        if (self.displacement is None) != (self.waveform is None):
            raise ValueError("displacement and waveform go together")

    def to(self, dtype=None, device=None) -> Motion:
        conv = lambda v: None if v is None else v.to(dtype=dtype, device=device)
        return Motion(conv(self.velocity), conv(self.displacement), self.waveform, self.rigid)

    def per_segment(self, dt: Tensor) -> dict:
        """The per-segment terms at mid-times: t (S,), w (S,) or None, A (S, 3, 3) and b (S, 3) or None."""
        d = dt.detach().double().cpu().numpy()
        t = np.cumsum(d) - 0.5 * d
        out = {"t": torch.as_tensor(t, dtype=dt.dtype, device=dt.device), "w": None, "A": None, "b": None}
        if self.waveform is not None:
            out["w"] = torch.as_tensor(np.asarray(self.waveform(t), dtype=float), dtype=dt.dtype, device=dt.device)
        if self.rigid is not None:
            A, b = self.rigid(t)
            out["A"] = torch.as_tensor(np.array(A, dtype=float), dtype=dt.dtype, device=dt.device)
            out["b"] = torch.as_tensor(np.array(b, dtype=float), dtype=dt.dtype, device=dt.device)
        return out


def positions(base: Tensor, motion: Motion, seg: dict, a: int, b: int) -> Tensor:
    """(N, b - a, 3) effective positions over segments a..b-1 from tissue-frame positions
    ``base`` ((N, 3), or (N, b - a, 3) with a diffusion walk)."""
    if base.dim() == 2:
        base = base[:, None, :].expand(-1, b - a, -1)
    r = base
    if motion.velocity is not None:
        r = r + motion.velocity[:, None, :] * seg["t"][None, a:b, None]
    if motion.displacement is not None:
        r = r + motion.displacement[:, None, :] * seg["w"][None, a:b, None]
    if seg["A"] is not None:
        r = torch.einsum("sij,nsj->nsi", seg["A"][a:b], r) + seg["b"][None, a:b, :]
    return r


def rotation_z(angle: float, centre=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """(A, b) of a rotation by ``angle`` [rad] about z through ``centre``."""
    c, s = np.cos(angle), np.sin(angle)
    A = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    centre = np.asarray(centre, dtype=float)
    return A, centre - A @ centre
