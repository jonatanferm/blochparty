"""Isochromat Bloch simulator.

Each spin's magnetization (mx, my, mz) is advanced segment by segment:

* free precession (no RF): exact rotation about z and exact relaxation
  (they commute, so this is exact for any segment length);
* RF segments: exact rotation about the effective field, with relaxation split
  symmetrically around it (Strang splitting, second order in dt).

Conventions: dM/dt = gamma * M x B, so with fields in Hz the rotation vector of
a segment is w = -2*pi*dt*(rf_x, rf_y, G.r + dB0) and the transverse
magnetization Mxy = Mx + i*My evolves as exp(-i*2*pi*f*t) at offset f.

Everything is plain tensor arithmetic, so autograd reaches the sequence
(flip angles, timings, gradients), the phantom (PD, T1, T2, dB0) and the coils.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .phantom import Phantom
from .sequence import Sequence

TWO_PI = 2.0 * math.pi


class _Spins(NamedTuple):
    pos: Tensor
    db0: Tensor
    pd: Tensor
    r1: Tensor
    r2: Tensor
    sens_re: Tensor  # (N, C)
    sens_im: Tensor


@dataclass
class SimResult:
    signal: Tensor  # (C, n_adc) complex, in ADC order
    m: Tensor  # (N, 3) magnetization after the last segment


def _relax(mx, my, mz, dt, spins: _Spins):
    e2 = torch.exp(-dt * spins.r2)
    e1 = torch.exp(-dt * spins.r1)
    return mx * e2, my * e2, spins.pd + (mz - spins.pd) * e1


def _precess(mx, my, phi):
    c, s = torch.cos(phi), torch.sin(phi)
    return c * mx - s * my, s * mx + c * my


def _rotate(mx, my, mz, wx, wy, wz):
    """Right-handed rotation by the rotation vector w (angle |w| in rad), Rodrigues form."""
    phi = torch.sqrt(wx * wx + wy * wy + wz * wz + 1e-30)  # offset keeps the gradient finite at w = 0
    c = torch.cos(phi)
    s1 = torch.sinc(phi / math.pi)  # sin(phi) / phi
    s2 = 0.5 * torch.sinc(phi / TWO_PI) ** 2  # (1 - cos(phi)) / phi^2
    dot = wx * mx + wy * my + wz * mz
    return (
        c * mx + s1 * (wy * mz - wz * my) + s2 * wx * dot,
        c * my + s1 * (wz * mx - wx * mz) + s2 * wy * dot,
        c * mz + s1 * (wx * my - wy * mx) + s2 * wz * dot,
    )


def _run(mx, my, mz, segments, spins: _Spins):
    samples = []
    for dt, rf, g, psi, is_rf, adc, spoil in segments:
        bz = spins.pos @ g + spins.db0  # off-resonance in the rotating frame [Hz]
        if is_rf:
            mx, my, mz = _relax(mx, my, mz, 0.5 * dt, spins)
            w = -TWO_PI * dt
            mx, my, mz = _rotate(mx, my, mz, w * rf[0], w * rf[1], w * bz)
            mx, my, mz = _relax(mx, my, mz, 0.5 * dt, spins)
        else:
            mx, my = _precess(mx, my, -TWO_PI * dt * bz)
            mx, my, mz = _relax(mx, my, mz, dt, spins)
        if adc:
            re = mx @ spins.sens_re - my @ spins.sens_im
            im = mx @ spins.sens_im + my @ spins.sens_re
            c, s = torch.cos(psi), torch.sin(psi)  # demodulate by exp(-i*psi)
            samples.append(torch.stack([re * c + im * s, im * c - re * s]))
        if spoil:
            mx, my = torch.zeros_like(mx), torch.zeros_like(my)
    n_coils = spins.sens_re.shape[1]
    sig = torch.stack(samples) if samples else mx.new_zeros(0, 2, n_coils)
    return mx, my, mz, sig


def simulate(
    phantom: Phantom,
    seq: Sequence,
    coils: Tensor | None = None,
    *,
    m_init: Tensor | None = None,
    checkpoint_every: int | None = None,
) -> SimResult:
    """Run ``seq`` on ``phantom``.

    coils: (N, C) complex receive sensitivities at the spin positions (default:
        one uniform coil).
    m_init: (N, 3) starting magnetization (default: equilibrium, Mz = PD).
    checkpoint_every: recompute the forward pass in chunks of this many segments
        during backward instead of storing it, trading compute for memory.
    """
    pd = phantom.pd
    seq = seq.to(dtype=pd.dtype, device=pd.device)
    n = len(phantom)
    if coils is None:
        sens_re, sens_im = torch.ones_like(pd)[:, None], torch.zeros_like(pd)[:, None]
    else:
        sens_re, sens_im = coils.real.to(pd.dtype), coils.imag.to(pd.dtype)
    spins = _Spins(phantom.pos, phantom.db0, pd, 1.0 / phantom.t1, 1.0 / phantom.t2, sens_re, sens_im)

    if m_init is None:
        mx, my, mz = torch.zeros_like(pd), torch.zeros_like(pd), pd
    else:
        mx, my, mz = m_init.unbind(-1)

    segments = list(zip(seq.dt.unbind(0), seq.rf.unbind(0), seq.grad.unbind(0),
                        seq.adc_phase.unbind(0), seq.is_rf, seq.adc, seq.spoil))
    chunk = checkpoint_every or max(len(segments), 1)
    use_ckpt = checkpoint_every is not None and torch.is_grad_enabled()
    sigs = [pd.new_zeros(0, 2, sens_re.shape[1])]
    for start in range(0, len(segments), chunk):
        part = segments[start : start + chunk]
        if use_ckpt:
            mx, my, mz, sig = checkpoint(_run, mx, my, mz, part, spins, use_reentrant=False)
        else:
            mx, my, mz, sig = _run(mx, my, mz, part, spins)
        sigs.append(sig)
    sig = torch.cat(sigs)  # (n_adc, 2, C)
    assert sig.shape[0] == seq.n_adc and n == mx.shape[0]
    return SimResult(torch.complex(sig[:, 0], sig[:, 1]).T, torch.stack([mx, my, mz], dim=-1))
