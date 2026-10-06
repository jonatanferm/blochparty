"""Isochromat Bloch simulator.

Each spin's magnetization (mx, my, mz) is advanced segment by segment:

* free precession (no RF): exact rotation about z and exact relaxation
  (they commute, so this is exact for any segment length). Consecutive RF-free
  segments are evaluated together in closed form, ADC samples included;
* RF segments: exact rotation about the effective field, with relaxation split
  symmetrically around it (Strang splitting, second order in dt).

Conventions: dM/dt = gamma * M x B, so with fields in Hz the rotation vector of
a segment is w = -2*pi*dt*(rf_x, rf_y, G.r + dB0) and the transverse
magnetization Mxy = Mx + i*My evolves as exp(-i*2*pi*f*t) at offset f.

Field model (``fields.py``): the off-resonance of a segment is
G.r + dB0 + b0_offset + f_c(G), with r the encoding positions (gradient nonlinearity)
and f_c the concomitant field; a free run's phase then needs the cumulative moments
of G, of b0_offset and of the products G_i G_j (leading order), or the per-segment
|B| (exact mode).

Everything is plain tensor arithmetic, so autograd reaches the sequence
(flip angles, timings, gradients), the phantom (PD, T1, T2, dB0, B1+) and the coils.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .fields import Scanner, exact_excess, gradient_monomials, spin_fields
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
    b1: Tensor | None  # (N,) transmit scale, or None for 1
    conc: Tensor | None = None  # concomitant-field coefficients (see fields.SpinFields)
    mode: int = 0  # 0 none, 1 leading order, 2 exact


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


# Spins x ADC samples evaluated per block in a free run; bounds temporary memory.
_MAX_BLOCK = 1 << 22


def _concomitant(lin, g, spins: _Spins):
    """Concomitant field [Hz] of every spin in a segment with gradient g, given the
    rest of its off-resonance ``lin`` (used by the exact mode only)."""
    c = spins.conc
    if spins.mode == 1:
        return c @ gradient_monomials(g)
    return exact_excess(lin, c[:, :3] @ g, c[:, 3:6] @ g, c[:, 6])


def _rf_segment(mx, my, mz, dt, rf, g, off, spins: _Spins):
    bz = spins.pos @ g + spins.db0  # off-resonance in the rotating frame [Hz]
    if off is not None:
        bz = bz + off
    if spins.mode:
        bz = bz + _concomitant(bz, g, spins)
    mx, my, mz = _relax(mx, my, mz, 0.5 * dt, spins)
    w = -TWO_PI * dt
    wx, wy = w * rf[0], w * rf[1]
    if spins.b1 is not None:  # B1+: every spin sees the RF field scaled by its b1
        wx, wy = wx * spins.b1, wy * spins.b1
    mx, my, mz = _rotate(mx, my, mz, wx, wy, w * bz)
    return _relax(mx, my, mz, 0.5 * dt, spins)


def _sample(mx, my, psi, spins: _Spins):
    """(2, C) demodulated signal of the current magnetization."""
    re = mx @ spins.sens_re - my @ spins.sens_im
    im = mx @ spins.sens_im + my @ spins.sens_re
    c, s = torch.cos(psi), torch.sin(psi)  # demodulate by exp(-i*psi)
    return torch.stack([re * c + im * s, im * c - re * s])


def _extra_phase(dt, g, off, spins: _Spins):
    """Cumulative phase [cycles] of the uniform offset and the concomitant field over a
    free run, (N, n) or (n,) (or None if both are absent): the moments of b0_offset and
    of G_i G_j at leading order, the per-segment |B| in exact mode."""
    out = None if off is None else torch.cumsum(off * dt, 0)  # (n,)
    if spins.mode == 1:
        q = torch.cumsum(gradient_monomials(g) * dt[:, None], 0)  # (n, 6)
        conc = spins.conc @ q.T
        out = conc if out is None else conc + out
    elif spins.mode == 2:
        lin = spins.pos @ g.T + spins.db0[:, None]  # (N, n)
        if off is not None:
            lin = lin + off
        c = spins.conc
        conc = torch.cumsum(exact_excess(lin, c[:, :3] @ g.T, c[:, 3:6] @ g.T, c[:, 6:]) * dt, 1)
        out = conc if out is None else conc + out
    return out


def _free_run(mx, my, mz, dt, g, off, psi, adc_idx, spins: _Spins):
    """A stretch of RF-free segments in closed form.

    With t and k the cumulative time and gradient moment from the run's start,
    Mxy(t) = Mxy(0) exp(-t/T2) exp(-i 2 pi (k.r + dB0 t)) and Mz relaxes as usual, so
    every ADC sample in the run is a sum over spins of one exponential: computed
    blockwise as matrix products instead of segment by segment. The field model adds
    the phase of ``_extra_phase``.
    """
    t = torch.cumsum(dt, 0)
    k = torch.cumsum(g * dt[:, None], 0)
    extra = _extra_phase(dt, g, off, spins)
    if extra is not None and extra.dim() == 1:
        extra = extra.expand(mx.shape[0], -1)
    sig = None
    if adc_idx:
        idx = torch.tensor(adc_idx, device=dt.device)
        ta, ka, pa = t[idx], k[idx], psi[idx]
        v_re = spins.sens_re * mx[:, None] - spins.sens_im * my[:, None]  # (N, C): coil * Mxy
        v_im = spins.sens_re * my[:, None] + spins.sens_im * mx[:, None]
        block = max(1, _MAX_BLOCK // max(mx.shape[0], 1))
        parts = []
        for j in range(0, len(adc_idx), block):
            tj, kj = ta[j : j + block], ka[j : j + block]
            cyc = spins.pos @ kj.T + spins.db0[:, None] * tj  # (N, n)
            if extra is not None:
                cyc = cyc + extra[:, idx[j : j + block]]
            phase = TWO_PI * cyc
            decay = torch.exp(-spins.r2[:, None] * tj)
            e_re, e_im = decay * torch.cos(phase), -decay * torch.sin(phase)
            parts.append(torch.stack([v_re.T @ e_re - v_im.T @ e_im, v_re.T @ e_im + v_im.T @ e_re]))
        s_re, s_im = torch.cat(parts, dim=-1)  # (C, n_adc) each
        c, s = torch.cos(pa), torch.sin(pa)
        sig = torch.stack([s_re * c + s_im * s, s_im * c - s_re * s]).permute(2, 0, 1)  # (n_adc, 2, C)
    cyc = spins.pos @ k[-1] + spins.db0 * t[-1]
    if extra is not None:
        cyc = cyc + extra[:, -1]
    mx, my = _precess(mx, my, -TWO_PI * cyc)
    mx, my, mz = _relax(mx, my, mz, t[-1], spins)
    return mx, my, mz, sig


def _steps(seq: Sequence, merge: bool) -> list[tuple[int, int]]:
    """Split the sequence into [start, end) steps: single RF segments, and free runs of
    consecutive RF-free segments (each run ends at a spoiled segment). Without
    ``merge`` every segment is its own step."""
    steps, s, n = [], 0, len(seq)
    while s < n:
        e = s + 1
        if merge and not seq.is_rf[s]:
            while e < n and not seq.is_rf[e] and not seq.spoil[e - 1]:
                e += 1
        steps.append((s, e))
        s = e
    return steps


def _run(mx, my, mz, steps, seq: Sequence, spins: _Spins, use_off: bool):
    samples = []
    for a, b in steps:
        if seq.is_rf[a]:
            off = seq.b0_offset[a] if use_off else None
            mx, my, mz = _rf_segment(mx, my, mz, seq.dt[a], seq.rf[a], seq.grad[a], off, spins)
            if seq.adc[a]:
                samples.append(_sample(mx, my, seq.adc_phase[a], spins)[None])
        else:
            adc_idx = [i for i in range(b - a) if seq.adc[a + i]]
            off = seq.b0_offset[a:b] if use_off else None
            mx, my, mz, sig = _free_run(mx, my, mz, seq.dt[a:b], seq.grad[a:b], off, seq.adc_phase[a:b],
                                        adc_idx, spins)
            if sig is not None:
                samples.append(sig)
        if seq.spoil[b - 1]:
            mx, my = torch.zeros_like(mx), torch.zeros_like(my)
    n_coils = spins.sens_re.shape[1]
    sig = torch.cat(samples) if samples else mx.new_zeros(0, 2, n_coils)
    return mx, my, mz, sig


def simulate(
    phantom: Phantom,
    seq: Sequence,
    coils: Tensor | None = None,
    *,
    m_init: Tensor | None = None,
    checkpoint_every: int | None = None,
    merge_free_runs: bool = True,
    scanner: Scanner | None = None,
) -> SimResult:
    """Run ``seq`` on ``phantom``.

    coils: (N, C) complex receive sensitivities at the spin positions (default:
        one uniform coil).
    m_init: (N, 3) starting magnetization (default: equilibrium, Mz = PD).
    checkpoint_every: recompute the forward pass in chunks of about this many
        segments during backward instead of storing it, trading compute for memory.
    merge_free_runs: evaluate stretches of RF-free segments in closed form (fast).
        False steps segment by segment; the results agree to rounding error.
    scanner: hardware field model (``fields.Scanner``: concomitant fields); None is
        the ideal scanner. ``seq.b0_offset`` (None: zeros) is always applied.
    """
    pd = phantom.pd
    seq = seq.to(dtype=pd.dtype, device=pd.device)
    n = len(phantom)
    if coils is None:
        sens_re, sens_im = torch.ones_like(pd)[:, None], torch.zeros_like(pd)[:, None]
    else:
        sens_re, sens_im = coils.real.to(pd.dtype), coils.imag.to(pd.dtype)
    fields = spin_fields(scanner, phantom.pos)
    spins = _Spins(fields.pos, phantom.db0, pd, 1.0 / phantom.t1, 1.0 / phantom.t2, sens_re, sens_im, phantom.b1,
                   fields.conc, fields.mode)
    # a zero uniform offset is skipped, so that results without one are unchanged bit for bit
    off = seq.b0_offset
    use_off = off is not None and (off.requires_grad or bool(off.any()))

    if m_init is None:
        mx, my, mz = torch.zeros_like(pd), torch.zeros_like(pd), pd
    else:
        mx, my, mz = m_init.unbind(-1)

    # group steps into chunks of about checkpoint_every segments
    steps, chunks = _steps(seq, merge_free_runs), []
    for step in steps:
        if not chunks or (checkpoint_every and chunks[-1][-1][1] - chunks[-1][0][0] >= checkpoint_every):
            chunks.append([])
        chunks[-1].append(step)
    use_ckpt = checkpoint_every is not None and torch.is_grad_enabled()
    sigs = [pd.new_zeros(0, 2, sens_re.shape[1])]
    for part in chunks:
        if use_ckpt:
            mx, my, mz, sig = checkpoint(_run, mx, my, mz, part, seq, spins, use_off, use_reentrant=False)
        else:
            mx, my, mz, sig = _run(mx, my, mz, part, seq, spins, use_off)
        sigs.append(sig)
    sig = torch.cat(sigs)  # (n_adc, 2, C)
    assert sig.shape[0] == seq.n_adc and n == mx.shape[0]
    return SimResult(torch.complex(sig[:, 0], sig[:, 1]).T, torch.stack([mx, my, mz], dim=-1))
