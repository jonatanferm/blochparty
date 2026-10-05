"""Extended Phase Graph (EPG) reference model.

An independent way to compute the signal of one voxel: instead of tracking
isochromats, the magnetization is expanded in Fourier "configuration states" of
the dephasing caused by gradients (Weigel, JMRI 41:266, 2015; Hennig 1988).
It captures stimulated echoes and gradient / RF spoiling exactly, so it is the
reference for realism work that ideal spoiling (``Sequence.spoil``) cannot check.

Conventions (identical to ``bloch.py``): dM/dt = gamma * M x B, so an RF pulse
along +x tips +z to +y; RF phase phi gives Mxy = i*exp(i*phi)*sin(alpha), and
Mxy = Mx + i*My evolves as exp(-i*2*pi*f*t) at offset f. One unit of
dephasing gradient (area 1/L over a voxel of length L) multiplies Mxy(x) by
exp(-i*theta(x)) with theta(x) = 2*pi*x/L. The states are defined by

    Mxy(x) = sum_k F_k exp(-i*k*theta(x)),    Mz(x) = sum_k Z_k exp(-i*k*theta(x)),

with F+_k = F_k, F-_k = conj(F_{-k}) and Z_k (k >= 0, Z_{-k} = conj(Z_k)). So a
positive unit gradient shifts F+ up and F- down, and **F+_0 is exactly the
voxel-mean Mxy in blochparty's convention** (no sign flip or conjugate needed).
This is Weigel's formulation with the RF rotation sense reversed (alpha ->
-alpha in his T matrix), because his pulse along +x tips +z to -y.

A state is a (3, n) complex tensor with rows (F+, F-, Z) over orders k = 0..n-1;
orders pushed beyond n - 1 are dropped. All operators are functional (no
in-place updates), so autograd reaches flip angles, phases, timings, T1 and T2.
RF pulses are instantaneous. Helper angles are in degrees, operator angles in rad.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


_REAL = {torch.complex128: torch.float64, torch.complex64: torch.float32}


def _cdtype(*xs) -> torch.dtype:
    single = any(torch.is_tensor(x) and x.dtype in (torch.float32, torch.complex64) for x in xs)
    return torch.complex64 if single else torch.complex128


def epg_init(n_states: int, m0=1.0, dtype=torch.complex128) -> Tensor:
    """Equilibrium: Z_0 = m0, everything else zero."""
    zero = torch.zeros(n_states, dtype=dtype)
    m0 = torch.as_tensor(m0, dtype=_REAL[dtype]).to(dtype).reshape(1)
    return torch.stack([zero, zero, torch.cat([m0, zero[1:]])])


def rf_matrix(alpha, phase=0.0, dtype=torch.complex128) -> Tensor:
    """3x3 RF operator acting on (F+_k, F-_k, Z_k) for every k; ``alpha``, ``phase`` in rad.

    The rotation dM/dt = gamma M x B gives for a pulse along (cos phase, sin phase, 0)
    (left-handed about that axis), written for (Mxy, conj(Mxy), Mz); it acts the same on
    every order because it is uniform over the voxel. Weigel's T matrix with alpha -> -alpha.
    """
    a = torch.as_tensor(alpha, dtype=_REAL[dtype])
    p = torch.exp(1j * torch.as_tensor(phase, dtype=_REAL[dtype])).to(dtype)
    c2, s2 = (torch.cos(a / 2) ** 2).to(dtype), (torch.sin(a / 2) ** 2).to(dtype)
    s, c = torch.sin(a).to(dtype), torch.cos(a).to(dtype)
    return torch.stack([
        torch.stack([c2, p * p * s2, 1j * p * s]),
        torch.stack([s2 / (p * p), c2, -1j * s / p]),
        torch.stack([0.5j * s / p, -0.5j * p * s, c]),
    ])


def epg_rf(state: Tensor, alpha, phase=0.0) -> Tensor:
    """Instantaneous RF pulse of flip ``alpha`` and phase ``phase`` [rad]."""
    return rf_matrix(alpha, phase, state.dtype) @ state


def epg_relax(state: Tensor, tau, t1, t2, m0=1.0, off_resonance_hz=0.0) -> Tensor:
    """Free evolution for ``tau`` seconds: T2 decay of F, T1 decay of Z with
    recovery of Z_0 towards ``m0``, and an optional uniform off-resonance
    phase (Mxy picks up exp(-i*2*pi*f*tau), so F+ and F- rotate oppositely)."""
    real = _REAL[state.dtype]
    tau, t1, t2 = (torch.as_tensor(x, dtype=real) for x in (tau, t1, t2))
    e1, e2 = torch.exp(-tau / t1), torch.exp(-tau / t2)
    w = torch.exp(-2j * math.pi * torch.as_tensor(off_resonance_hz, dtype=real) * tau)
    fp, fm, z = state[0] * (e2 * w), state[1] * (e2 / w), state[2] * e1
    z = torch.cat([z[:1] + torch.as_tensor(m0, dtype=real) * (1 - e1), z[1:]])
    return torch.stack([fp, fm, z])


def epg_grad(state: Tensor, n: int = 1) -> Tensor:
    """``n`` units of dephasing gradient (negative ``n`` rephases)."""
    for _ in range(abs(n)):
        fp, fm, z = state
        zero = torch.zeros_like(fp[:1])
        if n > 0:  # F_k <- F_{k-1}: F+ moves up, F- moves down, F+_0 = conj(F-_1)
            fp, fm = torch.cat([fm[1:2].conj(), fp[:-1]]), torch.cat([fm[1:], zero])
        else:
            fp, fm = torch.cat([fp[1:], zero]), torch.cat([fp[1:2].conj(), fm[:-1]])
        if state.shape[1] == 1:  # only order 0 kept: everything dephases out of it
            fp, fm = zero, zero
        state = torch.stack([fp, fm, z])
    return state


# --- sequence-level helpers --------------------------------------------------------


def rf_spoil_phases(n_pulses: int, increment_deg: float = 117.0) -> Tensor:
    """Quadratic RF-spoiling phases phi_j = increment * j(j+1)/2 [rad], j = 0..n-1
    (the phase step grows by ``increment`` every TR). Wrapped exactly in degrees
    so long trains do not lose precision."""
    return torch.tensor([math.radians((increment_deg * j * (j + 1) / 2) % 360.0) for j in range(n_pulses)],
                        dtype=torch.float64)


def epg_spoiled_gre(t1, t2, tr, te, flip, n_pulses: int, rf_spoil_increment_deg: float | None = 117.0,
                    *, ideal_spoil: bool = False, m0=1.0, n_states: int | None = None) -> Tensor:
    """Gradient-spoiled GRE: per TR an RF pulse of ``flip`` degrees (centre at t=0),
    echo at ``te``, then one unit of spoiler gradient before the next pulse.

    ``rf_spoil_increment_deg`` sets quadratic RF spoiling (None: constant phase 0).
    The signal is demodulated by the RF phase, as a scanner does. ``ideal_spoil``
    zeros all transverse states at the end of each TR (the ``Sequence.spoil`` model).
    Returns the (n_pulses,) complex echo signal.
    """
    phases = (rf_spoil_phases(n_pulses, rf_spoil_increment_deg) if rf_spoil_increment_deg is not None
              else torch.zeros(n_pulses, dtype=torch.float64))
    alpha = torch.deg2rad(torch.as_tensor(flip, dtype=torch.float64))
    state = epg_init(n_states or n_pulses + 1, m0, _cdtype(flip, t1, t2))
    out = []
    for phi in phases:
        state = epg_relax(epg_rf(state, alpha, phi), te, t1, t2, m0)
        out.append(state[0, 0] * torch.exp(-1j * phi))
        state = epg_grad(epg_relax(state, tr - te, t1, t2, m0))
        if ideal_spoil:
            state = torch.stack([torch.zeros_like(state[0]), torch.zeros_like(state[1]), state[2]])
    return torch.stack(out)


def epg_cpmg(t1, t2, esp, refocus_flips, excite_flip=90.0, *, m0=1.0, n_states: int | None = None) -> Tensor:
    """CPMG echo train: excitation (phase 0, so M lies along +y), then refocusing
    pulses ``refocus_flips`` [deg] with phase 90 deg (along +y, the CPMG condition)
    at esp/2 + (j - 1) * esp, each surrounded by one unit of crusher gradient.
    Returns the complex signal at the echoes, t = j * esp for j = 1..len(refocus_flips)."""
    flips = torch.deg2rad(torch.as_tensor(refocus_flips, dtype=torch.float64)).reshape(-1)
    n = flips.shape[0]
    state = epg_init(n_states or 2 * n + 2, m0, _cdtype(refocus_flips, excite_flip, t1, t2))
    state = epg_rf(state, torch.deg2rad(torch.as_tensor(excite_flip, dtype=torch.float64)), 0.0)
    out = []
    for beta in flips.unbind():
        state = epg_grad(epg_relax(state, esp / 2, t1, t2, m0))
        state = epg_rf(state, beta, math.pi / 2)
        state = epg_grad(epg_relax(state, esp / 2, t1, t2, m0))
        out.append(state[0, 0])
    return torch.stack(out)


def epg_bssfp(t1, t2, tr, flip, n_pulses: int, off_resonance_hz=0.0, *, phase_cycle_deg: float = 180.0,
              m0=1.0) -> Tensor:
    """Balanced SSFP: pulses of ``flip`` degrees with phase j * ``phase_cycle_deg``,
    no net gradient (so only order 0 is ever populated), echo at TE = TR/2,
    demodulated by the RF phase. Returns the (n_pulses,) complex signal."""
    alpha = torch.deg2rad(torch.as_tensor(flip, dtype=torch.float64))
    state = epg_init(1, m0, _cdtype(flip, t1, t2))
    out = []
    for j in range(n_pulses):
        phi = math.radians((j * phase_cycle_deg) % 360.0)
        state = epg_relax(epg_rf(state, alpha, phi), tr / 2, t1, t2, m0, off_resonance_hz)
        out.append(state[0, 0] * complex(math.cos(phi), -math.sin(phi)))
        state = epg_relax(state, tr / 2, t1, t2, m0, off_resonance_hz)
    return torch.stack(out)
