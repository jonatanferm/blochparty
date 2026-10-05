"""Reference models used to validate the Bloch simulator.

These are deliberately written differently from ``bloch.py``: closed-form
expressions and an explicit signal equation instead of incremental rotations,
so agreement is meaningful evidence that both are right.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .phantom import Phantom
from .sequence import Sequence

# --- closed-form results -----------------------------------------------------


def ernst_angle(tr, t1):
    """Flip angle [rad] maximizing the spoiled-GRE steady-state signal."""
    return torch.arccos(torch.exp(-torch.as_tensor(tr) / torch.as_tensor(t1)))


def spoiled_gre_signal(pd, t1, t2, tr, te, flip_rad):
    """Steady-state transverse magnetization of an ideally spoiled GRE at TE."""
    e1 = torch.exp(-torch.as_tensor(tr) / t1)
    a = torch.as_tensor(flip_rad)
    return pd * torch.sin(a) * (1 - e1) / (1 - torch.cos(a) * e1) * torch.exp(-torch.as_tensor(te) / t2)


def inversion_recovery_mz(pd, t1, ti):
    return pd * (1 - 2 * torch.exp(-torch.as_tensor(ti) / t1))


# --- signal equation -----------------------------------------------------------


def signal_equation(phantom: Phantom, seq: Sequence, coils: Tensor | None = None) -> Tensor:
    """(C, n_adc) signal from the explicit MR signal equation

        s(t) = sum_r C(r) * Mxy0(r) * exp(-t/T2) * exp(-i 2 pi (k(t).r + dB0 t)),

    with k(t) and t measured from the centre of the last RF pulse.

    Valid for spoiled-GRE-like sequences, which is asserted: no gradient during
    RF, and transverse magnetization ideally spoiled before every RF pulse, so
    only Mz carries over between excitations. RF pulses are treated as
    instantaneous rotations at their centre (ignoring relaxation and
    off-resonance during the pulse).
    """
    pd, t1, t2, db0, pos = phantom.pd, phantom.t1, phantom.t2, phantom.db0, phantom.pos
    seq = seq.to(dtype=pd.dtype, device=pd.device)
    cdtype = torch.complex128 if pd.dtype == torch.float64 else torch.complex64
    sens = torch.ones(len(phantom), 1, dtype=cdtype) if coils is None else coils.to(cdtype)

    mz_ref = pd  # Mz at the reference time (start, or centre of the last pulse)
    t = pd.new_zeros(())  # time since the reference
    k = pd.new_zeros(3)  # k-space position since the last pulse [cycles/m]
    mxy0 = torch.zeros_like(pd, dtype=cdtype)  # Mxy just after the last pulse
    coherent = False
    out = []
    for s in range(len(seq)):
        dt, g = seq.dt[s], seq.grad[s]
        if seq.is_rf[s]:
            assert not coherent, "transverse magnetization must be spoiled before each RF pulse"
            assert torch.all(g == 0), "gradients during RF are not supported"
            t_centre = t + dt / 2
            mz = pd + (mz_ref - pd) * torch.exp(-t_centre / t1)
            rf = seq.rf[s]
            alpha = 2 * math.pi * torch.linalg.vector_norm(rf) * dt
            phi = torch.atan2(rf[1], rf[0])
            mxy0 = 1j * torch.exp(1j * phi) * mz * torch.sin(alpha)
            mz_ref, t, k, coherent = mz * torch.cos(alpha), dt / 2, torch.zeros_like(k), True
            continue
        t, k = t + dt, k + g * dt
        if seq.adc[s]:
            phase = -2 * math.pi * (pos @ k + db0 * t)
            mxy = mxy0 * torch.exp(-t / t2) * torch.exp(1j * phase)
            out.append((sens * mxy[:, None]).sum(dim=0) * torch.exp(-1j * seq.adc_phase[s]))
        if seq.spoil[s]:
            mxy0, coherent = torch.zeros_like(mxy0), False
    return torch.stack(out, dim=1)
