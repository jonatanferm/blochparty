"""The transmit chain: what the RF amplifier does to the pulses the sequence asks for.

``amplifier(seq, ...)`` returns the sequence with every RF segment's (x, y) field replaced by
what a solid-state amplifier delivers, per pulse (a run of consecutive RF segments):

* AM/AM, Rapp's model: |out| = g |in| / (1 + (g |in| / a_sat)^(2p))^(1/(2p)): linear with gain g
  at low drive, saturating smoothly at a_sat [Hz of B1], sharpness p;
* AM/PM: a phase shift phi_max (|in| / a_sat)^2 / (1 + (|in| / a_sat)^2) [rad] (Saleh-type),
  growing with drive, which distorts the phase of shaped pulses;
* droop: the supply sags with the energy delivered within the pulse, the amplitude scaled by
  exp(-droop * int_0^t |out|^2 dt') [droop in 1/(Hz^2 s)].

The models' parameters are what a calibration on a real scanner would fit (``docs/REALISM.md``,
R7); defaults are an ideal amplifier (gain 1, no saturation, no AM/PM, no droop).
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch

from ..sequence import Sequence


def rapp(amp, gain: float = 1.0, a_sat: float = math.inf, p: float = 2.0):
    """AM/AM of Rapp's model."""
    a = gain * amp
    if math.isinf(a_sat):
        return a
    return a / (1 + (a / a_sat) ** (2 * p)) ** (1 / (2 * p))


def am_pm(amp, a_sat: float = math.inf, phi_max: float = 0.0):
    """AM/PM phase shift [rad]."""
    if math.isinf(a_sat) or phi_max == 0.0:
        return torch.zeros_like(amp)
    r2 = (amp / a_sat) ** 2
    return phi_max * r2 / (1 + r2)


def pulses(seq: Sequence) -> list[tuple[int, int]]:
    """[start, end) of every run of consecutive RF segments."""
    out, s = [], 0
    while s < len(seq):
        if seq.is_rf[s]:
            e = s + 1
            while e < len(seq) and seq.is_rf[e]:
                e += 1
            out.append((s, e))
            s = e
        else:
            s += 1
    return out


def amplifier(
    seq: Sequence, gain: float = 1.0, a_sat: float = math.inf, p: float = 2.0, phi_max: float = 0.0, droop: float = 0.0
) -> Sequence:
    """The sequence with the RF the amplifier delivers (see the module docstring)."""
    rf = seq.rf.clone()
    for a, b in pulses(seq):
        x = rf[a:b]
        amp = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        ph = torch.atan2(x[:, 1], x[:, 0])
        out = rapp(amp, gain, a_sat, p)
        ph = ph + am_pm(amp, a_sat, phi_max)
        if droop:
            # energy delivered before each segment's midpoint
            e = torch.cumsum(out**2 * seq.dt[a:b], 0) - 0.5 * out**2 * seq.dt[a:b]
            out = out * torch.exp(-droop * e)
        rf[a:b] = torch.stack([out * torch.cos(ph), out * torch.sin(ph)], -1)
    return replace(seq, rf=rf)
