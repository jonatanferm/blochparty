"""Sequences with learnable parameters (ROADMAP M7's parameterized sequence API starts here): a spoiled GRE whose
per-line flip angles, TE, TR and k-space line positions are tensors (``learnable_gre``), and ``Design``, those
parameters constrained to feasible ranges by smooth maps of unconstrained ones, as the lesion-classification
experiment (validation/design/lesion_design.py) learns them jointly with a network through the GPU adjoint."""

from __future__ import annotations

import math

import torch
from torch import nn

from ..sequence import Sequence
from ..sequences.builder import SequenceBuilder

# the experiment's acquisition: 64 x 64 over 220 mm, 10 us dwell, 0.2 ms hard pulses, 8 dummy excitations
FOV, NX, DWELL = 0.22, 64, 10e-6
RF_DURATION = 0.2e-3
N_DUMMY = 8
T_READ, T_ECHO = NX * DWELL, (NX / 2 + 0.5) * DWELL
T_TAIL = T_READ - T_ECHO + RF_DURATION / 2 + 0.2e-3  # the least TR - TE: the rest of the readout and a 0.2 ms spoiler


def learnable_gre(flips_deg: torch.Tensor, ky_cycles: torch.Tensor, te: torch.Tensor, tr: torch.Tensor, *, nx=NX, fov=FOV,
                  dwell=DWELL, n_dummy: int = N_DUMMY, dtype=torch.float32) -> Sequence:
    """A spoiled GRE of K = len(flips) lines: line j excites with ``flips_deg[j]`` and encodes ``ky_cycles[j]`` cycles
    across the field of view (continuous: not restricted to the Cartesian grid), all with echo time ``te`` and
    repetition time ``tr`` (tensors; gradients flow to all four through the builder and the adjoint). ``n_dummy``
    excitations with the mean flip bring the magnetisation toward its steady state first. Ideal gradients and spoiling."""
    dk = 1.0 / fov
    t_pre = te - RF_DURATION / 2 - T_ECHO
    t_fill = tr - RF_DURATION - t_pre - T_READ
    b = SequenceBuilder(dtype=dtype)
    mean_flip = flips_deg.mean()
    for _ in range(n_dummy):
        b.rf(RF_DURATION, flip_deg=mean_flip)
        b.delay(tr - RF_DURATION).spoil()
    for j in range(len(flips_deg)):
        ky = ky_cycles[j] * dk
        b.rf(RF_DURATION, flip_deg=flips_deg[j])
        b.delay(t_pre, grad=(-(nx / 2 + 0.5) * dk / t_pre, ky / t_pre, 0.0))
        b.readout(nx, dwell, grad=(dk / dwell, 0.0, 0.0))
        b.delay(t_fill).spoil()
    return b.build(kspace_shape=(len(flips_deg), nx), fov=fov, dwell=dwell)


class Design(nn.Module):
    """Arm C's parameters: flips in (2, 90) deg by a sigmoid, TE in ``te`` (s), TR = TE + the readout tail + an extra in
    ``tr_extra`` (so every design is feasible), ky in (-nx/2, nx/2) cycles by a tanh, all unconstrained underneath."""

    def __init__(self, lines: int, nx=NX, te=(2e-3, 80e-3), tr_extra=(0.5e-3, 100e-3), init_flip=25., init_te=6e-3, init_tr=15e-3,
                 ky_init: torch.Tensor | None = None):
        super().__init__()
        self.lines, self.nx, self.te_rng, self.tr_rng = lines, nx, te, tr_extra
        if not (te[0] < init_te < te[1] and tr_extra[0] < init_tr - init_te - T_TAIL < tr_extra[1]):
            raise ValueError(f"start TE {init_te * 1e3:.1f} / TR {init_tr * 1e3:.1f} ms outside the design's ranges")
        self.flip_raw = nn.Parameter(torch.full((lines,), math.log((init_flip - 2) / (90 - init_flip))))
        self.te_raw = nn.Parameter(torch.tensor(math.log((init_te - te[0]) / (te[1] - init_te))))
        extra = init_tr - init_te - T_TAIL
        self.tr_raw = nn.Parameter(torch.tensor(math.log((extra - tr_extra[0]) / (tr_extra[1] - extra))))
        if ky_init is None:
            ky_init = torch.linspace(-nx / 2 + nx / lines / 2, nx / 2 - nx / lines / 2, lines)  # evenly spread lines
        self.ky_raw = nn.Parameter(torch.atanh((ky_init.float() / (nx / 2)).clamp(-.999, .999)))

    def forward(self):
        flips = 2 + 88 * torch.sigmoid(self.flip_raw)
        te = self.te_rng[0] + (self.te_rng[1] - self.te_rng[0]) * torch.sigmoid(self.te_raw)
        tr = te + T_TAIL + self.tr_rng[0] + (self.tr_rng[1] - self.tr_rng[0]) * torch.sigmoid(self.tr_raw)
        ky = (self.nx / 2) * torch.tanh(self.ky_raw)
        return flips, ky, te, tr

    def sequence(self):
        return learnable_gre(*self())

    def describe(self):
        flips, ky, te, tr = (v.detach() for v in self())
        return {"flips_deg": flips.round(decimals=1).tolist(), "ky_cycles": ky.round(decimals=2).tolist(), "te_ms": round(te.item() * 1e3, 2),
                "tr_ms": round(tr.item() * 1e3, 2), "scan_ms": round(tr.item() * 1e3 * (self.lines + N_DUMMY), 1)}

    def summary(self):
        flips, ky, te, tr = (v.detach() for v in self())
        return (f"flip {flips.mean():.1f} ({flips.min():.1f}-{flips.max():.1f}) deg, TE {te * 1e3:.2f} ms, TR {tr * 1e3:.2f} ms, "
                f"ky {ky.min():.1f}..{ky.max():.1f} (std {ky.std():.1f}, |ky| < 4: {(ky.abs() < 4).sum().item()})")
