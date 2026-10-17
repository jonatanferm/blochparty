"""J-coupled protons under spin-echo trains: density matrices for small coupled spin fragments (ROADMAP M6.1).

A Bloch simulation treats every spin as uncoupled, so it cannot show J modulation: in a spin echo the coupling
between protons of neighbouring groups keeps evolving through the refocusing pulse, and a train of refocusing
pulses spaced closely against 1 / (shift difference) suppresses it (the "bright fat" of fast spin echo). This module
evolves the density matrix of a fragment of coupled groups (strong-coupling Hamiltonian, ideal hard pulses) and
returns each group's echo amplitude relative to the uncoupled case, for single spin echoes or a CPMG train.

``triglyceride_fragments`` is a reduced fat model: the coupled ends of the fatty-acid chains (methyl, carboxyl,
allylic, diallylic) and the glycerol backbone, weighted by triglyceride stoichiometry, plus the bulk methylene, whose
neighbours share its shift and which this model leaves unmodulated. ``fat_modulation`` sums them.

Finding (2026-10-13, validation/jcoupling/jcoupling_fat.py): with one intrinsic T2 calibrated so that CPMG at 8 ms spacing reads
145 ms (Keene 2020, 3 T), this model gives single-echo spin echoes an apparent T2 of 222 ms, against 48-68 ms
measured at 3 T (Jordan 2013, de Bazelaire 2004): the coupled end groups modulate and partly recover, while most of
the fat signal (the bulk methylene) is not J-modulated here. The in-vivo difference between spin-echo and CPMG fat
therefore needs more than these fragments (the methylene chain's near-equivalent many-spin dynamics, or mechanisms
fast refocusing suppresses besides J coupling), so the simulator keeps the CPMG fat T2 (tissuedb ``LIPIDS``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .constants import GAMMA_BAR

_DT = torch.complex128
_SX = torch.tensor([[0, 0.5], [0.5, 0]], dtype=_DT)
_SY = torch.tensor([[0, -0.5j], [0.5j, 0]], dtype=_DT)
_SZ = torch.tensor([[0.5, 0], [0, -0.5]], dtype=_DT)
_E2 = torch.eye(2, dtype=_DT)


@dataclass(frozen=True)
class Group:
    """Magnetically equivalent protons: name, chemical shift [ppm], count."""

    name: str
    ppm: float
    protons: int


@dataclass(frozen=True)
class Fragment:
    """Coupled groups and their couplings ((i, j, J [Hz]) between group indices); ``copies`` per molecule."""

    groups: tuple[Group, ...]
    couplings: tuple[tuple[int, int, float], ...]
    copies: float = 1.0


def _kron_all(mats):
    out = torch.ones(1, 1, dtype=_DT)
    for m in mats:
        out = torch.kron(out, m)
    return out


def _op(single, k, n):
    return _kron_all([single if j == k else _E2 for j in range(n)])


def _rotation(axis: str, theta: float, n: int):
    """exp(-i theta sum_k I_axis,k): the same pulse on every spin (hard, on resonance for all)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    r = (
        torch.tensor([[c, -1j * s], [-1j * s, c]], dtype=_DT)
        if axis == "x"
        else torch.tensor([[c, -s], [s, c]], dtype=_DT)
    )
    return _kron_all([r] * n)


def echo_amplitudes(
    fragment: Fragment,
    field_T: float,
    times,
    *,
    mode: str = "cpmg",
    esp: float | None = None,
    refocus_deg: float = 180.0,
) -> torch.Tensor:
    """(len(times), len(groups)) echo amplitude of each group (|<F+>| of its spins) relative to its amplitude right
    after the 90x excitation, without relaxation. ``mode="se"``: one refocusing pulse (y) at TE / 2 for each TE in
    ``times``; ``"cpmg"``: 90x then refocusing pulses (y) every ``esp``, ``times`` the echo times (k esp, k = 1, 2,
    ...). The Hamiltonian is 2 pi (sum nu_k Iz_k + sum J I_k . I_l) in the frame of the fragment's mean shift."""
    groups = fragment.groups
    idx, n = [], 0
    for g in groups:
        idx.append(list(range(n, n + g.protons)))
        n += g.protons
    I = {(a, k): _op(m, k, n) for a, m in (("x", _SX), ("y", _SY), ("z", _SZ)) for k in range(n)}
    ref = sum(g.ppm for g in groups) / len(groups)
    H = sum(
        2 * math.pi * (g.ppm - ref) * 1e-6 * GAMMA_BAR * field_T * I["z", k] for g, ks in zip(groups, idx) for k in ks
    )
    for i, j, J in fragment.couplings:
        for k in idx[i]:
            for l in idx[j]:
                H = H + 2 * math.pi * J * (I["x", k] @ I["x", l] + I["y", k] @ I["y", l] + I["z", k] @ I["z", l])
    E, V = torch.linalg.eigh(H)

    def evolve(rho, t):
        u = V @ torch.diag(torch.exp(-1j * E * t)) @ V.conj().T
        return u @ rho @ u.conj().T

    observe = [sum(I["x", k] + 1j * I["y", k] for k in ks) for ks in idx]
    signal = lambda rho: torch.stack([torch.trace(rho @ f.conj().T).abs() for f in observe])
    r90, rref = _rotation("x", math.pi / 2, n), _rotation("y", math.radians(refocus_deg), n)
    rho = r90 @ sum(I["z", k] for k in range(n)) @ r90.conj().T
    s0 = signal(rho)
    out = []
    if mode == "se":
        for te in times:
            r = evolve(rho, te / 2)
            r = evolve(rref @ r @ rref.conj().T, te / 2)
            out.append(signal(r) / s0)
    elif mode == "cpmg":
        if esp is None:
            raise ValueError("cpmg needs esp")
        r = rho
        for _ in times:
            r = evolve(r, esp / 2)
            r = evolve(rref @ r @ rref.conj().T, esp / 2)
            out.append(signal(r) / s0)
    else:
        raise ValueError(f"mode must be 'se' or 'cpmg', not {mode!r}")
    return torch.stack(out)


def triglyceride_fragments(
    chain_length: float = 17.3, double_bonds: float = 1.9, interrupted: float = 0.2, J: float = 7.0
) -> tuple[list[Fragment], float]:
    """The coupled fragments of a triglyceride and the uncoupled bulk-methylene protons per molecule. Shifts: methyl
    0.90, bulk methylene 1.30, beta-carboxyl 1.58, allylic 2.02, alpha-carboxyl 2.24, diallylic 2.77, glycerol 4.20 /
    5.22, olefinic 5.32 ppm (Hamilton et al. 2011, NMR Biomed 24:784); vicinal J 6-8 Hz (7; glycerol 5). Composition
    per molecule: three chains of ``chain_length`` carbons with ``double_bonds`` double bonds in all, ``interrupted``
    of them methylene-interrupted; the defaults reproduce the simulator's six-peak spectrum (``phantom.FAT_PEAKS``,
    Ren et al. 2008: methyl 8.7 %, 2.1 ppm 12.8 %, diallylic 0.4 %, olefinic + glycerol CH 4.8 %). Each fragment
    carries the methylene next to its coupled group; the rest of the chain methylene is returned as uncoupled (its
    neighbours share its shift)."""
    plain = double_bonds - interrupted  # double bonds with an allylic CH2 on each side
    frags = [
        Fragment((Group("methyl", 0.90, 3), Group("methylene next to methyl", 1.30, 2)), ((0, 1, J),), 3),
        Fragment(
            (Group("alpha-carboxyl", 2.24, 2), Group("beta-carboxyl", 1.58, 2), Group("gamma methylene", 1.30, 2)),
            ((0, 1, J), (1, 2, J)),
            3,
        ),
        Fragment(
            (Group("olefinic", 5.32, 1), Group("allylic", 2.02, 2), Group("methylene next to allylic", 1.30, 2)),
            ((0, 1, J), (1, 2, J)),
            2 * plain,
        ),
        Fragment(
            (Group("olefinic", 5.32, 1), Group("diallylic", 2.77, 2), Group("olefinic", 5.32, 1)),
            ((0, 1, J), (1, 2, J)),
            interrupted,
        ),
        Fragment(
            (Group("glycerol CH2", 4.20, 2), Group("glycerol CH", 5.22, 1), Group("glycerol CH2", 4.20, 2)),
            ((0, 1, 5.0), (1, 2, 5.0)),
            1,
        ),
    ]
    total = 5 + 3 * (2 * chain_length - 1) - 2 * double_bonds  # protons per molecule
    named = (
        9 + 2 * double_bonds + 5 + 6 + 6 + 4 * plain + 2 * interrupted
    )  # methyl, olefinic, glycerol, alpha, beta, allylic, diallylic
    methylene = total - named  # chain CH2 at ~1.3 ppm
    neighbours = 3 * 2 + 3 * 2 + 2 * plain * 2  # carried by the methyl, carboxyl and allylic fragments
    if interrupted > plain:
        raise ValueError("more methylene-interrupted double bonds than double bonds")
    return frags, max(methylene - neighbours, 0.0)


def fat_modulation(
    times,
    field_T: float = 3.0,
    *,
    mode: str = "cpmg",
    esp: float | None = None,
    refocus_deg: float = 180.0,
    **composition,
) -> torch.Tensor:
    """(len(times),) J modulation of the whole fat signal (all protons, uncoupled bulk methylene at 1): the
    proton-weighted sum of ``triglyceride_fragments``' echo amplitudes; no relaxation."""
    frags, bulk = triglyceride_fragments(**composition)
    total = torch.full((len(times),), float(bulk), dtype=torch.float64)
    weight = float(bulk)
    for f in frags:
        amp = echo_amplitudes(f, field_T, times, mode=mode, esp=esp, refocus_deg=refocus_deg)
        for gi, g in enumerate(f.groups):
            total = total + f.copies * g.protons * amp[:, gi]
            weight += f.copies * g.protons
    return total / weight
