"""J-coupled spin fragments (blochparty.jcoupling) against analytic limits."""
import math

import numpy as np
import pytest
import torch

from blochparty import jcoupling as jc
from blochparty.constants import GAMMA_BAR
from blochparty.phantom import FAT_PEAKS

TE = np.array([10, 40, 80, 120, 160]) * 1e-3


def test_no_coupling_no_modulation():
    f = jc.Fragment((jc.Group("a", 1.3, 2), jc.Group("b", 2.0, 2)), ((0, 1, 0.0),))
    for mode, esp in (("se", None), ("cpmg", 10e-3)):
        amp = jc.echo_amplitudes(f, 3.0, TE if mode == "se" else 10e-3 * np.arange(1, 9), mode=mode, esp=esp)
        assert torch.allclose(amp, torch.ones_like(amp), atol=1e-10)


def test_equivalent_spins_do_not_modulate():
    """Coupling between spins of one shift is unobservable: A2 coupled to A2 at the same ppm stays at 1."""
    f = jc.Fragment((jc.Group("a", 1.3, 2), jc.Group("a'", 1.3, 2)), ((0, 1, 7.0),))
    amp = jc.echo_amplitudes(f, 3.0, TE, mode="se")
    assert torch.allclose(amp, torch.ones_like(amp), atol=1e-10)


def test_weak_coupling_spin_echo_is_cos_pi_j_te():
    """AX with the shift difference far above J (5 ppm at 3 T = 639 Hz, J 7 Hz): each doublet's echo is
    |cos(pi J TE)| (the coupling is not refocused by a pulse that flips both spins)."""
    J = 7.0
    f = jc.Fragment((jc.Group("a", 0.0, 1), jc.Group("x", 5.0, 1)), ((0, 1, J),))
    amp = jc.echo_amplitudes(f, 3.0, TE, mode="se")
    expect = np.abs(np.cos(math.pi * J * TE))  # weak-coupling limit; corrections O(J / dnu) ~ 1 %
    assert np.allclose(amp[:, 0].numpy(), expect, atol=0.01)
    assert np.allclose(amp[:, 1].numpy(), expect, atol=0.01)


def test_fast_cpmg_suppresses_the_modulation():
    """An AX pair at 0.4 ppm (51 Hz at 3 T) with J 7 Hz: one spin echo at 100 ms is clearly modulated; a CPMG train
    with 2 ms spacing (well under 1 / 51 Hz) keeps the echo near 1 out to 100 ms."""
    f = jc.Fragment((jc.Group("a", 0.9, 1), jc.Group("b", 1.3, 1)), ((0, 1, 7.0),))
    se = jc.echo_amplitudes(f, 3.0, [0.1], mode="se")[0, 0].item()
    cp = jc.echo_amplitudes(f, 3.0, 2e-3 * np.arange(1, 51), mode="cpmg", esp=2e-3)[-1, 0].item()
    assert se < 0.8 and cp > 0.95


def test_triglyceride_composition_matches_the_fat_spectrum():
    """The model's shares of all fat protons against the simulator's six-peak spectrum (Ren 2008): methyl 8.7 %,
    alpha-carboxyl + allylic (2.1 ppm) 12.8 %, diallylic 0.4 %, olefinic + glycerol CH (5.3 ppm) 4.8 %."""
    frags, bulk = jc.triglyceride_fragments()
    protons = {}
    for f in frags:
        for g in f.groups:
            protons[g.name] = protons.get(g.name, 0) + f.copies * g.protons
    total = bulk + sum(protons.values())
    peaks = dict((round(4.7 + s, 1), a) for s, a in FAT_PEAKS)
    assert protons["methyl"] / total == pytest.approx(peaks[0.9], abs=0.015)
    assert (protons["olefinic"] + protons["glycerol CH"]) / total == pytest.approx(peaks[5.3], abs=0.015)
    assert (protons["alpha-carboxyl"] + protons["allylic"]) / total == pytest.approx(peaks[2.1], abs=0.015)
    assert protons["diallylic"] / total == pytest.approx(peaks[2.8], abs=0.005)
