"""Scanner-frame field maps (``maps.py``) sampled where each spin is. Trilinear interpolation is
exact for linear fields, so: static linear maps equal the same fields given per spin; a spin
moving through a linear B0 map equals a static field gradient in the sequence; a translated
object under mapped coils equals the static moved object; the GPU equals the core."""

import math
from dataclasses import replace

import numpy as np
import torch

import blochparty as bp

from helpers import rel

f64 = torch.float64


def grid(nz=3, ny=7, nx=7):
    zz, yy, xx = torch.meshgrid(torch.linspace(-0.1, 0.1, nz, dtype=f64), torch.linspace(-0.3, 0.3, ny, dtype=f64),
                                torch.linspace(-0.3, 0.3, nx, dtype=f64), indexing="ij")
    return (-0.3, -0.3, -0.1), (0.1, 0.1, 0.1), xx, yy, zz


def test_static_linear_maps_equal_per_spin_fields():
    origin, spacing, xx, yy, _ = grid()
    coils = torch.stack([1 + 0.5 * xx + 0.2j * yy, 0.8 - 0.3 * yy + 0.1j * xx])
    maps = bp.FieldMaps(origin, spacing, b0=30 * xx - 10 * yy, b1=1 + 0.4 * xx, coils=coils)
    ph = bp.shepp_logan(16)
    seq = bp.gre_2d(16, n_dummy=2)
    x, y = ph.pos[:, 0], ph.pos[:, 1]
    explicit = replace(ph, db0=ph.db0 + 30 * x - 10 * y, b1=1 + 0.4 * x)
    want = bp.simulate(explicit, seq, torch.stack([1 + 0.5 * x + 0.2j * y, 0.8 - 0.3 * y + 0.1j * x], -1))
    assert rel(bp.simulate(ph, seq, maps=maps).signal, want.signal) < 1e-12


def test_moving_through_linear_b0_is_a_gradient():
    origin, spacing, xx, _, _ = grid()
    c = 30.0  # Hz/m
    ph = bp.shepp_logan(16)
    vel = torch.zeros(len(ph), 3, dtype=f64)
    vel[:, 0] = 0.05
    ph = replace(ph, motion=bp.Motion(velocity=vel))
    seq = bp.gre_2d(16, n_dummy=2)
    a = bp.simulate(ph, seq, maps=bp.FieldMaps(origin, spacing, b0=c * xx))
    g = seq.grad.clone()
    g[:, 0] += c
    assert rel(a.signal, bp.simulate(ph, replace(seq, grad=g)).signal) < 1e-12


def test_translated_object_under_mapped_coils():
    origin, spacing, xx, yy, _ = grid()
    coils = torch.stack([torch.exp(-((xx - 0.1) ** 2 + yy**2) / 0.05), torch.exp(-((xx + 0.1) ** 2 + yy**2) / 0.05)])
    maps = bp.FieldMaps(origin, spacing, coils=coils.to(torch.complex128))
    ph = bp.shepp_logan(16)
    seq = bp.gre_2d(16, n_dummy=2)
    b = np.array([0.03, -0.02, 0.0])
    rig = bp.Motion(rigid=lambda t: (np.broadcast_to(np.eye(3), (len(t), 3, 3)), np.broadcast_to(b, (len(t), 3))))
    moving = bp.simulate(replace(ph, motion=rig), seq, maps=maps)
    moved = bp.simulate(replace(ph, pos=ph.pos + torch.as_tensor(b)), seq, maps=maps)
    assert rel(moving.signal, moved.signal) < 1e-12


def test_adc_during_rf_under_mapped_coils():
    """An ADC sample at the end of an RF segment sees the mapped coils where the moving spin is
    (it read the zero placeholders before)."""
    origin, spacing, xx, yy, _ = grid()
    coils = torch.stack([torch.exp(-((xx - 0.1) ** 2 + yy**2) / 0.05), torch.exp(-((xx + 0.1) ** 2 + yy**2) / 0.05)])
    maps = bp.FieldMaps(origin, spacing, coils=coils.to(torch.complex128))
    ph = bp.shepp_logan(16)
    seq = bp.gre_2d(16, n_dummy=2)
    seq = replace(seq, adc=[a or r for a, r in zip(seq.adc, seq.is_rf)])
    b = np.array([0.03, -0.02, 0.0])
    rig = bp.Motion(rigid=lambda t: (np.broadcast_to(np.eye(3), (len(t), 3, 3)), np.broadcast_to(b, (len(t), 3))))
    moving = bp.simulate(replace(ph, motion=rig), seq, maps=maps)
    moved = bp.simulate(replace(ph, pos=ph.pos + torch.as_tensor(b)), seq, maps=maps)
    rf_rows = torch.tensor([r for a, r in zip(seq.adc, seq.is_rf) if a])
    assert moved.signal[:, rf_rows].abs().max() > 0
    assert rel(moving.signal, moved.signal) < 1e-12


def test_gpu_matches_core(gpu):
    origin, spacing, xx, yy, _ = grid(4, 9, 9)
    coils = torch.stack([torch.exp(-((xx - 0.1) ** 2 + yy**2) / 0.02) * torch.exp(1j * 3 * yy),
                         torch.exp(-((xx + 0.1) ** 2 + yy**2) / 0.02) * torch.exp(-2j * xx)])
    maps = bp.FieldMaps(origin, spacing, b0=40 * xx**2 - 20 * yy, b1=1 + 0.3 * xx, coils=coils)
    ph = bp.shepp_logan(16)
    g = torch.Generator().manual_seed(1)
    mot = bp.Motion(velocity=(torch.rand(len(ph), 3, generator=g, dtype=f64) - 0.5) * 0.1,
                    rigid=lambda t: (np.stack([np.eye(3)] * len(t)),
                                     np.stack([[0.02 * math.sin(2 * math.pi * x / 0.1), 0, 0] for x in t])))
    seq = bp.gre_2d(16, n_dummy=2)
    for p in (ph, replace(ph, motion=mot), replace(ph, motion=mot, diffusion=torch.full((len(ph),), 2e-9, dtype=f64))):
        assert rel(gpu.simulate(p, seq, maps=maps, seed=1).signal, bp.simulate(p, seq, maps=maps, seed=1).signal) < 1e-5
