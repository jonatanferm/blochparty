"""Restricted diffusion (``restriction.py``): walkers in impermeable planes, cylinders and spheres.
References: the reflected walk's equilibrium must be uniform (Kolmogorov-Smirnov), and the
short-gradient-pulse signals: Tanner-Stejskal's series between planes at finite diffusion times,
the long-time limits |FT of the compartment|^2 for spheres ([3 (sin x - x cos x) / x^3]^2) and
cylinders ([2 J1(x) / x]^2), free diffusion along a cylinder. The GPU walks the same paths as
the core."""

import math

import numpy as np
import pytest
import torch
from scipy.special import j1

import blochparty as bp
from blochparty.restriction import Restriction, n_sub, start_offsets, tau_max, walk_segment
from cases import sgp

f64 = torch.float64
D = 2e-9


def walkers(n, kind, size, axis):
    return bp.Phantom(pos=torch.zeros(n, 3, dtype=f64), pd=torch.ones(n, dtype=f64), t1=torch.full((n,), 1e6, dtype=f64),
                      t2=torch.full((n,), 1e6, dtype=f64), db0=torch.zeros(n, dtype=f64),
                      diffusion=torch.full((n,), D, dtype=f64), restriction=bp.Restriction(kind, size, axis))


def signals(sim, ph, qs, delta, big_delta, **kw):
    return np.array([float(sim(ph, sgp(q, delta, big_delta), seed=2, **kw).signal[0, 0].abs()) / len(ph) for q in qs])


@pytest.mark.slow
@pytest.mark.parametrize("kind, axis, power", [("sphere", None, 3), ("cylinder", (0.0, 0.0, 1.0), 2)])
def test_equilibrium_is_uniform(kind, axis, power):
    # specular reflection keeps the uniform density (a radial mirror left the wall shell 11 % empty)
    n, r_c = 20000, 5e-6
    res = Restriction(kind, r_c, axis).expand(n, torch.zeros(n, dtype=f64))
    idx = np.arange(n, dtype=np.uint64)
    centre = torch.zeros(n, 3, dtype=f64)
    r = centre + start_offsets(res, 1, idx, centre)
    sd = torch.full((n,), D, dtype=f64).sqrt()
    tm = tau_max(res, torch.full((n,), D, dtype=f64))
    for seg in range(16):  # 80 ms >> R^2 / D
        _, r = walk_segment(r, centre, sd, res, torch.tensor(5e-3, dtype=f64), seg, n_sub(5e-3, tm), 1, idx)
    rr = r.norm(dim=-1) if kind == "sphere" else r[:, :2].norm(dim=-1)
    assert float(rr.max()) <= r_c * (1 + 1e-12)
    u = np.sort(((rr / r_c) ** power).numpy())
    assert np.abs(u - (np.arange(n) + 0.5) / n).max() < 1.63 / math.sqrt(n)  # KS at 1 %


def test_gpu_walks_the_core_paths(gpu):
    qs = np.array([0.6, 1.2]) / 5e-6
    for kind, axis, size in (("planes", (0.6, 0.8, 0.0), 10e-6), ("cylinder", (0.0, 0.6, 0.8), 5e-6),
                             ("sphere", None, 5e-6)):
        ph = walkers(1500, kind, size, axis)
        a = signals(bp.simulate, ph, qs, 1e-3, 20e-3)
        b = signals(gpu.simulate, ph, qs, 1e-3, 20e-3)
        assert np.abs(a - b).max() < 2e-5, (kind, a, b)


@pytest.mark.parametrize("big_delta", [10e-3, 25e-3, 200e-3])
def test_planes_tanner_stejskal(gpu, big_delta):
    n, L = 400_000, 10e-6
    qs = np.array([0.37, 0.8, 1.3]) * 2 * math.pi / L
    sim = signals(gpu.simulate, walkers(n, "planes", L, (1.0, 0.0, 0.0)), qs, 0.02e-3, big_delta)
    x = qs * L
    ref = 2 * (1 - np.cos(x)) / x**2 + 4 * x**2 * sum(
        np.exp(-(k * math.pi) ** 2 * D * big_delta / L**2) * (1 - (-1) ** k * np.cos(x)) / (x**2 - (k * math.pi) ** 2) ** 2
        for k in range(1, 400))
    assert np.abs(sim - ref).max() < 3e-3, (sim, ref)


@pytest.mark.slow
def test_sphere_and_cylinder_long_time_limits(gpu):
    n, r_c = 1_000_000, 5e-6
    qs = np.array([0.6, 1.0, 1.4, 2.0]) / r_c
    x = qs * r_c
    s = signals(gpu.simulate, walkers(n, "sphere", r_c, None), qs, 5e-6, 300e-3)
    c = signals(gpu.simulate, walkers(n, "cylinder", r_c, (0.0, 0.0, 1.0)), qs, 5e-6, 300e-3)
    assert np.abs(s - (3 * (np.sin(x) - x * np.cos(x)) / x**3) ** 2).max() < 2e-3
    assert np.abs(c - (2 * j1(x) / x) ** 2).max() < 2e-3
    # along the cylinder: free diffusion
    q = np.array([2e5, 5e5])
    free = signals(gpu.simulate, walkers(200_000, "cylinder", r_c, (1.0, 0.0, 0.0)), q, 1e-3, 30e-3)
    assert np.abs(free - np.exp(-q**2 * (30e-3 - 1e-3 / 3) * D)).max() < 3e-3
