"""Bloch simulator vs. direct numerical integration of the Bloch equations (SciPy).

The ODE solver knows nothing about rotations or operator splitting, so this
checks the integrator, sign conventions and units end to end on random input.
"""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

from blochparty import Phantom, simulate
from cases import random_case


def bloch_ode(phantom: Phantom, seq) -> np.ndarray:
    pos, pd = phantom.pos.numpy(), phantom.pd.numpy()
    r1, r2, db0 = 1 / phantom.t1.numpy(), 1 / phantom.t2.numpy(), phantom.db0.numpy()
    out = []
    for p in range(len(pd)):
        m = np.array([0.0, 0.0, pd[p]])
        for s in range(len(seq)):
            dt = float(seq.dt[s])
            bx, by = seq.rf[s].numpy()
            bz = pos[p] @ seq.grad[s].numpy() + db0[p]
            b = 2 * math.pi * np.array([bx, by, bz])  # rad/s

            def f(_, m):
                return np.cross(m, b) - np.array([r2[p] * m[0], r2[p] * m[1], r1[p] * (m[2] - pd[p])])

            m = solve_ivp(f, (0, dt), m, method="DOP853", rtol=1e-11, atol=1e-13).y[:, -1]
            if seq.spoil[s]:
                m[:2] = 0
        out.append(m)
    return np.array(out)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rotations_exact_without_relaxation(seed):
    phantom, seq = random_case(seed)
    phantom = replace(phantom, t1=torch.full_like(phantom.t1, 1e12), t2=torch.full_like(phantom.t2, 1e12))
    np.testing.assert_allclose(simulate(phantom, seq).m.numpy(), bloch_ode(phantom, seq), atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_ode_with_second_order_convergence(seed):
    # Relaxation inside RF segments uses Strang splitting: once in the asymptotic
    # regime the error must fall ~4x per halving of the RF step.
    phantom, seq = random_case(seed)
    m_ref = bloch_ode(phantom, seq)
    err = {k: np.abs(simulate(phantom, seq.subdivide_rf(k)).m.numpy() - m_ref).max() for k in (2, 4, 8)}
    for k in (2, 4):
        assert 3.5 < err[k] / err[2 * k] < 5.0, err
    assert err[8] < 2e-5, err
