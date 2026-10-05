"""Low-SAR turbo spin echo: a refocusing flip-angle train minimising RF energy subject to
every echo reaching a target amplitude.

The target for echo j is a fraction ``FRACTION`` of the ideal 180° CPMG echo,
exp(-j ESP / T2). A constant train needs ~100° throughout (set by the first echo); a
variable train can start there and drop, because stimulated-echo pathways (decaying with
T1, not T2) carry the later echoes — the idea behind Hennig's TRAPS and Alsop's
pseudo-steady-state trains. The echoes come from the isochromat simulator (one voxel of
``K`` spins with crusher gradients, fused kernel), the energy from the sequence's own RF
(``blochparty.design.rf_energy``), and the optimizer is the augmented Lagrangian in
``blochparty.design.minimize``.

Independent checks: the optimized train's echoes recomputed with the extended phase graph
model (``epg_cpmg``); the energy against the cheapest constant train that meets the
targets (bisection on EPG); and the optimum itself against SciPy's SLSQP solving the same
problem on the EPG model.

    uv run python examples/design_tse_flips.py
"""

import math
import time

import torch

from blochparty import Phantom, SequenceBuilder
from blochparty.design import b1_uT, minimize, rf_energy
from blochparty.epg import epg_cpmg

T1, T2, ESP, N_ECHO = 1.0, 0.1, 10e-3, 12
FRACTION = 0.6
RF_DUR, DWELL = 1e-3, 2e-6  # 1 ms hard pulses (a 180° pulse is 11.7 µT); RF is off during the crushers
K, L = 64, 1e-3  # isochromats across one voxel of length L: more than the 2 N_ECHO + 1 EPG orders reached
TARGET = FRACTION * torch.exp(-torch.arange(1, N_ECHO + 1, dtype=torch.float64) * ESP / T2)

_x = torch.arange(K, dtype=torch.float64) * (L / K)
_full = lambda v: torch.full((K,), v, dtype=torch.float64)
VOXEL = Phantom(pos=torch.stack([_x, 0 * _x, 0 * _x], dim=-1), pd=_full(1.0), t1=_full(T1), t2=_full(T2),
                db0=_full(0.0))


def crusher(duration):
    """Gradient that dephases by one cycle across the voxel (one EPG order) over ``duration``."""
    return (1.0 / (L * duration), 0.0, 0.0)


def cpmg_sequence(flips):
    """90° excitation (phase 0), then refocusing pulses ``flips`` [deg] (phase 90°, CPMG)
    centred at ESP/2 + (j - 1) ESP after the excitation's centre, a crusher on each side of
    every one, and an ADC sample at every echo, j ESP."""
    b = SequenceBuilder().rf(RF_DUR, flip_deg=90.0)
    b.delay(ESP / 2 - RF_DUR, grad=crusher(ESP / 2 - RF_DUR))
    half = ESP / 2 - RF_DUR / 2 - DWELL / 2
    for j in range(N_ECHO):
        b.rf(RF_DUR, flip_deg=flips[j], phase=math.pi / 2)
        b.delay(half, grad=crusher(half)).readout(1, DWELL)
        if j < N_ECHO - 1:
            b.delay(half, grad=crusher(half))
    return b.build(esp=ESP)


def echoes(seq, simulate):
    """Echo amplitudes |voxel-mean Mxy| at the ADC samples."""
    return simulate(VOXEL, seq).signal[0].abs() / K


def epg_echoes(flips):
    return epg_cpmg(T1, T2, ESP, torch.as_tensor(flips, dtype=torch.float64)).abs()


def constant_baseline():
    """Smallest constant refocusing angle whose echoes all meet the targets (EPG, bisection)."""
    lo, hi = 1.0, 180.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if bool((epg_echoes([mid] * N_ECHO) >= TARGET).all()) else (mid, hi)
    return hi


def optimize(start=None, simulate=None):
    if simulate is None:
        from blochparty.fused import simulate_fused as simulate
    base = constant_baseline()
    e_ref = float(rf_energy(cpmg_sequence([base] * N_ECHO)))
    flips = torch.full((N_ECHO,), 120.0 if start is None else start, dtype=torch.float64, requires_grad=True)
    state = {}

    def objective():
        state["seq"] = cpmg_sequence(flips.unbind())
        return rf_energy(state["seq"]) / e_ref

    def constraints():  # echo_j / target_j - 1 >= 0
        seq = state["seq"] if torch.is_grad_enabled() else cpmg_sequence(flips.unbind())
        return echoes(seq, simulate) / TARGET - 1

    t0 = time.perf_counter()
    info = minimize([flips], objective, constraints, rho=10.0)
    seconds = time.perf_counter() - t0
    found = flips.detach().abs()
    seq = cpmg_sequence(found.unbind())
    return {"flips": found, "energy": float(rf_energy(seq)), "baseline_flip": base, "baseline_energy": e_ref,
            "echoes": echoes(seq, simulate), "peak_b1": float(b1_uT(seq).max()), "n_evals": info["n_evals"],
            "seconds": seconds}


def slsqp_on_epg(start):
    """The same problem solved by SciPy's SLSQP on the EPG model: an independent model and
    optimizer. Energy is proportional to sum(flip^2) at fixed pulse duration."""
    import numpy as np
    from scipy.optimize import minimize as scipy_minimize

    def cons(b):
        return epg_echoes(b).numpy() / TARGET.numpy() - 1

    r = scipy_minimize(lambda b: float(np.sum(b ** 2)), np.asarray(start, dtype=float), jac=lambda b: 2 * b,
                       method="SLSQP", bounds=[(0, 180)] * N_ECHO, constraints=[{"type": "ineq", "fun": cons}],
                       options={"ftol": 1e-14, "maxiter": 500})
    return torch.as_tensor(r.x)


def main():
    res = optimize()
    flips = res["flips"]
    print(f"T1 {T1} s, T2 {T2} s, ESP {ESP * 1e3:g} ms, {N_ECHO} echoes, target {FRACTION:g} x exp(-t/T2)")
    print(f"constant train: {res['baseline_flip']:.2f} deg, RF energy {res['baseline_energy']:.2f} µT²s")
    print(f"optimized train (fused, {res['n_evals']} evaluations, {res['seconds']:.1f} s):")
    print("  " + " ".join(f"{a:.1f}" for a in flips.tolist()))
    print(f"  RF energy {res['energy']:.2f} µT²s = {100 * res['energy'] / res['baseline_energy']:.1f} % of the "
          f"constant train; peak B1 {res['peak_b1']:.2f} µT")
    epg = epg_echoes(flips)
    print(f"  echoes / target, Bloch: min {float((res['echoes'] / TARGET).min()):.6f}; "
          f"EPG: min {float((epg / TARGET).min()):.6f}; Bloch vs EPG {float((res['echoes'] - epg).abs().max()):.1e}")
    ref = slsqp_on_epg([res["baseline_flip"]] * N_ECHO)  # started from the constant train
    print(f"SLSQP on EPG: energy {100 * float((ref ** 2).sum() / (flips ** 2).sum() - 1):+.1e} % vs ours, "
          f"max flip difference {float((ref - flips).abs().max()):.1e} deg")
    assert float((epg / TARGET).min()) > 1 - 1e-6
    assert float((ref - flips).abs().max()) < 0.05 and res["energy"] < res["baseline_energy"]

if __name__ == "__main__":
    main()
