"""Variable-flip-angle (DESPOT1) T1 mapping, designed by minimising a Cramér–Rao bound.

Two spoiled-GRE steady states at flip angles a1, a2 (same TR) determine M0 and T1. The
bound on the variance of the T1 estimate comes from the Jacobian of the *simulated*
signals w.r.t. (M0, T1) (``blochparty.design.crb``), and gradient descent through the fused
kernel moves the two flip angles to minimise it. Known result (Deoni, Rutt & Peters, MRM
49:515, 2003): at the optimum both signals are ≈ 71 % of the Ernst-angle maximum. Checked
here against a brute-force grid search over the closed-form signal and its exact derivative.

    uv run python examples/design_despot1.py [--metal]
"""

import math
import time

import torch

from blochparty import Phantom, Sequence, SequenceBuilder
from blochparty.design import crb, minimize

T1, T2, TR, TE = 1.0, 0.05, 15e-3, 2e-3
RF_DUR, DWELL = 100e-6, 2e-6  # one segment per pulse: the instantaneous rotation of the closed form
# dummy pulses until (E1 cos a)^n < 1e-12 for any flip angle: the simulated signal is the steady state
N_DUMMY = math.ceil(math.log(1e-12) / (-TR / T1))
f64 = lambda v: torch.as_tensor(v, dtype=torch.float64)
METAL_STEP = 5e-2  # fd4 step for float32 simulators: 2e-2 to 1e-1 give the pair to <= 0.03 deg


def spgr_sequence(flip_deg, n_dummy=N_DUMMY):
    """Spoiled GRE driven to steady state, then one ADC sample at TE: ``n_dummy`` copies
    of one spoiled TR (built once and repeated, which is much faster than building them all)."""
    tr = SequenceBuilder().rf(RF_DUR, flip_deg=flip_deg, n_steps=1).delay(TR - RF_DUR).spoil().build()
    echo = SequenceBuilder().rf(RF_DUR, flip_deg=flip_deg, n_steps=1)
    echo = echo.delay(TE - RF_DUR / 2 - DWELL / 2).readout(1, DWELL).build()
    def cat(name):
        a, b = getattr(tr, name), getattr(echo, name)
        return a * n_dummy + b if isinstance(a, list) else torch.cat([a.repeat(n_dummy, *[1] * (a.ndim - 1)), b])

    return Sequence(**{f: cat(f) for f in ("dt", "rf", "grad", "adc_phase", "is_rf", "adc", "spoil")})


def signal_fn(seqs, simulate):
    """theta = (M0, T1) -> the complex signals of the acquisitions ``seqs``."""
    def f(theta):
        spin = Phantom(pos=torch.zeros(1, 3, dtype=torch.float64), pd=theta[:1], t1=theta[1:2],
                       t2=f64([T2]), db0=f64([0.0]))
        return torch.cat([simulate(spin, s).signal.flatten() for s in seqs])
    return f


def t1_crb(flips, simulate, *, method="fd", n_dummy=N_DUMMY, rel_step=1e-4):
    """sqrt(CRB) of T1 / T1 for unit noise and M0 = 1 (the T1 coefficient of variation per
    unit noise), both flip angles in one tensor (degrees). ``rel_step``: the finite-difference
    step of the Jacobian (float32 simulators need ~1e-2: their signals carry ~1e-6 of rounding)."""
    seqs = [spgr_sequence(a, n_dummy) for a in flips.unbind()]
    kw = {"rel_step": rel_step} if method in ("fd", "fd4") else {}
    var = crb(signal_fn(seqs, simulate), f64([1.0, T1]), method=method, **kw)
    return torch.sqrt(var[1]) / T1


def optimize(start=(10.0, 40.0), simulate=None, rel_step=1e-4, method="fd"):
    if simulate is None:
        from blochparty.fused import simulate_fused as simulate
    flips = f64(start).clone().requires_grad_(True)
    t0 = time.perf_counter()
    info = minimize([flips], lambda: t1_crb(flips, simulate, method=method, rel_step=rel_step))
    found = sorted(abs(a) for a in flips.tolist())
    cov = float(t1_crb(f64(found), simulate, method=method, rel_step=rel_step))
    return {"flips": found, "cov": cov, "n_evals": info["n_evals"],
            "seconds": time.perf_counter() - t0}


# --- independent reference: closed-form signal, exact derivative, brute-force grid -----------


def closed_form_cov(a1, a2):
    """sqrt(CRB(T1)) / T1 for flip angles a1, a2 [rad] (broadcasting), from
    S = M0 sin a (1 - E1) / (1 - E1 cos a) and its exact derivatives."""
    e1 = math.exp(-TR / T1)
    decay = math.exp(-TE / T2)

    def s_and_ds(a):
        s = torch.sin(a) * (1 - e1) / (1 - torch.cos(a) * e1) * decay  # dS/dM0 at M0 = 1
        ds = torch.sin(a) * (torch.cos(a) - 1) / (1 - torch.cos(a) * e1) ** 2 * e1 * TR / T1 ** 2 * decay
        return s, ds

    (s1, d1), (s2, d2) = s_and_ds(a1), s_and_ds(a2)
    det = s1 * d2 - s2 * d1  # J = [[s1, d1], [s2, d2]]; (J^T J)^-1_{T1,T1} = (s1^2 + s2^2) / det^2
    return torch.sqrt((s1 ** 2 + s2 ** 2) / det ** 2) / T1


def _grid_best(ax1, ax2):
    a1, a2 = torch.meshgrid(ax1, ax2, indexing="ij")
    cov = closed_form_cov(torch.deg2rad(a1), torch.deg2rad(a2))
    i = int(torch.where(torch.isfinite(cov), cov, torch.full_like(cov, math.inf)).argmin())
    return sorted([float(a1.flatten()[i]), float(a2.flatten()[i])])


def grid_optimum():
    """Brute force: a 0.1 deg grid over 0.1..90 deg squared, then 0.001 deg around its best."""
    coarse = torch.arange(1, 901, dtype=torch.float64) * 0.1
    b1, b2 = _grid_best(coarse, coarse)
    fine = torch.arange(-200, 201, dtype=torch.float64) * 0.001
    best = _grid_best(b1 + fine, b2 + fine)
    return best, float(closed_form_cov(*torch.deg2rad(f64(best))))


def ernst_fraction(flip_deg):
    """Steady-state signal at ``flip_deg`` relative to the Ernst-angle maximum."""
    e1 = math.exp(-TR / T1)
    s = lambda a: math.sin(a) * (1 - e1) / (1 - math.cos(a) * e1)
    return s(math.radians(flip_deg)) / s(math.acos(e1))


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--metal", action="store_true", help="through the GPU adjoint (float32) instead of the fused kernel")
    args = p.parse_args()
    if args.metal:  # float32: fourth-order differences with a step large against the signals' rounding
        from blochparty.metal import simulate_metal

        res, engine = optimize(simulate=simulate_metal, rel_step=METAL_STEP, method="fd4"), "Metal adjoint"
    else:
        res, engine = optimize(), "fused kernel"
    print(f"T1 {T1} s, TR {TR * 1e3:g} ms, Ernst angle {math.degrees(math.acos(math.exp(-TR / T1))):.3f} deg")
    print(f"gradient descent ({engine}): flips {res['flips'][0]:.3f}, {res['flips'][1]:.3f} deg, "
          f"CoV(T1)·SNR0 {res['cov']:.4f} ({res['n_evals']} evaluations, {res['seconds']:.1f} s)")
    (g1, g2), gcov = grid_optimum()
    print(f"closed-form grid:                flips {g1:.3f}, {g2:.3f} deg, CoV(T1)·SNR0 {gcov:.4f}")
    excess = float(closed_form_cov(*torch.deg2rad(f64(res["flips"])))) / gcov - 1
    print(f"closed-form CoV at the found flips: {excess:.1e} above the optimum")
    fr = [ernst_fraction(a) for a in res["flips"]]
    print(f"signals at {100 * fr[0]:.2f} % and {100 * fr[1]:.2f} % of the Ernst maximum (Deoni 2003: ≈ 71 %)")
    # the bound is flat along a2: float32 rounding moves the found pair by up to 0.03 deg, at < 1e-5 of the bound
    assert max(abs(res["flips"][0] - g1), abs(res["flips"][1] - g2)) < (0.05 if args.metal else 0.01)
    assert excess < 1e-5
    assert all(abs(f - 0.71) < 0.01 for f in fr)


if __name__ == "__main__":
    main()
