"""WM/GM contrast with flip angle *and* TR free, under a scan-time budget.

``optimize_contrast.py`` finds the flip angle maximizing white/grey-matter contrast at a
fixed TR of 10 ms (16.109°). Here TR is free too: longer TRs give more T1 contrast, so
without a limit the optimum drifts to TR ≈ 0.8 s. A budget on the total scan time
((dummies + lines) x TR) makes the constraint active, and the augmented-Lagrangian
optimizer (``blochparty.design.minimize``) has to find the optimum on its boundary. With a
budget of (400 + 32) x 10 ms that is again TR = 10 ms and 16.109°. Everything runs through
the full Bloch simulation and reconstruction (the core in float64; through the float32 Metal
adjoint the flip is found but the constraint only to ~1e-4) and is checked against a
brute-force grid over the closed-form steady state.

    uv run python examples/design_contrast_tr.py

A worked design problem: contrast per scan time (``examples/design_contrast_tr.py`` runs it).
"""

import math
import time
import warnings

import torch
import torch.nn.functional as F

import blochparty as bp
from blochparty.design import contrast, minimize
from blochparty.phantom import LABELS
from blochparty.reference import spoiled_gre_signal

# gre_cartesian checks float(TR) for a TR that requires grad; harmless here
warnings.filterwarnings("ignore", "Converting a tensor with requires_grad=True to a scalar")

N, FOV, TE, DWELL, N_DUMMY = 32, 0.22, 3e-3, 2e-6, 400
TR_MIN = 4e-3  # TE + readout + RF fit comfortably above this
SCALE = 0.01  # typical contrast, so the objective is O(1)

maps = bp.shepp_logan_maps(N)
phantom = bp.phantom_from_maps(maps["pd"], maps["t1"], maps["t2"], FOV)
wm, gm = maps["label"] == LABELS.index("wm"), maps["label"] == LABELS.index("gm")


def tr_of(u):
    """TR > TR_MIN for any real u (so no line-search step can produce an impossible TR)."""
    return TR_MIN + 1e-3 * F.softplus(u)


def sequence(flip, tr):
    return bp.gre_2d(N, FOV, flip_deg=flip, tr=tr, te=TE, dwell=DWELL, n_dummy=N_DUMMY)


def image_contrast(seq, simulate):
    img = bp.ifft2c(bp.kspace_grid(simulate(phantom, seq).signal, seq))[0]
    return contrast(img, wm, gm)


def optimize(budget, start=(40.0, 6e-3), simulate=None):
    """Maximize WM-GM contrast subject to scan time <= ``budget`` [s]."""
    if simulate is None:
        simulate = bp.simulate
    flip = torch.tensor(start[0], dtype=torch.float64, requires_grad=True)
    u0 = math.log(math.expm1((start[1] - TR_MIN) / 1e-3))  # softplus^-1
    u = torch.tensor(u0, dtype=torch.float64, requires_grad=True)
    state = {}

    def objective():
        state["seq"] = sequence(flip, tr_of(u))
        return -image_contrast(state["seq"], simulate) / SCALE

    def constraints():  # 1 - scan time / budget >= 0, on the sequence just simulated
        seq = state["seq"] if torch.is_grad_enabled() else sequence(flip, tr_of(u))
        return (1 - seq.duration / budget).reshape(1)

    t0 = time.perf_counter()
    info = minimize([flip, u], objective, constraints)
    seq = sequence(flip.detach(), tr_of(u.detach()))
    return {"flip": abs(flip.item()), "tr": tr_of(u).item(), "duration": float(seq.duration),
            "contrast": float(image_contrast(seq, simulate)), "n_evals": info["n_evals"],
            "seconds": time.perf_counter() - t0}


def grid_optimum(budget):
    """Brute force over the closed-form steady state: TR on a 0.01 ms grid up to the budget,
    flip on a 0.01 deg grid, then 0.001 deg around the best."""
    tr_max = budget / (N_DUMMY + N)
    props = {name: (maps["pd"][m][0], maps["t1"][m][0], maps["t2"][m][0]) for name, m in (("wm", wm), ("gm", gm))}
    c = lambda tr, a: (spoiled_gre_signal(*props["wm"], tr, TE, torch.deg2rad(a))
                       - spoiled_gre_signal(*props["gm"], tr, TE, torch.deg2rad(a)))
    trs = torch.arange(TR_MIN, tr_max + 1e-12, 1e-5, dtype=torch.float64)
    trs = torch.cat([trs, torch.tensor([tr_max], dtype=torch.float64)])  # the boundary itself
    angles = torch.arange(1, 9001, dtype=torch.float64) * 0.01
    vals = c(trs[:, None], angles[None, :])
    i, j = divmod(int(vals.argmax()), len(angles))
    fine = angles[j] + torch.arange(-100, 101, dtype=torch.float64) * 1e-4
    k = int(c(trs[i], fine).argmax())
    return float(fine[k]), float(trs[i]), float(c(trs[i], fine[k]))
