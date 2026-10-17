"""Differentiable sequence design, verified: find the flip angle that maximizes
white/grey-matter contrast in a simulated 2D spoiled GRE image by gradient
descent *through the full Bloch simulation and reconstruction*, then compare
with a brute-force search over the closed-form steady-state signal.

    uv run python examples/optimize_contrast.py
    uv run python examples/optimize_contrast.py --gpu   # through the GPU adjoint (Metal or CUDA; float32)
"""

import functools
import math
import sys
import time

import torch

import blochparty as bp
from blochparty.phantom import LABELS
from blochparty.reference import spoiled_gre_signal

N, FOV, TR, TE = 32, 0.22, 10e-3, 3e-3

maps = bp.shepp_logan_maps(N)
phantom = bp.phantom_from_maps(maps["pd"], maps["t1"], maps["t2"], FOV)
wm, gm = maps["label"] == LABELS.index("wm"), maps["label"] == LABELS.index("gm")
flip = torch.tensor(40.0, dtype=torch.float64, requires_grad=True)
if {"--gpu", "--metal"} & set(sys.argv[1:]):  # the GPU adjoint (Metal or CUDA)
    simulate = functools.partial(bp.simulate, engine="gpu")
else:
    simulate = bp.simulate


def contrast():
    seq = bp.gre_2d(N, FOV, flip_deg=flip, tr=TR, te=TE, dwell=2e-6, n_dummy=400)
    img = bp.ifft2c(bp.kspace_grid(simulate(phantom, seq).signal, seq))[0].abs()
    return img[wm].mean() - img[gm].mean()


opt = torch.optim.LBFGS([flip], lr=1, max_iter=30, line_search_fn="strong_wolfe")
n_evals = 0


def closure():
    global n_evals
    n_evals += 1
    opt.zero_grad()
    loss = -contrast()
    loss.backward()
    print(f"  eval {n_evals:2d}: flip {flip.item():7.3f} deg, contrast {-loss.item():.5f}")
    return loss


t0 = time.perf_counter()
opt.step(closure)
print(f"gradient descent: {flip.item():.3f} deg ({n_evals} evaluations, {time.perf_counter() - t0:.1f} s)")

# brute force on the analytic steady state
angles = torch.linspace(1, 90, 89001, dtype=torch.float64)
props = {name: (maps["pd"][m][0], maps["t1"][m][0], maps["t2"][m][0]) for name, m in (("wm", wm), ("gm", gm))}
c = (spoiled_gre_signal(*props["wm"], TR, TE, torch.deg2rad(angles))
     - spoiled_gre_signal(*props["gm"], TR, TE, torch.deg2rad(angles)))
best = angles[c.argmax()].item()
# +a and -a give identical magnitude images, so the optimizer may land on either
found = abs(flip.item())
print(f"analytic optimum: {best:.3f} deg  ->  difference {abs(found - best):.3f} deg")
assert math.isclose(found, best, abs_tol=0.05)
