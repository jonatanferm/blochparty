"""Constrained optimisation of sequence parameters (``minimize``: projected gradients with penalties)."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

# --- constrained optimisation ---------------------------------------------------------------


def minimize(params: list[Tensor], objective: Callable[[], Tensor],
             constraints: Callable[[], Tensor] | None = None, *, rho: float = 10.0, outer: int = 20,
             inner: int = 50, tol: float = 1e-7, callback=None) -> dict:
    """Minimise ``objective()`` subject to ``constraints() >= 0`` (a 1-D tensor, ideally
    normalised, e.g. 1 - value / limit) with the augmented Lagrangian method: L-BFGS on
    f + sum psi(g, lambda), psi = (relu(lambda - rho g)^2 - lambda^2) / (2 rho), then
    lambda <- relu(lambda - rho g), until the worst violation and the change in lambda are
    below ``tol``. Unlike a fixed quadratic penalty, the result satisfies active
    constraints exactly at convergence. Returns a dict with the multipliers and counts."""
    lam = None
    n_evals = 0

    def lagrangian():
        nonlocal n_evals
        n_evals += 1
        f = objective()
        if constraints is None:
            return f
        g = constraints()
        lm = torch.zeros_like(g) if lam is None else lam
        return f + ((torch.relu(lm - rho * g) ** 2 - lm ** 2) / (2 * rho)).sum()

    for it in range(outer if constraints is not None else 1):
        opt = torch.optim.LBFGS(params, lr=1, max_iter=inner, tolerance_grad=1e-12, tolerance_change=1e-15,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = lagrangian()
            loss.backward()
            return loss

        opt.step(closure)
        if constraints is None:
            break
        with torch.no_grad():
            g = constraints()
            new = torch.relu((torch.zeros_like(g) if lam is None else lam) - rho * g)
            change = float((new - (torch.zeros_like(g) if lam is None else lam)).abs().max())
            lam = new
            worst = float(torch.relu(-g).max())
        if callback is not None:
            callback(it, worst, lam)
        if worst < tol and change < tol:
            break
    return {"multipliers": lam, "n_evals": n_evals, "outer_iterations": it + 1}
