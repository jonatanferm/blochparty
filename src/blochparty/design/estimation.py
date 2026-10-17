"""Parameter estimation: the Jacobian of a signal model (autograd or finite differences), the Fisher information and
the Cramér-Rao bound."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor



# --- Cramér-Rao bound -----------------------------------------------------------------------


def _as_real(x: Tensor) -> Tensor:
    return torch.view_as_real(x).flatten() if x.is_complex() else x.flatten()


def jacobian(signal_fn: Callable[[Tensor], Tensor], theta: Tensor, *, method: str = "autograd",
             rel_step: float = 1e-4) -> Tensor:
    """(M, P) Jacobian of ``signal_fn(theta)`` (complex outputs as stacked real / imaginary
    parts) w.r.t. the (P,) parameters ``theta``.

    ``method="autograd"`` differentiates through the simulator with ``create_graph=True``:
    the Jacobian is itself differentiable (w.r.t. sequence parameters, say) when the
    simulator supports double backward, as ``simulate`` does and ``simulate_metal``
    (first-order adjoint) does not. ``method="fd"`` uses central differences with step
    ``rel_step * |theta|``: every evaluation is an ordinary forward pass, so the result is
    differentiable through ``simulate_metal`` too (truncation error ~ rel_step^2).
    ``method="fd4"``: the fourth-order central stencil (truncation ~ rel_step^4), for float32
    simulators, whose rounding needs steps near 1e-2.
    """
    theta = theta.detach()
    if method == "autograd":
        th = theta.clone().requires_grad_(True)
        s = _as_real(signal_fn(th))
        rows = [torch.autograd.grad(s[m], th, retain_graph=True, create_graph=True)[0] for m in range(s.numel())]
        return torch.stack(rows)
    if method in ("fd", "fd4"):
        cols = []
        for p in range(theta.numel()):
            h = rel_step * max(abs(float(theta[p])), 1e-12)
            e = torch.zeros_like(theta)
            e[p] = h
            d = lambda k: _as_real(signal_fn(theta + k * e)) - _as_real(signal_fn(theta - k * e))
            cols.append(d(1) / (2 * h) if method == "fd" else (8 * d(1) - d(2)) / (12 * h))
        return torch.stack(cols, dim=1)
    raise ValueError(f"unknown method {method!r}")


def fisher_information(signal_fn, theta: Tensor, sigma=1.0, **kwargs) -> Tensor:
    """(P, P) Fisher information J^T J / sigma^2 for measurements with independent Gaussian
    noise of standard deviation ``sigma`` (per real and imaginary part for complex data)."""
    j = jacobian(signal_fn, theta, **kwargs)
    return j.T @ j / sigma ** 2


def crb(signal_fn, theta: Tensor, sigma=1.0, **kwargs) -> Tensor:
    """(P,) Cramér–Rao lower bounds on the variance of unbiased estimates of ``theta``:
    the diagonal of the inverse Fisher information (all parameters estimated jointly)."""
    return torch.diagonal(torch.linalg.inv(fisher_information(signal_fn, theta, sigma, **kwargs)))
