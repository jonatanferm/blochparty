"""Sequence design: hardware / safety constraints, image-quality metrics and losses, and a
machine-readable design report.

Everything here is a differentiable function of a ``Sequence``'s tensors (or of simulated
images), so it can sit in a loss next to the simulator:

* constraints: gradient amplitude, slew rate, peak B1, RF energy / SAR proxy, a PNS
  estimate and scan time, each with a smooth penalty (``penalties``) and a report entry
  that names every violation by segment range (``check_hardware``);
* metrics: tissue contrast, CNR, SNR efficiency, PSF width / sidelobes from a simulated
  point spin, and the Cramér–Rao bound of tissue-parameter estimates (``crb``);
* ``minimize``: an augmented-Lagrangian wrapper around L-BFGS for constrained designs;
* ``design_report``: timing, every constraint with its margin, and image metrics, as a
  JSON-serialisable dict meant to be read and acted on by an agent.

Units follow the rest of the package internally (RF and gradients in Hz and Hz/m); limits
and reported values use scanner units: mT/m, T/m/s, µT. Segment boundary j is the start of
segment j (j = 0..S, boundary S is the end of the sequence); the gradient is zero before
the first and after the last segment, so a sequence must ramp up and down within itself.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import torch
from torch import Tensor

from .coils import add_noise
from .phantom import Phantom
from .recon import reconstruct, rss
from .sequence import Sequence

GAMMA_BAR = 42.577478518e6  # proton gyromagnetic ratio / 2 pi [Hz/T]
AXES = "xyz"


@dataclass
class Hardware:
    """Scanner limits. ``None`` disables a limit (the value is still reported).

    The PNS model parameters are generic whole-body values, not those of any particular
    gradient coil; see ``pns``.
    """

    max_grad: float = 40.0  # per axis [mT/m]
    max_slew: float = 150.0  # per axis [T/m/s]
    grad_raster: float = 10e-6  # longest time a gradient step may take [s] (see ``slew_rate``)
    max_b1: float = 20.0  # peak RF amplitude [µT]
    max_b1_rms: float | None = None  # RMS B1 over ``sar_window`` [µT]
    sar_scale: float = 1.0  # SAR proxy = sar_scale * mean |B1|^2 [sar_unit per µT^2]
    sar_unit: str = "µT²"
    max_sar: float | None = None  # [sar_unit], over ``sar_window``
    sar_window: float | None = None  # averaging window [s]; None: the whole sequence
    pns_rheobase: float = 100.0  # [T/m/s]: IEC 60601-2-33's 20 T/s dB/dt rheobase at 0.2 m from isocentre
    pns_chronaxie: float = 360e-6  # [s]
    max_pns: float | None = 0.8  # fraction of the stimulation threshold (IEC normal mode)
    max_duration: float | None = None  # total scan time [s]


# --- hardware quantities ----------------------------------------------------------------


def grad_mT_m(seq: Sequence) -> Tensor:
    """(S, 3) gradient per segment [mT/m]."""
    return seq.grad / GAMMA_BAR * 1e3


def boundary_times(seq: Sequence) -> Tensor:
    """(S + 1,) start time of every segment, plus the end of the sequence [s]."""
    return torch.cat([seq.dt.new_zeros(1), torch.cumsum(seq.dt, 0)])


def gradient_steps(seq: Sequence) -> Tensor:
    """(S + 1, 3) gradient change at every segment boundary [T/m], from zero before the
    first segment and back to zero after the last."""
    g = seq.grad / GAMMA_BAR
    zero = g.new_zeros(1, 3)
    return torch.diff(torch.cat([zero, g, zero]), dim=0)


def slew_rate(seq: Sequence, raster: float = 10e-6) -> Tensor:
    """(S + 1, 3) slew rate at every segment boundary [T/m/s]: the gradient step divided by
    the time over which it can happen, the distance between the centres of the two segments
    it separates but at most one gradient ``raster`` time.

    A gradient waveform sampled as segment averages on the raster (``read_pulseq`` cuts
    ramps at the gradient raster) gives the slew of the underlying waveform. A ramp must be
    represented by segments no longer than the raster: an instantaneous step between two long
    segments (today's builders) counts as happening within one raster time, so it shows up as
    the (far too high) slew it would need, rather than hidden by the long segments' spacing.
    """
    return gradient_steps(seq) / step_durations(seq, raster)[:, None]


def step_durations(seq: Sequence, raster: float = 10e-6) -> Tensor:
    """(S + 1,) time each gradient step is taken to ramp over (see ``slew_rate``) [s]."""
    pad = torch.cat([seq.dt.new_zeros(1), seq.dt, seq.dt.new_zeros(1)])
    return (0.5 * (pad[:-1] + pad[1:])).clamp(1e-12, raster)


def b1_uT(seq: Sequence) -> Tensor:
    """(S,) RF amplitude per segment [µT]."""
    return torch.sqrt((seq.rf ** 2).sum(-1) + 1e-30) / GAMMA_BAR * 1e6


def rf_energy(seq: Sequence) -> Tensor:
    """Sum of |B1|^2 dt over the sequence [µT^2 s], the quantity SAR is proportional to."""
    b1sq = (seq.rf ** 2).sum(-1) * (1e6 / GAMMA_BAR) ** 2
    return (b1sq * seq.dt).sum()


def rf_power(seq: Sequence, window: float | None = None) -> Tensor:
    """Mean |B1|^2 [µT^2]: over the whole sequence (``window=None``, a scalar), or (S,)
    over windows of ``window`` seconds starting at every segment (windows reaching past
    the end are averaged over what remains, but never over less than ``window``)."""
    b1sq = (seq.rf ** 2).sum(-1) * (1e6 / GAMMA_BAR) ** 2
    if window is None:
        return (b1sq * seq.dt).sum() / seq.dt.sum()
    t = boundary_times(seq)
    e = torch.cat([b1sq.new_zeros(1), torch.cumsum(b1sq * seq.dt, 0)])  # energy, linear in t per segment
    t_end = t[:-1] + window
    i = torch.searchsorted(t.detach(), t_end.detach()).clamp(1, len(seq))
    frac = ((t_end - t[i - 1]) / seq.dt[i - 1].clamp_min(1e-30)).clamp(0.0, 1.0)
    e_end = e[i - 1] + frac * (e[i] - e[i - 1])
    return (e_end - e[:-1]) / window


def sar_proxy(seq: Sequence, scale: float = 1.0, window: float | None = None) -> Tensor:
    """SAR proxy: ``scale`` times the mean |B1|^2 (see ``rf_power``)."""
    return scale * rf_power(seq, window)


def pns(seq: Sequence, rheobase: float = 100.0, chronaxie: float = 360e-6, raster: float = 10e-6,
        window: float = 20e-3) -> tuple[Tensor, Tensor]:
    """Approximate peripheral-nerve-stimulation estimate, as a fraction of the threshold.

    A chronaxie-filtered dG/dt model: per axis the nerve response is

        R(t) = 1/rheobase * integral dG/dt(s) * c / (c + t - s)^2 ds,      c = chronaxie,

    and stimulation is expected at R = 1. For a single ramp of slew S lasting tau this gives
    R = S/rheobase * tau / (c + tau), i.e. the hyperbolic strength-duration law
    S_threshold = rheobase * (1 + c / tau) of IEC 60601-2-33 (there for dB/dt; here for the
    slew, dB/dt divided by the distance from isocentre); a step of dG taken within one raster
    time gives R ~ dG / (c * rheobase) (36 mT/m at threshold by default). Axes combine as
    the root sum of squares.

    Every gradient step is the linear ramp ``slew_rate`` assumes (over ``step_durations``,
    centred on its boundary), and the integral over each ramp is exact, so ramps sampled on
    the raster give the continuous-waveform result. R is evaluated at the end of every ramp,
    where it peaks for monotone waveforms; ramps ending more than ``window`` earlier are
    ignored (they contribute at most c / (c + window), ~2 % of a sustained slew, and cancel
    for oscillating ones).

    This is not a coil-specific model (e.g. Siemens SAFE): rheobase and chronaxie depend on
    the gradient coil and body region, and the result is an estimate, not a safety check.

    Returns (values (J,), segment boundary index (J,)) at the J boundaries with a step.
    """
    dg = gradient_steps(seq)
    idx = torch.nonzero((dg.detach() != 0).any(-1)).flatten()
    if idx.numel() == 0:
        return dg.new_zeros(0), idx
    tj, tau = boundary_times(seq)[idx], step_durations(seq, raster)[idx]
    a, b, slope = tj - tau / 2, tj + tau / 2, dg[idx] / tau[:, None]  # ramp start, end, slew [T/m/s]
    kern = lambda lag: chronaxie / (chronaxie + lag)  # integral of c / (c + lag)^2 d(lag), up to a constant
    tj_d = tj.detach()
    out, k0 = [], 0
    while k0 < len(idx):  # evaluate at b[k] for a block of k, against every ramp that can reach it
        k1 = min(len(idx), k0 + 512)
        j0 = int(torch.searchsorted(tj_d, tj_d[k0] - window - raster))
        j1 = int(torch.searchsorted(tj_d, tj_d[k1 - 1] + raster, right=True))
        t = b[k0:k1, None]
        aj, bj = a[None, j0:j1], b[None, j0:j1]
        live = (aj < t) & (t - bj < window)
        upper = torch.minimum(t, bj)  # ramps still running at t count up to t
        w = torch.where(live, kern((t - upper).clamp_min(0)) - kern((t - aj).clamp_min(0)), torch.zeros_like(t - aj))
        out.append(w @ slope[j0:j1])
        k0 = k1
    r = torch.cat(out) / rheobase  # (J, 3)
    return torch.sqrt((r ** 2).sum(-1) + 1e-30), idx


# --- constraint checks, penalties and violation reports ------------------------------------


@dataclass
class Check:
    """One constraint evaluated over a sequence: ``values`` (n,) at segments ``segments``
    (n,) (boundary indices for slew and PNS), against ``limit`` (None: report only)."""

    name: str
    values: Tensor
    segments: Tensor
    limit: float | None
    unit: str
    where: str = "segments"  # what ``segments`` index: "segments", "boundaries" or "sequence"


def check_hardware(seq: Sequence, hw: Hardware | None = None) -> list[Check]:
    """Every constraint of ``hw`` evaluated on ``seq`` (values keep their autograd graph)."""
    hw = hw or Hardware()
    s = len(seq)
    segs = torch.arange(s)
    bounds = torch.arange(s + 1)
    g, slew = grad_mT_m(seq).abs(), slew_rate(seq, hw.grad_raster).abs()
    checks = [Check(f"gradient {a}", g[:, i], segs, hw.max_grad, "mT/m") for i, a in enumerate(AXES)]
    checks += [Check(f"slew {a}", slew[:, i], bounds, hw.max_slew, "T/m/s", "boundaries")
               for i, a in enumerate(AXES)]
    rf_segs = torch.tensor([i for i in range(s) if seq.is_rf[i]], dtype=torch.long)
    checks.append(Check("peak B1", b1_uT(seq)[rf_segs], rf_segs, hw.max_b1, "µT"))
    power = rf_power(seq, hw.sar_window)
    if hw.sar_window is None:
        win, where = torch.zeros(1, dtype=torch.long), "sequence"
        power = power.reshape(1)
    else:
        win, where = segs, "segments"
    checks.append(Check("B1 rms", torch.sqrt(power + 1e-30), win, hw.max_b1_rms, "µT", where))
    checks.append(Check("SAR proxy", hw.sar_scale * power, win, hw.max_sar, hw.sar_unit, where))
    r, at = pns(seq, hw.pns_rheobase, hw.pns_chronaxie, hw.grad_raster)
    checks.append(Check("PNS", r, at, hw.max_pns, "× threshold", "boundaries"))
    checks.append(Check("scan time", seq.duration.reshape(1), torch.zeros(1, dtype=torch.long),
                        hw.max_duration, "s", "sequence"))
    return checks


def hinge2(values: Tensor, limit: float) -> Tensor:
    """Smooth (C1) penalty: sum of squared relative excess over ``limit``, zero inside."""
    return (torch.relu(values / limit - 1.0) ** 2).sum()


def penalties(seq: Sequence, hw: Hardware | None = None, only: tuple[str, ...] | None = None
              ) -> dict[str, Tensor]:
    """``hinge2`` penalty per limited constraint (names as in ``check_hardware``; ``only``
    keeps those whose name starts with one of the given prefixes)."""
    out = {}
    for c in check_hardware(seq, hw):
        if c.limit is not None and (only is None or c.name.startswith(only)):
            out[c.name] = hinge2(c.values, c.limit)
    return out


def _runs(index: list[int]) -> list[tuple[int, int]]:
    """Consecutive runs [(first, last), ...] of sorted indices."""
    runs = []
    for i in index:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [tuple(r) for r in runs]


def violations(check: Check, seq: Sequence, max_items: int = 10) -> list[dict]:
    """Segment ranges where ``check`` exceeds its limit, worst first, with a message such
    as "slew x exceeded in segments 1200–1210 (t = 12.000–12.100 ms) by 12 %"."""
    if check.limit is None:
        return []
    v = check.values.detach().double()
    over = torch.nonzero(v > check.limit).flatten().tolist()
    if not over:
        return []
    seg = check.segments.tolist()
    t = boundary_times(seq).detach().double()
    # group by consecutive segment index (adjacent boundaries / segments form one run)
    pos = {seg[i]: i for i in over}
    items = []
    for a, b in _runs(sorted(pos)):
        worst = max(float(v[pos[j]]) for j in range(a, b + 1) if j in pos)
        excess = 100.0 * (worst / check.limit - 1.0)
        if check.where == "sequence":
            span, times = "", [0.0, 1e3 * float(t[-1])]
        else:
            end = b + 1 if check.where == "segments" else b  # a segment ends at the next boundary
            times = [1e3 * float(t[a]), 1e3 * float(t[end])]
            one, many = ("segment", "segments") if check.where == "segments" else ("boundary", "boundaries")
            span = f" at {one} {a}" if a == b else f" in {many} {a}–{b}"
            span += (f" (t = {times[0]:.3f} ms)" if times[0] == times[1]
                     else f" (t = {times[0]:.3f}–{times[1]:.3f} ms)")
        items.append({
            "constraint": check.name, "segments": [a, b], "index": check.where, "time_ms": times, "value": worst,
            "limit": check.limit, "unit": check.unit, "excess_pct": excess,
            "message": f"{check.name} exceeded{span} by {excess:.3g} % "
                       f"({worst:.4g} {check.unit} > {check.limit:.4g} {check.unit})",
        })
    items.sort(key=lambda d: -d["excess_pct"])
    return items[:max_items] + ([{"constraint": check.name, "message":
                                   f"{check.name}: {len(items) - max_items} more violating ranges not listed",
                                   "n_more": len(items) - max_items}] if len(items) > max_items else [])


# --- image-quality metrics ------------------------------------------------------------------


def tissue_mean(image: Tensor, mask: Tensor) -> Tensor:
    """Mean magnitude of ``image`` over ``mask``."""
    return image.abs()[mask].mean()


def contrast(image: Tensor, mask_a: Tensor, mask_b: Tensor) -> Tensor:
    """Mean magnitude in ``mask_a`` minus that in ``mask_b``."""
    return tissue_mean(image, mask_a) - tissue_mean(image, mask_b)


def cnr(image: Tensor, mask_a: Tensor, mask_b: Tensor, sigma) -> Tensor:
    """Contrast-to-noise ratio, with ``sigma`` the image-domain noise standard deviation
    (given, or from ``image_noise_sigma``)."""
    return contrast(image, mask_a, mask_b) / sigma


def snr(image: Tensor, mask: Tensor, sigma) -> Tensor:
    return tissue_mean(image, mask) / sigma


def snr_efficiency(image: Tensor, mask: Tensor, sigma, duration) -> Tensor:
    """SNR per square root of scan time [1/sqrt(s)]."""
    return snr(image, mask, sigma) / torch.sqrt(torch.as_tensor(duration))


def image_noise_sigma(signal: Tensor, recon: Callable[[Tensor], Tensor], kspace_sigma: float,
                      noise_cov: Tensor | None = None, mask: Tensor | None = None, n_replicas: int = 16,
                      generator: torch.Generator | None = None) -> Tensor:
    """Image-domain noise standard deviation by the pseudo-replica method: ``recon`` of
    (C, n_adc) ``signal`` plus k-space noise of per-channel ``kspace_sigma`` (and channel
    covariance ``noise_cov``, as ``coils.add_noise``), minus the noise-free image; the
    per-pixel standard deviation over replicas, RMS-averaged over ``mask``. Works for any
    reconstruction, including nonlinear ones such as RSS, at the actual signal level."""
    with torch.no_grad():
        clean = recon(signal)
        diffs = torch.stack([recon(add_noise(signal, kspace_sigma, noise_cov, generator)) - clean
                             for _ in range(n_replicas)])
        if diffs.is_complex():
            diffs = diffs.real  # one component of complex noise, the same sigma as the magnitude at high SNR
        var = diffs.var(dim=0)
        return torch.sqrt(var[mask].mean() if mask is not None else var.mean())


def magnitude_image(signal: Tensor, seq: Sequence) -> Tensor:
    """Cartesian reconstruction: magnitude for one coil, root-sum-of-squares for several."""
    images = reconstruct(signal, seq)
    return images[0].abs() if images.shape[0] == 1 else rss(images)


# --- point-spread function ------------------------------------------------------------------


def _default_simulate():
    from .fused import simulate_fused

    return simulate_fused


def point_spread_function(seq: Sequence, *, t1=1.0, t2=0.1, pd=1.0, db0=0.0, pos=(0.0, 0.0, 0.0),
                          oversample: int = 4, simulate=None) -> Tensor:
    """Magnitude PSF of a Cartesian ``seq``: simulate one spin at ``pos``, zero-pad k-space
    by ``oversample`` on every axis, reconstruct; normalised to its peak. Shape
    ``oversample * kspace_shape``; pixel ``oversample * n/2`` (per axis) is the grid centre."""
    simulate = simulate or _default_simulate()
    dtype = seq.dt.dtype
    f = lambda v: torch.as_tensor(v, dtype=dtype).reshape(1)
    spin = Phantom(pos=torch.as_tensor(pos, dtype=dtype).reshape(1, 3), pd=f(pd), t1=f(t1), t2=f(t2), db0=f(db0))
    k = simulate(spin, seq).signal[0].reshape(seq.meta["kspace_shape"])
    pads = []
    for n in reversed(k.shape):  # F.pad wants the last axis first
        before = (oversample * n) // 2 - n // 2
        pads += [before, oversample * n - n - before]
    big = torch.nn.functional.pad(torch.view_as_real(k), [0, 0] + pads)
    img = _ifft_centred(torch.view_as_complex(big.contiguous())).abs()
    return img / img.max()


def _ifft_centred(k: Tensor) -> Tensor:
    dims = tuple(range(k.ndim))
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(k, dim=dims), dim=dims), dim=dims)


def psf_metrics(psf: Tensor, oversample: int = 4) -> dict[str, Tensor]:
    """Per-axis FWHM [original pixels] and peak sidelobe level (largest value outside the
    main lobe, relative to the peak) of the profiles through the PSF's maximum. The FWHM
    interpolates linearly between samples, so it is differentiable in the PSF values.
    Axes are named as in the array order (z, y, x)."""
    peak = torch.unravel_index(torch.argmax(psf.detach()), psf.shape)
    out = {}
    names = AXES[: psf.ndim][::-1]
    for ax in range(psf.ndim):
        index = list(peak)
        index[ax] = slice(None)
        p = psf[tuple(index)]
        p = p / p[peak[ax]]
        c = int(peak[ax])
        half = []
        for step in (1, -1):  # walk out to the half-maximum crossing, interpolate linearly
            i = c
            while 0 <= i + step < len(p) and p[i + step] >= 0.5:
                i += step
            if not 0 <= i + step < len(p):
                half.append(p.new_tensor(float(abs(i - c))))
            else:
                half.append(abs(i - c) + (p[i] - 0.5) / (p[i] - p[i + step]))
        lobe = []  # main lobe: out to the first minimum on each side
        for step in (1, -1):
            i = c
            while 0 <= i + step < len(p) and p[i + step] <= p[i]:
                i += step
            lobe.append(i)
        lo, hi = min(lobe), max(lobe)
        outside = torch.cat([p[:lo], p[hi + 1:]])
        out[f"fwhm_{names[ax]}"] = (half[0] + half[1]) / oversample
        out[f"sidelobe_{names[ax]}"] = outside.max() if outside.numel() else p.new_zeros(())
    return out


# --- Cramér-Rao bound -----------------------------------------------------------------------


def _as_real(x: Tensor) -> Tensor:
    return torch.view_as_real(x).flatten() if x.is_complex() else x.flatten()


def jacobian(signal_fn: Callable[[Tensor], Tensor], theta: Tensor, *, method: str = "autograd",
             rel_step: float = 1e-4) -> Tensor:
    """(M, P) Jacobian of ``signal_fn(theta)`` (complex outputs as stacked real / imaginary
    parts) w.r.t. the (P,) parameters ``theta``.

    ``method="autograd"`` differentiates through the simulator with ``create_graph=True``:
    the Jacobian is itself differentiable (w.r.t. sequence parameters, say) when the
    simulator supports double backward, as ``simulate`` does and ``simulate_fused``
    (first-order adjoint) does not. ``method="fd"`` uses central differences with step
    ``rel_step * |theta|``: every evaluation is an ordinary forward pass, so the result is
    differentiable through ``simulate_fused`` too (truncation error ~ rel_step^2).
    """
    theta = theta.detach()
    if method == "autograd":
        th = theta.clone().requires_grad_(True)
        s = _as_real(signal_fn(th))
        rows = [torch.autograd.grad(s[m], th, retain_graph=True, create_graph=True)[0] for m in range(s.numel())]
        return torch.stack(rows)
    if method == "fd":
        cols = []
        for p in range(theta.numel()):
            h = rel_step * max(abs(float(theta[p])), 1e-12)
            e = torch.zeros_like(theta)
            e[p] = h
            cols.append((_as_real(signal_fn(theta + e)) - _as_real(signal_fn(theta - e))) / (2 * h))
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


# --- design report -------------------------------------------------------------------------


def _num(x):
    if torch.is_tensor(x):
        return float(x.detach()) if x.numel() == 1 else [float(v) for v in x.detach().flatten()]
    if isinstance(x, (int, float, bool, str)) or x is None:
        return x
    if isinstance(x, (tuple, list)):
        return [_num(v) for v in x]
    return str(x)


def design_report(seq: Sequence, phantom: Phantom | None = None, *, hardware: Hardware | None = None,
                  coils: Tensor | None = None, masks: dict[str, Tensor] | None = None,
                  contrast_pairs: list[tuple[str, str]] = (), noise_sigma: float | None = None,
                  kspace_sigma: float | None = None, noise_cov: Tensor | None = None, psf: bool = True,
                  psf_tissue: tuple[float, float] = (1.0, 0.1), simulate=None, max_items: int = 10) -> dict:
    """Machine-readable summary of ``seq``: timing, every constraint of ``hardware`` with
    its margin and specific violations, and (for Cartesian sequences) the PSF and, given a
    ``phantom`` and label ``masks`` on the image grid, per-tissue signal, contrast, CNR and
    SNR efficiency. Noise: image-domain ``noise_sigma``, or ``kspace_sigma`` (+
    ``noise_cov``) propagated by pseudo-replicas through the reconstruction; without either,
    noise-dependent metrics are omitted. Only plain JSON types are returned.
    """
    hw = hardware or Hardware()
    simulate = simulate or _default_simulate()
    with torch.no_grad():
        duration = float(seq.duration)
        report = {
            "timing": {
                "duration_s": duration, "n_segments": len(seq), "n_rf_segments": sum(seq.is_rf),
                "n_adc": seq.n_adc, "n_spoils": sum(seq.spoil),
                **{f"{k}_s": _num(seq.meta[k]) for k in ("tr", "te", "dwell") if k in seq.meta},
            },
            "hardware": asdict(hw),
            "constraints": {},
            "violations": [],
        }
        for c in check_hardware(seq, hw):
            v = c.values.detach().double()
            peak = float(v.max()) if v.numel() else 0.0
            items = violations(c, seq, max_items)
            entry = {"value": peak, "limit": c.limit, "unit": c.unit,
                     "margin_pct": None if c.limit is None else 100.0 * (1.0 - peak / c.limit),
                     "ok": c.limit is None or peak <= c.limit,
                     "n_violating": 0 if c.limit is None else int((v > c.limit).sum()), "violations": items}
            if v.numel():
                entry["at_segment"] = int(c.segments[int(v.argmax())])
            report["constraints"][c.name] = entry
            if items:  # one line per violated constraint: how often, and the worst place
                n_ranges = len(items) - 1 + items[-1].get("n_more", 1)
                where = "" if c.where == "sequence" else (
                    f" at {entry['n_violating']} {'boundaries' if c.where == 'boundaries' else 'segments'}"
                    f" ({n_ranges} ranges); worst")
                worst = items[0]["message"][len(f"{c.name} exceeded "):]
                report["violations"].append(f"{c.name} exceeded{where} {worst}" if where else items[0]["message"])
        report["ok"] = not report["violations"]

        cartesian = "kspace_shape" in seq.meta
        if cartesian and psf:
            p = point_spread_function(seq, t1=psf_tissue[0], t2=psf_tissue[1], simulate=simulate)
            report["psf"] = {k: float(v) for k, v in psf_metrics(p).items()}
            report["psf"]["tissue_t1_t2_s"] = list(psf_tissue)
        if cartesian and phantom is not None and masks:
            sig = simulate(phantom, seq, coils).signal
            img = magnitude_image(sig, seq)
            sigma = noise_sigma
            if sigma is None and kspace_sigma is not None:
                any_mask = torch.stack(list(masks.values())).any(0)
                recon = lambda s: magnitude_image(s, seq)
                sigma = float(image_noise_sigma(sig, recon, kspace_sigma, noise_cov, any_mask,
                                                generator=torch.Generator().manual_seed(0)))
            metrics = {"noise_sigma": sigma, "tissues": {}, "contrasts": {}}
            for name, m in masks.items():
                if m.any():
                    t = {"mean_signal": float(tissue_mean(img, m)), "n_pixels": int(m.sum())}
                    if sigma:
                        t["snr"] = float(snr(img, m, sigma))
                        t["snr_efficiency_per_sqrt_s"] = t["snr"] / math.sqrt(duration)
                    metrics["tissues"][name] = t
            for a, b in contrast_pairs:
                c = float(contrast(img, masks[a], masks[b]))
                entry = {"contrast": c}
                if sigma:
                    entry["cnr"] = c / sigma
                    entry["cnr_efficiency_per_sqrt_s"] = c / sigma / math.sqrt(duration)
                metrics["contrasts"][f"{a}-{b}"] = entry
            report["image"] = metrics
    return report


__all__ = [
    "AXES", "GAMMA_BAR", "Check", "Hardware", "b1_uT", "boundary_times", "check_hardware", "cnr", "contrast",
    "crb", "design_report", "fisher_information", "grad_mT_m", "gradient_steps", "hinge2", "image_noise_sigma",
    "jacobian", "magnitude_image", "minimize", "penalties", "pns", "point_spread_function", "psf_metrics",
    "rf_energy", "rf_power", "sar_proxy", "slew_rate", "snr", "snr_efficiency", "tissue_mean", "violations",
]
