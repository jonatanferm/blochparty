"""Hardware and safety: gradient amplitude, slew and timing, RF energy, power and a SAR proxy, peripheral nerve
stimulation; the limits (``Hardware``), their checks, penalties and violation reports."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..constants import GAMMA_BAR
from ..sequence import Sequence

AXES = "xyz"


@dataclass
class Hardware:
    """Scanner limits. ``None`` disables a limit (the value is still reported).

    The PNS model parameters are generic whole-body values, not those of any particular
    gradient coil; see ``pns``.
    """

    max_grad: float = 40.0  # per axis [mT/m]
    max_slew: float = 170.0  # per axis [T/m/s]; with max_grad, the defaults of sequence.System
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
    return torch.sqrt((seq.rf**2).sum(-1) + 1e-30) / GAMMA_BAR * 1e6


def rf_energy(seq: Sequence) -> Tensor:
    """Sum of |B1|^2 dt over the sequence [µT^2 s], the quantity SAR is proportional to."""
    b1sq = (seq.rf**2).sum(-1) * (1e6 / GAMMA_BAR) ** 2
    return (b1sq * seq.dt).sum()


def rf_power(seq: Sequence, window: float | None = None) -> Tensor:
    """Mean |B1|^2 [µT^2]: over the whole sequence (``window=None``, a scalar), or (S,)
    over windows of ``window`` seconds starting at every segment (windows reaching past
    the end are averaged over what remains, but never over less than ``window``)."""
    b1sq = (seq.rf**2).sum(-1) * (1e6 / GAMMA_BAR) ** 2
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


def pns(
    seq: Sequence, rheobase: float = 100.0, chronaxie: float = 360e-6, raster: float = 10e-6, window: float = 20e-3
) -> tuple[Tensor, Tensor]:
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
    return torch.sqrt((r**2).sum(-1) + 1e-30), idx


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
    checks += [Check(f"slew {a}", slew[:, i], bounds, hw.max_slew, "T/m/s", "boundaries") for i, a in enumerate(AXES)]
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
    checks.append(
        Check("scan time", seq.duration.reshape(1), torch.zeros(1, dtype=torch.long), hw.max_duration, "s", "sequence")
    )
    return checks


def hinge2(values: Tensor, limit: float) -> Tensor:
    """Smooth (C1) penalty: sum of squared relative excess over ``limit``, zero inside."""
    return (torch.relu(values / limit - 1.0) ** 2).sum()


def penalties(seq: Sequence, hw: Hardware | None = None, only: tuple[str, ...] | None = None) -> dict[str, Tensor]:
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
            span += f" (t = {times[0]:.3f} ms)" if times[0] == times[1] else f" (t = {times[0]:.3f}–{times[1]:.3f} ms)"
        items.append(
            {
                "constraint": check.name,
                "segments": [a, b],
                "index": check.where,
                "time_ms": times,
                "value": worst,
                "limit": check.limit,
                "unit": check.unit,
                "excess_pct": excess,
                "message": f"{check.name} exceeded{span} by {excess:.3g} % "
                f"({worst:.4g} {check.unit} > {check.limit:.4g} {check.unit})",
            }
        )
    items.sort(key=lambda d: -d["excess_pct"])
    return items[:max_items] + (
        [
            {
                "constraint": check.name,
                "message": f"{check.name}: {len(items) - max_items} more violating ranges not listed",
                "n_more": len(items) - max_items,
            }
        ]
        if len(items) > max_items
        else []
    )
