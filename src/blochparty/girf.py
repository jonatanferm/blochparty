"""Gradient impulse responses and eddy currents: sequence transforms (REALISM.md, R1).

``apply_girf(seq, model)`` returns the sequence the spins actually see: each gradient
axis and the uniform field (``Sequence.b0_offset``) become linear time-invariant
responses to the nominal gradients. Rows of every model are the outputs (x, y, z
gradient [Hz/m], B0 [Hz]), columns the input axes (x, y, z), so cross terms and B0 eddy
terms are entries like any other.

The nominal gradient g is piecewise constant on the segments, and so is what the spins
see, as the exact average of the response over each segment: with G = int g (the
k-space moment, piecewise linear), the response y = h * g averages to
[Y(t_{s+1}) - Y(t_s)] / dt_s with Y = h * G. k-space at every segment end is therefore exact
for the staircase input (a ramp on the gradient raster is a staircase of its averages;
its response differs from that of the true ramp by O((raster / tau)^2)).

* ``GIRF``: impulse responses as delta trains on a raster (taps h_n at delay + n raster,
  the discrete GIRF convolution), or from a transfer function on the FFT grid.
* ``EddyCurrents``: multi-exponential terms; output -a (dg/dt * exp(-t / tau)), the classic
  eddy-current model. Amplitudes and time constants are differentiable (tensors), and
  the exponentials are evaluated by an exact recurrence over segments (a blocked scan, so
  long sequences stay vectorized).

Everything is plain tensor arithmetic: gradients reach the model's tensors and the
nominal sequence. Timing (segment boundaries) is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from .sequence import Sequence

OUTPUTS = ("x", "y", "z", "b0")


def _boundaries(seq: Sequence) -> Tensor:
    """(S + 1,) segment boundary times [s] from the sequence start."""
    return torch.cat([seq.dt.new_zeros(1), torch.cumsum(seq.dt, 0)])


def _moment_at(t: Tensor, tb: Tensor, grad: Tensor, moment: Tensor) -> Tensor:
    """(*t.shape, 3) moments G(t) = int_0^t g of the piecewise-constant g (S, 3) on boundaries
    tb (S + 1,) with G(tb) = ``moment`` (S + 1, 3); g = 0 outside the sequence."""
    s = torch.searchsorted(tb.detach(), t.detach().reshape(-1), right=True) - 1  # segment containing t
    s = s.clamp(0, len(grad) - 1)
    tc = t.reshape(-1).clamp(tb[0], tb[-1])
    val = moment[s] + grad[s] * (tc - tb[s])[:, None]
    return val.reshape(*t.shape, 3)


def _segment_average(y_moment: Tensor, dt: Tensor) -> Tensor:
    """(S, ...) segment averages from the response's moment at the boundaries (S + 1, ...)."""
    return (y_moment[1:] - y_moment[:-1]) / dt.reshape(-1, *([1] * (y_moment.dim() - 1)))


@dataclass
class GIRF:
    """Linear time-invariant gradient response as delta trains: output o sees
    sum_j sum_n taps[o, j, n] g_j(t - delay[o, j] - n raster).

    taps: (4, 3, n) weights (rows x, y, z [Hz/m per Hz/m], B0 [Hz per Hz/m]); the DC gain of
        an axis is the sum of its taps.
    raster: tap spacing [s] (the gradient raster).
    delay: (4, 3) extra delays [s] (or a scalar for all); differentiable.
    """

    taps: Tensor
    raster: float = 10e-6
    delay: Tensor | float = 0.0

    @classmethod
    def identity(cls, delay=0.0, dtype=torch.float64) -> GIRF:
        """Ideal gradients, optionally delayed: ``delay`` scalar or (3,) per axis [s]."""
        taps = torch.zeros(4, 3, 1, dtype=dtype)
        taps[0, 0, 0] = taps[1, 1, 0] = taps[2, 2, 0] = 1.0
        d = torch.cat([torch.diag_embed(torch.as_tensor(delay, dtype=dtype).expand(3)), torch.zeros(1, 3, dtype=dtype)])
        return cls(taps, 10e-6, d)

    @classmethod
    def from_transfer_function(cls, h: Tensor, raster: float, n: int | None = None,
                               delay: Tensor | float = 0.0) -> GIRF:
        """From transfer functions H(f) (4, 3, F) complex on the real-FFT grid f_k = k / (n raster),
        k = 0 .. F - 1 (n = 2 (F - 1) unless given): taps = irfft(H). Exact for a response
        that is a delta train on the raster; otherwise the band-limited approximation."""
        n = n or 2 * (h.shape[-1] - 1)
        return cls(torch.fft.irfft(h, n=n), raster, delay)

    def transfer_function(self, n: int) -> Tensor:
        """(4, 3, n // 2 + 1) H on the real-FFT grid of n raster steps (without ``delay``)."""
        return torch.fft.rfft(self.taps, n=n)


@dataclass
class EddyCurrents:
    """Multi-exponential eddy currents: P terms, term p adding
    -amplitude[p] (dg_{input[p]}/dt * exp(-t / time_constant[p])) to output ``output[p]``
    (0, 1, 2: x, y, z gradient, amplitude a fraction; 3: B0, amplitude in Hz per Hz/m).
    Equivalently the impulse response a (exp(-t / tau) / tau - delta(t)), DC gain 0.
    ``amplitude`` and ``time_constant`` (P,) [s] may require grad.
    """

    output: list[int]
    input: list[int]
    amplitude: Tensor
    time_constant: Tensor

    def __post_init__(self):
        if not len(self.output) == len(self.input) == len(self.amplitude) == len(self.time_constant):
            raise ValueError("output, input, amplitude and time_constant need one entry per term")

    @classmethod
    def per_axis(cls, amplitude, time_constant, b0_amplitude=None, b0_time_constant=None,
                 dtype=torch.float64) -> EddyCurrents:
        """Each axis on itself: (3, K) amplitudes and time constants (K terms per axis), and
        optionally (3, K') B0 terms of each axis [Hz per Hz/m]. Tensors keep their graph."""
        amp = torch.as_tensor(amplitude, dtype=dtype).reshape(3, -1)
        tau = torch.as_tensor(time_constant, dtype=dtype).reshape(3, -1)
        k = amp.shape[1]
        out, inp = [j for j in range(3) for _ in range(k)], [j for j in range(3) for _ in range(k)]
        amps, taus = [amp.reshape(-1)], [tau.reshape(-1)]
        if b0_amplitude is not None:
            b0a = torch.as_tensor(b0_amplitude, dtype=dtype).reshape(3, -1)
            b0t = torch.as_tensor(b0_time_constant, dtype=dtype).reshape(3, -1)
            kb = b0a.shape[1]
            out += [3] * (3 * kb)
            inp += [j for j in range(3) for _ in range(kb)]
            amps.append(b0a.reshape(-1))
            taus.append(b0t.reshape(-1))
        return cls(out, inp, torch.cat(amps), torch.cat(taus))


def _linear_scan(log_a: Tensor, b: Tensor, block: int = 16) -> Tensor:
    """x_{s+1} = exp(log_a_s) x_s + b_s from x_0 = 0, over the last axis: returns x_1 .. x_S.
    Blocked: closed form inside blocks of ``block`` steps, the same scan over the blocks."""
    n = log_a.shape[-1]
    if n <= block:
        cum = torch.cumsum(log_a, -1)  # A_s
        diff = cum[..., :, None] - cum[..., None, :]  # A_s - A_j: decay of b_j by steps j+1 .. s
        mask = torch.ones(n, n, dtype=torch.bool, device=log_a.device).tril()
        decay = torch.exp(torch.where(mask, diff, torch.full_like(diff, -torch.inf)))
        return (decay * b[..., None, :]).sum(-1)
    nb = -(-n // block)
    pad = nb * block - n
    la = torch.nn.functional.pad(log_a, (0, pad)).reshape(*log_a.shape[:-1], nb, block)
    bb = torch.nn.functional.pad(b, (0, pad)).reshape(*b.shape[:-1], nb, block)
    local = _linear_scan(la, bb, block)  # (..., nb, block): zero state at each block start
    cum = torch.cumsum(la, -1)
    carry = _linear_scan(cum[..., -1], local[..., -1], block)  # state after each block
    start = torch.cat([torch.zeros_like(carry[..., :1]), carry[..., :-1]], -1)  # state at each block start
    x = local + start[..., None] * torch.exp(cum)
    return x.reshape(*b.shape[:-1], nb * block)[..., :n]


def _exponential_response(grad: Tensor, dt: Tensor, tau: Tensor) -> Tensor:
    """Segment averages of E = (1/tau) exp(-t/tau) * g for each term: g (S, P), tau (P,) ->
    (S, P). Exact for piecewise-constant g: E relaxes towards g_s within segment s,
    E(t_s + u) = g_s + (E_s - g_s) exp(-u / tau)."""
    z = dt[:, None] / tau  # (S, P)
    one_minus = -torch.expm1(-z)
    # E at segment starts: E_0 = 0, E_{s+1} = exp(-z_s) E_s + (1 - exp(-z_s)) g_s
    after = _linear_scan((-z).T, (one_minus * grad).T).T
    start = torch.cat([torch.zeros_like(after[:1]), after[:-1]])
    phi = one_minus / torch.where(z > 0, z, torch.ones_like(z))  # (1 - exp(-z)) / z
    phi = torch.where(z > 1e-8, phi, 1 - z / 2)
    return grad + (start - grad) * phi


def apply_girf(seq: Sequence, model: GIRF | EddyCurrents | list) -> Sequence:
    """The sequence as played by the hardware: gradients replaced by the model's response
    to the nominal ones, the B0 response added to ``b0_offset``. ``model`` may be a list:
    responses add up, as deviations from the nominal gradients (at most one ``GIRF``;
    ``EddyCurrents`` terms act on the nominal gradients, not on the GIRF's output)."""
    models = model if isinstance(model, (list, tuple)) else [model]
    if sum(isinstance(m, GIRF) for m in models) > 1:
        raise ValueError("give at most one GIRF")
    g, dt = seq.grad, seq.dt
    nominal = torch.cat([g, g.new_zeros(len(seq), 1)], -1)  # (S, 4): x, y, z, B0
    out = nominal
    for m in models:
        if isinstance(m, GIRF):
            out = out + (_girf_response(seq, m) - nominal)
        elif isinstance(m, EddyCurrents):
            gin = g[:, m.input]  # (S, P)
            e = _exponential_response(gin, dt, m.time_constant.to(g.dtype))
            # -a (dg/dt * exp(-t/tau)) = -a (g - E)
            dev = -m.amplitude.to(g.dtype) * (gin - e)
            idx = torch.tensor(m.output, device=g.device)
            out = out + torch.zeros_like(out).index_add(1, idx, dev)
        else:
            raise TypeError(f"unknown model {type(m).__name__}")
    off = out[:, 3] if seq.b0_offset is None else seq.b0_offset + out[:, 3]
    return replace(seq, grad=out[:, :3], b0_offset=off)


def _girf_response(seq: Sequence, m: GIRF) -> Tensor:
    """(S, 4) segment averages of the GIRF's outputs."""
    g, dt = seq.grad, seq.dt
    tb = _boundaries(seq)
    moment = torch.cat([g.new_zeros(1, 3), torch.cumsum(g * dt[:, None], 0)])  # (S + 1, 3)
    taps = m.taps.to(g.dtype)
    n = taps.shape[-1]
    delay = torch.as_tensor(m.delay, dtype=g.dtype).expand(4, 3)
    lags = delay[..., None] + m.raster * torch.arange(n, dtype=g.dtype)  # (4, 3, n)
    out = []
    for o in range(4):
        y = 0
        for j in range(3):
            if not bool((taps[o, j] != 0).any()) and not taps.requires_grad:
                continue
            t = tb[:, None] - lags[o, j][None, :]  # (S + 1, n)
            gm = _moment_at(t, tb, g, moment)[..., j]  # (S + 1, n): G_j at the shifted boundaries
            y = y + (gm * taps[o, j]).sum(-1)
        out.append(_segment_average(y, dt) if torch.is_tensor(y) else g.new_zeros(len(seq)))
    return torch.stack(out, -1)


__all__ = ["EddyCurrents", "GIRF", "OUTPUTS", "apply_girf"]
