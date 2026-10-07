"""The receive chain after the coils: what a scanner's electronics do to the signal.

Scanners sample each readout ``os`` times faster than its dwell, low-pass filter digitally and
decimate; thermal noise, interference and spikes enter before that filter (at the coil and
preamplifier), the ADC quantises with finite range. Here:

* ``oversample(seq, os)`` -> (sequence, info): every readout sampled ``os`` times faster, at
  (j + 1/2) dwell / os from its start (segments split there; piecewise-constant, so exact).
* ``decimate(signal, info, taps)``: a windowed-sinc FIR (Kaiser, cutoff at the original
  Nyquist, unit DC gain, centred on the original samples: no delay) per readout, back to the
  original samples. Its frequency
  response is ``response(info, f)``; near the band edges the signal rolls off, as on scanners.
* Impairments on (C, n) signals at the oversampled rate (before the filter) or after it:
  ``thermal_noise`` (complex Gaussian, channel covariance), ``interference`` (a narrowband
  source at a fixed absolute frequency: in the image column of its frequency; a line along
  phase encoding only if its phase is incoherent with the TR, as real ones), ``spikes`` (isolated
  broadband bursts), ``gain_drift`` (receiver gain changing over the scan) and ``quantise``
  (ADC bits and full-scale range: rounding and clipping).

``ReceiveChain`` applies them in the scanner's order: noise, interference and spikes at the
oversampled rate, the decimation filter, gain drift, then quantisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import Tensor

from .sequence import Sequence


@dataclass
class OversampleInfo:
    os: int
    readouts: list[tuple[int, int]]  # (first original sample, count) per readout
    dwell: list[float]  # per readout [s]
    times: Tensor  # (n_adc * os,) absolute times of the oversampled samples [s]


def _readouts(t: np.ndarray) -> tuple[list[tuple[int, int]], list[float]]:
    """Group the ADC sample times into readouts: runs with a constant spacing (the dwell)."""
    groups, dwells, start = [], [], 0
    for i in range(2, len(t) + 1):
        if i == len(t) or not math.isclose(t[i] - t[i - 1], t[start + 1] - t[start], rel_tol=1e-6):
            groups.append((start, i - start))
            dwells.append(float(t[start + 1] - t[start]))
            start = i
            if start == len(t) - 1:  # a lone last sample: no dwell
                groups.append((start, 1))
                dwells.append(float("nan"))
                break
    return groups, dwells


def oversample(seq: Sequence, os: int) -> tuple[Sequence, OversampleInfo]:
    """The sequence with every readout sampled ``os`` times faster (see the module docstring)."""
    if os < 1:
        raise ValueError("os must be >= 1")
    dt = seq.dt.detach().double().cpu().numpy()
    ends = np.cumsum(dt)
    adc = np.array(seq.adc)
    t_adc = ends[adc]
    groups, dwells = _readouts(t_adc)
    if any(not np.isfinite(d) for d in dwells):
        raise ValueError("every readout needs at least two samples (a dwell)")
    psi_all = seq.adc_phase.detach().double().cpu().numpy()[adc]
    new_t, new_psi = [], []
    for (a, n), dw in zip(groups, dwells):
        t0 = t_adc[a] - 0.5 * dw  # readout start
        j = np.arange(n * os)
        new_t.append(t0 + (j + 0.5) * dw / os)
        # receiver phase: linear in time along the readout (frequency offsets), from its samples
        ti = t_adc[a:a + n]
        ps = np.unwrap(psi_all[a:a + n])
        new_psi.append(np.interp(new_t[-1], ti, ps) if n > 1 else np.full(n * os, ps[0]))
    new_t, new_psi = np.concatenate(new_t), np.concatenate(new_psi)
    # split segments at the new sample times; an ADC flag at each
    starts = ends - dt
    cut = np.unique(np.concatenate([ends, new_t]))
    seg_of = np.searchsorted(ends, cut - 1e-15, side="left")  # the segment containing each piece's end
    seg_of = np.minimum(seg_of, len(dt) - 1)
    piece_dt = np.diff(np.concatenate([[0.0], cut]))
    keep = piece_dt > 1e-15
    cut, seg_of, piece_dt = cut[keep], seg_of[keep], piece_dt[keep]
    is_sample = np.isin(np.round(cut, 15), np.round(new_t, 15))
    idx = torch.as_tensor(seg_of)
    psi = torch.zeros(len(cut), dtype=seq.dt.dtype, device=seq.dt.device)
    psi[torch.as_tensor(np.nonzero(is_sample)[0])] = torch.as_tensor(new_psi, dtype=psi.dtype, device=psi.device)
    seg_end = np.isclose(cut, ends[seg_of], rtol=0, atol=1e-15)
    out = replace(seq, dt=torch.as_tensor(piece_dt, dtype=seq.dt.dtype, device=seq.dt.device),
                  rf=seq.rf[idx], grad=seq.grad[idx], adc_phase=psi,
                  b0_offset=None if seq.b0_offset is None else seq.b0_offset[idx],
                  rf_freq=None if seq.rf_freq is None else seq.rf_freq[idx],
                  is_rf=[seq.is_rf[k] for k in seg_of], adc=is_sample.tolist(),
                  spoil=[bool(seq.spoil[k] and e) for k, e in zip(seg_of, seg_end)])
    if any(out.is_rf[k] for k in np.nonzero(is_sample)[0]):
        raise ValueError("oversampled ADC samples fall inside RF segments")
    return out, OversampleInfo(os, groups, dwells, torch.as_tensor(new_t))


def fir(os: int, taps_per_sample: int = 16, beta: float = 8.0) -> np.ndarray:
    """Windowed-sinc low-pass, cutoff 0.5 / os cycles per sample, unit DC gain; even length for
    even os (each original sample lies midway between two oversampled ones), odd for odd os
    (it coincides with one), so it is centred on the original samples."""
    n = taps_per_sample * os
    if os % 2 and n % 2 == 0:
        n += 1
    m = np.arange(n) - (n - 1) / 2
    h = np.sinc(m / os) * np.kaiser(n, beta)
    return h / h.sum()


def response(os: int, f_hz: np.ndarray, dwell: float, taps_per_sample: int = 16, beta: float = 8.0) -> np.ndarray:
    """|H| of the decimation filter at frequencies [Hz] for a readout dwell (its DTFT)."""
    h = fir(os, taps_per_sample, beta)
    m = np.arange(len(h)) - (len(h) - 1) / 2
    w = 2 * np.pi * np.asarray(f_hz)[..., None] * dwell / os
    return np.abs((h * np.exp(-1j * w * m)).sum(-1))


def decimate(signal: Tensor, info: OversampleInfo, taps_per_sample: int = 16, beta: float = 8.0) -> Tensor:
    """(C, n_adc * os) oversampled signal -> (C, n_adc): filter each readout, keep the samples
    at the original times (no group delay: the filter is centred on them)."""
    os = info.os
    if os == 1:
        return signal
    h = torch.as_tensor(fir(os, taps_per_sample, beta), dtype=signal.real.dtype, device=signal.device)
    half = len(h) // 2
    out, pos = [], 0
    for _, n in info.readouts:
        x = signal[:, pos:pos + n * os]
        pos += n * os
        # original sample i sits between oversampled i os + os/2 - 1 and i os + os/2 (os even), or at i os + (os-1)/2
        pad = torch.nn.functional.pad
        xr = pad(x.real[:, None], (half, half))[:, 0]
        xi = pad(x.imag[:, None], (half, half))[:, 0]
        centre = np.arange(n) * os + (os - 1) / 2 + half  # in the padded signal, between taps for even os
        starts = np.round(centre - (len(h) - 1) / 2).astype(int)
        win = torch.as_tensor(starts[:, None] + np.arange(len(h))[None], device=signal.device)
        out.append(torch.complex((xr[:, win] * h).sum(-1), (xi[:, win] * h).sum(-1)))
    return torch.cat(out, 1)


def thermal_noise(signal: Tensor, sigma: float, cov: Tensor | None = None, generator=None) -> Tensor:
    """Complex Gaussian noise of std ``sigma`` per channel (real and imaginary each sigma / sqrt 2),
    correlated across channels by ``cov`` (C, C) if given."""
    c, n = signal.shape
    z = torch.randn(2, c, n, generator=generator, dtype=signal.real.dtype)
    noise = torch.complex(z[0], z[1]) * (sigma / math.sqrt(2))
    if cov is not None:
        noise = torch.linalg.cholesky(cov.to(noise.dtype)) @ noise
    return signal + noise.to(signal.device)


def interference(signal: Tensor, times: Tensor, freq_hz: float, amplitude: float, channels=None,
                 phase: float = 0.0) -> Tensor:
    """A narrowband source at a fixed frequency (relative to the receiver) on the given channels."""
    tone = amplitude * torch.exp(1j * (2 * math.pi * freq_hz * times.to(signal.real.dtype) + phase))
    out = signal.clone()
    out[channels if channels is not None else slice(None)] += tone.to(signal.device)
    return out


def spikes(signal: Tensor, count: int, amplitude: float, generator=None) -> Tensor:
    """``count`` isolated spikes at random samples (all channels), random phase."""
    n = signal.shape[1]
    where = torch.randint(0, n, (count,), generator=generator)
    ph = 2 * math.pi * torch.rand(count, generator=generator)
    out = signal.clone()
    out[:, where] += (amplitude * torch.exp(1j * ph)).to(signal.dtype).to(signal.device)
    return out


def gain_drift(signal: Tensor, times: Tensor, per_second: float) -> Tensor:
    """Receiver gain 1 + per_second t."""
    return signal * (1 + per_second * times.to(signal.real.dtype)).to(signal.device)


def quantise(signal: Tensor, bits: int, full_scale: float) -> Tensor:
    """Real and imaginary parts rounded to ``bits``-bit two's complement over +-full_scale, clipped."""
    step = full_scale / 2 ** (bits - 1)
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    q = lambda x: torch.clamp(torch.round(x / step), lo, hi) * step
    return torch.complex(q(signal.real), q(signal.imag))


@dataclass
class ReceiveChain:
    """The impairments a scanner adds after the coils, in their order (see the module docstring).
    Sigma is per oversampled sample; the filter then narrows the noise to the readout band."""

    os: int = 2
    sigma: float = 0.0
    cov: Tensor | None = None
    tone: tuple[float, float] | None = None  # (frequency [Hz], amplitude)
    n_spikes: int = 0
    spike_amplitude: float = 0.0
    drift_per_second: float = 0.0
    bits: int | None = None
    full_scale: float | None = None
    taps_per_sample: int = 16

    def acquire(self, simulate, seq: Sequence, generator=None) -> Tensor:
        """Run ``simulate(sequence) -> SimResult`` on the oversampled sequence and pass its signal
        through the chain: the (C, n_adc) signal a scanner would record."""
        seq_os, info = oversample(seq, self.os)
        s = simulate(seq_os).signal
        if self.sigma:
            s = thermal_noise(s, self.sigma, self.cov, generator)
        if self.tone is not None:
            s = interference(s, info.times, *self.tone)
        if self.n_spikes:
            s = spikes(s, self.n_spikes, self.spike_amplitude, generator)
        s = decimate(s, info, self.taps_per_sample)
        if self.drift_per_second:
            t = torch.as_tensor(np.cumsum(seq.dt.detach().double().cpu().numpy())[np.array(seq.adc)])
            s = gain_drift(s, t, self.drift_per_second)
        if self.bits is not None:
            s = quantise(s, self.bits, self.full_scale)
        return s
