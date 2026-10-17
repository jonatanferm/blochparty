"""Pulseq import: ``.seq`` file or ``pypulseq.Sequence`` -> ``Sequence``.

Every block is split at all of its event boundaries: RF raster steps (or the
points of a time-shaped pulse), the RF centre, gradient corner points, ADC sample
times and the block end. Each ADC sample therefore lands exactly on a segment end.
Gradients are piecewise linear in Pulseq and every corner is a breakpoint, so the
midpoint value of a segment is its exact average gradient: gradient moments, and so
k-space, are exact at every segment end. Ramps (and other sloped pieces) are further
cut at ``ramp_raster`` so that spins see a staircase close to the true ramp.

Conventions (Pulseq -> blochparty), checked in ``tests/sequences/test_pulseq.py``:

* Units already match: RF in Hz, gradients in Hz/m, time in s.
* Pulseq fixes the sign of a frequency offset physically: a pulse with frequency
  offset f excites the spins whose Larmor frequency is higher by f (slice position
  = f / G, as in ``pypulseq``'s ``calculate_kspace``). Those spins precess as
  exp(-i 2 pi f t) in blochparty, so the pulse must turn the same way. With the
  Pulseq RF phase defined as phase(t) = phase_offset + 2 pi f t (from the start of
  the shape, as in the Pulseq specification), the field is the complex conjugate
  of the Pulseq waveform: ``rf_x + i rf_y = conj(signal * exp(i phase(t)))``.
  So a real pulse (phase 0) still tips +z to +y (Mxy = +i), while Pulseq phase phi
  gives Mxy = i exp(-i phi) sin(alpha).
* ADC: the receiver phase is psi = -(phase_offset + 2 pi f t_i + modulation_i), with
  t_i = (i + 1/2) dwell from the ADC start. RF and ADC phases written with the same
  value cancel (RF spoiling) and an ADC frequency offset follows the spins at +f.
* ``convention="mrzero"`` uses the Pulseq phases as written instead (field
  ``signal * exp(i phase(t))``, receiver phase +phase): the reading of MRzero, whose
  PDG simulation then equals blochparty's up to the constant factor -i (its signal
  carries an extra +pi/2 receiver phase and it rotates the RF the other way).
  The two readings are mirror images of each other: conj(signal) on the spatially
  inverted phantom with negated off-resonance. Frequency offsets then excite -f.

Label, trigger and soft-delay extensions are ignored (soft delays at their default
duration). Anything else (unknown extensions such as rotations, RF shims) raises.
"""

from ...sequence import flip_angle
from .read import read_pulseq
from .write import write_pulseq

__all__ = ["flip_angle", "read_pulseq", "write_pulseq"]
