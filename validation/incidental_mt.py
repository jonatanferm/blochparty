"""Incidental magnetization transfer in multi-slice imaging, against Shin, Gu & Yang (2009).

    uv run python validation/incidental_mt.py

Shin et al. (MRM 2009; PMC2779549) measured at 3 T, in white and grey matter, that the steady
state of a multi-slice inversion-recovery Look-Locker EPI (TR 400 ms, 16 deg, 5 mm slices) with fat
saturation falls from ~94-98 % (1 slice) to ~70-72 % (9 slices): about 30 % signal loss and a
20 % T1 underestimation, attributed to the fat-saturation pulses saturating the bound pool.

Here: one spin at the centre of the imaged slice, the Look-Locker train run to its steady state
(25 TRs, as their 10 s), Mz before the imaged excitation relative to 1 slice without fat
saturation. Per TR and slice: a non-selective Gaussian fat-saturation pulse at -3.4 ppm
(-436 Hz) and an ideal spoiler, then a 16 deg slice-selective sinc (2 ms, TBW 4) at that slice's
position (the other slices' pulses reach our spin as off-resonant pulses through the slice
gradient), then the EPI readout as a 30 ms delay and a spoiler; slices 10 mm apart in 40 ms
slots. Everything except the bound pool comes from the Bloch equations, so the same run
without it isolates MT from direct saturation of the water.

Not reported by the paper, so scanned: the fat-saturation pulse's duration and flip angle.
Tissue (3 T): WM T1/T2 1084/69 ms, GM 1820/99 ms (Stanisz et al., MRM 2005); bound pool WM
f = 0.15, GM 0.07, k = 30 Hz (kf ~ 4.5 / 2.1 s^-1), T1b 1 s, T2b 12 us, super-Lorentzian:
typical qMT values (Gloor et al., MRM 2008; Stanisz et al. 2005).
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch

import blochparty as bp
from blochparty.fused import simulate_fused

F_FAT = -3.4e-6 * 3.0 * bp.GAMMA_BAR  # [Hz]
TR, SLOT, N_TR = 0.400, 0.040, 25
TISSUE = {"WM": dict(t1=1.084, t2=0.069, f=0.15), "GM": dict(t1=1.820, t2=0.099, f=0.07)}


def fat_sat(b: bp.SequenceBuilder, duration: float, flip_deg: float, dt=20e-6):
    n = round(duration / dt)
    t = (torch.arange(n, dtype=torch.float64) + 0.5) * dt - 0.5 * duration
    env = torch.exp(-0.5 * (t / (duration / 6)) ** 2)
    env = env * (flip_deg / 360.0) / (env.sum() * dt)
    ph = -2 * math.pi * F_FAT * t  # turns as exp(-i 2 pi f t): resonant with fat at F_FAT
    b.rf_waveform(torch.stack([env * torch.cos(ph), env * torch.sin(ph)], -1), dt)
    b.delay(1e-3).spoil()


def _count(b: bp.SequenceBuilder) -> tuple[int, float]:
    """Segments so far and their total duration."""
    return sum(t.numel() for t in b._dt), float(sum(t.sum() for t in b._dt)) if b._dt else 0.0


def sequence(n_slices: int, fs: tuple[float, float] | None) -> tuple[bp.Sequence, list[int]]:
    """N_TR Look-Locker TRs; returns the sequence and the segment index of every imaged excitation."""
    b, marks = bp.SequenceBuilder(), []
    centre = n_slices // 2
    for _ in range(N_TR):
        t_used = 0.0
        for k in range(n_slices):
            t0 = _count(b)[1]
            if fs:
                fat_sat(b, *fs)
            if k == centre:
                marks.append(_count(b)[0])
            b.sinc_pulse(16.0, 2e-3, slice_thickness=5e-3, slice_offset=(k - centre) * 10e-3)
            b.delay(30e-3).spoil()
            b.delay(SLOT - (_count(b)[1] - t0))
            t_used += SLOT
        b.delay(TR - t_used)
    return b.build(), marks


def steady_state(tissue: str, mt: bool, n_slices: int, fs) -> float:
    p = TISSUE[tissue]
    one = lambda v: torch.full((1,), v, dtype=torch.float64)
    pool = bp.BoundPool(f=p["f"], k=30.0, t1=1.0, t2=12e-6) if mt else None
    ph = bp.Phantom(pos=torch.zeros(1, 3, dtype=torch.float64), pd=one(1.0), t1=one(p["t1"]), t2=one(p["t2"]),
                    db0=one(0.0), bound=pool)
    seq, marks = sequence(n_slices, fs)
    cut = marks[-1]  # Mz just before the last imaged excitation
    part = replace(seq, dt=seq.dt[:cut], rf=seq.rf[:cut], grad=seq.grad[:cut], adc_phase=seq.adc_phase[:cut],
                   is_rf=seq.is_rf[:cut], adc=seq.adc[:cut], spoil=seq.spoil[:cut], meta={})
    return float(simulate_fused(ph, part).m[0, 2])


def main():
    slices = (1, 3, 5, 7, 9)
    print(f"fat saturation at {F_FAT:.0f} Hz; relative steady state Mz = Mz_ss / Mz_ss(1 slice, no fat sat)\n")
    print("| tissue | fat sat | bound pool | " + " | ".join(f"{n} sl." for n in slices) + " | loss 1→9 |")
    print("|---|---|---|" + "---|" * len(slices) + "---|")
    for tissue in ("WM", "GM"):
        for mt in (True, False):
            ref = steady_state(tissue, mt, 1, None)
            for fs in ((10e-3, 90.0), (10e-3, 110.0), (10e-3, 180.0), (20e-3, 110.0), (5e-3, 110.0)):
                rel = [steady_state(tissue, mt, n, fs) / ref for n in slices]
                print(f"| {tissue} | {fs[0] * 1e3:.0f} ms, {fs[1]:.0f}° | {'yes' if mt else 'no'} | "
                      + " | ".join(f"{100 * r:.1f} %" for r in rel) + f" | {100 * (1 - rel[-1] / rel[0]):.1f} % |",
                      flush=True)
            if not mt:
                break
    print("\nShin et al. 2009 (3 T, WM and GM): ~94-98 % (1 slice) -> ~70-72 % (9 slices), ~30 % loss")


if __name__ == "__main__":
    main()
