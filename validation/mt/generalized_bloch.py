"""Where the rate model of magnetization transfer fails: the bound pool after on-resonance
rectangular pulses, generalized Bloch model (``blochparty.generalized_bloch``) vs. the
Graham rate model of ``exchange.py`` (W = w1^2 G, the super-Lorentzian held at its 1 kHz value
on resonance), as Fig. 3 of Asslander et al. (arXiv 2107.11000).

    uv run python validation/mt/generalized_bloch.py

Isolated bound pool, super-Lorentzian, T2s = 12 us, R1s = 1/s. Rate model: z_s = exp(-W T_RF)
(plus R1, negligible here)."""

import math

import numpy as np
import torch

from blochparty import generalized_bloch as gb
from blochparty.exchange import superlorentzian

T2S = 12e-6


def main():
    g0 = float(superlorentzian(torch.tensor(0.0, dtype=torch.float64), torch.tensor(T2S, dtype=torch.float64)))
    print("z_s(T_RF) / z_s(0), on-resonance rectangular pulses; rate model with the 1 kHz cut\n")
    print("| T_RF | α = π/2: generalized | rate model | α = π: generalized | rate model |\n|---|---|---|---|---|")
    for t_rf in (10e-6, 30e-6, 100e-6, 300e-6, 1e-3, 3e-3):
        h = min(T2S / 24, t_rf / 40)
        row = []
        for alpha in (math.pi / 2, math.pi):
            zs = gb.simulate_pulse(gb.rectangular(alpha, t_rf, h), h, t2s=T2S, shape="superlorentzian")["zs"][-1]
            w = (alpha / t_rf) ** 2 * g0
            row += [zs, math.exp(-w * t_rf)]
        print(f"| {t_rf * 1e6:g} µs | {row[0]:+.3f} | {row[1]:+.3f} | {row[2]:+.3f} | {row[3]:+.3f} |", flush=True)


if __name__ == "__main__":
    main()
