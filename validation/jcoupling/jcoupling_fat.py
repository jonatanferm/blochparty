"""Does J coupling explain bright fat at 3 T? The triglyceride fragment model (blochparty.jcoupling) against fat T2
measured at 3 T with spin-echo trains of different echo spacing.

Measured (src/blochparty/anatomy/tissuedb): CPMG / multi-echo spin echo at 7.6-8.0 ms spacing, 17 echoes: 140-159 ms
(Keene 2020, subcutaneous fat, four cohorts); single-echo spin echoes at TE 10-160 ms: 48 ms (Jordan 2013, knee),
68 ms (de Bazelaire 2004, abdomen, multi-TE SE with HASTE readout).

The model has J modulation only; one intrinsic T2 (shared by every proton) is calibrated so that its CPMG at 8 ms
spacing (17 echoes, mono-exponential fit) reads 145 ms, then the single-echo series (TE 10 / 20 / 40 / 80 / 160 ms)
and other spacings are predicted.

    uv run python validation/jcoupling/jcoupling_fat.py

2026-10-13: intrinsic T2 266 ms; single-echo apparent T2 222 ms against the measured 48-68 ms, and fat at TE 100 ms
1.13 x brighter in a 10.26 ms train than in one spin echo (Henkelman 1992: up to ~2): the model's J modulation (the
coupled chain ends, about half the protons) cannot produce the in-vivo spin-echo / CPMG difference; most of the fat
signal (the bulk methylene, whose neighbours share its shift) is not J-modulated here, and the coupled groups'
modulation partly recovers by TE 160. At ~113 ms the modulation is 0.687 with 8 ms spacing and 0.663 with fastMRI's
10.26 ms: J coupling makes that protocol's fat ~3.5 % dimmer than the CPMG value, not brighter. The simulator keeps
the CPMG fat T2 (tissuedb LIPIDS); an echo-spacing-dependent fat T2 needs measured curves at intermediate spacings, or
a model of the methylene chain and of what else fast refocusing suppresses.
"""
import numpy as np
from scipy.optimize import brentq

from blochparty.jcoupling import fat_modulation


def mono_t2(t, s):
    A = np.stack([np.ones_like(t), -t], 1)
    return 1 / np.linalg.lstsq(A, np.log(s), rcond=None)[0][1]


def main():
    field = 3.0
    t_cpmg = 8e-3 * np.arange(1, 18)
    m_cpmg = fat_modulation(t_cpmg, field, mode="cpmg", esp=8e-3).numpy()
    t_se = np.array([10, 20, 40, 80, 160]) * 1e-3
    m_se = fat_modulation(t_se, field, mode="se").numpy()
    t2 = brentq(lambda T2: mono_t2(t_cpmg, m_cpmg * np.exp(-t_cpmg / T2)) - 0.145, 0.05, 2.0)
    print(f"J modulation alone: CPMG 8 ms at echoes 1 / 5 / 10 / 17: {np.round(m_cpmg[[0, 4, 9, 16]], 3)}; "
          f"single echo at TE 10 / 20 / 40 / 80 / 160 ms: {np.round(m_se, 3)}")
    print(f"intrinsic T2 that makes CPMG (8 ms, 17 echoes) read 145 ms: {t2 * 1e3:.0f} ms")
    print(f"predicted single-echo apparent T2 (TE 10-160 ms): {mono_t2(t_se, m_se * np.exp(-t_se / t2)) * 1e3:.0f} ms "
          f"(measured at 3 T: 48 ms Jordan 2013, 68 ms de Bazelaire 2004)")
    print("J modulation at ~113 ms (fastMRI's effective TE) by echo spacing (a fixed time: fits over 17 echoes span "
          "different times at each spacing and the modulation oscillates):")
    for esp in (6e-3, 7.6e-3, 8e-3, 10.26e-3, 12e-3, 16e-3):
        k = int(round(0.113 / esp))
        m = fat_modulation(esp * np.arange(1, k + 1), field, mode="cpmg", esp=esp).numpy()[-1]
        print(f"  CPMG {esp * 1e3:5.2f} ms: echo {k} at {k * esp * 1e3:.0f} ms, {m:.3f}")
    se100 = fat_modulation([0.1], field, mode="se").numpy()[0]
    fse100 = fat_modulation(10.26e-3 * np.arange(1, 11), field, mode="cpmg", esp=10.26e-3).numpy()[-1]
    print(f"fat at TE ~100 ms, J only: CPMG 10.26 ms / single echo = {fse100 / se100:.2f} "
          f"(Henkelman 1992: up to ~2 in vivo)")


if __name__ == "__main__":
    main()
