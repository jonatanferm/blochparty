"""Low-SAR turbo spin echo: a refocusing flip-angle train minimising RF energy subject to
every echo reaching a target amplitude.

The problem (sequence, objective, reference optimum) is ``blochparty.design.problems.tse_flips``; this runs it and
checks the optimizer against the reference.

    uv run python examples/design_tse_flips.py
"""

from __future__ import annotations

from blochparty.design.problems.tse_flips import (
    ESP,
    FRACTION,
    N_ECHO,
    T1,
    T2,
    TARGET,
    epg_echoes,
    optimize,
    slsqp_on_epg,
)


def main():
    res = optimize()
    flips = res["flips"]
    print(f"T1 {T1} s, T2 {T2} s, ESP {ESP * 1e3:g} ms, {N_ECHO} echoes, target {FRACTION:g} x exp(-t/T2)")
    print(f"constant train: {res['baseline_flip']:.2f} deg, RF energy {res['baseline_energy']:.2f} µT²s")
    print(f"optimized train (core, {res['n_evals']} evaluations, {res['seconds']:.1f} s):")
    print("  " + " ".join(f"{a:.1f}" for a in flips.tolist()))
    print(f"  RF energy {res['energy']:.2f} µT²s = {100 * res['energy'] / res['baseline_energy']:.1f} % of the "
          f"constant train; peak B1 {res['peak_b1']:.2f} µT")
    epg = epg_echoes(flips)
    print(f"  echoes / target, Bloch: min {float((res['echoes'] / TARGET).min()):.6f}; "
          f"EPG: min {float((epg / TARGET).min()):.6f}; Bloch vs EPG {float((res['echoes'] - epg).abs().max()):.1e}")
    ref = slsqp_on_epg([res["baseline_flip"]] * N_ECHO)  # started from the constant train
    print(f"SLSQP on EPG: energy {100 * float((ref ** 2).sum() / (flips ** 2).sum() - 1):+.1e} % vs ours, "
          f"max flip difference {float((ref - flips).abs().max()):.1e} deg")
    assert float((epg / TARGET).min()) > 1 - 1e-6
    assert float((ref - flips).abs().max()) < 0.05 and res["energy"] < res["baseline_energy"]


if __name__ == "__main__":
    main()
