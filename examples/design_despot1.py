"""Variable-flip-angle (DESPOT1) T1 mapping, designed by minimising a Cramér–Rao bound.

The problem (sequence, objective, reference optimum) is ``blochparty.design.problems.despot1``; this runs it and
checks the optimizer against the reference.

    uv run python examples/design_despot1.py [--gpu]
"""

from __future__ import annotations

import functools
import math

import torch

import blochparty as bp
from blochparty.design.problems.despot1 import (
    METAL_STEP,
    T1,
    TR,
    closed_form_cov,
    ernst_fraction,
    f64,
    grid_optimum,
    optimize,
)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--gpu", "--metal", action="store_true", help="through the GPU adjoint (float32) instead of the core"
    )
    args = p.parse_args()
    if args.gpu:  # float32: fourth-order differences with a step large against the signals' rounding
        gpu = functools.partial(bp.simulate, engine="gpu")
        res, engine = optimize(simulate=gpu, rel_step=METAL_STEP, method="fd4"), f"{bp.engines.resolve('gpu')} adjoint"
    else:
        res, engine = optimize(), "core"
    print(f"T1 {T1} s, TR {TR * 1e3:g} ms, Ernst angle {math.degrees(math.acos(math.exp(-TR / T1))):.3f} deg")
    print(
        f"gradient descent ({engine}): flips {res['flips'][0]:.3f}, {res['flips'][1]:.3f} deg, "
        f"CoV(T1)·SNR0 {res['cov']:.4f} ({res['n_evals']} evaluations, {res['seconds']:.1f} s)"
    )
    (g1, g2), gcov = grid_optimum()
    print(f"closed-form grid:                flips {g1:.3f}, {g2:.3f} deg, CoV(T1)·SNR0 {gcov:.4f}")
    excess = float(closed_form_cov(*torch.deg2rad(f64(res["flips"])))) / gcov - 1
    print(f"closed-form CoV at the found flips: {excess:.1e} above the optimum")
    fr = [ernst_fraction(a) for a in res["flips"]]
    print(f"signals at {100 * fr[0]:.2f} % and {100 * fr[1]:.2f} % of the Ernst maximum (Deoni 2003: ≈ 71 %)")
    # the bound is flat along a2: float32 rounding moves the found pair by up to 0.03 deg, at < 1e-5 of the bound
    assert max(abs(res["flips"][0] - g1), abs(res["flips"][1] - g2)) < (0.05 if args.gpu else 0.01)
    assert excess < 1e-5
    assert all(abs(f - 0.71) < 0.01 for f in fr)


if __name__ == "__main__":
    main()
