"""WM/GM contrast with flip angle *and* TR free, under a scan-time budget.

The problem (sequence, objective, reference optimum) is ``blochparty.design.problems.contrast_tr``; this runs it and
checks the optimizer against the reference.

    uv run python examples/design_contrast_tr.py
"""

from __future__ import annotations

from blochparty.design.problems.contrast_tr import N_DUMMY, N, grid_optimum, optimize


def main():
    for tr_budget in (10e-3, 25e-3):
        budget = (N_DUMMY + N) * tr_budget
        res = optimize(budget)
        print(f"budget {budget:.3f} s (TR <= {tr_budget * 1e3:g} ms):")
        print(f"  gradient descent (core):  flip {res['flip']:.3f} deg, TR {res['tr'] * 1e3:.4f} ms, "
              f"scan time {res['duration']:.4f} s, contrast {res['contrast']:.5f} "
              f"({res['n_evals']} evaluations, {res['seconds']:.1f} s)")
        flip, tr, c = grid_optimum(budget)
        print(f"  closed-form grid:         flip {flip:.3f} deg, TR {tr * 1e3:.4f} ms, contrast {c:.5f}")
        assert abs(res["flip"] - flip) < 0.01 and abs(res["tr"] - tr) < 1e-6
        assert res["duration"] <= budget * (1 + 1e-6)


if __name__ == "__main__":
    main()
