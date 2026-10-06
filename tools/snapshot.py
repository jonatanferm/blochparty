"""Bit-identity guard for changes to the simulators.

Saves outputs and gradients of both simulators (and default sequences / phantoms) on a fixed
set of default-argument inputs, so a change can be checked to leave default behaviour
bit-for-bit unchanged:

    git stash / switch to the base commit
    uv run python tools/snapshot.py save /tmp/before.pt
    switch to the change
    uv run python tools/snapshot.py save /tmp/after.pt
    uv run python tools/snapshot.py compare /tmp/before.pt /tmp/after.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for tests.test_bloch_vs_ode

import blochparty as bp  # noqa: E402
from blochparty.fused import simulate_fused  # noqa: E402
from tests.test_bloch_vs_ode import random_case  # noqa: E402


def snapshot() -> dict:
    out = {}
    sims = (("torch", bp.simulate), ("fused", lambda *a, **k: simulate_fused(*a, n_threads=1, **k)))
    for name, sim in sims:
        for dt in (torch.float64, torch.float32):
            ph = bp.shepp_logan(32).to(dtype=dt)
            ph.db0 = torch.linspace(-50, 50, len(ph), dtype=dt)
            coils = bp.birdcage_sensitivities(ph.pos, 8)
            t1 = ph.t1.clone().requires_grad_()
            flip = torch.tensor(20.0, dtype=dt, requires_grad=True)
            r = sim(bp.Phantom(ph.pos, ph.pd, t1, ph.t2, ph.db0), bp.gre_2d(32, flip_deg=flip, n_dummy=50, dtype=dt),
                    coils)
            grads = torch.autograd.grad((r.signal.abs() ** 2).sum() + r.m.sum(), [flip, t1])
            out[f"{name}-2d-{dt}"] = (r.signal.detach(), r.m.detach(), *grads)
            r = sim(bp.shepp_logan(12, ndim=3).to(dtype=dt), bp.gre_3d(12, n_dummy=5, dtype=dt))
            out[f"{name}-3d-{dt}"] = (r.signal, r.m)
        for s in range(3):
            p, q = random_case(s)
            r = sim(p, q)
            out[f"{name}-rand{s}"] = (r.signal, r.m)
    seqs = {"gre2d": bp.gre_2d(32, n_dummy=5), "gre3d": bp.gre_3d(8, n_dummy=5)}
    out["seqs"] = {k: (v.dt, v.rf, v.grad, v.adc_phase, v.is_rf, v.adc, v.spoil) for k, v in seqs.items()}
    maps = bp.shepp_logan_maps(32)
    ph = bp.phantom_from_maps(maps["pd"], maps["t1"], maps["t2"], 0.22)
    out["phantom"] = (ph.pos, ph.pd, ph.t1, ph.t2, ph.db0)
    return out


def compare(a: dict, b: dict) -> list[str]:
    bad = []
    for key, va in a.items():
        vb = b.get(key)
        if vb is None:
            bad.append(f"{key}: missing")
            continue
        items = va.items() if isinstance(va, dict) else [(key, va)]
        for name, xs in items:
            ys = vb[name] if isinstance(va, dict) else vb
            for i, (x, y) in enumerate(zip(xs, ys)):
                same = torch.equal(x, y) if torch.is_tensor(x) else x == y
                if not same:
                    diff = f" (max diff {(x - y).abs().max():.1e})" if torch.is_tensor(x) and x.shape == y.shape else ""
                    bad.append(f"{name}[{i}]{diff}")
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("save").add_argument("path")
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    args = p.parse_args()
    if args.cmd == "save":
        torch.save(snapshot(), args.path)
        print(f"saved {args.path}")
    else:
        bad = compare(torch.load(args.before, weights_only=False), torch.load(args.after, weights_only=False))
        print("bit-identical" if not bad else "DIFFERENCES:\n  " + "\n  ".join(bad))
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
