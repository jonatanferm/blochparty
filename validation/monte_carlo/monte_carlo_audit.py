"""Reproduce small, CPU-only diagnostics in docs/MONTE_CARLO_REVIEW.md.

Run: .venv/bin/python validation/monte_carlo_audit.py
These are estimator/analytic-integral experiments, not scanner throughput benchmarks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).parent / "real"))
from nist import _sobol_points, jackknife, mc_magnitude  # noqa: E402

import blochparty as bp
from blochparty.rng import normals


def magnitude_experiment():
    rng = np.random.default_rng(20261008)
    k, trials = 4, 200_000
    # Each component of the seed mean has SD=1; each seed component has SD=sqrt(k).
    noise = np.sqrt(k) * (rng.normal(size=(k, trials)) + 1j * rng.normal(size=(k, trials)))
    rows = []
    for amplitude in (0.0, 1.0, 2.0, 5.0, 10.0):
        images = amplitude + noise
        corrected, s2 = mc_magnitude(images)
        power = np.abs(images.mean(0)) ** 2 - s2
        rows.append(dict(amplitude=amplitude, raw_mean=float(np.abs(images.mean(0)).mean()),
                         corrected_mean=float(corrected.mean()), corrected_mean_se=float(corrected.std()/np.sqrt(trials)),
                         corrected_power_mean=float(power.mean()), corrected_power_se=float(power.std()/np.sqrt(trials)),
                         clip_fraction=float(np.mean(power <= 0))))
    return dict(seeds=k, trials=trials, mean_component_sd=1, rows=rows)


def aggregate_experiment():
    # Perfectly correlated sphere estimates: aggregate SE must equal each sphere SE.
    samples = np.repeat(np.array([1.0, 2.0, 3.0, 4.0])[:, None], 10, axis=1)
    _, marginal_se = jackknife(lambda x: x.mean(0), samples)
    _, aggregate_se = jackknife(lambda x: x.mean(0).mean(), samples)
    old = np.linalg.norm(marginal_se) / 10
    return dict(independence_formula=float(old), aggregate_jackknife=float(aggregate_se),
                underestimation_factor=float(aggregate_se/old))


def qmc_experiment():
    rows = []
    # Fourier transform of a unit cube and an inscribed radius-0.4 ball, analytic targets.
    freq = np.array([3.25, 1.5, 0.75])
    radius = 0.4
    cube_ref = np.prod(np.sinc(freq))
    z = 2 * np.pi * np.linalg.norm(freq) * radius
    ball_ref = 4*np.pi*radius**3/3 * 3*(np.sin(z)-z*np.cos(z))/z**3
    for n in (64, 256, 1024):
        estimates = {name: [] for name in ("iid", "scrambled_sobol", "current_sobol_plus_shift")}
        bad_bins = {name: [] for name in estimates}
        for seed in range(128):
            points = dict(iid=np.random.default_rng(seed).random((n, 3)),
                          scrambled_sobol=qmc.Sobol(3, scramble=True, seed=seed).random_base2(int(np.log2(n))),
                          current_sobol_plus_shift=_sobol_points(n, 1, seed)[0])
            for name, u in points.items():
                x = u - 0.5
                f = np.exp(-2j*np.pi*(x @ freq))
                estimates[name].append([f.mean(), (f*(np.linalg.norm(x, axis=1) <= radius)).mean()])
                counts = np.bincount(np.floor(n*u[:, 0]).astype(int), minlength=n)
                bad_bins[name].append(np.mean(counts != 1))
        for name, v in estimates.items():
            rmse = np.sqrt(np.mean(np.abs(np.array(v)-[cube_ref, ball_ref])**2, axis=0))
            rows.append(dict(n=n, method=name, cube_rmse=float(rmse[0]), ball_rmse=float(rmse[1]),
                             fraction_x_bins_not_singleton=float(np.mean(bad_bins[name]))))
    return dict(replicates=128, frequency_cycles=freq.tolist(), radius=radius, rows=rows)


def t2prime_experiment():
    k, t2p = 256, 0.02
    one = lambda x: torch.full((1, 1), x, dtype=torch.float64)
    t = torch.linspace(0, 3*t2p, 1001, dtype=torch.float64)
    signals = []
    for seed in range(8):
        ph = bp.phantom_from_maps(one(1), one(1), one(1), 1e-3, spins_per_voxel=(1, k),
                                  t2prime=one(t2p), seed=seed)
        signals.append((torch.exp(-2j*np.pi*ph.db0[:, None]*t)*ph.pd[:, None]).sum(0).numpy())
    values = np.array(signals)
    return dict(spins=k, seeds=8, max_seed_sd=float(values.std(0).max()),
                max_error_vs_exact=float(np.abs(values.mean(0)-np.exp(-t.numpy()/t2p)).max()))


def bridge_experiment():
    # One free segment, x(0)=0, constant gradient. Exact E exp(-i phi) = exp(-v/2).
    n = 300_000
    z = normals(123, np.arange(n), 0)
    rows = []
    for variance in (0.1, 1.0, 4.0):
        endpoint_phase = np.sqrt(3*variance/4)*z[:, 0]
        bridge_phase = np.sqrt(variance/4)*z[:, 3]
        raw = np.exp(-1j*(endpoint_phase+bridge_phase))
        conditioned = np.exp(-1j*endpoint_phase-variance/8)
        exact = np.exp(-variance/2)
        rows.append(dict(phase_variance=variance, exact=float(exact),
                         raw_real=float(raw.real.mean()), conditioned_real=float(conditioned.real.mean()),
                         raw_se=float(raw.real.std()/np.sqrt(n)), conditioned_se=float(conditioned.real.std()/np.sqrt(n)),
                         complex_variance_ratio=float(np.var(raw)/np.var(conditioned)),
                         analytic_variance_ratio=float((1-np.exp(-variance))/(np.exp(-variance/4)-np.exp(-variance)))))
    return dict(walkers=n, rows=rows)


def main():
    results = dict(environment=dict(torch=torch.__version__, scipy=scipy.__version__, numpy=np.__version__,
                                    mps_available=torch.backends.mps.is_available()),
                   magnitude=magnitude_experiment(), aggregate=aggregate_experiment(),
                   qmc=qmc_experiment(), t2prime=t2prime_experiment(), bridge=bridge_experiment())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
