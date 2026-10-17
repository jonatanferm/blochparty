"""Sequence-design toolkit: constraints vs closed forms, metrics vs theory, the Cramér–Rao
bound vs its closed form, the design report, and three optimizations that must recover
optima known independently (brute-force grids over closed forms, EPG + SciPy SLSQP).

The optimizations run through the core in float64 (the checks are at 1e-6 to 1e-12), DESPOT1
also through the GPU adjoint (Metal, CUDA); for each loss, the GPU adjoint and autograd through
``simulate`` must give the same gradient to float32 accuracy.
"""

import json
import math

import pytest
import torch

import blochparty as bp
from blochparty import SequenceBuilder, simulate
from blochparty import design as d
from blochparty.constants import PULSEQ_GAMMA_BAR
from blochparty.phantom import LABELS
from blochparty.reference import spoiled_gre_signal
from blochparty.sequence import System

import design_contrast_tr as contrast_tr
import design_despot1 as despot1
import design_tse_flips as tse
from helpers import rel

G = d.GAMMA_BAR
f64 = lambda v: torch.as_tensor(v, dtype=torch.float64)
# gre_cartesian checks float(TR) on a TR that requires grad; harmless
tr_with_grad = pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad=True to a scalar")


@pytest.fixture
def gpu_simulate(gpu):
    """The simulate function of each available GPU engine (the shared ``gpu`` fixture)."""
    return gpu.simulate


HW150 = d.Hardware(max_slew=150.0)  # the violation tests below assume a 150 T/m/s scanner


def test_hardware_defaults_match_the_builders_system():
    sysm = System()
    assert d.Hardware().max_grad == pytest.approx(sysm.max_grad / PULSEQ_GAMMA_BAR * 1e3)
    assert d.Hardware().max_slew == pytest.approx(sysm.max_slew / PULSEQ_GAMMA_BAR)


def trapezoid(slew, flat_mT_m, raster=10e-6, axis=0):
    """Trapezoid sampled as segment averages on ``raster`` (as read_pulseq does): ramps of
    ``slew`` [T/m/s] up to ``flat_mT_m`` and back, between two 1 ms gaps."""
    n = round(flat_mT_m * 1e-3 / slew / raster)
    ramp = [(i + 0.5) * slew * raster for i in range(n)]  # [T/m]
    b = SequenceBuilder().delay(1e-3)
    for v in ramp + [flat_mT_m * 1e-3] * 20 + ramp[::-1]:
        g = [0.0, 0.0, 0.0]
        g[axis] = v * G
        b.delay(raster, grad=g)
    return b.delay(1e-3).build()


# --- hardware quantities vs closed forms --------------------------------------------------


def test_units_peak_b1_and_rf_energy():
    seq = SequenceBuilder().rf(1e-3, flip_deg=90.0, n_steps=1).delay(1e-3, grad=(G * 1e-3, 0.0, 0.0)).build()
    b1 = 250.0 / G * 1e6  # 90 deg in 1 ms = 250 Hz
    assert abs(float(d.b1_uT(seq)[0]) - b1) < 1e-12 and abs(b1 - 5.8717) < 1e-4
    assert abs(float(d.grad_mT_m(seq)[1, 0]) - 1.0) < 1e-12
    assert abs(float(d.rf_energy(seq)) - b1 ** 2 * 1e-3) < 1e-12
    assert abs(float(d.rf_power(seq)) - b1 ** 2 / 2) < 1e-12  # over the whole 2 ms


def test_windowed_rf_power():
    # pulses of 1 ms at t = 0 and t = 5 ms; a 2 ms window holds at most one pulse
    b = SequenceBuilder()
    for _ in range(2):
        b.rf(1e-3, flip_deg=90.0, n_steps=1).delay(4e-3)
    p = d.rf_power(b.build(), window=2e-3)
    b1sq = (250.0 / G * 1e6) ** 2
    assert abs(float(p.max()) - b1sq * 1e-3 / 2e-3) < 1e-9
    assert abs(float(p[1]) - 0.0) < 1e-12  # window [1 ms, 3 ms]: no RF
    assert abs(float(p[-1]) - 0.0) < 1e-12  # last window, past the end: averaged over 2 ms


def test_slew_of_raster_sampled_trapezoid_is_exact():
    seq = trapezoid(slew=100.0, flat_mT_m=20.0)
    s = d.slew_rate(seq)[:, 0].abs()
    assert abs(float(s.max()) - 100.0) < 1e-9  # inside the ramps
    assert abs(float(d.grad_mT_m(seq).abs().max()) - 20.0) < 1e-12
    report = d.design_report(seq, psf=False, hardware=HW150)
    assert report["ok"], report["violations"]
    assert abs(report["constraints"]["slew x"]["margin_pct"] - 100 * (1 - 100 / 150)) < 1e-6


def test_instantaneous_steps_are_reported_with_segments_and_excess():
    # the builders' instantaneous gradients: a step counts as happening within one raster time
    seq = SequenceBuilder().delay(1e-3).delay(1e-3, grad=(G * 20e-3, 0.0, 0.0)).delay(1e-3).build()
    s = d.slew_rate(seq)[:, 0]
    assert abs(float(s[1]) - 20e-3 / 10e-6) < 1e-6 and abs(float(s[2]) + 2000.0) < 1e-6
    checks = {c.name: c for c in d.check_hardware(seq, HW150)}
    items = d.violations(checks["slew x"], seq)
    assert [i["segments"] for i in items] == [[1, 2]]  # boundaries 1 and 2: one run
    assert items[0]["message"] == ("slew x exceeded in boundaries 1–2 (t = 1.000–2.000 ms) by 1.23e+03 % "
                                   "(2000 T/m/s > 150 T/m/s)")
    # a ramp 12 % too steep over boundaries 101..119 of a raster-sampled trapezoid
    seq = trapezoid(slew=168.0, flat_mT_m=168.0 * 20 * 10e-6 * 1e3)
    report = d.design_report(seq, psf=False, hardware=HW150)
    slew = report["constraints"]["slew x"]
    assert not slew["ok"] and abs(slew["margin_pct"] + 12.0) < 1e-6
    msgs = [v["message"] for v in slew["violations"]]
    assert msgs[0].startswith("slew x exceeded in boundaries 2–20 (t = 1.010–1.190 ms) by 12 %")
    assert msgs[1].startswith("slew x exceeded in boundaries 42–60 (t = 1.410–1.590 ms) by 12 %")
    assert report["violations"][0].startswith("slew x exceeded at 38 boundaries (2 ranges); worst in boundaries 2–20")


@pytest.mark.slow
def test_pns_follows_the_hyperbolic_strength_duration_law():
    # one ramp: R = S/rheobase * tau / (c + tau), i.e. threshold slew rheobase * (1 + c / tau).
    # Sampled on a raster dt, the ramp's corners are rounded over dt: first-order error, < dt / tau
    hw = d.Hardware()
    slew = 30.0
    for tau in (100e-6, 360e-6, 1e-3, 3e-3):
        expected = slew / hw.pns_rheobase * tau / (hw.pns_chronaxie + tau)
        err = []
        for dt in (1e-6, 0.5e-6):
            seq = trapezoid(slew=slew, flat_mT_m=slew * tau * 1e3, raster=dt)
            r, at = d.pns(seq, hw.pns_rheobase, hw.pns_chronaxie, raster=dt)
            err.append(float(r.max()) / expected - 1)
            assert abs(err[-1]) < dt / tau, (tau, dt, err)
        assert abs(err[1] / err[0] - 0.5) < 0.02  # converges to the law as dt -> 0
    # an instantaneous step ramps over one raster time: R = dG / ((c + raster) * rheobase);
    # two axes add in quadrature
    seq = SequenceBuilder().delay(1e-3).delay(1e-3, grad=(G * 5e-3, G * 5e-3, 0.0)).build()
    r, at = d.pns(seq, hw.pns_rheobase, hw.pns_chronaxie, hw.grad_raster)
    expected = math.sqrt(2) * 5e-3 / ((hw.pns_chronaxie + hw.grad_raster) * hw.pns_rheobase)
    assert abs(float(r[0]) - expected) < 1e-12
    assert at.tolist() == [1, 2]


def test_penalties_are_zero_inside_and_differentiable_outside():
    seq = trapezoid(slew=100.0, flat_mT_m=20.0)
    assert all(float(p) == 0.0 for p in d.penalties(seq).values())
    amp = f64(300.0).requires_grad_(True)  # 300 Hz / gamma_bar = 7.05 µT

    def pen(a):
        s = SequenceBuilder().rf(1e-3, amplitude=a, n_steps=1).delay(1e-3).build()
        return d.penalties(s, d.Hardware(max_b1=5.0), only=("peak B1",))["peak B1"]

    assert float(pen(amp.detach())) > 0
    assert torch.autograd.gradcheck(pen, [amp])
    assert float(pen(f64(200.0))) == 0.0


# --- image metrics -------------------------------------------------------------------------


def test_pseudo_replica_noise_matches_ifft_theory():
    # ifftn (backward norm.) of N samples with per-component noise sigma: sigma / sqrt(N) per pixel
    seq = bp.gre_2d(32, n_dummy=0)
    sig = torch.randn(1, 32 * 32, dtype=torch.complex128, generator=torch.Generator().manual_seed(1)) * 10
    g = torch.Generator().manual_seed(2)
    complex_img = lambda s: bp.reconstruct(s, seq)[0]
    sigma = d.image_noise_sigma(sig, complex_img, 0.5, n_replicas=64, generator=g)
    assert abs(float(sigma) / (0.5 / 32) - 1) < 0.02
    # magnitude images at high SNR have the same noise
    sigma = d.image_noise_sigma(sig, lambda s: d.magnitude_image(s, seq), 0.5, n_replicas=64, generator=g)
    assert abs(float(sigma) / (0.5 / 32) - 1) < 0.02


def dirichlet(x, n):
    """|PSF| of n uniformly weighted Cartesian samples, x in pixels."""
    x = torch.as_tensor(x, dtype=torch.float64)
    out = torch.sin(math.pi * x) / (n * torch.sin(math.pi * x / n))
    return torch.where(x.abs() < 1e-12, torch.ones_like(out), out).abs()


def test_psf_is_the_dirichlet_kernel_without_decay():
    # T1 << TR: every line starts from the same Mz; T2 = inf: no weighting along the readout
    n, os = 32, 16
    seq = bp.gre_2d(n, flip_deg=30.0, tr=10e-3, n_dummy=2)
    psf = d.point_spread_function(seq, t1=1e-3, t2=1e6, oversample=os)
    x = (torch.arange(n * os, dtype=torch.float64) - n * os // 2) / os
    assert float((psf[n * os // 2] - dirichlet(x, n)).abs().max()) < 1e-9
    assert float((psf[:, n * os // 2] - dirichlet(x, n)).abs().max()) < 1e-9
    m = d.psf_metrics(psf, os)
    # sinc: FWHM 1.2067 px, first sidelobe 21.7 % (-13.3 dB); the Dirichlet kernel for n = 32 is
    # within 1e-3 of those, and the sampled metrics within ~1e-3 more
    lo, hi = 0.5, 0.7  # exact half-maximum point of the Dirichlet kernel by bisection
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if float(dirichlet(mid, n)) > 0.5 else (lo, mid)
    side = float(dirichlet(torch.linspace(1, 2, 100001), n).max())
    for ax in "xy":
        assert abs(float(m[f"fwhm_{ax}"]) - 2 * lo) < 2e-3 and abs(2 * lo - 1.2067) < 2e-3
        assert abs(float(m[f"sidelobe_{ax}"]) - side) < 1e-3 and abs(side - 0.2172) < 1e-3


def test_psf_with_t2_decay_matches_closed_form_weighting():
    # readout weighting exp(-i * dwell / T2) broadens the PSF along x only
    n, os, dwell, t2 = 32, 8, 100e-6, 2e-3
    seq = bp.gre_2d(n, flip_deg=30.0, tr=10e-3, dwell=dwell, n_dummy=2)
    psf = d.point_spread_function(seq, t1=1e-3, t2=t2, oversample=os)
    w = torch.zeros(n * os, dtype=torch.complex128)
    w[n * os // 2 - n // 2: n * os // 2 + n // 2] = torch.exp(-torch.arange(n, dtype=torch.float64) * dwell / t2)
    ref = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(w))).abs()
    assert float((psf[n * os // 2] - ref / ref.max()).abs().max()) < 1e-9
    m = d.psf_metrics(psf, os)
    assert float(m["fwhm_x"]) > 1.3 > 1.25 > float(m["fwhm_y"])


# --- Cramér-Rao bound ----------------------------------------------------------------------


@pytest.mark.slow
def test_crb_matches_closed_form():
    flips = f64([5.0, 30.0])
    expected = float(despot1.closed_form_cov(*torch.deg2rad(flips)))
    for sim, method in ((simulate, "fd"), (simulate, "autograd")):
        got = float(despot1.t1_crb(flips, sim, method=method).detach())
        assert abs(got / expected - 1) < 1e-7, (method, got, expected)


def test_finite_difference_jacobians_converge_at_their_order():
    f = lambda th: torch.stack([torch.sin(th[0]) * torch.exp(th[1]), th[0] ** 3 / th[1]])
    theta = f64([0.7, 0.3])
    exact = torch.autograd.functional.jacobian(f, theta)
    for method, order in (("fd", 2), ("fd4", 4)):
        e1, e2 = (float((d.jacobian(f, theta, method=method, rel_step=h) - exact).abs().max()) for h in (1e-2, 5e-3))
        assert abs(math.log2(e1 / e2) - order) < 0.1, (method, e1, e2)


def test_despot1_optimum_through_the_gpu_adjoint(gpu_simulate):
    # float32: fourth-order differences for the bound's Jacobian; the bound is flat along a2, so the pair
    # moves by up to 0.03 deg with the step while its closed-form bound stays within 3e-6 of the optimum
    # (the core in float64: 0.0002 deg, but 74 s)
    res = despot1.optimize(simulate=gpu_simulate, rel_step=despot1.METAL_STEP, method="fd4")
    (g1, g2), gcov = despot1.grid_optimum()
    assert abs(res["flips"][0] - g1) < 0.05 and abs(res["flips"][1] - g2) < 0.05, (res, g1, g2)
    assert float(despot1.closed_form_cov(*torch.deg2rad(f64(res["flips"])))) / gcov - 1 < 1e-5


def test_crb_gradient_gpu_and_differences_match_autograd(gpu_simulate):
    # d CRB / d flips: central differences in T1 through the core, and fourth-order ones through the
    # Metal adjoint (1.5e-5 measured), against the exact Jacobian's double backward through the core
    grads = []
    for sim, method, kw in ((simulate, "autograd", {}), (simulate, "fd", {}),
                            (gpu_simulate, "fd4", {"rel_step": despot1.METAL_STEP})):
        flips = f64([5.0, 30.0]).requires_grad_(True)
        (g,) = torch.autograd.grad(despot1.t1_crb(flips, sim, method=method, n_dummy=40, **kw), flips)
        grads.append(g)
    assert rel(grads[1], grads[0]) < 1e-6
    assert rel(grads[2], grads[0]) < 1e-4


# --- known optima --------------------------------------------------------------------------


@pytest.mark.slow
@tr_with_grad
def test_contrast_with_free_tr_under_scan_time_budget():
    # the core: through the float32 adjoint the flip is found (0.003 deg) but the constraint only to 8e-5
    budget = (contrast_tr.N_DUMMY + contrast_tr.N) * 10e-3
    res = contrast_tr.optimize(budget, simulate=simulate)
    flip, tr, _ = contrast_tr.grid_optimum(budget)
    assert abs(flip - 16.109) < 0.001 and abs(tr - 10e-3) < 1e-12  # the boundary, and the known optimum
    assert abs(res["flip"] - flip) < 0.01 and abs(res["tr"] - tr) < 1e-7, res
    assert res["duration"] <= budget * (1 + 1e-6)


@tr_with_grad
def test_contrast_gradient_gpu_equals_simulate(gpu_simulate):
    grads = []
    for sim in (gpu_simulate, simulate):
        flip, u = f64(20.0).requires_grad_(True), f64(2.0).requires_grad_(True)
        seq = contrast_tr.sequence(flip, contrast_tr.tr_of(u))
        grads.append(torch.stack(torch.autograd.grad(contrast_tr.image_contrast(seq, sim).double(), [flip, u])))
    assert rel(grads[0], grads[1]) < 5e-5  # float32: 4.2e-6 measured


def test_tse_low_sar_train():
    res = tse.optimize(simulate=simulate)
    flips = res["flips"]
    epg = tse.epg_echoes(flips)
    assert float((res["echoes"] - epg).abs().max()) < 1e-12  # Bloch vs EPG
    assert float((epg / tse.TARGET).min()) > 1 - 1e-6  # every echo meets its target
    assert float((epg / tse.TARGET).min()) < 1 + 1e-6  # and the binding one exactly
    assert res["energy"] < 0.3 * res["baseline_energy"]
    # constant baseline: the cheapest constant train meets the targets, 0.1 deg less does not
    base = res["baseline_flip"]
    assert bool((tse.epg_echoes([base] * tse.N_ECHO) >= tse.TARGET).all())
    assert not bool((tse.epg_echoes([base - 0.1] * tse.N_ECHO) >= tse.TARGET).all())
    # independent model and optimizer: SLSQP on EPG from the constant train
    ref = tse.slsqp_on_epg([base] * tse.N_ECHO)
    assert float((ref - flips).abs().max()) < 0.01
    assert abs(float((ref ** 2).sum() / (flips ** 2).sum()) - 1) < 1e-6


def test_tse_gradient_gpu_equals_simulate(gpu_simulate):
    grads = []
    for sim in (gpu_simulate, simulate):
        flips = torch.linspace(150.0, 60.0, tse.N_ECHO, dtype=torch.float64).requires_grad_(True)
        seq = tse.cpmg_sequence(flips.unbind())
        loss = (tse.echoes(seq, sim).double() / tse.TARGET - 1).pow(2).sum() + d.rf_energy(seq)
        (g,) = torch.autograd.grad(loss, flips)
        grads.append(g)
    assert rel(grads[0], grads[1]) < 3e-5  # float32: 2.2e-6 measured


# --- design report -------------------------------------------------------------------------


def test_design_report_is_json_and_matches_closed_forms(gpu_simulate):
    n, fov, tr, te, flip = 32, 0.22, 10e-3, 3e-3, 16.0
    maps = bp.shepp_logan_maps(n)
    phantom = bp.phantom_from_maps(maps["pd"], maps["t1"], maps["t2"], fov)
    masks = {t: maps["label"] == LABELS.index(t) for t in ("wm", "gm", "csf")}
    seq = bp.gre_2d(n, fov, flip_deg=flip, tr=tr, te=te, dwell=2e-6, n_dummy=400)
    report = d.design_report(seq, phantom, masks=masks, contrast_pairs=[("wm", "gm")], noise_sigma=1e-3,
                             simulate=gpu_simulate)
    report = json.loads(json.dumps(report))  # plain JSON types only
    assert abs(report["timing"]["duration_s"] - (400 + n) * tr) < 1e-12
    # readout gradient dk / dwell = 1 / (fov * dwell) Hz/m is the largest
    g_ro = 1 / (fov * 2e-6) / G * 1e3
    assert abs(report["constraints"]["gradient x"]["value"] - g_ro) < 1e-9
    assert abs(report["constraints"]["gradient x"]["margin_pct"] - 100 * (1 - g_ro / 40)) < 1e-9
    # builders: instantaneous gradients (slew, PNS) and 20 µs hard pulses (peak B1) are violations
    assert not report["ok"]
    assert {v.split(" exceeded")[0] for v in report["violations"]} == {"gradient x", "slew x", "peak B1", "PNS"}
    b1 = flip / 360 / 20e-6 / G * 1e6
    assert abs(report["constraints"]["peak B1"]["value"] - b1) < 1e-9
    # image metrics vs the closed-form steady state
    tissues = report["image"]["tissues"]
    for t in ("wm", "gm", "csf"):
        pd, t1, t2 = (maps[k][masks[t]][0] for k in ("pd", "t1", "t2"))
        expected = float(spoiled_gre_signal(pd, t1, t2, tr, te, math.radians(flip)))
        assert abs(tissues[t]["mean_signal"] / expected - 1) < 1e-3, t
        snr_eff = tissues[t]["mean_signal"] / 1e-3 / math.sqrt((400 + n) * tr)
        assert abs(tissues[t]["snr_efficiency_per_sqrt_s"] / snr_eff - 1) < 1e-6  # the same signal, float32
    wm_gm = report["image"]["contrasts"]["wm-gm"]
    assert abs(wm_gm["cnr"] - (tissues["wm"]["mean_signal"] - tissues["gm"]["mean_signal"]) / 1e-3) < 1e-9
    assert set(report["psf"]) >= {"fwhm_x", "fwhm_y", "sidelobe_x", "sidelobe_y"}
