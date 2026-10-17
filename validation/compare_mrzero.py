"""Cross-validation: blochparty vs. MRzero on identical Pulseq files.

    uv sync --extra pulseq --extra mrzero
    uv run python validation/compare_mrzero.py            # writes validation/RESULTS.md

Each sequence is written with pypulseq to a ``.seq`` file, which both simulators read
(blochparty: ``read_pulseq``; MRzero: ``Sequence.import_file``, pulseq-rs backend), and
simulated on the same tissue-labelled Shepp-Logan phantom with a B0 map and 4 coils.

Apples to apples:
* Isochromats: MRzero is run point-like (dephasing function 1, no Nyquist cut-off), so
  each voxel is one isochromat at its centre, as in blochparty. Separately, MRzero's
  box voxel model is compared with blochparty using K x K jittered isochromats per voxel.
* PDG pruning: state limit 10^7, every threshold 1e-12, one pre-pass per tissue (the
  default mean-tissue pre-pass drops long-T2 CSF states); MRzero's defaults are reported
  for comparison.
* Pulses: MRzero applies RF as instantaneous rotations at the pulse centre, so all
  pulses are 0.2 us hard pulses (relaxation and off-resonance during them < 1e-5).
* Phases: MRzero takes Pulseq RF/ADC phases as written and adds pi/2 to the receiver
  phase, so blochparty reads with ``convention="mrzero"`` and the signals then agree up
  to the constant factor -i (checked, not assumed: the fitted factor is reported).
* Ordering: both return ADC samples in time order; MRzero's own k-space trajectory at
  its samples is compared with pypulseq's ``calculate_kspace``.
An exact matrix-exponential Bloch solution on a few voxels decides who owns any residual.
MRzero computes in float32; a test-only patch runs its main pass in float64 for comparison.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pypulseq as pp
import torch

import blochparty as bp
from blochparty.pulseq import read_pulseq

FOV = 0.22
# 0.1 us RF / block raster so pulses can be 0.2 us long; generous gradient limits
SYS = pp.Opts(rf_dead_time=0, rf_ringdown_time=0, adc_dead_time=0, rf_raster_time=1e-7,
              block_duration_raster=1e-7, max_grad=80, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s")
T_RF = 0.2e-6


def _hard(flip_deg, phase=0.0, use="excitation"):
    return pp.make_block_pulse(math.radians(flip_deg), duration=T_RF, phase_offset=phase, system=SYS, use=use)


def _delay(t):
    return pp.make_delay(round(t / SYS.block_duration_raster) * SYS.block_duration_raster)


def _readout(n, dwell, sign=1.0):
    """Readout trapezoid + ADC on its flat top, and the prephaser area that puts sample i
    at kx = (i - n/2) / FOV (blochparty's grid)."""
    dk = 1 / FOV
    gx = pp.make_trapezoid("x", flat_area=sign * n * dk, flat_time=n * dwell, system=SYS)
    adc = pp.make_adc(n, dwell=dwell, delay=gx.rise_time, system=SYS)
    pre_area = sign * (-(n / 2 + 0.5) * dk) - gx.amplitude * gx.rise_time / 2
    return gx, adc, pre_area


def gre_rf_spoiled(n=32, tr=8e-3, flip=15.0, n_dummy=8):
    """Spoiled GRE: x spoiler of 2 cycles/voxel, ky rewinder, 117 deg quadratic RF spoiling."""
    dk, seq = 1 / FOV, pp.Sequence(SYS)
    gx, adc, pre = _readout(n, 10e-6)
    gx_pre = pp.make_trapezoid("x", area=pre, duration=0.6e-3, system=SYS)
    gx_spoil = pp.make_trapezoid("x", area=2 * n * dk, duration=0.8e-3, system=SYS)
    used = T_RF + 0.6e-3 + pp.calc_duration(gx) + 0.8e-3
    phase, inc = 0.0, 0.0
    for i in range(-n_dummy, n):
        line = max(i, 0) - n // 2
        seq.add_block(_hard(flip, phase))
        gy_pre = pp.make_trapezoid("y", area=line * dk, duration=0.6e-3, system=SYS)
        seq.add_block(gx_pre, gy_pre)
        adc.phase_offset = phase
        seq.add_block(gx, *([adc] if i >= 0 else []))
        seq.add_block(gx_spoil, pp.scale_grad(gy_pre, -1))  # exact rewinder, also as written to file
        seq.add_block(_delay(tr - used))
        inc = (inc + math.radians(117)) % (2 * math.pi)
        phase = (phase + inc) % (2 * math.pi)
    return seq


def spin_echo(n=32, te=20e-3, tr=15.0):
    """90x - 180y spin echo, one line per TR, x crushers around the refocusing pulse."""
    dk, seq = 1 / FOV, pp.Sequence(SYS)
    gx, adc, pre = _readout(n, 10e-6)
    crush = pp.make_trapezoid("x", area=4 * n * dk, duration=1e-3, system=SYS)
    gx_pre = pp.make_trapezoid("x", area=-pre, duration=1e-3, system=SYS)  # before the 180: sign flips
    t_echo = gx.rise_time + (n / 2 + 0.5) * 10e-6  # readout start -> k = 0 sample
    d1 = te / 2 - T_RF - 2e-3  # 90 centre to 180 centre is te / 2
    d2 = te / 2 - T_RF / 2 - 1e-3 - t_echo
    for i in range(n):
        seq.add_block(_hard(90))
        seq.add_block(gx_pre, pp.make_trapezoid("y", area=-(i - n // 2) * dk, duration=1e-3, system=SYS))
        seq.add_block(_delay(d1))
        seq.add_block(crush)
        seq.add_block(_hard(180, math.pi / 2, use="refocusing"))
        seq.add_block(crush)
        seq.add_block(_delay(d2))
        seq.add_block(gx, adc)
        seq.add_block(_delay(tr - te - pp.calc_duration(gx) + t_echo - T_RF / 2))
    return seq


def bssfp(n=32, flip=40.0, tr=4e-3, n_dummy=16):
    """Balanced SSFP with 0/180 RF phase cycling (ADC follows) and an alpha/2 preparation."""
    dk, seq = 1 / FOV, pp.Sequence(SYS)
    gx, adc, pre = _readout(n, 10e-6)
    gx_pre = pp.make_trapezoid("x", area=pre, duration=0.5e-3, system=SYS)
    gx_rew = pp.make_trapezoid("x", area=-gx.area - pre, duration=0.5e-3, system=SYS)
    used = T_RF + 1e-3 + pp.calc_duration(gx)
    seq.add_block(_hard(flip / 2, math.pi))
    seq.add_block(_delay(tr / 2 - T_RF))
    for i in range(-n_dummy, n):
        line = max(i, 0) - n // 2
        phase = math.pi * ((i + n_dummy) % 2)
        seq.add_block(_hard(flip, phase))
        gy_pre = pp.make_trapezoid("y", area=line * dk, duration=0.5e-3, system=SYS)
        seq.add_block(gx_pre, gy_pre)
        adc.phase_offset = phase
        seq.add_block(gx, *([adc] if i >= 0 else []))
        seq.add_block(gx_rew, pp.scale_grad(gy_pre, -1))
        seq.add_block(_delay(tr - used))
    return seq


def epi(n=32, dwell=5e-6):
    """Single-shot GRE-EPI, readouts alternating in x, blips in y."""
    dk, seq = 1 / FOV, pp.Sequence(SYS)
    gx, adc, pre = _readout(n, dwell)
    blip = pp.make_trapezoid("y", area=dk, system=SYS)
    seq.add_block(_hard(90))
    seq.add_block(pp.make_trapezoid("x", area=pre, duration=0.5e-3, system=SYS),
                  pp.make_trapezoid("y", area=-n / 2 * dk, duration=0.5e-3, system=SYS))
    for i in range(n):
        seq.add_block(gx if i % 2 == 0 else pp.scale_grad(gx, -1), adc)
        if i < n - 1:
            seq.add_block(blip)
    return seq


SEQUENCES = {
    "GRE, 117° RF + gradient spoiling": gre_rf_spoiled,
    "spin echo (90x-180y, crushers)": spin_echo,
    "bSSFP (0/180 phase cycling)": bssfp,
    "single-shot EPI": epi,
}


# --- phantom ---------------------------------------------------------------------------


def phantom_maps(n):
    """Shepp-Logan PD/T1/T2, a +-20 Hz B0 ramp, and 4 coil maps on the n x n grid."""
    maps = bp.shepp_logan_maps(n)
    grid = bp.grid_positions(n, FOV)
    maps["db0"] = 20.0 * (grid[..., 0] + 0.5 * grid[..., 1]) / (FOV / 2)
    maps["coils"] = bp.birdcage_sensitivities(grid.reshape(-1, 3), n_coils=4).T.reshape(4, n, n)
    return maps


def bp_phantom(maps, sub=1, seed=0):
    """Point isochromats: one per voxel centre (sub=1), or sub x sub stratified-jittered
    isochromats per voxel, each with PD / sub^2 and its voxel's B0 and coil weights."""
    n = maps["pd"].shape[-1]
    mask = maps["pd"] > 0
    pos = bp.grid_positions(n, FOV)[mask]
    take = lambda m: m[mask]
    pd, t1, t2, db0 = take(maps["pd"]), take(maps["t1"]), take(maps["t2"]), take(maps["db0"])
    coils = maps["coils"][:, mask].T
    if sub > 1:
        g = torch.Generator().manual_seed(seed)
        cell = (torch.arange(sub, dtype=torch.float64) + 0.0) / sub - 0.5  # sub-cell lower corners
        cx, cy = torch.meshgrid(cell, cell, indexing="ij")
        offs = torch.stack([cx.flatten(), cy.flatten()], -1)  # (sub^2, 2) in voxel units
        jit = torch.rand(len(pos), sub * sub, 2, generator=g, dtype=torch.float64) / sub
        xy = pos[:, None, :2] + (offs[None] + jit) * (FOV / n)
        pos = torch.cat([xy, torch.zeros_like(xy[..., :1])], -1).reshape(-1, 3)
        rep = lambda v: v.repeat_interleave(sub * sub, dim=0)
        pd, t1, t2, db0, coils = rep(pd) / sub**2, rep(t1), rep(t2), rep(db0), rep(coils)
    return bp.Phantom(pos=pos, pd=pd, t1=t1, t2=t2, db0=db0), coils


def mr0_data(maps, voxel="point"):
    import MRzeroCore as mr0

    n = maps["pd"].shape[-1]
    mask = maps["pd"] > 0
    f = lambda m: m[mask].to(torch.float32)
    pos = bp.grid_positions(n, FOV)[mask].to(torch.float32)
    nv = len(pos)
    size = torch.tensor([FOV / n, FOV / n, 1.0])
    if voxel == "point":
        dephasing = lambda k, nyq: torch.ones(k.shape[0], device=k.device)
    else:  # uniform box voxel: integral of exp(-i 2 pi k.r) over the voxel
        dephasing = lambda k, nyq: torch.prod(torch.sinc(k * size.to(k.device)), dim=1)
    return mr0.SimData(
        PD=f(maps["pd"]), T1=f(maps["t1"]), T2=f(maps["t2"]), T2dash=torch.full((nv,), 1e9),
        D=torch.zeros(nv), B0=f(maps["db0"]), B1=torch.ones(1, nv),
        coil_sens=maps["coils"][:, mask].to(torch.complex64), size=torch.tensor([FOV, FOV, 1.0]),
        affine=torch.eye(4), voxel_pos=pos, nyquist=torch.full((3,), float("inf")),
        dephasing_func=dephasing,
    )


# --- simulators --------------------------------------------------------------------------


def run_blochparty(path, maps, sub=1):
    phantom, coils = bp_phantom(maps, sub)
    seq = read_pulseq(path, convention="mrzero")
    return bp.simulate(phantom, seq, coils).signal.numpy()  # (C, n_adc)


TISSUES = {name: {"T1": t1, "T2": t2, "T2dash": 1e9, "D": 0.0} for name, (_, t1, t2) in bp.phantom.TISSUES.items()}
PRUNE = 1e-12  # "strict": state limit 10^7 and every threshold 1e-12 (exact 0 never prunes
#                decayed states, which makes long sequences explode without changing the signal)


@contextlib.contextmanager
def _mrzero_float64():
    """Run MRzero's import and main pass in float64 / complex128 (a test-only patch: both
    modules hard-code float32 / cfloat). The Rust pre-pass is untouched."""
    import types

    import MRzeroCore.sequence as mseq
    import MRzeroCore.simulation.main_pass as mmain

    shim = types.ModuleType("torch_float64")
    shim.__dict__.update(torch.__dict__)
    shim.float32, shim.cfloat = torch.float64, torch.complex128
    mseq.torch, mmain.torch = shim, shim
    try:
        yield
    finally:
        mseq.torch, mmain.torch = torch, torch


def _to_float64(data):
    import copy

    data = copy.copy(data)
    for k in ["PD", "T1", "T2", "T2dash", "D", "B0", "B1", "voxel_pos", "size"]:
        setattr(data, k, getattr(data, k).double())
    data.coil_sens = data.coil_sens.to(torch.complex128)
    return data


def run_mrzero(path, data, strict=True, float64=False):
    """strict: pruning thresholds at PRUNE; else MRzero's defaults. float64: build the graph
    from the stock import, then execute it on a float64 import (see _mrzero_float64)."""
    import MRzeroCore as mr0

    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seq = mr0.Sequence.import_file(str(path))
        k = seq.get_kspace()[:, :3].numpy()
        if strict:
            # one pre-pass per tissue: the default mean-tissue pre-pass would prune the
            # long-lived CSF states (T2 = 2 s) as decayed
            graph = mr0.compute_graph(seq, data, max_state_count=10**7, min_state_mag=PRUNE,
                                      tissues=TISSUES, threaded=False)
            run = dict(min_emitted_signal=PRUNE, min_latent_signal=PRUNE)
        else:
            graph = mr0.compute_graph(seq, data)
            run = {}
        if float64:
            with _mrzero_float64():
                seq = mr0.Sequence.import_file(str(path))
                for rep in seq:
                    rep.pulse.angle = torch.as_tensor(rep.pulse.angle, dtype=torch.float64)
                    rep.pulse.phase = torch.as_tensor(rep.pulse.phase, dtype=torch.float64)
                sig = mr0.execute_graph(graph, seq, _to_float64(data), print_progress=False, **run)
        else:
            sig = mr0.execute_graph(graph, seq, data, print_progress=False, **run)
    return sig.numpy().T, k  # (C, n_adc), MRzero's k-space at its samples (n_adc, 3)


def exact_signal(seq, phantom, coils):
    """Independent reference: for every segment, the exact propagator of the full Bloch
    equation with relaxation, exp(A dt) of the 4x4 generator in homogeneous coordinates
    (torch.linalg.matrix_exp), batched over spins. No splitting, no closed-form free runs."""
    n = len(phantom)
    r1, r2 = 1 / phantom.t1, 1 / phantom.t2
    m = torch.zeros(n, 4, 1, dtype=torch.float64)
    m[:, 2, 0], m[:, 3, 0] = phantom.pd, 1.0
    out = []
    for s in range(len(seq)):
        w = 2 * math.pi * torch.stack([seq.rf[s, 0].expand(n), seq.rf[s, 1].expand(n),
                                       phantom.pos @ seq.grad[s] + phantom.db0], -1)  # dM/dt = M x w
        a = torch.zeros(n, 4, 4, dtype=torch.float64)
        a[:, 0, 1], a[:, 0, 2], a[:, 1, 2] = w[:, 2], -w[:, 1], w[:, 0]
        a[:, 1, 0], a[:, 2, 0], a[:, 2, 1] = -w[:, 2], w[:, 1], -w[:, 0]
        a[:, 0, 0], a[:, 1, 1], a[:, 2, 2], a[:, 2, 3] = -r2, -r2, -r1, r1 * phantom.pd
        m = torch.linalg.matrix_exp(a * seq.dt[s]) @ m
        if seq.adc[s]:
            mxy = torch.complex(m[:, 0, 0], m[:, 1, 0]) * torch.exp(-1j * seq.adc_phase[s])
            out.append((coils * mxy[:, None]).sum(0))
    return torch.stack(out, 1).numpy()


def sample_voxels(maps, count=24, seed=1):
    """The maps with PD zeroed outside ``count`` random voxels, half of them CSF."""
    g = torch.Generator().manual_seed(seed)
    label = maps["label"].flatten()
    pick = []
    for want_csf in (True, False):
        is_csf = label == bp.phantom.LABELS.index("csf")
        idx = torch.nonzero((is_csf == want_csf) & (maps["pd"].flatten() > 0)).flatten()
        pick.append(idx[torch.randperm(len(idx), generator=g)[: count // 2]])
    keep = torch.zeros_like(label, dtype=torch.bool)
    keep[torch.cat(pick)] = True
    sub = dict(maps)
    sub["pd"] = torch.where(keep.reshape(maps["pd"].shape), maps["pd"], 0.0)
    return sub


# --- comparison --------------------------------------------------------------------------


def kspace_grid(sig, n, epi_order=False):
    k = sig.reshape(sig.shape[0], n, n).copy()
    if epi_order:
        k[:, 1::2] = k[:, 1::2, ::-1]  # odd EPI lines run backwards
    return k


def recon(k):
    return bp.ifft2c(torch.as_tensor(k)).numpy()


def rel_max(a, ref):
    return np.abs(a - ref).max() / np.abs(ref).max()


def metrics(s_mr, s_bp, n, epi_order=False):
    """Fitted constant c (s_mr ~ c * s_bp), max relative k-space difference and image NRMSE
    after removing the expected factor -i."""
    c = np.vdot(s_bp, s_mr) / np.vdot(s_bp, s_bp)
    ref = -1j * s_bp
    img_mr, img_bp = recon(kspace_grid(s_mr, n, epi_order)), recon(kspace_grid(ref, n, epi_order))
    nrmse = np.linalg.norm(img_mr - img_bp) / np.linalg.norm(img_bp)
    return c, rel_max(s_mr, ref), nrmse


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", default=str(Path(__file__).with_name("RESULTS.md")))
    ap.add_argument("--subs", default="1,2,4,8", help="isochromats per voxel side for the voxel study")
    args = ap.parse_args()
    n = args.n
    maps = phantom_maps(n)
    data_point = mr0_data(maps, "point")
    few = sample_voxels(maps)
    data_few = mr0_data(few, "point")
    rows, exact_rows = [], []
    tmp = Path(tempfile.mkdtemp())
    for name, make in SEQUENCES.items():
        pseq = make(n)
        assert pseq.check_timing()[0], name
        path = tmp / f"{make.__name__}.seq"
        pseq.write(str(path))
        epi_order = make is epi
        t0 = time.perf_counter()
        s_bp = run_blochparty(path, maps)
        t_bp = time.perf_counter() - t0
        t0 = time.perf_counter()
        s_mr, k_mr = run_mrzero(path, data_point)
        t_mr = time.perf_counter() - t0
        s_mr64, _ = run_mrzero(path, data_point, float64=True)
        s_mr_def, _ = run_mrzero(path, data_point, strict=False)
        assert s_mr.shape == s_bp.shape, (s_mr.shape, s_bp.shape)
        c, dk, nrmse = metrics(s_mr, s_bp, n, epi_order)
        _, dk64, nrmse64 = metrics(s_mr64, s_bp, n, epi_order)
        _, dk_def, nrmse_def = metrics(s_mr_def, s_bp, n, epi_order)
        k_pp = pseq.calculate_kspace()[0].T
        dtraj = rel_max(k_mr, k_pp)
        rows.append((name, dk, nrmse, dk64, nrmse64, dk_def, nrmse_def, c, dtraj, t_bp, t_mr))
        print(f"{name:34s} dk {dk:.1e} nrmse {nrmse:.1e} | float64 {dk64:.1e} {nrmse64:.1e} | "
              f"defaults {dk_def:.1e} {nrmse_def:.1e} | c {c:.6f} | traj {dtraj:.1e} | {t_bp:.1f}s {t_mr:.1f}s")

        # exact reference on a few voxels
        phantom, coils = bp_phantom(few)
        ref = -1j * exact_signal(read_pulseq(path, convention="mrzero"), phantom, coils.to(torch.complex128))
        e_bp = rel_max(-1j * run_blochparty(path, few), ref)
        e_mr = rel_max(run_mrzero(path, data_few)[0], ref)
        e_mr64 = rel_max(run_mrzero(path, data_few, float64=True)[0], ref)
        exact_rows.append((name, e_bp, e_mr, e_mr64))
        print(f"{'':34s} vs exact on {len(phantom)} voxels: bp {e_bp:.1e}  MRzero {e_mr:.1e}  MRzero float64 {e_mr64:.1e}")

    # voxel model: MRzero box voxels vs. blochparty with sub x sub jittered isochromats
    vox = {}
    for make in (epi, gre_rf_spoiled):
        path = tmp / f"{make.__name__}.seq"
        s_mr_box, _ = run_mrzero(path, mr0_data(maps, "box"))
        s_mr_pt, _ = run_mrzero(path, data_point)
        box_vs_point = np.linalg.norm(s_mr_box - s_mr_pt) / np.linalg.norm(s_mr_pt)
        conv = []
        for sub in [int(s) for s in args.subs.split(",")]:
            _, dk_s, nrmse_s = metrics(s_mr_box, run_blochparty(path, maps, sub), n, make is epi)
            conv.append((sub, dk_s, nrmse_s))
            print(f"{make.__name__}: box voxel vs {sub}x{sub} isochromats: dk {dk_s:.1e} nrmse {nrmse_s:.1e}")
        vox[make] = (box_vs_point, conv)

    write_results(args.out, n, rows, exact_rows, vox)


def write_results(out, n, rows, exact_rows, vox):
    lines = [
        "# Cross-validation: blochparty vs. MRzero",
        "",
        f"Generated by `validation/compare_mrzero.py` (MRzeroCore {_version('MRzeroCore')}, "
        f"pypulseq {_version('pypulseq')}). {n}×{n} Shepp-Logan (3 T tissue table), ±20 Hz B0 ramp, "
        "4 birdcage coils; every sequence written once with pypulseq and read by both simulators.",
        "",
        "## Point isochromats",
        "",
        "MRzero runs point-like (voxel dephasing function 1, no Nyquist cut-off) with pruning "
        f"thresholds at {PRUNE:g} and one pre-pass per tissue; blochparty reads with "
        "`convention=\"mrzero\"`. Differences are after removing the expected constant factor −i; "
        "*fitted c* is the least-squares factor s_MRzero ≈ c·s_blochparty. *Trajectory*: max "
        "difference between MRzero's own k-space at its ADC samples and pypulseq's "
        "`calculate_kspace`, relative to max |k| (same sample ordering, same trajectory).",
        "",
        "| sequence | max rel. k-space diff | image NRMSE | same, MRzero in float64 "
        "| MRzero default pruning | fitted c | trajectory | time bp / MRzero |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, dk, nrmse, dk64, nrmse64, dk_def, nrmse_def, c, dtraj, t_bp, t_mr in rows:
        lines.append(f"| {name} | {dk:.1e} | {nrmse:.1e} | {dk64:.1e} / {nrmse64:.1e} | "
                     f"{dk_def:.1e} / {nrmse_def:.1e} | {c.real:+.6f}{c.imag:+.6f}i | {dtraj:.1e} "
                     f"| {t_bp:.1f} s / {t_mr:.1f} s |")
    lines += [
        "",
        "Independent tie-breaker: an exact per-segment matrix-exponential solution of the Bloch "
        "equation (with relaxation) on 24 sampled voxels (12 CSF), max difference relative to its "
        "max |signal|:",
        "",
        "| sequence | blochparty vs exact | MRzero vs exact | MRzero (float64) vs exact |",
        "|---|---|---|---|",
    ]
    for name, e_bp, e_mr, e_mr64 in exact_rows:
        lines.append(f"| {name} | {e_bp:.1e} | {e_mr:.1e} | {e_mr64:.1e} |")
    lines += NOTES
    lines += [
        "",
        "## Voxel model: MRzero box voxels vs. jittered isochromats",
        "",
        "MRzero's box voxel (each state's signal weighted by sinc(k·Δx), the exact integral over a "
        "uniform square voxel; no Nyquist cut-off) against blochparty with K×K stratified-jittered "
        "isochromats per voxel (one uniform random point per sub-cell).",
        "",
        "| sequence | box vs. point voxels in MRzero | " + " | ".join(f"{s}×{s}" for s, _, _ in next(iter(vox.values()))[1]) + " |",
        "|---|---|" + "---|" * len(next(iter(vox.values()))[1]),
    ]
    for make, (box_vs_point, conv) in vox.items():
        name = next(k for k, v in SEQUENCES.items() if v is make)
        lines.append(f"| {name} | {box_vs_point:.0%} | " + " | ".join(f"{dk_s:.1e} / {nrmse_s:.1e}" for _, dk_s, nrmse_s in conv) + " |")
    lines += [
        "",
        "Entries: max rel. k-space difference / image NRMSE; the second column is the relative "
        "L2 k-space difference between MRzero's own box and point voxels (how much the voxel "
        "model matters).",
    ]
    lines += VOXEL_NOTES
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


NOTES = [
    "",
    "What the remaining differences are:",
    "- **MRzero's numerics, not physics.** Against the exact reference blochparty is at "
    "1e-9–1e-7 (6e-7 on bSSFP, below), MRzero at 1e-6–4e-4. Running MRzero's import and main pass "
    "in float64 (column 4; a test-only patch, `_mrzero_float64`) removes most of it on the GRE and "
    "spin echo, not on bSSFP: there ~4e-6 remains, about half of it off-resonance during the "
    "pulses (below); with 0.02 µs pulses 1–2e-6 is left, which we have not traced further. "
    "What is left on the RF-spoiled GRE "
    "comes from MRzero's pre-pass merging PDG states whose k-vectors agree only to float32 precision "
    "(|k| reaches ~4000 /m after 40 TRs of spoiling, where float32 resolves ~2e-4 /m) and then "
    "using the first ancestor's k: up to ~1e-4 rad of phase at the edge of the FOV.",
    "- **Finite pulses.** MRzero applies RF as instantaneous rotations at the pulse centre, so all "
    "pulses are 0.2 µs hard pulses. Off-resonance during them (±20 Hz) is a ~1e-6 effect (EPI: "
    "1e-7 without B0); blochparty's second-order Strang splitting of relaxation during RF gives "
    "the 6e-7 against the exact reference on bSSFP (4e-8 with `subdivide_rf(4)`).",
    "- **Pruning.** With MRzero's defaults (200 states, magnitude 1e-4, emitted/latent signal "
    "1e-2), the point-isochromat GRE and bSSFP lose most of their signal: those thresholds assume "
    "realistic voxels, where high-order states dephase. Here every threshold is 1e-12. MRzero's "
    "default pre-pass also uses the phantom's *mean* T1/T2 and prunes the long-lived CSF states "
    "(T2 = 2 s) as decayed; on the spin echo that alone was a 6e-5 difference (1e-4 NRMSE), so "
    "the pre-pass runs once per tissue (`compute_graph(..., tissues=...)`).",
    "- **Same file, same numbers.** Pulseq rounds amplitudes when writing, so a phase-encode "
    "rewinder built separately does not cancel its prephaser exactly. That is the same in both "
    "simulators, but it splits PDG states MRzero then merges; rewinders are built with "
    "`scale_grad(prephaser, -1)`.",
    "- **Conventions.** MRzero rotates by the Pulseq RF phase as written, tips +z to −y (it "
    "rotates the other way from blochparty) and multiplies the signal by exp(i(π/2 − ADC phase)); "
    "with `convention=\"mrzero\"` that is the constant −i, fitted above to 1e-5 or better. "
    "blochparty's default (\"physical\") reading conjugates RF and ADC phases so that a frequency "
    "offset excites the slice at +f/G; `tests/sequences/test_pulseq.py` checks that the two readings are exact "
    "mirror images. MRzero ignores ADC frequency offsets, so none are used here.",
    "- **Ordering.** Both return samples in time order; the trajectory column shows MRzero's own "
    "k-space at its samples equals pypulseq's to its float32 precision.",
]
VOXEL_NOTES = [
    "",
    "On the EPI, where a voxel never holds more than half a cycle of phase, the difference falls "
    "as ~K⁻² (stratified jitter on a smooth integrand); 1×1 is the voxel centre, which beats one "
    "random point per voxel. On the spoiled GRE the PDG holds states that have wound up to ~80 "
    "cycles across a voxel after 40 TRs: MRzero integrates them analytically, while K×K points "
    "resolve them only once K exceeds the cycles per voxel, so the difference falls as ~1/K. "
    "This is the guidance for sub-voxel isochromats: their number per voxel side must exceed the "
    "largest dephasing (in cycles per voxel) that still carries signal.",
]


def _version(pkg):
    from importlib.metadata import version

    return version(pkg)


if __name__ == "__main__":
    main()
