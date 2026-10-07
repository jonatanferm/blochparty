"""Cross-validation of magnetization transfer: blochparty vs. BMCTool (pulseq-cest) Z-spectra.

    uv run --with bmctool==1.0.3 python validation/compare_bmctool.py [--quick]

BMCTool solves the Bloch-McConnell equations with a Pade matrix exponential per RF sample;
its MT pool is the same two-pool model (Henkelman/Graham rate W = w1^2 G) as
``blochparty.exchange``. One Pulseq file holds the Z-spectrum: per offset a continuous-wave
block pulse with that frequency offset, a z spoiler and a pseudo ADC (BMCTool records Mz there
and resets to equilibrium). blochparty reads the same file with ``read_pulseq``; the spoiler
blocks become ideal spoils, as BMCTool treats them, and each offset is simulated from
equilibrium.

The bound pool sits at -3 ppm, so the spectrum is asymmetric and a sign error in the RF
frequency, the importer or the pool shift would show. Two blochparty paths:
* the .seq file, offset by offset (importer, RF frequency from the phase ramp);
* all offsets at once as spins with dB0 = -offset under one on-resonance pulse (the same
  physics in the RF's frame), which also checks the two descriptions agree.
Lineshapes: Lorentzian (identical definitions: an exact comparison) and super-Lorentzian,
where BMCTool's 101-point rectangle sum differs from blochparty's exact quadrature by
0.5-2 % in W at 1-10 kHz (and it uses its own spline within ~150 Hz of the pool).
"""

from __future__ import annotations

import argparse
import contextlib
import math
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pypulseq as pp
import torch

import blochparty as bp
from blochparty import exchange
from blochparty.pulseq import read_pulseq

GAMMA_RAD = 267.5154109126009  # BMCTool's gamma [rad/us/T]
B0 = 3.0
F0 = GAMMA_RAD * B0 / (2 * math.pi) * 1e6  # Larmor frequency [Hz]
GAMMA_BAR = F0 / B0  # [Hz/T]
hz = lambda ppm: ppm * 1e-6 * F0  # chemical shift [ppm] -> frequency [Hz]

WATER = dict(t1=1.0, t2=0.05)
MT = dict(f=0.1351, k=30.0, t1=1.0, dw_ppm=-3.0)
T2B = {"lorentzian": 40e-6, "superlorentzian": 12e-6}
B1_UT, T_SAT, RF_RASTER = 1.5, 1.0, 10e-6


def write_seq(path: Path, offsets_ppm) -> None:
    sys = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=130, slew_unit="T/m/s", rf_ringdown_time=0,
                  rf_dead_time=0, rf_raster_time=RF_RASTER, gamma=GAMMA_BAR, B0=B0)
    seq = pp.Sequence(system=sys)
    flip = 2 * math.pi * GAMMA_BAR * B1_UT * 1e-6 * T_SAT
    spoil = pp.make_trapezoid("z", amplitude=0.8 * sys.max_grad, duration=5e-3, system=sys)
    adc = pp.make_adc(num_samples=1, duration=1e-3, system=sys)
    for ppm in offsets_ppm:
        seq.add_block(pp.make_block_pulse(flip, duration=T_SAT, freq_offset=hz(ppm), system=sys))
        seq.add_block(spoil)
        seq.add_block(adc)
    seq.set_definition("offsets_ppm", list(offsets_ppm))
    seq.write(str(path))


def run_bmctool(path: Path, lineshape: str) -> np.ndarray:
    from bmctool.parameters import Parameters
    from bmctool.simulation import BMCSim

    cfg = {"water_pool": dict(f=1.0, **WATER),
           "mt_pool": dict(f=MT["f"], t1=MT["t1"], t2=T2B[lineshape], k=MT["k"], dw=MT["dw_ppm"], lineshape=lineshape),
           "b0": B0, "gamma": GAMMA_RAD, "b0_inhom": 0.0, "rel_b1": 1.0,
           "verbose": False, "reset_init_mag": True, "scale": 1.0, "max_pulse_samples": 10**6}
    sim = BMCSim(Parameters.from_dict(cfg), str(path), verbose=False)
    sim.run()
    return sim.get_zspec(return_abs=False)[1]


def bmctool_superlorentzian(delta: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Replica of BMCTool 1.0.3's super-Lorentzian (``get_mt_shape_at_offset``): a 101-point
    rectangle rule, and within w0 = gamma B0 [numerically rad/us] of the pool a cubic Hermite
    spline through four such values; offsets ``delta`` in Hz. Checked against BMCTool's own
    function in ``main``."""
    w0 = GAMMA_RAD * B0
    u = 0.01 * torch.arange(101, dtype=delta.dtype)
    s = (3 * u**2 - 1).abs()

    def rect(off):  # off [rad/s], any shape
        return math.pi * 0.01 * (math.sqrt(2 / math.pi) * t2[..., None] / s
                                 * torch.exp(-2 * (off[..., None] * t2[..., None] / s) ** 2)).sum(-1)

    dw = 2 * math.pi * delta
    px = torch.tensor([-300 - w0, -100 - w0, 100 + w0, 300 + w0], dtype=delta.dtype)
    py = [rect(x.expand_as(dw)) for x in px]
    d0, d1 = 30 * (py[1] - py[0]), 30 * (py[3] - py[2])
    c = ((dw - px[1] + 1) / (px[2] - px[1] + 1)).abs()
    spline = ((2 * c**3 - 3 * c**2 + 1) * py[1] + (-2 * c**3 + 3 * c**2) * py[2] + (c**3 - 2 * c**2 + c) * d0
              + (c**3 - c**2) * d1)
    return torch.where(dw.abs() >= w0, rect(dw), spline)


@contextlib.contextmanager
def bmctool_lineshape():
    """Run blochparty with BMCTool's super-Lorentzian, to show the lineshape is the only difference."""
    orig = exchange.lineshape
    exchange.lineshape = lambda pool, delta: (bmctool_superlorentzian(delta, pool.t2) if pool.lineshape == "superlorentzian"
                                              else orig(pool, delta))
    try:
        yield
    finally:
        exchange.lineshape = orig


def bound_pool(lineshape: str) -> bp.BoundPool:
    return bp.BoundPool(f=MT["f"], k=MT["k"], t1=MT["t1"], t2=T2B[lineshape], dw=hz(MT["dw_ppm"]), lineshape=lineshape)


def spin(db0: float | torch.Tensor, lineshape: str) -> bp.Phantom:
    db0 = torch.as_tensor(db0, dtype=torch.float64).reshape(-1)
    n = len(db0)
    return bp.Phantom(pos=torch.zeros(n, 3, dtype=torch.float64), pd=torch.ones(n, dtype=torch.float64),
                      t1=torch.full((n,), WATER["t1"], dtype=torch.float64),
                      t2=torch.full((n,), WATER["t2"], dtype=torch.float64), db0=db0, bound=bound_pool(lineshape))


def blochparty_from_seq(path: Path, lineshape: str) -> np.ndarray:
    seq = read_pulseq(str(path), rf_raster=RF_RASTER, max_rf_step=RF_RASTER)  # also the 0 ppm block pulse
    pps = pp.Sequence()
    pps.read(str(path))
    blocks = seq.meta["block_segments"]
    out, start = [], 0
    for j, idx in enumerate(pps.block_events):
        blk = pps.get_block(idx)
        a, b = blocks[j]
        if blk.gz is not None and blk.rf is None and blk.adc is None:  # BMCTool: a z gradient spoils
            seq.spoil[b - 1] = True
        if blk.adc is not None:  # Mz at the pseudo ADC, then back to equilibrium
            sl = slice(start, a)
            part = replace(seq, dt=seq.dt[sl], rf=seq.rf[sl], grad=seq.grad[sl], adc_phase=seq.adc_phase[sl],
                           is_rf=seq.is_rf[sl], adc=[False] * (a - start), spoil=seq.spoil[sl], meta={})
            out.append(float(bp.simulate(spin(0.0, lineshape), part).m[0, 2]))
            start = b
    return np.array(out)


def blochparty_as_spins(offsets_ppm, lineshape: str) -> np.ndarray:
    amp = GAMMA_BAR * B1_UT * 1e-6
    n = round(T_SAT / RF_RASTER)
    seq = bp.SequenceBuilder().rf(T_SAT, amplitude=amp, n_steps=n).delay(5e-3).spoil().build()
    ph = spin(-hz(torch.tensor(offsets_ppm, dtype=torch.float64)), lineshape)
    return bp.simulate(ph, seq).m[:, 2].numpy()


def check_replica() -> None:
    from bmctool.parameters import Parameters
    from bmctool.simulation import BlochMcConnellSolver

    cfg = {"water_pool": dict(f=1.0, **WATER), "b0": B0, "gamma": GAMMA_RAD, "b0_inhom": 0.0, "rel_b1": 1.0,
           "mt_pool": dict(f=MT["f"], t1=MT["t1"], t2=T2B["superlorentzian"], k=MT["k"], dw=0.0, lineshape="superlorentzian")}
    solver = BlochMcConnellSolver(Parameters.from_dict(cfg), 1)
    hz_grid = torch.tensor([0.0, 50.0, 127.0, 128.0, 140.0, 200.0, 1e3, 5e3, 3e4], dtype=torch.float64)
    ours = bmctool_superlorentzian(hz_grid, torch.full_like(hz_grid, T2B["superlorentzian"]))
    theirs = torch.tensor([solver.get_mt_shape_at_offset(2 * math.pi * float(h), GAMMA_RAD * B0) for h in hz_grid],
                          dtype=torch.float64)
    err = float(((ours - theirs).abs() / theirs.abs()).max())
    print(f"BMCTool super-Lorentzian replica vs BMCTool: max rel. difference {err:.1e} (0-30 kHz)")
    assert err < 1e-12, (ours, theirs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="fewer offsets on the .seq path")
    args = p.parse_args(argv)
    offsets = np.concatenate([[-300.0], np.arange(-50, -10, 10.0), np.arange(-10, 10.5, 1.0), np.arange(20, 51, 10.0)])
    seq_offsets = np.array([-300.0, -20.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 20.0])
    if args.quick:
        seq_offsets = seq_offsets[[0, 3, 6, 7]]
    check_replica()
    print(cest_comparison(args.quick), flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        full, short = Path(tmp) / "zspec.seq", Path(tmp) / "zspec_short.seq"
        write_seq(full, offsets)
        write_seq(short, seq_offsets)
        for shape in ("lorentzian", "superlorentzian"):
            t0 = time.perf_counter()
            ref = run_bmctool(full, shape)
            t_bmc = time.perf_counter() - t0
            t0 = time.perf_counter()
            ours = blochparty_as_spins(offsets, shape)
            t_ours = time.perf_counter() - t0
            patched = None
            if shape == "superlorentzian":
                with bmctool_lineshape():
                    patched = blochparty_as_spins(offsets, shape)
            t0 = time.perf_counter()
            ref_s, ours_s = run_bmctool(short, shape), blochparty_from_seq(short, shape)
            t_seq = time.perf_counter() - t0
            m0 = ref[0]
            print(f"\n{shape} (T2b {T2B[shape] * 1e6:.0f} us), Z = Mz / M0 at -300 ppm; BMCTool {t_bmc:.1f} s, "
                  f"blochparty {t_ours:.1f} s ({len(offsets)} offsets as spins), .seq path {t_seq:.0f} s")
            print("| offset [ppm] | BMCTool | blochparty | difference |\n|---|---|---|---|")
            for o, r, b in zip(offsets, ref / m0, ours / ours[0]):
                print(f"| {o:g} | {r:.5f} | {b:.5f} | {b - r:+.1e} |")
            dz = ours / ours[0] - ref / m0
            print(f"max |dZ| {np.abs(dz).max():.1e}; via .seq: max |dMz| {np.abs(ours_s - ref_s).max():.1e} "
                  f"(offsets {', '.join(f'{o:g}' for o in seq_offsets)}); MTR_asym at 3 ppm: "
                  f"BMCTool {float((ref / m0)[offsets == 3][0] - (ref / m0)[offsets == -3][0]):+.4f}, "
                  f"blochparty {float((ours / ours[0])[offsets == 3][0] - (ours / ours[0])[offsets == -3][0]):+.4f}")
            if patched is not None:
                print(f"blochparty with BMCTool's super-Lorentzian (replica): max |dZ| "
                      f"{np.abs(patched / patched[0] - ref / m0).max():.1e}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- CEST: free exchanging pools


def cest_comparison(quick: bool = False) -> str:
    """BMCTool's own 5-pool library model (water, amide, guanidine, NOE, and MT modelled as a
    free Lorentzian pool), as ``Phantom.pools`` in blochparty's exact Bloch-McConnell engine."""
    import importlib.resources

    import yaml
    from bmctool.parameters import Parameters
    from bmctool.simulation import BMCSim

    cfg_path = importlib.resources.files("bmctool") / "library" / "sim-library" / "config_5pool.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["max_pulse_samples"] = 10**6
    water = cfg["water_pool"]
    pools = tuple(bp.ExchangePool(f=q["f"], k=float(q["k"]), t1=q["t1"], t2=q["t2"], dw=hz(q["dw"]))
                  for q in cfg["cest_pool"].values())
    offsets = np.concatenate([[-300.0], np.arange(-6, 6.01, 0.25)])

    def ours_as_spins():
        amp = GAMMA_BAR * B1_UT * 1e-6
        seq = bp.SequenceBuilder().rf(T_SAT, amplitude=amp).delay(5e-3).spoil().build()  # one exact segment
        db0 = -hz(torch.tensor(offsets, dtype=torch.float64))
        n = len(db0)
        ph = bp.Phantom(pos=torch.zeros(n, 3, dtype=torch.float64), pd=torch.ones(n, dtype=torch.float64),
                        t1=torch.full((n,), water["t1"], dtype=torch.float64),
                        t2=torch.full((n,), water["t2"], dtype=torch.float64), db0=db0, pools=pools)
        return bp.simulate(ph, seq).m[:, 2].numpy()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cest.seq"
        write_seq(path, offsets)
        t0 = time.perf_counter()
        sim = BMCSim(Parameters.from_dict(cfg), str(path), verbose=False)
        sim.run()
        ref = sim.get_zspec(return_abs=False)[1]
        t_bmc = time.perf_counter() - t0
        t0 = time.perf_counter()
        ours = ours_as_spins()
        t_ours = time.perf_counter() - t0
    z_ref, z_ours = ref / ref[0], ours / ours[0]
    asym = lambda z, o: float(z[offsets == -o][0] - z[offsets == o][0])
    lines = [f"\nCEST, BMCTool's config_5pool.yaml (water + amide 3.5, guanidine 2, NOE -3, MT -3 ppm as free pools), "
             f"{B1_UT} uT block pulse {T_SAT} s; BMCTool {t_bmc:.1f} s, blochparty {t_ours:.1f} s ({len(offsets)} "
             f"offsets as spins, one exact segment each)",
             f"max |dZ| {np.abs(z_ours - z_ref).max():.1e}; MTR_asym(3.5 ppm) BMCTool {asym(z_ref, 3.5):+.5f}, "
             f"blochparty {asym(z_ours, 3.5):+.5f} (the .seq path is checked with MT above; the exact engine costs "
             f"~0.3 ms per segment with 4 pools)",
             "| offset [ppm] | BMCTool | blochparty | difference |", "|---|---|---|---|"]
    lines += [f"| {o:g} | {r:.6f} | {b:.6f} | {b - r:+.1e} |" for o, r, b in zip(offsets, z_ref, z_ours)]
    return "\n".join(lines)
