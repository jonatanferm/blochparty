"""Realistic gradients: trapezoid design and limits, exact moments of the segment
representation, k-space at every ADC sample of the new builders against analytic values,
a pypulseq-written GRE read with ``read_pulseq``, and bit-identical default sequences."""

import hashlib
import math

import numpy as np
import pytest
import torch

import blochparty as bp
from blochparty.builders import bssfp_2d, epi_2d, se_2d, tse_2d
from blochparty.constants import PULSEQ_GAMMA_BAR
from blochparty.sequence import (ADC, SequenceBuilder, System, gradient_waveform, sinc_pulse,
                                 trapezoid)

SYS = System()


def moments(seq):
    """Cumulative gradient moment [1/m] at every segment end."""
    return torch.cumsum(seq.grad * seq.dt[:, None], 0)


def rf_centres(seq):
    """Centre times of the RF pulses (the middle of each run of RF segments; all pulses
    here are symmetric)."""
    t_end = torch.cumsum(seq.dt, 0)
    out, s = [], None
    for i, r in enumerate(seq.is_rf + [False]):
        if r and s is None:
            s = float(t_end[i] - seq.dt[i])
        elif not r and s is not None:
            out.append(0.5 * (s + float(t_end[i - 1])))
            s = None
    return torch.tensor(out, dtype=torch.float64)


def moment_at(seq, t):
    """Gradient moment [1/m] at times ``t`` (piecewise-constant segments, exact)."""
    t_end = torch.cumsum(seq.dt, 0)
    m = torch.cat([torch.zeros(1, 3, dtype=torch.float64), moments(seq)])
    i = torch.searchsorted(t_end, torch.as_tensor(t, dtype=torch.float64) - 1e-13)
    return m[i] + seq.grad[i] * (torch.as_tensor(t, dtype=torch.float64) - (t_end[i] - seq.dt[i]))[:, None]


def k_at_adc(seq, refocus=()):
    """k at every ADC sample, from the last excitation centre; at the centre of each
    refocusing pulse (indices into ``rf_centres``) the moment so far is negated, as a 180 does."""
    t_adc = torch.cumsum(seq.dt, 0)[torch.tensor(seq.adc)]
    tc = rf_centres(seq)
    m_c, m_adc = moment_at(seq, tc), moment_at(seq, t_adc)
    out, k_c, last, j = [], None, None, 0  # k_c: effective k at the last pulse centre
    for i, t in enumerate(t_adc.tolist()):
        while j < len(tc) and tc[j] < t:
            k_c = -(k_c + m_c[j] - m_c[last]) if j in refocus else torch.zeros(3, dtype=torch.float64)
            last, j = j, j + 1
        out.append(k_c + m_adc[i] - m_c[last])
    return torch.stack(out)


def grid_k(n, fov, index=None):
    """(n*n, 3) Cartesian k of each sample: raster order, or placed by ``index``."""
    dk = 1.0 / fov
    ky, kx = torch.meshgrid(*(2 * [(torch.arange(n, dtype=torch.float64) - n // 2) * dk]), indexing="ij")
    k = torch.stack([kx, ky, torch.zeros_like(kx)], -1).reshape(-1, 3)
    return k if index is None else k[index]


# --- trapezoids and limits ------------------------------------------------------------


@pytest.mark.parametrize("kw", [
    dict(axis="x", area=50.0),  # triangle
    dict(axis="y", area=-2000.0),  # trapezoid at max_grad
    dict(axis="z", area=300.0, duration=1.5e-3),
    dict(axis="x", flat_area=290.9, flat_time=640e-6),
    dict(axis="x", amplitude=-4e5, flat_time=1e-4),
    dict(axis=None, area=(120.0, -45.0, 7.5)),
    dict(axis=None, area=(0.0, 333.0, -1000.0), duration=2e-3),
])
def test_trapezoid_area_limits_and_segments(kw):
    axis = kw.pop("axis")
    g = trapezoid(axis, **kw)
    amp = g.amplitude.abs().max()
    assert amp <= SYS.max_grad * (1 + 1e-12) and amp / g.rise <= SYS.max_slew * (1 + 1e-12)
    for t in (g.rise, g.flat if "duration" not in kw else g.rise, g.duration):  # on the gradient raster
        assert abs(t / SYS.grad_raster - round(t / SYS.grad_raster)) < 1e-9
    if "area" in kw:
        assert torch.allclose(g.area, torch.as_tensor(bp.sequence._vector(kw["area"], axis, torch.float64)), atol=1e-9)
    if "duration" in kw:
        assert abs(g.duration - kw["duration"]) < 1e-15
    # appended as segments: the total moment is the area, every ramp piece one raster step
    seq = SequenceBuilder().block(g).build()
    assert (moments(seq)[-1] - g.area).abs().max() < 1e-9
    assert float(seq.dt.max()) <= max(g.flat, SYS.grad_raster) + 1e-15


def test_minimum_time_trapezoid():
    # the triangle / trapezoid boundary and the minimum ramp, worked out by hand
    tri = trapezoid("x", area=50.0)
    r = math.ceil(math.sqrt(50.0 / SYS.max_slew) / SYS.grad_raster) * SYS.grad_raster
    assert tri.flat == 0 and abs(tri.rise - r) < 1e-15 and abs(float(tri.amplitude[0]) - 50.0 / r) < 1e-6
    big = trapezoid("x", area=2000.0)
    assert abs(big.rise - 240e-6) < 1e-15  # 1.70304e6 / 7.23792e9 = 235.3 us -> 240 us
    assert abs(big.flat - 940e-6) < 1e-15  # 2000 / max_grad - rise = 934 us -> 940 us
    # no shorter trapezoid exists: one raster step less in flat time exceeds max_grad
    assert 2000.0 / (big.rise + big.flat - SYS.grad_raster) > SYS.max_grad
    # amplitude-given trapezoids ramp at the slew limit, rounded up to the raster
    assert abs(trapezoid("x", amplitude=SYS.max_grad, flat_time=0.0).rise - 240e-6) < 1e-15


def test_limits_raise():
    with pytest.raises(ValueError, match="max_grad"):
        trapezoid("x", amplitude=1.1 * SYS.max_grad, flat_time=1e-3)
    with pytest.raises(ValueError, match="slew"):
        trapezoid("x", amplitude=1e6, flat_time=1e-3, rise_time=10e-6)
    with pytest.raises(ValueError, match="does not fit"):
        trapezoid("x", area=1000.0, duration=0.5e-3)
    with pytest.raises(ValueError, match="max_grad"):  # fits the slew rate, but needs 2.1 MHz/m
        trapezoid("x", area=1500.0, duration=1e-3)
    ramp = np.linspace(0, 1, 11)[:, None] * np.array([[1e6, 0.0, 0.0]])
    gradient_waveform(np.concatenate([ramp * 0.7, ramp[::-1] * 0.7]))  # 7e9 Hz/m/s: fine
    with pytest.raises(ValueError, match="slew"):
        gradient_waveform(np.concatenate([ramp, ramp[::-1]]))  # 1e10 Hz/m/s
    with pytest.raises(ValueError, match="max_grad"):
        gradient_waveform(np.full((5, 3), 2e6), raster=1e-3)
    with pytest.raises(ValueError, match="raster"):
        bp.gre_2d(8, ramps=True, dwell=1.05e-6, system=System(max_grad=1e9))
    with pytest.raises(ValueError, match="max_grad"):  # 4.5 MHz/m readout
        bp.gre_2d(8, ramps=True, dwell=1e-6)
    assert System().max_grad == pytest.approx(40e-3 * PULSEQ_GAMMA_BAR)


def test_gradient_waveform_and_adc_moments_exact():
    # a smooth arbitrary waveform on all three axes, ADC samples off the raster: the moment
    # at each sample equals the integral of the piecewise-linear waveform
    t = np.arange(41) * SYS.grad_raster
    w = np.stack([3e5 * np.sin(np.pi * t / t[-1]) ** 2, 2e5 * np.sin(2 * np.pi * t / t[-1]),
                  -1e5 * np.sin(np.pi * t / t[-1]) ** 3], -1)
    adc = ADC(23, 17e-6, delay=3e-6)
    seq = SequenceBuilder().block(gradient_waveform(w), adc=adc).build()
    k = moments(seq)[torch.tensor(seq.adc)].numpy()
    fine = np.linspace(0, t[-1], 400_001)  # trapezoid rule on the linear interpolant is exact at nodes
    wf = np.stack([np.interp(fine, t, w[:, i]) for i in range(3)], -1)
    cum = np.concatenate([np.zeros((1, 3)), np.cumsum(0.5 * (wf[1:] + wf[:-1]) * np.diff(fine)[:, None], 0)])
    ref = np.stack([np.interp(adc.times, fine, cum[:, i]) for i in range(3)], -1)
    assert np.abs(k - ref).max() < 1e-6  # limited by the reference's interpolation of k
    # exact check against the closed form on one piece: a linear ramp from 0
    g = trapezoid("x", amplitude=4e5, flat_time=50e-6)
    a = ADC(9, 13e-6, delay=1e-6)
    seq = SequenceBuilder().block(g, adc=a).build()
    amp, r = 4e5, g.rise
    kx = [amp * s * s / (2 * r) if s < r else amp * r / 2 + amp * (s - r) if s < r + g.flat else
          amp * (r / 2 + g.flat) + amp * (s - r - g.flat) - amp * (s - r - g.flat) ** 2 / (2 * r) for s in a.times]
    assert np.abs(moments(seq)[torch.tensor(seq.adc), 0].numpy() - kx).max() < 1e-9


# --- k-space of the builders --------------------------------------------------------------


@pytest.mark.parametrize("kw", [
    dict(ramps=True),
    dict(ramps=True, spoiling="rf", n_dummy=2),
    dict(spoiling="gradient", te=2.5e-3),
    dict(ramps=True, slice_thickness=5e-3, slice_offset=2e-3, spoiling="rf"),
])
def test_gre_kspace_and_per_tr_moment(kw):
    n, fov = 16, 0.22
    seq = bp.gre_2d(n, fov, **kw)
    k = k_at_adc(seq)
    assert (k - grid_k(n, fov)).abs().max() < 1e-9
    # TE: from the pulse centre to the k = 0 sample; TR between pulse centres
    tc = rf_centres(seq)[kw.get("n_dummy", 0):]
    t_adc = torch.cumsum(seq.dt, 0)[torch.tensor(seq.adc)].reshape(n, n)
    assert (t_adc[:, n // 2] - tc - seq.meta["te"]).abs().max() < 1e-12
    assert (tc.diff() - seq.meta["tr"]).abs().max() < 1e-12
    if kw.get("spoiling", "ideal") != "ideal":  # net x moment per TR: 4 cycles across a voxel, none on y
        per_tr = moment_at(seq, tc).diff(dim=0)
        assert (per_tr[:, 0] - 4 * n / fov).abs().max() < 1e-9 and per_tr[:, 1].abs().max() < 1e-9


def test_gre_3d_slab_kspace():
    seq = bp.gre_3d(6, 0.2, ramps=True, slice_thickness=0.15, spoiling="gradient")
    k = k_at_adc(seq)
    dk = 5.0
    kz, ky, kx = torch.meshgrid(*(3 * [(torch.arange(6, dtype=torch.float64) - 3) * dk]), indexing="ij")
    assert (k - torch.stack([kx, ky, kz], -1).reshape(-1, 3)).abs().max() < 1e-9


@pytest.mark.parametrize("kw", [dict(), dict(slice_thickness=5e-3), dict(ramps=False, n_dummy=2)])
def test_bssfp_balanced_and_kspace(kw):
    n, fov = 16, 0.22
    seq = bssfp_2d(n, fov, **kw)
    assert (k_at_adc(seq) - grid_k(n, fov)).abs().max() < 1e-9
    # zero net moment between consecutive pulse centres (and from the prep to the first)
    tc = rf_centres(seq)
    assert moment_at(seq, tc).diff(dim=0).abs().max() < 1e-9
    assert (tc.diff()[1:] - seq.meta["tr"]).abs().max() < 1e-12  # TR
    assert abs(float(tc[1] - tc[0]) - seq.meta["tr"] / 2) < 1e-12  # alpha/2 prep at -TR/2
    t_adc = torch.cumsum(seq.dt, 0)[torch.tensor(seq.adc)].reshape(n, n)
    first = kw.get("n_dummy", 0) + 1
    assert (t_adc[:, n // 2] - tc[first:] - seq.meta["tr"] / 2).abs().max() < 1e-12  # TE = TR / 2


@pytest.mark.parametrize("kw", [dict(etl=4), dict(etl=8, ordering="centric", slice_thickness=5e-3),
                                dict(etl=16, ramps=False, refocus_flip_deg=140.0)])
def test_tse_kspace_and_echo_times(kw):
    n, fov = 16, 0.22
    seq = tse_2d(n, fov, **kw)
    etl = kw["etl"]
    tc = rf_centres(seq)
    refocus = [j for j in range(len(tc)) if j % (etl + 1)]
    k = k_at_adc(seq, refocus)
    assert (k - grid_k(n, fov, seq.meta["kspace_index"])).abs().max() < 1e-9
    # the ADC order covers every grid point once
    assert sorted(seq.meta["kspace_index"].tolist()) == list(range(n * n))
    # echoes at j * esp after the excitation centre (k = 0 sample of every readout)
    t_echo = torch.cumsum(seq.dt, 0)[torch.tensor(seq.adc)].reshape(-1, n)[:etl, n // 2] - tc[0]
    assert (t_echo - torch.tensor(seq.meta["echo_times"], dtype=torch.float64)).abs().max() < 1e-12
    # centric ordering puts ky = 0 in the first echo
    if kw.get("ordering") == "centric":
        assert seq.meta["line_echo"][n // 2] == 0 and seq.meta["te"] == seq.meta["esp"]


@pytest.mark.parametrize("ramps", [True, False])
def test_epi_kspace(ramps):
    n, fov = 16, 0.22
    seq = epi_2d(n, fov, ramps=ramps, slice_thickness=5e-3 if ramps else None)
    idx = seq.meta["kspace_index"]
    assert (k_at_adc(seq) - grid_k(n, fov)[idx]).abs().max() < 1e-9
    # every line starts one echo spacing after the previous one
    t_adc = torch.cumsum(seq.dt, 0)[torch.tensor(seq.adc)].reshape(n, n)
    assert (t_adc[1:, 0] - t_adc[:-1, 0] - seq.meta["esp"]).abs().max() < 1e-12
    if ramps:  # blips overlap the readout ramps: no gap between lines
        r = trapezoid("x", amplitude=1 / fov / 4e-6, flat_time=n * 4e-6).rise
        assert abs(seq.meta["esp"] - (2 * r + n * 4e-6)) < 1e-12


# --- the same sequence written with pypulseq ---------------------------------------------


def test_gre_matches_pypulseq_written_equivalent():
    pp = pytest.importorskip("pypulseq")
    from blochparty.pulseq import read_pulseq

    n, fov, dwell, flip, tr, thk = 8, 0.22, 10e-6, 20.0, 10e-3, 5e-3
    dk = 1 / fov
    seq = bp.gre_2d(n, fov, flip_deg=flip, tr=tr, dwell=dwell, ramps=True, spoiling="rf", slice_thickness=thk)
    opts = pp.Opts(max_grad=SYS.max_grad, grad_unit="Hz/m", max_slew=SYS.max_slew, slew_unit="Hz/m/s",
                   rf_dead_time=SYS.rf_dead_time, rf_ringdown_time=SYS.rf_ringdown_time, adc_dead_time=0,
                   grad_raster_time=SYS.grad_raster, rf_raster_time=SYS.rf_raster, adc_raster_time=SYS.adc_raster)
    # pypulseq's own designs of the slice-select pulse and the readout ...
    rf, gz, gzr = pp.make_sinc_pulse(math.radians(flip), duration=2e-3, slice_thickness=thk, apodization=0.5,
                                     time_bw_product=4, system=opts, return_gz=True, use="excitation",
                                     delay=SYS.rf_dead_time)
    gx = pp.make_trapezoid("x", flat_area=n * dk, flat_time=n * dwell, system=opts)
    # ... and lobes with the areas and durations the builder documents
    pre_x = -(n / 2 + 0.5) * dk - gx.amplitude * gx.rise_time / 2
    t_pre = seq.meta["te"] - (pp.calc_duration(rf, gz) - rf.delay - pp.calc_rf_center(rf)[0]) \
        - gx.rise_time - (n / 2 + 0.5) * dwell
    spoil_x = 4 * n / fov - (pre_x + gx.amplitude * (gx.rise_time + gx.flat_time))

    def lobes(areas, duration=None):
        # the builder plays a multi-axis lobe as one vector trapezoid (shared ramps)
        g = trapezoid(area=areas, duration=duration)
        return [pp.make_trapezoid(ch, amplitude=float(a), rise_time=g.rise, flat_time=g.flat, fall_time=g.fall,
                                  system=opts) for ch, a in zip("xyz", g.amplitude) if a != 0]

    t_spoil = trapezoid("x", area=max(abs(spoil_x), n // 2 * dk)).duration
    pseq = pp.Sequence(opts)
    for j in range(n):
        ky = (j - n // 2) * dk
        psi = math.radians((117.0 * j * (j + 1) / 2) % 360)
        rf.phase_offset = -psi  # Pulseq phase -> blochparty -phase (see pulseq.py)
        pseq.add_block(rf, gz)
        pseq.add_block(*lobes((pre_x, ky, gzr.area), t_pre))
        pseq.add_block(gx, pp.make_adc(n, dwell=dwell, delay=gx.rise_time, phase_offset=-psi, system=opts))
        sp = trapezoid(area=(spoil_x, -ky, 0.0), duration=t_spoil)
        pseq.add_block(*[pp.make_trapezoid(ch, amplitude=float(a), rise_time=sp.rise, flat_time=sp.flat,
                                           fall_time=sp.fall, system=opts) for ch, a in zip("xy", sp.amplitude[:2])
                         if a != 0])
        pseq.add_block(pp.make_delay(round((tr - (pseq.duration()[0] - j * tr)) / 1e-6) * 1e-6))
    assert abs(pseq.duration()[0] - float(seq.duration)) < 1e-9
    ref = read_pulseq(pseq)
    # the same moments at every ADC sample (from the excitation centre), and pypulseq's own k
    k_pp = pseq.calculate_kspace()[0].T
    assert np.abs(k_at_adc(seq).numpy() - k_pp).max() < 1e-9
    assert np.abs(k_at_adc(ref).numpy() - k_pp).max() < 1e-9
    # and the same signal from a few spins in and around the slice
    g = torch.Generator().manual_seed(0)
    pos = (torch.rand(40, 3, generator=g, dtype=torch.float64) - 0.5) * torch.tensor([0.2, 0.2, 0.012])
    one = torch.ones(40, dtype=torch.float64)
    ph = bp.Phantom(pos=pos, pd=one, t1=0.8 * one, t2=0.05 * one, db0=torch.linspace(-20, 20, 40, dtype=torch.float64))
    s, s_ref = bp.simulate(ph, seq).signal, bp.simulate(ph, ref).signal
    assert (s - s_ref).abs().max() < 1e-9 * s.abs().max()


# --- defaults unchanged ----------------------------------------------------------------------

# sha256 of the default builders' sequences on main before the realistic options were added
_MAIN = {
    "gre_2d_default": "443a96a6e37f71a0dab80650d01e38660691e95c26f77bd19fe405d3e8829e75",
    "gre_2d_opts": "1415f4aec90379b80f091fa869d0ffc244b1c246680831e87132c33d8693b0d4",
    "gre_3d": "508b1ea99da86d3c626e3eae1382b7561f7f68ddb5aa432597b4a35993f1b12d",
    "gre_cartesian_rect": "754fd08bab151b83226055e23027f05d5aa7a62e1eb0f56b2a23746142b4693e",
}


def _digest(seq):
    h = hashlib.sha256()
    for t in (seq.dt, seq.rf, seq.grad, seq.adc_phase):
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().contiguous().numpy().tobytes())
    for f in (seq.is_rf, seq.adc, seq.spoil):
        h.update(bytes(f))
    h.update(repr(sorted(seq.meta.items())).encode())
    return h.hexdigest()


def test_default_builders_bit_identical_to_main():
    cases = {
        "gre_2d_default": lambda: bp.gre_2d(),
        "gre_2d_opts": lambda: bp.gre_2d(16, 0.2, flip_deg=30.0, tr=8e-3, te=3e-3, dwell=5e-6, rf_duration=40e-6,
                                         rf_phase=0.3, n_dummy=3, rf_steps=1),  # 40 us: 2 segments by default since 2026-10
        "gre_3d": lambda: bp.gre_3d(8, n_dummy=2),
        "gre_cartesian_rect": lambda: bp.gre_cartesian((4, 6, 10), 0.25, dtype=torch.float32),
    }
    for name, make in cases.items():
        assert _digest(make()) == _MAIN[name], name


def test_spin_echo_meta_reconstructs_with_bp_reconstruct():
    seq = se_2d(8, te=10e-3, tr=0.1)
    assert seq.meta["kspace_shape"] == (8, 8) and seq.n_adc == 64
    assert (k_at_adc(seq, refocus=[j for j in range(16) if j % 2]) - grid_k(8, 0.22)).abs().max() < 1e-9


def test_sinc_pulse_matches_pypulseq_design():
    pp = pytest.importorskip("pypulseq")
    opts = pp.Opts(max_grad=SYS.max_grad, grad_unit="Hz/m", max_slew=SYS.max_slew, slew_unit="Hz/m/s",
                   rf_dead_time=SYS.rf_dead_time, rf_ringdown_time=SYS.rf_ringdown_time)
    rf_pp, gz_pp, gzr_pp = pp.make_sinc_pulse(math.radians(70), duration=1.6e-3, slice_thickness=3e-3,
                                              apodization=0.46, time_bw_product=5.2, system=opts, return_gz=True,
                                              use="excitation", delay=SYS.rf_dead_time)
    rf, gz, reph = sinc_pulse(70.0, 1.6e-3, slice_thickness=3e-3, apodization=0.46, time_bw_product=5.2)
    # pypulseq's waveform is our field conjugated (phase convention); real here
    assert np.abs(rf.signal[:, 0].numpy() - rf_pp.signal.real).max() < 1e-9 * np.abs(rf_pp.signal).max()
    assert abs(float(gz.amplitude[2]) - gz_pp.amplitude) < 1e-6
    assert abs(gz.rise - gz_pp.rise_time) < 1e-15 and abs(gz.flat - gz_pp.flat_time) < 1e-15
    assert abs(rf.delay - rf_pp.delay) < 1e-12 and abs(float(reph[2]) - gzr_pp.area) < 1e-9
