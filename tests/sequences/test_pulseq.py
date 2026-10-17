"""Pulseq import: timing, units and phase conventions against closed forms; gradient
moments against pypulseq's own ``calculate_kspace``; a pypulseq GRE against the same
GRE built with ``SequenceBuilder``; and a pypulseq Shepp-Logan GRE against the
tissue-model image."""

import math

import numpy as np
import pytest
import torch

import blochparty as bp
from blochparty.reference import spoiled_gre_signal

pp = pytest.importorskip("pypulseq")
from blochparty.pulseq import read_pulseq  # noqa: E402

# No dead times: events start exactly where the test puts them. Generous gradient limits,
# since simulation is not bound by hardware (short dwells need strong readouts).
SYS = pp.Opts(
    rf_dead_time=0,
    rf_ringdown_time=0,
    adc_dead_time=0,
    max_grad=400,
    grad_unit="mT/m",
    max_slew=2000,
    slew_unit="T/m/s",
)
# 0.1 us RF and block raster, for near-instantaneous hard pulses that can still be written to a file
FINE = pp.Opts(rf_dead_time=0, rf_ringdown_time=0, adc_dead_time=0, rf_raster_time=1e-7, block_duration_raster=1e-7)


def spins(z=None, n=1, t1=1e9, t2=1e9, db0=0.0, pd=1.0):
    f = lambda v: torch.full((n,), float(v), dtype=torch.float64)
    pos = torch.zeros(n, 3, dtype=torch.float64)
    if z is not None:
        pos[:, 2] = z
    return bp.Phantom(pos=pos, pd=f(pd), t1=f(t1), t2=f(t2), db0=f(db0))


def fid_seq(phase=0.0, adc_phase=0.0, flip=90.0, adc_delay=0.0, n=8, dwell=100e-6, gap=1e-3):
    """0.2 us hard pulse, a gap, an ADC. (A pulse of a single RF raster step loses its time
    shape in pypulseq's event library, so pulses here span at least two.)"""
    seq = pp.Sequence(FINE)
    seq.add_block(pp.make_block_pulse(math.radians(flip), duration=0.2e-6, phase_offset=phase, system=FINE))
    seq.add_block(pp.make_delay(gap))
    seq.add_block(pp.make_adc(n, dwell=dwell, delay=adc_delay, phase_offset=adc_phase, system=FINE))
    return seq


# --- conventions ---------------------------------------------------------------------


def test_hard_pulse_along_x_gives_plus_i_and_phase_signs():
    # Pulseq phase 0 tips +z to +y (Mxy = +i). Pulseq phase phi gives i*exp(-i*phi): the
    # field is the conjugate of the Pulseq waveform (see the module docstring and the
    # slice-position test below, which pins the sign). The ADC phase demodulates by
    # exp(+i*psi), so equal RF and ADC phases cancel, as RF spoiling requires.
    for phi, psi in [(0.0, 0.0), (0.7, 0.0), (-math.pi / 2, 0.0), (0.0, 0.4), (1.3, 1.3)]:
        seq = read_pulseq(fid_seq(phi, psi, n=1, dwell=1e-6, gap=1e-6))
        s = bp.simulate(spins(), seq).signal[0, 0]
        assert abs(s - 1j * np.exp(-1j * (phi - psi))) < 1e-9, (phi, psi, s)
        # MRzero's reading uses the phases as written
        seq = read_pulseq(fid_seq(phi, psi, n=1, dwell=1e-6, gap=1e-6), convention="mrzero")
        s = bp.simulate(spins(), seq).signal[0, 0]
        assert abs(s - 1j * np.exp(1j * (phi - psi))) < 1e-9, (phi, psi, s)


def test_fid_timing_relaxation_and_off_resonance(tmp_path):
    # Through a .seq file: samples at (i + 1/2) * dwell after ADC start (incl. its delay),
    # T2 decay and off-resonance precession measured from the pulse centre.
    t2, df, dwell, n, gap, delay = 0.05, 210.0, 100e-6, 16, 1e-3, 30e-6
    path = tmp_path / "fid.seq"
    fid_seq(phase=0.3, adc_phase=0.1, adc_delay=delay, n=n, dwell=dwell, gap=gap).write(str(path))
    seq = read_pulseq(path)
    s = bp.simulate(spins(t2=t2, db0=df), seq).signal[0]
    t = 0.1e-6 + gap + delay + (np.arange(n) + 0.5) * dwell  # from the pulse centre
    expected = 1j * np.exp(-1j * (0.3 - 0.1)) * np.exp(-t / t2) * np.exp(-2j * math.pi * df * t)
    # a 0.2 us pulse is not quite instantaneous: off-resonance during it acts like ~0.03 us
    # of extra precession (4e-5 rad), relaxation during it ~1e-6
    assert np.abs(s.numpy() - expected).max() < 1e-4
    # half a dwell off would be 0.07 rad of phase: far outside the tolerance
    assert np.abs(s.numpy() - expected * np.exp(-1j * math.pi * df * dwell)).max() > 1e-2
    # segment ends of the ADC samples are exactly pypulseq's ADC times
    t_end = torch.cumsum(seq.dt, 0)[torch.tensor(seq.adc)].numpy()
    assert np.abs(t_end - pp.Sequence.adc_times(_read(path))[0]).max() < 1e-12
    assert seq.meta["adc_per_readout"] == [n]


def _read(path):
    seq = pp.Sequence()
    seq.read(str(path))
    return seq


def test_adc_frequency_offset_follows_off_resonant_spins():
    # Spins at +df demodulated with ADC freq_offset = +df give a constant signal.
    df, n, dwell = 400.0, 32, 50e-6
    seq = pp.Sequence(SYS)
    seq.add_block(pp.make_block_pulse(math.pi / 2, duration=20e-6, system=SYS))
    seq.add_block(pp.make_adc(n, dwell=dwell, freq_offset=df, system=SYS))
    s = bp.simulate(spins(db0=df), read_pulseq(seq)).signal[0].numpy()
    assert np.abs(s - s[0]).max() < 1e-9
    assert abs(s[0] - 1j) < 0.05  # up to the precession accrued before the ADC starts
    # with the opposite offset the phase winds by 2 * 2 pi df t
    seq.get_block(2).adc.freq_offset = -df
    s = bp.simulate(spins(db0=df), read_pulseq(seq)).signal[0].numpy()
    assert abs(np.angle(s[1] / s[0]) + 4 * math.pi * df * dwell) < 1e-9


def test_ppm_offsets_and_adc_phase_modulation():
    # ppm offsets are weighted with the system frequency (gamma * B0), as in the Pulseq spec;
    # an ADC phase-modulation shape adds per-sample phase.
    f0_mhz = SYS.gamma * SYS.B0 * 1e-6

    def imported(rf_kw, adc_kw):
        seq = pp.Sequence(SYS)
        seq.add_block(pp.make_block_pulse(math.radians(30), duration=100e-6, system=SYS, **rf_kw))
        seq.add_block(pp.make_adc(6, dwell=20e-6, system=SYS, **adc_kw))
        return read_pulseq(seq)

    a = imported(dict(freq_ppm=-3.4, phase_ppm=0.08), dict(freq_ppm=1.2, phase_ppm=-0.3))
    b = imported(
        dict(freq_offset=-3.4 * f0_mhz, phase_offset=0.08 * f0_mhz),
        dict(freq_offset=1.2 * f0_mhz, phase_offset=-0.3 * f0_mhz),
    )
    assert torch.allclose(a.rf, b.rf, atol=1e-9) and torch.allclose(a.adc_phase, b.adc_phase, atol=1e-12)
    assert a.rf[:, 1].abs().max() > 1.0  # the offsets really are there
    mod = np.linspace(0, 1, 6)
    c = imported({}, dict(phase_modulation=mod))
    assert torch.allclose(c.adc_phase[torch.tensor(c.adc)], torch.as_tensor(-mod))


def slice_profile(z0, convention="physical"):
    """|Mxy| and Mxy along z after a sinc pulse with freq_offset = Gz * z0 and the usual
    phase_offset = -2 pi f t_centre, plus the slice rephaser."""
    thk = 5e-3
    rf, gz, gzr = pp.make_sinc_pulse(
        math.radians(30),
        duration=2e-3,
        slice_thickness=thk,
        apodization=0.5,
        time_bw_product=4,
        system=SYS,
        return_gz=True,
        use="excitation",
    )
    rf.freq_offset = gz.amplitude * z0
    rf.phase_offset = -2 * math.pi * rf.freq_offset * pp.calc_rf_center(rf)[0]
    seq = pp.Sequence(SYS)
    seq.add_block(rf, gz)
    seq.add_block(gzr)
    z = torch.linspace(-0.03, 0.03, 241, dtype=torch.float64)
    m = bp.simulate(spins(z=z, n=len(z)), read_pulseq(seq, convention=convention)).m
    mxy = torch.complex(m[:, 0], m[:, 1])
    return z, mxy


def test_rf_frequency_offset_selects_slice_at_plus_f_over_g():
    # The Pulseq definition: slice position = freq_offset / G. With the phase offset that
    # zeroes the phase at the pulse centre, the shifted slice has the centred slice's
    # profile and phase (this also pins the time reference of the phase ramp).
    z, centred = slice_profile(0.0)
    _, shifted = slice_profile(0.01)
    dz = z[1] - z[0]
    assert abs(z[shifted.abs().argmax()] - 0.01) <= dz / 2
    i0, i1 = int(torch.argmin((z - 0.0).abs())), int(torch.argmin((z - 0.01).abs()))
    span = 30  # +-7.5 mm around each centre
    # (to the error of stepping the frequency-offset phase ramp once per 1 us raster step)
    assert (shifted[i1 - span : i1 + span + 1] - centred[i0 - span : i0 + span + 1]).abs().max() < 1e-4
    # slice centre: a 30 deg pulse, rephased, along +y
    assert abs(centred[i0] - 1j * math.sin(math.radians(30))) < 2e-3
    # read with MRzero's convention, the same file excites -z0
    _, mirrored = slice_profile(0.01, convention="mrzero")
    assert abs(z[mirrored.abs().argmax()] + 0.01) <= dz / 2


def test_conventions_are_mirror_images():
    # The "physical" and "mrzero" readings conjugate every RF and ADC phase. Reflecting
    # y -> -y turns one into the other with positions and off-resonance negated, so
    # s_physical(r, dB0) = -conj(s_mrzero(-r, -dB0)) exactly, for any sequence: here RF
    # spoiling, RF and ADC frequency offsets, gradients on all axes, relaxation.
    seq = pp.Sequence(SYS)
    g = np.random.default_rng(3)
    for i in range(6):
        ph = math.radians(117) * i * (i + 1) / 2
        seq.add_block(
            pp.make_block_pulse(math.radians(40), duration=60e-6, phase_offset=ph, freq_offset=300.0 * i, system=SYS)
        )
        seq.add_block(
            *[pp.make_trapezoid(ch, area=a, duration=0.4e-3, system=SYS) for ch, a in zip("xyz", g.uniform(-60, 60, 3))]
        )
        gx = pp.make_trapezoid("x", flat_area=50.0, flat_time=200e-6, system=SYS)
        seq.add_block(
            gx, pp.make_adc(10, dwell=20e-6, delay=gx.rise_time, phase_offset=ph, freq_offset=-150.0 * i, system=SYS)
        )
        seq.add_block(pp.make_delay(2e-3))
    r = torch.as_tensor(g.uniform(-0.05, 0.05, (40, 3)))
    db0 = torch.as_tensor(g.uniform(-80, 80, 40))
    ph = lambda pos, off: bp.Phantom(
        pos=pos,
        pd=torch.ones(40, dtype=torch.float64),
        t1=torch.full((40,), 0.8, dtype=torch.float64),
        t2=torch.full((40,), 0.05, dtype=torch.float64),
        db0=off,
    )
    s_phys = bp.simulate(ph(r, db0), read_pulseq(seq)).signal
    s_mr0 = bp.simulate(ph(-r, -db0), read_pulseq(seq, convention="mrzero")).signal
    assert (s_phys + s_mr0.conj()).abs().max() < 1e-12 * s_phys.abs().max()
    # and they are not trivially equal
    assert (s_phys - s_mr0).abs().max() > 0.1 * s_phys.abs().max()


# --- gradient moments vs pypulseq ---------------------------------------------------


def moments_at_adc(seq):
    """Gradient moment at every ADC sample, from the last excitation centre (pypulseq's
    calculate_kspace definition), straight from the imported segments."""
    t_end = torch.cumsum(seq.dt, 0).numpy()
    k_end = torch.cumsum(seq.grad * seq.dt[:, None], 0).numpy()
    return t_end, k_end


def check_kspace(pseq, tol=1e-9):
    seq = read_pulseq(pseq)
    t_end, k_end = moments_at_adc(seq)
    t_exc = np.asarray(pseq.rf_times()[0])
    i_exc = np.array([np.argmin(np.abs(t_end - t)) for t in t_exc])
    assert np.abs(t_end[i_exc] - t_exc).max() < 1e-12  # the RF centre is a segment end
    adc = np.flatnonzero(seq.adc)
    last = np.searchsorted(t_end[i_exc], t_end[adc]) - 1
    k_bp = k_end[adc] - k_end[i_exc[last]]
    k_pp = pseq.calculate_kspace()[0].T
    err = np.abs(k_bp - k_pp).max()
    assert err <= tol, err
    return seq, k_bp


def gre_pulseq(n=16, fov=0.22, dwell=10e-6, tr=None, n_dummy=0, flip=15.0, selective=False, spoil=True):
    """Cartesian 2D GRE with trapezoids (ramps included). Readout prephaser chosen so
    sample i sits at k = (i - n/2) / fov, matching blochparty's grid."""
    dk = 1 / fov
    seq = pp.Sequence(SYS)
    if selective:
        rf, gz, gzr = pp.make_sinc_pulse(
            math.radians(flip),
            duration=1e-3,
            slice_thickness=5e-3,
            time_bw_product=4,
            system=SYS,
            return_gz=True,
            use="excitation",
        )
    else:
        rf, gz, gzr = pp.make_block_pulse(math.radians(flip), duration=20e-6, system=SYS, use="excitation"), None, None
    gx = pp.make_trapezoid("x", flat_area=n * dk, flat_time=n * dwell, system=SYS)
    adc = pp.make_adc(n, dwell=dwell, delay=gx.rise_time, system=SYS)
    pre_area = -(n / 2 + 0.5) * dk - gx.amplitude * gx.rise_time / 2
    gx_pre = pp.make_trapezoid("x", area=pre_area, duration=0.6e-3, system=SYS)
    gx_spoil = pp.make_trapezoid("x", area=2 * n * dk, system=SYS)
    blocks_rf = [rf] + ([gz] if gz else [])
    t_fill = 0.0
    if tr is not None:
        t_used = (
            pp.calc_duration(*blocks_rf)
            + (pp.calc_duration(gzr) if gzr else 0)
            + 0.6e-3
            + pp.calc_duration(gx, adc)
            + (pp.calc_duration(gx_spoil) if spoil else 0)
        )
        t_fill = round((tr - t_used) / SYS.block_duration_raster) * SYS.block_duration_raster
        assert t_fill > 0
    for _ in range(n_dummy):
        seq.add_block(*blocks_rf)
        seq.add_block(
            pp.make_delay(
                round((tr - pp.calc_duration(*blocks_rf)) / SYS.block_duration_raster) * SYS.block_duration_raster
            )
        )
    for i in range(n):
        gy_pre = pp.make_trapezoid("y", area=(i - n // 2) * dk, duration=0.6e-3, system=SYS)
        seq.add_block(*blocks_rf)
        if gzr:
            seq.add_block(gzr)
        seq.add_block(gx_pre, gy_pre)
        seq.add_block(gx, adc)
        if spoil:
            gy_rew = pp.make_trapezoid("y", area=-(i - n // 2) * dk, duration=0.6e-3, system=SYS)
            seq.add_block(gx_spoil, gy_rew)
        if t_fill:
            seq.add_block(pp.make_delay(t_fill))
    return seq


def epi_pulseq(n=16, fov=0.22, dwell=5e-6, ramp_sampling=False):
    """Single-shot EPI: alternating readouts, blips, with or without ramp sampling."""
    dk = 1 / fov
    seq = pp.Sequence(SYS)
    rf = pp.make_block_pulse(math.radians(90), duration=50e-6, system=SYS, use="excitation")
    if ramp_sampling:  # ADC spans the whole trapezoid, ramps included
        gx = pp.make_trapezoid("x", area=n * dk, duration=n * dwell + 2e-4, system=SYS)
        n_adc, adc_dwell = 2 * n, round(pp.calc_duration(gx) / (2 * n) / 1e-7) * 1e-7
        adc = pp.make_adc(n_adc, dwell=adc_dwell, system=SYS)
    else:
        gx = pp.make_trapezoid("x", flat_area=n * dk, flat_time=n * dwell, system=SYS)
        adc = pp.make_adc(n, dwell=dwell, delay=gx.rise_time, system=SYS)
    blip = pp.make_trapezoid("y", area=dk, system=SYS)
    seq.add_block(rf)
    seq.add_block(
        pp.make_trapezoid("x", area=-gx.area / 2, duration=0.5e-3, system=SYS),
        pp.make_trapezoid("y", area=-n / 2 * dk, duration=0.5e-3, system=SYS),
    )
    for i in range(n):
        seq.add_block(gx if i % 2 == 0 else pp.scale_grad(gx, -1), adc)
        if i < n - 1:
            seq.add_block(blip)
    return seq


def test_gre_kspace_matches_pypulseq():
    _, k = check_kspace(gre_pulseq(selective=True))
    # and lands on the Cartesian grid: kx = (i - n/2)/fov, ky = (line - n/2)/fov
    n, dk = 16, 1 / 0.22
    grid = (np.arange(n) - n // 2) * dk
    assert np.abs(k[:, 0].reshape(n, n) - grid[None, :]).max() < 1e-9
    assert np.abs(k[:, 1].reshape(n, n) - grid[:, None]).max() < 1e-9


@pytest.mark.parametrize("ramp_sampling", [False, True])
def test_epi_kspace_matches_pypulseq(ramp_sampling):
    seq, _ = check_kspace(epi_pulseq(ramp_sampling=ramp_sampling))
    assert seq.meta["adc_per_readout"] == [32 if ramp_sampling else 16] * 16


def test_arbitrary_and_extended_gradients_kspace():
    seq = pp.Sequence(SYS)
    seq.add_block(pp.make_block_pulse(math.radians(10), duration=40e-6, system=SYS, use="excitation"))
    wave = 2e5 * np.sin(np.linspace(0, 3 * math.pi, 57)) ** 2
    seq.add_block(
        pp.make_arbitrary_grad("x", wave, first=0, last=wave[-1], delay=30e-6, system=SYS),
        pp.make_adc(40, dwell=13e-6, delay=17e-6, system=SYS),
    )
    ext = pp.make_extended_trapezoid(
        "y", times=np.array([0, 1e-4, 3e-4, 3.5e-4]), amplitudes=np.array([0, 3e5, -1e5, 0]), system=SYS
    )
    seq.add_block(
        pp.make_arbitrary_grad("x", np.linspace(wave[-1], 0, 10), first=wave[-1], last=0, system=SYS),
        ext,
        pp.make_adc(25, dwell=14e-6, system=SYS),
    )
    check_kspace(seq)


# --- equivalence with SequenceBuilder -------------------------------------------------


def test_pulseq_gre_equals_builder_gre():
    # Ideal case (no ramps): constant "block" gradients written as arbitrary gradients,
    # one per block, give the same blochparty signal as the SequenceBuilder version.
    n, fov, dwell, t_pre, tr, rf_dur, flip = 8, 0.2, 10e-6, 400e-6, 3e-3, 20e-6, 30.0
    dk = 1 / fov
    pseq = pp.Sequence(SYS)
    b = bp.SequenceBuilder()
    const = lambda ch, amp, dur: pp.make_arbitrary_grad(
        ch, np.full(round(dur / SYS.grad_raster_time), amp), first=amp, last=amp, system=SYS
    )
    t_spoil, t_fill = 500e-6, tr - rf_dur - t_pre - n * dwell - 500e-6
    for i in range(n):
        phase = 0.5 * 2 * math.pi * i * (i + 1) / 2 / 3  # some RF/ADC phase cycling
        gpre = (-(n / 2 + 0.5) * dk / t_pre, (i - n // 2) * dk / t_pre)
        pseq.add_block(pp.make_block_pulse(math.radians(flip), duration=rf_dur, phase_offset=phase, system=SYS))
        pseq.add_block(const("x", gpre[0], t_pre), const("y", gpre[1], t_pre))
        pseq.add_block(const("x", dk / dwell, n * dwell), pp.make_adc(n, dwell=dwell, phase_offset=phase, system=SYS))
        pseq.add_block(const("x", 3 * n * dk / t_spoil, t_spoil))
        pseq.add_block(pp.make_delay(t_fill))
        # Pulseq phase phi -> blochparty -phi (module docstring)
        b.rf(rf_dur, flip_deg=flip, phase=-phase, n_steps=2)  # the RF centre is a breakpoint
        b.delay(t_pre, grad=(*gpre, 0.0))
        b.readout(n, dwell, grad=(dk / dwell, 0.0, 0.0), phase=-phase)
        b.delay(t_spoil, grad=(3 * n * dk / t_spoil, 0.0, 0.0))
        b.delay(t_fill)
    phantom = bp.shepp_logan(n, fov)
    phantom.t2 = phantom.t2 * 0 + 0.02  # some decay
    s_pp = bp.simulate(phantom, read_pulseq(pseq)).signal
    s_b = bp.simulate(phantom, b.build()).signal
    assert (s_pp - s_b).abs().max() < 1e-10 * s_b.abs().max()


# --- imaging ---------------------------------------------------------------------------


def test_pulseq_gre_reconstructs_tissue_model_image():
    # As in test_imaging: ideal spoiling makes the closed-form steady state exact. Pulseq
    # cannot express ideal spoiling, so it is added after import, at the end of each TR.
    n, fov, flip, tr = 32, 0.22, 15.0, 10e-3
    maps = bp.shepp_logan_maps(n)
    phantom = bp.phantom_from_maps(maps["pd"], maps["t1"], maps["t2"], fov)
    pseq = gre_pulseq(n, fov, dwell=1e-6, tr=tr, n_dummy=300, flip=flip, spoil=False)
    seq = read_pulseq(pseq, kspace_shape=(n, n))
    rf_starts = [i for i, r in enumerate(seq.is_rf) if r and (i == 0 or not seq.is_rf[i - 1])]
    for i in rf_starts[1:]:
        seq.spoil[i - 1] = True
    seq.spoil[-1] = True
    # TE: from the RF centre to the k = 0 sample of a line
    t_end = torch.cumsum(seq.dt, 0)
    t_adc = t_end[torch.tensor(seq.adc)].reshape(n, n)
    t_rf = torch.tensor(pseq.rf_times()[0][300:], dtype=torch.float64)
    te = float(t_adc[0, n // 2] - t_rf[0])
    assert torch.allclose(t_adc[:, n // 2] - t_rf, torch.full((n,), te, dtype=torch.float64), atol=1e-12)

    img = bp.reconstruct(bp.simulate(phantom, seq).signal, seq)[0]
    mxy = spoiled_gre_signal(maps["pd"], maps["t1"], maps["t2"], tr, te, math.radians(flip))
    expected = torch.where(maps["pd"] > 0, 1j * mxy, 0)
    err = (img - expected).abs().max() / expected.abs().max()
    assert err < 2e-4, err  # T2 decay during the 32 us readout blurs at the 1e-4 level


def test_unsupported_events_raise():
    seq = fid_seq()
    seq.get_block(1).rotation = object()  # e.g. a rotation extension (pypulseq caches blocks)
    with pytest.raises(NotImplementedError, match="rotation"):
        read_pulseq(seq)
    seq = fid_seq()
    seq.extension_string_idx.append("ROTATIONS")
    with pytest.raises(NotImplementedError, match="ROTATIONS"):
        read_pulseq(seq)
    # labels and triggers are ignored
    seq = fid_seq()
    seq.add_block(pp.make_label(type="SET", label="LIN", value=1), pp.make_delay(1e-4))
    seq.add_block(pp.make_trigger("physio1", duration=1e-5), pp.make_delay(1e-4))
    assert read_pulseq(seq).n_adc == 8
