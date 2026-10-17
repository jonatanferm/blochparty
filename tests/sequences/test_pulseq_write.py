"""Pulseq export (``pulseq.write_pulseq``, ROADMAP M4): written, read back by pypulseq through
``read_pulseq`` and simulated, a sequence gives the same signal to 1e-9; pypulseq's own timing
check passes on the files; and k-space at every ADC sample (``sequence.kspace_trajectory``) equals
pypulseq's ``calculate_kspace`` of the written file. Block provenance (``meta["blocks"]``) follows
the segments through ``subdivide_rf``; what cannot be written raises."""

from dataclasses import replace

import numpy as np
import pytest
import torch

import blochparty as bp
from blochparty.pulseq import flip_angle, read_pulseq, write_pulseq
from blochparty.sequence import kspace_trajectory

pp = pytest.importorskip("pypulseq")
f64 = torch.float64

CASES = {
    "gre: ramps, slice selection, RF spoiling": lambda: bp.gre_2d(16, ramps=True, spoiling="rf", slice_thickness=5e-3,
                                                                  n_dummy=2),
    "tse: slice selection, crushers, gradient spoiling": lambda: bp.tse_2d(16, etl=4, tr=0.4, slice_thickness=5e-3,
                                                                            spoiling="gradient"),
    "tse, slice 2 cm off centre (phase-ramped pulses)": lambda: bp.tse_2d(16, etl=4, tr=0.4, slice_thickness=5e-3,
                                                                          slice_offset=0.02, spoiling="gradient"),
    "bssfp: slice selection, alpha/2 preparation": lambda: bp.bssfp_2d(16, slice_thickness=5e-3, n_dummy=4),
    "epi: blipped, ramps": lambda: bp.epi_2d(16, dwell=5e-6, slice_thickness=5e-3),
}


def phantom():
    ph = bp.shepp_logan(16)
    g = torch.Generator().manual_seed(0)
    return replace(ph, db0=(torch.rand(len(ph), generator=g, dtype=f64) - 0.5) * 60)


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


@pytest.fixture(params=list(CASES), scope="module")
def written(request, tmp_path_factory):
    seq = CASES[request.param]()
    path = tmp_path_factory.mktemp("seq") / "out.seq"
    write_pulseq(seq, path)
    return seq, path


def test_round_trip_simulates_identically(written):
    seq, path = written
    back = read_pulseq(path)
    ph = phantom()
    coils = bp.birdcage_sensitivities(ph.pos, 2)
    a, b = bp.simulate(ph, seq, coils), bp.simulate(ph, back, coils)
    assert back.n_adc == seq.n_adc and abs(float(back.duration - seq.duration)) < 1e-12
    assert rel(b.signal, a.signal) < 1e-9 and rel(b.m, a.m) < 1e-9


def test_pypulseq_timing_check_passes(written):
    _, path = written
    s = pp.Sequence()
    s.read(str(path), remove_duplicates=False)
    ok, report = s.check_timing()
    assert ok, report[:5]


def test_kspace_equals_pypulseq(written):
    seq, path = written
    s = pp.Sequence()
    s.read(str(path), remove_duplicates=False)
    k_pp = np.asarray(s.calculate_kspace()[0]).T  # (n_adc, 3) [1/m]
    ours = kspace_trajectory(seq).numpy()
    scale = np.abs(ours).max()
    assert np.abs(k_pp - ours).max() < 1e-9 * scale  # measured 4e-13 to 5e-11


def test_provenance_follows_subdivision_and_covers_the_segments():
    seq = CASES["gre: ramps, slice selection, RF spoiling"]()
    for s in (seq, seq.subdivide_rf(3)):
        spans = [b["span"] for b in s.meta["blocks"]]
        assert spans[0][0] == 0 and spans[-1][1] == len(s)
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    # the subdivided blocks keep their durations
    sub = seq.subdivide_rf(3)
    for b0, b1 in zip(seq.meta["blocks"], sub.meta["blocks"]):
        assert abs(float(seq.dt[slice(*b0["span"])].sum() - sub.dt[slice(*b1["span"])].sum())) < 1e-15


def test_what_cannot_be_written_raises(tmp_path):
    from blochparty.girf import GIRF, apply_girf

    with pytest.raises(NotImplementedError, match="ideal segments"):  # instantaneous gradients, no dead time
        write_pulseq(bp.gre_2d(8, n_dummy=0), tmp_path / "a.seq", spoil="drop")
    tse = bp.tse_2d(16, etl=4, tr=0.4, slice_thickness=5e-3)  # ideal spoiling at the end of each TR
    with pytest.raises(ValueError, match="ideal spoilers"):
        write_pulseq(tse, tmp_path / "b.seq")
    write_pulseq(tse, tmp_path / "b.seq", spoil="drop")
    gre = CASES["gre: ramps, slice selection, RF spoiling"]()
    with pytest.raises(ValueError, match="changed after building"):  # the played sequence is not the written one
        write_pulseq(apply_girf(gre, GIRF.identity(delay=2e-6)), tmp_path / "c.seq")
    with pytest.raises(ValueError, match="not on the raster"):  # a 64 us readout flat top
        write_pulseq(bp.epi_2d(16, slice_thickness=5e-3), tmp_path / "d.seq")


def test_full_precision_survives_where_six_digits_would_not(tmp_path):
    # pypulseq's writer rounds amplitudes to six digits (~5e-7); ours keeps doubles
    seq = CASES["gre: ramps, slice selection, RF spoiling"]()
    write_pulseq(seq, tmp_path / "e.seq")
    back = read_pulseq(tmp_path / "e.seq")
    assert len(back) == len(seq)
    assert rel(back.grad, seq.grad) < 1e-14 and rel(back.adc_phase, seq.adc_phase) < 1e-14


def assert_same_segments(a, b, exact=True):
    assert len(a) == len(b) and a.is_rf == b.is_rf and a.adc == b.adc
    for k in ("dt", "rf", "grad", "adc_phase"):
        x, y = getattr(a, k), getattr(b, k)
        if exact:
            assert torch.equal(x, y), k
        else:  # last bits: segment edges are absolute times from rounded delays
            assert float((x - y).abs().max()) <= (1e-17 if k == "dt" else 1e-14 * float(y.abs().max())), k


def test_imported_file_is_written_back_losslessly(tmp_path):
    # a sequence made and written by pypulseq itself (its six-digit numbers), imported, re-exported
    # by write_pulseq and imported again: the same segments, bit for bit
    sysm = pp.Opts(max_grad=32, grad_unit="mT/m", max_slew=130, slew_unit="T/m/s", rf_ringdown_time=20e-6,
                   rf_dead_time=100e-6, adc_dead_time=10e-6)
    s = pp.Sequence(sysm)
    rf, gz, gzr = pp.make_sinc_pulse(flip_angle=np.pi / 6, duration=2e-3, slice_thickness=5e-3, apodization=0.5,
                                     time_bw_product=4, system=sysm, return_gz=True)
    gx = pp.make_trapezoid("x", flat_area=64 / 0.2, flat_time=3.2e-3, system=sysm)
    adc = pp.make_adc(64, duration=gx.flat_time, delay=gx.rise_time, system=sysm)
    gx_pre = pp.make_trapezoid("x", area=-gx.area / 2, duration=1e-3, system=sysm)
    for i in range(4):
        rf.phase_offset = adc.phase_offset = np.deg2rad(117) * i * (i + 1) / 2
        gy = pp.make_trapezoid("y", area=(i - 2) / 0.2, duration=1e-3, system=sysm)
        s.add_block(rf, gz)
        s.add_block(gx_pre, gy, gzr)
        s.add_block(gx, adc)
        s.add_block(pp.make_delay(5e-3))
    s.write(str(tmp_path / "pp.seq"))
    first = read_pulseq(tmp_path / "pp.seq")
    write_pulseq(first, tmp_path / "ours.seq")
    assert_same_segments(read_pulseq(tmp_path / "ours.seq"), first)
    with pytest.raises(ValueError, match="changed after read_pulseq"):
        write_pulseq(first.subdivide_rf(2), tmp_path / "x.seq")


@pytest.mark.slow
def test_scanner_file_is_written_back_losslessly(tmp_path):
    from pathlib import Path

    path = Path(__file__).parents[1] / "validation" / "real" / "data" / "iohexol" / "Seq_04_IR_LL_FOV_192_SOS3.seq"
    if not path.exists():
        pytest.skip("validation data not fetched (validation/real/fetch_iohexol.py)")
    first = read_pulseq(path)
    write_pulseq(first, tmp_path / "ours.seq")
    # measured: durations within 1.7e-18 s, RF and gradients within 1e-15 relative, receiver phases exact
    assert_same_segments(read_pulseq(tmp_path / "ours.seq"), first, exact=False)


def test_flip_angle_about_the_pulse_frequency():
    rf, _, _ = bp.sequence.sinc_pulse(180.0, 2e-3, slice_thickness=5e-3, slice_offset=0.02)
    sig = (rf.signal[:, 0] + 1j * rf.signal[:, 1]).numpy()
    assert abs(abs(sig.sum()) * rf.raster * 360 - 180) > 170  # the phase ramp cancels the plain sum
    assert abs(flip_angle(sig, rf.raster) - 180) < 1e-9
    rf0, _, _ = bp.sequence.sinc_pulse(90.0, 2e-3, slice_thickness=5e-3)
    assert abs(flip_angle((rf0.signal[:, 0] + 1j * rf0.signal[:, 1]).numpy(), rf0.raster) - 90) < 1e-9
