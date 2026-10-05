"""Fat: ``phantom_from_maps(..., fat_fraction=...)`` adds one spin per peak of the 6-peak fat
spectrum, shifted by ppm x gamma_bar x B0 (negative: fat resonates below water).
Checked against the chemical-shift displacement along the readout, the k-space modulation
of each peak, and the analytic multi-peak in- / opposed-phase signal."""

import math

import pytest
import torch

import blochparty as bp
from blochparty import SequenceBuilder, simulate
from blochparty.phantom import FAT_PEAKS, GAMMA_BAR, TISSUES

one = lambda v: torch.full((1, 1), v, dtype=torch.float64)
PPM = torch.tensor([p for p, _ in FAT_PEAKS], dtype=torch.float64)
AMP = torch.tensor([a for _, a in FAT_PEAKS], dtype=torch.float64)
AMP = AMP / AMP.sum()


def test_spectrum_and_bookkeeping():
    n = 6
    m = bp.shepp_logan_maps(n)
    mask = m["pd"] > 0
    ff = torch.rand(n, n, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    ff[0, 0] = ff[3, 3] = 0.0  # no fat spins for these
    db0 = torch.linspace(-20, 20, n * n, dtype=torch.float64).reshape(n, n)
    for b0 in (1.5, 3.0):
        ph = bp.phantom_from_maps(m["pd"], m["t1"], m["t2"], 0.2, db0, fat_fraction=ff, field_strength=b0)
        fat = (ff[mask] > 0)
        n_w, n_f = int(mask.sum()), int(fat.sum())
        assert len(ph) == n_w + 6 * n_f
        # water first, PD (1 - f); then one block per peak with PD f a_p, same positions, own T1 / T2
        assert torch.allclose(ph.pd[:n_w], m["pd"][mask] * (1 - ff[mask]), rtol=1e-15)
        fat_pd = ph.pd[n_w:].reshape(6, n_f)
        assert torch.allclose(fat_pd, AMP[:, None] * (m["pd"] * ff)[mask][fat], rtol=1e-14)
        assert torch.allclose(ph.pd.sum(), m["pd"][mask].sum(), rtol=1e-14)  # PD is only split
        assert torch.equal(ph.pos[n_w:].reshape(6, n_f, 3), ph.pos[:n_w][fat].expand(6, n_f, 3))
        assert torch.all(ph.t1[n_w:] == TISSUES["fat"][1]) and torch.all(ph.t2[n_w:] == TISSUES["fat"][2])
        shift = ph.db0[n_w:].reshape(6, n_f) - db0[mask][fat]
        expected = PPM * 1e-6 * 42.577478518e6 * b0  # e.g. -434.3 Hz for the main peak at 3 T
        assert torch.allclose(shift, expected[:, None].expand(6, n_f), atol=1e-10)
        assert abs(expected[1] - (-3.4 * 42.577478518 * b0)) < 1e-9 and expected[1] < 0
    assert GAMMA_BAR == 42.577478518e6


def test_with_subvoxel_spins_t2prime_b1_and_maps():
    # fat spins inherit their water spin's position, T2' offset and B1+; T1 / T2 maps work
    n = 4
    m = bp.shepp_logan_maps(n)
    mask = m["pd"] > 0
    ff = torch.full((n, n), 0.4, dtype=torch.float64)
    ft1, ft2 = torch.full((n, n), 0.3, dtype=torch.float64), torch.full((n, n), 0.08, dtype=torch.float64)
    b1 = torch.linspace(0.8, 1.2, n * n, dtype=torch.float64).reshape(n, n)
    ph = bp.phantom_from_maps(m["pd"], m["t1"], m["t2"], 0.2, fat_fraction=ff, spins_per_voxel=4,
                              jitter="stratified", t2prime=torch.full((n, n), 0.05, dtype=torch.float64), b1=b1,
                              fat_t1=ft1, fat_t2=ft2)
    n_w = 4 * int(mask.sum())
    water, fat = slice(0, n_w), slice(n_w, None)
    for f in ("pos", "b1"):
        assert torch.equal(getattr(ph, f)[fat].reshape(6, n_w, -1), getattr(ph, f)[water].reshape(1, n_w, -1)
                           .expand(6, n_w, -1)), f
    shift = (ph.db0[fat].reshape(6, n_w) - ph.db0[water]) - (PPM * 1e-6 * GAMMA_BAR * 3.0)[:, None]
    assert shift.abs().max() < 1e-9
    assert torch.all(ph.t1[fat] == 0.3) and torch.all(ph.t2[fat] == 0.08)


@pytest.mark.parametrize("sim", ["core", "fused"])
def test_chemical_shift_displacement(sim):
    # Fat-only phantom, main peak only, readout bandwidth = |shift| / 2 per pixel: the fat
    # image is the water image moved by exactly 2 pixels towards -x (negative frequency;
    # +2 bandwidth-pixels of off-resonance move it to +x, see tests/test_imaging.py).
    if sim == "fused":
        simulate_fused = pytest.importorskip("blochparty.fused").simulate_fused
    run = simulate if sim == "core" else simulate_fused
    n, fov, b0 = 32, 0.22, 3.0
    m = bp.shepp_logan_maps(n)
    df = -3.4e-6 * GAMMA_BAR * b0
    dwell = 2 / (abs(df) * n)  # 144 us
    seq = bp.gre_2d(n, fov, flip_deg=15.0, tr=10e-3, dwell=dwell, rf_duration=0.1e-6, n_dummy=100)
    bw = bp.readout_bandwidth_per_pixel(seq)
    assert abs(df / bw + 2) < 1e-12
    kw = dict(fat_t1=m["t1"], fat_t2=m["t2"], field_strength=b0)  # same relaxation as the water image
    water = bp.phantom_from_maps(m["pd"], m["t1"], m["t2"], fov)
    fat = bp.phantom_from_maps(m["pd"], m["t1"], m["t2"], fov, fat_fraction=torch.ones(n, n, dtype=torch.float64),
                               fat_peaks=((-3.4, 1.0),), **kw)
    recon = lambda p: bp.ifft2c(bp.kspace_grid(run(p, seq).signal, seq))[0].abs()
    img, ref = recon(fat), recon(water)
    tol = 1e-6 * ref.max()
    assert (img - ref.roll(-2, dims=-1)).abs().max() < tol
    for wrong in (2, -1, -3):  # wrong sign or amount fails
        assert (img - ref.roll(wrong, dims=-1)).abs().max() > 100 * tol
    # All six peaks: each modulates k-space by exp(-i 2 pi f_p t), t from the excitation centre
    fat6 = bp.phantom_from_maps(m["pd"], m["t1"], m["t2"], fov, fat_fraction=torch.ones(n, n, dtype=torch.float64),
                                **kw)
    k_w, k_f = run(water, seq).signal[0], run(fat6, seq).signal[0]
    end = seq.dt.cumsum(0)
    rf_centre = end[torch.tensor(seq.is_rf)] - seq.dt[0] / 2  # 100 dummies, then one pulse per line
    t = end[torch.tensor(seq.adc)] - rf_centre[100:].repeat_interleave(n)
    f = PPM * 1e-6 * GAMMA_BAR * b0
    expected = k_w * (AMP[:, None] * torch.exp(-2j * math.pi * f[:, None] * t)).sum(0)
    assert (k_f - expected).abs().max() < 1e-4 * k_w.abs().max()  # off-resonance during the 0.1 us pulses


@pytest.mark.parametrize("b0", [1.5, 3.0])
def test_in_and_opposed_phase(b0):
    # water + fat in one voxel after a 1 ns 90 deg pulse:
    #   s(t) = i [(1 - f) exp(-t/T2w) + f exp(-t/T2f) sum_p a_p exp(-i 2 pi f_p t)]
    ff, t2w, t2f = 0.35, 0.04, 0.025
    ph = bp.phantom_from_maps(one(1.0), one(1e9), one(t2w), 1e-3, fat_fraction=one(ff), field_strength=b0,
                              fat_t1=1e9, fat_t2=t2f)
    f_main = -3.4e-6 * GAMMA_BAR * b0
    te_op, te_ip = 1 / (2 * abs(f_main)), 1 / abs(f_main)  # 1.15 / 2.30 ms at 3 T for the main peak
    tau = 1e-9
    times = torch.tensor([te_op, te_ip, 3 * te_op, 2 * te_ip, 0.37e-3, 4.1e-3], dtype=torch.float64)
    b = SequenceBuilder().rf(tau, flip_deg=90.0)
    prev = tau / 2
    for te in times.tolist():  # one ADC sample (1 ns dwell) centred at te after the pulse centre
        b.delay(te - prev - 0.5e-9).readout(1, 1e-9)
        prev = te + 0.5e-9
    for sim in (simulate, pytest.importorskip("blochparty.fused").simulate_fused):
        s = sim(ph, b.build()).signal[0]
        f = PPM * 1e-6 * GAMMA_BAR * b0
        fat = (AMP[:, None] * torch.exp(-2j * math.pi * f[:, None] * times)).sum(0)
        expected = 1j * ((1 - ff) * torch.exp(-times / t2w) + ff * torch.exp(-times / t2f) * fat)
        assert (s - expected).abs().max() < 2e-7  # first order in (fat shift x pulse length)
        # the multi-peak spectrum is never fully in phase: |sum a_p e^{..}| < 1 at the "in-phase" TE
        assert abs(fat[1]) < 0.95 and abs(s[1]) > abs(s[0])
    # single peak: the textbook two-point Dixon values, and the sign: Mxy ~ exp(-i 2 pi f t) with
    # f < 0, so at t = 1/(4|f|) fat's phasor is +90 deg from water's (a positive shift gives -90)
    ph1 = bp.phantom_from_maps(one(1.0), one(1e9), one(1e9), 1e-3, fat_fraction=one(ff), field_strength=b0,
                               fat_t1=1e9, fat_t2=1e9, fat_peaks=((-3.4, 1.0),))
    t4 = 1 / (4 * abs(f_main))
    seq = (SequenceBuilder().rf(tau, flip_deg=90.0).delay(t4 - tau / 2 - 0.5e-9).readout(1, 1e-9)
           .delay(te_op - t4 - 1e-9).readout(1, 1e-9).delay(te_ip - te_op - 1e-9).readout(1, 1e-9).build())
    s = simulate(ph1, seq).signal[0] / 1j
    assert abs(s[0] - complex(1 - ff, ff)) < 1e-6  # fat at +i: lower frequency, phase advancing
    assert abs(s[1] - (1 - 2 * ff)) < 1e-6 and abs(s[2] - 1) < 1e-6


def test_shepp_logan_chemical_shift():
    plain = bp.shepp_logan(16)
    cs = bp.shepp_logan(16, chemical_shift=True)
    maps = bp.shepp_logan_maps(16)
    n_fat = int((maps["label"] == bp.phantom.LABELS.index("fat")).sum())
    assert len(cs) == len(plain) + 6 * n_fat
    assert torch.allclose(cs.pd.sum(), plain.pd.sum(), rtol=1e-14)
    assert torch.equal(bp.shepp_logan(16, spins_per_voxel=1).pos, plain.pos)
