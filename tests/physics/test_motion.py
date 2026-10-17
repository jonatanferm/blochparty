"""Moving spins (``motion.py``): flow, periodic deformation and rigid motion, in the core and
on the GPU. References: the phase-contrast phase 2 pi G v tau^2 of a bipolar gradient (exact),
a constant rigid transform equals the static phantom moved, a translation between two halves
of k-space equals the two static k-spaces stitched with the Fourier shift theorem's phase,
inflow enhancement in a spoiled GRE against the time-of-flight series, and the GPU against the
core spin by spin with every kind of motion and diffusion at once."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

import blochparty as bp
from blochparty.motion import rotation_z

f64 = torch.float64


def spins(pos, **kw):
    n = pos.shape[0]
    return bp.Phantom(pos=pos, pd=torch.ones(n, dtype=f64), t1=torch.full((n,), 1e6, dtype=f64),
                      t2=torch.full((n,), 1e6, dtype=f64), db0=torch.zeros(n, dtype=f64), **kw)


def test_phase_contrast_exact():
    v = torch.tensor([[0.3, 0.0, 0.0], [-0.5, 0.1, 0.0], [0.0, 0.0, 0.0]], dtype=f64)
    ph = spins(torch.tensor([[0.01, 0, 0], [-0.02, 0.01, 0], [0.0, 0, 0]], dtype=f64), motion=bp.Motion(velocity=v))
    g, tau = 2e5, 2e-3
    seq = (bp.SequenceBuilder().rf(1e-6, flip_deg=90.0).delay(tau, grad=(g, 0, 0)).delay(tau, grad=(-g, 0, 0))
           .readout(1, 2e-6).build())
    m = bp.simulate(ph, seq).m
    phase = torch.atan2(m[:, 1], m[:, 0]) - math.pi / 2
    assert torch.allclose(phase, 2 * math.pi * g * v[:, 0] * tau**2, atol=1e-12)


def test_constant_rigid_transform_is_the_moved_phantom():
    ph = bp.shepp_logan(24)
    seq = bp.gre_2d(24, n_dummy=3)
    A, b = rotation_z(0.3, centre=(0.01, -0.02, 0.0))
    moved = bp.simulate(replace(ph, pos=ph.pos @ torch.as_tensor(A).T + torch.as_tensor(b)), seq)
    rig = bp.Motion(rigid=lambda t: (np.broadcast_to(A, (len(t), 3, 3)), np.broadcast_to(b, (len(t), 3))))
    moving = bp.simulate(replace(ph, motion=rig), seq)
    assert torch.allclose(moving.signal, moved.signal, atol=1e-10 * float(moved.signal.abs().max()))


def test_translation_between_k_space_halves():
    # ideal spoiling, no inhomogeneity: each line depends only on the current positions, so a jump
    # by b after half the lines multiplies those lines by exp(-i 2 pi k . b) (Fourier shift)
    n = 32
    ph = bp.shepp_logan(n)
    seq = bp.gre_2d(n, n_dummy=0)
    t_line = float(seq.duration) / n
    shift = np.array([0.02, -0.01, 0.0])
    rig = bp.Motion(rigid=lambda t: (np.broadcast_to(np.eye(3), (len(t), 3, 3)),
                                     np.where((t >= n // 2 * t_line)[:, None], shift[None], 0.0)))
    moving = bp.simulate(replace(ph, motion=rig), seq).signal[0].reshape(n, n)
    static = bp.simulate(ph, seq).signal[0].reshape(n, n)
    shifted = bp.simulate(replace(ph, pos=ph.pos + torch.as_tensor(shift)), seq).signal[0].reshape(n, n)
    ref = torch.cat([static[: n // 2], shifted[n // 2:]])
    assert torch.allclose(moving, ref, atol=1e-9 * float(ref.abs().max()))
    assert (shifted - static).abs().max() > 0.1 * static.abs().max()  # the shift matters


def _tof_sequence(n_tr, tr, alpha, slab):
    b = bp.SequenceBuilder()
    for k in range(n_tr):
        b.sinc_pulse(math.degrees(alpha), 2e-3, slice_thickness=slab, time_bw_product=8)
        b.readout(1, 10e-6)
        b.delay(tr * (k + 1) - float(torch.cat(b._dt).sum())).spoil()
    return b.build()


@pytest.mark.parametrize("velocity", [0.0, 0.05, 0.15])
def test_inflow_time_of_flight(gpu, velocity):
    # spins stream through a 5 mm slice (TBW 8 sinc) under ideal spoiling. Reference: the
    # time-of-flight recursion with the pulse's own complex slice profile (from static spins,
    # no relaxation): before pulse k a spin at z_k = z0 + v t_k has Mz_k; it gives Mz_k T(z_k) to
    # the signal and keeps Mz_k L(z_k), recovering by T1 until the next pulse.
    slab, tr, t1, alpha, n_tr = 5e-3, 20e-3, 1.0, math.radians(30), 120
    seq = _tof_sequence(n_tr, tr, alpha, slab)
    # the slice profile
    zp = torch.linspace(-8e-3, 8e-3, 3201, dtype=f64)
    one = bp.SequenceBuilder().sinc_pulse(math.degrees(alpha), 2e-3, slice_thickness=slab, time_bw_product=8)
    prof = gpu.simulate(spins(torch.stack([torch.zeros_like(zp), torch.zeros_like(zp), zp], -1)),
                                one.readout(1, 10e-6).build()).m.double()  # before the spoiler
    T, L = torch.complex(prof[:, 0], prof[:, 1]).numpy(), prof[:, 2].numpy()
    # an upstream column long enough to feed the slice for the whole run
    dz = 50e-6
    z0 = torch.arange(-(velocity * n_tr * tr + 0.02), 0.01, dz, dtype=f64)
    n = len(z0)
    ph = spins(torch.stack([torch.zeros(n, dtype=f64), torch.zeros(n, dtype=f64), z0], -1), motion=bp.Motion(
        velocity=torch.tensor([0.0, 0.0, velocity], dtype=f64).expand(n, 3).contiguous()))
    ph = replace(ph, t1=torch.full((n,), t1, dtype=f64))
    sim = gpu.simulate(ph, seq).signal[0].abs().double().numpy()
    t_pulse = np.arange(n_tr) * tr + 1e-3 + float(seq.dt[0])  # pulse centres
    mz = np.ones(n)
    ref = []
    e1 = math.exp(-tr / t1)
    for k in range(n_tr):
        zk = z0.numpy() + velocity * t_pulse[k]
        tk = np.interp(zk, zp.numpy(), T.real, left=0, right=0) + 1j * np.interp(zk, zp.numpy(), T.imag, left=0, right=0)
        lk = np.interp(zk, zp.numpy(), L, left=1, right=1)
        ref.append(abs((mz * tk).sum()))
        mz = 1 + (mz * lk - 1) * e1
    ref = np.array(ref)
    assert np.abs(sim[-20:] - ref[-20:]).max() < 0.01 * ref[-20:].mean(), (sim[-3:], ref[-3:])
    if velocity:  # flow raises the signal well above the static steady state
        assert ref[-1] > 2 * ref[0] * (1 - math.cos(alpha))


def test_gpu_matches_core_with_all_motion(gpu):
    g = torch.Generator().manual_seed(4)
    n = 1500
    pos = (torch.rand(n, 3, generator=g, dtype=f64) - 0.5) * 0.04
    mot = bp.Motion(velocity=(torch.rand(n, 3, generator=g, dtype=f64) - 0.5) * 0.2,
                    displacement=(torch.rand(n, 3, generator=g, dtype=f64) - 0.5) * 0.004,
                    waveform=lambda t: np.sin(2 * np.pi * t / 0.05),
                    rigid=lambda t: tuple(map(np.stack, zip(*[rotation_z(2.0 * x, (0.002, 0.0, 0.0)) for x in t]))))
    ph = replace(spins(pos, motion=mot), t2=torch.full((n,), 0.08, dtype=f64),
                 diffusion=torch.full((n,), 2e-9, dtype=f64))
    seq = bp.gre_2d(16, n_dummy=2)
    coils = bp.birdcage_sensitivities(pos, 4)
    for p in (ph, replace(ph, diffusion=None)):
        ref = bp.simulate(p, seq, coils, seed=3)
        for mma in (False, True):
            out = gpu.simulate(p, seq, coils, seed=3, mma=mma)
            assert float((out.signal.to(ref.signal.dtype) - ref.signal).abs().max() / ref.signal.abs().max()) < 1e-4


def _field_chain_case():
    """Five spins at constant velocity, a random RF / gradient / off-resonance sequence with
    gradients 20x stronger (concomitant fields of tens of Hz at 0.55 T), four coils."""
    from cases import random_case
    from helpers import with_readout

    g = torch.Generator().manual_seed(0)
    n = 5
    pos = (torch.rand(n, 3, generator=g, dtype=f64) - 0.5) * 0.2
    vel = (torch.rand(n, 3, generator=g, dtype=f64) - 0.5) * 0.5
    one = torch.ones(n, dtype=f64)
    ph = bp.Phantom(pos, one, 0.8 * one, 0.1 * one, torch.linspace(-30, 30, n, dtype=f64), motion=bp.Motion(velocity=vel))
    _, seq = random_case(0)
    seq = with_readout(seq)
    return ph, replace(seq, grad=seq.grad * 20), bp.birdcage_sensitivities(pos, 4)


def _scanners():
    from cases import distortion

    from blochparty.fields import Scanner

    return {"nonlinearity": Scanner(nonlinearity=distortion()), "leading": Scanner(field_strength=0.55),
            "exact + nonlinearity": Scanner(field_strength=0.55, concomitant="exact", nonlinearity=distortion())}


def _static_chain(ph, seq, coils, scanner):
    """Segment by segment, a static spin at the moving spin's mid-segment position (m_init chained)."""
    t = torch.cumsum(seq.dt, 0) - 0.5 * seq.dt
    m, sig = None, []
    for s in range(len(seq)):
        one = slice(s, s + 1)
        part = replace(seq, dt=seq.dt[one], rf=seq.rf[one], grad=seq.grad[one], adc_phase=seq.adc_phase[one],
                       is_rf=seq.is_rf[one], adc=seq.adc[one], spoil=seq.spoil[one], meta={})
        static = replace(ph, pos=ph.pos + ph.motion.velocity * t[s], motion=None)
        r = bp.simulate(static, part, coils, m_init=m, scanner=scanner)
        m = r.m
        sig.append(r.signal)
    return torch.cat(sig, 1), m


@pytest.mark.parametrize("which", ["nonlinearity", "leading", "exact + nonlinearity"])
def test_moving_under_field_model_equals_static_chain(which):
    # moving spins live in true coordinates; every segment encodes them through the nonlinearity and
    # re-evaluates their concomitant coefficients where they are (ROADMAP M1b)
    ph, seq, coils = _field_chain_case()
    scanner = _scanners()[which]
    ref_sig, ref_m = _static_chain(ph, seq, coils, scanner)
    out = bp.simulate(ph, seq, coils, scanner=scanner)
    assert float((out.m - ref_m).abs().max()) < 1e-10
    assert float((out.signal - ref_sig).abs().max() / ref_sig.abs().max()) < 1e-10
    moved = bp.simulate(ph, seq, coils, scanner=None)
    assert float((moved.signal - ref_sig).abs().max() / ref_sig.abs().max()) > 1e-3  # the field model matters


@pytest.mark.parametrize("which", ["nonlinearity", "leading", "exact + nonlinearity"])
def test_gpu_moving_under_field_model_equals_static_chain(gpu, which):
    ph, seq, coils = _field_chain_case()
    scanner = _scanners()[which]
    ref_sig, ref_m = _static_chain(ph, seq, coils, scanner)
    for mma in (False, True):  # per spin (the five dephase strongly: |signal| ~ 0.3 of their sum)
        out = gpu.simulate(ph, seq, coils, scanner=scanner, mma=mma)
        assert float((out.signal.to(ref_sig.dtype) - ref_sig).abs().max()) < 1e-5 * len(ph)
        assert float((out.m.double() - ref_m).abs().max()) < 1e-5


def test_gpu_diffusion_under_field_model(gpu):
    # walkers in true coordinates, encoded per segment; the same Philox walk in both engines
    ph, seq, coils = _field_chain_case()
    g = torch.Generator().manual_seed(5)
    n = 300
    pos = (torch.rand(n, 3, generator=g, dtype=f64) - 0.5) * 0.2
    one = torch.ones(n, dtype=f64)
    ph = bp.Phantom(pos, one, 0.8 * one, 0.1 * one, torch.linspace(-30, 30, n, dtype=f64), diffusion=2e-9 * one)
    coils = bp.birdcage_sensitivities(pos, 4)
    for scanner in _scanners().values():
        ref = bp.simulate(ph, seq, coils, scanner=scanner, seed=3)
        out = gpu.simulate(ph, seq, coils, scanner=scanner, seed=3)
        assert float((out.signal.to(ref.signal.dtype) - ref.signal).abs().max() / ref.signal.abs().max()) < 2e-4
        assert float((out.m.double() - ref.m).abs().max() / ref.m.abs().max()) < 3e-3


def test_tse_flow_void(gpu):
    """Blood streaming through a 5 mm slice at 10 cm/s keeps < 5 % of static blood's signal at a TSE
    echo near 113 ms (echo 11 at 10.26 ms spacing; static ~2/3 of echo 1): the premise of treating
    BrainWeb's vessels as flow voids (validation/realism/simulate_heads.py, flow_void)."""
    seq = bp.tse_2d(16, 0.22, etl=16, esp=0.01026, tr=0.5, slice_thickness=5e-3, n_dummy=0, dtype=torch.float32)
    dur = float(seq.dt.sum())

    def echo11(v, dz=25e-6):
        z = torch.arange(-(v * dur + 4e-3), 4e-3, dz, dtype=f64)
        n = len(z)
        ph = spins(torch.stack([torch.zeros(n, dtype=f64), torch.zeros(n, dtype=f64), z], -1),
                   motion=bp.Motion(velocity=torch.tensor([0., 0., v], dtype=f64).expand(n, 3).contiguous()))
        ph = replace(ph, t1=torch.full((n,), 1.74, dtype=f64), t2=torch.full((n,), .275, dtype=f64))
        return float(gpu.simulate(ph, seq).signal[0].abs().double().reshape(16, -1).mean(1)[10])

    static = echo11(0.)
    assert echo11(0.10) < 0.05 * static and echo11(0.20) < 0.05 * static
    assert 0.5 * static < echo11(0.005) < static  # slow flow loses only part
