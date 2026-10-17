"""Concomitant (Maxwell) fields and the uniform per-segment offset (``fields.py``).

Checked against: the explicit vector field of a cylindrical coil set (Maxwell's
equations, and |B| in 50-digit arithmetic), Bernstein's analytic concomitant phase of a
bipolar gradient pair, a Maxwell-compensated spin echo, the closed-form free runs against
segment-wise stepping, and finite differences for the field strength (Metal agrees with the core
in ``test_engines.py`` and ``test_gpu_grad.py``).
"""

import math
from dataclasses import replace

import mpmath
import pytest
import torch

from blochparty import GAMMA_BAR, Phantom, SequenceBuilder, simulate, trapezoid
from blochparty.fields import Scanner, concomitant_coefficients, exact_excess, transverse_coefficients

from cases import busy_sequence, random_phantom
from helpers import rel

SIMS = {"core": simulate}
G_MAX = 40e-3 * GAMMA_BAR  # 40 mT/m [Hz/m]


def vector_field(r, g, f0, a=0.5):
    """Explicit (Bx, By, Bz) [Hz] of linear gradients g on a cylindrical coil set."""
    x, y, z = r.unbind(-1)
    gx, gy, gz = g
    return torch.stack([gx * z - a * gz * x, gy * z - (1 - a) * gz * y, f0 + gx * x + gy * y + gz * z], dim=-1)


def bernstein(r, g, f0):
    """Bernstein et al. (MRM 1998) leading-order concomitant field [Hz], written out."""
    x, y, z = r.unbind(-1)
    gx, gy, gz = g
    return ((gx**2 + gy**2) * z**2 + gz**2 * (x**2 + y**2) / 4 - gx * gz * x * z - gy * gz * y * z) / (2 * f0)


def points(n=40, seed=0, extent=0.25):
    gen = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 3, dtype=torch.float64, generator=gen) - 0.5) * 2 * extent


def test_vector_field_obeys_maxwell():
    # div B = 0 and curl B = 0: the Jacobian is traceless and symmetric, for any asymmetry
    g = torch.tensor([0.7, -0.4, 1.0], dtype=torch.float64) * G_MAX
    for a in (0.5, 0.3):
        for r in points(5):
            jac = torch.autograd.functional.jacobian(lambda p: vector_field(p, g, 1e8, a), r)
            assert abs(float(jac.trace())) < 1e-6 * G_MAX
            assert torch.allclose(jac, jac.T, atol=1e-6 * G_MAX)


@pytest.mark.parametrize("a", [0.5, 0.3])
def test_leading_order_is_the_expansion_of_the_magnitude(a):
    r = points()
    g = torch.tensor([0.7, -0.4, 1.0], dtype=torch.float64) * G_MAX
    f0 = GAMMA_BAR * 0.55
    lin = r @ g
    lead = concomitant_coefficients(r, f0, a) @ torch.stack([g[0]**2, g[1]**2, g[2]**2, g[0] * g[1], g[0] * g[2],
                                                              g[1] * g[2]])
    if a == 0.5:
        assert rel(lead, bernstein(r, g, f0)) < 1e-14
    # transverse coefficients reproduce the vector field's (Bx, By)
    t = transverse_coefficients(r, a)
    assert torch.allclose(torch.stack([t[:, :3] @ g, t[:, 3:] @ g], -1), vector_field(r, g, f0, a)[:, :2])
    # |B| - f0 - lin in 50 digits: the leading term is its 1/f0 part, the rest falls as 1/f0^2
    mpmath.mp.dps = 50
    diffs = []
    for scale in (1.0, 2.0, 4.0):
        b = vector_field(r, g, 0.0, a)
        f0s = f0 * scale
        excess = torch.tensor([float(mpmath.sqrt(mpmath.mpf(float(bx))**2 + mpmath.mpf(float(by))**2
                                                 + (mpmath.mpf(f0s) + mpmath.mpf(float(bz)))**2) - f0s - mpmath.mpf(float(bz)))
                               for bx, by, bz in b.tolist()], dtype=torch.float64)
        t2 = b[:, 0] ** 2 + b[:, 1] ** 2
        assert rel(exact_excess(lin, b[:, 0], b[:, 1], f0s), excess) < 1e-13  # the stable form is exact
        assert torch.allclose(lead / scale, t2 / (2 * f0s), rtol=1e-13)
        diffs.append(excess - lead / scale)
        # next order: -t^2 lin / (2 f0^2)
        assert rel(diffs[-1], -t2 * lin / (2 * f0s**2)) < 0.05 / scale
    for d1, d2 in zip(diffs, diffs[1:]):
        ratio = d1 / d2
        assert torch.allclose(ratio, torch.full_like(ratio, 4.0), rtol=0.06)


def spins(r):
    """Spins at r with no relaxation or off-resonance; with ``eye`` coils each coil sees one."""
    n = r.shape[0]
    z = torch.zeros(n, dtype=torch.float64)
    return Phantom(r, z + 1, z + 1e9, z + 1e9, z), torch.eye(n, dtype=torch.complex128)


def phases(sim, r, seq, scanner):
    """Phase accumulated after excitation (Mxy = i), per spin, at the last ADC sample."""
    ph, coils = spins(r)
    return SIMS[sim](ph, seq, coils, scanner=scanner).signal[:, -1] / 1j


def bipolar(g, delta, ramps):
    b = SequenceBuilder()
    b.rf(1e-6, flip_deg=90.0)
    if ramps:
        lobe = lambda s: trapezoid(amplitude=s * g, flat_time=delta, dtype=torch.float64)
        b.block(lobe(1.0)).block(lobe(-1.0))
    else:
        b.delay(delta, grad=g).delay(delta, grad=-g)
    return b.readout(1, 1e-6).build()


@pytest.mark.parametrize("sim", SIMS)
@pytest.mark.parametrize("axis", [0, 2])
def test_bernstein_bipolar_phase(sim, axis):
    # a phase-contrast bipolar pair cancels G.r but not the concomitant phase,
    # -2 pi int f_c dt = -2 pi f_c(G) 2 delta, Bernstein's analytic phase
    f0 = GAMMA_BAR * 0.55
    g = torch.zeros(3, dtype=torch.float64)
    g[axis] = 0.8 * G_MAX
    r, delta = points(30, seed=1, extent=0.15), 2e-3
    scanner = Scanner(field_strength=0.55)
    got = phases(sim, r, bipolar(g, delta, False), scanner)
    expect = torch.exp(-2j * math.pi * bernstein(r, g, f0) * 2 * delta)
    assert (got - expect).abs().max() < 1e-9
    assert (phases(sim, r, bipolar(g, delta, False), None) - 1).abs().max() < 1e-9  # ideal scanner: none
    # trapezoids: int g^2 dt of the true waveform is G^2 (flat + 2 rise / 3) per lobe; the
    # raster staircase the simulator sees has G^2 (flat + 2 rise / 3 - rise / (6 n^2)) for
    # ramps of n raster steps
    seq = bipolar(g, delta, True)
    lobe = trapezoid(amplitude=g, flat_time=delta, dtype=torch.float64)
    n = round(lobe.rise / 10e-6)
    stair = 2 * (lobe.flat + 2 * lobe.rise / 3 - lobe.rise / (6 * n * n))
    got = phases(sim, r, seq, scanner)
    assert (got - torch.exp(-2j * math.pi * bernstein(r, g, f0) * stair)).abs().max() < 1e-9
    true = 2 * (lobe.flat + 2 * lobe.rise / 3)
    assert (got - torch.exp(-2j * math.pi * bernstein(r, g, f0) * true)).abs().max() < 2e-3


def spin_echo(lobes_before, lobes_after, axis=2):
    """90x, lobes, 180y, lobes, sample: G.r refocuses when the areas match, the concomitant
    phase when int G^2 dt does too (a Maxwell-compensated pair)."""
    b = SequenceBuilder()
    b.rf(1e-6, flip_deg=90.0)
    unit = torch.zeros(3, dtype=torch.float64)
    unit[axis] = 1.0
    for amp, dur in lobes_before:
        b.delay(dur, grad=amp * unit)
    b.rf(1e-6, flip_deg=180.0, phase=math.pi / 2)
    for amp, dur in lobes_after:
        b.delay(dur, grad=amp * unit)
    return b.readout(1, 1e-6).build()


@pytest.mark.parametrize("sim", SIMS)
def test_maxwell_compensated_echo(sim):
    f0, G, delta = GAMMA_BAR * 0.55, 0.8 * G_MAX, 3e-3
    r = points(30, seed=2, extent=0.15)
    scanner = Scanner(field_strength=0.55)
    # after the 180: two lobes of duration tau with total area G delta and int g^2 = G^2 delta
    tau = 0.8 * delta
    s, p = G * delta / tau, G * G * delta / tau
    root = math.sqrt(2 * p - s * s)
    after = [((s + root) / 2, tau), ((s - root) / 2, tau)]
    assert math.isclose(sum(a * d for a, d in after), G * delta) and math.isclose(sum(a * a * d for a, d in after),
                                                                                    G * G * delta)
    compensated = phases(sim, r, spin_echo([(G, delta)], after), scanner)
    assert (compensated - 1).abs().max() < 1e-9
    # uncompensated (half the amplitude for twice as long): the 180 negates the first half's phase
    plain = phases(sim, r, spin_echo([(G, delta)], [(G / 2, 2 * delta)]), scanner)
    fc = lambda amp: bernstein(r, torch.tensor([0.0, 0.0, amp], dtype=torch.float64), f0)
    expect = torch.exp(-2j * math.pi * (fc(G / 2) * 2 * delta - fc(G) * delta))
    assert (plain - expect).abs().max() < 1e-9
    assert (plain - 1).abs().max() > 0.1


def _moving_spin_in_a_compensated_echo(sim, engine):
    # Maxwell compensation cancels int f_c dt for a static spin only: a spin moving at v sees
    # f_c(r0 + v t), quadratic in t, and keeps -2 pi (int_after - int_before) f_c dt, integrated here
    # exactly per gradient segment (3-point Gauss-Legendre) along the path; the ratio to the ideal
    # scanner removes the G.r (velocity-encoding) phase, the same in both. The simulators take each
    # segment's position at its mid-time, exact for terms linear in t: the v^2 t^2 term leaves
    # C dt^3 / 12 per segment, so the lobes are cut into k pieces and the error falls as k^-2.
    from blochparty import Motion

    f0, G, delta = GAMMA_BAR * 0.55, 0.8 * G_MAX, 5e-3
    r0 = points(20, seed=3, extent=0.15)
    v = (torch.rand(20, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(4)) - 0.5) * 4.0  # <= 2 m/s
    tau = 0.8 * delta
    s_, p_ = G * delta / tau, G * G * delta / tau
    root = math.sqrt(2 * p_ - s_ * s_)
    ph, coils = spins(r0)
    moving = replace(ph, motion=Motion(velocity=v))
    scanner = Scanner(field_strength=0.55)
    nodes = torch.tensor([-math.sqrt(0.6), 0.0, math.sqrt(0.6)], dtype=torch.float64)
    weights = torch.tensor([5 / 9, 8 / 9, 5 / 9], dtype=torch.float64)

    def errors(k):
        before = [(G, delta / k)] * k
        after = [((s_ + root) / 2, tau / k)] * k + [((s_ - root) / 2, tau / k)] * k
        seq = spin_echo(before, after)
        ratio = sim(moving, seq, coils, scanner=scanner).signal[:, -1] / sim(moving, seq, coils).signal[:, -1]
        t_end = torch.cumsum(seq.dt, 0)
        total = torch.zeros(20, dtype=torch.float64)
        for j in range(len(seq)):
            if not bool(seq.grad[j].any()):
                continue
            a, b = float(t_end[j] - seq.dt[j]), float(t_end[j])
            fc = torch.stack([bernstein(r0 + v * t, seq.grad[j], f0) for t in 0.5 * (a + b) + 0.5 * (b - a) * nodes], -1)
            integral = 0.5 * (b - a) * (fc * weights).sum(-1)
            total += -integral if j <= k else integral  # segment 0 the 90, 1..k the first lobe, k + 1 the 180
        expect = torch.exp(-2j * math.pi * total)
        assert (expect - 1).abs().max() > 0.2  # the velocity term is there
        static = sim(ph, seq, coils, scanner=scanner).signal[:, -1] / sim(ph, seq, coils).signal[:, -1]
        return float((ratio.to(expect.dtype) - expect).abs().max()), float((static.to(expect.dtype) - 1).abs().max())

    (coarse, still), (fine, _) = errors(25), errors(50)
    if engine == "core":
        assert still < 1e-9  # a static spin stays compensated
        assert fine < 1e-6
        assert 3.8 < coarse / fine < 4.2  # second order in the segment length
    else:  # float32: the lobes are ~6400 rad of G.r phase at 15 cm, resolved to ~4e-4 rad (2^-24 of it)
        assert still < 2e-3 and fine < 2e-3


def test_moving_spin_in_a_compensated_echo():
    _moving_spin_in_a_compensated_echo(simulate, "core")


def test_moving_spin_in_a_compensated_echo_gpu(gpu):
    _moving_spin_in_a_compensated_echo(gpu.simulate, "gpu")


@pytest.mark.parametrize("sim", SIMS)
def test_exact_mode(sim):
    # one constant lobe: phase -2 pi delta (|B| - f0), |B| in 50 digits; exact minus leading
    # falls as 1/B0^2 (very low field here, so that it is large)
    r, delta = points(20, seed=3, extent=0.15), 0.1e-3
    g = torch.tensor([0.5, -0.3, 0.9], dtype=torch.float64) * G_MAX
    seq = SequenceBuilder().rf(1e-6, flip_deg=90.0).delay(delta, grad=g).readout(1, 1e-6).build()
    mpmath.mp.dps = 50
    gaps = []
    for b0 in (0.05, 0.1):
        f0 = GAMMA_BAR * b0
        exact = phases(sim, r, seq, Scanner(field_strength=b0, concomitant="exact"))
        b = vector_field(r, g, 0.0)
        mod = torch.tensor([float(mpmath.sqrt(mpmath.mpf(bx)**2 + mpmath.mpf(by)**2 + (mpmath.mpf(f0) + bz)**2) - f0)
                            for bx, by, bz in b.tolist()], dtype=torch.float64)
        assert (exact - torch.exp(-2j * math.pi * delta * mod)).abs().max() < 1e-9
        lead = phases(sim, r, seq, Scanner(field_strength=b0))
        gaps.append(torch.angle(exact / lead))
    ratio = gaps[0] / gaps[1]
    assert torch.allclose(ratio, torch.full_like(ratio, 4.0), rtol=0.1)


@pytest.mark.parametrize("sim", SIMS)
def test_uniform_offset_phase(sim):
    # b0_offset f over a free run of length T adds -2 pi f T to every spin, also during RF
    r = points(10, seed=4)
    seq = SequenceBuilder().rf(1e-6, flip_deg=90.0).delay(1e-3, grad=(500.0, 0.0, 0.0)).readout(4, 1e-5).build()
    off = torch.zeros(len(seq), dtype=torch.float64)
    off[1:] = 37.0
    base = phases(sim, r, seq, None)
    got = phases(sim, r, replace(seq, b0_offset=off), None)
    last = max(i for i, a in enumerate(seq.adc) if a)
    t = float(seq.dt[1 : last + 1].sum())
    assert (got - base * complex(math.cos(2 * math.pi * 37 * t), -math.sin(2 * math.pi * 37 * t))).abs().max() < 1e-12
    # during a slow RF pulse an offset acts like off-resonance
    ph, _ = spins(r)
    seq = SequenceBuilder().rf(2e-3, flip_deg=90.0, n_steps=1).build()
    with_off = SIMS[sim](ph, replace(seq, b0_offset=torch.full((1,), 120.0, dtype=torch.float64))).m
    as_db0 = SIMS[sim](replace(ph, db0=ph.db0 + 120.0), seq).m
    assert torch.allclose(with_off, as_db0, atol=1e-14)


def field_case(mode, seed=0):
    ph = random_phantom(40, seed)
    seq = busy_sequence(torch.tensor(800.0, dtype=torch.float64), torch.tensor(1500.0, dtype=torch.float64), seed)
    seq = replace(seq, grad=seq.grad * 400.0)  # strong gradients, so the concomitant field matters
    g = torch.Generator().manual_seed(seed + 7)
    off = 30.0 * torch.randn(len(seq), dtype=torch.float64, generator=g)
    coils = torch.randn(len(ph), 3, dtype=torch.complex128, generator=g)
    return ph, replace(seq, b0_offset=off), coils, Scanner(field_strength=torch.tensor(0.3, dtype=torch.float64),
                                                           concomitant=mode)


@pytest.mark.parametrize("mode", ["leading", "exact"])
def test_field_model_matters_and_stepping_agrees(mode):
    ph, seq, coils, scanner = field_case(mode)
    a = simulate(ph, seq, coils, scanner=scanner)
    # the field model matters here, and segment-wise stepping agrees with the closed form
    assert rel(simulate(ph, seq, coils).signal, a.signal) > 0.1
    assert rel(simulate(ph, seq, coils, scanner=scanner, merge_free_runs=False).signal, a.signal) < 1e-11


@pytest.mark.parametrize("mode", ["leading", "exact"])
def test_field_strength_gradient_vs_finite_differences(mode):
    ph, seq, coils, scanner = field_case(mode)

    def loss(b0):
        r = simulate(ph, seq, coils, scanner=replace(scanner, field_strength=b0))
        w = torch.randn(r.signal.shape, dtype=torch.complex128, generator=torch.Generator().manual_seed(9))
        return (r.signal * w).real.sum() + (r.m**2).sum()

    b0 = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    (g,) = torch.autograd.grad(loss(b0), [b0])
    h = 1e-5
    with torch.no_grad():
        fd = (loss(torch.tensor(0.3 + h, dtype=torch.float64)) - loss(torch.tensor(0.3 - h, dtype=torch.float64))) / (2 * h)
    assert abs(float(g) - float(fd)) < 1e-6 * abs(float(fd))
