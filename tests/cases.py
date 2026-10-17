"""Small cases shared by several test modules (until 2026-10-17 each lived in the module that used it first and
the others imported that module): random sequences and phantoms, the bound-pool and free-pool cases, a pulsed-gradient
spin echo, a gradient nonlinearity, a short-gradient-pulse sequence and a susceptibility sphere. Moved verbatim."""

import math
from dataclasses import replace

import torch

import blochparty as bp
from blochparty import ExchangePool, Phantom, SequenceBuilder
from blochparty.exchange import BoundPool
from blochparty.fields import GradientNonlinearity
from blochparty.phantom import b0_sphere

f64 = torch.float64

DELTA_SMALL, DELTA_BIG = 10e-3, 30e-3
R0 = 0.25
DCHI, FIELD = 9.4e-6, 3.0


def random_case(seed: int):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *shape: torch.rand(*shape, generator=g, dtype=torch.float64)
    n = 6
    phantom = Phantom(
        pos=(rnd(n, 3) - 0.5) * 0.2,
        pd=0.5 + rnd(n),
        t1=0.2 + rnd(n),
        t2=0.02 + 0.1 * rnd(n),
        db0=(rnd(n) - 0.5) * 200,
    )
    b = SequenceBuilder()
    for _ in range(4):
        # shaped RF on a 10 us raster with a slice-select-like gradient
        k = 40
        b.rf_waveform((rnd(k, 2) - 0.5) * 2000, 10e-6, grad=(rnd(3) - 0.5) * 2000)
        b.delay(rnd(()) * 3e-3, grad=(rnd(3) - 0.5) * 4000)
        b.rf(100e-6, flip_deg=180 * rnd(()), phase=2 * math.pi * rnd(()))
        b.delay(rnd(()) * 3e-3, grad=(rnd(3) - 0.5) * 4000)
    b.spoil()
    b.rf(200e-6, flip_deg=90.0, n_steps=10)
    b.delay(1e-3)
    return phantom, b.build()


def random_phantom(n=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64)
    return Phantom(pos=(rnd(n, 3) - 0.5) * 0.2, pd=0.5 + rnd(n), t1=0.2 + rnd(n),
                   t2=0.02 + 0.1 * rnd(n), db0=(rnd(n) - 0.5) * 200)


def busy_sequence(amp, gx, seed=0):
    """RF of varying phase, runs whose gradient changes from segment to segment,
    readouts split across runs, a spoiled run without ADC, and an ADC right after RF."""
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64)
    b = SequenceBuilder()
    for i in range(4):
        b.rf(100e-6, amplitude=amp, phase=1.3 * i, n_steps=1)
        b.delay(0.4e-3, grad=(gx, (rnd(()) - 0.5) * 3000, 0.0))
        b.readout(8, 20e-6, grad=(-gx, 0.0, 400.0), phase=0.2 * i)
        b.delay(0.2e-3, grad=(rnd(3) - 0.5) * 2000)
        b.readout(5, 30e-6, grad=(rnd(3) - 0.5) * 2000)
        b.delay(1e-3).spoil()
    b.rf(50e-6, amplitude=amp, n_steps=1).readout(3, 10e-6).delay(2e-3, grad=(1e3, 0.0, 0.0)).spoil()
    return b.build()


def bm_case(seed: int, shape: str):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.rand(*s, generator=g, dtype=f64)
    n = 5
    ph = Phantom(pos=(rnd(n, 3) - 0.5) * 0.2, pd=0.5 + rnd(n), t1=0.5 + rnd(n), t2=0.02 + 0.08 * rnd(n),
                 db0=(rnd(n) - 0.5) * 200,
                 bound=BoundPool(f=0.1 + 0.1 * rnd(n), k=20 + 40 * rnd(n), t1=torch.ones(n, dtype=f64),
                                 t2=(8 + 6 * rnd(n)) * 1e-6, dw=(rnd(n) - 0.5) * 600, lineshape=shape))
    b = SequenceBuilder()
    for _ in range(3):
        kk, dt = 40, 10e-6
        t = (torch.arange(kk, dtype=f64) + 0.5) * dt
        f_off = (rnd(()) - 0.5) * 8000  # off-resonant saturation-like pulse: a phase ramp
        amp = 300 + 1500 * rnd(kk)
        b.rf_waveform(torch.stack([amp * torch.cos(-2 * math.pi * f_off * t), amp * torch.sin(-2 * math.pi * f_off * t)], -1),
                      dt, grad=(rnd(3) - 0.5) * 2000)
        b.delay(rnd(()) * 3e-3, grad=(rnd(3) - 0.5) * 4000)
        b.rf(100e-6, flip_deg=180 * rnd(()), phase=2 * math.pi * rnd(()))
        b.delay(rnd(()) * 3e-3)
    b.spoil()
    b.rf(200e-6, flip_deg=90.0, n_steps=10)
    b.delay(1e-3)
    return ph, b.build()


def pool_case(seed, with_bound=True):
    ph, seq = bm_case(seed, "superlorentzian")
    g = torch.Generator().manual_seed(seed + 5)
    rnd = lambda: torch.rand(len(ph), generator=g, dtype=f64)
    amide = ExchangePool(f=0.01 + 0.02 * rnd(), k=50 + 200 * rnd(), t1=torch.ones(len(ph), dtype=f64),
                         t2=0.02 + 0.08 * rnd(), dw=3.5e-6 * 3 * bp.GAMMA_BAR + 100 * rnd())
    myelin = ExchangePool(f=0.15 + 0.05 * rnd(), k=5 + 10 * rnd(), t1=0.4 + 0.2 * rnd(), t2=0.008 + 0.01 * rnd(),
                          dw=10 * rnd())
    return replace(ph, pools=(amide, myelin), bound=ph.bound if with_bound else None), seq


def _maps_for(ph):
    zz, yy, xx = torch.meshgrid(torch.linspace(-0.2, 0.2, 5, dtype=f64), torch.linspace(-0.3, 0.3, 7, dtype=f64),
                                torch.linspace(-0.3, 0.3, 7, dtype=f64), indexing="ij")
    coils = torch.stack([torch.exp(-((xx - 0.1) ** 2 + yy**2) / 0.05) * torch.exp(1j * 2 * yy),
                         torch.exp(-((xx + 0.1) ** 2 + yy**2) / 0.05) * torch.exp(-1j * xx)])
    return bp.FieldMaps((-0.3, -0.3, -0.2), (0.1, 0.1, 0.1), b0=40 * xx**2 - 20 * yy + 10 * zz, b1=1 + 0.3 * xx,
                        coils=coils)


def pgse(g, readout=1):
    """90 - lobe (delta) - 180 - lobe, lobes DELTA_BIG apart; ADC at the echo."""
    b = bp.SequenceBuilder()
    b.rf(10e-6, flip_deg=90.0).delay(DELTA_SMALL, grad=g).delay(DELTA_BIG - DELTA_SMALL - 10e-6)
    b.rf(10e-6, flip_deg=180.0, phase=math.pi / 2).delay(DELTA_SMALL, grad=g).readout(readout, 10e-6)
    return b.build()


def distortion():
    """x, y coils with third- and fifth-order terms: up to ~3 pixels at the edge of the grid."""
    alpha, beta = torch.zeros(3, 6, 6, dtype=torch.float64), torch.zeros(3, 6, 6, dtype=torch.float64)
    alpha[0, 3, 1], alpha[0, 5, 1], alpha[0, 3, 3] = -1.2, 0.5, 0.25
    beta[1, 3, 1], beta[1, 5, 1], beta[1, 3, 3] = -1.0, 0.4, -0.25
    alpha[2, 3, 0] = -0.5
    return GradientNonlinearity(alpha, beta, R0)


def sgp(q, delta, big_delta, axis=0):
    """Pulsed-gradient spin echo with q = 2 pi G delta [rad/m]."""
    g = [0.0, 0.0, 0.0]
    g[axis] = q / (2 * math.pi * delta)
    b = bp.SequenceBuilder().rf(1e-6, flip_deg=90.0).delay(delta, grad=g).delay(big_delta - delta - 1e-6)
    b.rf(1e-6, flip_deg=180.0, phase=math.pi / 2).delay(delta, grad=g).readout(1, 2e-6)
    return b.build()


def partial_volume(inside, grids, h, sub=4):
    """Fraction of each cell inside the object, from sub^d samples per cell."""
    offs = (torch.arange(sub, dtype=torch.float64) + 0.5) / sub - 0.5
    frac = torch.zeros_like(grids[0])
    for o in torch.cartesian_prod(*[offs] * len(grids)).reshape(-1, len(grids)):
        frac += inside(*(g + oi * h for g, oi in zip(grids, o))).double()
    return frac / sub ** len(grids)


def coordinates(shape, voxel, origin=None):
    if origin is None:
        origin = tuple(-(n - 1) * h / 2 for n, h in reversed(list(zip(shape, voxel))))
    axes = [torch.arange(n, dtype=torch.float64) * h + o
            for n, h, o in zip(shape, voxel, reversed(origin))]
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    return torch.stack((x, y, z), -1)


def sphere(n=48):
    h, radius = 0.064 / n, 0.01
    pos = coordinates((n,) * 3, (h,) * 3)
    x, y, z = pos.unbind(-1)
    chi = DCHI * partial_volume(lambda x, y, z: x*x + y*y + z*z <= radius**2,
                               (x, y, z), h)
    ref = b0_sphere(pos, (0, 0, 0), radius, DCHI, FIELD)
    return chi, pos, h, radius, ref
