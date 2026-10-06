"""Gradient nonlinearity (``fields.GradientNonlinearity``).

Checked against: gradunwarp's ``siemens_B`` formula (spherical coordinates and SciPy's
associated Legendre functions, written out here), Laplace's equation for every basis
field, a synthetic coefficient file in both layouts, the image of a point grid (each
point must appear at its analytically mapped position), finite differences for the
coefficient gradients, and between the two simulators.
"""

import math
from dataclasses import replace

import numpy as np
import pytest
import scipy.special
import torch

import blochparty as bp
from blochparty import Phantom, simulate
from blochparty.fields import GradientNonlinearity, Scanner, solid_harmonics
from test_free_runs import busy_sequence, random_phantom
from test_fused import rel

fused = pytest.importorskip("blochparty.fused")
SIMS = {"core": simulate, "fused": fused.simulate_fused}
R0 = 0.25


@pytest.fixture(scope="module", autouse=True)
def extension():
    try:
        fused._ext()
    except Exception as e:  # no compiler / ninja: nothing to test
        pytest.skip(f"fused extension unavailable: {e}")


def gradunwarp_displacement(alpha, beta, pos, r0):
    """gradunwarp's siemens_B (core/unwarp_resample.py) times R0, per axis: spherical
    coordinates, scipy.special.lpmv (with the Condon-Shortley phase) and Siemens'
    (-1)^m sqrt((2n+1)(n-m)! / (2 (n+m)!)) normalisation."""
    x, y, z = np.asarray(pos, dtype=np.float64).T
    r = np.sqrt(x * x + y * y + z * z)
    cos_theta = z / np.where(r != 0, r, 1.0)
    phi = np.arctan2(y, x)
    out = []
    for a in range(3):
        b = np.zeros_like(r)
        for n in range(alpha.shape[1]):
            f = (r / r0) ** n
            for m in range(n + 1):
                f2 = alpha[a, n, m] * np.cos(m * phi) + beta[a, n, m] * np.sin(m * phi)
                p = scipy.special.lpmv(m, n, cos_theta)
                if m > 0:
                    p = p * (-1) ** m * math.sqrt((2 * n + 1) * math.factorial(n - m) / (2 * math.factorial(n + m)))
                b = b + f * p * f2
        out.append(r0 * b)
    return np.stack(out, -1)


def random_coefficients(size=9, seed=0, scale=0.05):
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(size, size, dtype=torch.bool).tril()
    mask[:2] = False  # no constant or linear terms
    rnd = lambda: scale * torch.randn(3, size, size, dtype=torch.float64, generator=g) * mask
    return rnd(), rnd()


def points(n=50, seed=0, extent=0.2):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 3, dtype=torch.float64, generator=g) - 0.5) * 2 * extent


def test_basis_matches_gradunwarp():
    alpha, beta = random_coefficients(11)
    pos = points()
    got = GradientNonlinearity(alpha, beta, R0).displacement(pos)
    ref = torch.from_numpy(gradunwarp_displacement(alpha.numpy(), beta.numpy(), pos.numpy(), R0))
    assert rel(got, ref) < 1e-12
    # each basis function on its own, including the linear ones
    c, s = solid_harmonics(pos, 6, R0)
    for n in range(7):
        for m in range(n + 1):
            unit = torch.zeros(3, 7, 7, dtype=torch.float64)
            unit[0, n, m] = 1.0
            ref_c = gradunwarp_displacement(unit.numpy(), 0 * unit.numpy(), pos.numpy(), R0)[:, 0]
            ref_s = gradunwarp_displacement(0 * unit.numpy(), unit.numpy(), pos.numpy(), R0)[:, 0]
            assert np.allclose(c[:, n, m].numpy(), ref_c, rtol=1e-11, atol=1e-14), (n, m)
            assert np.allclose(s[:, n, m].numpy(), ref_s, rtol=1e-11, atol=1e-14), (n, m)
    # the linear terms are the coordinates, scaled: x, y and z
    assert torch.allclose(c[:, 1, 1] / math.sqrt(3 / 4), pos[:, 0]) and torch.allclose(c[:, 1, 0], pos[:, 2])


@pytest.mark.filterwarnings("ignore:`torch.jit.script` is deprecated")  # raised inside torch.func
def test_basis_is_harmonic():
    # Laplacian of every basis field (to n = 10) vanishes; second derivatives by autograd
    n_max = 10
    for p in points(6, seed=1, extent=0.15):
        def field(q):
            c, s = solid_harmonics(q[None], n_max, R0)
            return torch.cat([c.reshape(-1), s.reshape(-1)])

        hess = torch.func.hessian(field)(p)  # (2 (n + 1)^2, 3, 3)
        lap = hess.diagonal(dim1=-2, dim2=-1).sum(-1)  # (2 (n+1)^2,)
        scale = hess.abs().amax(dim=(-2, -1)) + 1e-300
        assert (lap.abs() / scale).max() < 1e-11
        assert scale.max() > 1.0  # the high orders do curve


SIEMENS = """#*[ Script ****************************************************************\\
# Name        : synthetic.grad
#*] END: */
 synthetic, test , Gx,y,z = 80/80 mT/m
 0.25 m = R0, lnorm = 4? A(1,0) = B(1,1) = A(1,1) = 0;
 0 = CoSyMode,

 NO. TYPE            SPECTRUM                     AXIS

  1 A( 3, 0)           -0.0100000                   z
  2 A( 5, 0)            0.02000000                  z
101 A( 3, 1)            0.0100000                   x
102 A( 3, 3)           -0.02000000                  x
103 A( 5, 1)            3.0e-02                     x
201 B( 3, 1)            0.01000000                  y
202 B( 5, 3)           -0.04000000                  y
"""

NAMED = """# gradunwarp variable layout
alpha_z 3 0 -0.01
alpha_z 5 0 0.02
alpha_x 3 1 0.01
alpha_x 3 3 -0.02
alpha_x 5 1 0.03
beta_y 3 1 0.01
beta_y 5 3 -0.04
"""


def test_coefficient_files(tmp_path):
    (tmp_path / "a.grad").write_text(SIEMENS)
    (tmp_path / "b.coef").write_text(NAMED)
    a = GradientNonlinearity.from_file(tmp_path / "a.grad")
    b = GradientNonlinearity.from_file(tmp_path / "b.coef", r0=0.25)
    assert a.r0 == 0.25 and a.alpha.shape == (3, 6, 6)
    expect = {("alpha", 2, 3, 0): -0.01, ("alpha", 2, 5, 0): 0.02, ("alpha", 0, 3, 1): 0.01, ("alpha", 0, 3, 3): -0.02,
              ("alpha", 0, 5, 1): 0.03, ("beta", 1, 3, 1): 0.01, ("beta", 1, 5, 3): -0.04}
    for model in (a, b):
        full = {"alpha": model.alpha, "beta": model.beta}
        assert sum(int((t != 0).sum()) for t in full.values()) == len(expect)
        for (kind, axis, n, m), v in expect.items():
            assert float(full[kind][axis, n, m]) == v
    with pytest.raises(ValueError, match="R0"):
        GradientNonlinearity.from_file(tmp_path / "b.coef")
    (tmp_path / "c.grad").write_text(SIEMENS.replace(" 0.25 m = R0", " R0 = 330 mm"))
    assert GradientNonlinearity.from_file(tmp_path / "c.grad").r0 == pytest.approx(0.33)


@pytest.mark.parametrize("sim", SIMS)
def test_zero_deviation_is_bit_identical(sim, tmp_path):
    # a file stating only the (zero) linear terms changes nothing, bit for bit
    (tmp_path / "lin.grad").write_text(" 0.25 m = R0\n 1 A( 1, 1) 0.0 x\n 2 B( 1, 1) 0.0 y\n 3 A( 1, 0) 0.0 z\n")
    scanner = Scanner(nonlinearity=GradientNonlinearity.from_file(tmp_path / "lin.grad"))
    ph = random_phantom(40)
    seq = busy_sequence(torch.tensor(800.0, dtype=torch.float64), torch.tensor(1500.0, dtype=torch.float64))
    kw = {"n_threads": 1} if sim == "fused" else {}
    a = SIMS[sim](ph, seq, scanner=scanner, **kw)
    b = SIMS[sim](ph, seq, **kw)
    assert torch.equal(a.signal, b.signal) and torch.equal(a.m, b.m)


@pytest.mark.parametrize("sim", SIMS)
def test_linear_terms_scale_the_gradients(sim):
    # A(1,1) on x, B(1,1) on y, A(1,0) on z: gradient scale errors (1 + c N_1m) per axis
    alpha, beta = torch.zeros(3, 2, 2, dtype=torch.float64), torch.zeros(3, 2, 2, dtype=torch.float64)
    alpha[0, 1, 1], beta[1, 1, 1], alpha[2, 1, 0] = 0.02, -0.03, 0.01
    scale = torch.tensor([1 + 0.02 * math.sqrt(3 / 4), 1 - 0.03 * math.sqrt(3 / 4), 1.01], dtype=torch.float64)
    ph = random_phantom(40)
    seq = busy_sequence(torch.tensor(800.0, dtype=torch.float64), torch.tensor(1500.0, dtype=torch.float64))
    a = SIMS[sim](ph, seq, scanner=Scanner(nonlinearity=GradientNonlinearity(alpha, beta, R0)))
    b = SIMS[sim](replace(ph, pos=ph.pos * scale), seq)
    assert rel(a.signal, b.signal) < 1e-12


def distortion():
    """x, y coils with third- and fifth-order terms: up to ~3 pixels at the edge of the grid."""
    alpha, beta = torch.zeros(3, 6, 6, dtype=torch.float64), torch.zeros(3, 6, 6, dtype=torch.float64)
    alpha[0, 3, 1], alpha[0, 5, 1], alpha[0, 3, 3] = -1.2, 0.5, 0.25
    beta[1, 3, 1], beta[1, 5, 1], beta[1, 3, 3] = -1.0, 0.4, -0.25
    alpha[2, 3, 0] = -0.5
    return GradientNonlinearity(alpha, beta, R0)


def peaks(image, guesses, fov, n, up=8):
    """Sub-pixel peak positions [m] (x, y) of a point image near each guess: Hann-windowed
    k-space, zero-padded ``up`` times, argmax in a window, then a parabola per axis."""
    k = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image)))
    w = torch.hann_window(n, periodic=True, dtype=torch.float64)
    k = k * w[:, None] * w[None, :]
    pad = (n * up - n) // 2
    big = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(torch.nn.functional.pad(k, (pad, pad, pad, pad))))).abs()
    m = n * up
    out = []
    for gx, gy in guesses:
        ix, iy = round(gx / fov * m + m / 2), round(gy / fov * m + m / 2)
        h = 3 * up
        win = big[iy - h : iy + h + 1, ix - h : ix + h + 1]
        j = int(win.argmax())
        jy, jx = divmod(j, win.shape[1])
        vy = win[jy - 1 : jy + 2, jx]
        vx = win[jy, jx - 1 : jx + 2]
        para = lambda v: float(0.5 * (v[0] - v[2]) / (v[0] - 2 * v[1] + v[2]))
        py = iy - h + jy + para(vy)
        px = ix - h + jx + para(vx)
        out.append(((px - m / 2) * fov / m, (py - m / 2) * fov / m))
    return torch.tensor(out, dtype=torch.float64)


@pytest.mark.parametrize("sim", SIMS)
def test_point_grid_images_at_mapped_positions(sim):
    n, fov, z = 64, 0.25, 0.04
    px = fov / n
    side = torch.arange(-3, 4, dtype=torch.float64) * 8 * px
    yy, xx = torch.meshgrid(side, side, indexing="ij")
    pos = torch.stack([xx.ravel(), yy.ravel(), torch.full_like(xx.ravel(), z)], -1)
    m = len(pos)
    ones = torch.ones(m, dtype=torch.float64)
    ph = Phantom(pos, ones, 1e-4 * ones, 1e3 * ones, 0 * ones)  # full recovery each TR, no decay
    seq = bp.gre_2d(n, fov, flip_deg=30.0, tr=5e-3, n_dummy=0)
    model = distortion()
    image = bp.reconstruct(SIMS[sim](ph, seq, scanner=Scanner(nonlinearity=model)).signal, seq)[0]
    mapped = torch.from_numpy(gradunwarp_displacement(model.alpha.numpy(), model.beta.numpy(), pos.numpy(), R0)) + pos
    found = peaks(image, pos[:, :2].tolist(), fov, n)
    err = (found - mapped[:, :2]).abs() / px
    shift = (mapped[:, :2] - pos[:, :2]).abs() / px
    assert shift.max() > 2.0  # a real distortion
    assert err.max() < 0.02, err.max()
    # the ideal scanner images the true grid
    ideal = bp.reconstruct(SIMS[sim](ph, seq).signal, seq)[0]
    assert ((peaks(ideal, pos[:, :2].tolist(), fov, n) - pos[:, :2]).abs() / px).max() < 0.05


def nl_loss(sim, alpha, beta, field_strength=None, **kw):
    ph = random_phantom(30, seed=2)
    seq = busy_sequence(torch.tensor(800.0, dtype=torch.float64), torch.tensor(1500.0, dtype=torch.float64), seed=2)
    seq = replace(seq, grad=seq.grad * 50.0)
    coils = torch.randn(len(ph), 2, dtype=torch.complex128, generator=torch.Generator().manual_seed(4))
    scanner = Scanner(field_strength=field_strength, nonlinearity=GradientNonlinearity(alpha, beta, R0))
    r = SIMS[sim](replace(ph, pos=ph.pos * 1.5), seq, coils, scanner=scanner, **kw)
    w = torch.randn(r.signal.shape, dtype=torch.complex128, generator=torch.Generator().manual_seed(9))
    return (r.signal * w).real.sum() + (r.m**2).sum()


@pytest.mark.parametrize("field_strength", [None, 0.55])
def test_coefficient_gradients(field_strength):
    alpha, beta = random_coefficients(6, seed=3, scale=0.2)
    grads = {}
    for sim in SIMS:
        a, b = alpha.clone().requires_grad_(), beta.clone().requires_grad_()
        kw = {"checkpoint_every": 7} if sim == "fused" else {}
        grads[sim] = torch.autograd.grad(nl_loss(sim, a, b, field_strength, **kw), [a, b])
    for ga, gb in zip(grads["core"], grads["fused"]):
        assert rel(gb, ga) < 1e-10
    # central differences on a few coefficients, through the fused kernel
    h = 1e-6
    for kind, idx in [(0, (0, 3, 1)), (0, (2, 4, 0)), (1, (1, 5, 3)), (0, (1, 2, 2))]:
        plus, minus = [alpha.clone(), beta.clone()], [alpha.clone(), beta.clone()]
        plus[kind][idx] += h
        minus[kind][idx] -= h
        with torch.no_grad():
            fd = (nl_loss("fused", *plus, field_strength) - nl_loss("fused", *minus, field_strength)) / (2 * h)
        got = grads["fused"][kind][idx]
        assert abs(float(got - fd)) < 1e-5 * max(1.0, abs(float(fd))), (kind, idx, float(got), float(fd))
