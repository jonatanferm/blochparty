"""The capability matrix of ``docs/ENGINES.md``, cell by cell, so it cannot silently go stale.

The matrix is parsed from the document. For every feature row there is a small case below
(``CASES``, same name), and every cell is checked:

* ``✓`` forward: the engine agrees with the core (``blochparty.simulate``, float64) on the case,
  signal and final magnetization (bound pool and free pools included), to the engine's tolerance
  (``TOL``). For the core itself the cell says it runs; its references are the other test files
  named in the document.
* ``✓`` gradients: core: autograd matches a (Richardson-extrapolated) central difference of the
  same loss along a random direction; Metal, CUDA and chunked: their gradients match the core's autograd.
* ``✗``: the engine raises ``NotImplementedError`` (forward: on the plain case; gradients: on the
  case with its tensors requiring grad, grad mode on).
* ``–``: not applicable, not checked.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import torch

import blochparty as bp
from blochparty.chunked import simulate_chunked
from blochparty.fields import Scanner
from blochparty.girf import GIRF, EddyCurrents, apply_girf
from blochparty.motion import rotation_z

from cases import bm_case, distortion, pgse, pool_case, random_case, sgp
from helpers import rel, with_readout

f64 = torch.float64
DOC = Path(__file__).resolve().parents[2] / "docs" / "ENGINES.md"
ENGINES = ("core", "metal", "cuda", "chunked")
# agreement with the core, max |engine - core| / scale per output (``errors``); observed worst cases
# in docs/ENGINES.md ("Agreement")
TOL = {"metal": 5e-5, "cuda": 5e-5, "chunked": 1e-12}
# Metal and CUDA with walking spins (diffusion): the kernel walks absolute positions in float32, which
# perturbs every spin's phase; the per-spin states feel it most, the signal averages it
TOL_METAL_WALK = {"signal": 2e-4, "states": 3e-3}
GRAD_TOL = {"core": 1e-6, "chunked": 1e-12, "metal": 2e-4, "cuda": 2e-4}  # core: vs finite differences
# metal, cuda: the float32 adjoint vs the core's float64 autograd


# ---------------------------------------------------------------- the document


def read_matrix(path: Path = DOC) -> dict[str, dict[tuple[str, str], str]]:
    """{feature: {(engine, "forward" | "gradients"): "✓" | "✗" | "–"}} from the table under
    "## Capability matrix" (columns: feature, then forward and gradients for each engine)."""
    text = path.read_text()
    section = text.split("## Capability matrix", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("|")]
    head = [c.strip() for c in rows[0].strip("|").split("|")]
    cols = []
    for c in head[1:]:
        m = re.fullmatch(r"(\w+) (fwd|grad)", c)
        if m:
            cols.append((m[1], "forward" if m[2] == "fwd" else "gradients"))
    out = {}
    for line in rows[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        name = cells[0].replace("`", "")
        out[name] = dict(zip(cols, cells[1:1 + len(cols)]))
    return out


# ---------------------------------------------------------------- the cases


class Case(NamedTuple):
    phantom: bp.Phantom
    seq: bp.Sequence
    kw: dict  # coils, m_init, scanner, maps, seed
    leaves: list  # tensors that require grad (empty without grad)


class Leaves:
    """``leaf(t)``: with grad on, a fresh leaf copy of ``t`` that requires grad, remembered. With
    ``shift`` (a list, one per leaf in call order), ``t + shift[i]`` instead, without grad: the
    case rebuilt at a perturbed point, derived tensors included."""

    def __init__(self, on: bool, shift=None):
        self.on, self.shift, self.items = on, shift, []

    def __call__(self, t):
        if self.shift is not None:
            t = t + self.shift[len(self.items)]
            self.items.append(t)
            return t
        if not self.on:
            return t
        t = t.detach().clone().requires_grad_()
        self.items.append(t)
        return t


def _g(seed):
    return torch.Generator().manual_seed(seed)


def base(leaf, coils=True):
    """Random RF / gradients / off-resonance on six spins with a readout, two coils."""
    ph, seq = random_case(0)
    seq = with_readout(seq)
    ph = replace(ph, t1=leaf(ph.t1))
    seq = replace(seq, rf=leaf(seq.rf))
    return ph, seq, {"coils": leaf(bp.birdcage_sensitivities(ph.pos, 2))} if coils else {}


def case_basic(leaf):
    return base(leaf)


def case_b1(leaf):
    ph, seq, kw = base(leaf)
    return replace(ph, b1=leaf(torch.linspace(0.8, 1.2, len(ph), dtype=f64))), seq, kw


def _maps(leaf):
    """B0, B1+ and two coils on a 4 x 9 x 9 grid around the phantom."""
    zz, yy, xx = torch.meshgrid(torch.linspace(-0.1, 0.1, 4, dtype=f64), torch.linspace(-0.3, 0.3, 9, dtype=f64),
                                torch.linspace(-0.3, 0.3, 9, dtype=f64), indexing="ij")
    coils = torch.stack([torch.exp(-((xx - 0.1 * (c - 0.5)) ** 2 + yy ** 2) / 0.02) * torch.exp(1j * (3 * yy - 2 * c * xx))
                         for c in range(2)])
    return bp.FieldMaps((-0.3, -0.3, -0.1), (0.075, 0.075, 0.2 / 3), b0=leaf(40 * xx ** 2 - 20 * yy + 5 * zz),
                        b1=leaf(1 + 0.3 * xx), coils=leaf(coils))


def case_maps_static(leaf):
    ph, seq, _ = base(leaf, coils=False)
    return ph, seq, {"maps": _maps(leaf)}


def _motion(leaf, n):
    """Flow, a periodic displacement and a slow rotation about z, all at once."""
    g = _g(4)
    rnd = lambda *s: torch.rand(*s, generator=g, dtype=f64)
    return bp.Motion(velocity=leaf((rnd(n, 3) - 0.5) * 0.2), displacement=leaf((rnd(n, 3) - 0.5) * 0.004),
                     waveform=lambda t: np.sin(2 * np.pi * t / 0.02),
                     rigid=lambda t: tuple(map(np.stack, zip(*[rotation_z(2.0 * x, (0.002, 0.0, 0.0)) for x in t]))))


def case_maps_moving(leaf):
    ph, seq, kw = case_maps_static(leaf)
    return replace(ph, motion=_motion(leaf, len(ph))), seq, kw


def _with_bound(leaf, seed, model, shape="superlorentzian"):
    ph, seq = bm_case(seed, shape)
    seq = with_readout(seq)
    b = ph.bound
    ph = replace(ph, t1=leaf(ph.t1), b1=torch.linspace(0.8, 1.2, len(ph), dtype=f64),
                 bound=replace(b, f=leaf(b.f), k=leaf(b.k), model=model))
    seq = replace(seq, rf=leaf(seq.rf))
    m0 = torch.randn(len(ph), 4, dtype=f64, generator=_g(1)).abs()
    return ph, seq, {"coils": bp.birdcage_sensitivities(ph.pos, 2), "m_init": leaf(m0)}


def case_bound_rate(leaf):
    return _with_bound(leaf, 0, "rate")


def case_bound_generalized(leaf):
    return _with_bound(leaf, 1, "generalized")


def case_bound_rate_nocut(leaf):
    ph, seq, kw = _with_bound(leaf, 2, "rate")
    return replace(ph, bound=replace(ph.bound, cutoff=None)), seq, kw


def _pools(leaf, with_bound):
    ph, seq = pool_case(0, with_bound)
    seq = with_readout(seq)
    amide, myelin = ph.pools
    ph = replace(ph, t1=leaf(ph.t1), pools=(replace(amide, f=leaf(amide.f), k=leaf(amide.k)), myelin))
    return ph, replace(seq, rf=leaf(seq.rf)), {"coils": bp.birdcage_sensitivities(ph.pos, 2)}


def case_pools(leaf):
    return _pools(leaf, False)


def case_pools_rate(leaf):
    return _pools(leaf, True)


def _diffusing(leaf, n=200, seed=2):
    g = _g(seed)
    rnd = lambda *s: torch.rand(*s, generator=g, dtype=f64)
    th = 0.4
    rot = torch.tensor([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]], dtype=f64)
    d = rot @ torch.diag(torch.tensor([0.5e-9, 1.5e-9, 2.5e-9], dtype=f64)) @ rot.T
    # the leaf is a per-spin scale: a perturbation of the tensor itself must stay symmetric
    scale = leaf(0.5 + rnd(n))
    ph = bp.Phantom(pos=(rnd(n, 3) - 0.5) * 0.02, pd=torch.ones(n, dtype=f64), t1=leaf(torch.full((n,), 1.0, dtype=f64)),
                    t2=torch.full((n,), 0.08, dtype=f64), db0=(rnd(n) - 0.5) * 50, diffusion=scale[:, None, None] * d)
    seq = pgse((1.2e5, 3e4, -2e4), readout=8)
    return ph, replace(seq, rf=leaf(seq.rf)), {"coils": bp.birdcage_sensitivities(ph.pos, 2), "seed": 9}


def case_diffusion(leaf):
    return _diffusing(leaf)


def case_restricted(leaf):
    # D per spin: with D = 2e-9 everywhere the sub-step count ceil(dt / tau_max) sits exactly on an
    # integer for 1 ms segments, where the walk (and so the signal) jumps (docs/LESSONS.md)
    n = 300
    g = _g(5)
    ph = bp.Phantom(pos=(torch.rand(n, 3, dtype=f64, generator=g) - 0.5) * 0.01, pd=torch.ones(n, dtype=f64),
                    t1=leaf(torch.full((n,), 1.0, dtype=f64)), t2=torch.full((n,), 0.08, dtype=f64),
                    db0=torch.zeros(n, dtype=f64), diffusion=leaf((1.5 + torch.rand(n, dtype=f64, generator=g)) * 1e-9),
                    restriction=bp.Restriction("cylinder", 5e-6, (0.0, 0.6, 0.8)))
    seq = sgp(1.0 / 5e-6, 1e-3, 10e-3, axis=1)
    return ph, replace(seq, rf=leaf(seq.rf)), {"seed": 2}


def case_motion(leaf):
    ph, seq, kw = base(leaf)
    return replace(ph, motion=_motion(leaf, len(ph))), seq, kw


def _strong(leaf):
    """The base case with gradients 50 times stronger (~5 mT/m): concomitant fields of ~10 Hz at 0.55 T."""
    ph, seq, kw = base(leaf)
    return ph, replace(seq, grad=seq.grad * 50), kw


def case_conc_leading(leaf):
    ph, seq, kw = _strong(leaf)
    return ph, seq, {**kw, "scanner": Scanner(field_strength=leaf(torch.tensor(0.55, dtype=f64)))}


def case_conc_exact(leaf):
    ph, seq, kw = _strong(leaf)
    return ph, seq, {**kw, "scanner": Scanner(field_strength=leaf(torch.tensor(0.55, dtype=f64)), concomitant="exact")}


def _nonlinear(leaf):
    nl = distortion()
    return Scanner(nonlinearity=replace(nl, alpha=leaf(nl.alpha)))


def case_nonlinearity(leaf):
    ph, seq, kw = base(leaf)
    return ph, seq, {**kw, "scanner": _nonlinear(leaf)}


def case_girf(leaf):
    ph = bp.shepp_logan(8)
    amp = leaf(torch.tensor([[0.01], [0.02], [0.015]], dtype=f64))
    b0a = leaf(torch.tensor([[1e-5], [2e-5], [-1e-5]], dtype=f64))
    model = [GIRF.identity(delay=[2e-6, 2e-6, 3e-6]),
             EddyCurrents.per_axis(amp, [[0.5e-3]] * 3, b0_amplitude=b0a, b0_time_constant=[[1e-3]] * 3)]
    seq = apply_girf(bp.gre_2d(8, n_dummy=2, ramps=True), model)
    return replace(ph, t1=leaf(ph.t1)), seq, {"coils": bp.birdcage_sensitivities(ph.pos, 2)}


def case_b0_offset(leaf):
    ph, seq, kw = base(leaf)
    off = 30.0 * torch.sin(torch.arange(len(seq), dtype=f64) * 0.05)
    return ph, replace(seq, b0_offset=leaf(off)), kw


def case_rf_freq(leaf):
    # an explicit carrier per segment (leaves must not depend on other leaves: the finite
    # difference rebuilds the case, autograd does not see through a leaf)
    ph, seq, kw = _with_bound(leaf, 0, "rate")
    carrier = 2000.0 * torch.cos(torch.arange(len(seq), dtype=f64) * 0.1)
    return ph, replace(seq, rf_freq=leaf(carrier)), kw


def case_bound_only(leaf):
    # every third RF segment saturates the bound pool alone (the far slices' pulses of tse_2d)
    ph, seq, kw = _with_bound(leaf, 0, "rate")
    flags = [bool(rf) and i % 3 == 0 for i, rf in enumerate(seq.is_rf)]
    assert any(flags)
    return ph, replace(seq, bound_only=flags), kw


def case_many_coils(leaf):
    ph, seq, kw = base(leaf)
    coils = torch.randn(len(ph), 40, dtype=torch.complex128, generator=_g(0))
    return ph, seq, {**kw, "coils": leaf(coils)}


def case_m_init(leaf):
    ph, seq, kw = base(leaf)
    return ph, seq, {**kw, "m_init": leaf(torch.randn(len(ph), 3, dtype=f64, generator=_g(3)))}


def case_pools_generalized(leaf):
    ph, seq, kw = _pools(leaf, True)
    return replace(ph, bound=replace(ph.bound, model="generalized")), seq, kw


def case_pools_diffusion(leaf):
    ph, seq, kw = _pools(leaf, False)
    return replace(ph, diffusion=torch.full((len(ph),), 2e-9, dtype=f64)), seq, kw


def case_pools_motion(leaf):
    ph, seq, kw = _pools(leaf, False)
    return replace(ph, motion=_motion(leaf, len(ph))), seq, kw


def case_pools_maps(leaf):
    ph, seq, _ = _pools(leaf, False)
    return ph, seq, {"maps": _maps(leaf)}


def case_motion_nonlinearity(leaf):
    ph, seq, kw = case_motion(leaf)
    return ph, seq, {**kw, "scanner": _nonlinear(leaf)}


def case_motion_conc(leaf):
    ph, seq, kw = case_motion(leaf)
    return ph, seq, {**kw, "scanner": Scanner(field_strength=0.55)}


def _diffusing_strong(leaf):
    ph, seq, kw = _diffusing(leaf)
    return replace(ph, pos=ph.pos * 10), seq, kw  # 20 cm: concomitant and nonlinear terms matter


def case_diffusion_conc_leading(leaf):
    ph, seq, kw = _diffusing_strong(leaf)
    return ph, seq, {**kw, "scanner": Scanner(field_strength=leaf(torch.tensor(0.55, dtype=f64)))}


def case_diffusion_conc_exact(leaf):
    ph, seq, kw = _diffusing_strong(leaf)
    return ph, seq, {**kw, "scanner": Scanner(field_strength=0.55, concomitant="exact")}


def case_diffusion_nonlinearity(leaf):
    ph, seq, kw = _diffusing_strong(leaf)
    return ph, seq, {**kw, "scanner": _nonlinear(leaf)}


def case_bound_field_model(leaf):
    ph, seq, kw = _with_bound(leaf, 0, "rate")
    seq = replace(seq, grad=seq.grad * 20)
    return ph, seq, {**kw, "scanner": Scanner(field_strength=leaf(torch.tensor(0.55, dtype=f64)),
                                              nonlinearity=distortion())}


def case_pools_field_model(leaf):
    ph, seq, kw = _pools(leaf, True)
    seq = replace(seq, grad=seq.grad * 30)  # ~3 mT/m: concomitant fields of a few Hz at 0.55 T
    return ph, seq, {**kw, "scanner": Scanner(field_strength=leaf(torch.tensor(0.55, dtype=f64)),
                                              nonlinearity=distortion())}


def _bound_diffusing(leaf, model):
    ph, seq, kw = _diffusing(leaf, n=64)
    n = len(ph)
    bound = bp.BoundPool(f=leaf(torch.full((n,), 0.12, dtype=f64)), k=leaf(torch.full((n,), 40.0, dtype=f64)),
                         t1=torch.ones(n, dtype=f64), t2=torch.full((n,), 12e-6, dtype=f64), model=model)
    return replace(ph, bound=bound), seq, kw


def case_bound_diffusion(leaf):
    return _bound_diffusing(leaf, "rate")


def case_generalized_diffusion(leaf):
    return _bound_diffusing(leaf, "generalized")


CASES = {
    "basic Bloch: PD, T1, T2, dB0, RF, gradients, ADC, spoils, coils": case_basic,
    "B1+ (Phantom.b1)": case_b1,
    "FieldMaps, static spins": case_maps_static,
    "FieldMaps, moving spins": case_maps_moving,
    "bound pool, rate model": case_bound_rate,
    "bound pool, rate model without a cut (cutoff=None)": case_bound_rate_nocut,
    "bound pool, generalized Bloch": case_bound_generalized,
    "free pools (Phantom.pools)": case_pools,
    "free pools + rate bound pool": case_pools_rate,
    "diffusion, free (isotropic or tensor)": case_diffusion,
    "diffusion, restricted": case_restricted,
    "motion (flow, waveform, rigid)": case_motion,
    "concomitant fields, leading order": case_conc_leading,
    "concomitant fields, exact": case_conc_exact,
    "gradient nonlinearity": case_nonlinearity,
    "GIRF / eddy currents (apply_girf)": case_girf,
    "uniform field per segment (Sequence.b0_offset)": case_b0_offset,
    "RF carrier (Sequence.rf_freq)": case_rf_freq,
    "bound-pool-only RF segments (Sequence.bound_only)": case_bound_only,
    "many coils (40)": case_many_coils,
    "initial magnetization (m_init)": case_m_init,
    "bound pool + scanner field model": case_bound_field_model,
    "free pools + scanner field model": case_pools_field_model,
    "bound pool + diffusion": case_bound_diffusion,
    "generalized Bloch + diffusion": case_generalized_diffusion,
    "diffusion + leading-order concomitant fields": case_diffusion_conc_leading,
    "diffusion + gradient nonlinearity": case_diffusion_nonlinearity,
    "diffusion + exact concomitant fields": case_diffusion_conc_exact,
    "motion + gradient nonlinearity": case_motion_nonlinearity,
    "motion + concomitant fields": case_motion_conc,
    "free pools + generalized Bloch": case_pools_generalized,
    "free pools + diffusion": case_pools_diffusion,
    "free pools + motion": case_pools_motion,
    "free pools + FieldMaps": case_pools_maps,
}


def build(name: str, grad: bool, shift=None) -> Case:
    leaf = Leaves(grad, shift)
    ph, seq, kw = CASES[name](leaf)
    return Case(ph, seq, kw, leaf.items)


# ---------------------------------------------------------------- the engines


def engine(name: str):
    if name == "core":
        return bp.simulate
    if name == "metal":
        metal = pytest.importorskip("blochparty.metal")
        if not metal.available():
            pytest.skip("no Apple GPU (MPS)")
        return metal.simulate_metal
    if name == "cuda":
        cuda = pytest.importorskip("blochparty.cuda")
        if not cuda.available():
            pytest.skip("no CUDA device with NVRTC")
        return cuda.simulate_cuda
    if name == "chunked":
        return lambda *a, seed=None, **k: simulate_chunked(*a, spin_chunk=4, **k)
    raise ValueError(name)


def run(name: str, case: Case) -> bp.SimResult:
    return engine(name)(case.phantom, case.seq, **case.kw)


def outputs(r: bp.SimResult) -> dict:
    out = {"signal": r.signal, "m": r.m}
    if r.m_bound is not None:
        out["m_bound"] = r.m_bound
    if r.m_pools is not None:
        out["m_pools"] = r.m_pools
    return out


def errors(out: dict, ref: dict) -> dict:
    """Relative error of every output; the bound pool and free pools relative to the water's
    magnetization (a saturated bound pool is tiny, its own maximum no scale)."""
    m_scale = ref["m"].abs().max()
    return {k: rel(out[k], ref[k], None if k in ("signal", "m") else m_scale) for k in ref}


def loss(r: bp.SimResult) -> torch.Tensor:
    """Linear in every output, with fixed random weights."""
    total = 0
    for i, (k, v) in enumerate(sorted(outputs(r).items())):
        x = (torch.view_as_real(v) if v.is_complex() else v).double()  # the same weights for float32 engines
        w = torch.randn(x.shape, dtype=x.dtype, generator=_g(10 + i))
        total = total + (w * x).sum()
    return total


def directional_fd(feature: str, eps=1e-5) -> tuple[float, float]:
    """The core's (autograd, finite difference) derivative of the loss along leaves * u (relative
    steps, u ~ N(0, 1)); the difference rebuilds the case at every point. Central differences at
    eps and 2 eps, Richardson-extrapolated (error O(eps^4)): at a plain 1e-6 step the rounding of
    the loss alone was ~1e-6 relative on spins 10 cm out under a gradient nonlinearity."""
    case = build(feature, grad=True)
    grads = torch.autograd.grad(loss(run("core", case)), case.leaves, allow_unused=True)
    dirs = []
    for i, t in enumerate(case.leaves):
        u = torch.randn(t.shape, dtype=f64, generator=_g(20 + i))
        dirs.append((u * t.detach().abs()).to(t.dtype))
    ad = sum(float((g.conj() * d).real.sum()) for g, d in zip(grads, dirs) if g is not None)
    with torch.no_grad():
        val = lambda h: float(loss(run("core", build(feature, False, [h * d for d in dirs]))))
        d1, d2 = ((val(h) - val(-h)) / (2 * h) for h in (eps, 2 * eps))
    return ad, (4 * d1 - d2) / 3


# ---------------------------------------------------------------- the tests

MATRIX = read_matrix()
CELLS = [(feature, eng, kind, mark) for feature, row in MATRIX.items() for (eng, kind), mark in row.items()
         if mark != "–"]
SLOW = {("free pools + generalized Bloch", "core", "gradients")}  # 60-dimensional matrix exponentials, ~3 s


def test_matrix_rows_are_the_cases():
    assert list(MATRIX) == list(CASES)
    for feature, row in MATRIX.items():
        assert set(row) == {(e, k) for e in ENGINES for k in ("forward", "gradients")}, feature
        assert set(row.values()) <= {"✓", "✗", "–"}, feature
        for e in ENGINES:  # no gradients without a forward
            if row[(e, "forward")] == "✗":
                assert row[(e, "gradients")] == "✗", (feature, e)


@pytest.mark.parametrize("feature, eng, kind, mark",
                         [pytest.param(*c, marks=pytest.mark.slow) if c[:3] in SLOW else c for c in CELLS],
                         ids=[f"{e}-{k}-{f}" for f, e, k, _ in CELLS])
def test_cell(feature, eng, kind, mark):
    case = build(feature, grad=kind == "gradients")
    if mark == "✗":
        with pytest.raises(NotImplementedError):
            run(eng, case)
        return
    if kind == "forward":
        if eng == "core":
            assert torch.isfinite(run("core", case).signal).all()
            return
        ref, out = outputs(run("core", case)), outputs(run(eng, case))
        assert set(out) == set(ref)
        err = errors(out, ref)
        tol = {k: TOL[eng] for k in ref}
        if eng in ("metal", "cuda") and case.phantom.diffusion is not None:
            tol = {k: TOL_METAL_WALK["signal" if k == "signal" else "states"] for k in ref}
        assert all(err[k] < tol[k] for k in ref), err
        return
    assert case.leaves
    if eng == "core":
        ad, fd = directional_fd(feature)
        assert abs(ad - fd) <= GRAD_TOL["core"] * max(abs(fd), 1e-12), (ad, fd)
        return
    ref_case = build(feature, grad=True)  # its own graph: derived tensors are not shared between the runs
    ref = torch.autograd.grad(loss(run("core", ref_case)), ref_case.leaves, allow_unused=True)
    got = torch.autograd.grad(loss(run(eng, case)), case.leaves, allow_unused=True)
    for i, (a, b) in enumerate(zip(got, ref)):
        if b is None:
            assert a is None or not a.any(), i
            continue
        assert a is not None and rel(a, b) < GRAD_TOL[eng], (i, rel(a, b))


def test_capabilities_declare_the_matrix():
    """engines/capabilities.py says what the document says, cell by cell, for every engine whether or not it runs
    here: test_cell runs the engines that are present; this checks the declaration the engines refuse by."""
    from blochparty.engines import capabilities as C

    for feature, row in MATRIX.items():
        for kind in ("forward", "gradients"):
            case = build(feature, grad=kind == "gradients")
            used = C.features(case.phantom, case.seq, case.kw.get("m_init"), case.kw.get("scanner"), case.kw.get("maps"))
            for eng in ENGINES:
                if row[(eng, kind)] == "–":
                    continue
                declared = "✗" if C.unsupported(eng, used, grad=kind == "gradients") else "✓"
                assert declared == row[(eng, kind)], (feature, eng, kind, sorted(used))


def test_refusals_name_the_engine_and_the_feature():
    from blochparty.engines import capabilities as C

    with pytest.raises(NotImplementedError, match=r"chunked engine: simulating with diffusion \(Phantom.diffusion\)"):
        C.check("chunked", frozenset({"diffusion"}))
    with pytest.raises(NotImplementedError, match="metal engine: gradients through motion .* torch.no_grad"):
        C.check("metal", frozenset({"motion"}), grad=True)
    with pytest.raises(NotImplementedError, match="the generalized Bloch bound pool with free exchanging pools"):
        C.check("cuda", frozenset({"pools", "generalized"}))
    C.check("core", frozenset(C.FEATURES) - {"bound_only"})  # everything but the one refused pair


def test_dtypes():
    """Core and chunked compute in the phantom's dtype (float64 here)."""
    case = build("basic Bloch: PD, T1, T2, dB0, RF, gradients, ADC, spoils, coils", grad=False)
    for eng in ("core", "chunked"):
        assert run(eng, case).m.dtype == f64


def test_dtypes_gpu(gpu):
    """The GPU engines compute in float32 and return float32 whatever the input dtype."""
    case = build("basic Bloch: PD, T1, T2, dB0, RF, gradients, ADC, spoils, coils", grad=False)
    r = gpu.simulate(case.phantom, case.seq, **case.kw)
    assert case.phantom.pd.dtype == f64
    assert r.m.dtype == torch.float32 and r.signal.dtype == torch.complex64


def test_metal_runs_under_no_grad():
    """Under no_grad, inputs that require grad take the plain forward."""
    metal = pytest.importorskip("blochparty.metal")
    if not metal.available():
        pytest.skip("no Apple GPU (MPS)")
    case = build("B1+ (Phantom.b1)", grad=True)
    with torch.no_grad():
        out = metal.simulate_metal(case.phantom, case.seq, **case.kw)
    assert rel(out.signal, bp.simulate(case.phantom, case.seq, **case.kw).signal.detach()) < TOL["metal"]
