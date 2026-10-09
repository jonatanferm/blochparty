"""Digital-twin loop on NIST benchmarks 1 and 2 (ROADMAP M3): the reduced model used for fitting.

The full simulation of either benchmark images the whole plane (128 TRs, every readout sample, all
spheres and the water around them) and costs one to two minutes per forward pass on the GPU, and
several times that for a gradient. Fitting scanner parameters through it would take hours. The
fits here use a reduced model that keeps the physics of each sphere's ROI mean and drops the
imaging:

- **One column of spins per sphere** at its ROI centre: positions across the slab (+-1.5 slice
  thicknesses, so the slice profile and the slice-axis crushers are resolved) and across one pixel
  along the readout (the intra-voxel phase of the readout-axis crushers), at the sphere's centre
  along the phase-encoding axis; inside the ball the sphere's T1 / T2, beyond the wall water.
- **The TR that acquires the k-space centre line**, after the TRs that set up its steady state.
  With no phase encoding, the column's magnetization at the readout's k = 0 point is the image
  intensity of a uniform region, which is what a ROI mean of the interior measures.
- **Two ADC samples per echo**, the pair either side of k = 0 (+-1.9 cycles/m: 0.004 cycles across
  a pixel), averaged.

``check`` compares the reduced model's curves with the full simulation's ROI means at nominal
parameters; the fits are only meaningful where the two agree.

    uv run python validation/real/nist_twin.py check
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

import blochparty as bp
from blochparty.pulseq import read_pulseq
from nist import CENTRE, PIXEL, SLICE, WATER_T1, WATER_T2, coarsen_rf, shape_error

HERE = Path(__file__).parent


# ---------------------------------------------------------------- sequences


def tr_bounds(seq: bp.Sequence) -> list[int]:
    """Segment index of each TR's first RF segment. The period is the median spacing of readouts
    that follow more than a second without ADC (one per TR in both benchmarks; an RF gap would not
    do: the IR-SE's inversion-to-excitation gap is up to 3 s); TRs start at the RF runs whose start
    time is a whole number of periods after the first run's."""
    t = np.r_[0, np.cumsum(seq.dt.double().numpy())]
    rf, adc = np.asarray(seq.is_rf), np.asarray(seq.adc)
    ta = t[np.flatnonzero(adc)]
    groups = ta[np.r_[True, np.diff(ta) > 1.0]]
    period = float(np.median(np.diff(groups)))
    runs = np.flatnonzero(rf & ~np.r_[False, rf[:-1]])
    phase = (t[runs] - t[runs[0]]) / period
    return [int(r) for r, ph in zip(runs, phase) if abs(ph - round(ph)) * period < 1e-3]


def axes(seq: bp.Sequence) -> tuple[int, int]:
    """(readout, phase-encoding) gradient axes: the readout is the axis with the largest gradient
    during ADC samples, phase encoding the remaining in-plane axis (the slice axis is x for both)."""
    g = seq.grad.double().abs()[torch.as_tensor(seq.adc)]
    ro = int(g.max(0).values[1:].argmax()) + 1
    return ro, 3 - ro


def k0_tr(seq: bp.Sequence, starts: list[int], first: int = 0) -> int:
    """Index (into ``starts``) of the TR whose phase-encoding moment (z) before its first ADC
    sample is smallest in magnitude, among TRs ``first`` onwards: the k-space centre line."""
    dt, g, adc = seq.dt.double().numpy(), seq.grad.double().numpy(), np.asarray(seq.adc)
    pe = axes(seq)[1]
    bounds = list(starts) + [len(seq)]
    best, arg = math.inf, None
    for n in range(first, len(starts)):
        s, e = bounds[n], bounds[n + 1]
        hits = np.flatnonzero(adc[s:e])
        if not len(hits):
            continue
        area = abs(float((g[s:s + hits[0], pe] * dt[s:s + hits[0]]).sum()))
        if area < best:
            best, arg = area, n
    return arg


def centre_samples(seq: bp.Sequence, a: int, b: int, n_read: int) -> np.ndarray:
    """Segment indices (within [a, b)) of the two ADC samples either side of k = 0 in each echo,
    found from the readout (y) moment since the excitation centre, its sign flipped at every
    refocusing pulse centre. Shape (n_echo, 2)."""
    dt, g = seq.dt[a:b].double().numpy(), seq.grad[a:b, axes(seq)[0]].double().numpy()
    rf, adc = np.asarray(seq.is_rf[a:b]), np.asarray(seq.adc[a:b])
    t = np.r_[0, np.cumsum(dt)]
    centres, i = [], 0
    while i < len(rf):
        if rf[i]:
            j = i
            while j < len(rf) and rf[j]:
                j += 1
            centres.append(0.5 * (t[i] + t[j]))
            i = j
        else:
            i += 1
    # the excitation is the second-to-last pulse before the first echo: 90 then the first 180
    # (TSE: 90, 180; IR-SE: inversion, 90, 180)
    first_adc_t = t[np.flatnonzero(adc)[0]]
    before = [c for c in centres if c < first_adc_t]
    exc = before[-2]
    flips = [c for c in centres if c > exc]
    m, cur = np.zeros(len(dt)), 0.0
    for k in range(len(dt)):
        pts = [t[k]] + [c for c in flips if t[k] < c < t[k + 1]] + [t[k + 1]]
        for p, q in zip(pts[:-1], pts[1:]):
            if p >= exc:
                cur += g[k] * (q - p)
            if q in flips:
                cur = -cur
        m[k] = cur  # moment at the end of segment k (the sample time of an ADC segment)
    idx = np.flatnonzero(adc)
    out = []
    for e in range(len(idx) // n_read):
        ii = idx[e * n_read:(e + 1) * n_read]
        j = int(np.flatnonzero(np.diff(np.sign(m[ii])) != 0)[0])  # sign change between samples j and j + 1
        out.append([a + ii[j], a + ii[j + 1]])
    return np.array(out)


def reduced(seq: bp.Sequence, n_warm: int, n_read: int, first: int = 0) -> tuple[bp.Sequence, np.ndarray]:
    """The k-space centre TR and the ``n_warm`` TRs before it, ADC only at each echo's k = 0 pair.
    Returns the sequence and the (n_echo, 2) ADC sample indices of the pairs in its signal."""
    starts = tr_bounds(seq)
    k = k0_tr(seq, starts, first)
    a = starts[k - n_warm]
    b = starts[k + 1] if k + 1 < len(starts) else len(seq)
    pairs = centre_samples(seq, starts[k], b, n_read)
    keep = np.zeros(len(seq), dtype=bool)
    keep[pairs.ravel()] = True
    adc = [bool(keep[i]) for i in range(a, b)]
    out = replace(seq, dt=seq.dt[a:b], rf=seq.rf[a:b], grad=seq.grad[a:b], adc_phase=seq.adc_phase[a:b],
                  is_rf=list(seq.is_rf[a:b]), adc=adc, spoil=list(seq.spoil[a:b]), meta={})
    order = np.cumsum(adc) - 1  # signal column of each kept sample
    return out, order[pairs - a]


# ---------------------------------------------------------------- spins


@dataclass
class Columns:
    """Columns of spins: ``phantom`` (all columns, listed column by column), ``column`` (column index
    per spin) and ``sphere`` (sphere index per column)."""
    phantom: bp.Phantom
    column: torch.Tensor
    sphere: torch.Tensor


def columns(rois, radii, t1_ms, t2_ms, x0, *, n_slice=128, sub_read=16, grid=1, step=2, wall=1e-3,
            water=(WATER_T1, WATER_T2), dtype=torch.float32) -> Columns:
    """Columns at each ROI's centre (y from its row, z from its column, as ``nist.phantom``), or on
    a ``grid`` x ``grid`` lattice ``step`` pixels apart around it: x stratified over x0 +- 1.5
    slice thicknesses, y regularly over one pixel. Inside the sphere's ball (3D distance to its
    centre) the sphere's relaxation, nothing in the wall, water beyond (``water=None``: none). PD
    per spin so that a column's signal is its mean magnetization."""
    half = 1.5 * SLICE
    xs = (torch.arange(n_slice, dtype=torch.float64) + 0.5) / n_slice * 2 * half - half
    ys = (torch.arange(sub_read, dtype=torch.float64) + 0.5) / sub_read * PIXEL - PIXEL / 2
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    gx, gy = gx.flatten(), gy.flatten()
    offs = (torch.arange(grid, dtype=torch.float64) - (grid - 1) / 2) * step * PIXEL
    pos, t1, t2, pd, col, sph = [], [], [], [], [], []
    w = 1.0 / (n_slice * sub_read)
    for k, (roi, rad) in enumerate(zip(rois, radii)):
        r0, c0 = ndimage.center_of_mass(roi)
        y0, z0 = (r0 - CENTRE) * PIXEL, (c0 - CENTRE) * PIXEL
        for oy in offs:
            for oz in offs:
                c = len(sph)
                d2 = gx**2 + (gy + oy) ** 2 + oz**2
                inside, outside = d2 <= rad**2, d2 > (rad + wall) ** 2
                parts = [(inside, t1_ms[k] * 1e-3, t2_ms[k] * 1e-3)]
                if water is not None:
                    parts.append((outside, water[0], water[1]))
                for sel, tt1, tt2 in parts:
                    n = int(sel.sum())
                    pos.append(torch.stack([x0 + gx[sel], y0 + oy + gy[sel], torch.full((n,), float(z0 + oz),
                                                                                        dtype=torch.float64)], -1))
                    t1.append(torch.full((n,), tt1, dtype=torch.float64))
                    t2.append(torch.full((n,), tt2, dtype=torch.float64))
                    pd.append(torch.full((n,), w, dtype=torch.float64))
                    col.append(torch.full((n,), c))
                sph.append(k)
    pos, t1, t2, pd = (torch.cat(v) for v in (pos, t1, t2, pd))
    ph = bp.Phantom(pos=pos, pd=pd, t1=t1, t2=t2, db0=torch.zeros_like(pd)).to(dtype=dtype)
    return Columns(ph, torch.cat(col), torch.tensor(sph))


def curves(sim, cols: Columns, seq: bp.Sequence, pairs: np.ndarray, n_rois: int, phantom=None, **kw) -> torch.Tensor:
    """(n_echo, n_rois) magnitude at k = 0 of each column, averaged over each sphere's columns (as
    a ROI mean of a magnitude image): one coil per column (an indicator sensitivity), so one
    simulation gives every column's signal. ``phantom`` replaces ``cols.phantom`` (e.g. with B1+)."""
    ph = cols.phantom if phantom is None else phantom
    n_col = len(cols.sphere)
    sens = torch.zeros(len(ph), n_col, dtype=torch.complex64)
    sens[torch.arange(len(ph)), cols.column] = 1.0
    s = sim(ph, seq, sens, **kw).signal  # (n_col, n_adc)
    mag = s[:, torch.as_tensor(pairs)].mean(-1).abs()  # (n_col, n_echo)
    avg = torch.zeros(n_rois, n_col, dtype=mag.dtype)
    avg[cols.sphere, torch.arange(n_col)] = 1.0
    avg = avg / avg.sum(1, keepdim=True)
    return (avg @ mag).T  # (n_echo, n_rois)


# ---------------------------------------------------------------- the twin's parameters

R_B1 = 0.1  # [m] length scale of the B1+ model's coefficients


def b1_field(pos: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Transmit scale at ``pos`` (..., 3): s (1 + a (x^2 + y^2) / R^2 + b z^2 / R^2 + c_y y / R + c_z z / R),
    theta = (s, a, b, c_y, c_z). Axially symmetric about the bore (z) to second order, plus linear
    in-plane terms: one sagittal plane (fixed x) determines all five, and the symmetry carries the
    field to another plane (a free x term would not be identifiable from one plane)."""
    x, y, z = (pos[..., i].to(theta.dtype) for i in range(3))
    s, a, b, cy, cz = theta
    return s * (1 + a * (x**2 + y**2) / R_B1**2 + b * z**2 / R_B1**2 + cy * y / R_B1 + cz * z / R_B1)


def played(seq: bp.Sequence, delays: torch.Tensor | None) -> bp.Sequence:
    """The sequence with per-axis gradient delays [s] (``girf.GIRF.identity``), or as written."""
    if delays is None:
        return seq
    from blochparty.girf import GIRF, apply_girf

    return apply_girf(seq.to(dtype=torch.float64), GIRF.identity(delay=delays)).to(dtype=torch.float32)


class Bench:
    """One benchmark's reduced model: its sequences (one per image series: TIs or one TSE file) with
    their k = 0 pairs, the columns, and the measured ROI means."""

    def __init__(self, name: str, wall=1e-3, **col_kw):
        sys.path.insert(0, str(HERE))
        from nist import roi_means, slice_position, sphere_radius

        self.name = name
        if name == "irse":
            import nist_irse as m

            self.rois = m.sphere_rois()
            img = m.real_images()
            radii = [sphere_radius(img, r) for r in self.rois]
            t1, t2 = m.T1_REF["v2"], m.T2_REF["v2"]
            self.seqs = [reduced(coarsen_rf(read_pulseq(m.DATA / "acq" / f"irse_T1_TI{ti}.seq").to(dtype=torch.float32), 4), 4, 128)
                         for ti in m.TI]
            x0 = slice_position(m.DATA / "acq" / f"irse_T1_TI{m.TI[0]}.seq")
        else:
            import nist_tse as m

            self.rois = m.sphere_rois()
            img = m.real_images()
            radii = [m.sphere_radius_t1_plane()] * 14
            t1, t2 = m.T1_REF, m.T2_REF
            full = coarsen_rf(m.last_trs(read_pulseq(m.SEQ), 128 + 1 + 8).to(dtype=torch.float32), 4)
            self.seqs = [reduced(full, 8, 128, first=len(tr_bounds(full)) - 128)]
            x0 = m.imaged_slice()
        self.real = roi_means(img, self.rois)
        self.cols = columns(self.rois, radii, t1, t2, x0, wall=wall, **col_kw)

    def curves(self, theta=None, delays=None, sim=None) -> torch.Tensor:
        """(n_images, 14) model curves; differentiable in ``theta`` (B1+) and ``delays``."""
        if sim is None:
            from blochparty.metal import simulate_metal as sim
        ph = self.cols.phantom
        if theta is not None:
            ph = replace(ph, b1=b1_field(ph.pos, theta).to(ph.pd.dtype))
        return torch.cat([curves(sim, self.cols, played(s, delays), p, len(self.rois), phantom=ph) for s, p in self.seqs])


def shape_loss(sim_curves: torch.Tensor, real: np.ndarray, spheres) -> torch.Tensor:
    """Mean squared shape error over ``spheres`` (differentiable ``nist.shape_error``)."""
    s = sim_curves[:, spheres].double()
    r = torch.as_tensor(real[:, spheres], dtype=torch.float64)
    a = (s * r).sum(0) / (s * s).sum(0)
    return (((a * s - r) ** 2).sum(0) / (r * r).sum(0)).mean()


def explore():
    """Shape error (spheres 1-10 and per sphere) of the reduced model against global B1+ and
    per-axis gradient delays, both benchmarks: what can move the curves."""
    np.set_printoptions(precision=2, suppress=True, linewidth=170)
    one = torch.tensor([1.0, 0, 0, 0, 0], dtype=torch.float64)
    for name in ("tse", "irse"):
        b = Bench(name)
        print(f"{name}: {sum(len(s) for s, _ in b.seqs)} segments, {len(b.cols.phantom)} spins", flush=True)
        for s in (0.8, 0.9, 1.0, 1.1):
            e = 100 * shape_error(b.curves(one * torch.tensor([s, 1, 1, 1, 1])).double().numpy(), b.real)
            print(f"  B1 {s:.2f}: 1-10 {e[:10].mean():.2f} %  {e}", flush=True)
        for ax in range(3):
            for d in (-10e-6, 10e-6):
                dl = torch.zeros(3, dtype=torch.float64)
                dl[ax] = d
                e = 100 * shape_error(b.curves(delays=dl).double().numpy(), b.real)
                print(f"  delay {'xyz'[ax]} {d * 1e6:+.0f} us: 1-10 {e[:10].mean():.2f} %  {e}", flush=True)


FIT_SPHERES = list(range(10))  # spheres 1-10, as the shape metric (11-14 are limited by something B1+ does not reach)


def fit(bench: Bench, start=(0.9, 0.0, 0.0, 0.0, 0.0), max_iter=40, log=print) -> dict:
    """B1+ model fitted to ``bench`` by L-BFGS through the Metal adjoint (shape loss, spheres 1-10).
    Start off s = 1: inversion and refocusing are nearly symmetric in B1+ about 1, so the gradient
    vanishes there. Returns theta, the loss history and the fitted curves."""
    theta = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([theta], lr=1, max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                            tolerance_change=1e-12)
    hist = []

    def closure():
        opt.zero_grad()
        loss = shape_loss(bench.curves(theta), bench.real, FIT_SPHERES)
        loss.backward()
        hist.append((float(loss), theta.detach().clone(), theta.grad.detach().clone()))
        log(f"    loss {100 * float(loss) ** 0.5:.3f} %  theta {np.round(theta.detach().numpy(), 4)}")
        return loss

    opt.step(closure)
    best = min(hist, key=lambda h: h[0])
    return {"theta": best[1], "loss": best[0], "evals": len(hist)}


def fits(out: str):
    """Both benchmarks, fitted from s = 0.9 and from s = 1.1 (the two branches of |B1+ - 1|); each fit
    scored on its own benchmark and, with the reduced model, predicted on the other. Saves the
    thetas to ``out`` (.json) for the full-model predictions."""
    import json

    np.set_printoptions(precision=2, suppress=True, linewidth=170)
    benches = {n: Bench(n) for n in ("irse", "tse")}
    res = {}
    for name, b in benches.items():
        nominal = shape_error(b.curves().double().numpy(), b.real)
        print(f"{name}: nominal B1+  1-10 {100 * nominal[:10].mean():.2f} %  {100 * nominal}", flush=True)
        for s0 in (0.9, 1.1):
            print(f"  fit {name} from s = {s0}:", flush=True)
            f = fit(b, start=(s0, 0, 0, 0, 0), log=lambda m: print(m, flush=True))
            th = f["theta"]
            row = {"theta": th.tolist(), "evals": f["evals"]}
            for other, ob in benches.items():
                with torch.no_grad():
                    e = shape_error(ob.curves(th).double().numpy(), ob.real)
                row[other] = e.tolist()
                tag = "fitted" if other == name else "PREDICTED"
                print(f"    {tag} {other}: 1-10 {100 * e[:10].mean():.2f} %  {100 * e}", flush=True)
            pos = [ob.cols.phantom.pos[ob.cols.column == c][0] for ob in benches.values() for c in range(0, len(ob.cols.sphere))]
            b1 = b1_field(torch.stack(pos).double(), th).numpy()
            print(f"    B1+ at the sphere columns: irse {b1[:14].round(3)}  tse {b1[14:].round(3)}", flush=True)
            res[f"{name}@{s0}"] = row
    Path(out).write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------- full-model predictions (Monte Carlo)


@dataclass
class Physics:
    """What the full model includes; the ablation rows switch one thing at a time."""
    theta: tuple | None = None  # B1+ model; None: B1+ = 1
    fill: bool = True  # the water around the spheres (1 mm walls)
    slice_profile: bool = True  # False: spins of the nominal slice only, all at its centre (ideal slab)
    concomitant: bool = False  # leading-order concomitant fields at 3 T
    girf: bool = False  # the measured Prisma GIRF of benchmark 7 (Wu et al. 2022), gradients and B0 terms


def prisma_girf():
    """The GIRF estimated from benchmark 7's Meas1 (``girf.py``), cached next to its data (the raw
    files take minutes to decompress)."""
    import girf as g7

    from blochparty.girf import GIRF

    cache = g7.DATA / "prisma_girf_meas1.npz"
    if cache.exists():
        z = np.load(cache)
        return GIRF(torch.from_numpy(z["taps"]), float(z["raster"]), float(z["delay"]))
    i1 = g7.nominal(1)
    hs, hb = {}, {}
    for ax in "xyz":
        o, b0 = g7.outputs(1, ax)
        hs[ax], hb[ax] = g7.estimate(o, i1), g7.estimate(b0, i1)
    model = g7.to_girf(hs, hb)
    np.savez(cache, taps=model.taps.numpy(), raster=model.raster, delay=float(model.delay))
    return model


class Full:
    """One benchmark's full simulation (every TR and sample of the scanner's files), Monte Carlo over
    Sobol seeds as ``nist_irse.monte_carlo`` / ``nist_tse.monte_carlo``, with the twin's inputs."""

    def __init__(self, name: str, n=256, wall=1e-3, margin=8, dense=3, sparse=0.25):
        sys.path.insert(0, str(HERE))
        from nist import fill_mask, roi_means, slice_position, sphere_radius

        self.name, self.n, self.wall, self.margin, self.dense, self.sparse = name, n, wall, margin, dense, sparse
        if name == "irse":
            import nist_irse as m

            self.m = m
            self.rois = m.sphere_rois()
            self.img = m.real_images()
            self.radii = [sphere_radius(self.img, r) for r in self.rois]
            self.t1, self.t2 = m.T1_REF["v2"], m.T2_REF["v2"]
            self.seqs = [coarsen_rf(read_pulseq(m.DATA / "acq" / f"irse_T1_TI{ti}.seq").to(dtype=torch.float32), 4)
                         for ti in m.TI]
            self.x0 = slice_position(m.DATA / "acq" / f"irse_T1_TI{m.TI[0]}.seq")
            self.m0_image = self.img[-1]  # TI 3000 ms
        else:
            import nist_tse as m

            self.m = m
            self.rois = m.sphere_rois()
            self.img = m.real_images()
            self.radii = [m.sphere_radius_t1_plane()] * 14
            self.t1, self.t2 = m.T1_REF, m.T2_REF
            self.seqs = [coarsen_rf(m.last_trs(read_pulseq(m.SEQ), 128 + 1 + 8).to(dtype=torch.float32), 4)]
            self.x0 = m.imaged_slice()
            self.m0_image = self.img[0]  # echo 1
        self.real = roi_means(self.img, self.rois)
        r_out = (self.radii[0] if name == "tse" else max(self.radii)) + wall
        self.water_mask = fill_mask(self.img, self.rois, r_out)
        self._girf = None

    def phantoms(self, seed: int, phys: Physics) -> list[bp.Phantom]:
        from nist import fill_phantom_qmc, phantom_qmc

        r_in = self.radii[0] if self.name == "tse" else self.radii
        phs = [phantom_qmc(self.rois, self.radii, self.t1, self.t2, self.x0, n=self.n, seed=seed)]
        if phys.fill:
            phs.append(fill_phantom_qmc(self.rois, self.water_mask, r_in, self.wall, self.x0, margin=self.margin * PIXEL,
                                        t1_ms=WATER_T1 * 1e3, t2_ms=WATER_T2 * 1e3, n=self.n, dense=self.dense * PIXEL,
                                        sparse=self.sparse, seed=seed))
        out = []
        for ph in phs:
            if not phys.slice_profile:  # an ideal slab: the nominal slice's spins, at its centre
                keep = (ph.pos[:, 0] - self.x0).abs() <= SLICE / 2
                pos = ph.pos[keep].clone()
                pos[:, 0] = self.x0
                ph = bp.Phantom(pos, ph.pd[keep], ph.t1[keep], ph.t2[keep], ph.db0[keep])
            if phys.theta is not None:
                ph = replace(ph, b1=b1_field(ph.pos.double(), torch.as_tensor(phys.theta, dtype=torch.float64)).to(ph.pd.dtype))
            out.append(ph)
        return out

    def sequences(self, phys: Physics) -> list[bp.Sequence]:
        if not phys.girf:
            return self.seqs
        from blochparty.girf import apply_girf

        if self._girf is None:
            self._girf = prisma_girf()
        return [apply_girf(s.to(dtype=torch.float64), self._girf).to(dtype=torch.float32) for s in self.seqs]

    def images(self, phys: Physics, seeds: int, log=print) -> np.ndarray:
        """(seeds, n_images, N, N) complex single-coil images."""
        import time

        from blochparty.fields import Scanner
        from blochparty.metal import simulate_metal

        kw = {"scanner": Scanner(field_strength=3.0, concomitant="leading")} if phys.concomitant else {}
        seqs = self.sequences(phys)
        out = []
        for k in range(seeds):
            t0 = time.perf_counter()
            phs = self.phantoms(k, phys)
            if self.name == "irse":
                imgs = [self.m.sim_complex_image(sum(simulate_metal(ph, s, **kw).signal for ph in phs)) for s in seqs]
                out.append(np.stack(imgs))
            else:
                out.append(self.m.sim_complex_images(sum(simulate_metal(ph, seqs[0], **kw).signal for ph in phs)))
            torch.mps.empty_cache()  # long runs on a 16 GB machine: give cached GPU buffers back between seeds
            log(f"    seed {k}: {' + '.join(str(len(p)) for p in phs)} spins, {time.perf_counter() - t0:.0f} s, "
                f"GPU driver memory {torch.mps.driver_allocated_memory() / 2**30:.1f} GB")
        return np.array(out)


def scores(full: Full, imgs: np.ndarray) -> dict:
    """Shape (with jackknife SE) and absolute errors (uniform receive, and the water-fill receive
    profile) of seed images, spheres 1-10."""
    from nist import jackknife, mc_errors, mc_magnitude, receive_profile, roi_means
    from nist import absolute_error

    sh = mc_errors(imgs, full.rois, full.real)
    prof = receive_profile(full.m0_image, full.water_mask)

    def absolute(x, weight=None):
        mag = mc_magnitude(x)[0]
        if weight is not None:
            mag = mag * weight
        return float(absolute_error(roi_means(mag, full.rois), full.real)[:10].mean())

    ab, ab_se = jackknife(absolute, imgs)
    abr, abr_se = jackknife(lambda x: absolute(x, prof), imgs)
    return {"shape": float(sh.mean10), "shape_se": float(sh.mean10_se), "per_sphere": sh.err.tolist(),
            "abs": float(ab), "abs_se": float(ab_se), "abs_rx": float(abr), "abs_rx_se": float(abr_se)}


def predict(fits_json: str, key: str, target: str, out: str, seeds: str = "4"):
    """Full-model Monte Carlo on ``target`` with B1+ from fit ``key`` of ``fits_json`` (e.g.
    "irse@0.9"), next to nothing fitted and the ablations around the twin, all on the same seeds;
    paired differences against the twin row. Seed images to ``out`` (.npz)."""
    import json

    from nist import mc_paired_difference

    theta = tuple(json.loads(Path(fits_json).read_text())[key]["theta"])
    full = Full(target)
    fill = target == "tse"  # benchmark 1's water fill changed nothing significant (RESULTS.md) at 7x the spins
    rows = [("nothing fitted (B1+ = 1)", Physics(fill=fill)), (f"twin: B1+ fitted on {key}", Physics(theta=theta, fill=fill)),
            ("twin, no water fill" if fill else "twin + water fill", Physics(theta=theta, fill=not fill)),
            ("twin, ideal slab (no slice profile)", Physics(theta=theta, fill=fill, slice_profile=False)),
            ("twin + concomitant fields (3 T)", Physics(theta=theta, fill=fill, concomitant=True)),
            ("twin + Prisma GIRF (benchmark 7)", Physics(theta=theta, fill=fill, girf=True))]
    k = int(seeds)
    results, images = [], {}
    for i, (name, phys) in enumerate(rows):
        print(f"{target}: {name}", flush=True)
        imgs = full.images(phys, k, log=lambda m: print(m, flush=True))
        images[f"imgs{i}"] = imgs
        sc = scores(full, imgs)
        results.append((name, sc))
        print(f"  shape {100 * sc['shape']:.2f} ± {100 * sc['shape_se']:.2f} %, absolute {100 * sc['abs']:.2f} ± "
              f"{100 * sc['abs_se']:.2f} % (receive profile {100 * sc['abs_rx']:.2f} ± {100 * sc['abs_rx_se']:.2f} %)",
              flush=True)
        np.savez_compressed(out, names=[r[0] for r in rows[:i + 1]], **images)
    print(f"\n| {target}, B1+ from {key} | shape, spheres 1-10 | vs twin (paired) | absolute | absolute, receive profile |")
    print("|---|---|---|---|---|")
    for i, (name, sc) in enumerate(results):
        if i == 1:
            d = ""
        else:
            dd, se = mc_paired_difference(images["imgs1"], images[f"imgs{i}"], full.rois, full.real)
            d = f"{100 * dd:+.2f} ± {100 * se:.2f}"
        print(f"| {name} | {100 * sc['shape']:.2f} ± {100 * sc['shape_se']:.2f} % | {d} | {100 * sc['abs']:.2f} ± "
              f"{100 * sc['abs_se']:.2f} % | {100 * sc['abs_rx']:.2f} ± {100 * sc['abs_rx_se']:.2f} % |")
    Path(out).with_suffix(".json").write_text(json.dumps({n: sc for n, sc in results}, indent=1))


def gradcheck():
    """d(shape loss)/d(theta) through the Metal adjoint against central differences (reduced TSE)."""
    b = Bench("tse")
    theta = torch.tensor([0.92, -0.2, 0.1, 0.05, -0.03], dtype=torch.float64, requires_grad=True)
    loss = shape_loss(b.curves(theta), b.real, FIT_SPHERES)
    (g,) = torch.autograd.grad(loss, theta)
    for i in range(5):
        h = 1e-3
        e = torch.zeros(5, dtype=torch.float64)
        e[i] = h
        with torch.no_grad():
            fd = (shape_loss(b.curves(theta + e), b.real, FIT_SPHERES) - shape_loss(b.curves(theta - e), b.real, FIT_SPHERES)) / (2 * h)
        print(f"  d/dtheta[{i}]: adjoint {float(g[i]):+.5e}  central difference {float(fd):+.5e}  rel {abs(float(g[i] - fd)) / abs(float(fd)):.1e}")


# ---------------------------------------------------------------- check against the full model


VARIANTS = [("centre, no water ", dict(water=None)), ("3x3 grid, no water", dict(water=None, grid=3, sub_read=8, n_slice=64)),
            ("3x3 grid, water   ", dict(grid=3, sub_read=8, n_slice=64))]


def check():
    """Reduced vs full model at nominal parameters, both benchmarks: per-sphere curve shapes."""
    from blochparty.metal import simulate_metal

    sys.path.insert(0, str(HERE))
    import nist_irse as b1
    import nist_tse as b2
    from nist import phantom, roi_means, slice_position, sphere_radius

    np.set_printoptions(precision=2, suppress=True, linewidth=160)
    # benchmark 2
    rois = b2.sphere_rois()
    real = roi_means(b2.real_images(), rois)
    radii = [b2.sphere_radius_t1_plane()] * 14
    x0 = b2.imaged_slice()
    full_seq = coarsen_rf(b2.last_trs(read_pulseq(b2.SEQ), 128 + 1 + 8).to(dtype=torch.float32), 4)
    red, pairs = reduced(full_seq, 8, 128, first=len(tr_bounds(full_seq)) - 128)
    ph, _ = phantom(rois, radii, b2.T1_REF, b2.T2_REF, x0, n_slice=64, sub=1, sub_read=4)
    full_c = b2.simulate(ph, full_seq, rois)
    print(f"benchmark 2: {len(red)} segments (full: {len(full_seq)}, {len(ph)} spins)")
    print("  full vs data:           ", 100 * shape_error(full_c, real))
    for name, kw in VARIANTS:
        cols = columns(rois, radii, b2.T1_REF, b2.T2_REF, x0, **kw)
        red_c = curves(simulate_metal, cols, red, pairs, 14).double().numpy()
        print(f"  {name}: vs data {100 * shape_error(red_c, real)}\n  {' ' * len(name)}  vs full {100 * shape_error(red_c, full_c)}")
    # benchmark 1
    rois = b1.sphere_rois()
    real_img = b1.real_images()
    real = roi_means(real_img, rois)
    radii = [sphere_radius(real_img, r) for r in rois]
    t1, t2 = b1.T1_REF["v2"], b1.T2_REF["v2"]
    full_seqs, reds = [], []
    for ti in b1.TI:
        s = coarsen_rf(read_pulseq(b1.DATA / "acq" / f"irse_T1_TI{ti}.seq").to(dtype=torch.float32), 4)
        full_seqs.append(s)
        reds.append(reduced(s, 4, 128))
    x0 = slice_position(b1.DATA / "acq" / f"irse_T1_TI{b1.TI[0]}.seq")
    ph, _ = phantom(rois, radii, t1, t2, x0, n_slice=128, sub=2)
    full_c = roi_means(np.stack([b1.sim_images(s) for s in b1.simulate_all(ph, full_seqs)]), rois)
    print(f"benchmark 1: {len(reds[0][0])} segments per TI (full: {len(full_seqs[0])}, {len(ph)} spins)")
    print("  full vs data:           ", 100 * shape_error(full_c, real))
    for name, kw in VARIANTS:
        cols = columns(rois, radii, t1, t2, x0, **kw)
        red_c = np.concatenate([curves(simulate_metal, cols, s, p, 14).double().numpy() for s, p in reds])
        print(f"  {name}: vs data {100 * shape_error(red_c, real)}\n  {' ' * len(name)}  vs full {100 * shape_error(red_c, full_c)}")


if __name__ == "__main__":
    cmd = {"check": check, "explore": explore, "gradcheck": gradcheck, "fits": fits, "predict": predict}[sys.argv[1]]
    cmd(*sys.argv[2:])
