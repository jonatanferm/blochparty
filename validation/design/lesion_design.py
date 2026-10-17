"""Sequence design by classification (ROADMAP M7, agentic design, exploratory): can a spoiled GRE whose per-line flip
angles, TE, TR and k-space line positions are learned together with a network that reads its raw samples detect
white-matter lesions as well as, or better than, a standard T2w TSE read through a reconstruction, when every arm may
acquire only a fraction of k-space?

    uv run python validation/design/lesion_design.py --slabs ~/data/slabs --out ~/data/design_f0.2 --fraction 0.2

Task: "is there a lesion in this slab?" on the slabs of lesion_slabs.py (64 x 64 over 220 mm, a 5 mm slab, 10-20 mm
T2-hyperintense lesions at the imaged levels; the same heads with and without lesions; split by subject; slabs whose
lesion covers fewer than ``--min-lesion-px`` pixels are left out as ambiguous). Every arm acquires K = fraction x 64
phase-encode lines through the same 8-element receive ring with the same complex Gaussian noise per sample (``--snr``:
the image SNR of a reference GRE in the head, see ``snr_unit``):
  A  the T2w TSE undersampled uniformly to K lines (the centre lines kept) -> CG-SENSE with the exact coil maps (an
     upper bound for ESPIRiT-style calibration) -> a CNN on the magnitude image: the clinical baseline;
  B  the same K TSE lines -> the encoding's adjoint (per-coil zero-filled inverse DFT, no reconstruction) -> the CNN;
  G  a fixed spoiled GRE (25 deg, TE 6 ms, TR 15 ms, K lines evenly spread) -> adjoint -> CNN: what the design adds;
  C  the GRE's flips, TE, TR and line positions learned jointly with the CNN (started from G's) by gradient descent
     through the GPU adjoint (``bp.simulate(..., engine="gpu")``: Metal, or CUDA), with a replay buffer of recent
     acquisitions for extra network updates.
The adjoint is a fixed, parameter-free linear map of the raw samples, differentiable in the line positions (``--front
raw`` feeds the samples to the CNN as they are, which fails: a CNN cannot read a small lesion out of k-space). The K
lines are evenly spaced (``--sampling uniform``, SENSE's case) or drawn from a Gaussian density (``gauss``: incoherent
aliasing in the zero-filled image), the same set for every arm; arms G and C start from the best of ``--starts``. Reports
the test AUC per arm (A, B, G over ``--seeds`` seeds, accuracy at the threshold that is best on the training set), the
scan time and arm C's learned sequence; run at 1.0, 0.2, 0.1 and 0.05 to see where each arm fails.
"""
from __future__ import annotations

import argparse
import collections
import copy
import functools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import blochparty as bp
from blochparty.sequences.builder import SequenceBuilder
from blochparty.design import auc
from blochparty.recon import cg_sense, uniform_mask, variable_density_mask

simulate_metal = functools.partial(bp.simulate, engine="gpu")  # Metal, or CUDA on an NVIDIA machine

FOV, NX, DWELL = 0.22, 64, 10e-6
RF_DURATION = 0.2e-3
N_COILS, COIL_RADIUS = 8, 0.14
N_DUMMY = 8
T_READ, T_ECHO = NX * DWELL, (NX / 2 + 0.5) * DWELL
T_TAIL = T_READ - T_ECHO + RF_DURATION / 2 + 0.2e-3  # the least TR - TE: the rest of the readout and a 0.2 ms spoiler


def learnable_gre(flips_deg: torch.Tensor, ky_cycles: torch.Tensor, te: torch.Tensor, tr: torch.Tensor, *, nx=NX, fov=FOV,
                  dwell=DWELL, n_dummy: int = N_DUMMY, dtype=torch.float32) -> bp.Sequence:
    """A spoiled GRE of K = len(flips) lines: line j excites with ``flips_deg[j]`` and encodes ``ky_cycles[j]`` cycles
    across the field of view (continuous: not restricted to the Cartesian grid), all with echo time ``te`` and
    repetition time ``tr`` (tensors; gradients flow to all four through the builder and the adjoint). ``n_dummy``
    excitations with the mean flip bring the magnetisation toward its steady state first. Ideal gradients and spoiling."""
    dk = 1.0 / fov
    t_pre = te - RF_DURATION / 2 - T_ECHO
    t_fill = tr - RF_DURATION - t_pre - T_READ
    b = SequenceBuilder(dtype=dtype)
    mean_flip = flips_deg.mean()
    for _ in range(n_dummy):
        b.rf(RF_DURATION, flip_deg=mean_flip)
        b.delay(tr - RF_DURATION).spoil()
    for j in range(len(flips_deg)):
        ky = ky_cycles[j] * dk
        b.rf(RF_DURATION, flip_deg=flips_deg[j])
        b.delay(t_pre, grad=(-(nx / 2 + 0.5) * dk / t_pre, ky / t_pre, 0.0))
        b.readout(nx, dwell, grad=(dk / dwell, 0.0, 0.0))
        b.delay(t_fill).spoil()
    return b.build(kspace_shape=(len(flips_deg), nx), fov=fov, dwell=dwell)


class Design(nn.Module):
    """Arm C's parameters: flips in (2, 90) deg by a sigmoid, TE in ``te`` (s), TR = TE + the readout tail + an extra in
    ``tr_extra`` (so every design is feasible), ky in (-nx/2, nx/2) cycles by a tanh, all unconstrained underneath."""

    def __init__(self, lines: int, nx=NX, te=(2e-3, 80e-3), tr_extra=(0.5e-3, 100e-3), init_flip=25., init_te=6e-3, init_tr=15e-3,
                 ky_init: torch.Tensor | None = None):
        super().__init__()
        self.lines, self.nx, self.te_rng, self.tr_rng = lines, nx, te, tr_extra
        if not (te[0] < init_te < te[1] and tr_extra[0] < init_tr - init_te - T_TAIL < tr_extra[1]):
            raise ValueError(f"start TE {init_te * 1e3:.1f} / TR {init_tr * 1e3:.1f} ms outside the design's ranges")
        self.flip_raw = nn.Parameter(torch.full((lines,), math.log((init_flip - 2) / (90 - init_flip))))
        self.te_raw = nn.Parameter(torch.tensor(math.log((init_te - te[0]) / (te[1] - init_te))))
        extra = init_tr - init_te - T_TAIL
        self.tr_raw = nn.Parameter(torch.tensor(math.log((extra - tr_extra[0]) / (tr_extra[1] - extra))))
        if ky_init is None:
            ky_init = torch.linspace(-nx / 2 + nx / lines / 2, nx / 2 - nx / lines / 2, lines)  # evenly spread lines
        self.ky_raw = nn.Parameter(torch.atanh((ky_init.float() / (nx / 2)).clamp(-.999, .999)))

    def forward(self):
        flips = 2 + 88 * torch.sigmoid(self.flip_raw)
        te = self.te_rng[0] + (self.te_rng[1] - self.te_rng[0]) * torch.sigmoid(self.te_raw)
        tr = te + T_TAIL + self.tr_rng[0] + (self.tr_rng[1] - self.tr_rng[0]) * torch.sigmoid(self.tr_raw)
        ky = (self.nx / 2) * torch.tanh(self.ky_raw)
        return flips, ky, te, tr

    def sequence(self):
        return learnable_gre(*self())

    def describe(self):
        flips, ky, te, tr = (v.detach() for v in self())
        return {"flips_deg": flips.round(decimals=1).tolist(), "ky_cycles": ky.round(decimals=2).tolist(), "te_ms": round(te.item() * 1e3, 2),
                "tr_ms": round(tr.item() * 1e3, 2), "scan_ms": round(tr.item() * 1e3 * (self.lines + N_DUMMY), 1)}

    def summary(self):
        flips, ky, te, tr = (v.detach() for v in self())
        return (f"flip {flips.mean():.1f} ({flips.min():.1f}-{flips.max():.1f}) deg, TE {te * 1e3:.2f} ms, TR {tr * 1e3:.2f} ms, "
                f"ky {ky.min():.1f}..{ky.max():.1f} (std {ky.std():.1f}, |ky| < 4: {(ky.abs() < 4).sum().item()})")


def coils_for(ph) -> torch.Tensor:
    """(N, 8) sensitivities of an 8-element ring (radius 0.14 m) at the spins, complex64 (the same in every arm)."""
    return bp.birdcage_sensitivities(ph.pos, N_COILS, radius=COIL_RADIUS).to(torch.complex64)


def coil_maps(n=NX, fov=FOV) -> torch.Tensor:
    """(8, n, n) the ring's sensitivities on the image grid (the exact maps SENSE is given): rows along +y, columns
    along +x, the grid ``ifft2c`` of a (ky, kx) k-space maps to."""
    u = (torch.arange(n, dtype=torch.float32) - n / 2) * (fov / n)
    yy, xx = torch.meshgrid(u, u, indexing="ij")
    pos = torch.stack([xx.ravel(), yy.ravel(), torch.zeros(n * n)], 1)
    return bp.birdcage_sensitivities(pos, N_COILS, radius=COIL_RADIUS).to(torch.complex64).T.reshape(N_COILS, n, n)


def sampled_lines(n: int, lines: int, sampling: str, centre: int = 4, seed: int = 0) -> torch.Tensor:
    """(n,) bool: the phase-encode lines every arm acquires. ``uniform``: ``uniform_mask`` (evenly spaced, SENSE-style;
    arms G and C then start from evenly spread continuous positions). ``gauss``: the central ``centre`` lines and the
    rest drawn without replacement from a Gaussian density of width n/5 over the grid (variable density: a zero-filled
    image shows incoherent aliasing instead of folded copies of the head), deterministic in ``seed``; arms G and C
    start from the same lines."""
    if sampling == "uniform":
        return uniform_mask(n, lines, centre)
    return variable_density_mask(n, lines, centre, seed)


def adjoint_image(k: torch.Tensor, ky_cycles: torch.Tensor, nx: int = NX) -> torch.Tensor:
    """(C, L, nx) samples of L lines at ``ky_cycles`` (cycles across the field of view, any real values) with a
    Cartesian readout -> (C, nx, nx) the encoding's adjoint, normalised like ``bp.ifft2c`` and equal to it for on-grid
    lines (checked to 1e-6); differentiable in the samples and in the line positions."""
    kx = torch.fft.ifftshift(torch.fft.ifft(torch.fft.ifftshift(k, dim=-1), dim=-1), dim=-1)  # the readout axis exactly
    y = torch.arange(nx, dtype=torch.float32) - nx / 2
    E = torch.exp(2j * math.pi * ky_cycles[:, None] / nx * y[None, :]) / nx  # (L, nx): line j -> image row y
    return torch.einsum("clx,ly->cyx", kx.to(torch.complex64), E.to(torch.complex64))


def add_noise(k: torch.Tensor, sigma: float, gen: torch.Generator | None) -> torch.Tensor:
    noise = torch.randn(k.shape, generator=gen, dtype=torch.float32) + 1j * torch.randn(k.shape, generator=gen, dtype=torch.float32)
    return k + sigma * noise.to(k.dtype)


def snr_unit(ph) -> float:
    """``--snr`` is the image SNR of a reference acquisition (a 64-line 25 deg, TE 6 / TR 15 ms spoiled GRE, the
    arm-G/C starting point) in the head: the median head pixel of its root-sum-of-squares image over the noise standard
    deviation of a pixel. With ``bp.ifft2c``'s 1/N^2 normalisation a k-space sample noise of sigma gives sigma / N per
    pixel, so sigma = N x (median head pixel) / snr; this returns N x (median head pixel), the unit of sigma."""
    with torch.no_grad():
        seq = bp.gre_2d(NX, FOV, flip_deg=25., tr=15e-3, te=6e-3, dwell=DWELL, n_dummy=N_DUMMY).to(torch.float32)
        img = bp.ifft2c(bp.cartesian_kspace(simulate_metal(ph, seq, coils_for(ph)).signal, seq)).abs().pow(2).sum(0).sqrt()
        return float(img[img > 0.2 * img.max()].median()) * NX


class Net(nn.Module):
    """A small CNN with max-pooling over space at the end ("a lesion anywhere"): three conv blocks, dropout, a linear
    read-out. Pools each axis only while it has room (a few raw k-space lines are a short axis)."""

    def __init__(self, channels: int, shape: tuple[int, int], width=(32, 64, 128)):
        super().__init__()
        c = [channels, *width]
        layers = []
        h, w = shape
        for i in range(3):
            pool = (2 if h >= 2 else 1, 2 if w >= 2 else 1)
            h, w = h // pool[0], w // pool[1]
            layers += [nn.Conv2d(c[i], c[i + 1], 3, padding=1), nn.BatchNorm2d(c[i + 1]), nn.ReLU(), nn.MaxPool2d(pool)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(c[-1], 1))

    def forward(self, x):
        return self.head(self.features(x).amax((-2, -1))).squeeze(-1)


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Image inputs: a random left-right flip and a shift of up to 4 pixels (the field of view is centred on the head;
    a lesion may be anywhere). Raw k-space inputs are left alone."""
    if tuple(x.shape[-2:]) != (NX, NX):
        return x
    if torch.rand((), generator=gen) < .5:
        x = x.flip(-1)
    dy, dx = (int(v) for v in torch.randint(-4, 5, (2,), generator=gen))
    return torch.roll(x, (dy, dx), (-2, -1))


def sense_input(k_full: torch.Tensor, maps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(C, ky, kx) k-space with the lines of ``mask`` sampled -> the CG-SENSE magnitude, (1, n, n) scaled by its mean."""
    img = cg_sense(k_full * mask[None, :, None], maps, mask).abs()
    return (img / (img.mean() + 1e-12))[None].float()


def front_input(front: str, k_lines: torch.Tensor, ky_cycles: torch.Tensor) -> torch.Tensor:
    """(C, L, nx) raw samples of L lines at ``ky_cycles`` -> the CNN's input: ``rss`` the root-sum-of-squares of the
    adjoint images (1, n, n); ``coils`` the adjoint images' real and imaginary parts (2 C, n, n); ``raw`` the samples'
    real and imaginary parts (2 C, L, nx). Each scaled by its mean magnitude."""
    if front == "raw":
        k = k_lines / (k_lines.abs().mean() + 1e-12)
        return torch.cat([k.real, k.imag], 0).float()
    img = adjoint_image(k_lines, ky_cycles)
    if front == "rss":
        img = img.abs().pow(2).sum(0).sqrt()
        return (img / (img.mean() + 1e-12))[None].float()
    img = img / (img.abs().mean() + 1e-12)
    return torch.cat([img.real, img.imag], 0).float()


def best_threshold(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s)
    cand = np.concatenate([[0.], np.sort(np.unique(s)), [1.]])
    mids = (cand[1:] + cand[:-1]) / 2
    accs = [((s > t) == y).mean() for t in mids]
    return float(mids[int(np.argmax(accs))])


def load_index(folder: Path, min_lesion_px: float):
    """The slabs' records by split, without the ambiguous ones (a lesion drawn but fewer than ``min_lesion_px`` of its
    pixels in the slab)."""
    index = json.loads((folder / "index.json").read_text())
    kept = [r for r in index if not (0 < r["lesion_px"] < min_lesion_px)]
    data = {"train": [r for r in kept if r["split"] == "train"], "test": [r for r in kept if r["split"] == "test"]}
    if not data["test"]:  # a functional test on a handful of slabs: score on the training slabs and say so
        print("no test split: scoring on the training slabs (a functional test, not a result)", flush=True)
        data["test"] = data["train"]
    return data, len(index) - len(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slabs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fraction", type=float, default=0.2, help="the share of k-space every arm may acquire (K = fraction x 64 lines)")
    ap.add_argument("--snr", type=float, default=20., help="image SNR of the reference GRE in the head (snr_unit)")
    ap.add_argument("--front", default="rss", choices=("rss", "coils", "raw"), help="what the CNN of arms B, G and C reads (rss read best)")
    ap.add_argument("--epochs", type=int, default=100, help="arms A, B, G: epochs over the cached acquisitions")
    ap.add_argument("--seeds", type=int, default=3, help="arms A, B, G: repeats (initialisation, noise, order)")
    ap.add_argument("--steps", type=int, default=400, help="arm C: joint design + network steps, one simulated batch each")
    ap.add_argument("--batch", type=int, default=8, help="arm C: slabs per simulated step")
    ap.add_argument("--replay", type=int, default=4, help="arm C: extra network-only updates per step from the replay buffer")
    ap.add_argument("--design-lr", type=float, default=3e-3, help="arm C: Adam's rate for the flips, TE and TR (raw parameters)")
    ap.add_argument("--ky-lr", type=float, default=1e-3, help="arm C: Adam's rate for the line positions. Adam moves every parameter "
                    "~lr per step whatever the gradient's signal-to-noise, and a central line moves nx/2 cycles per unit of its raw "
                    "parameter: at 1e-2 the central lines wandered 0.3 cycles a step and the joint loss went 0.19 -> 0.69 in 20 steps")
    ap.add_argument("--net-lr", type=float, default=3e-5, help="arm C: the network's rate in the joint phase. It starts from G's converged "
                    "net (cosine-decayed to 1e-5); at 3e-4 with a fresh Adam and batches of 8 the joint phase wrecked it (100 %%: G 0.95 -> C 0.66 "
                    "with the design nearly unchanged), so the joint phase keeps batch-norm statistics and dropout fixed and moves the net gently")
    ap.add_argument("--starts", default="25:6:15,10:6:15,25:20:30,15:30:40,15:45:55,15:60:70",
                    help="arm G / C's starting designs as flip_deg:TE_ms:TR_ms (K lines evenly spread): each is fitted with one seed and the "
                         "best on the training set is arm G (refitted with --seeds) and arm C's start. A coarse search over the generic "
                         "GRE contrasts first, because a joint gradient from a chance-level network has nothing to follow: from the "
                         "25 deg / 6 ms / 15 ms start alone (a lesion 16 %% darker than white matter, like grey matter) the design did not move")
    ap.add_argument("--min-lesion-px", type=float, default=4.)
    ap.add_argument("--sampling", default="uniform", choices=("uniform", "gauss"), help="which K lines every arm acquires (sampled_lines)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="A,B,G,C")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data, dropped = load_index(args.slabs, args.min_lesion_px)
    train, test = data["train"], data["test"]
    y_train, y_test = [r["label"] for r in train], [r["label"] for r in test]
    print(f"slabs: {len(train)} train / {len(test)} test ({dropped} ambiguous left out); positives {sum(y_train)} / {sum(y_test)}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    lines = max(int(round(args.fraction * NX)), 2)
    mask = sampled_lines(NX, lines, args.sampling, seed=args.seed)
    cols = torch.nonzero(mask).squeeze(1)
    results = {"n": {"train": len(train), "test": len(test)}, "snr": args.snr, "fraction": args.fraction, "lines": lines, "coils": N_COILS,
               "front": args.front, "sampling": args.sampling, "epochs": args.epochs, "seeds": args.seeds, "steps": args.steps}
    print(f"fraction {args.fraction}: {lines} of {NX} lines, {args.sampling}; the TSE keeps ky indices {cols.tolist()}; front {args.front}; "
          f"device {device}", flush=True)

    def load_ph(rec):
        return torch.load(args.slabs / rec["file"], weights_only=False)["phantom"]

    def cached(tag, fn):
        """``fn(ph)`` for every slab, kept in ``<slabs>/cache_<tag>.pt`` (extended when slabs are added)."""
        path = args.slabs / f"cache_{tag}.pt"
        store = torch.load(path) if path.exists() else {}
        todo = [r for r in {r["file"]: r for r in train + test}.values() if r["file"] not in store]
        if todo:
            t0 = time.perf_counter()
            for i, rec in enumerate(todo):
                with torch.no_grad():
                    store[rec["file"]] = fn(load_ph(rec))
                if i % 100 == 99:
                    print(f"  cache {tag}: {i + 1} / {len(todo)}, {time.perf_counter() - t0:.0f} s", flush=True)
            torch.save(store, path)
            print(f"cache {tag}: {len(todo)} slabs in {time.perf_counter() - t0:.0f} s", flush=True)
        return store

    unit = cached("snr_unit", snr_unit)
    sigma = {f: unit[f] / args.snr for f in unit}

    def fit(name, make_input, seeds=None):
        """Arms A, B, G: ``make_input(rec, gen)`` builds the CNN's input with fresh noise; ``--seeds`` repeats; the
        final-epoch network is scored (no selection on the test set)."""
        runs, nets = [], []
        for s in range(args.seeds if seeds is None else seeds):
            torch.manual_seed(args.seed + s); gen = torch.Generator().manual_seed(args.seed + s)
            x0 = make_input(train[0], gen)
            net = Net(x0.shape[0], tuple(x0.shape[1:])).to(device)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-5)
            t0 = time.perf_counter()
            for ep in range(args.epochs):
                net.train()
                perm = torch.randperm(len(train), generator=gen); losses = []
                for i in range(0, len(perm), 16):
                    j = perm[i:i + 16]
                    xb = torch.stack([augment(make_input(train[q], gen), gen) for q in j]).to(device)
                    yb = torch.tensor([y_train[q] for q in j], dtype=torch.float32, device=device)
                    loss = nn.functional.binary_cross_entropy_with_logits(net(xb), yb)
                    opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
                sched.step()
                if ep % 25 == 24 or ep == args.epochs - 1:
                    print(f"  {name} seed {s} epoch {ep + 1}: train loss {np.mean(losses):.3f}, {time.perf_counter() - t0:.0f} s", flush=True)
            net.eval()
            with torch.no_grad():
                s_tr = [float(torch.sigmoid(net(make_input(r, gen)[None].to(device)))) for r in train]
                s_te = [float(torch.sigmoid(net(make_input(r, gen)[None].to(device)))) for r in test]
            thr = best_threshold(y_train, s_tr)
            runs.append({"auc": auc(y_test, s_te), "accuracy": float(np.mean([(v > thr) == l for v, l in zip(s_te, y_test)])),
                         "train_auc": auc(y_train, s_tr), "threshold": thr})
            nets.append(net)
            print(f"{name} seed {s}: test AUC {runs[-1]['auc']:.3f}, accuracy {runs[-1]['accuracy']:.2f} (train AUC {runs[-1]['train_auc']:.3f})", flush=True)
        out = {k: float(np.mean([r[k] for r in runs])) for k in ("auc", "accuracy", "train_auc")}
        out["auc_range"] = [min(r["auc"] for r in runs), max(r["auc"] for r in runs)]
        out["runs"] = runs
        return out, nets

    arms = args.arms.split(",")
    if "A" in arms or "B" in arms:
        tse = bp.tse_2d(NX, FOV, etl=16, esp=12.5e-3, tr=6.0, n_dummy=1).to(torch.float32)  # ky = 0 at echo 9: TE_eff ~113 ms
        k_tse = cached("tse", lambda ph: bp.cartesian_kspace(simulate_metal(ph, tse, coils_for(ph)).signal, tse).cpu())  # (C, ky, kx)
        scan_tse = float(tse.dt.sum()) * lines / NX  # the kept lines' share of the shots
        if "A" in arms:
            maps = coil_maps()
            results["A_tse_sense_image"], _ = fit("A", lambda r, g: sense_input(add_noise(k_tse[r["file"]], sigma[r["file"]], g), maps, mask))
            results["A_tse_sense_image"]["scan_s"] = scan_tse
            print("A", {k: v for k, v in results["A_tse_sense_image"].items() if k != "runs"}, flush=True)
        if "B" in arms:
            ky_tse = cols.float() - NX / 2
            results["B_tse_" + args.front], _ = fit("B", lambda r, g: front_input(args.front, add_noise(k_tse[r["file"]][:, cols, :], sigma[r["file"]], g), ky_tse))
            results["B_tse_" + args.front]["scan_s"] = scan_tse
            print("B", {k: v for k, v in results["B_tse_" + args.front].items() if k != "runs"}, flush=True)
    if "G" in arms or "C" in arms:
        trials = []
        for start in args.starts.split(","):
            flip, te_ms, tr_ms = (float(v) for v in start.split(":"))
            d = Design(lines, init_flip=flip, init_te=te_ms * 1e-3, init_tr=tr_ms * 1e-3,
                       ky_init=None if args.sampling == "uniform" else cols.float() - NX / 2)
            with torch.no_grad():
                flips0, ky0, te0, tr0 = d()
                gre0 = learnable_gre(flips0, ky0, te0, tr0)
            k0 = cached(f"gre{lines}_{flip:g}_{te_ms:g}_{tr_ms:g}" + ("" if args.sampling == "uniform" else f"_{args.sampling}{args.seed}"),
                        lambda ph, seq=gre0: simulate_metal(ph, seq, coils_for(ph)).signal.reshape(N_COILS, lines, NX).cpu())
            r, nets = fit(f"G[{flip:g} deg, TE {te_ms:g}, TR {tr_ms:g} ms]",
                          lambda rec, g, k0=k0, ky0=ky0: front_input(args.front, add_noise(k0[rec["file"]], sigma[rec["file"]], g), ky0), seeds=1)
            trials.append({"start": [flip, te_ms, tr_ms], "auc": r["auc"], "train_auc": r["train_auc"], "scan_s": float(gre0.dt.sum()),
                           "design": d, "k": k0, "ky": ky0, "net": nets[0]})
        best = max(trials, key=lambda t: t["train_auc"])  # chosen on the training set, never on the test set
        results["G_starts"] = [{k: t[k] for k in ("start", "auc", "train_auc", "scan_s")} for t in trials]
        design, k_gre, ky0 = best["design"], best["k"], best["ky"]
        print(f"G starts (test AUC / train AUC): " + ", ".join(f"{t['start']}: {t['auc']:.3f} / {t['train_auc']:.3f}" for t in trials)
              + f"; best on the training set {best['start']}", flush=True)
        if args.seeds > 1:
            results["G_gre_" + args.front], nets_g = fit("G", lambda r, g: front_input(args.front, add_noise(k_gre[r["file"]], sigma[r["file"]], g), ky0))
        else:
            results["G_gre_" + args.front] = {"auc": best["auc"], "train_auc": best["train_auc"], "accuracy": float("nan"), "runs": []}
            nets_g = [best["net"]]
        results["G_gre_" + args.front].update({"scan_s": best["scan_s"], "design": design.describe(), "start": best["start"]})
        print("G", {k: v for k, v in results["G_gre_" + args.front].items() if k not in ("runs", "design")}, flush=True)
    if "C" in arms:
        gen = torch.Generator().manual_seed(args.seed + 100)
        net = copy.deepcopy(nets_g[0])
        net.eval()  # the joint phase fine-tunes G's net: batch-norm statistics and dropout fixed, gradients still flow
        opt_design = torch.optim.Adam([{"params": [design.flip_raw, design.te_raw, design.tr_raw], "lr": args.design_lr},
                                       {"params": [design.ky_raw], "lr": args.ky_lr}])
        opt_net = torch.optim.Adam(net.parameters(), lr=args.net_lr, weight_decay=1e-4)
        buffer = collections.deque(maxlen=32 * args.batch)
        t0 = time.perf_counter()
        recent = collections.deque(maxlen=20)
        for step in range(args.steps):
            flips, ky, te, tr = design()
            seq = learnable_gre(flips, ky, te, tr)
            idx = torch.randint(len(train), (args.batch,), generator=gen).tolist()
            xb, yb = [], []
            for j in idx:
                rec = train[j]
                ph = load_ph(rec)
                sig = add_noise(simulate_metal(ph, seq, coils_for(ph)).signal.reshape(N_COILS, lines, NX), sigma[rec["file"]], gen)
                xb.append(front_input(args.front, sig, ky)); yb.append(y_train[j])
            loss = nn.functional.binary_cross_entropy_with_logits(net(torch.stack(xb).to(device)), torch.tensor(yb, dtype=torch.float32, device=device))
            opt_design.zero_grad(); opt_net.zero_grad(); loss.backward(); opt_design.step(); opt_net.step()
            recent.append(loss.item())
            buffer.extend((x.detach().cpu(), y) for x, y in zip(xb, yb))
            for _ in range(args.replay):  # the network alone, on recent acquisitions (a slowly moving design)
                pick = torch.randint(len(buffer), (16,), generator=gen).tolist()
                xr = torch.stack([augment(buffer[p][0], gen) for p in pick]).to(device)
                yr = torch.tensor([buffer[p][1] for p in pick], dtype=torch.float32, device=device)
                l_r = nn.functional.binary_cross_entropy_with_logits(net(xr), yr)
                opt_net.zero_grad(); l_r.backward(); opt_net.step()
            if step % 20 == 0 or step == args.steps - 1:
                print(f"C step {step}: loss {np.mean(recent):.3f} over the last {len(recent)} batches (replay {l_r.item():.3f}), {design.summary()}, "
                      f"{time.perf_counter() - t0:.0f} s", flush=True)
        with torch.no_grad():
            flips, ky, te, tr = design()
            seq = learnable_gre(flips, ky, te, tr)

            def score(records):
                out = []
                for rec in records:
                    ph = load_ph(rec)
                    sig = add_noise(simulate_metal(ph, seq, coils_for(ph)).signal.reshape(N_COILS, lines, NX), sigma[rec["file"]], gen)
                    out.append(float(torch.sigmoid(net(front_input(args.front, sig, ky)[None].to(device)))))
                return out
            s_tr = score(train); s_te = score(test)
        thr = best_threshold(y_train, s_tr)
        results["C_learned_" + args.front] = {"auc": auc(y_test, s_te), "accuracy": float(np.mean([(v > thr) == l for v, l in zip(s_te, y_test)])),
                                              "train_auc": auc(y_train, s_tr), "scan_s": float(seq.dt.sum()), "design": design.describe()}
        torch.save({"design": design.state_dict(), "net": net.state_dict(), "describe": design.describe()}, args.out / "arm_C.pt")
        print("C", {k: v for k, v in results["C_learned_" + args.front].items() if k != "design"}, design.summary(), flush=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
