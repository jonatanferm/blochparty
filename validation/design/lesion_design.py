"""Sequence design by classification (ROADMAP M7, agentic design, exploratory): can a sequence optimised jointly with a
network that reads its raw samples detect white-matter lesions better, or faster, than a standard sequence read through
an image?

    uv run python validation/design/lesion_design.py --slabs ~/data/slabs --out ~/data/design_1 [--lines 16] [--steps 300]

Task: "is there a lesion in this slab?" on the slabs of lesion_slabs.py (64 x 64 over 220 mm, a 5 mm slab, lesions of
6-15 mm at the imaged levels; pairs of the same head with and without lesions; split by subject).
Every arm acquires at most a fraction ``--fraction`` of k-space (K = fraction x 64 lines) through the same 8-element
receive ring, with complex Gaussian noise at a fixed SNR on every acquisition:
  A  the standard T2w TSE undersampled uniformly to K lines (the centre lines kept) -> CG-SENSE with the exact coil
     sensitivities (an upper bound for any ESPIRiT-style calibration) -> a small CNN on the image: the clinical baseline;
  B  the same undersampled TSE, the CNN on the raw multi-coil lines (real and imaginary channels): the raw input alone;
  C  a spoiled GRE of K lines whose per-line flip angles, TE, TR and k-space positions are learned together with a
     network on its raw multi-coil samples, by gradient descent through the Metal adjoint (``simulate_metal``).
Reports test AUC and accuracy per arm, the scan time, and arm C's learned sequence; run it at 0.2, 0.1, 0.05 to see
where each arm fails. Every acquisition is simulated (no reconstruction in C); the phantoms are the cached slabs.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import blochparty as bp
from blochparty.metal import simulate_metal
from blochparty.sequence import SequenceBuilder

FOV, NX, DWELL = 0.22, 64, 10e-6
RF_DURATION = 0.2e-3


def learnable_gre(flips_deg: torch.Tensor, ky_cycles: torch.Tensor, te: torch.Tensor, tr: torch.Tensor, *, nx=NX, fov=FOV,
                  dwell=DWELL, n_dummy: int = 8, dtype=torch.float32) -> bp.Sequence:
    """A spoiled GRE of K = len(flips) lines: line j excites with ``flips_deg[j]`` and encodes ``ky_cycles[j]`` cycles
    across the field of view (continuous: not restricted to the Cartesian grid), all with echo time ``te`` and
    repetition time ``tr`` (tensors; gradients flow to all four through the builder and the adjoint). ``n_dummy``
    excitations with the mean flip bring the magnetisation toward its steady state first. Ideal gradients and spoiling."""
    dk = 1.0 / fov
    t_read = nx * dwell
    t_echo = (nx / 2 + 0.5) * dwell
    t_pre = te - RF_DURATION / 2 - t_echo
    t_fill = tr - RF_DURATION - t_pre - t_read
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
    """Arm C's parameters: flips in (2, 90) deg by a sigmoid, TE in (te_lo, te_hi), TR in (tr_lo, tr_hi), ky in
    (-nx/2, nx/2) cycles, all unconstrained underneath."""

    def __init__(self, lines: int, nx=NX, te=(2e-3, 30e-3), tr=(8e-3, 120e-3), init_flip=25., init_te=6e-3, init_tr=15e-3):
        super().__init__()
        self.lines, self.nx, self.te_rng, self.tr_rng = lines, nx, te, tr
        self.flip_raw = nn.Parameter(torch.full((lines,), math.log((init_flip - 2) / (90 - init_flip))))
        self.te_raw = nn.Parameter(torch.tensor(math.log((init_te - te[0]) / (te[1] - init_te))))
        self.tr_raw = nn.Parameter(torch.tensor(math.log((init_tr - tr[0]) / (tr[1] - init_tr))))
        start = torch.linspace(-nx / 2 + nx / lines / 2, nx / 2 - nx / lines / 2, lines)  # evenly spread lines
        self.ky_raw = nn.Parameter(torch.atanh((start / (nx / 2)).clamp(-.999, .999)))

    def forward(self):
        flips = 2 + 88 * torch.sigmoid(self.flip_raw)
        te = self.te_rng[0] + (self.te_rng[1] - self.te_rng[0]) * torch.sigmoid(self.te_raw)
        tr = self.tr_rng[0] + (self.tr_rng[1] - self.tr_rng[0]) * torch.sigmoid(self.tr_raw)
        ky = (self.nx / 2) * torch.tanh(self.ky_raw)
        return flips, ky, te, tr

    def describe(self):
        flips, ky, te, tr = (v.detach() for v in self())
        return {"flips_deg": flips.round(decimals=1).tolist(), "ky_cycles": ky.round(decimals=2).tolist(), "te_ms": round(te.item() * 1e3, 2),
                "tr_ms": round(tr.item() * 1e3, 2), "scan_ms": round(tr.item() * 1e3 * (self.lines + 8), 1)}


N_COILS, COIL_RADIUS = 8, 0.14


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centred orthonormal 2D FFT over the last two axes (``ifft2c_o`` is its exact inverse and adjoint)."""
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ifft2c_o(k: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def coils_for(ph) -> torch.Tensor:
    """(N, 8) sensitivities of an 8-element ring (radius 0.14 m) at the spins, complex64 (the same in every arm)."""
    return bp.birdcage_sensitivities(ph.pos, N_COILS, radius=COIL_RADIUS).to(torch.complex64)


def coil_maps(ph, n=NX, fov=FOV) -> torch.Tensor:
    """(8, n, n) the ring's sensitivities on the image grid (the exact maps SENSE is given), display orientation of
    ``image_input``: the same ifft2c convention, so built on the grid the k-space maps to."""
    u = (torch.arange(n, dtype=torch.float32) - n / 2 + .5) * (fov / n)
    yy, xx = torch.meshgrid(u, u, indexing="ij")
    pos = torch.stack([xx.ravel(), yy.ravel(), torch.zeros(n * n)], 1)
    return bp.birdcage_sensitivities(pos, N_COILS, radius=COIL_RADIUS).to(torch.complex64).T.reshape(N_COILS, n, n)


def uniform_mask(n: int, lines: int, centre: int = 4) -> torch.Tensor:
    """(n,) bool: ``lines`` phase-encode lines, the central ``centre`` always, the rest evenly spaced (SENSE-style)."""
    keep = torch.zeros(n, dtype=torch.bool)
    keep[n // 2 - centre // 2: n // 2 - centre // 2 + centre] = True
    rest = max(lines - centre, 0)
    if rest:
        cand = torch.nonzero(~keep).squeeze(1)
        pick = torch.linspace(0, len(cand) - 1, rest).round().long()
        keep[cand[pick]] = True
    return keep


def cg_sense(k: torch.Tensor, maps: torch.Tensor, mask: torch.Tensor, iters: int = 20, lam: float = 1e-3) -> torch.Tensor:
    """Least-squares SENSE by conjugate gradient: x = argmin ||M F S x - k||^2 + lam ||x||^2 for (C, n, n) k-space with
    the phase-encode columns of ``mask`` sampled, (C, n, n) coil maps; returns the (n, n) complex image."""
    def A(x):  # image -> masked multi-coil k-space
        return fft2c(maps * x[None]) * mask[None, None, :]

    def AH(y):  # the adjoint of A (orthonormal transforms on both sides)
        return (maps.conj() * ifft2c_o(y * mask[None, None, :])).sum(0)

    b = AH(k)
    x = torch.zeros_like(b)
    r = b.clone(); p = r.clone(); rs = (r.conj() * r).sum().real
    for _ in range(iters):
        Ap = AH(A(p)) + lam * p
        alpha = rs / (p.conj() * Ap).sum().real
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r.conj() * r).sum().real
        if rs_new < 1e-12:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def acquire(ph, seq, snr: float, gen: torch.Generator | None):
    """The raw samples of ``seq`` on ``ph`` with complex Gaussian noise: sigma = (mean |signal| of a noiseless
    reference acquisition, held by the caller as ``snr``'s unit) / snr. Returns (C, n_adc) complex."""
    out = simulate_metal(ph, seq, coils_for(ph))
    sig = out.signal
    if snr is not None:
        sigma = snr_unit(ph) / snr
        noise = torch.randn(sig.shape, generator=gen, dtype=torch.float32) + 1j * torch.randn(sig.shape, generator=gen, dtype=torch.float32)
        sig = sig + sigma * noise.to(sig.dtype)
    return sig


_UNIT = {}


def snr_unit(ph) -> float:
    """One noise unit per phantom: the mean |sample| of a 64-line, 25 deg, TE 6 / TR 15 ms GRE on it (cached), so an
    SNR is comparable across arms and phantoms."""
    key = id(ph)
    if key not in _UNIT:
        with torch.no_grad():
            seq = bp.gre_2d(NX, FOV, flip_deg=25., tr=15e-3, te=6e-3, dwell=DWELL, n_dummy=8)
            _UNIT[key] = float(simulate_metal(ph, seq.to(torch.float32), coils_for(ph)).signal.abs().mean())
    return _UNIT[key]


class RawNet(nn.Module):
    """A small CNN on raw samples (2 channels: real and imaginary, scaled per sample) or on an image (1 channel)."""

    def __init__(self, channels: int, shape: tuple[int, int]):
        super().__init__()
        c = [channels, 16, 32, 64]
        layers = []
        h, w = shape
        for i in range(3):  # pool each axis only while it has room (a few k-space lines are a short axis)
            pool = (2 if h >= 2 else 1, 2 if w >= 2 else 1)
            h, w = h // pool[0], w // pool[1]
            layers += [nn.Conv2d(c[i], c[i + 1], 3, padding=1), nn.BatchNorm2d(c[i + 1]), nn.ReLU(), nn.MaxPool2d(pool)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(c[-1], 1))

    def forward(self, x):
        return self.head(self.features(x).mean((-2, -1))).squeeze(-1)


def raw_input(k, lines=None, nx=NX):
    """(C, lines, nx) complex (or (C, n_adc) to be reshaped) -> (2 C, lines, nx) float, scaled by the mean magnitude."""
    if k.ndim == 2:
        k = k.reshape(k.shape[0], lines, nx)
    k = k / (k.abs().mean() + 1e-12)
    return torch.cat([k.real, k.imag], 0).float()


def sense_input(k, maps, mask):
    """Undersampled (C, n, n) k-space -> the CG-SENSE magnitude image, (1, n, n) float scaled by its mean."""
    img = cg_sense(k, maps, mask).abs()
    return (img / (img.mean() + 1e-12))[None].float()


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def load_slabs(folder: Path, device):
    index = json.loads((folder / "index.json").read_text())
    data = {"train": [], "test": []}
    for rec in index:
        d = torch.load(folder / rec["file"], weights_only=False)
        ph = d["phantom"]
        data[rec["split"]].append((ph, int(rec["label"]), rec))
    if not data["test"]:  # a functional test on a handful of slabs: score on the training slabs and say so
        print("no test split: scoring on the training slabs (a functional test, not a result)", flush=True)
        data["test"] = data["train"]
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slabs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fraction", type=float, default=0.2, help="the share of k-space every arm may acquire (K = fraction x 64 lines)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--snr", type=float, default=20.)
    ap.add_argument("--epochs-baseline", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="A,B,C")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data = load_slabs(args.slabs, device)
    print(f"slabs: {len(data['train'])} train / {len(data['test'])} test; "
          f"positives {sum(l for _, l, _ in data['train'])} / {sum(l for _, l, _ in data['test'])}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    lines = max(int(round(args.fraction * NX)), 2)
    args.lines = lines
    mask = uniform_mask(NX, lines)
    results = {"n": {k: len(v) for k, v in data.items()}, "snr": args.snr, "fraction": args.fraction, "lines": lines, "coils": N_COILS}
    print(f"fraction {args.fraction}: {lines} of {NX} lines; TSE mask keeps columns {torch.nonzero(mask).squeeze(1).tolist()}", flush=True)

    def train_fixed(name, seq, featurise, channels, shape):
        """Arms A and B: a fixed sequence, acquisitions precomputed once per slab (noise redrawn per epoch would cost a
        simulation each; here one noise draw per slab per epoch from a cached noiseless signal)."""
        t0 = time.perf_counter()
        clean = {}
        for split in ("train", "test"):
            for ph, _, rec in data[split]:
                with torch.no_grad():
                    clean[rec["file"]] = bp.cartesian_kspace(simulate_metal(ph, seq, coils_for(ph)).signal, seq) * mask[None, None, :]
        print(f"{name}: {len(clean)} acquisitions in {time.perf_counter() - t0:.0f} s", flush=True)
        net = RawNet(channels, shape).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        train = data["train"]
        for ep in range(args.epochs_baseline):
            net.train()
            perm = torch.randperm(len(train), generator=gen)
            for i in range(0, len(perm), args.batch):
                xb, yb = [], []
                for j in perm[i:i + args.batch]:
                    ph, lab, rec = train[j]
                    sig = clean[rec["file"]]
                    sigma = snr_unit(ph) / args.snr
                    sig = sig + sigma * (torch.randn(sig.shape, generator=gen) + 1j * torch.randn(sig.shape, generator=gen)).to(sig.dtype)
                    xb.append(featurise(sig)); yb.append(lab)
                loss = nn.functional.binary_cross_entropy_with_logits(net(torch.stack(xb).to(device)), torch.tensor(yb, dtype=torch.float32, device=device))
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        scores, labels = [], []
        with torch.no_grad():
            for ph, lab, rec in data["test"]:
                sig = clean[rec["file"]]
                sigma = snr_unit(ph) / args.snr
                sig = sig + sigma * (torch.randn(sig.shape, generator=gen) + 1j * torch.randn(sig.shape, generator=gen)).to(sig.dtype)
                scores.append(float(torch.sigmoid(net(featurise(sig)[None].to(device))))); labels.append(lab)
        acc = float(np.mean([(s > .5) == l for s, l in zip(scores, labels)]))
        return {"auc": auc(labels, scores), "accuracy": acc, "scan_s": float(seq.dt.sum()) * lines / NX}  # the kept lines' share of the shots

    arms = args.arms.split(",")
    if "A" in arms or "B" in arms:
        tse = bp.tse_2d(NX, FOV, etl=16, esp=12.5e-3, tr=6.0, n_dummy=1).to(torch.float32)  # ky = 0 at echo 9: TE_eff ~113 ms
        maps = coil_maps(None)
        if "A" in arms:
            results["A_tse_sense_image"] = train_fixed("A", tse, lambda k: sense_input(k, maps, mask), 1, (NX, NX))
            print("A", results["A_tse_sense_image"], flush=True)
        if "B" in arms:
            cols = torch.nonzero(mask).squeeze(1)
            results["B_tse_raw"] = train_fixed("B", tse, lambda k: raw_input(k[:, :, cols].transpose(1, 2)), 2 * N_COILS, (lines, NX))
            print("B", results["B_tse_raw"], flush=True)
    if "C" in arms:
        design = Design(args.lines).to("cpu")
        net = RawNet(2 * N_COILS, (args.lines, NX)).to(device)
        opt = torch.optim.Adam([{"params": design.parameters(), "lr": 3e-2}, {"params": net.parameters(), "lr": 1e-3}])
        train = data["train"]
        t0 = time.perf_counter()
        for step in range(args.steps):
            net.train()
            idx = torch.randint(len(train), (args.batch,), generator=gen)
            flips, ky, te, tr = design()
            seq = learnable_gre(flips, ky, te, tr)
            xb, yb = [], []
            for j in idx:
                ph, lab, rec = train[j]
                sig = acquire(ph, seq, args.snr, gen)
                xb.append(raw_input(sig, args.lines)); yb.append(lab)
            loss = nn.functional.binary_cross_entropy_with_logits(net(torch.stack(xb).to(device)), torch.tensor(yb, dtype=torch.float32, device=device))
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 20 == 0 or step == args.steps - 1:
                print(f"C step {step}: loss {loss.item():.3f}, {design.describe()}, {time.perf_counter() - t0:.0f} s", flush=True)
        net.eval()
        with torch.no_grad():
            flips, ky, te, tr = design()
            seq = learnable_gre(flips, ky, te, tr)
            scores, labels = [], []
            for ph, lab, rec in data["test"]:
                sig = acquire(ph, seq, args.snr, gen)
                scores.append(float(torch.sigmoid(net(raw_input(sig, args.lines)[None].to(device))))); labels.append(lab)
        results["C_learned_raw"] = {"auc": auc(labels, scores), "accuracy": float(np.mean([(s > .5) == l for s, l in zip(scores, labels)])),
                                    "scan_s": float(seq.dt.sum()), "design": design.describe()}
        torch.save({"design": design.state_dict(), "net": net.state_dict()}, args.out / "arm_C.pt")
        print("C", results["C_learned_raw"], flush=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
