"""Sim-to-real transfer: the number M7 is judged by. A small U-Net trained on generated samples only (generate.py)
reconstructs 4x-undersampled multi-coil k-space to the fully sampled RSS image; the same net is scored on real fastMRI
k-space it has never seen. A net trained on real scans (person-wise held out) gives the ceiling, zero-filling the floor.

    uv run python validation/realism/transfer.py --sim ~/data/gen_t2w --fastmri ~/data/fastmri_more --out ~/data/transfer_1

Task: Cartesian undersampling along the phase axis, factor ``--accel`` (4), the central ``--centre`` fraction (8 %)
fully sampled, the rest random (one mask per image, seeded). Input: the zero-filled RSS image of the undersampled
k-space; target: the RSS of the full k-space; both normalised by the target's p99 (as the realism score does). Metrics
on the held-out real slices: NMSE and SSIM against the fully sampled RSS, for (a) zero-filling, (b) the net trained on
synthetic data, (c) the net trained on real data (other people), (d) synthetic pre-training then real fine-tuning.
Everything at the realism resolution (128 x 128 on 220 mm) where the real k-space is already cached (score.real_set);
the simulated set's noisy k-space is ``/truth/kspace``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realism as R  # noqa: E402
import score as SC  # noqa: E402


def undersample(k: np.ndarray, accel: int, centre: float, rng) -> np.ndarray:
    """(C, n, n) k-space with phase-encoding columns (last axis) kept at random at 1 / accel, the central fraction always."""
    n = k.shape[-1]
    keep = np.zeros(n, bool)
    c = int(round(centre * n))
    keep[n // 2 - c // 2:n // 2 - c // 2 + c] = True
    rest = np.flatnonzero(~keep)
    m = max(n // accel - c, 0)
    keep[rng.choice(rest, size=m, replace=False)] = True
    return k * keep[None, None, :]


def pair(k: np.ndarray, accel: int, centre: float, rng, orient) -> tuple[np.ndarray, np.ndarray]:
    """(input, target) images from full k-space: zero-filled RSS of the undersampled k-space and the full RSS, both
    over the target's p99, in the display orientation (``orient`` applied to both)."""
    full = orient(R.recon(k))
    scale = np.percentile(full, 99)
    zero = orient(R.recon(undersample(k, accel, centre, rng)))
    return (zero / scale).astype(np.float32), (full / scale).astype(np.float32)


class UNet(nn.Module):
    def __init__(self, ch=(16, 32, 64, 128)):
        super().__init__()
        block = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                           nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
        self.down = nn.ModuleList([block(1, ch[0])] + [block(ch[i], ch[i + 1]) for i in range(len(ch) - 1)])
        self.up = nn.ModuleList([block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(len(ch) - 1))])
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = nn.functional.max_pool2d(x, 2)
        for u in self.up:
            x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = u(torch.cat([x, skips.pop()], 1))
        return self.out(x) + 0  # residual-free: the input's aliasing is not a good prior to keep


def train(net, X, Y, epochs, device, lr=1e-3, seed=0, batch=16):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt, Yt = torch.tensor(X[:, None]), torch.tensor(Y[:, None])
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb, yb = Xt[idx].to(device), Yt[idx].to(device)
            if torch.rand(1, generator=gen) < .5:
                xb, yb = xb.flip(-1), yb.flip(-1)
            loss = nn.functional.l1_loss(net(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def ssim(a, b, k1=.01, k2=.03, win=7):
    """SSIM of two (n, n) images (data range 1), Gaussian-free uniform window, as a plain mean."""
    from scipy import ndimage
    mu_a, mu_b = ndimage.uniform_filter(a, win), ndimage.uniform_filter(b, win)
    s_aa = ndimage.uniform_filter(a * a, win) - mu_a ** 2
    s_bb = ndimage.uniform_filter(b * b, win) - mu_b ** 2
    s_ab = ndimage.uniform_filter(a * b, win) - mu_a * mu_b
    c1, c2 = k1 ** 2, k2 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return float(m.mean())


def evaluate(net, X, Y, device):
    net.eval()
    with torch.no_grad():
        P = net(torch.tensor(X[:, None]).to(device)).cpu().numpy()[:, 0]
    nmse = float(np.mean([np.sum((p - y) ** 2) / np.sum(y ** 2) for p, y in zip(P, Y)]))
    return {"nmse": nmse, "ssim": float(np.mean([ssim(np.clip(p, 0, 2), y) for p, y in zip(P, Y)]))}, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=Path, required=True, help="generate.py output directory")
    ap.add_argument("--fastmri", type=Path, default=R.FASTMRI)
    ap.add_argument("--cache", type=Path, default=Path.home() / "data" / "realism_cache")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--accel", type=int, default=4)
    ap.add_argument("--centre", type=float, default=0.08)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--finetune-epochs", type=int, default=10)
    ap.add_argument("--slices", default="1-7")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import h5py
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    lo, _, hi = args.slices.partition("-")
    slices = range(int(lo), int(hi or lo) + 1)
    # real k-space (full), per scan, through the realism reconstruction: cached per file by score.real_set's recipe
    import ismrmrd  # noqa: F401  (realism.fastmri_volume needs it)
    real = []
    for f in R.manifest(root=args.fastmri):
        with h5py.File(f, "r") as h:
            k = h["kspace"][()]
            hdr = ismrmrd.xsd.CreateFromDocument(h["ismrmrd_header"][()])
        enc = hdr.encoding[0]
        rx, ry = enc.reconSpace.matrixSize.x, enc.reconSpace.matrixSize.y
        imgs = R.crop_centre(R.centred_ifft(k, (-2, -1)), (rx, ry))
        for s in slices:
            if s < k.shape[0]:
                real.append((f.stem, s, R.lowres_kspace(imgs[s], 128)))
    people = sorted({p for p, _, _ in real})
    rng.shuffle(people)
    test_people = set(people[: max(len(people) // 3, 1)])
    orient = lambda img: img  # the real reconstruction is already in display orientation
    Xr_tr, Yr_tr, Xr_te, Yr_te = [], [], [], []
    for p, s, k in real:
        x, y = pair(k, args.accel, args.centre, rng, orient)
        (Xr_te if p in test_people else Xr_tr)[0:0] = []
        (Xr_te if p in test_people else Xr_tr).append(x); (Yr_te if p in test_people else Yr_tr).append(y)
    Xr_tr, Yr_tr, Xr_te, Yr_te = (np.stack(v) for v in (Xr_tr, Yr_tr, Xr_te, Yr_te))
    # simulated: /truth/kspace (noisy, display orientation after recon as score.sim_set does)
    import simulate_heads as SH
    Xs, Ys = [], []
    for f in sorted(args.sim.glob("sample_*.h5")):
        with h5py.File(f, "r") as h:
            k = h["truth"]["kspace"][()].astype(np.complex128)
        x, y = pair(k, args.accel, args.centre, rng, SH.to_real_orientation)
        Xs.append(x); Ys.append(y)
    Xs, Ys = np.stack(Xs), np.stack(Ys)
    print(f"real: {len(Xr_tr)} train / {len(Xr_te)} test slices ({len(people) - len(test_people)} / {len(test_people)} people); "
          f"simulated: {len(Xs)} slices", flush=True)
    results = {"n": {"real_train": len(Xr_tr), "real_test": len(Xr_te), "sim": len(Xs)}, "accel": args.accel}
    results["zero_fill"], _ = evaluate(nn.Identity(), Xr_te, Yr_te, device)
    torch.manual_seed(args.seed)
    net_s = train(UNet().to(device), Xs, Ys, args.epochs, device, seed=args.seed)
    results["synthetic"], P_s = evaluate(net_s, Xr_te, Yr_te, device)
    torch.manual_seed(args.seed)
    net_r = train(UNet().to(device), Xr_tr, Yr_tr, args.epochs, device, seed=args.seed)
    results["real"], P_r = evaluate(net_r, Xr_te, Yr_te, device)
    net_f = train(net_s, Xr_tr, Yr_tr, args.finetune_epochs, device, lr=3e-4, seed=args.seed)
    results["synthetic_then_real"], P_f = evaluate(net_f, Xr_te, Yr_te, device)
    results["synthetic_on_synthetic"], _ = evaluate(net_s, Xs[-max(len(Xs) // 10, 1):], Ys[-max(len(Ys) // 10, 1):], device)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    np.savez_compressed(args.out / "examples.npz", X=Xr_te[:8], Y=Yr_te[:8], synthetic=P_s[:8], real=P_r[:8], finetuned=P_f[:8])
    for k, v in results.items():
        if isinstance(v, dict) and "nmse" in v:
            print(f"{k:22s} NMSE {v['nmse']:.4f}  SSIM {v['ssim']:.3f}")


if __name__ == "__main__":
    main()
