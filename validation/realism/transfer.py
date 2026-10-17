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
the simulated set's noisy k-space is ``/truth/kspace``, transposed into the display orientation before the mask so the
phase axis is the same image axis on both sides.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import blochparty as bp

import realism as R  # noqa: E402


def undersample(k: np.ndarray, accel: int, centre: float, rng) -> np.ndarray:
    """(C, n, n) k-space with phase-encoding columns (last axis) kept at random at 1 / accel, the central fraction always."""
    return k * bp.recon.random_mask(k.shape[-1], accel, centre, rng)[None, None, :]


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
        x_in = x
        skips = []
        for i, d in enumerate(self.down):
            x = d(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = nn.functional.max_pool2d(x, 2)
        for u in self.up:
            x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = u(torch.cat([x, skips.pop()], 1))
        return x_in + self.out(x)  # residual: the net learns the correction to the zero-filled image


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


def load_sim(folder, accel, centre, rng):
    """(inputs, targets, lesion masks, skipped) from a generate.py set: /truth/kspace undersampled in the display
    orientation (the FFT of a transposed image is the transposed k-space, so the phase axis is the same image axis on
    both sides; 2026-10-14: masking the simulated k-space's last axis and then transposing the image trained the net
    on streaks along the other axis), /truth/lesion as stored (already the display grid)."""
    import h5py
    Xs, Ys, Ls, skipped = [], [], [], []
    for f in sorted(Path(folder).glob("sample_*.h5")):
        try:
            with h5py.File(f, "r") as h:
                k = h["truth"]["kspace"][()].astype(np.complex128)
                les = h["truth"]["lesion"][()] if "lesion" in h["truth"] else np.zeros(k.shape[-2:], np.float32)
        except Exception as e:  # a file cut short
            skipped.append((f.name, type(e).__name__)); continue
        if not np.isfinite(k).all() or np.abs(k).max() == 0:
            skipped.append((f.name, "non-finite or empty")); continue
        x, y = pair(k.transpose(0, 2, 1), accel, centre, rng, lambda i: i)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            skipped.append((f.name, "non-finite image")); continue
        Xs.append(x); Ys.append(y); Ls.append(les.astype(np.float32))
    if skipped:
        print(f"skipped {len(skipped)} simulated samples in {folder}: {skipped[:5]}", flush=True)
    return np.stack(Xs), np.stack(Ys), np.stack(Ls), len(skipped)


def lesion_scores(nets, X, Y, L, device):
    """Per net: NMSE over lesion pixels (mask > 0.5) and over the rest of the brain (target > 0.15), pooled over the
    samples that carry lesion pixels; None if none do."""
    has = [i for i in range(len(X)) if (L[i] > .5).sum() >= 4]
    if not has:
        return None
    out = {"n_samples": len(has), "lesion_px": int(sum((L[i] > .5).sum() for i in has))}
    with torch.no_grad():
        for name, net in nets.items():
            P = X[has] if isinstance(net, nn.Identity) else net(torch.tensor(X[has][:, None]).to(device)).cpu().numpy()[:, 0]
            e2 = (P - Y[has]) ** 2
            inl = L[has] > .5
            brain = (Y[has] > .15) & ~inl
            out[name] = {"lesion_nmse": float(e2[inl].sum() / (Y[has][inl] ** 2).sum()),
                         "brain_nmse": float(e2[brain].sum() / (Y[has][brain] ** 2).sum()),
                         "lesion_mean_ratio": float(P[inl].mean() / Y[has][inl].mean())}
    return out


def spectra(Y, nbands=6):
    """Mean radial power of the target images by band of Nyquist (the fine-content check: a higher zero-fill floor on
    one side means its targets carry more energy where the mask takes it away)."""
    n = Y.shape[-1]
    f = np.fft.fftshift(np.fft.fftfreq(n)) * 2
    rr = np.sqrt(f[:, None] ** 2 + f[None] ** 2)
    P = np.mean([np.abs(np.fft.fftshift(np.fft.fft2(y - y.mean()))) ** 2 for y in Y], 0)
    edges = np.linspace(0, 1, nbands + 1)
    return [float(P[(rr >= a) & (rr < b)].mean()) for a, b in zip(edges[:-1], edges[1:])]


def evaluate(net, X, Y, device):
    net.eval()
    with torch.no_grad():
        P = net(torch.tensor(X[:, None]).to(device)).cpu().numpy()[:, 0]
    nmse = float(np.mean([np.sum((p - y) ** 2) / np.sum(y ** 2) for p, y in zip(P, Y)]))
    return {"nmse": nmse, "ssim": float(np.mean([ssim(np.clip(p, 0, 2), y) for p, y in zip(P, Y)]))}, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=Path, required=True, help="generate.py output directory")
    ap.add_argument("--sim2", type=Path, default=None, help="a second synthetic set for an ablation (trained and scored as synthetic2)")
    ap.add_argument("--lesion-eval", type=Path, default=None,
                    help="an evaluation-only synthetic set with many large lesions (never trained on) for the lesion test; "
                         "without it the held-out tenth of --sim is used, which rarely holds enough lesion pixels")
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
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    lo, _, hi = args.slices.partition("-")
    slices = range(int(lo), int(hi or lo) + 1)
    # real k-space (full), per scan, through the realism reconstruction: cached per file by score.real_set's recipe
    real = []
    for f in R.manifest(root=args.fastmri):
        low, _ = R.coil_kspace(f, 128)  # (S, C, 128, 128), raw or compact
        for s in slices:
            if s < low.shape[0]:
                real.append((f.stem, s, low[s].astype(np.complex128)))
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
    # simulated: /truth/kspace (noisy) and /truth/lesion, both into the display orientation
    Xs, Ys, Ls, n_skipped = load_sim(args.sim, args.accel, args.centre, rng)
    hold = max(len(Xs) // 10, 1)  # the last tenth of each synthetic set (the highest seeds) is held out of training
    print(f"real: {len(Xr_tr)} train / {len(Xr_te)} test slices ({len(people) - len(test_people)} / {len(test_people)} people); "
          f"simulated: {len(Xs)} slices ({hold} held out)", flush=True)
    results = {"n": {"real_train": len(Xr_tr), "real_test": len(Xr_te), "sim": len(Xs), "sim_skipped": n_skipped}, "accel": args.accel}
    results["zero_fill"], _ = evaluate(nn.Identity(), Xr_te, Yr_te, device)
    sr, ss_ = spectra(Yr_te), spectra(Ys)
    results["target_power_sim_over_real"] = [round(b / a, 3) for a, b in zip(sr, ss_)]  # six bands of Nyquist
    torch.manual_seed(args.seed)
    net_s = train(UNet().to(device), Xs[:-hold], Ys[:-hold], args.epochs, device, seed=args.seed)
    results["synthetic"], P_s = evaluate(net_s, Xr_te, Yr_te, device)
    nets = {"synthetic": net_s}
    if args.sim2 is not None:  # a second synthetic set (the ablation: e.g. the same heads without lesions)
        Xs2, Ys2, _, _ = load_sim(args.sim2, args.accel, args.centre, rng)
        hold2 = max(len(Xs2) // 10, 1)
        torch.manual_seed(args.seed)
        nets["synthetic2"] = train(UNet().to(device), Xs2[:-hold2], Ys2[:-hold2], args.epochs, device, seed=args.seed)
        results["synthetic2"], _ = evaluate(nets["synthetic2"], Xr_te, Yr_te, device)
        results["n"]["sim2"] = len(Xs2)
    torch.manual_seed(args.seed)
    net_r = train(UNet().to(device), Xr_tr, Yr_tr, args.epochs, device, seed=args.seed)
    results["real"], P_r = evaluate(net_r, Xr_te, Yr_te, device)
    import copy
    net_f = train(copy.deepcopy(net_s), Xr_tr, Yr_tr, args.finetune_epochs, device, lr=3e-4, seed=args.seed)  # a copy: until
    # 2026-10-16 this fine-tuned net_s in place, so every later use of "synthetic" (its in-domain score, the lesion test,
    # the saved net) was the fine-tuned net; the synthetic-on-real number above was taken before and is unaffected
    results["synthetic_then_real"], P_f = evaluate(net_f, Xr_te, Yr_te, device)
    nets |= {"real": net_r, "synthetic_then_real": net_f, "zero_fill": nn.Identity()}
    results["synthetic_on_synthetic"], _ = evaluate(net_s, Xs[-hold:], Ys[-hold:], device)
    results["zero_fill_on_synthetic"], _ = evaluate(nn.Identity(), Xs[-hold:], Ys[-hold:], device)
    results["real_on_synthetic"], _ = evaluate(net_r, Xs[-hold:], Ys[-hold:], device)
    # the lesion test: on the held-out synthetic samples that carry lesions, each net's error inside the lesion pixels
    # against its error in the rest of the brain (does a reconstruction keep what is not in its training data?)
    if args.lesion_eval is not None:
        Xl, Yl, Ll, _ = load_sim(args.lesion_eval, args.accel, args.centre, rng)
    else:
        Xl, Yl, Ll = Xs[-hold:], Ys[-hold:], Ls[-hold:]
    lesion = lesion_scores(nets, Xl, Yl, Ll, device)
    if lesion:
        results["lesion"] = lesion
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    for name, net in nets.items():  # 2 MB each: the nets are kept so a later evaluation set needs no retraining
        if not isinstance(net, nn.Identity):
            torch.save(net.state_dict(), args.out / f"net_{name}.pt")
    np.savez_compressed(args.out / "examples.npz", X=Xr_te[:8], Y=Yr_te[:8], synthetic=P_s[:8], real=P_r[:8], finetuned=P_f[:8])
    if lesion:
        idx = [i for i in range(len(Xl)) if (Ll[i] > .5).sum() >= 4][:6]
        with torch.no_grad():
            P = {k: v(torch.tensor(Xl[idx][:, None]).to(device)).cpu().numpy()[:, 0] for k, v in nets.items() if k != "zero_fill"}
        np.savez_compressed(args.out / "lesion_examples.npz", X=Xl[idx], Y=Yl[idx], L=Ll[idx], **P)
    for k, v in results.items():
        if isinstance(v, dict) and "nmse" in v:
            print(f"{k:22s} NMSE {v['nmse']:.4f}  SSIM {v['ssim']:.3f}")
    print("target power, synthetic / real, by sixth of Nyquist:", results["target_power_sim_over_real"])
    if lesion:
        print(f"lesion test on {lesion['n_samples']} held-out synthetic samples ({lesion['lesion_px']} lesion px): "
              + "; ".join(f"{k} lesion {v['lesion_nmse']:.3f} / brain {v['brain_nmse']:.3f} (level {v['lesion_mean_ratio']:.2f})"
                          for k, v in lesion.items() if isinstance(v, dict)))


if __name__ == "__main__":
    main()
