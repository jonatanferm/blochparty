"""Realism score v0 (ROADMAP M5, P2): can a classifier tell real fastMRI 3 T T2w slices from
simulated ones?

Both sets pass through ``realism.recon`` (central 128 x 128 of k-space on a 220 mm field of view,
root-sum-of-squares) and ``realism.normalise``. Simulated k-space (``simulate_heads.py``) gets
complex Gaussian noise whose level puts the background (image corners) at the same fraction of the
99th percentile as in the real scan it is paired with. Slices 0-12 of each stack are used
(``--slices``); real and simulated sets always use the same ones.

The score is the cross-validated AUC of a small CNN on magnitude images, with folds split by
person (a real volume, or all heads drawn from one BrainWeb subject, is never in training and test at
once) and balanced by class, pooled over all
held-out slices, with a 95 % bootstrap interval over people. AUC 0.5: indistinguishable; 1.0:
always told apart. Checks that the test can say 0.5: the same procedure on real-vs-real (two
random halves of the real volumes) and on sim-vs-sim. A baseline without a CNN (logistic
regression on 32 intensity quantiles) shows how much the plain intensity histogram gives away.

    uv run python validation/realism/score.py --sim ~/data/realism_sim --out outputs/realism
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
import simulate_heads as SH  # noqa: E402

SLICES = range(13)
CORNER = 12


def background(img):
    """Median of the four 12 x 12 corners (outside the head) of a normalised image."""
    c = CORNER
    return float(np.median(np.concatenate([img[:c, :c].ravel(), img[:c, -c:].ravel(), img[-c:, :c].ravel(),
                                           img[-c:, -c:].ravel()])))


def real_set(cache: Path, root: Path = R.FASTMRI, slices=SLICES):
    """[(person, slice, image)], {file: noise floor} from the real scans (cached as .npy)."""
    cache.mkdir(parents=True, exist_ok=True)
    items, floor = [], {}
    for path in R.manifest(root=root):
        f = cache / f"{path.stem}.npy"
        if not f.exists():
            vol, _ = R.fastmri_volume(path)
            np.save(f, vol.astype(np.float32))
        vol = np.load(f).astype(np.float64)
        imgs = [R.normalise(v) for v in vol]
        floor[path.name] = float(np.median([background(i) for i in imgs]))
        items += [(f"real:{path.stem}", s, imgs[s]) for s in slices]
    return items, floor


def unit_floor(coils, n=128, rng=None):
    """Background median of the RSS image of unit-sigma complex noise in ``coils`` channels."""
    rng = rng or np.random.default_rng(0)
    k = (rng.standard_normal((coils, n, n)) + 1j * rng.standard_normal((coils, n, n)))
    return float(np.median(R.recon(k)))


def sim_set(sim_dir: Path, floor: dict, seed=0, slices=SLICES):
    rng = np.random.default_rng(seed)
    items, units = [], {}
    for f in sorted(sim_dir.glob("sim_*.npz")):
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        target = floor[meta["protocol"]["file"]]
        for s in slices:
            key = f"slice{s:02d}"
            if key not in d:
                continue
            k = d[key].astype(np.complex128)
            c = k.shape[0]
            units.setdefault(c, unit_floor(c))
            clean = R.recon(k)
            sigma = target * np.percentile(clean, 99) / units[c]
            noisy = k + sigma * (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape))
            img = R.normalise(SH.to_real_orientation(R.recon(noisy)))
            items.append((f"sim:subject{meta['subject']:02d}", s, img))  # heads of one subject: one person
    return items


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        layers, ch = [], 1
        for out in (16, 32, 64, 64):
            layers += [nn.Conv2d(ch, out, 3, padding=1), nn.BatchNorm2d(out), nn.ReLU(), nn.MaxPool2d(2)]
            ch = out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(ch, 1))

    def forward(self, x):
        return self.head(self.features(x).mean((-2, -1))).squeeze(-1)


def augment(x, gen):
    """Random shifts of up to 4 pixels, left-right flips and +-10 % intensity: nothing either set
    could be identified by."""
    b = x.shape[0]
    shift = torch.randint(-4, 5, (b, 2), generator=gen)
    out = torch.stack([torch.roll(xi, (int(s[0]), int(s[1])), (-2, -1)) for xi, s in zip(x, shift)])
    flip = torch.rand(b, generator=gen) < 0.5
    out[flip] = out[flip].flip(-1)
    return out * (1 + 0.1 * (2 * torch.rand(b, 1, 1, 1, generator=gen) - 1))


def person_folds(groups, y, folds=5, seed=0):
    """Fold of every row: each class's people shuffled and dealt round-robin, so every fold holds
    out the same share of each class (unbalanced held-out people bias a classifier against them)."""
    rng = np.random.default_rng(seed)
    fold_of = {}
    for c in (0, 1):
        people = np.array(sorted({g for g, yy in zip(groups, y) if yy == c}))
        rng.shuffle(people)
        fold_of |= {p: i % folds for i, p in enumerate(people)}
    return np.array([fold_of[g] for g in groups])


def cnn_scores(X, y, groups, folds=5, epochs=30, seed=0, device="cpu"):
    """Held-out scores of the CNN, with folds split by person."""
    fold = person_folds(groups, y, folds, seed)
    scores = np.zeros(len(y))
    Xt = torch.tensor(np.clip(X, 0, 1.5)[:, None], dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    for k in range(folds):
        torch.manual_seed(seed + k)
        gen = torch.Generator().manual_seed(seed + k)
        train, test = np.nonzero(fold != k)[0], np.nonzero(fold == k)[0]
        net = Net().to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(epochs):
            net.train()
            perm = train[torch.randperm(len(train), generator=gen).numpy()]
            for i in range(0, len(perm), 32):
                idx = perm[i:i + 32]
                xb = augment(Xt[idx], gen).to(device)
                loss = nn.functional.binary_cross_entropy_with_logits(net(xb), yt[idx].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            scores[test] = torch.sigmoid(net(Xt[test].to(device))).cpu().numpy()
    return scores


def histogram_scores(X, y, groups, folds=5, seed=0):
    """Held-out scores of an L2-regularised logistic regression on 32 intensity quantiles."""
    feats = np.quantile(X.reshape(len(X), -1), np.linspace(0.02, 0.999, 32), axis=1).T
    fold = person_folds(groups, y, folds, seed)
    scores = np.zeros(len(y))
    for k in range(folds):
        tr, te = fold != k, fold == k
        mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-9
        f = torch.tensor((feats - mu) / sd, dtype=torch.float64)
        w = torch.zeros(f.shape[1] + 1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=200)
        yt = torch.tensor(y[tr], dtype=torch.float64)

        def loss():
            opt.zero_grad()
            z = f[tr] @ w[:-1] + w[-1]
            out = nn.functional.binary_cross_entropy_with_logits(z, yt) + 1e-2 * (w[:-1] ** 2).sum()
            out.backward()
            return out
        opt.step(loss)
        with torch.no_grad():
            scores[te] = torch.sigmoid(f[te] @ w[:-1] + w[-1]).numpy()
    return scores


def auc(y, s):
    """Mann-Whitney AUC (ties count half)."""
    pos, neg = s[y == 1], s[y == 0]
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def bootstrap(y, s, groups, b=2000, seed=0):
    """95 % interval of the AUC, resampling people (not slices)."""
    rng = np.random.default_rng(seed)
    people = np.array(sorted(set(groups)))
    rows = {p: np.nonzero(groups == p)[0] for p in people}
    pos = [p for p in people if y[rows[p][0]] == 1]
    neg = [p for p in people if y[rows[p][0]] == 0]
    out = []
    for _ in range(b):
        idx = np.concatenate([rows[p] for p in rng.choice(pos, len(pos))] + [rows[p] for p in rng.choice(neg, len(neg))])
        out.append(auc(y[idx], s[idx]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def intensity_gap(items_a, items_b, threshold=0.15):
    """How far apart the two sets' head intensities lie, without a classifier (it does not saturate):
    the 1-Wasserstein distance between the pooled distributions of pixels above ``threshold``
    (normalised units), averaged over slice indices, and the quartiles of each set."""
    out, qa, qb = [], [], []
    for s in sorted({i[1] for i in items_a} & {i[1] for i in items_b}):
        a = np.sort(np.concatenate([i[2][i[2] > threshold] for i in items_a if i[1] == s]))
        b = np.sort(np.concatenate([i[2][i[2] > threshold] for i in items_b if i[1] == s]))
        q = np.linspace(0, 1, 501)[1:-1]
        out.append(float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q)))))
        qa.append(np.quantile(a, [0.25, 0.5, 0.75]))
        qb.append(np.quantile(b, [0.25, 0.5, 0.75]))
    return {"w1": float(np.mean(out)), "quartiles_a": np.round(np.mean(qa, 0), 3).tolist(),
            "quartiles_b": np.round(np.mean(qb, 0), 3).tolist()}


def run(name, items_a, items_b, device, epochs):
    X = np.stack([i[2] for i in items_a + items_b])
    y = np.array([0] * len(items_a) + [1] * len(items_b))
    groups = np.array([i[0] for i in items_a + items_b])
    s = cnn_scores(X, y, groups, epochs=epochs, device=device)
    h = histogram_scores(X, y, groups)
    res = {"test": name, "intensity": intensity_gap(items_a, items_b), "n": [len(items_a), len(items_b)], "people": [len({i[0] for i in items_a}), len({i[0] for i in items_b})],
           "cnn_auc": auc(y, s), "cnn_ci": bootstrap(y, s, groups), "histogram_auc": auc(y, h), "histogram_ci": bootstrap(y, h, groups)}
    slices = np.array([i[1] for i in items_a + items_b])
    res["cnn_auc_by_slice"] = {int(k): round(auc(y[slices == k], s[slices == k]), 3) for k in sorted(set(slices))}
    print(json.dumps({k: v for k, v in res.items() if k != "cnn_auc_by_slice"}), flush=True)
    return res, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=Path, default=Path.home() / "data" / "realism_sim")
    ap.add_argument("--cache", type=Path, default=Path.home() / "data" / "realism_cache")
    ap.add_argument("--fastmri", type=Path, default=R.FASTMRI, help="directory with the fastMRI files and manifest.json")
    ap.add_argument("--out", type=Path, default=Path("outputs/realism"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--slices", default="0-12", help="slice indices used on both sides, a-b or a-b:step")
    args = ap.parse_args()
    rng_, _, step = args.slices.partition(":")
    slices = range(int(rng_.split("-")[0]), int(rng_.split("-")[-1]) + 1, int(step or 1))
    args.out.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    real, floor = real_set(args.cache, args.fastmri, slices)
    sim = sim_set(args.sim, floor, slices=slices)
    print(f"{len(real)} real slices from {len({i[0] for i in real})} scans, {len(sim)} simulated from "
          f"{len({i[0] for i in sim})} BrainWeb subjects; real noise floor {np.median(list(floor.values())):.4f} of p99", flush=True)
    results = []
    res, s = run("real vs simulated", real, sim, device, args.epochs)
    results.append(res)
    # null checks: the same test on two halves of one kind must give ~0.5
    rng = np.random.default_rng(1)
    for kind, items in (("real vs real", real), ("simulated vs simulated", sim)):
        people = sorted({i[0] for i in items})
        rng.shuffle(people)
        half = set(people[: len(people) // 2])
        results.append(run(kind, [i for i in items if i[0] in half], [i for i in items if i[0] not in half], device, args.epochs)[0])
    (args.out / "score.json").write_text(json.dumps(results, indent=1))
    np.savez(args.out / "examples.npz", real=np.stack([i[2] for i in real]), sim=np.stack([i[2] for i in sim]),
             sim_scores=s[len(real):], real_scores=s[: len(real)], real_groups=[i[0] for i in real],
             sim_groups=[i[0] for i in sim], real_slices=[i[1] for i in real], sim_slices=[i[1] for i in sim])


if __name__ == "__main__":
    main()
