"""The README's figures, every one simulated by blochparty: a sampled head in three contrasts (`hero.png`), the
pipeline from tissue maps through k-space to an image (`pipeline.png`), a spin echo as an animation checked
against its closed form (`spin_echo.gif`), and gradient descent through the simulator onto the analytic optimum
(`optimize_flip[_dark].png`).

    uv run python tools/readme_figures.py [--only hero,echo,design] [--out docs/images]

The head needs BrainWeb (`validation/anatomy/fetch_brainweb.py`) and runs on the GPU engine where there is one
(~6 min on an M4 laptop, cached in outputs/readme_head.npz for redraws); the rest takes seconds on the CPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import blochparty as bp

REPO = Path(__file__).resolve().parents[1]
BG, FG, MUTED = "#0d1117", "#e6edf3", "#8b949e"


def load_example(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------- the head


def head_figures(out: Path, seed: int = 3, n: int = 160, cache: Path = REPO / "outputs" / "readme_head.npz"):
    """Simulates once into ``cache`` (the images, coil 1's k-space and the tissue maps); later runs only redraw."""
    if not cache.exists():
        simulate_head(cache, seed, n)
    maps_cache = cache.with_name(cache.stem + "_maps.npz")
    if not maps_cache.exists():
        tissue_maps(maps_cache, seed, n)
    d = {**np.load(cache, allow_pickle=True), **np.load(maps_cache)}
    names, images = list(d["names"]), list(d["images"])

    # hero: the three contrasts, edge to edge
    fig, ax = plt.subplots(1, 3, figsize=(12, 5.4), facecolor=BG)
    for a, name, img in zip(ax, names, images):
        a.imshow(crop(img), cmap="gray", vmin=0, vmax=np.percentile(img, 99.6))
        a.set_axis_off()
        title, params = name.split(" (")
        a.text(0.5, -0.03, title, color=FG, fontsize=13, ha="center", va="top", transform=a.transAxes, weight="bold")
        a.text(0.5, -0.09, params.rstrip(")"), color=MUTED, fontsize=9.5, ha="center", va="top", transform=a.transAxes)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.1, wspace=0.03)
    fig.savefig(out / "hero.png", dpi=110, facecolor=BG)
    plt.close(fig)

    # pipeline: the tissue the spins carry -> raw k-space -> image
    body = np.nan_to_num(d["pd"]) > 0.05 * np.nanmax(d["pd"])
    mask = lambda m: np.where(body, m, np.nan)  # noqa: E731
    panels = [
        ("proton density", mask(d["pd"]), "gray", None),
        ("T1  [s]", mask(d["t1"]), "magma", (0, 4.0)),
        ("T2  [ms]", mask(1e3 * d["t2"]), "viridis", (20, 160)),
        ("B0 from susceptibility  [Hz]", mask(d["db0"]), "RdBu_r", (-40, 40)),
        ("raw k-space (coil 1)", np.log10(d["kspace"] + d["kspace"].max() * 1e-5), "inferno", None),
        ("T1w image", images[0], "gray", None),
    ]
    fig, ax = plt.subplots(1, 6, figsize=(16, 4.0), facecolor=BG)
    for i, (a, (title, img, cmap, lim)) in enumerate(zip(ax, panels)):
        raw = title.startswith("raw")
        view = img[:, round(0.1 * img.shape[1]) : round(0.9 * img.shape[1])] if raw else crop(img)
        lo, hi = lim or (np.nanpercentile(view, 0.5), np.nanpercentile(view, 99.6))
        a.imshow(view, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest" if raw else None)
        a.set_facecolor(BG)
        a.set_xticks([]), a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        a.set_title(title, color=FG if i >= 4 else MUTED, fontsize=10.5, pad=6)
    fig.text(0.34, 0.03, "what every spin carries", color=MUTED, fontsize=10, ha="center")
    fig.text(0.75, 0.03, "Bloch simulation  →  8 coils  →  reconstruction", color=MUTED, fontsize=10, ha="center")
    plt.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.09, wspace=0.04)
    fig.savefig(out / "pipeline.png", dpi=110, facecolor=BG)
    plt.close(fig)


def simulate_head(cache: Path, seed: int, n: int):
    ex = load_example("sampled_heads")
    fov, thickness = 0.24, 5e-3
    head = bp.sample_head(seed, root=ex.ROOT, voxel=(1e-3,) * 3)
    k = round(-head.anatomy.origin[2] / head.anatomy.voxel[0])
    ph = head.to_phantom(fov_region=(slice(k - 4, k + 5), slice(None), slice(None)), spins_per_voxel=(2, 2, 2))
    coils = bp.birdcage_sensitivities(ph.pos, 8)
    print(f"head {seed}: subject {head.meta['subject']}, {len(ph):,} spins", flush=True)
    names, images, kspace = [], [], None
    for name, seq in ex.protocols(n, fov, thickness).items():
        signal, engine = ex.simulate(ph, seq, coils)
        level = float(np.percentile(bp.rss(bp.cartesian_recon(signal, seq)).numpy(), 99))
        signal = bp.add_noise(signal, level * n / 60.0, generator=torch.Generator().manual_seed(seed))
        names.append(name)
        images.append(bp.rss(bp.cartesian_recon(signal, seq)).numpy()[::-1])
        if kspace is None:
            kspace = bp.cartesian_kspace(signal, seq)[0].abs().numpy()[::-1]
        print(f"  {name} ({engine})", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, names=np.array(names), images=np.stack(images), kspace=kspace)


def tissue_maps(cache: Path, seed: int, n: int, fov: float = 0.24):
    """Per-pixel maps of the spins in the central 1 mm. A voxel carries one set of spins per tissue in it, each with
    PD = fraction * PD / K, so PD is the pixel's sum and T1, T2 and B0 are PD-weighted medians: a plain mean or
    median lets a minor tissue (CSF at T2 ~2 s) or a fat spin (its -434 Hz chemical shift is in db0) set the pixel."""
    ex = load_example("sampled_heads")
    head = bp.sample_head(seed, root=ex.ROOT, voxel=(1e-3,) * 3)
    k = round(-head.anatomy.origin[2] / head.anatomy.voxel[0])
    ph = head.to_phantom(fov_region=(slice(k - 1, k + 2), slice(None), slice(None)), spins_per_voxel=(2, 2, 2))
    sel = (ph.pos[:, 2].abs() < 0.5e-3).numpy()
    pos, w = ph.pos.numpy()[sel], ph.pd.numpy()[sel]
    iy = np.clip(((pos[:, 1] + fov / 2) / fov * n).astype(int), 0, n - 1)
    ix = np.clip(((pos[:, 0] + fov / 2) / fov * n).astype(int), 0, n - 1)
    pixel = iy * n + ix
    order = np.argsort(pixel, kind="stable")
    bins, starts = np.unique(pixel[order], return_index=True)
    groups = lambda v: np.split(v[order], starts[1:])  # noqa: E731

    def weighted_median(v, wt):
        i = np.argsort(v)
        c = np.cumsum(wt[i])
        return v[i][np.searchsorted(c, 0.5 * c[-1])] if c[-1] > 0 else np.nan

    def grid(values):
        out = np.full(n * n, np.nan)
        out[bins] = values
        return out.reshape(n, n)[::-1]

    maps = {"pd": grid([g.sum() for g in groups(w)])}
    for key in ("t1", "t2", "db0"):
        values = getattr(ph, key).numpy()[sel]
        maps[key] = grid([weighted_median(v, wt) for v, wt in zip(groups(values), groups(w))])
    np.savez(cache, **maps)


def crop(img: np.ndarray, x: float = 0.1, y: float = 0.01) -> np.ndarray:
    """Trims the sides (heads are taller than wide) and a sliver top and bottom."""
    my, mx = round(y * img.shape[0]), round(x * img.shape[1])
    return img[my : img.shape[0] - my, mx : img.shape[1] - mx]


# ------------------------------------------------------------------------------------------- spin echo


def spin_echo_gif(out: Path, n_spins: int = 36, spread: float = 60.0, te: float = 60e-3, t2: float = 0.12):
    """Isochromats with uniform offsets in +-spread/2 Hz fan out after a 90x and refocus after a 180y at TE/2.
    The arrows are blochparty's magnetization at each frame; the signal trace is blochparty's ADC (dots)
    against the closed form e^{-t/T2} |mean e^{-i2 pi f tau}| (line), tau the phase-accrual time."""
    f = torch.linspace(-spread / 2, spread / 2, n_spins, dtype=torch.float64)
    ph = bp.Phantom(
        pos=torch.zeros(n_spins, 3, dtype=torch.float64),
        pd=torch.full((n_spins,), 1.0 / n_spins, dtype=torch.float64),
        t1=torch.full((n_spins,), 10.0, dtype=torch.float64),
        t2=torch.full((n_spins,), t2, dtype=torch.float64),
        db0=f,
    )
    t_rf = 50e-6

    def seq_until(t: float) -> bp.Sequence:
        """90x at 0, 180y at TE/2 (hard pulses), stopped at time t."""
        b = bp.SequenceBuilder()
        b.rf(t_rf, flip_deg=90.0)
        if t <= te / 2:
            if t > t_rf:
                b.delay(t - t_rf)
            return b.build()
        b.delay(te / 2 - t_rf).rf(t_rf, flip_deg=180.0, phase=math.pi / 2)
        if t > te / 2 + t_rf:
            b.delay(t - te / 2 - t_rf)
        return b.build()

    # the trace: one run with ADC samples over the whole echo, 0 .. 1.5 TE
    n_adc = 30
    dw1, dw2 = (te / 2 - t_rf) / n_adc, (te - t_rf) / (2 * n_adc)
    b = bp.SequenceBuilder().rf(t_rf, flip_deg=90.0).readout(n_adc, dw1)
    seq = b.rf(t_rf, flip_deg=180.0, phase=math.pi / 2).readout(2 * n_adc, dw2).build()
    sig = bp.simulate(ph, seq).signal[0].abs().numpy()
    t_adc = np.concatenate([t_rf + (np.arange(n_adc) + 0.5) * dw1, te / 2 + t_rf + (np.arange(2 * n_adc) + 0.5) * dw2])

    def closed_form(t):  # phase accrues from the end of the 90 and is undone from TE/2 to TE
        return np.exp(-t / t2) * discrete_sum(f.numpy(), np.where(t < te / 2, t - t_rf, t - te))

    tt = np.linspace(t_rf, 1.5 * te, 600)
    closed = closed_form(tt)
    err = np.abs(sig - closed_form(t_adc)).max()
    print(f"spin echo: blochparty vs closed form, max |difference| {err:.1e} (hard pulses {t_rf * 1e6:.0f} us)")

    from matplotlib.animation import FuncAnimation, PillowWriter

    frames = np.linspace(0, 1.5 * te, 75)
    colors = plt.cm.cool((f.numpy() - f.min().item()) / spread)
    fig = plt.figure(figsize=(7.2, 3.6), facecolor=BG)
    ax = fig.add_axes([0.01, 0.06, 0.42, 0.88], facecolor=BG)
    tr = fig.add_axes([0.52, 0.2, 0.45, 0.66], facecolor=BG)
    ax.set_xlim(-1.15, 1.15), ax.set_ylim(-1.15, 1.15), ax.set_aspect("equal"), ax.set_axis_off()
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#30363d", lw=1))
    ax.text(1.12, 0, "x", color=MUTED, fontsize=9, va="center"), ax.text(0, 1.08, "y", color=MUTED, fontsize=9)
    arrows = [ax.annotate("", (0, 0), (0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4)) for c in colors]
    total = ax.annotate("", (0, 0), (0, 0), arrowprops=dict(arrowstyle="-|>", color=FG, lw=2.6))
    label = ax.text(0, -1.13, "", color=FG, fontsize=9.5, ha="center")
    tr.plot(1e3 * tt, closed, color=MUTED, lw=1.6, label="closed form")
    tr.plot(1e3 * t_adc, sig, "o", color="#58a6ff", ms=3.2, mfc="none", mew=1.1, label="blochparty ADC")
    (cursor,) = tr.plot([0, 0], [0, 1.05], color="#f778ba", lw=1)
    tr.axvline(1e3 * te / 2, color="#30363d", lw=1, ls="--")
    tr.text(1e3 * te / 2, 1.08, "180°", color=MUTED, fontsize=8, ha="center")
    tr.text(1e3 * te, 1.08, "echo", color=MUTED, fontsize=8, ha="center")
    tr.set_xlim(0, 1.5e3 * te), tr.set_ylim(0, 1.05)
    tr.set_xlabel("time  [ms]", color=MUTED, fontsize=9), tr.set_ylabel("|signal|", color=MUTED, fontsize=9)
    tr.tick_params(colors=MUTED, labelsize=8)
    for s in tr.spines.values():
        s.set_color("#30363d")
    tr.legend(frameon=False, fontsize=8, labelcolor=FG, loc="upper right", bbox_to_anchor=(1.0, 0.92))

    def draw(t):
        m = bp.simulate(ph, seq_until(t)).m.numpy() * n_spins
        for a, (mx, my, _) in zip(arrows, m):
            a.xy = (mx, my)
        mean = m.mean(axis=0)
        total.xy = (mean[0], mean[1])
        cursor.set_xdata([1e3 * t] * 2)
        label.set_text(f"t = {1e3 * t:4.1f} ms")
        return [*arrows, total, cursor, label]

    anim = FuncAnimation(fig, draw, frames=list(frames) + [frames[-1]] * 12, blit=False)
    anim.save(out / "spin_echo.gif", writer=PillowWriter(fps=15), dpi=90, savefig_kwargs={"facecolor": BG})
    plt.close(fig)


def discrete_sum(f: np.ndarray, t: np.ndarray) -> np.ndarray:
    """|mean over spins of e^{-i 2 pi f t}| for the spins' own (discrete) offsets."""
    return np.abs(np.exp(-2j * np.pi * np.outer(t, f)).mean(axis=1))


# --------------------------------------------------------------------------------------------- design


def design_figure(out: Path):
    """The flip angle that maximises WM/GM contrast in a simulated spoiled-GRE image: L-BFGS on gradients
    through the Bloch simulation and the reconstruction, against the closed-form steady state."""
    from blochparty.phantom import LABELS
    from blochparty.reference import spoiled_gre_signal

    n, fov, tr, te = 32, 0.22, 10e-3, 3e-3
    maps = bp.shepp_logan_maps(n)
    phantom = bp.phantom_from_maps(maps["pd"], maps["t1"], maps["t2"], fov)
    wm, gm = maps["label"] == LABELS.index("wm"), maps["label"] == LABELS.index("gm")
    flip = torch.tensor(40.0, dtype=torch.float64, requires_grad=True)
    path = []

    def contrast():
        seq = bp.gre_2d(n, fov, flip_deg=flip, tr=tr, te=te, dwell=2e-6, n_dummy=400)
        img = bp.ifft2c(bp.kspace_grid(bp.simulate(phantom, seq).signal, seq))[0].abs()
        return img[wm].mean() - img[gm].mean()

    opt = torch.optim.LBFGS([flip], lr=1, max_iter=30, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -contrast()
        loss.backward()
        path.append((abs(flip.item()), -loss.item(), -flip.grad.item()))
        return loss

    opt.step(closure)
    angles = torch.linspace(1, 90, 2000, dtype=torch.float64)
    props = {k: (maps["pd"][m][0], maps["t1"][m][0], maps["t2"][m][0]) for k, m in (("wm", wm), ("gm", gm))}
    curve = (
        spoiled_gre_signal(*props["wm"], tr, te, torch.deg2rad(angles))
        - spoiled_gre_signal(*props["gm"], tr, te, torch.deg2rad(angles))
    ).numpy()
    best = angles[curve.argmax()].item()
    print(f"design: gradient descent {path[-1][0]:.3f} deg in {len(path)} simulations, analytic {best:.3f} deg")
    # the image is the centred iFFT of n x n samples: scale the closed form to it from the first point
    scale = path[0][1] / np.interp(path[0][0], angles.numpy(), curve)

    for theme, (bg, fg, muted, grid) in {
        "": ("#ffffff", "#1f2328", "#656d76", "#d0d7de"),
        "_dark": (BG, FG, MUTED, "#30363d"),
    }.items():
        fig, ax = plt.subplots(figsize=(7.2, 3.3), facecolor=bg)
        ax.set_facecolor(bg)
        ax.plot(angles, scale * curve, color=muted, lw=1.5, label="closed-form steady state")
        xs, ys = np.array([p[0] for p in path]), np.array([p[1] for p in path])
        order = np.arange(len(xs)) / (len(xs) - 1)
        ax.scatter(xs, ys, c=order, cmap="plasma", vmin=-0.15, vmax=1.0, s=42, zorder=3, edgecolor=bg, linewidth=0.9)
        ax.scatter([], [], color="#e16f24", s=36, label=f"L-BFGS on simulator gradients ({len(xs)} simulations)")
        ax.annotate(f"start {xs[0]:.0f}°", (xs[0], ys[0]), (14, -16), textcoords="offset points", color=fg, fontsize=9)
        ax.annotate(
            f"{xs[-1]:.2f}°  (analytic {best:.2f}°)",
            (xs[-1], ys[-1]),
            (40, 22),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color=muted, lw=0.8),
            color=fg,
            fontsize=9,
        )
        ax.set_xlabel("flip angle  [°]", color=muted), ax.set_ylabel("WM − GM contrast", color=muted)
        ax.set_xlim(0, 60), ax.set_ylim(top=1.22 * ys.max())
        ax.tick_params(colors=muted, labelsize=9)
        ax.grid(color=grid, lw=0.6)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.legend(frameon=False, fontsize=9, labelcolor=fg, loc="lower right")
        plt.tight_layout()
        fig.savefig(out / f"optimize_flip{theme}.png", dpi=110, facecolor=bg)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="hero,echo,design")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "images")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(","))
    if "echo" in only:
        spin_echo_gif(args.out)
    if "design" in only:
        design_figure(args.out)
    if "hero" in only:
        head_figures(args.out)


if __name__ == "__main__":
    main()
