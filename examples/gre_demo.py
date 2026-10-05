"""Simulate an 8-channel 2D spoiled GRE of the Shepp-Logan phantom and plot
tissue maps, coil images, reconstructions and the Bloch-vs-signal-equation error.

    uv run python examples/gre_demo.py      -> outputs/gre_demo.png
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import mrsim
from mrsim.reference import signal_equation

N, FOV, COILS = 96, 0.22, 8

maps = mrsim.shepp_logan_maps(N)
phantom = mrsim.phantom_from_maps(maps["pd"], maps["t1"], maps["t2"], FOV)
coils = mrsim.birdcage_sensitivities(phantom.pos, n_coils=COILS)
seq = mrsim.gre_2d(N, FOV, flip_deg=15.0, tr=10e-3, te=4e-3, n_dummy=200)

t0 = time.perf_counter()
signal = mrsim.simulate(phantom, seq, coils).signal
t_sim = time.perf_counter() - t0
print(f"{len(phantom)} spins, {len(seq)} segments, {seq.n_adc} ADC samples x {COILS} coils: {t_sim:.2f} s")

ref = signal_equation(phantom, seq, coils)
rel_err = ((signal - ref).abs().max() / ref.abs().max()).item()
print(f"max |Bloch - signal equation| / max |signal| = {rel_err:.2e}")

noisy = mrsim.add_noise(signal, sigma=0.02 * signal.abs().max().item() / N,
                        generator=torch.Generator().manual_seed(0))
coil_imgs = mrsim.ifft2c(mrsim.kspace_grid(noisy, seq))
grid = mrsim.grid_positions(N, FOV).reshape(-1, 3)
sens_grid = mrsim.birdcage_sensitivities(grid, n_coils=COILS).T.reshape(COILS, N, N)
err_img = mrsim.ifft2c(mrsim.kspace_grid(signal - ref, seq))

fig, ax = plt.subplots(3, 4, figsize=(13, 10))
show = lambda a, img, title, **kw: (a.imshow(img, origin="lower", cmap=kw.pop("cmap", "gray"), **kw),
                                    a.set_title(title), a.axis("off"))
show(ax[0, 0], maps["pd"], "PD")
show(ax[0, 1], maps["t1"].where(maps["pd"] > 0, 0), "T1 [s]", cmap="magma")
show(ax[0, 2], maps["t2"].where(maps["pd"] > 0, 0).clamp(max=0.3), "T2 [s] (clipped 0.3)", cmap="magma")
show(ax[0, 3], mrsim.kspace_grid(noisy, seq)[0].abs().log10(), "log10 |k-space|, coil 1", cmap="viridis")
for c in range(4):
    show(ax[1, c], coil_imgs[2 * c].abs(), f"coil {2 * c + 1} image")
show(ax[2, 0], mrsim.rss(sens_grid), "RSS of coil sensitivities", cmap="viridis")
show(ax[2, 1], mrsim.rss(coil_imgs), "RSS recon (with noise)")
show(ax[2, 2], mrsim.coil_combine(coil_imgs, sens_grid).abs(), "sensitivity-weighted combination")
show(ax[2, 3], err_img.abs().sum(0), f"|Bloch - signal eq.| image (max rel {rel_err:.0e})", cmap="inferno")
fig.suptitle(f"mrsim: 2D spoiled GRE, {N}x{N}, TR 10 ms, TE 4 ms, 15 deg, {COILS} coils ({t_sim:.1f} s)")
fig.tight_layout()
out = Path("outputs") / "gre_demo.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=110)
print(f"wrote {out}")
