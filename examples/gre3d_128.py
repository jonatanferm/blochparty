"""Full 3D 128^3 spoiled GRE (565k spins, 2.2M segments, 8 coils) on the Apple GPU (Metal);
saves central slices. 192 s on an M4's GPU (2026-10-09; the retired fused CPU kernel took 6 min).

    uv run python examples/gre3d_128.py      -> outputs/gre3d_128.png
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import blochparty as bp

N = 128

phantom = bp.shepp_logan(N, ndim=3).to(dtype=torch.float32)
coils = bp.birdcage_sensitivities(phantom.pos, n_coils=8)
seq = bp.gre_3d(N, flip_deg=15.0, tr=10e-3, n_dummy=200, dtype=torch.float32)

t0 = time.perf_counter()
signal = bp.simulate(phantom, seq, coils, engine="gpu").signal
t_sim = time.perf_counter() - t0
print(
    f"{len(phantom)} spins x {len(seq)} segments, {seq.n_adc} samples x 8 coils: {t_sim:.0f} s "
    f"({len(phantom) * len(seq) / t_sim:.2e} spin-segments/s)"
)

img = bp.rss(bp.reconstruct(signal, seq))  # (z, y, x)
fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
for a, im, title in zip(ax, (img[N // 2], img[:, N // 2], img[:, :, N // 2]), ("axial (z = 0)", "coronal", "sagittal")):
    a.imshow(im, cmap="gray", origin="lower")
    a.set_title(title)
    a.axis("off")
fig.suptitle(f"Bloch Party: 3D spoiled GRE 128³, 8 coils, Apple GPU ({t_sim:.0f} s)")
fig.tight_layout()
out = Path("outputs") / "gre3d_128.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=110)
print(f"wrote {out}")
