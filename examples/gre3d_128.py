"""Full 3D 128^3 spoiled GRE (565k spins, 2.2M segments, 8 coils) with the fused C++
kernel; saves central slices. About 6 min on an Apple M4.

    uv run python examples/gre3d_128.py      -> outputs/gre3d_128.png
"""
import time

import matplotlib
import torch
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import blochparty as bp
from blochparty.fused import simulate_fused
N = 128
t0 = time.perf_counter()
phantom = bp.shepp_logan(N, ndim=3, dtype=torch.float64).to(dtype=torch.float32)
coils = bp.birdcage_sensitivities(phantom.pos, n_coils=8)
seq = bp.gre_3d(N, flip_deg=15.0, tr=10e-3, n_dummy=200, dtype=torch.float32)
t_build = time.perf_counter() - t0
t0 = time.perf_counter()
sig = simulate_fused(phantom, seq, coils).signal
t_sim = time.perf_counter() - t0
img = bp.rss(bp.reconstruct(sig, seq))
print(f"{len(phantom)} spins x {len(seq)} segments, {seq.n_adc} samples x 8 coils | build {t_build:.0f} s | simulate {t_sim:.0f} s ({len(phantom)*len(seq)/t_sim:.2e} spin-seg/s)", flush=True)
fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
for a, im, t in zip(ax, (img[N // 2], img[:, N // 2], img[:, :, N // 2]), ("axial (z = 0)", "coronal", "sagittal")):
    a.imshow(im, cmap="gray", origin="lower"); a.set_title(t); a.axis("off")
fig.suptitle(f"Bloch Party: 3D spoiled GRE 128³, 8 coils, fused CPU kernel ({t_sim / 60:.1f} min on Apple M4)")
fig.tight_layout(); fig.savefig("outputs/gre3d_128.png", dpi=110); print("wrote outputs/gre3d_128.png")
