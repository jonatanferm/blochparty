"""The real AXT2 scans' echo trains, read from their k-space: in a linear-ordered TSE each echo fills one block of
adjacent phase-encode lines (as many as there are shots), and the line energy steps where one echo's block meets the
next (relaxation and the train's own amplitude pattern differ between echoes). Per line: the energy summed over coils
and readout (slices 1-7 of every 3 T AXT2 volume), its log minus a 15-line running median, averaged; the periodicity of
what is left, the largest steps, and the +ky / -ky balance.

    uv run python validation/realism/echo_train.py [--fastmri ~/data/fastmri_brain]

2026-10-12 (21 volumes, all 396 x 768): period 18.9 lines on the -ky side; steps at ky -161, -142, -123, -104, -85,
-66, -47, -28, -9 and +10 (between lines -10 / -9 and 9 / 10): blocks of 19 lines, the one holding ky = 0 centred on
it, so 396 lines over 21 echoes (19 shots), ky = 0 in echo 11; the headers' echo spacing gives TE / spacing = 113 /
10.26 = 11.0 and 105 / 9.56 = 11.0. +ky / -ky energy 0.86-0.92 from 5 to 150 lines out (later echoes on +ky).
simulate_heads.py's ``--train header`` plays this train; the fitted one was 16 echoes at TE / 9 (12.6 ms, 25 shots).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import realism as R  # noqa: E402
from scipy import ndimage, signal


def line_energy(paths, slices=range(1, 8)):
    """[(log energy per phase-encode line)], [(file, TE ms, echo spacing ms)] for 3 T volumes."""
    import h5py
    rows, info = [], []
    for f in paths:
        with h5py.File(f, "r") as h:
            x = h["ismrmrd_header"][()].decode()
            esp, te = re.search(r"<echo_spacing>([\d.]+)", x), re.search(r"<TE>([\d.]+)", x)
            k = h["kspace"]
            for s in slices:
                rows.append(np.log((np.abs(k[s]) ** 2).sum((0, 1)) + 1e-30))
        info.append((f.name, float(te.group(1)) if te else None, float(esp.group(1)) if esp else None))
    return rows, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fastmri", type=Path, default=R.FASTMRI)
    args = ap.parse_args()
    rows, info = line_energy(R.manifest(root=args.fastmri))
    n = {len(r) for r in rows}
    if len(n) != 1:
        raise SystemExit(f"volumes differ in phase-encode lines: {sorted(n)}")
    L = np.array(rows)
    c = L.shape[1] // 2
    res = (L - ndimage.median_filter(L, size=(1, 15), mode="nearest")).mean(0)
    print(f"{len(info)} volumes, {L.shape[1]} lines; (TE, echo spacing) [ms]: {sorted({i[1:] for i in info})}")
    for side, sl in (("-ky", slice(c - 190, c - 20)), ("+ky", slice(c + 20, c + 190))):
        f, p = signal.periodogram(res[sl] - res[sl].mean())
        top = [i for i in np.argsort(p)[::-1] if f[i] > 0][:3]
        print(f"{side}: strongest periods {[round(1 / f[i], 1) for i in top]} lines")
    jumps = np.argsort(np.abs(np.diff(res)))[::-1][:24]
    print("largest steps (ky of the line after the step):", sorted(int(j + 1 - c) for j in jumps))
    q = (5, 10, 20, 40, 64, 100, 150)
    print("+ky / -ky energy at", q, "lines:", np.round([np.exp(np.mean(L[:, c + d] - L[:, c - d])) for d in q], 3).tolist())


if __name__ == "__main__":
    main()
