"""Split a learned GRE (arm C's ``arm_C.pt``) into its two parts, for lesion_design.py's arm L:

    uv run python validation/design/split_design.py ~/data/design6_gauss_f0.5/arm_C.pt --out ~/data/design10_split

writes ``<out>/<run>_<part>/arm_C.pt`` for four parts: ``tetr``, the learned flip level, TE and TR with flat flips on
the start's lines (the contrast point a coarse search over flip, TE and TR could find); ``perline``, the learned
per-line flip pattern (shifted to the start's flip level) and line positions at the start's TE and TR (what only a
per-line design can do); and the per-line part's two halves at the start's TE and TR, ``flips`` (the flip pattern on
the start's lines) and ``shifts`` (the learned line positions with the start's flat flips). ``<run>`` is the name of
the folder of the input file. The start is lesion_design.py's arm-G
start (``--start``, flip:TE:TR) on the Gaussian line mask of ``--mask-seed``.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

from blochparty.design.parameters import NX, Design

_spec = importlib.util.spec_from_file_location("lesion_design", Path(__file__).with_name("lesion_design.py"))
lesion_design = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesion_design)


def _raw_flips(flips: torch.Tensor) -> torch.Tensor:
    return torch.log((flips - 2) / (90 - flips))  # Design's flip = 2 + 88 sigmoid(raw)


def split(learned: Design, start: Design) -> dict[str, Design]:
    """{part: design} for the parts of the module docstring."""
    with torch.no_grad():
        flips, ky, te, tr = learned()
        flips0, ky0, te0, tr0 = start()
        lines, pattern = len(flips), _raw_flips(flips - flips.mean() + flips0.mean())

        def at_start(ky_init, flip_raw=None):
            d = Design(lines, init_flip=float(flips0.mean()), init_te=float(te0), init_tr=float(tr0), ky_init=ky_init)
            if flip_raw is not None:
                d.flip_raw.copy_(flip_raw)
            return d

        tetr = Design(lines, init_flip=float(flips.mean()), init_te=float(te), init_tr=float(tr), ky_init=ky0)
        return {"tetr": tetr, "perline": at_start(ky, pattern), "flips": at_start(ky0, pattern), "shifts": at_start(ky)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design", type=Path, help="an arm_C.pt of lesion_design.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", default="15:60:70", help="arm G's start, flip_deg:TE_ms:TR_ms")
    ap.add_argument("--mask-seed", type=int, default=0)
    ap.add_argument("--lines", type=int, default=32)
    args = ap.parse_args()
    learned = Design(args.lines)
    learned.load_state_dict(torch.load(args.design, weights_only=False, map_location="cpu")["design"])
    flip, te_ms, tr_ms = (float(v) for v in args.start.split(":"))
    cols = torch.nonzero(lesion_design.sampled_lines(NX, args.lines, "gauss", seed=args.mask_seed)).squeeze(1)
    start = Design(
        args.lines, init_flip=flip, init_te=te_ms * 1e-3, init_tr=tr_ms * 1e-3, ky_init=cols.float() - NX / 2
    )
    run = args.design.parent.name
    print(f"{'start':>{len(run) + 8}}: {start.summary()}")
    print(f"{run + ' learned':>{len(run) + 8}}: {learned.summary()}")
    for part, d in split(learned, start).items():
        folder = args.out / f"{run}_{part}"
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({"design": d.state_dict(), "describe": d.describe(), "from": str(args.design)}, folder / "arm_C.pt")
        print(f"{run + '_' + part:>{len(run) + 8}}: {d.summary()}")


if __name__ == "__main__":
    main()
