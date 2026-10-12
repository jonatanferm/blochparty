"""Helpers of the realism experiment (validation/realism/simulate_heads.py, validation/anatomy/prepare_ixi.py):
the display orientation, the receive ring's position and the soft-tissue references."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

import blochparty as bp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation" / "realism"))
sys.path.insert(0, str(ROOT / "validation" / "anatomy"))
import simulate_heads as SH  # noqa: E402
import prepare_ixi as P  # noqa: E402


def test_display_is_posterior_up():
    """turn() makes the phantom's x the anterior direction (x' = y); a reconstructed (y', x') image with a mark at
    the most anterior x' shows it on the bottom row, as fastMRI shows the head (validation/realism/orientation.py)."""
    pos = torch.tensor([[0.0, 0.05, 0.0]])  # 5 cm anterior
    ph = SH.turn(bp.Phantom(pos=pos, pd=torch.ones(1), t1=torch.ones(1), t2=torch.ones(1), db0=torch.zeros(1)))
    assert ph.pos[0, 0].item() == pytest.approx(0.05)
    img = np.zeros((8, 8))
    img[3, 7] = 1  # row y', column x' at the anterior end
    shown = SH.to_real_orientation(img)
    assert np.unravel_index(shown.argmax(), shown.shape) == (7, 3)


def test_coil_offset_puts_the_ring_behind_the_occiput():
    """A box head from y = -90 to +90 mm: the ring's back ``gap`` behind the head's back."""
    n, h = (40, 46, 40), 4e-3
    head = torch.zeros(n)
    y = -0.092 + np.arange(n[1]) * h  # voxel centres
    head[:, (y >= -0.09) & (y <= 0.09), 5:35] = 1
    an = bp.Anatomy({"skin": head, "background": 1 - head}, voxel=(h,) * 3, origin=(-0.08, -0.092, -0.08))
    back = y[(y >= -0.09)].min()
    off = SH.coil_offset(an, np.array([0.0]), 0.0, radius=0.14, gap=0.025)
    assert off == pytest.approx(back - 0.025 + 0.14)
    assert SH.coil_offset(an, np.array([0.0]), 0.01, radius=0.14, gap=0.025) == pytest.approx(off - 0.01)


def test_soft_references_follow_the_tissue_table():
    """Skin and muscle relative to white matter in IXI's T2 (TR 5725.8 ms, TE 100 ms), spin-echo signal."""
    from blochparty import tissuedb
    table = tissuedb.tissue_table(3.0)
    tr, te = P.IXI_T2["tr"], P.IXI_T2["te"]
    s = {k: table[k].pd.mean * (1 - math.exp(-tr / table[k].t1.mean)) * math.exp(-te / table[k].t2.mean)
         for k in ("skin", "muscle", "wm")}
    skin, muscle = P.soft_references()
    assert skin == pytest.approx(s["skin"] / s["wm"])
    assert muscle == pytest.approx(s["muscle"] / s["wm"])
    assert 0 < muscle < 0.5 < skin < 1
