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


def test_residual_maps_recover_known_multipliers():
    """A synthetic head (white matter, a CSF block for the echo-ratio scale, a grey-matter block, a skin slab) whose
    two echoes are made with grey matter's T2 x1.2 and PD x0.9 and skin's T2 x1.3 against the tissue table (and an
    arbitrary scale on the PD echo): the fitted t2_scale / pd_scale are ~1.2 / ~0.9 in the grey matter, 1 in the white
    matter and 1 in the skin (outside the skull the residual is not fitted)."""
    from blochparty import tissuedb
    table = tissuedb.tissue_table(3.0)
    n = (24, 24, 24)
    frac = np.zeros((len(P.CLASSES),) + n, np.float32)
    k = {c: P.CLASSES.index(c) for c in P.CLASSES}
    frac[k["wm"]] = 1
    csf = (slice(2, 12),) * 3
    gm = (slice(13, 22), slice(13, 22), slice(13, 22))
    frac[k["wm"]][csf] = 0; frac[k["csf"]][csf] = 1
    frac[k["wm"]][gm] = 0; frac[k["gm"]][gm] = 1
    skin = (slice(None), slice(None), slice(22, 24))
    frac[k["wm"]][skin] = 0; frac[k["skin"]][skin] = 1
    tr = P.IXI_T2["tr"]

    def echo(te, t2_gm, pd_gm):
        out = np.zeros(n, np.float32)
        for c, t2f, pdf in (("wm", 1, 1), ("csf", 1, 1), ("gm", t2_gm, pd_gm), ("skin", 1.3, 1)):
            p = table[c]
            out += frac[k[c]] * p.pd.mean * pdf * (1 - np.exp(-tr / p.t1.mean)) * np.exp(-te / (p.t2.mean * t2f))
        return out
    t2w = echo(P.IXI_T2["te"], 1.2, .9)
    pdw = 3.7 * echo(P.IXI_PD_TE, 1.2, .9)
    origin = (0.05, 0.05, 0.05)  # far from MNI's deep grey regions and midline
    t2s, pds, diag = P.residual_maps(frac, t2w, pdw, np.ones(n, bool), origin, (1e-3,) * 3)
    inner = (slice(15, 20),) * 3
    assert np.median(t2s[inner]) == pytest.approx(1.2, rel=0.03)
    assert np.median(pds[inner]) == pytest.approx(0.9, rel=0.03)
    assert np.median(t2s[frac[k["wm"]] > .9]) == pytest.approx(1.0, abs=0.01)
    assert np.median(pds[frac[k["wm"]] > .9]) == pytest.approx(1.0, abs=0.01)
    assert np.all(t2s[skin] == 1) and np.all(pds[skin] == 1)


def test_header_train_matches_the_real_echo_blocks():
    """The real AXT2 k-space energy steps between the echoes' blocks of 19 lines at ky -161, -142, ..., -47, -28, -9 and
    +10 (validation/realism/echo_train.py over 21 scans): 396 lines over 21 echoes, ky = 0 in echo 11. The header train
    (REAL_ETL at the header's 10.26 ms) puts the 128 central lines' block boundaries there and TE at 11 echo spacings."""
    seq = SH.sequence(128, 6.19, 0.113, ordering="linear_full", etl=SH.REAL_ETL, esp=0.01026)
    echo = np.asarray(seq.meta["line_echo"])
    ky = np.arange(128) - 64
    assert echo[ky == 0][0] == 10  # 0-based: echo 11
    assert set(ky[1:][np.diff(echo) != 0].tolist()) == {-47, -28, -9, 10, 29, 48}
    assert seq.meta["te"] == pytest.approx(11 * 0.01026)
