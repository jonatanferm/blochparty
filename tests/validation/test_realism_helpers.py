"""Helpers of the realism experiment (validation/realism/simulate_heads.py, validation/anatomy/prepare_ixi.py):
the display orientation, the receive ring's position and the soft-tissue references."""
import math

import numpy as np
import pytest
import torch

import blochparty as bp

pytest.importorskip("nibabel")  # prepare_ixi reads NIfTI: the optional anatomy extra
import simulate_heads as SH  # noqa: E402
import prepare_ixi as P  # noqa: E402


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
    two echoes are made with grey matter's T2 x1.2 and PD x0.9, a sub-block of it at PD x0.85 more, and skin's T2 x1.3
    against the tissue table (and an arbitrary scale on the PD echo). t2_scale: ~1.2 in the grey matter, 1 in the white
    matter. pd_scale: re-centred per tissue (a tissue-wide PD offset is the subject's sequence's MT, not anatomy), so
    ~1 in the grey matter's bulk and below it in the sub-block. Both 1 in the skin (not fitted outside the skull)."""
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

    sub = (slice(13, 18),) * 3  # a corner of the grey-matter block
    pd_gm_map = np.full(n, .9, np.float32); pd_gm_map[sub] *= .85

    def echo(te, t2_gm, pd_gm):
        out = np.zeros(n, np.float32)
        for c, t2f, pdf in (("wm", 1, 1), ("csf", 1, 1), ("gm", t2_gm, pd_gm), ("skin", 1.3, 1)):
            p = table[c]
            out += frac[k[c]] * p.pd.mean * pdf * (1 - np.exp(-tr / p.t1.mean)) * np.exp(-te / (p.t2.mean * t2f))
        return out
    t2w = echo(P.IXI_T2["te"], 1.2, pd_gm_map)
    pdw = 3.7 * echo(P.IXI_PD_TE, 1.2, pd_gm_map)
    origin = (0.05, 0.05, 0.05)  # far from MNI's deep grey regions and midline
    t2s, pds, diag = P.residual_maps(frac, t2w, pdw, np.ones(n, bool), origin, (1e-3,) * 3)
    inner = (slice(15, 20),) * 3
    assert np.median(t2s[inner]) == pytest.approx(1.2, rel=0.03)
    gm = frac[k["gm"]] > .9
    assert np.median(pds[gm]) == pytest.approx(1.0, abs=0.01)
    assert np.median(pds[(slice(14, 16),) * 3]) < 0.9 * np.median(pds[(slice(19, 21),) * 3])
    assert np.median(t2s[frac[k["wm"]] > .9]) == pytest.approx(1.0, abs=0.01)
    assert np.median(pds[frac[k["wm"]] > .9]) == pytest.approx(1.0, abs=0.01)
    assert np.all(t2s[skin] == 1) and np.all(pds[skin] == 1)


def test_refine_csf_takes_the_csf_share_from_the_t2():
    """A synthetic brain: white matter, a deep CSF block (the calibration; its T2 at x1.1 the table, as IXI's MT makes
    it), and a sulcal region the segmentation calls 50 % CSF / 50 % grey matter whose T2 was made with 30 % CSF. The
    refined CSF share there is ~0.3, the grey matter takes the rest, and the deep CSF and white matter are unchanged."""
    from blochparty import tissuedb
    table = tissuedb.tissue_table(3.0)
    tr, te = P.IXI_T2["tr"], P.IXI_T2["te"]
    sig = {c: table[c].pd.mean * (1 - np.exp(-tr / table[c].t1.mean)) * np.exp(-te / table[c].t2.mean) for c in ("csf", "gm", "wm")}
    n = (20, 20, 20)
    k = {c: P.CLASSES.index(c) for c in P.CLASSES}
    frac = np.zeros((len(P.CLASSES),) + n, np.float32)
    frac[k["wm"]] = 1
    deep = (slice(2, 10),) * 3
    sulcus = (slice(12, 18), slice(12, 18), slice(12, 18))
    frac[k["wm"]][deep] = 0; frac[k["csf"]][deep] = 1
    frac[k["wm"]][sulcus] = 0; frac[k["csf"]][sulcus] = .5; frac[k["gm"]][sulcus] = .5
    t2w = np.full(n, sig["wm"], np.float32)
    t2w[deep] = 1.1 * sig["csf"]
    t2w[sulcus] = .3 * 1.1 * sig["csf"] + .7 * sig["gm"]
    out, diag = P.refine_csf(frac, t2w * 2.7, np.ones(n, bool))
    assert np.median(out[k["csf"]][sulcus]) == pytest.approx(.3, abs=.01)
    assert np.median(out[k["gm"]][sulcus]) == pytest.approx(.7, abs=.01)
    assert np.allclose(out[k["csf"]][deep], 1) and np.allclose(out.sum(0), 1, atol=1e-5)
    assert diag["csf_over_wm_deep"] == pytest.approx(1.1 * sig["csf"] / sig["wm"], rel=1e-3)


def test_orbit_vitreous_finds_the_globes_only():
    """Two vitreous-like spheres (T2 4 x, T1 0.3 x white matter) in the orbits, a fat blob there (T1 bright), a
    small fluid pocket beside a globe and a CSF-like sphere outside the orbits: only the globes come out."""
    n, h = (90, 110, 120), 1e-3
    origin = (-60., 0., -80.)  # mm, (x, y, z)
    z = origin[2] + np.arange(n[0]); y = origin[1] + np.arange(n[1]); x = origin[0] + np.arange(n[2])
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    r1 = np.ones(n, np.float32); r2 = np.ones(n, np.float32)
    def ball(c, r):
        return (zz - c[2]) ** 2 + (yy - c[1]) ** 2 + (xx - c[0]) ** 2 <= r ** 2
    globes = ball((-33, 55, -39), 11) | ball((33, 55, -39), 11)
    fat = ball((-35, 70, -20), 6)
    pocket = ball((-30, 35, -55), 3)
    outside = ball((0, 60, -40), 6)  # between the orbits
    for m in (globes, pocket, outside):
        r1[m], r2[m] = .3, 4.
    r1[fat], r2[fat] = 1.4, 3.5
    head = np.ones(n, bool)
    p = P.orbit_vitreous(r1, r2, np.zeros(n, np.float32), head, np.zeros(n, bool), origin, (h,) * 3)
    assert (p[globes] > .9).mean() > .95
    assert p[fat].max() < .05 and p[pocket].max() < .05 and p[outside].max() < .05
