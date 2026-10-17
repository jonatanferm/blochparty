"""Opt-in cross-simulator check (``uv run pytest -m crosscheck``, needs the ``mrzero`` extra):
the harness in validation/compare_mrzero.py on a 16 x 16 phantom.

Tolerances, from the full 32 x 32 runs in validation/RESULTS.md:
* MRzero vs blochparty < 1e-4 (max relative k-space difference). MRzero's float32 and its
  merging of PDG states whose k-vectors agree only to float32 precision give up to ~4e-5
  on the RF-spoiled GRE; everything else is at 1e-6..1e-5.
* blochparty vs the exact matrix-exponential reference < 2e-6: the second-order RF
  splitting error of the 0.2 us pulses (6e-7 on bSSFP, shrinking as subdivide_rf(k)**-2).
"""

import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.crosscheck, pytest.mark.slow]  # slow: excluded by -m 'not slow' too
pytest.importorskip("pypulseq")
pytest.importorskip("MRzeroCore")
sys.path.insert(0, str(Path(__file__).parents[1] / "validation"))
import compare_mrzero as cm  # noqa: E402

from blochparty.pulseq import read_pulseq  # noqa: E402

N = 16


@pytest.fixture(scope="module")
def maps():
    return cm.phantom_maps(N)


@pytest.mark.parametrize("make", list(cm.SEQUENCES.values()), ids=lambda f: f.__name__)
def test_blochparty_matches_mrzero(make, maps, tmp_path):
    pseq = make(N)
    path = tmp_path / "seq.seq"
    pseq.write(str(path))
    s_bp = cm.run_blochparty(path, maps)
    s_mr, k_mr = cm.run_mrzero(path, cm.mr0_data(maps))
    c, dk, nrmse = cm.metrics(s_mr, s_bp, N, epi_order=make is cm.epi)
    assert abs(c + 1j) < 1e-4  # the convention factor, fitted
    assert dk < 1e-4 and nrmse < 1e-4, (dk, nrmse)
    # same ADC ordering and trajectory as pypulseq
    assert cm.rel_max(k_mr, pseq.calculate_kspace()[0].T) < 1e-4

    few = cm.sample_voxels(maps, count=12)
    phantom, coils = cm.bp_phantom(few)
    ref = cm.exact_signal(read_pulseq(path, convention="mrzero"), phantom, coils.to(torch.complex128))
    assert cm.rel_max(cm.run_blochparty(path, few), ref) < 2e-6
