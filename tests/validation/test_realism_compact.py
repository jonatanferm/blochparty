"""The compact real set (validation/realism/realism.py: a scan as its central 128 x 128 coil k-space in a .npz) feeds
the realism loaders exactly as a raw fastMRI file would: manifest(), manifest_key(), coil_kspace(), fastmri_volume(),
score.real_set's floors and score.noise_covariances, keyed by the fastMRI file name throughout."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation" / "realism"))
R = pytest.importorskip("realism")
SC = pytest.importorskip("score")


def _fake_set(root, n=128, coils=4, slices=3):
    rng = np.random.default_rng(0)
    img = np.zeros((n, n)); yy, xx = np.mgrid[:n, :n]
    img[(yy - n / 2) ** 2 + (xx - n / 2) ** 2 < (n / 3) ** 2] = 1.0  # a disc
    sens = rng.standard_normal((coils, n, n)) * 0 + np.linspace(0.5, 1.5, coils)[:, None, None]
    k = np.stack([R.centred_fft(sens * img, (-2, -1)) + 0.01 * (rng.standard_normal((coils, n, n)) + 1j * rng.standard_normal((coils, n, n)))
                  for _ in range(slices)]).astype(np.complex64)
    info = {"file": "file_brain_AXT2_200_0000001.h5", "coils": coils, "slices": slices, "field_T": 2.89, "model": "Skyra",
            "TR_ms": 6190.0, "TE_ms": 113.0, "compact": n}
    np.savez_compressed(root / "file_brain_AXT2_200_0000001.npz", kspace=k, info=json.dumps(info))
    (root / "manifest.json").write_text(json.dumps({"file_brain_AXT2_200_0000001.h5": {
        "contrast": "AXT2", "bytes": 1, "field_T": 2.89, "model": "Skyra", "kspace_shape": [slices, coils, 2 * n, n],
        "compact": "file_brain_AXT2_200_0000001.npz", "TR_ms": 6190.0, "TE_ms": 113.0, "flip_deg": 180.0, "echo_spacing_ms": 10.26}}))
    return k


def test_compact_real_set(tmp_path):
    k = _fake_set(tmp_path)
    paths = R.manifest(root=tmp_path)
    assert [p.name for p in paths] == ["file_brain_AXT2_200_0000001.npz"]
    assert R.manifest_key(paths[0]) == "file_brain_AXT2_200_0000001.h5" == R.manifest_key(tmp_path / "file_brain_AXT2_200_0000001.h5")
    low, info = R.coil_kspace(paths[0], 128)
    assert low.shape == (3, 4, 128, 128) and np.allclose(low, k) and info["model"] == "Skyra"
    low64, _ = R.coil_kspace(paths[0], 64)
    assert low64.shape == (3, 4, 64, 64)
    vol, info = R.fastmri_volume(paths[0])
    assert vol.shape == (3, 128, 128) and vol.max() > 0 and info["coils"] == 4
    with pytest.raises(ValueError):
        R.coil_kspace(paths[0], 256)
    cache = tmp_path / "cache"
    items, floor = SC.real_set(cache, tmp_path, range(3))
    assert len(items) == 3 and items[0][0] == "real:file_brain_AXT2_200_0000001" and list(floor) == ["file_brain_AXT2_200_0000001.h5"]
    covs = SC.noise_covariances(cache, tmp_path)
    assert list(covs) == ["file_brain_AXT2_200_0000001.h5"] and covs["file_brain_AXT2_200_0000001.h5"].shape == (4, 4)
    assert np.allclose(np.diag(covs["file_brain_AXT2_200_0000001.h5"]).real, 2e-4, rtol=0.3)  # 2 sigma^2 per complex sample (sigma 0.01), the orthonormal FFT keeps it
    again = SC.noise_covariances(cache, tmp_path)  # cached, incremental
    assert np.allclose(again["file_brain_AXT2_200_0000001.h5"], covs["file_brain_AXT2_200_0000001.h5"])
