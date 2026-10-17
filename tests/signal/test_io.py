"""Dataset writer (``io.write_dataset``, ROADMAP M5): the ISMRMRD file read back with
ismrmrd-python gives the signal (complex64, exactly), the k-space of every sample and the Cartesian
encode indices, from which k-space is assembled as ``cartesian_kspace`` assembles it; the
provenance rebuilds an equal Sequence whose Pulseq file is the stored one, and keeps the phantom's
draws, seeds, engine and versions."""

from dataclasses import replace

import numpy as np
import pytest
import torch

import blochparty as bp
from blochparty.anatomy import DEFAULT_TISSUES, Anatomy, DigitalPhantom
from blochparty.builders import cartesian_kspace
from blochparty.io import read_dataset, write_dataset
from blochparty.pulseq import write_pulseq
from blochparty.sequence import kspace_trajectory

ismrmrd = pytest.importorskip("ismrmrd")
pytest.importorskip("pypulseq")

CASES = {
    "gre": lambda: bp.gre_2d(16, ramps=True, spoiling="rf", slice_thickness=5e-3, n_dummy=2),
    "tse (k-space index, ideal spoilers)": lambda: bp.tse_2d(16, etl=4, tr=0.4, slice_thickness=5e-3),
    "epi (reversed lines)": lambda: bp.epi_2d(16, dwell=5e-6, slice_thickness=5e-3),
}


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if hasattr(a, "__dataclass_fields__"):
        return type(a) is type(b) and all(equal(getattr(a, k), getattr(b, k)) for k in a.__dataclass_fields__)
    return a == b or (a != a and b != b)


@pytest.fixture(params=list(CASES), scope="module")
def written(request, tmp_path_factory):
    seq = CASES[request.param]()
    ph = bp.shepp_logan(16)
    ph = replace(ph, db0=torch.linspace(-20, 20, len(ph), dtype=torch.float64))
    result = bp.simulate(ph, seq, bp.birdcage_sensitivities(ph.pos, 4))
    path = tmp_path_factory.mktemp("ds") / "raw.h5"
    write_dataset(path, result, seq, coils={"model": "birdcage", "n": 4}, noise=None, seed=0, engine="core")
    return seq, result.signal, path


def test_signal_trajectory_and_kspace_survive(written):
    seq, signal, path = written
    back = read_dataset(path)
    assert torch.equal(back["signal"], signal.to(torch.complex64))
    traj = kspace_trajectory(seq).to(torch.float32)
    assert torch.equal(back["trajectory"].isnan(), traj.isnan())
    assert torch.equal(back["trajectory"].nan_to_num(), traj.nan_to_num())
    # a reader that knows only ISMRMRD: place every acquisition by its encode index
    dset = ismrmrd.Dataset(str(path), "dataset", create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    enc = header.encoding[0]
    nx, ny = enc.encodedSpace.matrixSize.x, enc.encodedSpace.matrixSize.y
    assert header.acquisitionSystemInformation.receiverChannels == 4 and (nx, ny) == (16, 16)
    k = np.zeros((4, ny, nx), np.complex64)
    for i in range(dset.number_of_acquisitions()):
        acq = dset.read_acquisition(i)
        line = acq.data if not acq.is_flag_set(ismrmrd.ACQ_IS_REVERSE) else acq.data[:, ::-1]
        start = nx // 2 - (
            acq.center_sample
            if not acq.is_flag_set(ismrmrd.ACQ_IS_REVERSE)
            else acq.number_of_samples - 1 - acq.center_sample
        )
        k[:, acq.idx.kspace_encode_step_1, start : start + acq.number_of_samples] = line
        last = acq.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
    dset.close()
    assert last
    assert np.array_equal(k, cartesian_kspace(signal, seq).to(torch.complex64).numpy())


def test_sequence_and_pulseq_file_survive(written, tmp_path):
    seq, _, path = written
    back = read_dataset(path)
    s = back["sequence"]
    for key in ("dt", "rf", "grad", "adc_phase"):
        assert torch.equal(getattr(s, key), getattr(seq, key)), key
    assert (s.is_rf, s.adc, s.spoil) == (seq.is_rf, seq.adc, seq.spoil)
    assert equal(s.meta, seq.meta)
    write_pulseq(s, tmp_path / "again.seq", spoil="drop")
    assert (tmp_path / "again.seq").read_text() == back["seq_file"]
    prov = back["provenance"]
    assert prov["engine"] == "core" and prov["seed"] == 0 and prov["field_T"] == 3.0
    assert "torch" in prov["versions"] and back["coils"] == {"model": "birdcage", "n": 4} and back["noise"] is None
    assert ("seq_note" in prov) == any(seq.spoil)


def test_digital_phantom_provenance(tmp_path):
    a = Anatomy({"wm": torch.full((4, 4, 4), 0.7, dtype=torch.float64)}, (0.002,) * 3, (0, 0, 0))
    head = DigitalPhantom(
        a,
        DEFAULT_TISSUES[1.5],
        1.5,
        {"seed": 3},
        {"sampler": "test", "draws": {"wm": {"t1": 0.9}}, "control": [[1.5, -2e-3]]},
    )
    seq = bp.gre_2d(4, ramps=True, spoiling="rf", n_dummy=0)
    ph = head.to_phantom(chi_to_b0=False)
    write_dataset(tmp_path / "h.h5", bp.simulate(ph, seq), seq, phantom=head, seed=11, engine="core")
    back = read_dataset(tmp_path / "h.h5")
    assert back["provenance"]["field_T"] == 1.5 and back["signal"].shape == (1, 16)
    assert back["phantom"]["meta"] == head.meta and back["phantom"]["options"] == {"seed": 3}
    assert back["phantom"]["tissues"]["wm"]["t1"]["citation"] == DEFAULT_TISSUES[1.5]["wm"].t1.citation
    assert f"{round(bp.GAMMA_BAR * 1.5)}" in back["header"]
    with pytest.raises(ValueError, match="signal must be"):
        write_dataset(tmp_path / "x.h5", torch.zeros(3), seq)


def test_unwritable_sequence_is_recorded_not_fatal(tmp_path):
    seq = bp.SequenceBuilder().rf(1e-3, flip_deg=30, n_steps=10).readout(8, 1e-5).build()  # no dead time
    write_dataset(tmp_path / "f.h5", bp.simulate(bp.shepp_logan(4), seq), seq)
    back = read_dataset(tmp_path / "f.h5")
    assert back["seq_file"] is None and "NotImplementedError" in back["provenance"]["seq_error"]
    assert "OTHER" in back["header"].upper() or "other" in back["header"]
    assert torch.equal(back["sequence"].dt, seq.dt)
