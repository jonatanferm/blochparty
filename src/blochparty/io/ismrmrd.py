"""Simulated raw data as ISMRMRD HDF5, with what it takes to make it again (ROADMAP M5).

``write_dataset`` stores the signal the way a scanner's converter would: one ISMRMRD acquisition
per run of consecutive ADC samples (complex64, coils x samples), with the k-space of every sample
(``sequence.kspace_trajectory``, cycles/m, x y z) as its trajectory, Cartesian encode indices
where the sequence's ``meta`` has them, and an XML header (field, matrix, FOV, TR/TE, channels).
Readers of ISMRMRD (Gadgetron, ismrmrd-python, BART) see an ordinary dataset; they ignore the
``/provenance`` group, which holds:

* ``sequence/``: the segment arrays (dt, rf, grad, adc_phase, flags, b0_offset, rf_freq),
  bit-exact, and ``meta.json`` (tensors, arrays, tuples and the builder's event records tagged,
  so ``read_dataset`` rebuilds an equal ``Sequence``);
* ``sequence.seq``: the Pulseq file (``pulseq.write_pulseq``) when the sequence can be written
  (ideal spoilers are dropped and the attribute ``seq_note`` says so; otherwise ``seq_error``);
* ``phantom.json`` (a ``DigitalPhantom``'s field, table, ``to_phantom`` options and every draw
  of ``sample_head``, or any JSON-able description), ``coils.json``, ``noise.json``;
* attributes: engine, seed, git commit (and whether the tree was dirty), package versions.

Requires the optional extra ``io`` (``ismrmrd``, ``h5py``)."""
from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np
import torch
from torch import Tensor

from . import sequence as _sequence
from .sequences import builder as _events
from .constants import GAMMA_BAR
from .sequence import Sequence, kspace_trajectory

TICK = 2.5e-3  # [s] ISMRMRD time-stamp unit
_ARRAYS = ("dt", "rf", "grad", "adc_phase", "b0_offset", "rf_freq")
_FLAGS = ("is_rf", "adc", "spoil")


def _imports():
    try:
        import h5py
        import ismrmrd
    except ImportError as err:
        raise ImportError("write_dataset/read_dataset need the optional extra: uv sync --extra io") from err
    return h5py, ismrmrd


def _encode(x):
    """JSON-able form of ``meta`` values: tensors, arrays, tuples and dataclasses tagged."""
    if isinstance(x, Tensor):
        x = x.detach().cpu()
        return {"__tensor__": str(x.dtype).removeprefix("torch."), "shape": list(x.shape),
                "data": x.reshape(-1).tolist()}
    if isinstance(x, np.ndarray):
        return {"__ndarray__": x.dtype.str, "shape": list(x.shape), "data": x.reshape(-1).tolist()}
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, tuple):
        return {"__tuple__": [_encode(v) for v in x]}
    if isinstance(x, list):
        return [_encode(v) for v in x]
    if isinstance(x, dict):
        if not all(isinstance(k, str) for k in x):
            return {"__items__": [[_encode(k), _encode(v)] for k, v in x.items()]}
        return {k: _encode(v) for k, v in x.items()}
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {"__dataclass__": type(x).__qualname__,
                **{f.name: _encode(getattr(x, f.name)) for f in dataclasses.fields(x)}}
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return {"__repr__": repr(x)}


def _decode(x):
    if isinstance(x, list):
        return [_decode(v) for v in x]
    if not isinstance(x, dict):
        return x
    if "__tensor__" in x:
        return torch.tensor(x["data"], dtype=getattr(torch, x["__tensor__"])).reshape(x["shape"])
    if "__ndarray__" in x:
        return np.asarray(x["data"], dtype=np.dtype(x["__ndarray__"])).reshape(x["shape"])
    if "__tuple__" in x:
        return tuple(_decode(v) for v in x["__tuple__"])
    if "__items__" in x:
        return {_decode(k): _decode(v) for k, v in x["__items__"]}
    if "__dataclass__" in x:
        name = x["__dataclass__"]  # only the sequence's own event types (System, Trapezoid, RFPulse, ADC, ...)
        cls = getattr(_events, name, None) or getattr(_sequence, name, None)
        values = {k: _decode(v) for k, v in x.items() if k != "__dataclass__"}
        if dataclasses.is_dataclass(cls):
            return cls(**values)
        return {"__dataclass__": x["__dataclass__"], **values}
    return {k: _decode(v) for k, v in x.items()}


def _json_bytes(value) -> np.ndarray:
    return np.frombuffer(json.dumps(_encode(value)).encode(), dtype=np.uint8)


def _from_json_bytes(dataset) -> object:
    return _decode(json.loads(bytes(dataset[()]).decode()))


def _git():
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True, timeout=10).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return "", False


def versions() -> dict:
    """Versions of the packages that make a simulation, and the platform."""
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("blochparty", "torch", "numpy", "pypulseq", "ismrmrd", "h5py"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


def _readouts(seq: Sequence) -> list[np.ndarray]:
    """ADC segment indices, split into runs of consecutive segments."""
    idx = np.flatnonzero(np.asarray(seq.adc, bool))
    return np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if idx.size else []


def _cartesian(seq: Sequence):
    """(matrix (z, y, x), per-sample (kz, ky, kx) indices) when the builder recorded them."""
    meta = seq.meta
    shape = meta.get("matrix", meta.get("kspace_shape"))
    if shape is None:
        return None, None
    shape = tuple(int(n) for n in shape)
    index = np.asarray(meta["kspace_index"]) if "kspace_index" in meta else np.arange(math.prod(shape))
    if index.size != seq.n_adc:
        return None, None
    return (1,) * (3 - len(shape)) + shape, np.stack(np.unravel_index(index, (1,) * (3 - len(shape)) + shape), -1)


def _header(ismrmrd, seq, field_T, channels, shape, name):
    xsd = ismrmrd.xsd
    meta = seq.meta
    fov = float(meta.get("fov", 0.0)) * 1e3
    nz, ny, nx = shape or (1, 1, seq.n_adc)
    thickness = meta.get("slice_thickness")
    z_mm = fov if nz > 1 else (float(thickness) * 1e3 if thickness else fov)
    space = xsd.encodingSpaceType(matrixSize=xsd.matrixSizeType(x=nx, y=ny, z=nz),
                                  fieldOfView_mm=xsd.fieldOfViewMm(x=fov, y=fov, z=z_mm))
    limit = lambda n: xsd.limitType(minimum=0, maximum=max(n - 1, 0), center=n // 2)
    encoding = xsd.encodingType(
        encodedSpace=space, reconSpace=space,
        encodingLimits=xsd.encodingLimitsType(kspace_encoding_step_0=limit(nx), kspace_encoding_step_1=limit(ny),
                                              kspace_encoding_step_2=limit(nz)),
        trajectory=xsd.trajectoryType.CARTESIAN if shape else xsd.trajectoryType.OTHER,
        echoTrainLength=int(meta["etl"]) if "etl" in meta else None)
    ms = lambda key: [float(meta[key]) * 1e3] if key in meta and meta[key] is not None else []
    params = xsd.sequenceParametersType(TR=ms("tr"), TE=ms("te"), TI=ms("ti"), sequence_type=name)
    user = xsd.userParametersType(userParameterString=[
        xsd.userParameterStringType(name="trajectory_units", value="1/m (cycles per metre); x, y, z"),
        xsd.userParameterStringType(name="simulator", value="blochparty")])
    header = xsd.ismrmrdHeader(
        experimentalConditions=xsd.experimentalConditionsType(H1resonanceFrequency_Hz=int(round(GAMMA_BAR * field_T))),
        acquisitionSystemInformation=xsd.acquisitionSystemInformationType(
            systemVendor="blochparty (simulated)", systemFieldStrength_T=float(field_T), receiverChannels=channels),
        encoding=[encoding], sequenceParameters=params, userParameters=user)
    return xsd.ToXML(header)


def write_dataset(path, signal, seq: Sequence, *, phantom=None, coils=None, noise=None, seed=None,
                  engine: str | None = None, field_T: float | None = None, name: str = "blochparty") -> Path:
    """Write ``signal`` ((C, n_adc) or (n_adc,), or a ``SimResult``) of ``seq`` to ``path`` as an
    ISMRMRD HDF5 file with a ``/provenance`` group (module docstring). ``phantom`` is a
    ``DigitalPhantom`` (its field strength is then the default ``field_T``, else 3 T) or any
    JSON-able description; ``coils`` and ``noise`` are descriptions, e.g. ``{"model":
    "birdcage", "n": 8}`` and ``{"sd": 1e-3, "seed": 4}``; ``seed`` and ``engine`` say how it
    was run. Overwrites ``path``."""
    h5py, ismrmrd = _imports()
    from .anatomy import DigitalPhantom
    from .sequences.pulseq import write_pulseq

    path = Path(path)
    sig = getattr(signal, "signal", signal)
    sig = sig.detach().cpu()
    if sig.ndim == 1:
        sig = sig[None]
    if sig.ndim != 2 or sig.shape[1] != seq.n_adc:
        raise ValueError(f"signal must be (coils, {seq.n_adc}) for this sequence, got {tuple(sig.shape)}")
    if isinstance(phantom, DigitalPhantom):
        field_T = phantom.field_T if field_T is None else field_T
        phantom = {"field_T": phantom.field_T, "options": phantom.options, "meta": phantom.meta,
                   "tissues": {k: dataclasses.asdict(p) for k, p in phantom.tissues.items()}}
    field_T = 3.0 if field_T is None else float(field_T)
    data = sig.to(torch.complex64).numpy()
    traj = kspace_trajectory(seq).to(torch.float32).numpy()
    shape, kidx = _cartesian(seq)
    tb = np.concatenate([[0.0], np.cumsum(seq.dt.detach().cpu().double().numpy())])
    adc_rank = np.cumsum(np.asarray(seq.adc, bool)) - 1  # sample number of each ADC segment
    if path.exists():
        path.unlink()
    dset = ismrmrd.Dataset(str(path), "dataset", create_if_needed=True)
    try:
        dset.write_xml_header(_header(ismrmrd, seq, field_T, data.shape[0], shape, name))
        runs = _readouts(seq)
        for n, run in enumerate(runs):
            samples = adc_rank[run]
            acq = ismrmrd.Acquisition.from_array(np.ascontiguousarray(data[:, samples]),
                                                 np.ascontiguousarray(traj[samples]))
            acq.scan_counter = n
            acq.sample_time_us = float(np.median(seq.dt.detach().cpu().double().numpy()[run])) * 1e6
            acq.acquisition_time_stamp = int(round(tb[run[0]] / TICK))
            for k, v in enumerate((1.0, 0.0, 0.0)):
                acq.read_dir[k], acq.phase_dir[k], acq.slice_dir[k] = v, (0.0, 1.0, 0.0)[k], (0.0, 0.0, 1.0)[k]
            if kidx is not None:
                kz, ky, kx = kidx[samples].T
                acq.idx.kspace_encode_step_1, acq.idx.kspace_encode_step_2 = int(ky[0]), int(kz[0])
                centre = np.flatnonzero(kx == shape[2] // 2)
                acq.center_sample = int(centre[0]) if centre.size else 0
                if kx.size > 1 and kx[-1] < kx[0]:
                    acq.set_flag(ismrmrd.ACQ_IS_REVERSE)
            if n == len(runs) - 1:
                acq.set_flag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
            dset.append_acquisition(acq)
    finally:
        dset.close()
    commit, dirty = _git()
    with h5py.File(path, "a") as f:
        prov = f.create_group("provenance")
        group = prov.create_group("sequence")
        for key in _ARRAYS:
            value = getattr(seq, key)
            if value is not None:
                group.create_dataset(key, data=value.detach().cpu().numpy())
        for key in _FLAGS:
            group.create_dataset(key, data=np.asarray(getattr(seq, key), bool))
        group.create_dataset("meta.json", data=_json_bytes(seq.meta), compression="gzip")
        with tempfile.TemporaryDirectory() as tmp:
            try:
                try:
                    write_pulseq(seq, Path(tmp) / "s.seq")
                except ValueError as err:
                    if "ideal spoilers" not in str(err):
                        raise
                    write_pulseq(seq, Path(tmp) / "s.seq", spoil="drop")
                    prov.attrs["seq_note"] = "ideal spoilers dropped (Pulseq has no ideal spoiler)"
                text = (Path(tmp) / "s.seq").read_bytes()
                prov.create_dataset("sequence.seq", data=np.frombuffer(text, np.uint8), compression="gzip")
            except (ValueError, NotImplementedError) as err:
                prov.attrs["seq_error"] = f"{type(err).__name__}: {err}"
        for key, value in (("phantom", phantom), ("coils", coils), ("noise", noise)):
            if value is not None:
                prov.create_dataset(f"{key}.json", data=_json_bytes(value), compression="gzip")
        prov.attrs["engine"] = engine or ""
        prov.attrs["seed"] = json.dumps(seed)
        prov.attrs["git_commit"] = commit
        prov.attrs["git_dirty"] = dirty
        prov.attrs["versions"] = json.dumps(versions())
        prov.attrs["field_T"] = field_T
    return path


def read_dataset(path) -> dict:
    """The contents of a ``write_dataset`` file: ``signal`` (C, n_adc) complex64 in sequence order,
    ``trajectory`` (n_adc, 3) [1/m], ``header`` (ISMRMRD XML), ``sequence`` (an equal ``Sequence``),
    ``seq_file`` (Pulseq text or None), ``phantom``/``coils``/``noise`` (decoded JSON or None) and
    ``provenance`` (the attributes)."""
    h5py, ismrmrd = _imports()
    dset = ismrmrd.Dataset(str(path), "dataset", create_if_needed=False)
    try:
        header = dset.read_xml_header()
        acqs = [dset.read_acquisition(i) for i in range(dset.number_of_acquisitions())]
    finally:
        dset.close()
    acqs.sort(key=lambda a: a.scan_counter)
    out = {"header": header.decode() if isinstance(header, bytes) else header,
           "signal": torch.from_numpy(np.concatenate([a.data for a in acqs], 1)) if acqs else None,
           "trajectory": torch.from_numpy(np.concatenate([a.traj for a in acqs], 0)) if acqs else None}
    with h5py.File(path, "r") as f:
        prov = f.get("provenance")
        if prov is None:
            return out
        group = prov["sequence"]
        arrays = {k: torch.from_numpy(group[k][()]) for k in _ARRAYS if k in group}
        flags = {k: [bool(v) for v in group[k][()]] for k in _FLAGS}
        out["sequence"] = Sequence(arrays["dt"], arrays["rf"], arrays["grad"], arrays["adc_phase"], flags["is_rf"],
                                   flags["adc"], flags["spoil"], _from_json_bytes(group["meta.json"]),
                                   arrays.get("b0_offset"), arrays.get("rf_freq"))
        out["seq_file"] = bytes(prov["sequence.seq"][()]).decode() if "sequence.seq" in prov else None
        for key in ("phantom", "coils", "noise"):
            out[key] = _from_json_bytes(prov[f"{key}.json"]) if f"{key}.json" in prov else None
        attrs = dict(prov.attrs)
        attrs["seed"] = json.loads(attrs["seed"])
        attrs["versions"] = json.loads(attrs["versions"])
        out["provenance"] = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in attrs.items()}
    return out
