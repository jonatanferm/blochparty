"""Minimal reader for Siemens raw data (twix VD / VE: multi-measurement files), imaging ADCs only.

Layout (as in mapVBVD / twixtools): a file header (uint32 header size, uint32 measurement count,
then 152-byte entries: uint32 measId, uint32 fileId, uint64 offset, uint64 length, 64-byte patient
and protocol names); each measurement starts with a uint32 header length and the text header, then
scans: a 192-byte measurement data header (MDH: DMA length in the low 25 bits of the first uint32,
evaluation-info mask at byte 40, samples at 48, channels at 50, loop counters at 52), and per
channel a 32-byte channel header followed by complex64 samples. ACQEND (eval bit 0) ends the
measurement; SYNCDATA (bit 5) and adjustment scans (noise adjust bit 25, phase-correction bit 21,
reflect-free bits we do not use) are skipped.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

ACQEND, SYNCDATA, PHASCOR, NOISEADJ = 1 << 0, 1 << 5, 1 << 21, 1 << 25


def measurements(path) -> list[tuple[int, int]]:
    with open(path, "rb") as f:
        _, n = struct.unpack("<II", f.read(8))
        out = []
        for i in range(n):
            f.seek(8 + 152 * i)
            _, _, off, length = struct.unpack("<IIQQ", f.read(24))
            out.append((off, length))
    return out


def header_text(path, meas: int = -1) -> str:
    off, _ = measurements(path)[meas]
    with open(path, "rb") as f:
        f.seek(off)
        (hlen,) = struct.unpack("<I", f.read(4))
        return f.read(hlen - 4).decode("latin-1")


def read_adcs(path, meas: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """(n_adc, channels, samples) complex64 imaging ADCs in acquisition order, and their eval masks."""
    off, length = measurements(path)[meas]
    data, masks = [], []
    with open(path, "rb") as f:
        f.seek(off)
        (hlen,) = struct.unpack("<I", f.read(4))
        pos, end = off + hlen, off + length
        while pos + 192 <= end:
            f.seek(pos)
            mdh = f.read(192)
            dma = struct.unpack_from("<I", mdh, 0)[0] & 0x1FFFFFF
            (mask,) = struct.unpack_from("<Q", mdh, 40)
            ns, nc = struct.unpack_from("<HH", mdh, 48)
            if mask & ACQEND or dma == 0:
                break
            if not mask & (SYNCDATA | PHASCOR | NOISEADJ) and nc > 0:
                raw = np.frombuffer(f.read(nc * (32 + 8 * ns)), dtype=np.uint8).reshape(nc, 32 + 8 * ns)
                data.append(raw[:, 32:].copy().view(np.complex64))
                masks.append(mask)
            pos += dma
    return np.stack(data), np.array(masks)


__all__ = ["header_text", "measurements", "read_adcs"]


if __name__ == "__main__":
    import sys

    p = Path(sys.argv[1])
    d, m = read_adcs(p)
    print(p.name, d.shape, np.unique(m))
