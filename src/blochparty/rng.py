"""Counter-based random numbers shared by every engine: Philox4x32-10 (Salmon et al., SC 2011).

A draw is a pure function of (seed, spin, segment, block), so the PyTorch reference and the GPU
kernel produce the same random walk for the same spin, whatever the evaluation order; results
can be compared spin by spin, not only statistically. ``normals`` gives 8 standard normals per
(spin, segment) by Box-Muller from two Philox blocks. Reference implementation: numpy uint64,
the same rounds as the Metal kernel's uint32 arithmetic (``metal/__init__.py``).
"""

from __future__ import annotations

import numpy as np

M0, M1 = 0xD2511F53, 0xCD9E8D57
W0, W1 = 0x9E3779B9, 0xBB67AE85
_MASK = 0xFFFFFFFF


def philox(c0, c1, c2, c3, k0: int, k1: int) -> tuple[np.ndarray, ...]:
    """Philox4x32-10 of counters (uint32 arrays, broadcast) and key (k0, k1): four uint32 arrays."""
    c = [np.asarray(x, dtype=np.uint64) & _MASK for x in (c0, c1, c2, c3)]
    c = np.broadcast_arrays(*c)
    c0, c1, c2, c3 = (x.copy() for x in c)
    k0, k1 = np.uint64(k0 & _MASK), np.uint64(k1 & _MASK)
    m0, m1, mask, sh = np.uint64(M0), np.uint64(M1), np.uint64(_MASK), np.uint64(32)
    for _ in range(10):
        p0, p1 = m0 * c0, m1 * c2  # < 2^64: exact in uint64
        hi0, lo0 = p0 >> sh, p0 & mask
        hi1, lo1 = p1 >> sh, p1 & mask
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0, k1 = (k0 + np.uint64(W0)) & mask, (k1 + np.uint64(W1)) & mask
    return c0, c1, c2, c3


def uniform(x: np.ndarray) -> np.ndarray:
    """uint32 -> (0, 1), never 0 (so log is finite)."""
    return (x.astype(np.float64) + 0.5) * 2.0**-32


def normals(seed: int, spin: np.ndarray, segment: int, sub: int = 0) -> np.ndarray:
    """(len(spin), 8) standard normals for (seed, spin, segment, sub-step): Box-Muller on two
    Philox blocks (counter: spin, segment, sub, block; key: the seed's low and high 32 bits).
    sub 0 is the free walk; restricted walks use sub-steps 1, 2, ...; segment 2^32 - 1 with
    sub 2^32 - 1 is the start offsets (``uniforms``)."""
    spin = np.asarray(spin, dtype=np.uint64)
    out = []
    for block in (0, 1):
        x = philox(spin, np.uint64(segment & _MASK), np.uint64(sub & _MASK), np.uint64(block),
                   seed & _MASK, (seed >> 32) & _MASK)
        for a, b in ((x[0], x[1]), (x[2], x[3])):
            r = np.sqrt(-2.0 * np.log(uniform(a)))
            th = 2.0 * np.pi * uniform(b)
            out += [r * np.cos(th), r * np.sin(th)]
    return np.stack(out, -1)


START = 0xFFFFFFFF  # the (segment, sub) counter of the start offsets


def uniforms(seed: int, spin: np.ndarray) -> np.ndarray:
    """(len(spin), 4) uniforms in (0, 1) for a walker's start within its compartment."""
    x = philox(np.asarray(spin, dtype=np.uint64), np.uint64(START), np.uint64(START), np.uint64(0),
               seed & _MASK, (seed >> 32) & _MASK)
    return np.stack([uniform(v) for v in x], -1)
