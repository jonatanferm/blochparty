"""Bloch simulation on the Apple GPU (Metal), forward only.

The same physics as the CPU paths (``bloch.py``, ``fused/kernel.h``): per segment exact
free precession and relaxation (relaxation in the ``m += (m - m0) * expm1(-dt/T)`` form, which
float32 needs), Strang-split Rodrigues rotations on RF segments, B1+ scaling and per-segment
uniform B0 offsets; ADC samples at segment ends, demodulated by exp(-i psi); ideal spoils.

One GPU thread per spin steps through the segments with its magnetization in registers. The
coil sum at ADC samples has two implementations. With two or more coils (``bloch_chunk_mma``),
each simdgroup buffers 8 samples of its 32 spins and does the sum as a complex matrix product
(coils x spins) . (spins x samples) on 8x8 simdgroup matrices, demodulating afterwards; the cost
barely grows with the coil count (48^3 GRE: 0.41 s for 1 coil, 0.42 s for 8, 1.6 s for 64). With
one coil (``bloch_chunk``) each sample is summed with simd_sum. Either way the per-simdgroup sums
are reduced across the threadgroup and added with one atomic per threadgroup. Segments are
dispatched in chunks so no single command buffer runs for long. Metal has no double precision:
everything is float32.

Not (yet) supported here: the scanner field model (``fields.Scanner``) and gradients; use
``blochparty.fused.simulate_fused`` for those.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..bloch import SimResult
from ..phantom import Phantom
from ..sequence import Sequence

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define TWO_PI 6.283185307179586f
#define MAX_SG 8        // simdgroups per threadgroup (256 threads / 32)
#define MAX_VALS 64     // 2 x coils
#define TG_FLOATS 7680  // threadgroup buffer: MAX_SG * block * 2C floats (30 KB; + row indices < 32 KB)

inline float expm1_(float x) {
    if (fabs(x) < 1e-2f) return x * (1.0f + x * (0.5f + x * (1.0f / 6.0f + x * (1.0f / 24.0f))));
    return precise::exp(x) - 1.0f;
}

inline void relax(thread float& mx, thread float& my, thread float& mz, float t, float pd, float r1, float r2) {
    float e1 = expm1_(-t * r1), e2 = expm1_(-t * r2);
    mx += mx * e2;
    my += my * e2;
    mz += (mz - pd) * e1;
}


struct Spin { float px, py, pz, pd, r1, r2, db0, b1; };

// Factor cache for free-precession segments: runs of identical segments (every readout sample)
// reuse each spin's rotation and relaxation factors. All threads see the same segment, so the
// check never diverges.
struct Cache { float dt = -1.0f, gx = 0.0f, gy = 0.0f, gz = 0.0f, b0 = 0.0f, c = 1.0f, s = 0.0f, e1 = 0.0f, e2 = 0.0f; };

inline void step(constant float* q, int f, thread float& mx, thread float& my, thread float& mz, thread const Spin& p,
                 thread Cache& k) {
    const float dt = q[0];
    const float bz = p.px * q[3] + p.py * q[4] + p.pz * q[5] + p.db0 + q[7];  // off-resonance [Hz]
    if (f & 1) {  // RF: relax dt/2, rotate about the effective field, relax dt/2
        relax(mx, my, mz, 0.5f * dt, p.pd, p.r1, p.r2);
        const float w = -TWO_PI * dt;
        const float wx = w * p.b1 * q[1], wy = w * p.b1 * q[2], wz = w * bz;
        const float phi2 = wx * wx + wy * wy + wz * wz, phi = sqrt(phi2);
        float c, s1v, s2v;
        if (phi < 1e-3f) {
            c = 1.0f - 0.5f * phi2;
            s1v = 1.0f - phi2 / 6.0f;
            s2v = 0.5f - phi2 / 24.0f;
        } else {
            const float sh = precise::sin(0.5f * phi) / phi;
            c = precise::cos(phi);
            s1v = precise::sin(phi) / phi;
            s2v = 2.0f * sh * sh;  // (1 - cos phi) / phi^2 without cancellation
        }
        const float dot = wx * mx + wy * my + wz * mz;
        const float nx = c * mx + s1v * (wy * mz - wz * my) + s2v * wx * dot;
        const float ny = c * my + s1v * (wz * mx - wx * mz) + s2v * wy * dot;
        const float nz = c * mz + s1v * (wx * my - wy * mx) + s2v * wz * dot;
        mx = nx; my = ny; mz = nz;
        relax(mx, my, mz, 0.5f * dt, p.pd, p.r1, p.r2);
    } else {  // free precession about z, exact relaxation
        if (dt != k.dt || q[3] != k.gx || q[4] != k.gy || q[5] != k.gz || q[7] != k.b0) {
            const float phi = -TWO_PI * dt * bz;
            k.c = precise::cos(phi); k.s = precise::sin(phi);
            k.e1 = expm1_(-dt * p.r1); k.e2 = expm1_(-dt * p.r2);
            k.dt = dt; k.gx = q[3]; k.gy = q[4]; k.gz = q[5]; k.b0 = q[7];
        }
        const float nx = k.c * mx - k.s * my, ny = k.s * mx + k.c * my;
        mx = nx + nx * k.e2; my = ny + ny * k.e2;
        mz += (mz - p.pd) * k.e1;
    }
}

inline Spin load_spin(constant float* spins, uint n, uint j) {
    return Spin{spins[j], spins[n + j], spins[2 * n + j], spins[3 * n + j], spins[4 * n + j], spins[5 * n + j],
                spins[6 * n + j], spins[7 * n + j]};
}

inline void flush(threadgroup float* partial, threadgroup int* block_rows, device atomic_float* sig, uint nb, uint nv,
                  uint nsg, uint tid, uint ntg) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = tid; k < nb * nv; k += ntg) {  // item k: sample k / nv, value k % nv
        const uint b = k / nv, v = k % nv;
        float t = 0.0f;
        for (uint g = 0; g < nsg; ++g) t += partial[(g * (TG_FLOATS / (MAX_SG * nv)) + b) * nv + v];
        atomic_fetch_add_explicit(&sig[block_rows[b] * nv + v], t, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void bloch_chunk(device float* M            [[buffer(0)]],   // (3, N) magnetization, in/out
                        constant float* spins      [[buffer(1)]],   // (8, N) px py pz pd r1 r2 db0 b1
                        constant float* sens       [[buffer(2)]],   // (2C, N) per coil: real, imag
                        constant float* segs       [[buffer(3)]],   // (S, 8) dt rfx rfy gx gy gz psi b0
                        constant int* flags        [[buffer(4)]],   // (S,) 1 rf, 2 adc, 4 spoil
                        constant int* adc_row      [[buffer(5)]],   // (S,) signal row of an ADC segment
                        device atomic_float* sig   [[buffer(6)]],   // (n_adc, C, 2)
                        constant uint& n           [[buffer(7)]],
                        constant uint& n_coils     [[buffer(8)]],
                        constant uint& s0          [[buffer(9)]],
                        constant uint& s1          [[buffer(10)]],
                        uint i    [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg   [[simdgroup_index_in_threadgroup]],
                        uint nsg  [[simdgroups_per_threadgroup]],
                        uint tid  [[thread_index_in_threadgroup]],
                        uint ntg  [[threads_per_threadgroup]]) {
    // ADC samples are buffered per simdgroup in threadgroup memory and reduced a block at a time:
    // two barriers per block of samples instead of two per sample, and the final sums are spread
    // over all threads of the group.
    threadgroup float partial[TG_FLOATS];
    threadgroup int block_rows[TG_FLOATS / (MAX_SG * 2)];
    const bool active = i < n;
    const uint j = active ? i : 0;
    float mx = active ? M[j] : 0.0f, my = active ? M[n + j] : 0.0f, mz = active ? M[2 * n + j] : 0.0f;
    const Spin p = load_spin(spins, n, j);
    const uint nv = 2 * n_coils;
    const uint block = TG_FLOATS / (MAX_SG * nv);  // samples buffered per flush
    uint nb = 0;
    Cache cache;

    for (uint s = s0; s < s1; ++s) {
        constant float* q = segs + 8 * s;
        const int f = flags[s];
        step(q, f, mx, my, mz, p, cache);
        if (f & 2) {  // ADC sample: simdgroup sums into the block buffer, flushed when full
            const float cp = precise::cos(q[6]), sp = precise::sin(q[6]);
            for (uint v = 0; v < nv; v += 2) {
                const float sr = active ? sens[v * n + j] : 0.0f, si = active ? sens[(v + 1) * n + j] : 0.0f;
                const float re = mx * sr - my * si, im = mx * si + my * sr;
                const float tr = simd_sum(re * cp + im * sp), ti = simd_sum(im * cp - re * sp);
                if (lane == 0) {
                    partial[(sg * block + nb) * nv + v] = tr;
                    partial[(sg * block + nb) * nv + v + 1] = ti;
                }
            }
            if (tid == 0) block_rows[nb] = adc_row[s];
            if (++nb == block) { flush(partial, block_rows, sig, nb, nv, nsg, tid, ntg); nb = 0; }
        }
        if (f & 4) { mx = 0.0f; my = 0.0f; }
    }
    if (nb > 0) flush(partial, block_rows, sig, nb, nv, nsg, tid, ntg);
    if (active) { M[j] = mx; M[n + j] = my; M[2 * n + j] = mz; }
}

// Many coils: the coil sum of a block of 8 ADC samples is a complex matrix product
// (coils x 32 spins) . (32 spins x 8 samples) per simdgroup, done on 8x8 simdgroup matrices:
//   P = Sr.Mr - Si.Mi,  Q = Si.Mr + Sr.Mi  (signal before demodulation: P + iQ).
// Demodulation by exp(-i psi) is the same for every spin, so it is applied after the sum.
#define MMA_REGION 768  // floats per simdgroup: Mr, Mi, -Mi (32 x 8 each); then P, Q of <= 6 coil tiles
#define MMA_TILES 6

inline void flush_mma(threadgroup float* buf, threadgroup float* mine, threadgroup const int* rows,
                      threadgroup const float* psi, device const float* sens, device atomic_float* sig, uint nb,
                      uint n_coils, uint n_tiles, uint n_pad, uint j0, uint nsg, uint tid, uint ntg) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 mr[4], mi[4], mn[4];
    for (uint k = 0; k < 4; ++k) {
        simdgroup_load(mr[k], mine + 64 * k, 8);
        simdgroup_load(mi[k], mine + 256 + 64 * k, 8);
        simdgroup_load(mn[k], mine + 512 + 64 * k, 8);
    }
    device const float* sr = sens;
    device const float* si = sens + 8 * n_tiles * n_pad;
    for (uint t0 = 0; t0 < n_tiles; t0 += MMA_TILES) {  // passes of up to 48 coils through the buffer
        const uint nt = min(n_tiles - t0, (uint)MMA_TILES);
        simdgroup_barrier(mem_flags::mem_threadgroup);  // the buffer's M (first pass) or last pass's sums are consumed
        if (t0 > 0) threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0; t < nt; ++t) {
            simdgroup_float8x8 P = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), Q = P, a, b;
            for (uint k = 0; k < 4; ++k) {
                const ulong2 origin = ulong2(j0 + 8 * k, 8 * (t0 + t));
                simdgroup_load(a, sr, n_pad, origin);
                simdgroup_load(b, si, n_pad, origin);
                simdgroup_multiply_accumulate(P, a, mr[k], P);
                simdgroup_multiply_accumulate(P, b, mn[k], P);
                simdgroup_multiply_accumulate(Q, b, mr[k], Q);
                simdgroup_multiply_accumulate(Q, a, mi[k], Q);
            }
            simdgroup_store(P, mine + 128 * t, 8);
            simdgroup_store(Q, mine + 128 * t + 64, 8);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint nc = min(n_coils - 8 * t0, 8 * nt);
        for (uint k = tid; k < nc * nb; k += ntg) {  // item k: coil k / nb of this pass, sample k % nb
            const uint c = k / nb, b = k % nb, off = 128 * (c / 8) + 8 * (c % 8) + b;
            float re = 0.0f, im = 0.0f;
            for (uint g = 0; g < nsg; ++g) { re += buf[g * MMA_REGION + off]; im += buf[g * MMA_REGION + off + 64]; }
            const float cp = precise::cos(psi[b]), sp = precise::sin(psi[b]);
            device atomic_float* out = sig + 2 * (rows[b] * n_coils + 8 * t0 + c);
            atomic_fetch_add_explicit(out, re * cp + im * sp, memory_order_relaxed);
            atomic_fetch_add_explicit(out + 1, im * cp - re * sp, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void bloch_chunk_mma(device float* M            [[buffer(0)]],   // (3, N) magnetization, in/out
                            constant float* spins      [[buffer(1)]],   // (8, N) px py pz pd r1 r2 db0 b1
                            device const float* sens   [[buffer(2)]],   // (2, 8T, N_pad): real, imag; zero-padded
                            constant float* segs       [[buffer(3)]],   // (S, 8) dt rfx rfy gx gy gz psi b0
                            constant int* flags        [[buffer(4)]],   // (S,) 1 rf, 2 adc, 4 spoil
                            constant int* adc_row      [[buffer(5)]],   // (S,) signal row of an ADC segment
                            device atomic_float* sig   [[buffer(6)]],   // (n_adc, C, 2)
                            constant uint& n           [[buffer(7)]],
                            constant uint& n_coils     [[buffer(8)]],
                            constant uint& s0          [[buffer(9)]],
                            constant uint& s1          [[buffer(10)]],
                            constant uint& n_tiles     [[buffer(11)]],  // coil tiles of 8
                            constant uint& n_pad       [[buffer(12)]],  // spins padded to the grid size
                            uint i    [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg   [[simdgroup_index_in_threadgroup]],
                            uint nsg  [[simdgroups_per_threadgroup]],
                            uint tid  [[thread_index_in_threadgroup]],
                            uint ntg  [[threads_per_threadgroup]]) {
    threadgroup float buf[MAX_SG * MMA_REGION];
    threadgroup int rows[8];
    threadgroup float psi[8];
    threadgroup float* mine = buf + sg * MMA_REGION;
    const bool active = i < n;
    const uint j = active ? i : 0;
    float mx = active ? M[j] : 0.0f, my = active ? M[n + j] : 0.0f, mz = active ? M[2 * n + j] : 0.0f;
    const Spin p = load_spin(spins, n, j);
    const uint j0 = i - lane;
    uint nb = 0;
    Cache cache;

    for (uint s = s0; s < s1; ++s) {
        constant float* q = segs + 8 * s;
        const int f = flags[s];
        step(q, f, mx, my, mz, p, cache);
        if (f & 2) {  // ADC sample: column nb of this simdgroup's (32 spins x 8 samples) block
            mine[8 * lane + nb] = mx;
            mine[256 + 8 * lane + nb] = my;
            mine[512 + 8 * lane + nb] = -my;
            if (tid == 0) { rows[nb] = adc_row[s]; psi[nb] = q[6]; }
            if (++nb == 8) { flush_mma(buf, mine, rows, psi, sens, sig, nb, n_coils, n_tiles, n_pad, j0, nsg, tid, ntg); nb = 0; }
        }
        if (f & 4) { mx = 0.0f; my = 0.0f; }
    }
    if (nb > 0) flush_mma(buf, mine, rows, psi, sens, sig, nb, n_coils, n_tiles, n_pad, j0, nsg, tid, ntg);
    if (active) { M[j] = mx; M[n + j] = my; M[2 * n + j] = mz; }
}
"""

_LIB = None
MAX_COILS = 32  # the direct (simd_sum) reduction; the matrix-product one has no limit
MMA_MIN_COILS = 2


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(_SOURCE)
    return _LIB


def available() -> bool:
    try:
        return torch.backends.mps.is_available() and _lib() is not None
    except Exception:
        return False


@torch.no_grad()
def simulate_metal(phantom: Phantom, seq: Sequence, coils: Tensor | None = None, *, m_init: Tensor | None = None,
                   chunk: int = 32768, group: int = 256, scanner=None, mma: bool | None = None) -> SimResult:
    """Forward simulation on the Apple GPU; same arguments and result as ``blochparty.simulate``
    (results come back on the CPU, float32). ``chunk`` segments per dispatch; ``group`` threads
    per threadgroup (must be 256). ``mma`` picks the coil reduction: matrix products on simdgroup
    matrices (default from ``MMA_MIN_COILS`` coils) or one simd_sum per coil and sample."""
    if scanner is not None:
        raise NotImplementedError("the scanner field model is not implemented on Metal; use simulate_fused")
    if phantom.bound is not None:
        raise NotImplementedError("magnetization transfer (Phantom.bound) is not implemented here yet; use blochparty.simulate")

    dev, f32 = torch.device("mps"), torch.float32
    n = len(phantom)
    b1 = phantom.b1 if getattr(phantom, "b1", None) is not None else torch.ones(n)
    spins = torch.stack([phantom.pos[:, 0], phantom.pos[:, 1], phantom.pos[:, 2], phantom.pd, 1 / phantom.t1,
                         1 / phantom.t2, phantom.db0, b1.to(phantom.pd.dtype)]).to(dev, f32).contiguous()
    if coils is None:
        coils = torch.ones(n, 1, dtype=torch.complex64)
    n_coils = coils.shape[1]
    mma = n_coils >= MMA_MIN_COILS if mma is None else mma
    if not mma and n_coils > MAX_COILS:
        raise ValueError(f"at most {MAX_COILS} coils with mma=False")
    if group != 256:
        raise ValueError("the Metal kernel's sample buffer is sized for threadgroups of 256")
    threads = math.ceil(n / group) * group
    if mma:  # (2, 8T, threads): real and imaginary planes, zero-padded to whole coil tiles and the grid
        n_tiles = math.ceil(n_coils / 8)
        sens = torch.zeros(2, 8 * n_tiles, threads, dtype=f32)
        sens[0, :n_coils, :n] = coils.real.T.float()
        sens[1, :n_coils, :n] = coils.imag.T.float()
        sens = sens.to(dev)
    else:
        sens = torch.stack([coils.real, coils.imag], -1).reshape(n, 2 * n_coils).T.to(dev, f32).contiguous()
    b0 = getattr(seq, "b0_offset", None)
    b0 = torch.zeros_like(seq.dt) if b0 is None else b0
    psi = torch.remainder(seq.adc_phase.double(), 2 * math.pi)  # RF spoiling makes phases large: wrap before float32
    segs = torch.stack([seq.dt.double(), seq.rf[:, 0].double(), seq.rf[:, 1].double(), seq.grad[:, 0].double(),
                        seq.grad[:, 1].double(), seq.grad[:, 2].double(), psi, b0.double()], 1).to(dev, f32).contiguous()
    adc = torch.tensor(seq.adc, dtype=torch.int32)
    flags = (torch.tensor(seq.is_rf, dtype=torch.int32) | (adc << 1) |
             (torch.tensor(seq.spoil, dtype=torch.int32) << 2)).to(dev)
    rows = (torch.cumsum(adc, 0) - 1).clamp_min(0).to(torch.int32).to(dev)
    n_adc = int(adc.sum())
    if m_init is None:
        M = torch.stack([torch.zeros(n), torch.zeros(n), phantom.pd.detach().cpu().float()])
    else:
        M = m_init.detach().cpu().float().T
    M = M.to(dev, f32).contiguous()
    sig = torch.zeros(max(n_adc, 1) * n_coils * 2, device=dev, dtype=f32)
    lib, S = _lib(), len(seq)
    for s0 in range(0, S, chunk):
        if mma:
            lib.bloch_chunk_mma(M, spins, sens, segs, flags, rows, sig, n, n_coils, s0, min(S, s0 + chunk), n_tiles,
                                threads, threads=threads, group_size=group)
        else:
            lib.bloch_chunk(M, spins, sens, segs, flags, rows, sig, n, n_coils, s0, min(S, s0 + chunk),
                            threads=threads, group_size=group)
    torch.mps.synchronize()
    sig = sig[: n_adc * n_coils * 2].cpu().view(n_adc, n_coils, 2)
    return SimResult(torch.complex(sig[..., 0], sig[..., 1]).T.contiguous(), M.cpu().T.contiguous())
