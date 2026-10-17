"""Bloch simulation on NVIDIA GPUs (CUDA), forward and adjoint: the Metal engine ported.

The physics is the Metal engine's (``blochparty.metal``), not a copy of it: the device functions of its shader
(``metal._SOURCE``, ``metal._ADJOINT_COMMON``: steps, relaxation, exchange, lineshapes, walks, the field model,
the reverse sweep) are turned into CUDA C++ by a few textual rules (address-space qualifiers dropped, Metal's
vector types and precise math mapped onto a small compatibility layer, ``_PRELUDE``) and compiled as they are.
The kernels are written here, because what differs between the two GPUs lives in them: a warp is a simdgroup of 32,
``__shfl_xor_sync`` sums replace ``simd_sum``, shared memory and ``atomicAdd`` replace threadgroup memory and Metal's
atomics. The coil sum of an ADC sample replaces Metal's 8 x 8 simdgroup matrix products (``bloch_chunk_mma``) by a
reduce-scatter across the warp: each lane forms its spin's products with eight coils (16 values), and five
butterfly steps (16 shuffles) leave every coil's sum in one lane, in full float32 (tensor cores would round the
products to TF32). Everything else, the per-spin state, the two-sum low parts, the NUFFT readouts, reused pulses,
the checkpointed adjoint with its in-order segment reduction, follows the Metal kernels line by line.

The Python driver is Metal's (``metal._forward``, ``metal._MetalSimulate``) with this module's backend: the
kernels take the same buffers in the same order, so the two engines cannot drift apart in what they compute
from a phantom and a sequence. ``simulate_cuda`` has ``simulate_metal``'s arguments and results (float32, on
the CPU).

Kernels are compiled at run time by NVRTC (the ``nvidia-cuda-nvrtc`` wheel PyTorch depends on, or a CUDA
toolkit), one kernel per program on its first use, for the current device's architecture, and cached on disk
(``BLOCHPARTY_CUDA_CACHE``, default ``~/.cache/blochparty/cuda``).
"""

from __future__ import annotations

import ctypes
import functools
import glob
import hashlib
import os
import re
import sys
from pathlib import Path

import torch

from .. import metal
from ..bloch import SimResult
from ..phantom import Phantom
from ..sequence import Sequence

MAX_COILS = 480  # the block buffer holds at least one sample of 2 x coils floats per warp (TG_FLOATS / MAX_SG)

# ---------------------------------------------------------------- the CUDA source

# Metal's built-in vector types and functions that the shared device code uses, for CUDA. Metal's names are
# renamed in the shared text (float3 -> bp_float3, ...), so CUDA's own vector types are left alone.
_PRELUDE = r"""
typedef unsigned int uint;

struct bp_int3;
struct bp_float2 {
    float x, y;
    bp_float2() = default;
    bp_float2(float a, float b) : x(a), y(b) {}
};
struct bp_float3 {
    float x, y, z;
    bp_float3() = default;
    explicit bp_float3(float s) : x(s), y(s), z(s) {}
    bp_float3(float a, float b, float c) : x(a), y(b), z(c) {}
    explicit bp_float3(const bp_int3& v);
};
struct bp_int3 {
    int x, y, z;
    bp_int3() = default;
    bp_int3(int a, int b, int c) : x(a), y(b), z(c) {}
    explicit bp_int3(const bp_float3& v) : x(int(v.x)), y(int(v.y)), z(int(v.z)) {}
};
inline bp_float3::bp_float3(const bp_int3& v) : x(float(v.x)), y(float(v.y)), z(float(v.z)) {}
struct bp_uint2 {
    uint x, y;
    bp_uint2() = default;
    bp_uint2(uint a, uint b) : x(a), y(b) {}
};
struct bp_uint4 {
    uint x, y, z, w;
    bp_uint4() = default;
    bp_uint4(uint a, uint b, uint c, uint d) : x(a), y(b), z(c), w(d) {}
};

inline bp_float3 operator+(bp_float3 a, bp_float3 b) { return bp_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline bp_float3 operator-(bp_float3 a, bp_float3 b) { return bp_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline bp_float3 operator-(bp_float3 a, float s) { return bp_float3(a.x - s, a.y - s, a.z - s); }
inline bp_float3 operator-(bp_float3 a) { return bp_float3(-a.x, -a.y, -a.z); }
inline bp_float3 operator*(bp_float3 a, float s) { return bp_float3(a.x * s, a.y * s, a.z * s); }
inline bp_float3 operator*(float s, bp_float3 a) { return bp_float3(s * a.x, s * a.y, s * a.z); }
inline bp_float3 operator/(bp_float3 a, bp_float3 b) { return bp_float3(a.x / b.x, a.y / b.y, a.z / b.z); }
inline bp_float3 operator/(bp_float3 a, float s) { return bp_float3(a.x / s, a.y / s, a.z / s); }
inline bp_float3& operator+=(bp_float3& a, bp_float3 b) { a.x += b.x, a.y += b.y, a.z += b.z; return a; }
inline bp_float3& operator-=(bp_float3& a, bp_float3 b) { a.x -= b.x, a.y -= b.y, a.z -= b.z; return a; }
inline float dot(bp_float3 a, bp_float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float length(bp_float3 a) { return sqrtf(dot(a, a)); }
inline bp_float3 floor(bp_float3 a) { return bp_float3(floorf(a.x), floorf(a.y), floorf(a.z)); }
inline bp_float3 min(bp_float3 a, bp_float3 b) { return bp_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }
inline bp_float3 max(bp_float3 a, bp_float3 b) { return bp_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
inline bp_int3 operator+(bp_int3 a, int s) { return bp_int3(a.x + s, a.y + s, a.z + s); }
inline bp_int3 min(bp_int3 a, bp_int3 b) { return bp_int3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
inline bp_uint2& operator+=(bp_uint2& a, bp_uint2 b) { a.x += b.x, a.y += b.y; return a; }
inline float clamp(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
"""

# ---- CUDA-only parts: warp reductions, the coil sums of one column (an ADC sample or a NUFFT grid column)
# into the block buffer, and its flush (as metal's flush: sums over the warps, one atomic per value).
_COMMON = r"""
#define FULL_MASK 0xffffffffu

__forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL_MASK, v, o);
    return v;
}
__forceinline__ int warp_min(int v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = min(v, __shfl_xor_sync(FULL_MASK, v, o));
    return v;
}
__forceinline__ int warp_max(int v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(FULL_MASK, v, o));
    return v;
}

// Sixteen values per lane summed over the warp, scattered: lane l ends with the sum of value (l >> 1) & 15
// (lanes l and l ^ 1 the same). Each butterfly step keeps half of the values and adds the partner's copy of
// that half (8 + 4 + 2 + 1 shuffles), the last adds the two halves of the warp.
__forceinline__ float reduce16(float* v, uint lane) {
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const bool up = (lane & 16u) != 0u;
        const float s = up ? v[k] : v[k + 8], keep = up ? v[k + 8] : v[k];
        v[k] = keep + __shfl_xor_sync(FULL_MASK, s, 16);
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const bool up = (lane & 8u) != 0u;
        const float s = up ? v[k] : v[k + 4], keep = up ? v[k + 4] : v[k];
        v[k] = keep + __shfl_xor_sync(FULL_MASK, s, 8);
    }
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const bool up = (lane & 4u) != 0u;
        const float s = up ? v[k] : v[k + 2], keep = up ? v[k + 2] : v[k];
        v[k] = keep + __shfl_xor_sync(FULL_MASK, s, 4);
    }
    {
        const bool up = (lane & 2u) != 0u;
        const float s = up ? v[0] : v[1], keep = up ? v[1] : v[0];
        v[0] = keep + __shfl_xor_sync(FULL_MASK, s, 2);
    }
    return v[0] + __shfl_xor_sync(FULL_MASK, v[0], 1);
}

// This lane's coil sensitivities: layout 0 (2C, N) real, imag per coil (metal's bloch_chunk); 1 (2, 8T, n_pad)
// real planes then imaginary planes, zero-padded (bloch_chunk_mma); mapped: sampled where the spin is now.
// The first eight coils are held in registers.
struct Sens {
    const float* sens;
    const float* maps;
    const float* mg;
    uint layout, n, n_pad, n_tiles, i, j;
    bool active, mapped;
    float r0[8], i0[8];
};

__forceinline__ void coil_at(const Sens& S, uint c, bp_float3 at, float& sr, float& si) {
    if (!S.active) { sr = si = 0.0f; return; }
    if (S.mapped) { sr = map_sample(S.maps, S.mg, 2u + 2u * c, at), si = map_sample(S.maps, S.mg, 3u + 2u * c, at); return; }
    if (S.layout) sr = S.sens[c * S.n_pad + S.i], si = S.sens[(8u * S.n_tiles + c) * S.n_pad + S.i];
    else sr = S.sens[2u * c * S.n + S.j], si = S.sens[(2u * c + 1u) * S.n + S.j];
}

// the block buffer: per warp `block` columns of nv = 2 C floats (metal's partial / block_rows)
struct Block {
    float* partial;
    int* rows;
    float* sig;
    uint block, nv, n_coils, nsg, tid, ntg, lane, sg, nb;
};

__forceinline__ void flush(Block& o) {
    __syncthreads();
    for (uint k = o.tid; k < o.nb * o.nv; k += o.ntg) {  // item k: column k / nv, value k % nv
        const uint b = k / o.nv, v = k % o.nv;
        float t = 0.0f;
        for (uint g = 0; g < o.nsg; ++g) t += o.partial[(g * o.block + b) * o.nv + v];
        atomicAdd(&o.sig[o.rows[b] * o.nv + v], t);
    }
    __syncthreads();
}

// Column of signal row `row`: this lane's spin contributes (ar + i ai) S_c to coil c (ar, ai already
// demodulated); the warp's sums go to the block buffer, flushed when full. Called by every thread of the
// block in step (the flush has barriers).
__forceinline__ void emit(Block& o, const Sens& S, float ar, float ai, int row, bp_float3 at) {
    float* dst = o.partial + (o.sg * o.block + o.nb) * o.nv;
    if (o.n_coils == 1u) {
        float sr = S.r0[0], si = S.i0[0];
        if (S.mapped) coil_at(S, 0u, at, sr, si);
        const float tr = warp_sum(ar * sr - ai * si), ti = warp_sum(ar * si + ai * sr);
        if (o.lane == 0u) dst[0] = tr, dst[1] = ti;
    } else {
        for (uint c0 = 0; c0 < o.n_coils; c0 += 8u) {
            float v[16];
#pragma unroll
            for (uint c = 0; c < 8u; ++c) {
                float sr = 0.0f, si = 0.0f;
                if (c0 + c < o.n_coils) {
                    if (c0 == 0u && !S.mapped) sr = S.r0[c], si = S.i0[c];
                    else coil_at(S, c0 + c, at, sr, si);
                }
                v[2 * c] = ar * sr - ai * si, v[2 * c + 1] = ar * si + ai * sr;
            }
            const float t = reduce16(v, o.lane);
            const uint idx = (o.lane >> 1) & 15u;
            if (!(o.lane & 1u) && 2u * c0 + idx < o.nv) dst[2u * c0 + idx] = t;
        }
    }
    if (o.tid == 0u) o.rows[o.nb] = row;
    if (++o.nb == o.block) { flush(o); o.nb = 0; }
}
"""

# The forward kernel, as metal's bloch_chunk (@MMA@ 0) and bloch_chunk_mma (@MMA@ 1: the zero-padded
# sensitivity layout, NUFFT readouts and reused pulses); placeholders as metal._KERNELS.
_KERNEL = r"""
extern "C" __global__ void __launch_bounds__(256) @NAME@(
        float* M, const float* spins, const float* sens, const float* segs, const int* flags, const int* adc_row,
        float* sig, float* MB, const float* pool, const float* frf, const float* line, const float* log_f,
        const float* conc, float* U, const float* modes, const int* nsub, float* P, const float* chol,
        const int* wsub, const float* geom, const float* mseg, const float* mspin, const float* maps,
        const float* mg, const uint* prm, float* BMv, const float* bmp, const float* fm, float* L
#if @MMA@
        , const int* run_of, const float* nu
#endif
        ) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x & 31u, sg = threadIdx.x >> 5,
               nsg = blockDim.x >> 5, tid = threadIdx.x, ntg = blockDim.x;
    const uint n = prm[0], n_coils = prm[1], s0 = prm[2], s1 = prm[3], n_tiles = prm[4], n_pad = prm[5], mt = prm[6],
               mode = prm[7], nj = prm[8], seed_lo = prm[9], seed_hi = prm[10], kind = prm[11], move = prm[12],
               diffuse = prm[13], mapped = prm[14], fmove = prm[15];
    __shared__ float partial[TG_FLOATS];
    __shared__ int block_rows[TG_FLOATS / (MAX_SG * 2)];
    const bool active = i < n;
    const uint j = active ? i : 0;
    State st;
    st.mx = active ? M[j] : 0.0f, st.my = active ? M[n + j] : 0.0f, st.mz = active ? M[2 * n + j] : 0.0f;
    if (active) st.lx = L[j], st.ly = L[n + j], st.lz = L[2 * n + j], st.lb = L[3 * n + j];  // dispatches continue exactly
    st.p = load_spin(spins, n, j);
    if (@V@ != 0 && mt) {  // inactive threads: an inert pool with unit rates (det A > 0, nothing to add)
        st.b = load_pool(pool, n, j), st.mb = active ? MB[j] : 0.0f;
        if (!active) st.p.pd = 0.0f, st.b.f = 0.0f;
    }
    if (@V@ != 0 && mode)  // inactive threads read spin 0's coefficients (finite, nothing added to the signal)
        for (uint c = 0; c < 7; ++c) st.conc.c[c] = conc[c * n + j];
    Walk wk;
    if (@DIFF@) {
        wk.x = active ? P[j] : 0.0f, wk.y = active ? P[n + j] : 0.0f, wk.z = active ? P[2 * n + j] : 0.0f;
        for (uint c = 0; c < 6; ++c) wk.l[c] = active && diffuse ? chol[c * n + j] : 0.0f;
        wk.seed = bp_uint2(seed_lo, seed_hi), wk.spin = j;
    }
    Move mv;
    if (@DIFF@) {
        mv.diffuse = diffuse != 0, mv.move = move != 0, mv.maps = mapped != 0;
        mv.db0t = st.p.db0, mv.b1t = st.p.b1;
        if (mv.move && active) {
            mv.v = bp_float3(mspin[j], mspin[n + j], mspin[2 * n + j]);
            mv.d = bp_float3(mspin[3 * n + j], mspin[4 * n + j], mspin[5 * n + j]);
        }
    }
    Geom gm;
    if (@DIFF@ && kind) {
        gm.kind = kind, gm.size = geom[j];
        gm.ax = bp_float3(geom[n + j], geom[2 * n + j], geom[3 * n + j]);
        gm.c = bp_float3(geom[4 * n + j], geom[5 * n + j], geom[6 * n + j]);
    }
    Modes md;
    for (uint m = 0; @V@ == 2 && m < nj; ++m) md.ur[m] = active ? U[2 * m * n + j] : 0.0f, md.ui[m] = active ? U[(2 * m + 1) * n + j] : 0.0f;
    BMS bm;
    if (@V@ == 3) {  // inactive threads: zero state and PD (inert)
        bm.bound = mt != 0;
        bm.hi = bm_load(BMv, 0, n, j), bm.lo = bm_load(BMv, BM_NC, n, j);
        if (!active) bm.hi = BMVec{}, bm.lo = BMVec{}, st.p.pd = 0.0f;
        bm.c0 = st.p.r1 * st.p.pd, bm.kex = 0.0f;
        for (uint p = 0; p < @NP@; ++p) {
            const float f = bmp[(5 * p) * n + j], k = bmp[(5 * p + 1) * n + j];
            bm.kin[p] = k, bm.kout[p] = k * f, bm.r1[p] = bmp[(5 * p + 2) * n + j], bm.r2[p] = bmp[(5 * p + 3) * n + j];
            bm.cz[p] = bm.r1[p] * f * st.p.pd, bm.dw[p] = TWO_PI * bmp[(5 * p + 4) * n + j], bm.kex += k + k * f;
        }
        bm.kfb = bm.kb = bm.r1b = bm.cb = 0.0f;
        if (mt) bm.kfb = st.b.k * st.b.f, bm.kb = st.b.k, bm.r1b = st.b.r1b, bm.cb = st.b.r1b * st.b.f * st.p.pd;
    }
    float &mx = st.mx, &my = st.my, &mz = st.mz;

    Block o;
    o.partial = partial, o.rows = block_rows, o.sig = sig, o.nv = 2 * n_coils, o.block = TG_FLOATS / (MAX_SG * o.nv);
    o.n_coils = n_coils, o.nsg = nsg, o.tid = tid, o.ntg = ntg, o.lane = lane, o.sg = sg, o.nb = 0;
    Sens S;
    S.sens = sens, S.maps = maps, S.mg = mg, S.layout = @MMA@, S.n = n, S.n_pad = n_pad, S.n_tiles = n_tiles;
    S.i = i, S.j = j, S.active = active, S.mapped = @DIFF@ && mapped > 1u;
#pragma unroll
    for (uint c = 0; c < 8u; ++c) {
        S.r0[c] = S.i0[c] = 0.0f;
        if (c < n_coils && !S.mapped) coil_at(S, c, bp_float3(0.0f), S.r0[c], S.i0[c]);
    }
#if @MMA@
    __shared__ int tgred[2 * MAX_SG];  // NUFFT readouts: per-warp window ends
    __shared__ int tgwin[2];
    const uint nuon = prm[16], pon = prm[17], pbase = prm[18];
#endif

    for (uint s = s0; s < s1; ++s) {
        const float* q = segs + 8 * s;
        const int f = flags[s];
#if @MMA@
        if (nuon && run_of[s] >= 0) {
            // A readout as a type-1 NUFFT (metal's bloch_chunk_mma): each spin spreads its decay terms onto the
            // oversampled grid, one emitted column per (cell, term), then jumps over the run in one free step.
            const int r = run_of[s];
            const float* rp = nu + NU_HEAD + 3 * r;
            const uint nr = uint(rp[0]), mrow = uint(rp[1]);
            const float shift = rp[2];
            const int ng = int(nu[0]), nadc = int(nu[4]);
            const uint w = uint(nu[1]), nterm = uint(nu[2]), wmax = uint(nu[5]);
            const float beta = nu[3];
            const float phi = TWO_PI * q[0] * offset(st.p, st.conc, 0u, q);
            const float turns = phi / TWO_PI;
            const float xg = (turns - floorf(turns)) * float(ng);
            const int base = int(floorf(xg - 0.5f * float(w))) + 1;
            int lo_b = active ? base : 0x7fffffff, hi_b = active ? base : -0x7fffffff;
            lo_b = warp_min(lo_b), hi_b = warp_max(hi_b);
            if (lane == 0) tgred[sg] = lo_b, tgred[MAX_SG + sg] = hi_b;
            __syncthreads();
            if (tid == 0) {
                int a = 0x7fffffff, b = -0x7fffffff;
                for (uint g = 0; g < nsg; ++g) a = min(a, tgred[g]), b = max(b, tgred[MAX_SG + g]);
                tgwin[0] = a, tgwin[1] = b;
            }
            __syncthreads();
            const int wb = tgwin[0], we = tgwin[1];
            __syncthreads();  // tgwin is read by all before any thread can reach the next run's writes
            if (wb > we || uint(we - wb) + w <= wmax) {  // all inactive, or a narrow window
                if (wb <= we) {
                    const uint ncol = (uint(we - wb) + w) * nterm;
                    const float ax = mx + st.lx, ay = my + st.ly;
                    const float cs = cosf(shift * phi), sn = sinf(shift * phi);
                    const float sr = ax * cs + ay * sn, si = ay * cs - ax * sn;  // (ax + i ay) e^{-i shift phi}
                    float er[MAX_TERMS];
                    for (uint m = 0; m < nterm; ++m) er[m] = expf(-nu[6 + m] * st.p.r2);
                    const bp_float3 at = bp_float3(st.p.px, st.p.py, st.p.pz);
                    for (uint col = 0; col < ncol; ++col) {
                        const uint cell = col / nterm, m = col % nterm;
                        const float z = 2.0f * (float(wb + int(cell)) - xg) / float(w);
                        const float kv = (active && fabsf(z) < 1.0f) ? expf(beta * (sqrtf(1.0f - z * z) - 1.0f)) * er[m] : 0.0f;
                        int u = (wb + int(cell)) % ng;
                        if (u < 0) u += ng;
                        emit(o, S, kv * sr, kv * si, nadc + (r * int(nterm) + int(m)) * ng + u, at);
                    }
                }
                advance<@V@, @DIFF@, @NP@>(st, md, bm, wk, gm, mv, segs, flags, frf, mrow, mt, line, log_f, mode, nsub, modes, nj, wsub, mseg, maps, mg, fm, fmove);
                s += nr - 1;
                if (flags[s] & 4) { mx = 0.0f; my = 0.0f; st.lx = 0.0f; st.ly = 0.0f; }
                continue;
            }
        }
        if (@V@ < 2 && pon && run_of[s] <= -2) {
            // A reused pulse (metal's bloch_chunk_mma): the spin's affine map of this pulse turned about z by this
            // occurrence's phase q[6]. U holds the maps (P, K, n_pad), K = R (R + 1) for R rows (x y z [b]).
            const uint pid = uint(-2 - run_of[s]);
            const uint R = (@V@ != 0 && mt) ? 4u : 3u;
            mx += st.lx, my += st.ly, mz += st.lz, st.lx = st.ly = st.lz = 0.0f;
            if (R == 4) st.mb += st.lb, st.lb = 0.0f;
            float sph, cph;
            sincosf(q[6], &sph, &cph);
            const float ux = cph * mx + sph * my, uy = cph * my - sph * mx;
            const float* D = U + pid * R * (R + 1) * n_pad + j;
            float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint r = 0; r < R; ++r) {
                const float* row = D + r * (R + 1) * n_pad;
                float a = row[R * n_pad] + row[0] * ux + row[n_pad] * uy + row[2 * n_pad] * mz;
                if (R == 4) a += row[3 * n_pad] * st.mb;
                v[r] = a;
            }
            two_sum(mx, st.lx, cph * v[0] - sph * v[1]), two_sum(my, st.ly, sph * v[0] + cph * v[1]);
            two_sum(mz, st.lz, v[2]);
            if (R == 4) two_sum(st.mb, st.lb, v[3]);
            s += uint(nu[pbase + pid]) - 1;
            if (flags[s] & 4) { mx = 0.0f; my = 0.0f; st.lx = 0.0f; st.ly = 0.0f; }
            continue;
        }
#endif
        advance<@V@, @DIFF@, @NP@>(st, md, bm, wk, gm, mv, segs, flags, frf, s, mt, line, log_f, mode, nsub, modes, nj, wsub, mseg, maps, mg, fm, fmove);
        if (f & 2) {  // ADC sample: m e^{-i psi} into the coil sums
            float sp, cp;
            sincosf(q[6], &sp, &cp);
            const float ax = mx + st.lx, ay = my + st.ly;
            emit(o, S, ax * cp + ay * sp, ay * cp - ax * sp, adc_row[s], bp_float3(st.p.px, st.p.py, st.p.pz));
        }
        if (f & 4) { mx = 0.0f; my = 0.0f; st.lx = 0.0f; st.ly = 0.0f; }
    }
    if (o.nb > 0) flush(o);
    if (active) {
        M[j] = mx; M[n + j] = my; M[2 * n + j] = mz;
        L[j] = st.lx, L[n + j] = st.ly, L[2 * n + j] = st.lz, L[3 * n + j] = st.lb;
        if (@V@ != 0 && mt) MB[j] = st.mb;
        for (uint m = 0; @V@ == 2 && m < nj; ++m) U[2 * m * n + j] = md.ur[m], U[(2 * m + 1) * n + j] = md.ui[m];
        if (@DIFF@) P[j] = wk.x, P[n + j] = wk.y, P[2 * n + j] = wk.z;
        if (@V@ == 3) bm_store(bm.hi, BMv, 0, n, j), bm_store(bm.lo, BMv, BM_NC, n, j);
    }
}
"""

_PULSE = r"""
// metal's pulse_maps: per spin the affine map of each distinct pulse, thread (p, j): pulse p, spin j; maps (P, K, T)
extern "C" __global__ void __launch_bounds__(256) @NAME@(float* maps, const float* spins, const float* segs, const int* flags,
        const float* pool, const float* frf, const float* line, const float* log_f, const int* pinfo, const uint* prm) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint n = prm[0], T = prm[1], P = prm[2], mt = prm[3];
    const uint pid = i / T, j = i % T;
    if (pid >= P || j >= n) return;
    const uint R = (@V@ != 0 && mt) ? 4u : 3u, K = R * (R + 1);
    const Spin p0 = load_spin(spins, n, j);
    Pool b = Pool{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (@V@ != 0 && mt) b = load_pool(pool, n, j);
    const int a = pinfo[2 * pid], len = pinfo[2 * pid + 1];
    Conc cc;
    for (uint col = 0; col <= R; ++col) {
        Spin p = p0;
        if (col < R) p.pd = 0.0f;
        float mx = col == 0 ? 1.0f : 0.0f, my = col == 1 ? 1.0f : 0.0f, mz = col == 2 ? 1.0f : 0.0f;
        float mb = col == 3 ? 1.0f : 0.0f, lx = 0.0f, ly = 0.0f, lz = 0.0f, lb = 0.0f;
        Cache k;
        ZCache zc;
        for (int s = a; s < a + len; ++s) {
            const float* q = segs + 8 * s;
            const float bz = offset(p, cc, 0u, q);
            if (@V@ != 0 && mt) step_mt(q, flags[s], bz, frf[s], mx, my, mz, mb, lx, ly, lz, lb, p, b, line, log_f, k, zc);
            else step(q, flags[s], bz, mx, my, mz, lx, ly, lz, p, k);
        }
        const float v[4] = {(mx - (col == 0 ? 1.0f : 0.0f)) + lx, (my - (col == 1 ? 1.0f : 0.0f)) + ly,
                            (mz - (col == 2 ? 1.0f : 0.0f)) + lz, (mb - (col == 3 ? 1.0f : 0.0f)) + lb};
        for (uint r = 0; r < R; ++r) maps[(pid * K + r * (R + 1) + col) * T + j] = v[r];
    }
}
"""

_SEG_REDUCE = r"""
// metal's seg_reduce: the per-warp segment partials of one interval summed over warps in order (two-sum)
extern "C" __global__ void seg_reduce(const float* GSEG, float* OUT, const uint* prm) {
    const uint t = blockIdx.x * blockDim.x + threadIdx.x;
    const uint G = prm[0], K = prm[1], s0 = prm[2], len = prm[3];
    if (t >= 8 * len) return;
    const uint slot = t / 8, k = t % 8;
    float hi = 0.0f, lo = 0.0f;
    for (uint gi = 0; gi < G; ++gi) two_sum(hi, lo, GSEG[(gi * K + slot) * 8 + k]);
    OUT[(s0 + slot) * 8 + k] = hi + lo;
}
"""

_REPLAY = r"""
// metal's bloch_replay: the forward over one interval from its checkpoint, m before every RBLOCK-th segment into R
extern "C" __global__ void __launch_bounds__(256) @NAME@(const float* CK, const float* spins, const float* segs, const int* flags,
        const float* pool, const float* frf, const float* line, const float* log_f, const float* conc, float* R,
        const uint* prm) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint n = prm[0], s0 = prm[2], s1 = prm[3], mt = prm[4], mode = prm[5];
    if (i >= n) return;
    const uint j = i;
    State st;
    st.p = load_spin(spins, n, j);
    if (@V@ != 0 && mt) st.b = load_pool(pool, n, j);
    if (@V@ != 0 && mode)
        for (uint c = 0; c < 7; ++c) st.conc.c[c] = conc[c * n + j];
    st.mx = CK[j], st.my = CK[n + j], st.mz = CK[2 * n + j], st.mb = CK[3 * n + j];
    st.lx = CK[4 * n + j], st.ly = CK[5 * n + j], st.lz = CK[6 * n + j], st.lb = CK[7 * n + j];
    for (uint s = s0; s < s1; ++s) {
        const float* q = segs + 8 * s;
        const int f = flags[s];
        if ((s - s0) % RBLOCK == 0) {
            float* r = R + (s - s0) / RBLOCK * rrows<@V@>() * n;
            r[j] = st.mx, r[n + j] = st.my, r[2 * n + j] = st.mz;
            if (rrows<@V@>() > 3) r[3 * n + j] = st.mb;
        }
        const float bz = offset(st.p, st.conc, @V@ == 0 ? 0u : mode, q);
        if (@V@ != 0 && mt)
            step_mt(q, f, bz, frf[s], st.mx, st.my, st.mz, st.mb, st.lx, st.ly, st.lz, st.lb, st.p, st.b, line, log_f,
                    st.cache, st.zc);
        else
            step(q, f, bz, st.mx, st.my, st.mz, st.lx, st.ly, st.lz, st.p, st.cache);
        if (f & 4) st.mx = st.my = st.lx = st.ly = 0.0f;
    }
}
"""

_ADJOINT = r"""
// metal's bloch_adjoint: the reverse sweep of one interval, a block of RBLOCK segments at a time recomputed from R
// into shared memory (each thread its own slots). NC: coils held in registers, padded with zero coils; 0: any
// count, from memory.
extern "C" __global__ void __launch_bounds__(256) @NAME@(const float* CK, const float* spins, const float* sens,
        const float* segs, const int* flags, const int* adc_row, const float* hb, const float* pool, const float* frf,
        const float* line, const float* log_f, const float* conc, float* R, float* LAM, float* GS, float* GC,
        float* GSEG, const uint* prm) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x & 31u, sg = threadIdx.x >> 5,
               nsg = blockDim.x >> 5, tid = threadIdx.x, group = blockIdx.x;
    // need_seg: 2 every segment's gradients (dt, grad, b0_offset requested), 1 those of RF segments (rf, rf_freq)
    const uint n = prm[0], nc = prm[1], s0 = prm[2], s1 = prm[3], mt = prm[4], mode = prm[5], need_seg = prm[6],
               need_sens = prm[7], need_adc = prm[8], K1 = prm[9];
    const uint nch = @NC@ > 0 ? @NC@ : nc;  // coil rows of sens and hb (padded with zero coils)
    const bool active = i < n;
    const uint j = active ? i : 0;
    State st;
    st.p = load_spin(spins, n, j);
    if (@V@ != 0 && mt) st.b = load_pool(pool, n, j);
    if (@V@ != 0 && mode)
        for (uint c = 0; c < 7; ++c) st.conc.c[c] = conc[c * n + j];
    if (!active) st.p.pd = 0.0f, st.b.f = 0.0f;
    float lx = active ? LAM[j] : 0.0f, ly = active ? LAM[n + j] : 0.0f, lz = active ? LAM[2 * n + j] : 0.0f;
    float lb = active ? LAM[3 * n + j] : 0.0f;
    Acc g = Acc{};
    RCache rc;
    __shared__ float tg[RBLOCK * 4 * 256];
    const uint sgid = group * nsg + sg;  // this warp's row of the segment partials
    float gcr[@NC@ + 1], gci[@NC@ + 1];
    for (uint c = 0; c < @NC@; ++c) gcr[c] = 0.0f, gci[c] = 0.0f;
    const uint nb = (s1 - s0 + RBLOCK - 1) / RBLOCK;
    for (int blk = int(nb) - 1; blk >= 0; --blk) {
        const uint b0 = s0 + uint(blk) * RBLOCK, b1 = min(b0 + RBLOCK, s1);
        const float* r = R + uint(blk) * rrows<@V@>() * n;
        st.mx = active ? r[j] : 0.0f, st.my = active ? r[n + j] : 0.0f, st.mz = active ? r[2 * n + j] : 0.0f;
        st.mb = (rrows<@V@>() > 3 && active) ? r[3 * n + j] : 0.0f;
        st.lx = st.ly = st.lz = st.lb = 0.0f;
        st.cache = Cache{}, st.zc = ZCache{};
        for (uint s = b0; s < b1; ++s) {
            const float* q = segs + 8 * s;
            const int f = flags[s];
            float* t = tg + (s - b0) * 4 * 256;
            t[tid] = st.mx, t[256 + tid] = st.my, t[512 + tid] = st.mz, t[768 + tid] = st.mb;
            const float bz = offset(st.p, st.conc, @V@ == 0 ? 0u : mode, q);
            if (@V@ != 0 && mt)
                step_mt(q, f, bz, frf[s], st.mx, st.my, st.mz, st.mb, st.lx, st.ly, st.lz, st.lb, st.p, st.b, line,
                        log_f, st.cache, st.zc);
            else
                step(q, f, bz, st.mx, st.my, st.mz, st.lx, st.ly, st.lz, st.p, st.cache);
            if ((f & 2) && need_adc && need_sens) {  // coil gradients from the sample (before the spoil)
                const float* h = hb + 2 * nch * uint(adc_row[s]);
                const float ox = st.mx + st.lx, oy = st.my + st.ly;
                if (@NC@ > 0) {
                    for (uint c = 0; c < @NC@; ++c) gcr[c] += h[2 * c] * ox + h[2 * c + 1] * oy, gci[c] += h[2 * c + 1] * ox - h[2 * c] * oy;
                } else if (active) {
                    for (uint c = 0; c < nc; ++c)
                        GC[2 * c * n + j] += h[2 * c] * ox + h[2 * c + 1] * oy, GC[(2 * c + 1) * n + j] += h[2 * c + 1] * ox - h[2 * c] * oy;
                }
            }
            if (f & 4) st.mx = st.my = st.lx = st.ly = 0.0f;
        }
        // the samples' source terms dL/dm = sum_c h_c conj(sens_c), coils outermost
        float ur[RBLOCK], ui[RBLOCK];
        int row[RBLOCK];
        bool any = false;
#pragma unroll
        for (uint k = 0; k < RBLOCK; ++k) {
            ur[k] = ui[k] = 0.0f;
            row[k] = (need_adc && b0 + k < b1 && (flags[b0 + k] & 2)) ? adc_row[b0 + k] : -1;
            any = any || row[k] >= 0;
        }
        if (any) {
            for (uint c = 0; c < nch; ++c) {
                const float a = sens[2 * c * n + j], b = sens[(2 * c + 1) * n + j];
#pragma unroll
                for (uint k = 0; k < RBLOCK; ++k) {
                    if (row[k] < 0) continue;
                    const float* h = hb + 2 * nch * uint(row[k]) + 2 * c;
                    ur[k] += h[0] * a + h[1] * b, ui[k] += h[1] * a - h[0] * b;
                }
            }
        }
        for (int s = int(b1) - 1; s >= int(b0); --s) {
            const float* q = segs + 8 * s;
            const int f = flags[s];
            const float* t = tg + (uint(s) - b0) * 4 * 256;
            const float x0 = t[tid], y0 = t[256 + tid], z0 = t[512 + tid], v0 = t[768 + tid];
            if (f & 4) lx = ly = 0.0f;
            if ((f & 2) && need_adc) {  // the sample: lambda += dL/dm
                const uint k = uint(s) - b0;
                lx += k == 0 ? ur[0] : (k == 1 ? ur[1] : (k == 2 ? ur[2] : ur[3]));
                ly += k == 0 ? ui[0] : (k == 1 ? ui[1] : (k == 2 ? ui[2] : ui[3]));
            }
            SegG sgv = SegG{};
            reverse_seg<@V@>(q, f, (@V@ != 0 && mt) ? frf[s] : 0.0f, x0, y0, z0, v0, st.p, st.b, st.conc, mode, mt != 0,
                             line, log_f, lx, ly, lz, lb, g, sgv, rc);
            // one partial per warp and segment (seg_reduce sums them in order)
            float* out = GSEG + (sgid * K1 + (uint(s) - s0)) * 8;
            if (need_seg == 2 || (need_seg == 1 && (f & 1))) {
                const float v[8] = {sgv.dt, sgv.rx, sgv.ry, sgv.gx, sgv.gy, sgv.gz, sgv.off, sgv.frf};
#pragma unroll
                for (uint k = 0; k < 8; ++k) {
                    const float tt = warp_sum(active ? v[k] : 0.0f);
                    if (lane == 0) out[k] = tt;
                }
            } else if (need_seg == 1 && lane < 8) {  // the partials buffer is reused: this segment contributes nothing
                out[lane] = 0.0f;
            }
            if (s % 64 == 0 && active) {
                acc_flush(g, GS, n, j);
                for (uint c = 0; need_sens && c < @NC@; ++c)
                    GC[2 * c * n + j] += gcr[c], GC[(2 * c + 1) * n + j] += gci[c], gcr[c] = gci[c] = 0.0f;
            }
        }
    }
    if (active) {
        acc_flush(g, GS, n, j);
        for (uint c = 0; need_sens && c < @NC@; ++c) GC[2 * c * n + j] += gcr[c], GC[(2 * c + 1) * n + j] += gci[c];
        LAM[j] = lx, LAM[n + j] = ly, LAM[2 * n + j] = lz, LAM[3 * n + j] = lb;
    }
}
"""


@functools.lru_cache(maxsize=None)
def _shared_source() -> str:
    """The Metal shader's device functions (forward and adjoint) as CUDA C++: without its Metal-only flushes
    and kernels; address spaces dropped (NVRTC's -default-device makes every function a device function);
    vector types renamed onto ``_PRELUDE``'s; precise math onto CUDA's (correctly rounded to within 2 ulp
    without fast math, which is not used)."""
    src = metal._SOURCE
    a, b = src.index("inline void flush(threadgroup"), src.index("// Many coils:")
    src = src[:a] + src[b:]
    src = src[:src.index("inline void flush_mma(")]
    adj = metal._ADJOINT_COMMON
    adj = adj[:adj.index("kernel void seg_reduce(")]
    text = (src + adj).replace("#include <metal_stdlib>", "").replace("using namespace metal;", "")
    text = re.sub(r"\b(thread|threadgroup|device)\s+(?=[A-Za-z_])", "", text)
    text = re.sub(r"\bconstant\s+(?=(?:float|int|uint)\b)", "const ", text)
    text = re.sub(r"\bprecise::(sin|cos|exp|log)\(", r"\1f(", text)
    # the scalar math of the shared code in float (CUDA's double versions run at 1/64 rate on consumer GPUs);
    # floor is also called on a vector (map_sample), the prelude's overload
    text = re.sub(r"\b(sin|cos|exp|log|sqrt|fabs|ceil)\(", r"\1f(", text)
    text = re.sub(r"\bfloor\(pos\)", "floorf(pos)", text)
    text = re.sub(r"\bmulhi\(", "__umulhi(", text)
    for t in ("float2", "float3", "int3", "uint2", "uint4"):
        text = re.sub(rf"\b{t}\b", f"bp_{t}", text)
    for define in ("MAX_TERMS", "NU_HEAD", "RBLOCK", "TG_FLOATS", "MAX_SG", "BM_NC"):
        assert f"#define {define}" in text, define
    return text


_VARIANTS = {"": (0, 0), "_full": (1, 0), "_gen": (2, 0), **{f"_bm{k}": (3, k) for k in range(1, metal.MAX_POOLS + 1)}}


def _variant(suffix: str) -> tuple[int, bool, int]:
    """(V, DIFF, NP) of a kernel suffix (``_full_diff``, ``_bm2``, ...), as metal._lib()."""
    diff = suffix.endswith("_diff")
    v, np_ = _VARIANTS[suffix[: -len("_diff")] if diff else suffix]
    return v, diff, np_


@functools.lru_cache(maxsize=None)
def kernel_source(name: str) -> str:
    """The CUDA program of one kernel (metal's kernel names): the prelude, the shared device code, the
    CUDA-only helpers and that kernel."""
    m = re.fullmatch(r"(bloch_chunk_mma|bloch_chunk|bloch_replay|bloch_adjoint|pulse_maps|seg_reduce)(\w*)", name)
    if m is None:
        raise AttributeError(f"no CUDA kernel {name!r}")
    family, rest = m[1], m[2]
    body = _PRELUDE + _shared_source() + _COMMON
    if family == "seg_reduce":
        if rest:
            raise AttributeError(f"no CUDA kernel {name!r}")
        return body + _SEG_REDUCE
    if family == "bloch_adjoint":
        mc = re.fullmatch(r"(\w*)_c(\d+)", rest)
        if mc is None or int(mc[2]) not in metal.ADJOINT_COILS or mc[1] not in ("", "_full"):
            raise AttributeError(f"no CUDA kernel {name!r}")
        v, nc = _VARIANTS[mc[1]][0], mc[2]
        return body + _ADJOINT.replace("@NAME@", name).replace("@V@", str(v)).replace("@NC@", nc)
    if family in ("bloch_replay", "pulse_maps"):
        if rest not in ("", "_full"):
            raise AttributeError(f"no CUDA kernel {name!r}")
        text = _REPLAY if family == "bloch_replay" else _PULSE
        return body + text.replace("@NAME@", name).replace("@V@", str(_VARIANTS[rest][0]))
    try:
        v, diff, np_ = _variant(rest)
    except KeyError:
        raise AttributeError(f"no CUDA kernel {name!r}") from None
    mma = family == "bloch_chunk_mma"
    return body + (_KERNEL.replace("@NAME@", name).replace("@MMA@", "1" if mma else "0").replace("@V@", str(v))
                   .replace("@DIFF@", "true" if diff else "false").replace("@NP@", str(np_)))


# ---------------------------------------------------------------- NVRTC and the driver API


def _load_nvrtc() -> ctypes.CDLL:
    """libnvrtc: PyTorch's (already loaded, or the nvidia-cuda-nvrtc wheel next to it), else a CUDA toolkit."""
    major = (torch.version.cuda or "12").split(".")[0]
    names = [f"libnvrtc.so.{major}", "libnvrtc.so"]
    for root in sys.path:
        names += sorted(glob.glob(os.path.join(root, "nvidia", "**", f"libnvrtc.so.{major}*"), recursive=True))
    names += sorted(glob.glob(f"/usr/local/cuda*/lib64/libnvrtc.so.{major}*"))
    for name in names:
        try:
            return ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
    raise OSError("NVRTC (libnvrtc) not found: install the nvidia-cuda-nvrtc wheel or a CUDA toolkit")


class _Driver:
    """NVRTC and the CUDA driver API through ctypes; modules per device."""

    def __init__(self):
        self.nvrtc = _load_nvrtc()
        self.nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p
        self.cuda = ctypes.CDLL("libcuda.so.1")
        self.cuda.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        major, minor = ctypes.c_int(), ctypes.c_int()
        self._nvrtc_check(self.nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)))
        self.nvrtc_version = (major.value, minor.value)
        self.functions: dict[tuple[int, str], ctypes.c_void_p] = {}
        self.modules: list[ctypes.c_void_p] = []

    def _nvrtc_check(self, code: int) -> None:
        if code != 0:
            raise RuntimeError(f"NVRTC error {code}: {self.nvrtc.nvrtcGetErrorString(code).decode()}")

    def _cuda_check(self, code: int) -> None:
        if code != 0:
            msg = ctypes.c_char_p()
            self.cuda.cuGetErrorString(code, ctypes.byref(msg))
            raise RuntimeError(f"CUDA error {code}: {(msg.value or b'?').decode()}")

    def compile(self, name: str, arch: str) -> bytes:
        """The cubin of kernel ``name`` for ``arch`` (e.g. "sm_89"), from the disk cache or NVRTC."""
        source = kernel_source(name)
        options = [f"--gpu-architecture={arch}", "-default-device", "--std=c++17"]
        key = hashlib.sha256("\0".join([source, *options, str(self.nvrtc_version)]).encode()).hexdigest()[:32]
        cache = Path(os.environ.get("BLOCHPARTY_CUDA_CACHE", Path.home() / ".cache" / "blochparty" / "cuda"))
        path = cache / f"{name}-{arch}-{key}.cubin"
        if path.exists():
            return path.read_bytes()
        prog = ctypes.c_void_p()
        self._nvrtc_check(self.nvrtc.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), f"{name}.cu".encode(),
                                                        0, None, None))
        try:
            opts = (ctypes.c_char_p * len(options))(*[o.encode() for o in options])
            code = self.nvrtc.nvrtcCompileProgram(prog, len(options), opts)
            size = ctypes.c_size_t()
            self.nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value)
            self.nvrtc.nvrtcGetProgramLog(prog, log)
            if code != 0:
                raise RuntimeError(f"NVRTC could not compile {name}:\n{log.value.decode(errors='replace')}")
            self._nvrtc_check(self.nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(size)))
            cubin = ctypes.create_string_buffer(size.value)
            self._nvrtc_check(self.nvrtc.nvrtcGetCUBIN(prog, cubin))
            data = cubin.raw
        finally:
            self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        try:
            cache.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        except OSError:
            pass
        return data

    def function(self, name: str) -> ctypes.c_void_p:
        dev = torch.cuda.current_device()
        key = (dev, name)
        if key not in self.functions:
            torch.empty(1, device=f"cuda:{dev}")  # the primary context exists and is current
            major, minor = torch.cuda.get_device_capability(dev)
            cubin = self.compile(name, f"sm_{major}{minor}")
            module, func = ctypes.c_void_p(), ctypes.c_void_p()
            self._cuda_check(self.cuda.cuModuleLoadData(ctypes.byref(module), ctypes.c_char_p(cubin)))
            self._cuda_check(self.cuda.cuModuleGetFunction(ctypes.byref(func), module, name.encode()))
            self.modules.append(module)
            self.functions[key] = func
        return self.functions[key]

    def attribute(self, name: str, attr: int) -> int:
        """cuFuncGetAttribute (3: local memory bytes per thread, 4: registers per thread)."""
        value = ctypes.c_int()
        self._cuda_check(self.cuda.cuFuncGetAttribute(ctypes.byref(value), attr, self.function(name)))
        return value.value

    def launch(self, name: str, args, threads: int, group_size: int = 256) -> None:
        func = self.function(name)
        holders = []
        for a in args:
            if not isinstance(a, torch.Tensor) or not a.is_cuda:
                raise TypeError(f"{name}: every argument must be a CUDA tensor")
            holders.append(ctypes.c_void_p(a.data_ptr()))
        params = (ctypes.c_void_p * len(holders))(*[ctypes.addressof(h) for h in holders])
        grid = max(1, -(-threads // group_size))
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        self._cuda_check(self.cuda.cuLaunchKernel(func, grid, 1, 1, group_size, 1, 1, 0, stream, params, None))


_DRIVER: _Driver | None = None


def _driver() -> _Driver:
    global _DRIVER
    if _DRIVER is None:
        _DRIVER = _Driver()
    return _DRIVER


class _Kernels:
    """metal._lib()'s interface: ``lib.<kernel name>(*buffers, threads=, group_size=)``."""

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        kernel_source(name)  # raises AttributeError for names that are no kernel

        def launch(*args, threads: int, group_size: int = 256):
            _driver().launch(name, args, threads, group_size)

        return launch


class _Cuda:
    """The driver's backend (metal._Metal's interface) on the current CUDA device."""

    name = "cuda"
    max_coils = MAX_COILS

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", torch.cuda.current_device())

    def lib(self) -> _Kernels:
        return _Kernels()

    def synchronize(self) -> None:
        torch.cuda.synchronize()


CUDA = _Cuda()


def available() -> bool:
    """A CUDA device, NVRTC and the driver library are there (kernels compile on first use)."""
    try:
        return torch.cuda.is_available() and _driver() is not None
    except Exception:
        return False


def simulate_cuda(phantom: Phantom, seq: Sequence, coils: torch.Tensor | None = None, *, m_init: torch.Tensor | None = None,
                  chunk: int = 32768, group: int = 256, scanner=None, mma: bool | None = None, seed: int = 0, maps=None,
                  checkpoint_every: int | None = None, readouts: str = "direct", nufft_width: int = 6,
                  nufft_terms: int = 3, pulses: str = "step") -> SimResult:
    """Forward simulation on an NVIDIA GPU, with ``simulate_metal``'s arguments, results (float32, on the CPU),
    gradients (the same hand-written adjoint) and limits; see its docstring. ``mma`` picks the sensitivity layout
    and kernel as on Metal (``bloch_chunk_mma``: any coil count up to ``MAX_COILS``, NUFFT readouts, reused
    pulses; ``bloch_chunk``: at most 32 coils); both reduce the coils by warp shuffles here."""
    return metal._simulate(CUDA, phantom, seq, coils, m_init, chunk, group, scanner, mma, seed, maps, checkpoint_every,
                           readouts, nufft_width, nufft_terms, pulses)
