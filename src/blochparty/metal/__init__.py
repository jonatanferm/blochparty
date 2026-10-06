"""Bloch simulation on the Apple GPU (Metal), forward only.

The same physics as the CPU paths (``bloch.py``, ``fused/kernel.h``): per segment exact
free precession and relaxation (relaxation in the ``m += (m - m0) * expm1(-dt/T)`` form, which
float32 needs), Strang-split Rodrigues rotations on RF segments, B1+ scaling and per-segment
uniform B0 offsets; ADC samples at segment ends, demodulated by exp(-i psi); ideal spoils.

One GPU thread per spin steps through the segments with its magnetization in registers. At each
ADC sample the spins' coil contributions are summed with simd_sum, then across the threadgroup,
then added to the signal with one atomic per threadgroup. Segments are dispatched in chunks so
no single command buffer runs for long. Metal has no double precision: everything is float32.

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
    const float px = spins[j], py = spins[n + j], pz = spins[2 * n + j], pd = spins[3 * n + j];
    const float r1 = spins[4 * n + j], r2 = spins[5 * n + j], db0 = spins[6 * n + j], b1 = spins[7 * n + j];
    const uint nv = 2 * n_coils;
    const uint block = TG_FLOATS / (MAX_SG * nv);  // samples buffered per flush
    uint nb = 0;
    // Factor cache for free-precession segments: runs of identical segments (every readout
    // sample) reuse each spin's rotation and relaxation factors. All threads see the same
    // segment, so the check never diverges.
    float k_dt = -1.0f, k_gx = 0.0f, k_gy = 0.0f, k_gz = 0.0f, k_b0 = 0.0f;
    float k_c = 1.0f, k_s = 0.0f, k_e1 = 0.0f, k_e2 = 0.0f;

    for (uint s = s0; s < s1; ++s) {
        constant float* q = segs + 8 * s;
        const float dt = q[0];
        const float bz = px * q[3] + py * q[4] + pz * q[5] + db0 + q[7];  // off-resonance [Hz]
        const int f = flags[s];
        if (f & 1) {  // RF: relax dt/2, rotate about the effective field, relax dt/2
            relax(mx, my, mz, 0.5f * dt, pd, r1, r2);
            const float w = -TWO_PI * dt;
            const float wx = w * b1 * q[1], wy = w * b1 * q[2], wz = w * bz;
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
            relax(mx, my, mz, 0.5f * dt, pd, r1, r2);
        } else {  // free precession about z, exact relaxation
            if (dt != k_dt || q[3] != k_gx || q[4] != k_gy || q[5] != k_gz || q[7] != k_b0) {
                const float phi = -TWO_PI * dt * bz;
                k_c = precise::cos(phi); k_s = precise::sin(phi);
                k_e1 = expm1_(-dt * r1); k_e2 = expm1_(-dt * r2);
                k_dt = dt; k_gx = q[3]; k_gy = q[4]; k_gz = q[5]; k_b0 = q[7];
            }
            const float nx = k_c * mx - k_s * my, ny = k_s * mx + k_c * my;
            mx = nx + nx * k_e2; my = ny + ny * k_e2;
            mz += (mz - pd) * k_e1;
        }
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
"""

_LIB = None
MAX_COILS = 32


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
                   chunk: int = 32768, group: int = 256, scanner=None) -> SimResult:
    """Forward simulation on the Apple GPU; same arguments and result as ``blochparty.simulate``
    (results come back on the CPU, float32). ``chunk`` segments per dispatch; ``group`` threads
    per threadgroup (a multiple of 32, at most 1024)."""
    if scanner is not None:
        raise NotImplementedError("the scanner field model is not implemented on Metal; use simulate_fused")
    dev, f32 = torch.device("mps"), torch.float32
    n = len(phantom)
    b1 = phantom.b1 if getattr(phantom, "b1", None) is not None else torch.ones(n)
    spins = torch.stack([phantom.pos[:, 0], phantom.pos[:, 1], phantom.pos[:, 2], phantom.pd, 1 / phantom.t1,
                         1 / phantom.t2, phantom.db0, b1.to(phantom.pd.dtype)]).to(dev, f32).contiguous()
    if coils is None:
        coils = torch.ones(n, 1, dtype=torch.complex64)
    n_coils = coils.shape[1]
    if n_coils > MAX_COILS:
        raise ValueError(f"at most {MAX_COILS} coils on Metal")
    if group != 256:
        raise ValueError("the Metal kernel's sample buffer is sized for threadgroups of 256")
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
    threads = math.ceil(n / group) * group
    lib, S = _lib(), len(seq)
    for s0 in range(0, S, chunk):
        lib.bloch_chunk(M, spins, sens, segs, flags, rows, sig, n, n_coils, s0, min(S, s0 + chunk),
                        threads=threads, group_size=group)
    torch.mps.synchronize()
    sig = sig[: n_adc * n_coils * 2].cpu().view(n_adc, n_coils, 2)
    return SimResult(torch.complex(sig[..., 0], sig[..., 1]).T.contiguous(), M.cpu().T.contiguous())
