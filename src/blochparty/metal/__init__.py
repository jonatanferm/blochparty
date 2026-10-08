"""Bloch simulation on the Apple GPU (Metal), forward only.

The same physics as the CPU paths (``bloch.py``, ``fused/kernel.h``): per segment exact
free precession and relaxation (relaxation in the ``m += (m - m0) * expm1(-dt/T)`` form, which
float32 needs), Strang-split Rodrigues rotations on RF segments, B1+ scaling and per-segment
uniform B0 offsets; ADC samples at segment ends, demodulated by exp(-i psi); ideal spoils. RF
segments add their increments (rotation as m += s1 w x m + s2 w x (w x m), relaxation, exchange)
by an exact two-sum into a per-spin low part: float32 otherwise rounds all of m every segment, and
long RF trains (CW saturation, fine-raster shaped pulses) drifted from the core by 1e-2 after 1e5
segments.

One GPU thread per spin steps through the segments with its magnetization in registers. The
coil sum at ADC samples has two implementations. With two or more coils (``bloch_chunk_mma``),
each simdgroup buffers 8 samples of its 32 spins and does the sum as a complex matrix product
(coils x spins) . (spins x samples) on 8x8 simdgroup matrices, demodulating afterwards; the cost
barely grows with the coil count (48^3 GRE: 0.41 s for 1 coil, 0.42 s for 8, 1.6 s for 64). With
one coil (``bloch_chunk``) each sample is summed with simd_sum. Either way the per-simdgroup sums
are reduced across the threadgroup and added with one atomic per threadgroup. Segments are
dispatched in chunks so no single command buffer runs for long. Metal has no double precision:
everything is float32.

Magnetization transfer (``Phantom.bound``): the bound pool's Mz is a fourth state per thread and
longitudinal relaxation the exact exchange step of ``exchange.py``, saturated during RF (about
+15 % on a 3D GRE).

Free exchanging pools (``Phantom.pools``, up to ``MAX_POOLS``, with or without the rate-model bound
pool): the Bloch-McConnell system of ``bloch._simulate_exchange``, one kernel per pool count
(``_bm1`` ...). Each segment applies exp(A dt) to the state by a Taylor series on sub-steps, the
common precession of free segments factored out exactly, increments added by two-sum; a float32
matrix exponential per segment was 1e-2 off over a CEST saturation train. A segment costs sub-steps
in proportion to |A| dt, so spins are sorted by that cost before the dispatch (a simdgroup waits
for its slowest lane) and returned in the caller's order.

The scanner field model (``fields.Scanner``: gradient nonlinearity through the encoding
positions, concomitant fields at leading order or exact) enters the off-resonance as in
``fused/kernel.h``.

Gradients: a hand-written adjoint (``_MetalSimulate``), the reverse sweep of fused/kernel.h on the GPU.
The forward stores the state (hi and lo parts) before every ``checkpoint_every`` segments; per interval
in reverse order, ``bloch_replay`` reruns the forward keeping every 4th state, and ``bloch_adjoint``
recomputes each block of 4 into threadgroup memory and sweeps it backward. Per-spin gradients stay with
their thread; per-segment ones are summed per simdgroup and then in order (``seg_reduce``), so the result
is the same for any interval and every run. Not for moving spins, free pools or the generalized bound
pool. What runs where: ``docs/ENGINES.md``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from .. import exchange
from ..bloch import SimResult
from ..fields import spin_fields
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

// hi + lo += d, exactly up to the rounding of lo (Knuth's two-sum). RF segments add their increments
// this way: float32 rounding of a slowly changing m otherwise accumulates, 1e-3 in Mz after 1e4 RF
// segments of CW saturation and 1e-2 after 1e5 (the core is float64). Survives Metal's fast math.
inline void two_sum(thread float& hi, thread float& lo, float d) {
    d += lo;
    const float t = hi + d, bp = t - hi;
    lo = (hi - (t - bp)) + (d - bp);
    hi = t;
}

inline void relax(thread float& mx, thread float& my, thread float& mz, float t, float pd, float r1, float r2) {
    float e1 = expm1_(-t * r1), e2 = expm1_(-t * r2);
    mx += mx * e2;
    my += my * e2;
    mz += (mz - pd) * e1;
}


struct Spin { float px, py, pz, pd, r1, r2, db0, b1; };

// Concomitant-field coefficients (fields.py, as kernel.h): mode 1 (leading order) c . (gx^2, gy^2,
// gz^2, gx gy, gx gz, gy gz); mode 2 (exact |B|) t = (c0..2 . g, c3..5 . g), f0 = c6.
struct Conc { float c[7]; };

// Off-resonance [Hz] of a spin in segment q (gradient q[3..5], uniform offset q[7]).
inline float offset(thread const Spin& p, thread const Conc& k, uint mode, constant float* q) {
    const float lin = p.px * q[3] + p.py * q[4] + p.pz * q[5] + p.db0 + q[7];
    if (mode == 0) return lin;
    const float gx = q[3], gy = q[4], gz = q[5];
    if (mode == 1)
        return lin + (k.c[0] * (gx * gx) + k.c[1] * (gy * gy) + k.c[2] * (gz * gz) + k.c[3] * (gx * gy) +
                      k.c[4] * (gx * gz) + k.c[5] * (gy * gz));
    const float tx = k.c[0] * gx + k.c[1] * gy + k.c[2] * gz, ty = k.c[3] * gx + k.c[4] * gy + k.c[5] * gz;
    const float t2 = tx * tx + ty * ty, u = k.c[6] + lin;
    return lin + t2 / (sqrt(u * u + t2) + u);
}

// Factor cache for free-precession segments: runs of identical segments (every readout sample)
// reuse each spin's rotation and relaxation factors. All threads see the same segment, so the
// check never diverges.
struct Cache { float dt = -1.0f, gx = 0.0f, gy = 0.0f, gz = 0.0f, b0 = 0.0f, c = 1.0f, s = 0.0f, e1 = 0.0f, e2 = 0.0f; };

// Rotation about the effective field of an RF segment q for a spin at offset bz [Hz], as an increment
// m += s1 (w x m) + s2 w x (w x m) (Rodrigues with cos = 1 - s2 phi^2) added by two_sum into l.
inline void rotate_inc(float wx, float wy, float wz, thread float& mx, thread float& my, thread float& mz,
                       thread float& lx, thread float& ly, thread float& lz) {
    const float phi2 = wx * wx + wy * wy + wz * wz, phi = sqrt(phi2);
    float s1v, s2v;
    if (phi < 1e-3f) {
        s1v = 1.0f - phi2 / 6.0f;
        s2v = 0.5f - phi2 / 24.0f;
    } else {
        const float sh = precise::sin(0.5f * phi) / phi;
        s1v = precise::sin(phi) / phi;
        s2v = 2.0f * sh * sh;  // (1 - cos phi) / phi^2 without cancellation
    }
    const float cx = wy * mz - wz * my, cy = wz * mx - wx * mz, cz = wx * my - wy * mx;  // w x m
    const float dx = wy * cz - wz * cy, dy = wz * cx - wx * cz, dz = wx * cy - wy * cx;  // w x (w x m)
    two_sum(mx, lx, s1v * cx + s2v * dx), two_sum(my, ly, s1v * cy + s2v * dy), two_sum(mz, lz, s1v * cz + s2v * dz);
}

inline void rf_rotate(constant float* q, float bz, thread float& mx, thread float& my, thread float& mz,
                      thread float& lx, thread float& ly, thread float& lz, thread const Spin& p) {
    const float w = -TWO_PI * q[0];
    rotate_inc(w * p.b1 * q[1], w * p.b1 * q[2], w * bz, mx, my, mz, lx, ly, lz);
}

// The low parts l of a run of RF segments are folded back at the next free segment.
inline void step(constant float* q, int f, float bz, thread float& mx, thread float& my, thread float& mz,
                 thread float& lx, thread float& ly, thread float& lz, thread const Spin& p, thread Cache& k) {
    const float dt = q[0];
    if (f & 1) {  // RF: relax dt/2, rotate about the effective field, relax dt/2 (increments compensated)
        const float e1 = expm1_(-0.5f * dt * p.r1), e2 = expm1_(-0.5f * dt * p.r2);
        two_sum(mx, lx, mx * e2), two_sum(my, ly, my * e2), two_sum(mz, lz, (mz - p.pd) * e1);
        rf_rotate(q, bz, mx, my, mz, lx, ly, lz, p);
        two_sum(mx, lx, mx * e2), two_sum(my, ly, my * e2), two_sum(mz, lz, (mz - p.pd) * e1);
    } else {  // free precession about z, exact relaxation
        mx += lx, my += ly, mz += lz, lx = ly = lz = 0.0f;
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

// ---- magnetization transfer (exchange.py): the bound pool's Mz `mb` exchanges with mz; RF
// saturates it at W = w1^2 G(f_rf - offset - dw). Longitudinal relaxation becomes the exact
// 2x2 exchange step in increment form (exchange.z_exchange).
struct Pool { float f, k, r1b, t2b, dw; };

inline Pool load_pool(constant float* pool, uint n, uint j) {
    return Pool{pool[j], pool[n + j], pool[2 * n + j], pool[3 * n + j], pool[4 * n + j]};
}

// line: kind (1 Lorentzian, 2 super-Lorentzian), cutoff [Hz], table log x0, step, x_min, size
inline float lineshape(float delta, float t2b, constant float* line, constant float* log_f) {
    if (line[0] < 1.5f) {
        const float x = TWO_PI * delta * t2b;
        return t2b / (1.0f + x * x);
    }
    const float x = TWO_PI * max(fabs(delta), line[1]) * t2b;
    if (x > 30.0f) return 0.0f;
    const int n = int(line[5]);
    const float pos = clamp((precise::log(max(x, line[4])) - line[2]) / line[3], 0.0f, float(n - 1));
    const int i = min(int(floor(pos)), n - 2);
    const float a = pos - float(i);
    return t2b * precise::exp(log_f[i] * (1.0f - a) + log_f[i + 1] * a);
}

struct ZStep { float chm1, sh, z1, z2, h, a12, a21; };

inline ZStep zstep(float t, thread const Spin& p, thread const Pool& b, float w) {
    const float kf = b.k * b.f;
    const float a11 = -(p.r1 + kf), a12 = b.k, a21 = kf, a22 = -(b.r1b + b.k + w);
    const float c1 = p.r1 * p.pd, c2 = b.r1b * b.f * p.pd;
    const float m = 0.5f * (a11 + a22), h = 0.5f * (a11 - a22), q = sqrt(h * h + a12 * a21);
    const float det = a11 * a22 - a12 * a21;
    ZStep z;
    z.z1 = -(a22 * c1 - a12 * c2) / det, z.z2 = -(a11 * c2 - a21 * c1) / det;
    z.chm1 = 0.5f * (expm1_((m + q) * t) + expm1_((m - q) * t));
    if (2.0f * q * t < 1.0f) {
        const float lo = precise::exp((m - q) * t), e2q = expm1_(2.0f * q * t);
        z.sh = q > 0.0f ? lo * 0.5f * e2q / q : t * precise::exp(m * t);
    } else {
        z.sh = 0.5f * (precise::exp((m + q) * t) - precise::exp((m - q) * t)) / q;
    }
    z.h = h, z.a12 = a12, z.a21 = a21;
    return z;
}

inline void zapply(thread const ZStep& z, thread float& u, thread float& v) {
    const float d1 = u - z.z1, d2 = v - z.z2;
    const float n1 = z.chm1 * d1 + z.sh * (z.h * d1 + z.a12 * d2), n2 = z.chm1 * d2 + z.sh * (z.a21 * d1 - z.h * d2);
    u += n1, v += n2;
}

inline void zapply_c(thread const ZStep& z, thread float& u, thread float& v, thread float& lu, thread float& lv) {
    const float d1 = u - z.z1, d2 = v - z.z2;
    two_sum(u, lu, z.chm1 * d1 + z.sh * (z.h * d1 + z.a12 * d2)), two_sum(v, lv, z.chm1 * d2 + z.sh * (z.a21 * d1 - z.h * d2));
}

// step() with a bound pool; zc caches the unsaturated exchange step of free segments (key dt).
struct ZCache { float dt = -1.0f; ZStep z; };

inline void step_mt(constant float* q, int f, float bz, float f_rf, thread float& mx, thread float& my,
                    thread float& mz, thread float& mb, thread float& lx, thread float& ly, thread float& lz,
                    thread float& lb, thread const Spin& p, thread const Pool& b, constant float* line,
                    constant float* log_f, thread Cache& k, thread ZCache& zc) {
    const float dt = q[0];
    if (f & 1) {  // as step(): increments compensated, low parts folded back at the next free segment
        const float w1sq = TWO_PI * TWO_PI * p.b1 * p.b1 * (q[1] * q[1] + q[2] * q[2]);
        const ZStep z = zstep(0.5f * dt, p, b, w1sq * lineshape(f_rf - bz - b.dw, b.t2b, line, log_f));
        const float e2 = expm1_(-0.5f * dt * p.r2);
        two_sum(mx, lx, mx * e2), two_sum(my, ly, my * e2);
        zapply_c(z, mz, mb, lz, lb);
        rf_rotate(q, bz, mx, my, mz, lx, ly, lz, p);
        two_sum(mx, lx, mx * e2), two_sum(my, ly, my * e2);
        zapply_c(z, mz, mb, lz, lb);
    } else {
        mx += lx, my += ly, mz += lz, mb += lb, lx = ly = lz = lb = 0.0f;
        if (dt != k.dt || q[3] != k.gx || q[4] != k.gy || q[5] != k.gz || q[7] != k.b0) {
            const float phi = -TWO_PI * dt * bz;
            k.c = precise::cos(phi); k.s = precise::sin(phi);
            k.e1 = expm1_(-dt * p.r1); k.e2 = expm1_(-dt * p.r2);
            k.dt = dt; k.gx = q[3]; k.gy = q[4]; k.gz = q[5]; k.b0 = q[7];
        }
        if (dt != zc.dt) { zc.z = zstep(dt, p, b, 0.0f); zc.dt = dt; }
        const float nx = k.c * mx - k.s * my, ny = k.s * mx + k.c * my;
        mx = nx + nx * k.e2; my = ny + ny * k.e2;
        zapply(zc.z, mz, mb);
    }
}

// ---- generalized Bloch bound pool (generalized_bloch.modes): modes u_j of the pool's transverse
// magnetization; RF segments sub-stepped (nsub per segment, from Python); the pool rotated by the RF
// alone, each mode taking c_j of the change of S = sum u_j. As bloch._rf_segment_generalized.
#define MAX_MODES 32

struct Modes {
    float ur[MAX_MODES], ui[MAX_MODES];   // state
    float fr[MAX_MODES], fi[MAX_MODES];   // free-segment factors, cached on (dt, g, b0)
    float k_dt = -1.0f, k_gx = 0.0f, k_gy = 0.0f, k_gz = 0.0f, k_b0 = 0.0f;
};

inline void mode_factor(constant float* modes, uint j, float t, float t2b, float dbz, thread float& a, thread float& b) {
    const float amp = precise::exp(-modes[4 * j] * t / t2b);
    const float ph = -TWO_PI * dbz * t - modes[4 * j + 1] * t / t2b;
    a = amp * precise::cos(ph), b = amp * precise::sin(ph);
}

inline void step_gen(constant float* q, int f, float bz, uint nsub, thread float& mx, thread float& my, thread float& mz,
                     thread float& mb, thread float& lx, thread float& ly, thread float& lz, thread float& lb,
                     thread Modes& md, thread const Spin& p, thread const Pool& b, constant float* modes, uint nj,
                     thread Cache& k, thread ZCache& zc) {
    const float dt = q[0];
    const float dbz = bz + b.dw;
    if (f & 1) {
        const float h = dt / float(nsub), hstep = 0.5f * h;
        const ZStep z = zstep(hstep, p, b, 0.0f);
        const float e2 = expm1_(-hstep * p.r2);
        float hr[MAX_MODES], hi[MAX_MODES];
        for (uint j = 0; j < nj; ++j) mode_factor(modes, j, hstep, b.t2b, dbz, hr[j], hi[j]);
        const float w = -TWO_PI * h;
        const float wx = w * p.b1 * q[1], wy = w * p.b1 * q[2], wz = w * bz;
        for (uint s = 0; s < nsub; ++s) {
            for (uint hs = 0; hs < 2; ++hs) {
                if (hs) {
                    // the water about its effective field (compensated increments, as step())
                    rotate_inc(wx, wy, wz, mx, my, mz, lx, ly, lz);
                    // the pool about the RF alone
                    float sx = 0.0f, sy = 0.0f;
                    for (uint j = 0; j < nj; ++j) sx += md.ur[j], sy += md.ui[j];
                    const float ph2 = wx * wx + wy * wy, ph = sqrt(ph2);
                    float pc, ps1, ps2;
                    if (ph < 1e-3f) { pc = 1.0f - 0.5f * ph2; ps1 = 1.0f - ph2 / 6.0f; ps2 = 0.5f - ph2 / 24.0f; }
                    else { const float sh = precise::sin(0.5f * ph) / ph; pc = precise::cos(ph); ps1 = precise::sin(ph) / ph; ps2 = 2.0f * sh * sh; }
                    const float pdot = wx * sx + wy * sy;
                    const float tx = pc * sx + ps1 * (wy * mb) + ps2 * wx * pdot;
                    const float ty = pc * sy - ps1 * (wx * mb) + ps2 * wy * pdot;
                    two_sum(mb, lb, ps1 * (wx * sy - wy * sx) - ps2 * ph2 * mb);  // pc = 1 - ps2 ph^2
                    const float dx = tx - sx, dy = ty - sy;
                    for (uint j = 0; j < nj; ++j) {
                        const float cr = modes[4 * j + 2], ci = modes[4 * j + 3];
                        md.ur[j] += cr * dx - ci * dy, md.ui[j] += cr * dy + ci * dx;
                    }
                }
                two_sum(mx, lx, mx * e2), two_sum(my, ly, my * e2);
                zapply_c(z, mz, mb, lz, lb);
                for (uint j = 0; j < nj; ++j) {
                    const float x = md.ur[j], y = md.ui[j];
                    md.ur[j] = hr[j] * x - hi[j] * y, md.ui[j] = hr[j] * y + hi[j] * x;
                }
            }
        }
    } else {
        mx += lx, my += ly, mz += lz, mb += lb, lx = ly = lz = lb = 0.0f;
        if (dt != k.dt || q[3] != k.gx || q[4] != k.gy || q[5] != k.gz || q[7] != k.b0) {
            const float phi = -TWO_PI * dt * bz;
            k.c = precise::cos(phi); k.s = precise::sin(phi);
            k.e1 = expm1_(-dt * p.r1); k.e2 = expm1_(-dt * p.r2);
            k.dt = dt; k.gx = q[3]; k.gy = q[4]; k.gz = q[5]; k.b0 = q[7];
            for (uint j = 0; j < nj; ++j) mode_factor(modes, j, dt, b.t2b, dbz, md.fr[j], md.fi[j]);
        }
        if (dt != zc.dt) { zc.z = zstep(dt, p, b, 0.0f); zc.dt = dt; }
        const float nx = k.c * mx - k.s * my, ny = k.s * mx + k.c * my;
        mx = nx + nx * k.e2; my = ny + ny * k.e2;
        zapply(zc.z, mz, mb);
        for (uint j = 0; j < nj; ++j) {
            const float x = md.ur[j], y = md.ui[j];
            md.ur[j] = md.fr[j] * x - md.fi[j] * y, md.ui[j] = md.fr[j] * y + md.fi[j] * x;
        }
    }
    if (f & 4) for (uint j = 0; j < nj; ++j) md.ur[j] = 0.0f, md.ui[j] = 0.0f;
}

// ---- free exchanging pools (exchange.ExchangePool, bloch._simulate_exchange): the Bloch-McConnell
// system of the water, up to MAX_POOLS free pools (x, y, z each) and the rate-model bound pool's Mz.
// A segment applies exp(A dt) to the state as a Taylor series of the structured A on sub-steps of
// norm <= BM_THETA, truncated when a term falls below 2^-27 of the magnetization: no matrix is
// formed, the state stays in registers. (A float32 matrix exponential per segment, as the core does
// in float64, was 1e-2 off over the 2000 segments of a CEST saturation train.) Free segments leave
// the common precession bz out of A and rotate every pool's Mxy by it exactly afterwards (it
// commutes with the rest there), so readout gradients cost no sub-steps. The state is kept as
// hi + lo with every increment added by an exact two-sum: rounding of a slowly changing Mz otherwise
// accumulates over thousands of segments (3e-5 after 2000 saturation steps).
#define MAX_POOLS 5                         // = metal.MAX_POOLS
#define BM_NC (3 * (MAX_POOLS + 1) + 1)     // state components: water, pools (x y z), bound Mz
#define BM_THETA 1.5f

struct BMVec { float x, y, z, px[MAX_POOLS], py[MAX_POOLS], pz[MAX_POOLS], b; };

// Per spin, fixed: the state (hi + lo) and the coefficients that do not change between segments.
struct BMS {
    BMVec hi, lo;
    bool bound = false;
    float c0;                                                          // water: r1 PD
    float kin[MAX_POOLS], kout[MAX_POOLS], r1[MAX_POOLS], r2[MAX_POOLS], cz[MAX_POOLS], dw[MAX_POOLS];  // dw [rad/s]
    float kex;                                                         // sum of the pools' kin + kout
    float kfb, kb, r1b, cb;                                            // bound pool: k f, k, r1b, r1b f PD
};

// Per segment: RF [rad/s] (B1+ included), the water's precession wz (pools: wz + dw) and the bound
// pool's loss rate r1b + k + W.
struct BMSeg { float wx, wy, wz, rb; };

// o = A u (exchange._bm_matrix); `inh`: u's augmented entry is 1 (the first Taylor term only). NP, the
// number of free pools, is a compile-time constant (kernels _bm1 .. _bm3): with a runtime count the
// pool loops are not unrolled, the pools' arrays live in memory, and the kernel ran ~3x slower.
template <uint NP>
inline void bm_apply(thread const BMS& s, thread const Spin& p, thread const BMSeg& c, thread const BMVec& u, bool inh,
                     thread BMVec& o) {
    o.x = c.wz * u.y - c.wy * u.z - p.r2 * u.x;
    o.y = -c.wz * u.x + c.wx * u.z - p.r2 * u.y;
    o.z = c.wy * u.x - c.wx * u.y - p.r1 * u.z + (inh ? s.c0 : 0.0f);
    for (uint q = 0; q < NP; ++q) {
        const float x = u.px[q], y = u.py[q], z = u.pz[q], wz = c.wz + s.dw[q];
        const float ex = s.kin[q] * x - s.kout[q] * u.x, ey = s.kin[q] * y - s.kout[q] * u.y,
                    ez = s.kin[q] * z - s.kout[q] * u.z;
        o.px[q] = wz * y - c.wy * z - s.r2[q] * x - ex;
        o.py[q] = -wz * x + c.wx * z - s.r2[q] * y - ey;
        o.pz[q] = c.wy * x - c.wx * y - s.r1[q] * z - ez + (inh ? s.cz[q] : 0.0f);
        o.x += ex, o.y += ey, o.z += ez;
    }
    o.b = 0.0f;
    if (s.bound) {
        o.z += s.kb * u.b - s.kfb * u.z;
        o.b = s.kfb * u.z - c.rb * u.b + (inh ? s.cb : 0.0f);
    }
}

template <uint NP>
inline float bm_max(thread const BMVec& u) {
    float m = max(max(fabs(u.x), fabs(u.y)), max(fabs(u.z), fabs(u.b)));
    for (uint p = 0; p < NP; ++p) m = max(m, max(max(fabs(u.px[p]), fabs(u.py[p])), fabs(u.pz[p])));
    return m;
}

inline void rot2(thread float& x, thread float& y, float c, float s) {
    const float nx = c * x - s * y, ny = s * x + c * y;
    x = nx, y = ny;
}

// state buffer (2 BM_NC, N): hi then lo, components x y z, pools' x y z (3 + 3p ...), bound Mz (last).
// No pointers into the state: taking addresses keeps it out of registers for the whole kernel.
inline BMVec bm_load(device const float* buf, uint off, uint n, uint j) {
    BMVec v;
    v.x = buf[off * n + j], v.y = buf[(off + 1) * n + j], v.z = buf[(off + 2) * n + j];
    for (uint p = 0; p < MAX_POOLS; ++p)
        v.px[p] = buf[(off + 3 + 3 * p) * n + j], v.py[p] = buf[(off + 4 + 3 * p) * n + j], v.pz[p] = buf[(off + 5 + 3 * p) * n + j];
    v.b = buf[(off + BM_NC - 1) * n + j];
    return v;
}

inline void bm_store(thread const BMVec& v, device float* buf, uint off, uint n, uint j) {
    buf[off * n + j] = v.x, buf[(off + 1) * n + j] = v.y, buf[(off + 2) * n + j] = v.z;
    for (uint p = 0; p < MAX_POOLS; ++p)
        buf[(off + 3 + 3 * p) * n + j] = v.px[p], buf[(off + 4 + 3 * p) * n + j] = v.py[p], buf[(off + 5 + 3 * p) * n + j] = v.pz[p];
    buf[(off + BM_NC - 1) * n + j] = v.b;
}

template <uint NP>
inline void step_bm(constant float* q, int f, float bz, float f_rf, thread BMS& s, thread const Spin& p,
                    thread const Pool& b, constant float* line, constant float* log_f, thread float& mx,
                    thread float& my, thread float& mz) {
    const float dt = q[0];
    const bool rf = (f & 1) != 0;
    BMSeg c;
    c.wx = rf ? TWO_PI * p.b1 * q[1] : 0.0f;
    c.wy = rf ? TWO_PI * p.b1 * q[2] : 0.0f;
    c.wz = rf ? TWO_PI * bz : 0.0f;  // free segments: the common precession is rotated exactly below
    c.rb = 0.0f;
    if (s.bound)  // saturated during RF at W = w1^2 G (exchange.saturation_rate)
        c.rb = s.r1b + s.kb + (rf ? TWO_PI * TWO_PI * p.b1 * p.b1 * (q[1] * q[1] + q[2] * q[2]) *
                                        lineshape(f_rf - bz - b.dw, b.t2b, line, log_f) : 0.0f);
    // the max row sum of |A| (the series' terms then shrink at least as BM_THETA / k)
    const float w = fabs(c.wx) + fabs(c.wy);
    float nrm = max(p.r2 + fabs(c.wz), p.r1 + s.kfb + s.kb) + w + s.kex;
    for (uint j = 0; j < NP; ++j)
        nrm = max(nrm, max(s.r2[j] + fabs(c.wz + s.dw[j]), s.r1[j]) + w + s.kin[j] + s.kout[j]);
    if (s.bound) nrm = max(nrm, c.rb + s.kfb);
    const uint ns = max(1u, uint(ceil(dt * nrm / BM_THETA)));
    const float h = dt / float(ns);
    for (uint i = 0; i < ns; ++i) {
        const float tol = 7.450580596923828e-9f * (bm_max<NP>(s.hi) + p.pd);  // 2^-27
        BMVec term = s.hi, acc = {}, nxt;
        for (uint k = 1; k <= 40; ++k) {
            bm_apply<NP>(s, p, c, term, k == 1, nxt);
            const float g = h / float(k);
            term.x = nxt.x * g, term.y = nxt.y * g, term.z = nxt.z * g, term.b = nxt.b * g;
            acc.x += term.x, acc.y += term.y, acc.z += term.z, acc.b += term.b;
            for (uint j = 0; j < NP; ++j) {
                term.px[j] = nxt.px[j] * g, term.py[j] = nxt.py[j] * g, term.pz[j] = nxt.pz[j] * g;
                acc.px[j] += term.px[j], acc.py[j] += term.py[j], acc.pz[j] += term.pz[j];
            }
            if (bm_max<NP>(term) <= tol) break;
        }
        two_sum(s.hi.x, s.lo.x, acc.x), two_sum(s.hi.y, s.lo.y, acc.y), two_sum(s.hi.z, s.lo.z, acc.z);
        two_sum(s.hi.b, s.lo.b, acc.b);
        for (uint j = 0; j < NP; ++j) {
            two_sum(s.hi.px[j], s.lo.px[j], acc.px[j]), two_sum(s.hi.py[j], s.lo.py[j], acc.py[j]);
            two_sum(s.hi.pz[j], s.lo.pz[j], acc.pz[j]);
        }
    }
    if (!rf) {  // Mxy *= exp(-i 2 pi bz dt) for every pool
        const float phi = -TWO_PI * dt * bz, cs = precise::cos(phi), sn = precise::sin(phi);
        rot2(s.hi.x, s.hi.y, cs, sn), rot2(s.lo.x, s.lo.y, cs, sn);
        for (uint j = 0; j < NP; ++j) rot2(s.hi.px[j], s.hi.py[j], cs, sn), rot2(s.lo.px[j], s.lo.py[j], cs, sn);
    }
    // the signal is every free pool's Mxy (the kernel samples mx, my and then spoils them)
    mx = s.hi.x + s.lo.x, my = s.hi.y + s.lo.y, mz = s.hi.z + s.lo.z;
    for (uint j = 0; j < NP; ++j) mx += s.hi.px[j] + s.lo.px[j], my += s.hi.py[j] + s.lo.py[j];
    if (f & 4) {
        s.hi.x = s.hi.y = s.lo.x = s.lo.y = 0.0f;
        for (uint j = 0; j < NP; ++j) s.hi.px[j] = s.hi.py[j] = s.lo.px[j] = s.lo.py[j] = 0.0f;
    }
}

// ---- diffusion: exact Brownian segments drawn from Philox4x32-10 (rng.py), as bloch._walk
inline uint4 philox(uint4 c, uint2 k) {
    for (int r = 0; r < 10; ++r) {
        const uint hi0 = mulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
        const uint hi1 = mulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
        c = uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
        k += uint2(0x9E3779B9u, 0xBB67AE85u);
    }
    return c;
}

inline float2 box_muller(uint a, uint b) {
    const float u1 = (float(a) + 0.5f) * 2.3283064365386963e-10f, u2 = (float(b) + 0.5f) * 2.3283064365386963e-10f;
    const float r = sqrt(-2.0f * precise::log(u1)), th = TWO_PI * u2;
    return float2(r * precise::cos(th), r * precise::sin(th));
}

struct Walk {
    float x, y, z;           // where the spin is
    float l[6];              // Cholesky factor of D, row-major lower triangle: l00 l10 l11 l20 l21 l22
    uint2 seed;
    uint spin;
};

// Effective position over segment s (duration dt) and the move to its end (rng.normals: n1 = z[0:3], n2 = z[3:6]).
inline float3 walk_segment(thread Walk& w, uint s, float dt) {
    const uint4 a = philox(uint4(w.spin, s, 0u, 0u), w.seed), b = philox(uint4(w.spin, s, 0u, 1u), w.seed);
    const float2 z01 = box_muller(a.x, a.y), z23 = box_muller(a.z, a.w), z45 = box_muller(b.x, b.y);
    const float3 n1 = float3(z01.x, z01.y, z23.x), n2 = float3(z23.y, z45.x, z45.y);
    const float3 l1 = float3(w.l[0] * n1.x, w.l[1] * n1.x + w.l[2] * n1.y, w.l[3] * n1.x + w.l[4] * n1.y + w.l[5] * n1.z);
    const float3 l2 = float3(w.l[0] * n2.x, w.l[1] * n2.x + w.l[2] * n2.y, w.l[3] * n2.x + w.l[4] * n2.y + w.l[5] * n2.z);
    const float3 step = sqrt(2.0f * dt) * l1;
    const float3 eff = float3(w.x, w.y, w.z) + 0.5f * step + sqrt(2.0f * dt / 12.0f) * l2;
    w.x += step.x, w.y += step.y, w.z += step.z;
    return eff;
}

// Restricted walks (restriction.py): sub-steps reflecting off the compartment walls. kind 1 planes
// (width size, normal ax), 2 cylinder (radius size, axis ax), 3 sphere (radius size).
// End offset of a step from offset a (inside) to b, reflected off the walls (restriction.reflect):
// a mirror at +-L/2 for planes; specular at the crossing point for cylinders and spheres.
inline float3 fold(float3 a, float3 b, uint kind, float size, float3 ax) {
    if (kind == 1) {
        const float s0 = dot(b, ax), hw = 0.5f * size;
        float t = s0;
        for (int k = 0; k < 8; ++k) t = t > hw ? 2.0f * hw - t : (t < -hw ? -2.0f * hw - t : t);
        return b + (t - s0) * ax;
    }
    float3 along = float3(0.0f);
    if (kind == 2) along = dot(b, ax) * ax, a -= dot(a, ax) * ax, b -= along;
    const float R2 = size * size;
    for (int k = 0; k < 8 && dot(b, b) > R2; ++k) {
        const float3 v = b - a;
        const float vv = max(dot(v, v), 1e-30f), av = dot(a, v);
        const float t = clamp((-av + sqrt(max(av * av + vv * (R2 - dot(a, a)), 0.0f))) / vv, 0.0f, 1.0f);
        const float3 h = a + t * v, nrm = h / max(length(h), 1e-30f);
        float3 rest = (1.0f - t) * v;
        rest -= 2.0f * dot(rest, nrm) * nrm;
        a = h, b = h + rest;
    }
    return along + b;
}

inline float3 walk_restricted(thread Walk& w, uint s, float dt, uint nsub, uint kind, float size, float3 ax, float3 c) {
    const float h = dt / float(nsub), sd = w.l[0];
    float3 r = float3(w.x, w.y, w.z), acc = float3(0.0f);
    for (uint k = 1; k <= nsub; ++k) {
        const uint4 a = philox(uint4(w.spin, s, k, 0u), w.seed), b = philox(uint4(w.spin, s, k, 1u), w.seed);
        const float2 z01 = box_muller(a.x, a.y), z23 = box_muller(a.z, a.w), z45 = box_muller(b.x, b.y);
        const float3 n1 = float3(z01.x, z01.y, z23.x), n2 = float3(z23.y, z45.x, z45.y);
        const float3 rb = c + fold(r - c, r + sqrt(2.0f * h) * sd * n1 - c, kind, size, ax);
        acc += 0.5f * (r + rb) + sqrt(2.0f * h / 12.0f) * sd * n2;
        r = rb;
    }
    w.x = r.x, w.y = r.y, w.z = r.z;
    return acc / float(nsub);
}

struct Geom {
    uint kind = 0;
    float size = 0.0f;
    float3 ax = float3(0.0f), c = float3(0.0f);
};

// The per-spin state and segment step of both kernels: plain Bloch, or with the bound pool.
struct State {
    float mx, my, mz, mb;
    float lx = 0.0f, ly = 0.0f, lz = 0.0f, lb = 0.0f;  // low parts of a run of RF segments (two_sum)
    Spin p;
    Pool b;
    Cache cache;
    ZCache zc;
    Conc conc;
};

// V: 0 plain Bloch (no field model, no bound pool), 1 rate-model bound pool and/or field model,
// 2 generalized Bloch bound pool (and/or field model), 3 free exchanging pools (and/or the rate-model
// bound pool, field model). Compiled separately so the plain kernels
// carry none of the others' per-thread state.
struct Move {
    bool diffuse = false, move = false, maps = false;
    float3 v = float3(0.0f), d = float3(0.0f);
    float db0t = 0.0f, b1t = 1.0f;  // the tissue-attached parts (maps add / multiply)
};

// Trilinear sample of channel k of the scanner-frame maps (maps.sample): mg = origin (3), spacing (3),
// dims nx ny nz (3); maps (K, nz, ny, nx); clamped to the grid's edges.
inline float map_sample(constant float* maps, constant float* mg, uint k, float3 p) {
    const int nx = int(mg[6]), ny = int(mg[7]), nz = int(mg[8]);
    const float3 hi = float3(nx - 1, ny - 1, nz - 1);
    const float3 f = min(max((p - float3(mg[0], mg[1], mg[2])) / float3(mg[3], mg[4], mg[5]), float3(0.0f)), hi);
    const int3 i0 = int3(min(floor(f), max(hi - 1.0f, float3(0.0f))));
    const float3 w = f - float3(i0);
    const int3 i1 = min(i0 + 1, int3(hi));
    constant float* g = maps + uint(k) * uint(nx * ny * nz);
    float out = 0.0f;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
                const int x = dx ? i1.x : i0.x, y = dy ? i1.y : i0.y, z = dz ? i1.z : i0.z;
                const float wt = (dx ? w.x : 1.0f - w.x) * (dy ? w.y : 1.0f - w.y) * (dz ? w.z : 1.0f - w.z);
                out += wt * g[(z * ny + y) * nx + x];
            }
    return out;
}

// ---- the field model at a moving spin's true position (fields.spin_fields per segment). fm: nonlinearity
// flag, L, R0, f0, z asymmetry, then the coefficients A, B (3, L, L) with R0 N_nm folded in (zero for m > n).
// The solid harmonics by the recurrences of fields.solid_harmonics, rolling (no arrays).
inline float3 nl_displacement(float3 r, constant float* fm) {
    const int L = int(fm[1]);
    constant float* A = fm + 5;
    constant float* B = A + 3 * L * L;
    const float x = r.x / fm[2], y = r.y / fm[2], z = r.z / fm[2], r2 = x * x + y * y + z * z;
    float3 d = float3(0.0f);
    float cm = 1.0f, sm = 0.0f, qmm = 1.0f;  // (x + i y)^m, (2m - 1)!!
    for (int m = 0; m < L; ++m) {
        if (m > 0) {
            const float c2 = x * cm - y * sm;
            sm = x * sm + y * cm, cm = c2, qmm *= float(2 * m - 1);
        }
        float qa = 0.0f, qb = qmm;  // q_{n-1}, q_n (r^n P_n^m / rho^m)
        for (int n = m; n < L; ++n) {
            if (n == m + 1) {
                qa = qb, qb = float(2 * m + 1) * z * qa;
            } else if (n > m + 1) {
                const float qn = (float(2 * n - 1) * z * qb - float(n + m - 1) * r2 * qa) / float(n - m);
                qa = qb, qb = qn;
            }
            const float tc = qb * cm, ts = qb * sm;
            const int k = n * L + m;
            d += float3(tc * A[k] + ts * B[k], tc * A[L * L + k] + ts * B[L * L + k], tc * A[2 * L * L + k] + ts * B[2 * L * L + k]);
        }
    }
    return d;
}

// concomitant coefficients at true position r (fields.concomitant_coefficients / transverse_coefficients)
inline void conc_at(float3 r, constant float* fm, uint mode, thread Conc& c) {
    const float f0 = fm[3], a = fm[4];
    if (mode == 1) {
        const float h = 0.5f / f0;
        c.c[0] = r.z * r.z * h, c.c[1] = r.z * r.z * h, c.c[2] = (a * a * r.x * r.x + (1.0f - a) * (1.0f - a) * r.y * r.y) * h;
        c.c[3] = 0.0f, c.c[4] = -a * r.x * r.z / f0, c.c[5] = -(1.0f - a) * r.y * r.z / f0;
    } else {
        c.c[0] = r.z, c.c[1] = 0.0f, c.c[2] = -a * r.x, c.c[3] = 0.0f, c.c[4] = r.z, c.c[5] = -(1.0f - a) * r.y, c.c[6] = f0;
    }
}

template <int V, bool DIFF, uint NP>
inline void advance(thread State& st, thread Modes& md, thread BMS& bm, thread Walk& wk, thread const Geom& gm, thread const Move& mv,
                    constant float* segs, constant int* flags, constant float* frf, uint s, uint mt,
                    constant float* line, constant float* log_f, uint mode, constant int* nsub, constant float* modes,
                    uint nj, constant int* wsub, constant float* mseg, constant float* maps, constant float* mg,
                    constant float* fm, uint fmove) {
    constant float* q = segs + 8 * s;
    if (DIFF) {  // a moving spin: its effective position for this segment; position-keyed caches are stale
        float3 eff = !mv.diffuse ? float3(wk.x, wk.y, wk.z)
                     : gm.kind ? walk_restricted(wk, s, q[0], uint(wsub[s]), gm.kind, gm.size, gm.ax, gm.c)
                               : walk_segment(wk, s, q[0]);
        if (mv.move) {  // motion.positions: A (r + v t + d w) + b at the segment's mid-time
            constant float* ms = mseg + 14 * s;
            eff += mv.v * ms[0] + mv.d * ms[1];
            eff = float3(ms[2] * eff.x + ms[3] * eff.y + ms[4] * eff.z, ms[5] * eff.x + ms[6] * eff.y + ms[7] * eff.z,
                         ms[8] * eff.x + ms[9] * eff.y + ms[10] * eff.z) + float3(ms[11], ms[12], ms[13]);
        }
        if (mv.maps) {  // the scanner's B0 and B1+ where the spin is
            st.p.db0 = mv.db0t + map_sample(maps, mg, 0u, eff);
            st.p.b1 = mv.b1t * map_sample(maps, mg, 1u, eff);
        }
        if (fmove) {  // under the field model eff is a true position: encode it, and its concomitant coefficients
            if (V != 0 && mode) conc_at(eff, fm, mode, st.conc);
            if (fm[0] > 0.5f) eff += nl_displacement(eff, fm);
        }
        st.p.px = eff.x, st.p.py = eff.y, st.p.pz = eff.z;
        st.cache.dt = -1.0f, md.k_dt = -1.0f;
    }
    const float bz = offset(st.p, st.conc, V == 0 ? 0u : mode, q);
    if (V == 0) step(q, flags[s], bz, st.mx, st.my, st.mz, st.lx, st.ly, st.lz, st.p, st.cache);
    else if (V == 3) step_bm<NP>(q, flags[s], bz, mt ? frf[s] : 0.0f, bm, st.p, st.b, line, log_f, st.mx, st.my, st.mz);
    else if (V == 2) step_gen(q, flags[s], bz, uint(nsub[s]), st.mx, st.my, st.mz, st.mb, st.lx, st.ly, st.lz, st.lb, md, st.p,
                              st.b, modes, nj, st.cache, st.zc);
    else if (mt) step_mt(q, flags[s], bz, frf[s], st.mx, st.my, st.mz, st.mb, st.lx, st.ly, st.lz, st.lb, st.p, st.b, line,
                         log_f, st.cache, st.zc);
    else step(q, flags[s], bz, st.mx, st.my, st.mz, st.lx, st.ly, st.lz, st.p, st.cache);
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

"""

_KERNELS = r"""
kernel void bloch_chunk@SUFFIX@(device float* M            [[buffer(0)]],  // (3, N) magnetization, in/out
                                constant float* spins      [[buffer(1)]],  // (8, N) px py pz pd r1 r2 db0 b1
                                constant float* sens       [[buffer(2)]],  // (2C, N) per coil: real, imag
                                constant float* segs       [[buffer(3)]],  // (S, 8) dt rfx rfy gx gy gz psi b0
                                constant int* flags        [[buffer(4)]],  // (S,) 1 rf, 2 adc, 4 spoil
                                constant int* adc_row      [[buffer(5)]],  // (S,) signal row of an ADC segment
                                device atomic_float* sig   [[buffer(6)]],  // (n_adc, C, 2)
                                device float* MB           [[buffer(7)]],  // (N,) bound-pool Mz, in/out (mt)
                                constant float* pool       [[buffer(8)]],  // (5, N) f k r1b t2b dw (mt)
                                constant float* frf        [[buffer(9)]],  // (S,) RF frequency [Hz] (mt 1)
                                constant float* line       [[buffer(10)]],  // (6,) lineshape (mt 1)
                                constant float* log_f      [[buffer(11)]],  // lineshape table (mt 1)
                                constant float* conc       [[buffer(12)]],  // (7, N) concomitant-field coefficients
                                device float* U            [[buffer(13)]],  // (2J, N) generalized pool modes, in/out (mt 2)
                                constant float* modes      [[buffer(14)]],  // (J, 4) Re p, Im p, Re c, Im c (mt 2)
                                constant int* nsub         [[buffer(15)]],  // (S,) RF sub-steps (mt 2)
                                device float* P            [[buffer(16)]],  // (3, N) walker positions, in/out (diffusion)
                                constant float* chol       [[buffer(17)]],  // (6, N) Cholesky factors of D (diffusion)
                                constant int* wsub         [[buffer(18)]],  // (S,) restricted-walk sub-steps
                                constant float* geom       [[buffer(19)]],  // (7, N) size, axis, centre (restriction)
                                constant float* mseg       [[buffer(20)]],  // (S, 14) motion: t, w, A (row-major), b
                                constant float* mspin      [[buffer(21)]],  // (6, N) motion: velocity, displacement
                                constant float* maps       [[buffer(22)]],  // (K, nz, ny, nx) b0, b1, coils (moving spins in FieldMaps)
                                constant float* mg         [[buffer(23)]],  // (9,) map origin, spacing, dims
                                constant uint* prm         [[buffer(24)]],  // (16,) n n_coils s0 s1 n_tiles n_pad mt mode nj seed_lo seed_hi kind move diffuse mapped fmove
                                device float* BMv          [[buffer(25)]],  // (2 BM_NC, N) free-pool state hi, lo, in/out (V 3)
                                constant float* bmp        [[buffer(26)]],  // (5 MAX_POOLS, N) per free pool: f k r1 r2 dw (V 3)
                                constant float* fm         [[buffer(27)]],  // field model of moving spins (nl_displacement, conc_at)
                                device float* L            [[buffer(28)]],  // (4, N) low parts of m, mb (two_sum), in/out
                                uint i    [[thread_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg   [[simdgroup_index_in_threadgroup]],
                                uint nsg  [[simdgroups_per_threadgroup]],
                                uint tid  [[thread_index_in_threadgroup]],
                                uint ntg  [[threads_per_threadgroup]]) {
    const uint n = prm[0], n_coils = prm[1], s0 = prm[2], s1 = prm[3], n_tiles = prm[4], n_pad = prm[5], mt = prm[6], mode = prm[7], nj = prm[8], seed_lo = prm[9], seed_hi = prm[10], kind = prm[11], move = prm[12], diffuse = prm[13], mapped = prm[14], fmove = prm[15];

    // ADC samples are buffered per simdgroup in threadgroup memory and reduced a block at a time:
    // two barriers per block of samples instead of two per sample, and the final sums are spread
    // over all threads of the group.
    threadgroup float partial[TG_FLOATS];
    threadgroup int block_rows[TG_FLOATS / (MAX_SG * 2)];
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
        wk.seed = uint2(seed_lo, seed_hi), wk.spin = j;
    }
    Move mv;
    if (@DIFF@) {
        mv.diffuse = diffuse != 0, mv.move = move != 0, mv.maps = mapped != 0;
        mv.db0t = st.p.db0, mv.b1t = st.p.b1;
        if (mv.move && active) {
            mv.v = float3(mspin[j], mspin[n + j], mspin[2 * n + j]);
            mv.d = float3(mspin[3 * n + j], mspin[4 * n + j], mspin[5 * n + j]);
        }
    }
    Geom gm;
    if (@DIFF@ && kind) {
        gm.kind = kind, gm.size = geom[j];
        gm.ax = float3(geom[n + j], geom[2 * n + j], geom[3 * n + j]);
        gm.c = float3(geom[4 * n + j], geom[5 * n + j], geom[6 * n + j]);
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
    thread float &mx = st.mx, &my = st.my, &mz = st.mz;
    const uint nv = 2 * n_coils;
    const uint block = TG_FLOATS / (MAX_SG * nv);  // samples buffered per flush
    uint nb = 0;

    for (uint s = s0; s < s1; ++s) {
        constant float* q = segs + 8 * s;
        const int f = flags[s];
        advance<@V@, @DIFF@, @NP@>(st, md, bm, wk, gm, mv, segs, flags, frf, s, mt, line, log_f, mode, nsub, modes, nj, wsub, mseg, maps, mg, fm, fmove);
        if (f & 2) {  // ADC sample: simdgroup sums into the block buffer, flushed when full
            const float cp = precise::cos(q[6]), sp = precise::sin(q[6]);
            for (uint v = 0; v < nv; v += 2) {
                float sr = active ? sens[v * n + j] : 0.0f, si = active ? sens[(v + 1) * n + j] : 0.0f;
                if (@DIFF@ && mapped > 1u && active) {  // mapped coils, where the spin is now
                    const float3 at = float3(st.p.px, st.p.py, st.p.pz);
                    sr = map_sample(maps, mg, 2u + v, at), si = map_sample(maps, mg, 3u + v, at);
                }
                const float re = (mx + st.lx) * sr - (my + st.ly) * si, im = (mx + st.lx) * si + (my + st.ly) * sr;
                const float tr = simd_sum(re * cp + im * sp), ti = simd_sum(im * cp - re * sp);
                if (lane == 0) {
                    partial[(sg * block + nb) * nv + v] = tr;
                    partial[(sg * block + nb) * nv + v + 1] = ti;
                }
            }
            if (tid == 0) block_rows[nb] = adc_row[s];
            if (++nb == block) { flush(partial, block_rows, sig, nb, nv, nsg, tid, ntg); nb = 0; }
        }
        if (f & 4) { mx = 0.0f; my = 0.0f; st.lx = 0.0f; st.ly = 0.0f; }
    }
    if (nb > 0) flush(partial, block_rows, sig, nb, nv, nsg, tid, ntg);
    if (active) {
        M[j] = mx; M[n + j] = my; M[2 * n + j] = mz;
        L[j] = st.lx, L[n + j] = st.ly, L[2 * n + j] = st.lz, L[3 * n + j] = st.lb;
        if (@V@ != 0 && mt) MB[j] = st.mb;
        for (uint m = 0; @V@ == 2 && m < nj; ++m) U[2 * m * n + j] = md.ur[m], U[(2 * m + 1) * n + j] = md.ui[m];
        if (@DIFF@) P[j] = wk.x, P[n + j] = wk.y, P[2 * n + j] = wk.z;
        if (@V@ == 3) bm_store(bm.hi, BMv, 0, n, j), bm_store(bm.lo, BMv, BM_NC, n, j);
    }
}

kernel void bloch_chunk_mma@SUFFIX@(device float* M            [[buffer(0)]],  // (3, N) magnetization, in/out
                                    constant float* spins      [[buffer(1)]],  // (8, N) px py pz pd r1 r2 db0 b1
                                    device const float* sens   [[buffer(2)]],  // (2, 8T, N_pad): real, imag; zero-padded
                                    constant float* segs       [[buffer(3)]],  // (S, 8) dt rfx rfy gx gy gz psi b0
                                    constant int* flags        [[buffer(4)]],  // (S,) 1 rf, 2 adc, 4 spoil
                                    constant int* adc_row      [[buffer(5)]],  // (S,) signal row of an ADC segment
                                    device atomic_float* sig   [[buffer(6)]],  // (n_adc, C, 2)
                                    device float* MB           [[buffer(7)]],  // (N,) bound-pool Mz, in/out (mt)
                                    constant float* pool       [[buffer(8)]],  // (5, N) f k r1b t2b dw (mt)
                                    constant float* frf        [[buffer(9)]],  // (S,) RF frequency [Hz] (mt 1)
                                    constant float* line       [[buffer(10)]],  // (6,) lineshape (mt 1)
                                    constant float* log_f      [[buffer(11)]],  // lineshape table (mt 1)
                                    constant float* conc       [[buffer(12)]],  // (7, N) concomitant-field coefficients
                                    device float* U            [[buffer(13)]],  // (2J, N) generalized pool modes, in/out (mt 2)
                                    constant float* modes      [[buffer(14)]],  // (J, 4) Re p, Im p, Re c, Im c (mt 2)
                                    constant int* nsub         [[buffer(15)]],  // (S,) RF sub-steps (mt 2)
                                    device float* P            [[buffer(16)]],  // (3, N) walker positions, in/out (diffusion)
                                    constant float* chol       [[buffer(17)]],  // (6, N) Cholesky factors of D (diffusion)
                                    constant int* wsub         [[buffer(18)]],  // (S,) restricted-walk sub-steps
                                    constant float* geom       [[buffer(19)]],  // (7, N) size, axis, centre (restriction)
                                    constant float* mseg       [[buffer(20)]],  // (S, 14) motion: t, w, A (row-major), b
                                    constant float* mspin      [[buffer(21)]],  // (6, N) motion: velocity, displacement
                                    constant float* maps       [[buffer(22)]],  // (K, nz, ny, nx) b0, b1, coils (moving spins in FieldMaps)
                                    constant float* mg         [[buffer(23)]],  // (9,) map origin, spacing, dims
                                    constant uint* prm         [[buffer(24)]],  // (16,) n n_coils s0 s1 n_tiles n_pad mt mode nj seed_lo seed_hi kind move diffuse mapped fmove
                                    device float* BMv          [[buffer(25)]],  // (2 BM_NC, N) free-pool state hi, lo, in/out (V 3)
                                    constant float* bmp        [[buffer(26)]],  // (5 MAX_POOLS, N) per free pool: f k r1 r2 dw (V 3)
                                    constant float* fm         [[buffer(27)]],  // field model of moving spins (nl_displacement, conc_at)
                                    device float* L            [[buffer(28)]],  // (4, N) low parts of m, mb (two_sum), in/out
                                    uint i    [[thread_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg   [[simdgroup_index_in_threadgroup]],
                                    uint nsg  [[simdgroups_per_threadgroup]],
                                    uint tid  [[thread_index_in_threadgroup]],
                                    uint ntg  [[threads_per_threadgroup]]) {
    const uint n = prm[0], n_coils = prm[1], s0 = prm[2], s1 = prm[3], n_tiles = prm[4], n_pad = prm[5], mt = prm[6], mode = prm[7], nj = prm[8], seed_lo = prm[9], seed_hi = prm[10], kind = prm[11], move = prm[12], diffuse = prm[13], mapped = prm[14], fmove = prm[15];

    threadgroup float buf[MAX_SG * MMA_REGION];
    threadgroup int rows[8];
    threadgroup float psi[8];
    threadgroup float* mine = buf + sg * MMA_REGION;
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
        wk.seed = uint2(seed_lo, seed_hi), wk.spin = j;
    }
    Move mv;
    if (@DIFF@) {
        mv.diffuse = diffuse != 0, mv.move = move != 0, mv.maps = mapped != 0;
        mv.db0t = st.p.db0, mv.b1t = st.p.b1;
        if (mv.move && active) {
            mv.v = float3(mspin[j], mspin[n + j], mspin[2 * n + j]);
            mv.d = float3(mspin[3 * n + j], mspin[4 * n + j], mspin[5 * n + j]);
        }
    }
    Geom gm;
    if (@DIFF@ && kind) {
        gm.kind = kind, gm.size = geom[j];
        gm.ax = float3(geom[n + j], geom[2 * n + j], geom[3 * n + j]);
        gm.c = float3(geom[4 * n + j], geom[5 * n + j], geom[6 * n + j]);
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
    thread float &mx = st.mx, &my = st.my, &mz = st.mz;
    const uint j0 = i - lane;
    uint nb = 0;

    for (uint s = s0; s < s1; ++s) {
        constant float* q = segs + 8 * s;
        const int f = flags[s];
        advance<@V@, @DIFF@, @NP@>(st, md, bm, wk, gm, mv, segs, flags, frf, s, mt, line, log_f, mode, nsub, modes, nj, wsub, mseg, maps, mg, fm, fmove);
        if (f & 2) {  // ADC sample: column nb of this simdgroup's (32 spins x 8 samples) block
            mine[8 * lane + nb] = mx + st.lx;
            mine[256 + 8 * lane + nb] = my + st.ly;
            mine[512 + 8 * lane + nb] = -(my + st.ly);
            if (tid == 0) { rows[nb] = adc_row[s]; psi[nb] = q[6]; }
            if (++nb == 8) { flush_mma(buf, mine, rows, psi, sens, sig, nb, n_coils, n_tiles, n_pad, j0, nsg, tid, ntg); nb = 0; }
        }
        if (f & 4) { mx = 0.0f; my = 0.0f; st.lx = 0.0f; st.ly = 0.0f; }
    }
    if (nb > 0) flush_mma(buf, mine, rows, psi, sens, sig, nb, n_coils, n_tiles, n_pad, j0, nsg, tid, ntg);
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

_ADJOINT_COMMON = r"""
// ==== Adjoint (reverse mode), as fused/kernel.h's Adjoint, one thread per spin. Segment s maps the
// magnetization affinely, m_s = A_s m_{s-1} + b_s; with lambda = dL/dm after segment s (spoil
// included) the sweep is lambda_{s-1} = A_s^T (P_s lambda + ADC term), and the parameter gradients of
// segment s follow from lambda and m_{s-1}. The forward stores the state (hi and lo) before every K1
// segments (one dispatch each); the backward replays one such interval into R and sweeps it in reverse.
// Per-spin gradients are summed in registers and flushed into GS at every segment index that is a
// multiple of 64 (K1 is one too), so the grouping, and the result, does not depend on K1. Segment
// gradients are reduced over the threadgroup (simd_sum, threadgroup memory) into one partial per group
// and segment, then summed over groups in order by seg_reduce: no atomics, the same result every run.

struct Acc { float px, py, pz, db0, pd, r1, r2, b1, c[7], bf, bk, br1, bt2, bdw; };
#define N_ACC 20
#define RBLOCK 4  // segments per recomputed block (divides 64; tg: RBLOCK x 4 x 256 floats = 16 KB)

inline void acc_flush(thread Acc& g, device float* GS, uint n, uint j) {
    const float v[N_ACC] = {g.px, g.py, g.pz, g.db0, g.pd, g.r1, g.r2, g.b1, g.c[0], g.c[1], g.c[2], g.c[3], g.c[4],
                            g.c[5], g.c[6], g.bf, g.bk, g.br1, g.bt2, g.bdw};
    for (uint k = 0; k < N_ACC; ++k) GS[k * n + j] += v[k];
    g = Acc{};
}

struct SegG { float dt, rx, ry, gx, gy, gz, off, frf; };

// rotation factors c, s1, s2 of w and f1, f2 (fused rotation_derivatives); float32: the series below
// phi^2 = 1, where (c - s1) / phi^2 and (s1 - 2 s2) / phi^2 cancel
inline void rot_factors(float wx, float wy, float wz, thread float& c, thread float& s1, thread float& s2,
                        thread float& f1, thread float& f2) {
    const float p2 = wx * wx + wy * wy + wz * wz, phi = sqrt(p2);
    if (phi < 1e-3f) {
        c = 1.0f - 0.5f * p2, s1 = 1.0f - p2 / 6.0f, s2 = 0.5f - p2 / 24.0f;
    } else {
        const float sh = precise::sin(0.5f * phi) / phi;
        c = precise::cos(phi), s1 = precise::sin(phi) / phi, s2 = 2.0f * sh * sh;
    }
    if (p2 < 1.0f) {
        f1 = -1.0f / 3 + p2 * (1.0f / 30 + p2 * (-1.0f / 840 + p2 * (1.0f / 45360 + p2 * (-1.0f / 3991680))));
        f2 = -1.0f / 12 + p2 * (1.0f / 180 + p2 * (-1.0f / 6720 + p2 * (1.0f / 453600 + p2 * (-1.0f / 47900160))));
    } else {
        f1 = (c - s1) / p2, f2 = (s1 - 2.0f * s2) / p2;
    }
}

// n = R(w) m; given a = dL/dn: b = R^T a = dL/dm and gw = dL/dw (fused rodrigues_reverse)
inline void rod_reverse(float c, float s1, float s2, float f1, float f2, float wx, float wy, float wz, float mx,
                        float my, float mz, float ax, float ay, float az, thread float& bx, thread float& by,
                        thread float& bz, thread float& gwx, thread float& gwy, thread float& gwz) {
    const float wa = wx * ax + wy * ay + wz * az, dot = wx * mx + wy * my + wz * mz;
    bx = c * ax - s1 * (wy * az - wz * ay) + s2 * wx * wa;
    by = c * ay - s1 * (wz * ax - wx * az) + s2 * wy * wa;
    bz = c * az - s1 * (wx * ay - wy * ax) + s2 * wz * wa;
    const float am = ax * mx + ay * my + az * mz;
    const float awm = ax * (wy * mz - wz * my) + ay * (wz * mx - wx * mz) + az * (wx * my - wy * mx);
    const float k = -s1 * am + f1 * awm + f2 * wa * dot;
    gwx = k * wx + s1 * (my * az - mz * ay) + s2 * (wa * mx + dot * ax);
    gwy = k * wy + s1 * (mz * ax - mx * az) + s2 * (wa * my + dot * ay);
    gwz = k * wz + s1 * (mx * ay - my * ax) + s2 * (wa * mz + dot * az);
}

// The off-resonance of segment q and its partials (fused Adjoint::field): d/d(lin) for lin = pos.g + db0 +
// b0_offset, d/dg; exact mode also ex = tx / S, ey = ty / S and d/df0.
struct OffD { float bz, dlin, dgx, dgy, dgz, ex, ey, ef0; };

inline OffD offset_d(thread const Spin& p, thread const Conc& k, uint mode, constant float* q) {
    const float gx = q[3], gy = q[4], gz = q[5];
    const float lin = p.px * gx + p.py * gy + p.pz * gz + p.db0 + q[7];
    OffD o;
    o.ex = o.ey = o.ef0 = 0.0f, o.dlin = 1.0f;
    if (mode == 0) {
        o.bz = lin, o.dgx = p.px, o.dgy = p.py, o.dgz = p.pz;
    } else if (mode == 1) {
        o.bz = lin + (k.c[0] * (gx * gx) + k.c[1] * (gy * gy) + k.c[2] * (gz * gz) + k.c[3] * (gx * gy) + k.c[4] * (gx * gz) +
                      k.c[5] * (gy * gz));
        o.dgx = p.px + 2.0f * k.c[0] * gx + k.c[3] * gy + k.c[4] * gz;
        o.dgy = p.py + 2.0f * k.c[1] * gy + k.c[3] * gx + k.c[5] * gz;
        o.dgz = p.pz + 2.0f * k.c[2] * gz + k.c[4] * gx + k.c[5] * gy;
    } else {
        const float tx = k.c[0] * gx + k.c[1] * gy + k.c[2] * gz, ty = k.c[3] * gx + k.c[4] * gy + k.c[5] * gz;
        const float t2 = tx * tx + ty * ty, u = k.c[6] + lin, S = sqrt(u * u + t2);
        o.bz = lin + t2 / (S + u);
        o.dlin = u / S, o.ex = tx / S, o.ey = ty / S, o.ef0 = -t2 / (S * (S + u));
        o.dgx = o.dlin * p.px + o.ex * k.c[0] + o.ey * k.c[3];
        o.dgy = o.dlin * p.py + o.ex * k.c[1] + o.ey * k.c[4];
        o.dgz = o.dlin * p.pz + o.ex * k.c[2] + o.ey * k.c[5];
    }
    return o;
}

// gb = dL/d(offset) into the per-spin gradients and this spin's share of the segment gradients
inline void offset_grads(float gb, thread const OffD& o, uint mode, constant float* q, thread Acc& g, thread SegG& sg) {
    const float gx = q[3], gy = q[4], gz = q[5], gl = gb * o.dlin;
    g.db0 += gl, g.px += gl * gx, g.py += gl * gy, g.pz += gl * gz;
    sg.gx += gb * o.dgx, sg.gy += gb * o.dgy, sg.gz += gb * o.dgz, sg.off += gl;
    if (mode == 1) {
        g.c[0] += gb * gx * gx, g.c[1] += gb * gy * gy, g.c[2] += gb * gz * gz;
        g.c[3] += gb * gx * gy, g.c[4] += gb * gx * gz, g.c[5] += gb * gy * gz;
    } else if (mode == 2) {
        const float a = gb * o.ex, b = gb * o.ey;
        g.c[0] += a * gx, g.c[1] += a * gy, g.c[2] += a * gz, g.c[3] += b * gx, g.c[4] += b * gy, g.c[5] += b * gz;
        g.c[6] += gb * o.ef0;
    }
}

// the 2 x 2 exchange step with what its reverse needs (fused ZStep)
struct ZR { float a11, a12, a21, a22, m, h, p, det, z1, z2, chm1, sh; };

inline ZR zr_make(float t, thread const Spin& sp, thread const Pool& b, float w) {
    ZR z;
    const float kf = b.k * b.f;
    z.a11 = -(sp.r1 + kf), z.a12 = b.k, z.a21 = kf, z.a22 = -(b.r1b + b.k + w);
    const float c1 = sp.r1 * sp.pd, c2 = b.r1b * b.f * sp.pd;
    z.m = 0.5f * (z.a11 + z.a22), z.h = 0.5f * (z.a11 - z.a22), z.p = z.h * z.h + z.a12 * z.a21;
    const float q = sqrt(z.p);
    z.det = z.a11 * z.a22 - z.a12 * z.a21;
    z.z1 = -(z.a22 * c1 - z.a12 * c2) / z.det, z.z2 = -(z.a11 * c2 - z.a21 * c1) / z.det;
    z.chm1 = 0.5f * (expm1_((z.m + q) * t) + expm1_((z.m - q) * t));
    if (2.0f * q * t < 1.0f) {
        const float lo = precise::exp((z.m - q) * t), e2q = expm1_(2.0f * q * t);
        z.sh = q > 0.0f ? lo * 0.5f * e2q / q : t * precise::exp(z.m * t);
    } else {
        z.sh = 0.5f * (precise::exp((z.m + q) * t) - precise::exp((z.m - q) * t)) / q;
    }
    return z;
}

inline void zr_apply(thread const ZR& z, thread float& u, thread float& v) {
    const float d1 = u - z.z1, d2 = v - z.z2;
    const float n1 = z.chm1 * d1 + z.sh * (z.h * d1 + z.a12 * d2), n2 = z.chm1 * d2 + z.sh * (z.a21 * d1 - z.h * d2);
    u += n1, v += n2;
}

// (l1, l2) = dL/dz' in, dL/dz out for input z = (u, v); y: dL/d(a11, a12, a21, a22, c1, c2), yt = dL/dt
inline void zr_reverse(thread const ZR& z, float t, float u, float v, thread float& l1, thread float& l2,
                       thread float& yt, thread float* y) {
    const float ch = 1.0f + z.chm1, x2 = z.p * t * t;
    const float sg = x2 < 0.25f
        ? precise::exp(z.m * t) * 0.5f * t * t * t *
              (1.0f / 3 + x2 * (1.0f / 30 + x2 * (1.0f / 840 + x2 * (1.0f / 45360 + x2 * (1.0f / 3991680)))))
        : (t * ch - z.sh) / (2.0f * z.p);
    const float d1 = u - z.z1, d2 = v - z.z2;
    const float e1 = z.chm1 * l1 + z.sh * (z.h * l1 + z.a21 * l2), e2 = z.chm1 * l2 + z.sh * (z.a12 * l1 - z.h * l2);
    const float ych = l1 * d1 + l2 * d2, ysh = l1 * (z.h * d1 + z.a12 * d2) + l2 * (z.a21 * d1 - z.h * d2);
    float yh = z.sh * (l1 * d1 - l2 * d2), ya12 = z.sh * l1 * d2, ya21 = z.sh * l2 * d1;
    const float ym = t * (ych * ch + ysh * z.sh), yp = ych * 0.5f * t * z.sh + ysh * sg;
    yt = ych * (z.m * ch + z.p * z.sh) + ysh * (z.m * z.sh + ch);
    yh += 2.0f * z.h * yp, ya12 += z.a21 * yp, ya21 += z.a12 * yp;
    const float n1 = (z.a22 * e1 - z.a21 * e2) / z.det, n2 = (z.a11 * e2 - z.a12 * e1) / z.det;
    y[0] = 0.5f * (ym + yh) + n1 * z.z1, y[3] = 0.5f * (ym - yh) + n2 * z.z2;
    y[1] = ya12 + n1 * z.z2, y[2] = ya21 + n2 * z.z1, y[4] = n1, y[5] = n2;
    l1 += e1, l2 += e2;
}

// dL/d(A, c) of the exchange step into the pool parameters; returns dL/dW (fused exchange_grads)
inline float exchange_grads(thread const float* y, thread const Spin& sp, thread const Pool& b, thread Acc& g) {
    const float f = b.f, k = b.k, rb = b.r1b, pd = sp.pd;
    g.r1 += -y[0] + pd * y[4];
    g.pd += sp.r1 * y[4] + rb * f * y[5];
    g.bf += k * (y[2] - y[0]) + rb * pd * y[5];
    g.bk += f * (y[2] - y[0]) + y[1] - y[3];
    g.br1 += -y[3] + f * pd * y[5];
    return -y[3];
}

// the lineshape (as lineshape()) and its partials d/dt2b, d/ddelta (fused Lineshape::eval)
inline float lineshape_d(float delta, float t2b, constant float* line, constant float* log_f, thread float& g_t2,
                         thread float& g_d) {
    if (line[0] < 1.5f) {
        const float x = TWO_PI * delta * t2b, den = 1.0f + x * x;
        g_t2 = (1.0f - x * x) / (den * den), g_d = -t2b * 2.0f * x * TWO_PI * t2b / (den * den);
        return t2b / den;
    }
    const float ad = fabs(delta);
    const bool cut = ad < line[1];
    const float d = cut ? line[1] : ad, x = TWO_PI * d * t2b;
    if (x > 30.0f) { g_t2 = g_d = 0.0f; return 0.0f; }
    const int n = int(line[5]);
    const bool low = x < line[4];
    float pos = (precise::log(low ? line[4] : x) - line[2]) / line[3];
    const bool clamped = pos < 0.0f || pos > float(n - 1);
    pos = clamp(pos, 0.0f, float(n - 1));
    const int i = min(int(floor(pos)), n - 2);
    const float a = pos - float(i);
    const float val = precise::exp(log_f[i] * (1.0f - a) + log_f[i + 1] * a);
    const float dval_dx = low || clamped ? 0.0f : val * (log_f[i + 1] - log_f[i]) / line[3] / x;
    g_t2 = val + t2b * dval_dx * TWO_PI * d;
    g_d = cut ? 0.0f : t2b * dval_dx * TWO_PI * t2b * (delta > 0.0f ? 1.0f : (delta < 0.0f ? -1.0f : 0.0f));
    return t2b * val;
}

// Free-segment factors of the reverse sweep, kept across runs of identical segments (as the forward's Cache:
// readout samples): cos, sin of the precession, expm1 of the relaxation, the unsaturated exchange step.
struct RCache { float dt = -1.0f, gx = 0.0f, gy = 0.0f, gz = 0.0f, b0 = 0.0f, c, sn, d1, d2; float zdt = -1.0f; ZR zs; };

// Reverse through segment q from the state (x0, y0, z0, v0) before it; lambda = dL/d(its output) in, dL/d(its
// input) out. V 0: plain; V 1: the field model (mode) and / or the rate-model bound pool (mt).
template <int V>
inline void reverse_seg(constant float* q, int f, float f_rf, float x0, float y0, float z0, float v0,
                        thread const Spin& p, thread const Pool& b, thread const Conc& cf, uint mode, bool mt,
                        constant float* line, constant float* log_f, thread float& lx, thread float& ly,
                        thread float& lz, thread float& lb, thread Acc& g, thread SegG& sg, thread RCache& rc) {
    const float w_per_s = -TWO_PI, dt = q[0], w = w_per_s * dt;
    const uint md = V == 0 ? 0u : mode;
    const OffD o = offset_d(p, cf, md, q);
    float gb, sdt;
    if (f & 1) {
        const float h = 0.5f * dt, rx = q[1], ry = q[2];
        const float wx0 = w * rx, wy0 = w * ry, wx = wx0 * p.b1, wy = wy0 * p.b1, wz = w * o.bz;
        float c, s1, s2, f1, f2;
        rot_factors(wx, wy, wz, c, s1, s2, f1, f2);
        const float d2 = expm1_(-h * p.r2), e2 = 1.0f + d2;
        float gwx, gwy, gwz, srx = 0.0f, sry = 0.0f, sfrf = 0.0f, gb1x = 0.0f;
        if (V == 0 || !mt) {
            const float d1 = expm1_(-h * p.r1), e1 = 1.0f + d1;
            const float m1x = x0 + x0 * d2, m1y = y0 + y0 * d2, m1z = z0 + (z0 - p.pd) * d1;
            const float dot = wx * m1x + wy * m1y + wz * m1z;
            const float nx = c * m1x + s1 * (wy * m1z - wz * m1y) + s2 * wx * dot;
            const float ny = c * m1y + s1 * (wz * m1x - wx * m1z) + s2 * wy * dot;
            const float nz = c * m1z + s1 * (wx * m1y - wy * m1x) + s2 * wz * dot;
            float gd2 = lx * nx + ly * ny, gd1 = lz * (nz - p.pd), gpd = -lz * d1;
            const float ax = lx + lx * d2, ay = ly + ly * d2, az = lz + lz * d1;
            float bx, by, bz;
            rod_reverse(c, s1, s2, f1, f2, wx, wy, wz, m1x, m1y, m1z, ax, ay, az, bx, by, bz, gwx, gwy, gwz);
            gd2 += bx * x0 + by * y0, gd1 += bz * (z0 - p.pd), gpd -= bz * d1;
            lx = bx + bx * d2, ly = by + by * d2, lz = bz + bz * d1;
            g.r1 -= gd1 * h * e1, g.r2 -= gd2 * h * e2, g.pd += gpd;
            gb = gwz * w;
            sdt = w_per_s * gwz * o.bz - 0.5f * (gd1 * p.r1 * e1 + gd2 * p.r2 * e2);
        } else {  // the exchange step on (z, mb) in place of longitudinal relaxation, saturated at W = kw G
            float g_t2, g_d;
            const float G = lineshape_d(f_rf - o.bz - b.dw, b.t2b, line, log_f, g_t2, g_d);
            const float a2 = rx * rx + ry * ry, kw = TWO_PI * TWO_PI * p.b1 * p.b1 * a2;
            const ZR zs = zr_make(h, p, b, kw * G);
            const float m1x = x0 + x0 * d2, m1y = y0 + y0 * d2;
            float m1z = z0, v1 = v0;
            zr_apply(zs, m1z, v1);
            const float dot = wx * m1x + wy * m1y + wz * m1z;
            const float nx = c * m1x + s1 * (wy * m1z - wz * m1y) + s2 * wx * dot;
            const float ny = c * m1y + s1 * (wz * m1x - wx * m1z) + s2 * wy * dot;
            const float nz = c * m1z + s1 * (wx * m1y - wy * m1x) + s2 * wz * dot;
            float gd2 = lx * nx + ly * ny;
            const float ax = lx + lx * d2, ay = ly + ly * d2;
            float az = lz, lv = lb, yt, ya[6], yb[6];
            zr_reverse(zs, h, nz, v1, az, lv, yt, ya);
            float yt_sum = yt;
            float bx, by, bz;
            rod_reverse(c, s1, s2, f1, f2, wx, wy, wz, m1x, m1y, m1z, ax, ay, az, bx, by, bz, gwx, gwy, gwz);
            gd2 += bx * x0 + by * y0;
            lx = bx + bx * d2, ly = by + by * d2;
            zr_reverse(zs, h, z0, v0, bz, lv, yt, yb);
            lz = bz, lb = lv;
            yt_sum += yt;
            for (int k = 0; k < 6; ++k) ya[k] += yb[k];
            const float yw = exchange_grads(ya, p, b, g);
            const float yd = yw * kw * g_d;  // delta = f_rf - offset - dw
            g.bt2 += yw * kw * g_t2, g.bdw -= yd;
            sfrf = yd;
            const float ykw = yw * G * TWO_PI * TWO_PI;
            gb1x = ykw * 2.0f * p.b1 * a2;
            srx = ykw * (p.b1 * p.b1) * 2.0f * rx, sry = ykw * (p.b1 * p.b1) * 2.0f * ry;
            g.r2 -= gd2 * h * e2;
            gb = gwz * w - yd;
            sdt = w_per_s * gwz * o.bz - 0.5f * gd2 * p.r2 * e2 + 0.5f * yt_sum;
        }
        const float swx = gwx * p.b1, swy = gwy * p.b1;
        g.b1 += gwx * wx0 + gwy * wy0 + gb1x;
        sg.dt += sdt + w_per_s * (rx * swx + ry * swy);
        sg.rx += w * swx + srx, sg.ry += w * swy + sry, sg.frf += sfrf;
    } else {
        if (dt != rc.dt || q[3] != rc.gx || q[4] != rc.gy || q[5] != rc.gz || q[7] != rc.b0) {
            const float phi = w * o.bz;
            rc.c = precise::cos(phi), rc.sn = precise::sin(phi);
            rc.d1 = expm1_(-dt * p.r1), rc.d2 = expm1_(-dt * p.r2);
            rc.dt = dt, rc.gx = q[3], rc.gy = q[4], rc.gz = q[5], rc.b0 = q[7];
        }
        const float c = rc.c, sn = rc.sn, d2 = rc.d2, e2 = 1.0f + d2;
        const float ux = c * x0 - sn * y0, uy = sn * x0 + c * y0;
        const float gd2 = lx * ux + ly * uy, gphi = e2 * (ly * ux - lx * uy);
        const float ax = c * lx + sn * ly, ay = c * ly - sn * lx;
        lx = ax + ax * d2, ly = ay + ay * d2;
        g.r2 -= gd2 * dt * e2;
        gb = gphi * w;
        sdt = w_per_s * gphi * o.bz - gd2 * p.r2 * e2;
        if (V == 0 || !mt) {
            const float d1 = rc.d1, e1 = 1.0f + d1;
            const float gd1 = lz * (z0 - p.pd);
            g.pd -= lz * d1;
            lz = lz + lz * d1;
            g.r1 -= gd1 * dt * e1;
            sdt -= gd1 * p.r1 * e1;
        } else {
            if (dt != rc.zdt) rc.zs = zr_make(dt, p, b, 0.0f), rc.zdt = dt;
            float yt, y[6];
            zr_reverse(rc.zs, dt, z0, v0, lz, lb, yt, y);
            exchange_grads(y, p, b, g);
            sdt += yt;
        }
        sg.dt += sdt;
    }
    offset_grads(gb, o, md, q, g, sg);
}

kernel void seg_reduce(device const float* GSEG [[buffer(0)]],  // (simdgroups, K1, 8) partials of one interval
                       device float* OUT         [[buffer(1)]],  // (S, 8) segment gradients
                       constant uint* prm        [[buffer(2)]],  // simdgroups, K1, s0, segments in the interval
                       uint t [[thread_position_in_grid]]) {
    const uint G = prm[0], K = prm[1], s0 = prm[2], len = prm[3];
    if (t >= 8 * len) return;
    const uint slot = t / 8, k = t % 8;
    float hi = 0.0f, lo = 0.0f;
    for (uint gi = 0; gi < G; ++gi) two_sum(hi, lo, GSEG[(gi * K + slot) * 8 + k]);
    OUT[(s0 + slot) * 8 + k] = hi + lo;
}
"""

_ADJOINT_KERNEL = r"""
// The forward again over one interval, from its checkpoint, keeping the full state (hi and lo) before every
// RBLOCK-th segment in R; bloch_adjoint recomputes each block from there into threadgroup memory. Storing
// every state was DRAM-bound (12 B per spin and segment written and read again: 1 s of 3d-48's 2.5 s).
kernel void bloch_replay@SUFFIX@(device const float* CK  [[buffer(0)]],   // (8, N) state at the interval start: m, mb, lo
                                 constant float* spins   [[buffer(1)]],
                                 constant float* segs    [[buffer(2)]],
                                 constant int* flags     [[buffer(3)]],
                                 constant float* pool    [[buffer(4)]],
                                 constant float* frf     [[buffer(5)]],
                                 constant float* line    [[buffer(6)]],
                                 constant float* log_f   [[buffer(7)]],
                                 constant float* conc    [[buffer(8)]],
                                 device float* R         [[buffer(9)]],   // (K1 / RBLOCK, 8, N) block-start states, out
                                 constant uint* prm      [[buffer(10)]],  // as bloch_adjoint
                                 uint i [[thread_position_in_grid]]) {
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
        constant float* q = segs + 8 * s;
        const int f = flags[s];
        if ((s - s0) % RBLOCK == 0) {
            device float* r = R + (s - s0) / RBLOCK * 8 * n;
            r[j] = st.mx, r[n + j] = st.my, r[2 * n + j] = st.mz, r[3 * n + j] = st.mb;
            r[4 * n + j] = st.lx, r[5 * n + j] = st.ly, r[6 * n + j] = st.lz, r[7 * n + j] = st.lb;
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

// NC: coils held in registers (sensitivities, their gradients), padded with zero coils; 0: any count, from memory
kernel void bloch_adjoint@SUFFIX@_c@NC@(device const float* CK  [[buffer(0)]],   // (8, N) state at the interval start: m, mb, lo
                                  constant float* spins   [[buffer(1)]],   // (8, N) px py pz pd r1 r2 db0 b1
                                  device const float* sens [[buffer(2)]],  // (2C, N) per coil: real, imag
                                  constant float* segs    [[buffer(3)]],   // (S, 8) dt rfx rfy gx gy gz psi b0
                                  constant int* flags     [[buffer(4)]],   // (S,) 1 rf, 2 adc, 4 spoil
                                  constant int* adc_row   [[buffer(5)]],   // (S,) signal row of an ADC segment
                                  device const float* hb  [[buffer(6)]],   // (n_adc, C, 2) dL/d(raw coil sums)
                                  constant float* pool    [[buffer(7)]],   // (5, N) f k r1b t2b dw (mt)
                                  constant float* frf     [[buffer(8)]],   // (S,) RF frequency (mt)
                                  constant float* line    [[buffer(9)]],   // (6,) lineshape (mt)
                                  constant float* log_f   [[buffer(10)]],  // lineshape table (mt)
                                  constant float* conc    [[buffer(11)]],  // (7, N) concomitant-field coefficients
                                  device float* R         [[buffer(12)]],  // (K1 / RBLOCK, 8, N) block-start states
                                  device float* LAM       [[buffer(13)]],  // (4, N) lambda, in/out
                                  device float* GS        [[buffer(14)]],  // (20, N) per-spin gradients, in/out
                                  device float* GC        [[buffer(15)]],  // (2C, N) coil gradients, in/out
                                  device float* GSEG      [[buffer(16)]],  // (simdgroups, K1, 8) segment partials, out
                                  constant uint* prm      [[buffer(17)]],  // n nc s0 s1 mt mode need_seg need_sens need_adc K1
                                  uint i     [[thread_position_in_grid]],
                                  uint lane  [[thread_index_in_simdgroup]],
                                  uint sg    [[simdgroup_index_in_threadgroup]],
                                  uint nsg   [[simdgroups_per_threadgroup]],
                                  uint tid   [[thread_index_in_threadgroup]],
                                  uint group [[threadgroup_position_in_grid]]) {
    // need_seg: 2 every segment's gradients (dt, grad, b0_offset requested), 1 those of RF segments (rf, rf_freq)
    const uint n = prm[0], nc = prm[1], s0 = prm[2], s1 = prm[3], mt = prm[4], mode = prm[5], need_seg = prm[6],
               need_sens = prm[7], need_adc = prm[8], K1 = prm[9];
    const bool active = i < n;
    const uint j = active ? i : 0;
    State st;
    st.p = load_spin(spins, n, j);
    if (@V@ != 0 && mt) st.b = load_pool(pool, n, j);
    if (@V@ != 0 && mode)
        for (uint c = 0; c < 7; ++c) st.conc.c[c] = conc[c * n + j];
    if (!active) st.p.pd = 0.0f, st.b.f = 0.0f;
    // reverse sweep, a block of RBLOCK segments at a time: the block's states recomputed from R into tg
    float lx = active ? LAM[j] : 0.0f, ly = active ? LAM[n + j] : 0.0f, lz = active ? LAM[2 * n + j] : 0.0f;
    float lb = active ? LAM[3 * n + j] : 0.0f;
    Acc g = Acc{};
    RCache rc;
    threadgroup float tg[RBLOCK * 4 * 256];
    const uint sgid = group * nsg + sg;  // this simdgroup's row of the segment partials
    // coil gradients summed in registers (+ 1: no zero-length arrays); the sensitivities are read per sample: in
    // registers too they cost more occupancy than the reads (3d-48, 8 coils: 1.9 s against 1.4 s)
    float gcr[@NC@ + 1], gci[@NC@ + 1];
    for (uint c = 0; c < @NC@; ++c) gcr[c] = 0.0f, gci[c] = 0.0f;
    const uint nb = (s1 - s0 + RBLOCK - 1) / RBLOCK;
    for (int blk = int(nb) - 1; blk >= 0; --blk) {
        const uint b0 = s0 + uint(blk) * RBLOCK, b1 = min(b0 + RBLOCK, s1);
        device const float* r = R + uint(blk) * 8 * n;
        st.mx = active ? r[j] : 0.0f, st.my = active ? r[n + j] : 0.0f, st.mz = active ? r[2 * n + j] : 0.0f;
        st.mb = active ? r[3 * n + j] : 0.0f;
        st.lx = active ? r[4 * n + j] : 0.0f, st.ly = active ? r[5 * n + j] : 0.0f;
        st.lz = active ? r[6 * n + j] : 0.0f, st.lb = active ? r[7 * n + j] : 0.0f;
        st.cache = Cache{}, st.zc = ZCache{};
        for (uint s = b0; s < b1; ++s) {
            constant float* q = segs + 8 * s;
            const int f = flags[s];
            threadgroup float* t = tg + (s - b0) * 4 * 256;
            t[tid] = st.mx, t[256 + tid] = st.my, t[512 + tid] = st.mz, t[768 + tid] = st.mb;
            const float bz = offset(st.p, st.conc, @V@ == 0 ? 0u : mode, q);
            if (@V@ != 0 && mt)
                step_mt(q, f, bz, frf[s], st.mx, st.my, st.mz, st.mb, st.lx, st.ly, st.lz, st.lb, st.p, st.b, line,
                        log_f, st.cache, st.zc);
            else
                step(q, f, bz, st.mx, st.my, st.mz, st.lx, st.ly, st.lz, st.p, st.cache);
            if (f & 4) st.mx = st.my = st.lx = st.ly = 0.0f;
        }
        for (int s = int(b1) - 1; s >= int(b0); --s) {
            constant float* q = segs + 8 * s;
            const int f = flags[s];
            threadgroup float* t = tg + (uint(s) - b0) * 4 * 256;
            const float x0 = t[tid], y0 = t[256 + tid], z0 = t[512 + tid], v0 = t[768 + tid];
            if (f & 4) lx = ly = 0.0f;
            if ((f & 2) && need_adc) {  // the sample: lambda += dL/dm, and the coil gradients from the output
                device const float* h = hb + 2 * (@NC@ > 0 ? @NC@ : nc) * uint(adc_row[s]);
                float ox = 0.0f, oy = 0.0f;
                if (need_sens) {  // the segment's output (before the spoil), recomputed
                    Cache kc;
                    ZCache zc;
                    float mx = x0, my = y0, mz = z0, mb = v0, ax = 0.0f, ay = 0.0f, az = 0.0f, ab = 0.0f;
                    const float bz = offset(st.p, st.conc, @V@ == 0 ? 0u : mode, q);
                    if (@V@ != 0 && mt)
                        step_mt(q, f, bz, frf[s], mx, my, mz, mb, ax, ay, az, ab, st.p, st.b, line, log_f, kc, zc);
                    else
                        step(q, f, bz, mx, my, mz, ax, ay, az, st.p, kc);
                    ox = mx + ax, oy = my + ay;
                }
                if (@NC@ > 0) {
                    for (uint c = 0; c < @NC@; ++c) {
                        const float hr = h[2 * c], hi = h[2 * c + 1];
                        const float a = sens[2 * c * n + j], b = sens[(2 * c + 1) * n + j];
                        lx += hr * a + hi * b, ly += hi * a - hr * b;
                        if (need_sens) gcr[c] += hr * ox + hi * oy, gci[c] += hi * ox - hr * oy;
                    }
                } else {
                    for (uint c = 0; c < nc; ++c) {
                        const float hr = h[2 * c], hi = h[2 * c + 1];
                        const float a = sens[2 * c * n + j], b = sens[(2 * c + 1) * n + j];
                        lx += hr * a + hi * b, ly += hi * a - hr * b;
                        if (need_sens && active) GC[2 * c * n + j] += hr * ox + hi * oy, GC[(2 * c + 1) * n + j] += hi * ox - hr * oy;
                    }
                }
            }
            SegG sgv = SegG{};
            reverse_seg<@V@>(q, f, (@V@ != 0 && mt) ? frf[s] : 0.0f, x0, y0, z0, v0, st.p, st.b, st.conc, mode, mt != 0,
                             line, log_f, lx, ly, lz, lb, g, sgv, rc);
            // one partial per simdgroup and segment (no barriers; seg_reduce sums them in order)
            device float* out = GSEG + (sgid * K1 + (uint(s) - s0)) * 8;
            if (need_seg == 2 || (need_seg == 1 && (f & 1))) {
                const float v[8] = {sgv.dt, sgv.rx, sgv.ry, sgv.gx, sgv.gy, sgv.gz, sgv.off, sgv.frf};
                for (uint k = 0; k < 8; ++k) {
                    const float tt = simd_sum(active ? v[k] : 0.0f);
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


_LIB = None
MAX_COILS = 32  # the direct (simd_sum) reduction; the matrix-product one has no limit
MAX_POOLS = 5  # free exchanging pools (the shader's MAX_POOLS; one kernel per count)
ADJOINT_COILS = (0, 1, 2, 4, 8, 16)  # adjoint kernels with the coils in registers (padded up); 0: any, from memory
MMA_MIN_COILS = 2


def _lib():
    global _LIB
    if _LIB is None:
        # each kernel four times (advance<V>): plain, with the rate-model bound pool and/or the field
        # model, with the generalized Bloch modes, and with free exchanging pools; extra per-thread
        # state costs the plain kernels ~10-30 % in register pressure otherwise
        variants = [(suffix + dsuf, v, diff, "0") for suffix, v in (("", "0"), ("_full", "1"), ("_gen", "2"))
                    for dsuf, diff in (("", "false"), ("_diff", "true"))]
        # free pools, one kernel per pool count
        variants += [(f"_bm{k}{dsuf}", "3", diff, str(k)) for k in range(1, MAX_POOLS + 1)
                     for dsuf, diff in (("", "false"), ("_diff", "true"))]
        kernels = "".join(_KERNELS.replace("@SUFFIX@", suffix).replace("@V@", v).replace("@DIFF@", diff)
                          .replace("@NP@", np_) for suffix, v, diff, np_ in variants)
        assert f"#define MAX_POOLS {MAX_POOLS} " in _SOURCE
        replay, adjoint = _ADJOINT_KERNEL.split("// NC: coils held in registers")
        adjoint = "// NC: coils held in registers" + adjoint
        for suffix, v in (("", "0"), ("_full", "1")):
            kernels += replay.replace("@SUFFIX@", suffix).replace("@V@", v)
            kernels += "".join(adjoint.replace("@SUFFIX@", suffix).replace("@V@", v).replace("@NC@", str(c)) for c in ADJOINT_COILS)
        _LIB = torch.mps.compile_shader(_SOURCE + _ADJOINT_COMMON + kernels)
    return _LIB


def available() -> bool:
    try:
        return torch.backends.mps.is_available() and _lib() is not None
    except Exception:
        return False


def _requires_grad(*objs) -> bool:
    """Whether any tensor in ``objs`` (tensors, dataclasses such as Phantom, Sequence, Scanner,
    FieldMaps and their pools, tuples) requires grad."""
    for o in objs:
        if isinstance(o, Tensor):
            if o.requires_grad:
                return True
        elif dataclasses.is_dataclass(o) and not isinstance(o, type):
            if _requires_grad(*(getattr(o, f.name) for f in dataclasses.fields(o))):
                return True
        elif isinstance(o, tuple) and _requires_grad(*o):
            return True
    return False


def simulate_metal(phantom: Phantom, seq: Sequence, coils: Tensor | None = None, *, m_init: Tensor | None = None,
                   chunk: int = 32768, group: int = 256, scanner=None, mma: bool | None = None,
                   seed: int = 0, maps=None, checkpoint_every: int | None = None) -> SimResult:
    """Forward simulation on the Apple GPU; same arguments and result as ``blochparty.simulate``
    (results come back on the CPU, float32). ``chunk`` segments per dispatch; ``group`` threads
    per threadgroup (must be 256). ``mma`` picks the coil reduction: matrix products on simdgroup
    matrices (default from ``MMA_MIN_COILS`` coils) or one simd_sum per coil and sample.

    Gradients (when grad mode is on and an input requires grad) come from a hand-written adjoint on the GPU
    (``bloch_adjoint``): to the sequence (dt, rf, grad, adc_phase, b0_offset), the phantom (pos, pd, t1, t2,
    db0, b1, the rate-model bound pool), coils, m_init and, through the encoding positions and concomitant
    coefficients, the scanner field model. The forward stores the state before every ``checkpoint_every``
    segments (a multiple of 64; default about sqrt(segments)); the backward replays one interval at a time.
    Not with diffusion, motion, free pools or the generalized bound pool (raises)."""
    if torch.is_grad_enabled() and _requires_grad(phantom, seq, coils, m_init, scanner, maps):
        return _simulate_grad(phantom, seq, coils, m_init, scanner, maps, mma, checkpoint_every)
    with torch.no_grad():
        return _forward(phantom, seq, coils, m_init, chunk, group, scanner, mma, seed, maps)


def default_checkpoint_every(n_segments: int) -> int:
    """About sqrt(segments), a multiple of 64 (the adjoint's per-spin gradients are flushed every 64
    segments; with intervals of whole flushes the result does not depend on the interval)."""
    return 64 * max(1, math.ceil(math.sqrt(max(n_segments, 1)) / 64))


class _Opts(NamedTuple):
    is_rf: list
    adc: list
    spoil: list
    every: int
    mode: int
    mma: bool | None
    lineshape: str
    cutoff: float


N_SPIN_GRADS = 20  # rows of the adjoint's per-spin gradients: pos (3), db0, pd, r1, r2, b1, conc (7), bound (5)


class _MetalSimulate(torch.autograd.Function):
    """The forward kernels with checkpoints; backward: bloch_adjoint, one dispatch per interval."""

    @staticmethod
    def forward(ctx, opts: _Opts, pos, pd, t1, t2, db0, coils_ri, dt, rf, grad, adc_phase, m0, b1, b0_offset, conc,
                bound, rf_freq):
        ctx.set_materialize_grads(False)
        d = lambda t: None if t is None else t.detach()
        ph = Phantom(d(pos), d(pd), d(t1), d(t2), d(db0), b1=d(b1),
                     bound=None if bound is None else exchange.BoundPool(*d(bound).unbind(-1), lineshape=opts.lineshape,
                                                                         cutoff=opts.cutoff))
        sq = Sequence(d(dt), d(rf), d(grad), d(adc_phase), opts.is_rf, opts.adc, opts.spoil, {}, d(b0_offset), d(rf_freq))
        coils = torch.complex(coils_ri[..., 0], coils_ri[..., 1]).detach()
        from ..fields import SpinFields

        res, cx = _forward(ph, sq, coils, d(m0), opts.every, 256, None, opts.mma, 0, None, True,
                           SpinFields(d(pos), d(conc), opts.mode), keep=True)
        ctx.opts, ctx.cx = opts, cx
        ctx.like = [None if t is None else (t.dtype, t.device) for t in
                    (pos, pd, t1, t2, db0, coils_ri, dt, rf, grad, adc_phase, m0, b1, b0_offset, conc, bound, rf_freq)]
        sig = torch.view_as_real(res.signal.T.contiguous())  # (n_adc, C, 2)
        m = res.m if res.m_bound is None else torch.cat([res.m, res.m_bound[:, None]], 1)
        ctx.save_for_backward(t1, t2, adc_phase, sig, bound, b1, conc)
        return sig, m

    @staticmethod
    @once_differentiable
    def backward(ctx, g_sig, g_m):
        opts, cx = ctx.opts, ctx.cx
        t1, t2, adc_phase, sig, bound, b1, conc = ctx.saved_tensors
        # pos, pd, t1, t2, db0, coils, dt, rf, grad, adc_phase, m0, b1, b0_offset, conc, bound, rf_freq
        need = list(ctx.needs_input_grad[1:])
        if g_sig is None:
            need[5] = need[9] = False
        if (g_sig is None and g_m is None) or not any(need):
            return (None,) * 17
        dev, f32 = torch.device("mps"), torch.float32
        n, nc, S, every = t1.shape[0], cx["n_coils"], len(opts.is_rf), opts.every
        adc_idx = torch.tensor(opts.adc).nonzero()[:, 0]
        h = torch.zeros(max(cx["n_adc"], 1), nc, 2, dtype=f32)
        if g_sig is not None and cx["n_adc"]:  # dL/d(raw coil sums): dL/dsignal rotated by exp(+i psi)
            psi = adc_phase.detach().double().cpu()[adc_idx][:, None]
            gr, gi = g_sig[..., 0].double().cpu(), g_sig[..., 1].double().cpu()
            h = torch.stack([gr * torch.cos(psi) - gi * torch.sin(psi), gr * torch.sin(psi) + gi * torch.cos(psi)], -1).float()
        lam = torch.zeros(4, n, dtype=f32)
        if g_m is not None:
            lam[: g_m.shape[1]] = g_m.detach().float().cpu().T
        need_seg = 2 if any(need[i] for i in (6, 8, 12)) else (1 if need[7] or need[15] else 0)
        need_sens = need[5]
        lam = lam.to(dev).contiguous()
        GS = torch.zeros(N_SPIN_GRADS, n, device=dev, dtype=f32)
        GC = torch.zeros(2 * max(nc, 16) if need_sens else 1, n, device=dev, dtype=f32)
        threads, sgs = cx["threads"], cx["threads"] // 32
        R = torch.zeros(every // 4 * 8 * n, device=dev, dtype=f32)  # RBLOCK = 4
        GSEG = torch.zeros(sgs * every * 8 if need_seg else 1, device=dev, dtype=f32)
        # coil gradients in registers (padded with zero coils) only when they are wanted: the arrays cost occupancy
        ncr = next((c for c in ADJOINT_COILS[1:] if c >= nc), 0) if need_sens else 0
        sens = cx["sens"]
        if ncr > nc:
            sens = torch.cat([sens, torch.zeros(2 * (ncr - nc), n, device=dev, dtype=f32)]).contiguous()
            h = torch.cat([h, torch.zeros(h.shape[0], ncr - nc, 2, dtype=f32, device=h.device)], 1)
        SEG = torch.zeros(S * 8 if need_seg else 1, device=dev, dtype=f32)
        lib = _lib()
        h = h.to(dev).contiguous()
        kernel = getattr(lib, f"bloch_adjoint{cx['suffix']}_c{ncr}")
        replay = getattr(lib, "bloch_replay" + cx["suffix"])
        n_int = len(cx["ckpts"])
        u32 = lambda rows: torch.from_numpy(np.array(rows, dtype=np.uint32).view(np.int32)).to(dev)
        prms = u32([[n, nc, c * every, min(S, (c + 1) * every), cx["mt"], cx["mode"], need_seg, need_sens,
                     g_sig is not None, every] for c in range(n_int)])
        reds = u32([[sgs, every, c * every, min(S, (c + 1) * every) - c * every] for c in range(n_int)])
        for c in reversed(range(n_int)):
            s0, s1 = c * every, min(S, (c + 1) * every)
            prm = prms[c]
            replay(cx["ckpts"][c], cx["spins"], cx["segs"], cx["flags"], cx["pool"], cx["frf"], cx["line"], cx["log_f"],
                   cx["conc"], R, prm, threads=threads, group_size=256)
            kernel(cx["ckpts"][c], cx["spins"], sens, cx["segs"], cx["flags"], cx["rows"], h, cx["pool"], cx["frf"],
                   cx["line"], cx["log_f"], cx["conc"], R, lam, GS, GC, GSEG, prm, threads=threads, group_size=256)
            if need_seg:
                lib.seg_reduce(GSEG, SEG, reds[c], threads=8 * (s1 - s0))
        torch.mps.synchronize()
        GS, lam = GS.cpu().double(), lam.cpu().double()
        out = [None] * 16
        if need[0]:
            out[0] = GS[0:3].T.contiguous()
        if need[1]:
            out[1] = GS[4]
        if need[2]:
            out[2] = -GS[5] / t1.detach().double().cpu() ** 2  # r1 = 1 / t1
        if need[3]:
            out[3] = -GS[6] / t2.detach().double().cpu() ** 2
        if need[4]:
            out[4] = GS[3]
        if need_sens:
            gc = GC.cpu().double()[: 2 * nc]
            out[5] = torch.stack([gc[0::2].T, gc[1::2].T], -1)
        if need_seg:
            seg = SEG.cpu().double().view(S, 8)
            out[6], out[7], out[8], out[12] = seg[:, 0], seg[:, 1:3], seg[:, 3:6], seg[:, 6]
            out[15] = seg[:, 7]
        if need[9]:  # signal = raw exp(-i psi): dsignal/dpsi = -i signal
            g = torch.zeros(S, dtype=torch.float64)
            sg, gs = sig.double(), g_sig.double().cpu()
            g[adc_idx] = (gs[..., 0] * sg[..., 1] - gs[..., 1] * sg[..., 0]).sum(-1)
            out[9] = g
        if need[10]:
            out[10] = lam[: 4 if bound is not None else 3].T.contiguous()
        if need[11]:
            out[11] = GS[7]
        if need[13]:
            out[13] = GS[8:8 + conc.shape[1]].T.contiguous()
        if need[14]:  # f, k, t1b (r1b = 1 / t1b), t2b, dw
            t1b = bound.detach().double().cpu()[:, 2]
            out[14] = torch.stack([GS[15], GS[16], -GS[17] / t1b**2, GS[18], GS[19]], -1)
        return (None, *[None if g is None or not need[i] else g.to(dtype=ctx.like[i][0], device=ctx.like[i][1]) for i, g in enumerate(out)])


def _simulate_grad(phantom: Phantom, seq: Sequence, coils, m_init, scanner, maps, mma, every) -> SimResult:
    """simulate_metal with gradients: the inputs as tensors of an autograd Function (_MetalSimulate)."""
    for what, present in (("diffusion", phantom.diffusion is not None), ("motion", phantom.motion is not None),
                          ("free pools", bool(getattr(phantom, "pools", None))),
                          ("the generalized Bloch bound pool", phantom.bound is not None and phantom.bound.model == "generalized")):
        if present:
            raise NotImplementedError(f"gradients through simulate_metal with {what} are not implemented; "
                                      "use blochparty.simulate, or run under torch.no_grad()")
    n, pd = len(phantom), phantom.pd
    fields = spin_fields(scanner, phantom.pos)
    db0, b1 = phantom.db0, phantom.b1
    if maps is not None:  # static spins: the maps sampled at the true positions, differentiably
        from ..maps import sample

        if maps.coils is not None and coils is not None:
            raise ValueError("coil sensitivities come from either `coils` or `maps`, not both")
        v = sample(maps.channels(pd.dtype), torch.as_tensor(maps.origin, dtype=pd.dtype),
                   torch.as_tensor(maps.spacing, dtype=pd.dtype), phantom.pos)
        db0 = db0 + v[:, 0]
        b1 = v[:, 1] if b1 is None else b1 * v[:, 1]
        if maps.coils is not None:
            coils = torch.complex(v[:, 2::2], v[:, 3::2])
    if coils is None:
        coils = torch.ones(n, 1, dtype=torch.complex64 if pd.dtype == torch.float32 else torch.complex128)
    bound, rf_freq, lineshape, cutoff = None, None, "superlorentzian", 0.0
    if phantom.bound is not None:
        pool = phantom.bound.expand(n, pd)
        bound = torch.stack([pool.f, pool.k, pool.t1, pool.t2, pool.dw], -1)
        rf_freq = exchange.rf_frequency(seq)
        lineshape, cutoff = pool.lineshape, float(pool.cutoff)
    if m_init is None:
        m0 = torch.stack([torch.zeros_like(pd), torch.zeros_like(pd), pd], -1)
        if bound is not None:
            m0 = torch.cat([m0, (bound[:, 0] * pd)[:, None]], -1)
    else:
        m0 = m_init
        if bound is not None and m0.shape[-1] == 3:
            m0 = torch.cat([m0, (bound[:, 0] * pd)[:, None]], -1)
        elif bound is None and m0.shape[-1] == 4:
            raise ValueError("m_init has a bound-pool column but the phantom has no bound pool")
    every = every or default_checkpoint_every(len(seq))
    if every % 64:
        raise ValueError(f"checkpoint_every must be a multiple of 64, got {every}")
    opts = _Opts(list(seq.is_rf), list(seq.adc), list(seq.spoil), every, fields.mode, mma, lineshape, cutoff)
    sig, m = _MetalSimulate.apply(opts, fields.pos, pd, phantom.t1, phantom.t2, db0, torch.view_as_real(coils),
                                  seq.dt, seq.rf, seq.grad, seq.adc_phase, m0, b1, seq.b0_offset, fields.conc, bound,
                                  rf_freq)
    signal = torch.view_as_complex(sig.contiguous()).T
    if bound is not None:
        return SimResult(signal, m[:, :3], m[:, 3])
    return SimResult(signal, m)


def _field_model_buffer(scanner) -> Tensor:
    """The shader's fm buffer (nl_displacement, conc_at): nonlinearity flag, L, R0, f0, z asymmetry, then
    A and B (3, L, L) with R0 N_nm folded in and zero for m > n (fields.GradientNonlinearity)."""
    head = [0.0, 1.0, 1.0, 1.0, float(scanner.z_asymmetry)]
    if scanner.field_strength is not None and scanner.concomitant is not None:
        head[3] = float(scanner.larmor)
    coef = torch.zeros(0, dtype=torch.float64)
    nl = scanner.nonlinearity
    if nl is not None:
        size = nl.n_max + 1
        norm = torch.zeros(size, size, dtype=torch.float64)
        for nn in range(size):
            for m in range(nn + 1):
                norm[nn, m] = 1.0 if m == 0 else math.sqrt((2 * nn + 1) * math.factorial(nn - m) / (2 * math.factorial(nn + m)))
        scale = nl.r0 * norm
        a, b = (t.detach().double().cpu() * scale for t in (nl.alpha, nl.beta))
        head[:3] = [1.0, float(size), float(nl.r0)]
        coef = torch.cat([a.reshape(-1), b.reshape(-1)])
    return torch.cat([torch.tensor(head, dtype=torch.float64), coef]).float().contiguous()


def _cost_order(phantom: Phantom, seq: Sequence) -> Tensor | None:
    """Static spins in order of their free-pool kernel cost: an RF segment takes sub-steps in proportion to
    the spin's off-resonance there, and the lanes of a simdgroup wait for its slowest spin (a CEST
    spectrum's M0 offset at -1500 ppm is ~80 sub-steps per 100 us sample, the other offsets ~2)."""
    rf = torch.tensor(seq.is_rf)
    if not bool(rf.any()):
        return None
    g = seq.grad.detach().double().cpu()[rf].abs().amax(0)
    off = 0.0 if seq.b0_offset is None else float(seq.b0_offset.detach().double()[rf].abs().max())
    cost = phantom.db0.detach().double().cpu().abs() + phantom.pos.detach().double().cpu().abs() @ g + off
    return torch.argsort(cost, stable=True)


def _take(phantom: Phantom, idx: Tensor) -> Phantom:
    """The (static) spins ``idx`` of a phantom with free pools."""
    n, pd = len(phantom), phantom.pd
    pick = lambda t: None if t is None else t[idx]
    pools = tuple(type(q)(*(getattr(q.expand(n, pd), a)[idx] for a in ("f", "k", "t1", "t2", "dw"))) for q in phantom.pools)
    bound = None if phantom.bound is None else phantom.bound.expand(n, pd)[idx]
    return dataclasses.replace(phantom, pos=phantom.pos[idx], pd=pd[idx], t1=phantom.t1[idx], t2=phantom.t2[idx],
                               db0=phantom.db0[idx], b1=pick(phantom.b1), bound=bound, pools=pools)


def _forward(phantom: Phantom, seq: Sequence, coils, m_init, chunk, group, scanner, mma, seed, maps,
             ordered: bool = False, fields_in=None, keep: bool = False):
    """The forward kernels. ``fields_in``: the field model's inputs (``fields.SpinFields``) in place of
    spin_fields(scanner, pos). ``keep``: also return what the adjoint needs, with the state before every
    ``chunk`` segments (one dispatch each)."""
    pools = tuple(getattr(phantom, "pools", None) or ())
    if pools:  # as bloch.simulate
        if phantom.bound is not None and phantom.bound.model == "generalized":
            raise NotImplementedError("free pools with the generalized Bloch bound pool are not implemented")
        if len(pools) > MAX_POOLS:
            raise NotImplementedError(f"at most {MAX_POOLS} free pools on the GPU, got {len(pools)}")
        # moving spins keep their order: a walk's random numbers belong to the spin's index
        moving = phantom.diffusion is not None or phantom.motion is not None
        order = None if ordered or moving else _cost_order(phantom, seq)
        if order is not None:  # run the spins sorted by cost, return them in the caller's order
            inv = torch.argsort(order)
            r = _forward(_take(phantom, order), seq, None if coils is None else coils[order],
                         None if m_init is None else m_init[order], chunk, group, scanner, mma, seed, maps, True)
            return SimResult(r.signal, r.m[inv], None if r.m_bound is None else r.m_bound[inv], r.m_pools[inv])
    dev, f32 = torch.device("mps"), torch.float32
    n = len(phantom)
    b1 = phantom.b1 if getattr(phantom, "b1", None) is not None else torch.ones(n)
    # encoding positions (nonlinearity), concomitant coefficients
    fields = fields_in if fields_in is not None else spin_fields(scanner, phantom.pos)
    moving = phantom.diffusion is not None or phantom.motion is not None
    db0 = phantom.db0
    mapped, map_bufs = 0, [torch.zeros(1, device=dev, dtype=f32), torch.zeros(9, device=dev, dtype=f32)]
    if maps is not None:  # scanner-frame fields: sampled once for static spins, per segment for moving ones
        from ..maps import sample

        if maps.coils is not None and coils is not None:
            raise ValueError("coil sensitivities come from either `coils` or `maps`, not both")
        grid = maps.channels(torch.float64)
        origin = torch.as_tensor(maps.origin, dtype=torch.float64)
        spacing = torch.as_tensor(maps.spacing, dtype=torch.float64)
        if not moving:  # at the true positions (the maps are scanner-frame fields)
            v = sample(grid, origin, spacing, phantom.pos.detach().double().cpu())
            db0 = phantom.db0 + v[:, 0].to(phantom.db0.dtype)
            b1 = b1.to(v.dtype) * v[:, 1]
            if maps.coils is not None:
                coils = torch.complex(v[:, 2::2], v[:, 3::2])
        else:
            mapped = 2 if maps.coils is not None else 1
            if maps.coils is not None:  # placeholders: the kernel samples the maps at every ADC sample
                coils = torch.zeros(n, maps.coils.shape[0], dtype=torch.complex64)
                mma = False
            nz, ny, nx = maps.shape()
            mg = torch.cat([origin, spacing, torch.tensor([nx, ny, nz], dtype=torch.float64)])
            map_bufs = [grid.reshape(-1).to(dev, f32).contiguous(), mg.to(dev, f32).contiguous()]
    spins = torch.stack([fields.pos[:, 0], fields.pos[:, 1], fields.pos[:, 2], phantom.pd, 1 / phantom.t1,
                         1 / phantom.t2, db0, b1.to(phantom.pd.dtype)]).to(dev, f32).contiguous()
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
    # the state as float32 hi + lo (L), split exactly from float64: the kernels carry lo (two_sum) across dispatches
    if m_init is None:
        M64 = torch.stack([torch.zeros(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64),
                           phantom.pd.detach().double().cpu()])
    else:
        M64 = m_init.detach().double().cpu().T
    mb64 = torch.zeros(n, dtype=torch.float64)
    pool = None if phantom.bound is None else phantom.bound.expand(n, phantom.pd)
    if pool is not None:  # magnetization transfer: the bound pool's Mz, parameters, RF frequency, lineshape
        mb = pool.f * phantom.pd if m_init is None or m_init.shape[-1] == 3 else m_init[:, 3]
        mb64 = mb.detach().double().cpu()
        if m_init is not None and m_init.shape[-1] == 4:
            M64 = M64[:3]
        mt_bufs = [mb64.float().to(dev).contiguous(),
                   torch.stack([pool.f, pool.k, 1 / pool.t1, pool.t2, pool.dw]).detach().to(dev, f32).contiguous()]
        gen_bufs = None
        if pool.model == "generalized":  # modes instead of the saturation lineshape
            from ..generalized_bloch import modes as green_modes

            pm, cm = green_modes(pool.lineshape, pool.memory)
            table = torch.tensor([[q.real, q.imag, c.real, c.imag] for q, c in zip(pm, cm)], dtype=f32)
            t2b_min = float(pool.t2.detach().min())
            dts = seq.dt.detach().double().cpu()
            nsub = torch.tensor([max(1, math.ceil(float(d) / (0.5 * t2b_min))) if r else 1
                                 for d, r in zip(dts, seq.is_rf)], dtype=torch.int32)
            gen_bufs = [torch.zeros(2 * len(pm), n, device=dev, dtype=f32), table.to(dev), nsub.to(dev), len(pm)]
            dummy = torch.zeros(1, device=dev, dtype=f32)
            mt_bufs += [dummy, dummy, dummy, 2]
        else:
            frf = exchange.rf_frequency(seq.to(dtype=torch.float64, device="cpu")).detach()
            lx, log_f = exchange._sl_table(torch.float64, torch.device("cpu"))
            kind = 1.0 if pool.lineshape == "lorentzian" else 2.0
            line = torch.tensor([kind, float(pool.cutoff), float(lx[0]), float(lx[1] - lx[0]), exchange._SL_XMIN,
                                 float(len(lx))], dtype=f32)
            mt_bufs += [frf.to(dev, f32).contiguous(), line.to(dev), log_f.to(dev, f32).contiguous(), 1]
    else:
        if m_init is not None and m_init.shape[-1] == 4:
            raise ValueError("m_init has a bound-pool column but the phantom has no bound pool")
        dummy = torch.zeros(1, device=dev, dtype=f32)
        mt_bufs = [dummy, dummy, dummy, dummy, dummy, 0]
        gen_bufs = None
    M = M64.float()
    lo_buf = torch.cat([M64 - M.double(), (mb64 - mb64.float().double())[None]]).float().to(dev).contiguous()
    M = M.to(dev).contiguous()
    if fields.mode:
        conc = torch.zeros(7, n, dtype=torch.float64)
        conc[: fields.conc.shape[1]] = fields.conc.detach().double().cpu().T
        conc_buf = [conc.to(dev, f32).contiguous(), int(fields.mode)]
    else:
        conc_buf = [torch.zeros(1, device=dev, dtype=f32), 0]
    sig = torch.zeros(max(n_adc, 1) * n_coils * 2, device=dev, dtype=f32)
    if gen_bufs is None:
        dz = torch.zeros(1, device=dev, dtype=f32)
        gen_bufs = [dz, dz, torch.zeros(1, device=dev, dtype=torch.int32), 0]
    # moving spins under the field model move in true coordinates; the kernel encodes them per segment
    fmove = moving and scanner is not None and bool(fields.mode or scanner.nonlinearity is not None)
    fm_buf = torch.zeros(1, device=dev, dtype=f32)
    if fmove:
        fm_buf = _field_model_buffer(scanner).to(dev)
    r_start = (phantom.pos if fmove else fields.pos).detach().double().cpu()
    if phantom.diffusion is not None:  # walkers start at their positions (encoding ones without fmove); L L^T = D
        D = phantom.diffusion.detach().double().cpu()
        L = torch.diag_embed(torch.sqrt(D).unsqueeze(-1).expand(n, 3)) if D.dim() == 1 else torch.linalg.cholesky(D)
        chol = torch.stack([L[:, 0, 0], L[:, 1, 0], L[:, 1, 1], L[:, 2, 0], L[:, 2, 1], L[:, 2, 2]])
        start = r_start
        dzi = torch.zeros(1, device=dev, dtype=torch.int32)
        geom_bufs = [dzi, torch.zeros(1, device=dev, dtype=f32), 0]
        if phantom.restriction is not None:  # walkers start inside their compartments, as the core
            from ..restriction import KINDS, n_sub, start_offsets, tau_max

            if D.dim() != 1:
                raise ValueError("restricted diffusion needs an isotropic diffusivity (N,)")
            res = phantom.restriction.expand(n, start)
            centre = start
            start = centre + start_offsets(res, int(seed), np.arange(n, dtype=np.uint64), centre)
            tmax = tau_max(res, D)
            wsub = torch.tensor([n_sub(float(d), tmax) for d in seq.dt.detach().double().cpu()], dtype=torch.int32)
            ax = res.axis if res.axis is not None else torch.zeros(n, 3, dtype=torch.float64)
            geom = torch.cat([res.size[None], ax.T, centre.T]).to(dev, f32).contiguous()
            geom_bufs = [wsub.to(dev), geom, KINDS.index(res.kind) + 1]
        walk_bufs = [start.T.to(dev, f32).contiguous(), chol.to(dev, f32).contiguous(),
                     int(seed) & 0xFFFFFFFF, (int(seed) >> 32) & 0xFFFFFFFF, *geom_bufs]
    else:  # motion alone: the tissue-frame position stays r0
        dz = torch.zeros(1, device=dev, dtype=f32)
        walk_bufs = [r_start.T.to(dev, f32).contiguous() if phantom.motion is not None else dz, dz, 0, 0,
                     torch.zeros(1, device=dev, dtype=torch.int32), dz, 0]
    move_bufs = [torch.zeros(1, device=dev, dtype=f32), torch.zeros(1, device=dev, dtype=f32)]
    if phantom.motion is not None:
        mo = phantom.motion
        seg = mo.per_segment(seq.dt.detach().double().cpu())
        S_ = len(seq)
        w = seg["w"] if seg["w"] is not None else torch.zeros(S_, dtype=torch.float64)
        A = seg["A"] if seg["A"] is not None else torch.eye(3, dtype=torch.float64).expand(S_, 3, 3)
        b = seg["b"] if seg["b"] is not None else torch.zeros(S_, 3, dtype=torch.float64)
        mseg = torch.cat([seg["t"][:, None], w[:, None], A.reshape(S_, 9), b], 1)
        zero3 = torch.zeros(n, 3, dtype=torch.float64)
        vel = mo.velocity.detach().double().cpu() if mo.velocity is not None else zero3
        dsp = mo.displacement.detach().double().cpu() if mo.displacement is not None else zero3
        move_bufs = [mseg.to(dev, f32).contiguous(), torch.cat([vel.T, dsp.T]).to(dev, f32).contiguous()]
    dz = torch.zeros(1, device=dev, dtype=f32)
    bm_bufs = [dz, dz]
    if pools:  # state (2 nc, N): hi, lo of water, pools (x y z) and the bound pool's Mz (bm_load); split exactly
        pd64 = phantom.pd.detach().double().cpu()
        nc = 3 * (MAX_POOLS + 1) + 1
        init = torch.zeros(nc, n, dtype=torch.float64)
        if m_init is None:
            init[2] = pd64
        else:
            init[:3] = m_init[:, :3].detach().double().cpu().T
        params = torch.zeros(5 * MAX_POOLS, n, dtype=torch.float64)
        for k, q in enumerate(pools):
            q = q.expand(n, pd64)
            init[5 + 3 * k] = q.f.detach() * pd64
            params[5 * k: 5 * k + 5] = torch.stack([q.f, q.k, 1 / q.t1, 1 / q.t2, q.dw]).detach()
        if pool is not None:
            init[nc - 1] = mb.detach().double().cpu()
        hi = init.float()
        bm_bufs = [torch.cat([hi, (init - hi.double()).float()]).to(dev).contiguous(), params.to(dev, f32).contiguous()]
    suffix = "_gen" if gen_bufs[3] else ("_full" if (mt_bufs[5] or conc_buf[1]) else "")
    if pools:
        suffix = f"_bm{len(pools)}"
    if phantom.diffusion is not None or phantom.motion is not None:
        suffix += "_diff"
    lib, S = _lib(), len(seq)
    # buffers 7-19 (MB pool frf line log_f | conc | U modes nsub | P chol | wsub geom) and the scalars
    MB, pool_b, frf_b, line_b, logf_b, mt_flag = mt_bufs
    conc_b, mode_flag = conc_buf
    U_b, modes_b, nsub_b, nj_v = gen_bufs
    P_b, chol_b, seed_lo, seed_hi, wsub_b, geom_b, kind_v = walk_bufs
    bufs = (MB, pool_b, frf_b, line_b, logf_b, conc_b, U_b, modes_b, nsub_b, P_b, chol_b, wsub_b, geom_b, *move_bufs,
            *map_bufs)

    starts = list(range(0, S, chunk))
    tiles = (n_tiles, threads) if mma else (0, 0)
    prms = torch.from_numpy(np.array([[n, n_coils, s0, min(S, s0 + chunk), *tiles, mt_flag, mode_flag, nj_v, seed_lo, seed_hi,
                                       kind_v, int(phantom.motion is not None), int(phantom.diffusion is not None), mapped,
                                       int(fmove)] for s0 in starts], dtype=np.uint32).view(np.int32)).to(dev)

    ckpts = []
    mb_row = mt_bufs[0][None] if pool is not None else torch.zeros(1, n, device=dev, dtype=f32)
    for c, s0 in enumerate(starts):  # one upload of every dispatch's parameters (prms[c]: a view)
        if keep:  # the state before this dispatch: m (3), mb, their low parts (4)
            ckpts.append(torch.cat([M, mb_row, lo_buf]).clone())
        if mma:
            getattr(lib, "bloch_chunk_mma" + suffix)(M, spins, sens, segs, flags, rows, sig, *bufs, prms[c],
                                                     *bm_bufs, fm_buf, lo_buf, threads=threads, group_size=group)
        else:
            getattr(lib, "bloch_chunk" + suffix)(M, spins, sens, segs, flags, rows, sig, *bufs, prms[c],
                                                 *bm_bufs, fm_buf, lo_buf, threads=threads, group_size=group)
    torch.mps.synchronize()
    sig = sig[: n_adc * n_coils * 2].cpu().view(n_adc, n_coils, 2)
    signal = torch.complex(sig[..., 0], sig[..., 1]).T.contiguous()
    if pools:
        st = bm_bufs[0].cpu().double()
        st = (st[:len(st) // 2] + st[len(st) // 2:]).float()
        m_pools = torch.stack([st[3 + 3 * k: 6 + 3 * k].T for k in range(len(pools))], 1).contiguous()
        return SimResult(signal, st[:3].T.contiguous(), st[-1].clone() if pool is not None else None, m_pools)
    lo = lo_buf.cpu().double()
    m = (M.cpu().double() + lo[:3]).float().T.contiguous()
    res = SimResult(signal, m, (mt_bufs[0].cpu().double() + lo[3]).float() if pool is not None else None)
    if not keep:
        return res
    sens_d = torch.stack([coils.real, coils.imag], -1).reshape(n, 2 * n_coils).T.to(dev, f32).contiguous()
    return res, dict(ckpts=ckpts, spins=spins, sens=sens_d, segs=segs, flags=flags, rows=rows, pool=mt_bufs[1],
                     frf=mt_bufs[2], line=mt_bufs[3], log_f=mt_bufs[4], mt=mt_flag, conc=conc_buf[0], mode=conc_buf[1],
                     n_adc=n_adc, n_coils=n_coils, threads=threads, suffix=suffix, sig=sig)
