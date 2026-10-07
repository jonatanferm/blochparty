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

Magnetization transfer (``Phantom.bound``): the bound pool's Mz is a fourth state per thread and
longitudinal relaxation the exact exchange step of ``exchange.py``, saturated during RF (about
+15 % on a 3D GRE).

The scanner field model (``fields.Scanner``: gradient nonlinearity through the encoding
positions, concomitant fields at leading order or exact) enters the off-resonance as in
``fused/kernel.h``. Not (yet) supported here: gradients; use ``blochparty.fused.simulate_fused``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

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

// Rotation about the effective field of an RF segment q for a spin at offset bz [Hz].
inline void rf_rotate(constant float* q, float bz, thread float& mx, thread float& my, thread float& mz,
                      thread const Spin& p) {
    const float w = -TWO_PI * q[0];
    {
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
    }
}

inline void step(constant float* q, int f, float bz, thread float& mx, thread float& my, thread float& mz,
                 thread const Spin& p, thread Cache& k) {
    const float dt = q[0];
    if (f & 1) {  // RF: relax dt/2, rotate about the effective field, relax dt/2
        relax(mx, my, mz, 0.5f * dt, p.pd, p.r1, p.r2);
        rf_rotate(q, bz, mx, my, mz, p);
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

// step() with a bound pool; zc caches the unsaturated exchange step of free segments (key dt).
struct ZCache { float dt = -1.0f; ZStep z; };

inline void step_mt(constant float* q, int f, float bz, float f_rf, thread float& mx, thread float& my,
                    thread float& mz, thread float& mb, thread const Spin& p, thread const Pool& b,
                    constant float* line, constant float* log_f, thread Cache& k, thread ZCache& zc) {
    const float dt = q[0];
    if (f & 1) {
        const float w1sq = TWO_PI * TWO_PI * p.b1 * p.b1 * (q[1] * q[1] + q[2] * q[2]);
        const ZStep z = zstep(0.5f * dt, p, b, w1sq * lineshape(f_rf - bz - b.dw, b.t2b, line, log_f));
        const float e2 = expm1_(-0.5f * dt * p.r2);
        mx += mx * e2, my += my * e2;
        zapply(z, mz, mb);
        rf_rotate(q, bz, mx, my, mz, p);
        mx += mx * e2, my += my * e2;
        zapply(z, mz, mb);
    } else {
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
                     thread float& mb, thread Modes& md, thread const Spin& p, thread const Pool& b,
                     constant float* modes, uint nj, thread Cache& k, thread ZCache& zc) {
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
                    // the water about its effective field
                    const float phi2 = wx * wx + wy * wy + wz * wz, phi = sqrt(phi2);
                    float c, s1v, s2v;
                    if (phi < 1e-3f) { c = 1.0f - 0.5f * phi2; s1v = 1.0f - phi2 / 6.0f; s2v = 0.5f - phi2 / 24.0f; }
                    else { const float sh = precise::sin(0.5f * phi) / phi; c = precise::cos(phi); s1v = precise::sin(phi) / phi; s2v = 2.0f * sh * sh; }
                    const float dot = wx * mx + wy * my + wz * mz;
                    const float nx = c * mx + s1v * (wy * mz - wz * my) + s2v * wx * dot;
                    const float ny = c * my + s1v * (wz * mx - wx * mz) + s2v * wy * dot;
                    const float nz = c * mz + s1v * (wx * my - wy * mx) + s2v * wz * dot;
                    mx = nx; my = ny; mz = nz;
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
                    mb = pc * mb + ps1 * (wx * sy - wy * sx);
                    const float dx = tx - sx, dy = ty - sy;
                    for (uint j = 0; j < nj; ++j) {
                        const float cr = modes[4 * j + 2], ci = modes[4 * j + 3];
                        md.ur[j] += cr * dx - ci * dy, md.ui[j] += cr * dy + ci * dx;
                    }
                }
                mx += mx * e2, my += my * e2;
                zapply(z, mz, mb);
                for (uint j = 0; j < nj; ++j) {
                    const float x = md.ur[j], y = md.ui[j];
                    md.ur[j] = hr[j] * x - hi[j] * y, md.ui[j] = hr[j] * y + hi[j] * x;
                }
            }
        }
    } else {
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
    Spin p;
    Pool b;
    Cache cache;
    ZCache zc;
    Conc conc;
};

// V: 0 plain Bloch (no field model, no bound pool), 1 rate-model bound pool and/or field model,
// 2 generalized Bloch bound pool (and/or field model). Compiled separately so the plain kernels
// carry none of the others' per-thread state.
template <int V, bool DIFF>
inline void advance(thread State& st, thread Modes& md, thread Walk& wk, thread const Geom& gm, constant float* segs,
                    constant int* flags, constant float* frf, uint s, uint mt, constant float* line,
                    constant float* log_f, uint mode, constant int* nsub, constant float* modes, uint nj,
                    constant int* wsub) {
    constant float* q = segs + 8 * s;
    if (DIFF) {  // a moving spin: its effective position for this segment; position-keyed caches are stale
        const float3 eff = gm.kind ? walk_restricted(wk, s, q[0], uint(wsub[s]), gm.kind, gm.size, gm.ax, gm.c)
                                   : walk_segment(wk, s, q[0]);
        st.p.px = eff.x, st.p.py = eff.y, st.p.pz = eff.z;
        st.cache.dt = -1.0f, md.k_dt = -1.0f;
    }
    const float bz = offset(st.p, st.conc, V == 0 ? 0u : mode, q);
    if (V == 0) step(q, flags[s], bz, st.mx, st.my, st.mz, st.p, st.cache);
    else if (V == 2) step_gen(q, flags[s], bz, uint(nsub[s]), st.mx, st.my, st.mz, st.mb, md, st.p, st.b, modes, nj, st.cache, st.zc);
    else if (mt) step_mt(q, flags[s], bz, frf[s], st.mx, st.my, st.mz, st.mb, st.p, st.b, line, log_f, st.cache, st.zc);
    else step(q, flags[s], bz, st.mx, st.my, st.mz, st.p, st.cache);
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
                                constant uint* prm         [[buffer(20)]],  // (12,) n n_coils s0 s1 n_tiles n_pad mt mode nj seed_lo seed_hi kind
                                uint i    [[thread_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg   [[simdgroup_index_in_threadgroup]],
                                uint nsg  [[simdgroups_per_threadgroup]],
                                uint tid  [[thread_index_in_threadgroup]],
                                uint ntg  [[threads_per_threadgroup]]) {
    const uint n = prm[0], n_coils = prm[1], s0 = prm[2], s1 = prm[3], n_tiles = prm[4], n_pad = prm[5], mt = prm[6], mode = prm[7], nj = prm[8], seed_lo = prm[9], seed_hi = prm[10], kind = prm[11];

    // ADC samples are buffered per simdgroup in threadgroup memory and reduced a block at a time:
    // two barriers per block of samples instead of two per sample, and the final sums are spread
    // over all threads of the group.
    threadgroup float partial[TG_FLOATS];
    threadgroup int block_rows[TG_FLOATS / (MAX_SG * 2)];
    const bool active = i < n;
    const uint j = active ? i : 0;
    State st;
    st.mx = active ? M[j] : 0.0f, st.my = active ? M[n + j] : 0.0f, st.mz = active ? M[2 * n + j] : 0.0f;
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
        for (uint c = 0; c < 6; ++c) wk.l[c] = active ? chol[c * n + j] : 0.0f;
        wk.seed = uint2(seed_lo, seed_hi), wk.spin = j;
    }
    Geom gm;
    if (@DIFF@ && kind) {
        gm.kind = kind, gm.size = geom[j];
        gm.ax = float3(geom[n + j], geom[2 * n + j], geom[3 * n + j]);
        gm.c = float3(geom[4 * n + j], geom[5 * n + j], geom[6 * n + j]);
    }
    Modes md;
    for (uint m = 0; @V@ == 2 && m < nj; ++m) md.ur[m] = active ? U[2 * m * n + j] : 0.0f, md.ui[m] = active ? U[(2 * m + 1) * n + j] : 0.0f;
    thread float &mx = st.mx, &my = st.my, &mz = st.mz;
    const uint nv = 2 * n_coils;
    const uint block = TG_FLOATS / (MAX_SG * nv);  // samples buffered per flush
    uint nb = 0;

    for (uint s = s0; s < s1; ++s) {
        constant float* q = segs + 8 * s;
        const int f = flags[s];
        advance<@V@, @DIFF@>(st, md, wk, gm, segs, flags, frf, s, mt, line, log_f, mode, nsub, modes, nj, wsub);
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
    if (active) {
        M[j] = mx; M[n + j] = my; M[2 * n + j] = mz;
        if (@V@ != 0 && mt) MB[j] = st.mb;
        for (uint m = 0; @V@ == 2 && m < nj; ++m) U[2 * m * n + j] = md.ur[m], U[(2 * m + 1) * n + j] = md.ui[m];
        if (@DIFF@) P[j] = wk.x, P[n + j] = wk.y, P[2 * n + j] = wk.z;
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
                                    constant uint* prm         [[buffer(20)]],  // (12,) n n_coils s0 s1 n_tiles n_pad mt mode nj seed_lo seed_hi kind
                                    uint i    [[thread_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg   [[simdgroup_index_in_threadgroup]],
                                    uint nsg  [[simdgroups_per_threadgroup]],
                                    uint tid  [[thread_index_in_threadgroup]],
                                    uint ntg  [[threads_per_threadgroup]]) {
    const uint n = prm[0], n_coils = prm[1], s0 = prm[2], s1 = prm[3], n_tiles = prm[4], n_pad = prm[5], mt = prm[6], mode = prm[7], nj = prm[8], seed_lo = prm[9], seed_hi = prm[10], kind = prm[11];

    threadgroup float buf[MAX_SG * MMA_REGION];
    threadgroup int rows[8];
    threadgroup float psi[8];
    threadgroup float* mine = buf + sg * MMA_REGION;
    const bool active = i < n;
    const uint j = active ? i : 0;
    State st;
    st.mx = active ? M[j] : 0.0f, st.my = active ? M[n + j] : 0.0f, st.mz = active ? M[2 * n + j] : 0.0f;
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
        for (uint c = 0; c < 6; ++c) wk.l[c] = active ? chol[c * n + j] : 0.0f;
        wk.seed = uint2(seed_lo, seed_hi), wk.spin = j;
    }
    Geom gm;
    if (@DIFF@ && kind) {
        gm.kind = kind, gm.size = geom[j];
        gm.ax = float3(geom[n + j], geom[2 * n + j], geom[3 * n + j]);
        gm.c = float3(geom[4 * n + j], geom[5 * n + j], geom[6 * n + j]);
    }
    Modes md;
    for (uint m = 0; @V@ == 2 && m < nj; ++m) md.ur[m] = active ? U[2 * m * n + j] : 0.0f, md.ui[m] = active ? U[(2 * m + 1) * n + j] : 0.0f;
    thread float &mx = st.mx, &my = st.my, &mz = st.mz;
    const uint j0 = i - lane;
    uint nb = 0;

    for (uint s = s0; s < s1; ++s) {
        constant float* q = segs + 8 * s;
        const int f = flags[s];
        advance<@V@, @DIFF@>(st, md, wk, gm, segs, flags, frf, s, mt, line, log_f, mode, nsub, modes, nj, wsub);
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
    if (active) {
        M[j] = mx; M[n + j] = my; M[2 * n + j] = mz;
        if (@V@ != 0 && mt) MB[j] = st.mb;
        for (uint m = 0; @V@ == 2 && m < nj; ++m) U[2 * m * n + j] = md.ur[m], U[(2 * m + 1) * n + j] = md.ui[m];
        if (@DIFF@) P[j] = wk.x, P[n + j] = wk.y, P[2 * n + j] = wk.z;
    }
}
"""

_LIB = None
MAX_COILS = 32  # the direct (simd_sum) reduction; the matrix-product one has no limit
MMA_MIN_COILS = 2


def _lib():
    global _LIB
    if _LIB is None:
        # each kernel three times (advance<V>): plain, with the rate-model bound pool and/or the field
        # model, and with the generalized Bloch modes; extra per-thread state costs the plain
        # kernels ~10-30 % in register pressure otherwise
        kernels = "".join(_KERNELS.replace("@SUFFIX@", suffix + dsuf).replace("@V@", v).replace("@DIFF@", diff)
                          for suffix, v in (("", "0"), ("_full", "1"), ("_gen", "2"))
                          for dsuf, diff in (("", "false"), ("_diff", "true")))
        _LIB = torch.mps.compile_shader(_SOURCE + kernels)
    return _LIB


def available() -> bool:
    try:
        return torch.backends.mps.is_available() and _lib() is not None
    except Exception:
        return False


@torch.no_grad()
def simulate_metal(phantom: Phantom, seq: Sequence, coils: Tensor | None = None, *, m_init: Tensor | None = None,
                   chunk: int = 32768, group: int = 256, scanner=None, mma: bool | None = None,
                   seed: int = 0) -> SimResult:
    """Forward simulation on the Apple GPU; same arguments and result as ``blochparty.simulate``
    (results come back on the CPU, float32). ``chunk`` segments per dispatch; ``group`` threads
    per threadgroup (must be 256). ``mma`` picks the coil reduction: matrix products on simdgroup
    matrices (default from ``MMA_MIN_COILS`` coils) or one simd_sum per coil and sample."""

    if getattr(phantom, "pools", None):
        raise NotImplementedError("free exchanging pools (Phantom.pools) are only in blochparty.simulate")
    dev, f32 = torch.device("mps"), torch.float32
    n = len(phantom)
    b1 = phantom.b1 if getattr(phantom, "b1", None) is not None else torch.ones(n)
    fields = spin_fields(scanner, phantom.pos)  # encoding positions (nonlinearity), concomitant coefficients
    spins = torch.stack([fields.pos[:, 0], fields.pos[:, 1], fields.pos[:, 2], phantom.pd, 1 / phantom.t1,
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
    pool = None if phantom.bound is None else phantom.bound.expand(n, phantom.pd)
    if pool is not None:  # magnetization transfer: the bound pool's Mz, parameters, RF frequency, lineshape
        mb = pool.f * phantom.pd if m_init is None or m_init.shape[-1] == 3 else m_init[:, 3]
        if m_init is not None and m_init.shape[-1] == 4:
            M = M[:3]
        mt_bufs = [mb.detach().to(dev, f32).contiguous(),
                   torch.stack([pool.f, pool.k, 1 / pool.t1, pool.t2, pool.dw]).detach().to(dev, f32).contiguous()]
        gen_bufs = None
        if pool.model == "generalized":  # modes instead of the saturation lineshape
            from ..generalized_bloch import modes as green_modes

            pm, cm = green_modes(pool.lineshape)
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
    M = M.to(dev, f32).contiguous()
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
    if phantom.diffusion is not None:  # walkers start at the (encoding) positions; L L^T = D
        D = phantom.diffusion.detach().double().cpu()
        L = torch.diag_embed(torch.sqrt(D).unsqueeze(-1).expand(n, 3)) if D.dim() == 1 else torch.linalg.cholesky(D)
        chol = torch.stack([L[:, 0, 0], L[:, 1, 0], L[:, 1, 1], L[:, 2, 0], L[:, 2, 1], L[:, 2, 2]])
        start = fields.pos.detach().double().cpu()
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
    else:
        dz = torch.zeros(1, device=dev, dtype=f32)
        walk_bufs = [dz, dz, 0, 0, torch.zeros(1, device=dev, dtype=torch.int32), dz, 0]
    suffix = "_gen" if gen_bufs[3] else ("_full" if (mt_bufs[5] or conc_buf[1]) else "")
    if phantom.diffusion is not None:
        suffix += "_diff"
    lib, S = _lib(), len(seq)
    # buffers 7-19 (MB pool frf line log_f | conc | U modes nsub | P chol | wsub geom) and the scalars
    MB, pool_b, frf_b, line_b, logf_b, mt_flag = mt_bufs
    conc_b, mode_flag = conc_buf
    U_b, modes_b, nsub_b, nj_v = gen_bufs
    P_b, chol_b, seed_lo, seed_hi, wsub_b, geom_b, kind_v = walk_bufs
    bufs = (MB, pool_b, frf_b, line_b, logf_b, conc_b, U_b, modes_b, nsub_b, P_b, chol_b, wsub_b, geom_b)

    def prm(s0, n_tiles, n_pad):
        v = np.array([n, n_coils, s0, min(S, s0 + chunk), n_tiles, n_pad, mt_flag, mode_flag, nj_v, seed_lo, seed_hi,
                      kind_v], dtype=np.uint32)
        return torch.from_numpy(v.view(np.int32)).to(dev)

    for s0 in range(0, S, chunk):
        if mma:
            getattr(lib, "bloch_chunk_mma" + suffix)(M, spins, sens, segs, flags, rows, sig, *bufs, prm(s0, n_tiles, threads),
                                                     threads=threads, group_size=group)
        else:
            getattr(lib, "bloch_chunk" + suffix)(M, spins, sens, segs, flags, rows, sig, *bufs, prm(s0, 0, 0),
                                                 threads=threads, group_size=group)
    torch.mps.synchronize()
    sig = sig[: n_adc * n_coils * 2].cpu().view(n_adc, n_coils, 2)
    signal = torch.complex(sig[..., 0], sig[..., 1]).T.contiguous()
    if pool is not None:
        return SimResult(signal, M.cpu().T.contiguous(), mt_bufs[0].cpu())
    return SimResult(signal, M.cpu().T.contiguous())
