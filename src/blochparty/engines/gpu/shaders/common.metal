
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
    if (line[0] > 2.5f) {  // no cut (cutoff=None): the tapered Green's function's transform from its
        // modes, G = T2b Re sum_j c_j / (p_j + i x); log_f holds (Re p, Im p, Re c, Im c) per mode
        const float x = TWO_PI * delta * t2b;
        const int nj = int(line[5]);
        float g = 0.0f;
        for (int j = 0; j < nj; ++j) {
            const float pr = log_f[4 * j], pim = log_f[4 * j + 1] + x, cr = log_f[4 * j + 2], ci = log_f[4 * j + 3];
            g += (cr * pr + ci * pim) / (pr * pr + pim * pim);
        }
        return t2b * g;
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
        if (f & 16) rotate_inc(0.0f, 0.0f, -TWO_PI * dt * bz, mx, my, mz, lx, ly, lz);  // bound-only RF: water precesses
        else rf_rotate(q, bz, mx, my, mz, lx, ly, lz, p);
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
#define MAX_TERMS 8     // NUFFT readouts: decay terms (time segmentation)
#define NU_HEAD 14      // NUFFT header floats: mg w terms beta n_adc wmax tau[8]

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

