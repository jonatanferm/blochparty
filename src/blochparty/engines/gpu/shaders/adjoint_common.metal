
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
#define RBLOCK 4  // segments per recomputed block (divides 64; tg: RBLOCK x 4 x 256 floats = 16 KB; the selects
                  // of the samples' source terms in bloch_adjoint assume 4)
// rows of R per block start: the magnetization only (the bound pool's in the variants that have one); the
// recomputation starts each block with zero low parts, one ulp from the forward and the same for any K1
template <int V> constexpr uint rrows() { return V == 0 ? 3u : 4u; }

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
    if (line[0] > 2.5f) {  // no cut: G = T2b sum_j Re c_j / (p_j + i x), x = 2 pi delta T2b (lineshape kind 3)
        const float x = TWO_PI * delta * t2b;
        float g = 0.0f, dg = 0.0f;  // the sum and its derivative in x
        for (int k = 0; k < int(line[5]); ++k) {
            const float pr = log_f[4 * k], pim = log_f[4 * k + 1] + x, cr = log_f[4 * k + 2], ci = log_f[4 * k + 3];
            const float den = pr * pr + pim * pim, num = cr * pr + ci * pim;
            g += num / den, dg += (ci * den - 2.0f * pim * num) / (den * den);
        }
        g_t2 = g + t2b * dg * TWO_PI * delta, g_d = t2b * dg * TWO_PI * t2b;
        return t2b * g;
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
