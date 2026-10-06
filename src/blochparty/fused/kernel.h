// Fused Bloch kernel: the math of bloch.py, but one pass over the sequence per block of
// spins, with the block's state in L1 instead of ~30 tensor ops over all spins per
// segment. simulate() runs it forward; adjoint() is its hand-written reverse mode
// (gradients w.r.t. every input, see the section "Adjoint" below).
//
// Written against raw pointers and sizes only (no PyTorch), so a standalone C++
// program can include this header and call simulate<T>() directly.
//
// Layout: spins are processed in blocks of B (structure of arrays inside a block, so
// the loops over the B lanes vectorize); each block steps through every segment; a
// pool of std::threads pulls blocks from a shared counter. Each thread accumulates
// raw (re, im) coil sums into its own (n_adc, C, 2) buffer; the buffers are added in
// thread order at the end and the receiver phase is applied once (it is linear).
// Which thread gets which block varies, so results agree run to run only to rounding.
//
// Per-lane factors depend only on a few segment parameters: relaxation on the duration,
// free precession on (dt, grad), an RF rotation on (dt, rf, grad). A block keeps the
// last few of each and recomputes only on a miss, so a readout of n identical segments
// costs one sin/cos per spin, not n; the inputs are the same, so the results are too.
//
// Field model (see fields.py): a lane's off-resonance in segment s is
//     lin = pos.g + db0 + b0_offset[s]  plus  the concomitant field f_c(g),
// f_c = conc . (gx^2, gy^2, gz^2, gx gy, gx gz, gy gz) at leading order (conc: (n, 6)), or
// in exact mode (conc: (n, 7) = ax, ay, f0) |B| - f0 - lin = t^2 / (sqrt(u^2 + t^2) + u) with
// t = (ax.g, ay.g) and u = f0 + lin. Without it the arithmetic is that of the ideal model
// (b0_offset adds an exact zero), so results are unchanged bit for bit.
//
// One deliberate difference in form: relaxation is m += (m - m0) * expm1(-t / T)
// instead of m = m0 + (m - m0) * exp(-t / T). Algebraically identical; in float32 it
// keeps the small per-step decrement exact (exp(-dwell / T1) rounds to ~1).

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace blochparty::fused {

// Segment flags
constexpr uint8_t kRF = 1;     // rotate about the RF + off-resonance field (Strang-split relaxation)
constexpr uint8_t kADC = 2;    // sample the signal at the end of the segment
constexpr uint8_t kSpoil = 4;  // ideal spoiler: zero Mx, My at the end of the segment

// Bound-pool lineshape G(2 pi delta) [s] (exchange.py; BMCTool's normalisation G = pi g) and
// its partial derivatives d/d(t2b), d/d(delta). The super-Lorentzian is interpolated linearly
// in log F over a uniform log-x table (x = 2 pi max(|delta|, cutoff) t2b), as exchange.py.
constexpr int kLineNone = 0;
constexpr int kLorentzian = 1;
constexpr int kSuperLorentzian = 2;
constexpr double kSlXmax = 30.0;  // exchange._SL_XMAX: G = 0 beyond

template <typename T>
struct Lineshape {
    int kind = kLineNone;
    T cutoff = T(0);
    const T* log_f = nullptr;  // (n,) log F at log x = x0 + i h
    T x0 = T(0), h = T(1), xmin = T(0);
    int64_t n = 0;

    T eval(T delta, T t2b, T& g_t2b, T& g_delta) const {
        const T two_pi = T(2.0 * 3.14159265358979323846);
        if (kind == kLorentzian) {
            const T x = two_pi * delta * t2b, den = T(1) + x * x;
            g_t2b = (T(1) - x * x) / (den * den);
            g_delta = -t2b * T(2) * x * two_pi * t2b / (den * den);
            return t2b / den;
        }
        const T ad = std::abs(delta);
        const bool cut = ad < cutoff;
        const T d = cut ? cutoff : ad;
        const T x = two_pi * d * t2b;
        if (x > T(kSlXmax)) return g_t2b = g_delta = T(0), T(0);
        const bool low = x < xmin;
        const T lx = std::log(low ? xmin : x);
        T pos = (lx - x0) / h;
        const T top = T(n - 1) - T(1e-9);
        const bool clamped = pos < T(0) || pos > top;
        pos = pos < T(0) ? T(0) : (pos > top ? top : pos);
        const int64_t i = static_cast<int64_t>(std::floor(pos));
        const T a = pos - T(i);
        const T val = std::exp(log_f[i] * (T(1) - a) + log_f[i + 1] * a);
        // d val / dx = val * dlogF/dlogx / x, zero where log x or the table position is clamped
        const T dval_dx = low || clamped ? T(0) : val * (log_f[i + 1] - log_f[i]) / h / x;
        g_t2b = val + t2b * dval_dx * two_pi * d;
        g_delta = cut ? T(0) : t2b * dval_dx * two_pi * t2b * (delta > T(0) ? T(1) : (delta < T(0) ? T(-1) : T(0)));
        return t2b * val;
    }
};

template <typename T>
struct Spins {
    int64_t n;        // number of spins
    int64_t n_coils;  // C
    const T* pos;     // (n, 3) [m]
    const T* pd;      // (n,)
    const T* t1;      // (n,) [s]
    const T* t2;      // (n,) [s]
    const T* db0;     // (n,) [Hz]
    const T* sens;    // (n, C, 2) receive sensitivities, (re, im) interleaved
    const T* b1;      // (n,) transmit scale of every RF pulse, or null for 1
    const T* conc;    // (n, 6) or (n, 7) concomitant-field coefficients (see conc_mode), or null
    int conc_mode;    // kConcNone, kConcLeading (conc is (n, 6)) or kConcExact ((n, 7))
    // Magnetization transfer: (n, 5) bound pool f, k [Hz], t1b [s], t2b [s], dw [Hz], or null.
    // With it the magnetization (m, checkpoints) is (n, 4): mx, my, mz, mzb.
    const T* bound = nullptr;
    Lineshape<T> line;

    int n_m() const { return bound ? 4 : 3; }

    Spins(int64_t n, int64_t n_coils, const T* pos, const T* pd, const T* t1, const T* t2, const T* db0,
          const T* sens, const T* b1 = nullptr, const T* conc = nullptr, int conc_mode = 0)
        : n(n), n_coils(n_coils), pos(pos), pd(pd), t1(t1), t2(t2), db0(db0), sens(sens), b1(b1), conc(conc),
          conc_mode(conc ? conc_mode : 0) {}
};

template <typename T>
struct Segments {
    int64_t n;             // number of segments
    const T* dt;           // (n,) [s]
    const T* rf;           // (n, 2) [Hz]
    const T* grad;         // (n, 3) [Hz/m]
    const T* adc_phase;    // (n,) [rad]
    const uint8_t* flags;  // (n,) kRF | kADC | kSpoil
    const T* b0_offset = nullptr;  // (n,) uniform off-resonance [Hz], or null for 0
    const T* rf_freq = nullptr;    // (n,) RF frequency [Hz] for the bound-pool saturation (Spins::bound)

    T offset(int64_t s) const { return b0_offset ? b0_offset[s] : T(0); }
};

// Concomitant-field modes (Spins::conc_mode)
constexpr int kConcNone = 0;
constexpr int kConcLeading = 1;
constexpr int kConcExact = 2;
constexpr int kConcMax = 7;  // coefficients per spin in exact mode

// Gradient outputs; a null pointer means "not needed". Sequence gradients are summed
// over spins; everything else is per spin.
template <typename T>
struct Grads {
    T* dt = nullptr;         // (n_seg,)
    T* rf = nullptr;         // (n_seg, 2)
    T* grad = nullptr;       // (n_seg, 3)
    T* adc_phase = nullptr;  // (n_seg,)
    T* pos = nullptr;        // (n, 3)
    T* pd = nullptr;         // (n,)
    T* t1 = nullptr;         // (n,)
    T* t2 = nullptr;         // (n,)
    T* db0 = nullptr;        // (n,)
    T* sens = nullptr;       // (n, C, 2): d/d(re), d/d(im)
    T* b1 = nullptr;         // (n,)
    T* m = nullptr;          // (n, 3) initial magnetization
    T* b0_offset = nullptr;  // (n_seg,)
    T* conc = nullptr;       // (n, 6) or (n, 7), as Spins::conc
    T* bound = nullptr;      // (n, 5), as Spins::bound
    T* rf_freq = nullptr;    // (n_seg,)
};

inline int64_t count_adc(const uint8_t* flags, int64_t n) {
    int64_t k = 0;
    for (int64_t s = 0; s < n; ++s) k += (flags[s] & kADC) != 0;
    return k;
}

// Number of checkpoints simulate() stores: one before every `every` segments.
inline int64_t count_checkpoints(int64_t n_segments, int64_t every) { return (n_segments + every - 1) / every; }

namespace detail {

constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------------------------------
// Magnetization transfer (exchange.py): the exact 2x2 longitudinal step of the free pool's
// Mz and the bound pool's Mzb, d/dt z = A z + c, in the increment form of exchange.z_exchange:
//   z' = z + (e^{At} - I)(z - z_ss),  e^{At} = e^{mt} [cosh(qt) I + sinh(qt)/q H],
// m = tr A / 2, H = A - m I = [[h, a12], [a21, -h]], q^2 = p = h^2 + a12 a21, z_ss = -A^-1 c.
// chm1 = e^{mt} cosh(qt) - 1 and sh = e^{mt} sinh(qt) / q come from the eigenvalues m -/+ q
// (both <= 0) without cancellation or overflow.
template <typename T>
struct ZStep {
    T a11, a12, a21, a22, m, h, p, det, z1, z2, chm1, sh;

    ZStep() = default;
    ZStep(T t, T a11_, T a12_, T a21_, T a22_, T c1, T c2) : a11(a11_), a12(a12_), a21(a21_), a22(a22_) {
        m = T(0.5) * (a11 + a22), h = T(0.5) * (a11 - a22);
        p = h * h + a12 * a21;
        const T q = std::sqrt(p);
        det = a11 * a22 - a12 * a21;
        z1 = -(a22 * c1 - a12 * c2) / det, z2 = -(a11 * c2 - a21 * c1) / det;
        chm1 = T(0.5) * (std::expm1((m + q) * t) + std::expm1((m - q) * t));
        if (T(2) * q * t < T(1)) {
            const T lo = std::exp((m - q) * t), e2q = std::expm1(T(2) * q * t);
            sh = q > T(0) ? lo * T(0.5) * e2q / q : t * std::exp(m * t);
        } else {
            sh = T(0.5) * (std::exp((m + q) * t) - std::exp((m - q) * t)) / q;
        }
    }

    void apply(T& u, T& v) const {
        const T d1 = u - z1, d2 = v - z2;
        const T n1 = chm1 * d1 + sh * (h * d1 + a12 * d2), n2 = chm1 * d2 + sh * (a21 * d1 - h * d2);
        u = u + n1, v = v + n2;
    }

    // Reverse: (l1, l2) = dL/dz' in, dL/dz out; y* receive dL/d(t, A, c) for input z = (u, v).
    void reverse(T t, T u, T v, T& l1, T& l2, T& yt, T& y11, T& y12, T& y21, T& y22, T& yc1, T& yc2) const {
        const T ch = T(1) + chm1;
        const T x2 = p * t * t;
        // d sh / dp = e^{mt} t^3 / 2 g(qt), g(x) = (x cosh x - sinh x) / x^3 = 1/3 + x^2/30 + ...
        const T sg = x2 < T(0.25)
                         ? std::exp(m * t) * T(0.5) * t * t * t *
                               (T(1.0 / 3) + x2 * (T(1.0 / 30) + x2 * (T(1.0 / 840) + x2 * (T(1.0 / 45360) + x2 * T(1.0 / 3991680)))))
                         : (t * ch - sh) / (T(2) * p);
        const T d1 = u - z1, d2 = v - z2;
        const T e1 = chm1 * l1 + sh * (h * l1 + a21 * l2), e2 = chm1 * l2 + sh * (a12 * l1 - h * l2);  // (E - I)^T l
        const T ych = l1 * d1 + l2 * d2, ysh = l1 * (h * d1 + a12 * d2) + l2 * (a21 * d1 - h * d2);
        T yh = sh * (l1 * d1 - l2 * d2), ya12 = sh * l1 * d2, ya21 = sh * l2 * d1;
        const T ym = t * (ych * ch + ysh * sh);
        const T yp = ych * T(0.5) * t * sh + ysh * sg;
        yt = ych * (m * ch + p * sh) + ysh * (m * sh + ch);
        yh += T(2) * h * yp, ya12 += a21 * yp, ya21 += a12 * yp;
        // through z_ss = -A^-1 c: dL/dz_ss = -(E - I)^T l = -e; nu = -A^-T dL/dz_ss = dL/dc
        const T n1 = (a22 * e1 - a21 * e2) / det, n2 = (a11 * e2 - a12 * e1) / det;
        y11 = T(0.5) * (ym + yh) + n1 * z1, y22 = T(0.5) * (ym - yh) + n2 * z2;
        y12 = ya12 + n1 * z2, y21 = ya21 + n2 * z1;
        yc1 = n1, yc2 = n2;
        l1 = l1 + e1, l2 = l2 + e2;
    }
};


// Per-lane factors for the last K distinct keys (NK values each), least recently used
// out. A slot holds NA arrays of B lanes.
template <typename T, int B, int NK, int NA, int K>
struct Cache {
    alignas(64) T val[K][NA][B];
    T key[K][NK];
    uint64_t used[K];  // last use; 0 = empty
    uint64_t clock;

    void clear() { std::fill(used, used + K, uint64_t(0)), clock = 0; }

    // The slot for `k`; `fill` is set when it is new and its values must be computed.
    T (*find(const T (&k)[NK], bool& fill))[B] {
        ++clock;
        int slot = 0;
        for (int s = 0; s < K; ++s) {
            if (used[s] && std::equal(k, k + NK, key[s])) {
                used[s] = clock, fill = false;
                return val[s];
            }
            if (used[s] < used[slot]) slot = s;
        }
        std::copy(k, k + NK, key[slot]);
        used[slot] = clock, fill = true;
        return val[slot];
    }
};

template <typename T, int B>
struct Block {
    alignas(64) T px[B], py[B], pz[B], db0[B], pd[B], r1[B], r2[B], b1[B];
    alignas(64) T mx[B], my[B], mz[B];
    alignas(64) T cf[kConcMax][B];  // concomitant-field coefficients (Spins::conc)
    alignas(64) T pf[2][B];         // precession factors including a uniform offset (precession())
    // bound pool (Spins::bound): Mzb and the pool parameters, k and f as kb = k, kf = k f
    alignas(64) T mb[B], bf[B], bk[B], br1[B], bt2[B], bdw[B];
    bool bound = false;
    int nm = 3;  // magnetization components per spin in m and the checkpoints
    Lineshape<T> line;
    int mode = kConcNone;
    std::vector<T> sr, si;  // (C, B) sensitivities
    // Relaxation over a duration: expm1(-t * R1), expm1(-t * R2). Key: t.
    Cache<T, B, 1, 2, 8> relax;
    // Free precession: cos and sin of the phase. Key: dt, grad, b0_offset.
    Cache<T, B, 5, 2, 8> prec;
    // RF rotation: cos(phi), sin(phi) / phi, (1 - cos(phi)) / phi^2 and wz. Key: dt, rf, grad, b0_offset.
    Cache<T, B, 7, 4, 4> rot;
    // Unsaturated exchange step over a duration (ZStep): chm1, sh, z_ss (2), h. Key: t.
    Cache<T, B, 1, 5, 8> zfree;
    // Saturated half step of an RF segment: the same factors. The saturation depends on the RF
    // through |rf|^2 only, so RF-spoiled repeats hit. Key: dt, |rf|^2, grad, b0_offset, f_rf.
    Cache<T, B, 7, 5, 4> zrf;

    explicit Block(int64_t n_coils) : sr(n_coils * B), si(n_coils * B) {}

    // Copy spins [start, start + B) in; lanes past the end get PD = 0, rates 0 and
    // zero sensitivity, so they stay at zero and add nothing to the signal.
    void load(const Spins<T>& sp, const T* m, int64_t start) {
        const int64_t nc = sp.n_coils;
        relax.clear(), prec.clear(), rot.clear(), zfree.clear(), zrf.clear();
        mode = sp.conc_mode;
        bound = sp.bound != nullptr, nm = sp.n_m(), line = sp.line;
        const int n_cf = mode == kConcExact ? 7 : 6;
        for (int q = 0; q < kConcMax; ++q)
            for (int l = 0; l < B; ++l) {
                // lanes past the end get a huge f0 in exact mode (no transverse field), so that
                // u = f0 + offset > 0 and their excess is 0, not 0 / 0
                const T pad = mode == kConcExact && q == 6 ? T(1e15) : T(0);
                cf[q][l] = mode && q < n_cf && start + l < sp.n ? sp.conc[n_cf * (start + l) + q] : pad;
            }
        for (int l = 0; l < B; ++l) {
            const int64_t i = start + l;
            const bool in = i < sp.n;
            px[l] = in ? sp.pos[3 * i] : T(0);
            py[l] = in ? sp.pos[3 * i + 1] : T(0);
            pz[l] = in ? sp.pos[3 * i + 2] : T(0);
            db0[l] = in ? sp.db0[i] : T(0);
            pd[l] = in ? sp.pd[i] : T(0);
            r1[l] = in ? T(1) / sp.t1[i] : T(0);
            r2[l] = in ? T(1) / sp.t2[i] : T(0);
            b1[l] = in && sp.b1 ? sp.b1[i] : T(1);
            if (bound) {  // padded lanes: an inert pool (f = 0, no exchange, unit rates: det A > 0, no 0 / 0)
                if (!in) r1[l] = T(1);
                const T* q = sp.bound + 5 * i;
                bf[l] = in ? q[0] : T(0), bk[l] = in ? q[1] : T(0), br1[l] = in ? T(1) / q[2] : T(1);
                bt2[l] = in ? q[3] : T(1), bdw[l] = in ? q[4] : T(0);
            }
            for (int64_t c = 0; c < nc; ++c) {
                sr[c * B + l] = in ? sp.sens[2 * (i * nc + c)] : T(0);
                si[c * B + l] = in ? sp.sens[2 * (i * nc + c) + 1] : T(0);
            }
        }
        load_m(m, start, sp.n);
    }

    // m may be null (zeros); (n, nm).
    void load_m(const T* m, int64_t start, int64_t n) {
        for (int l = 0; l < B; ++l) {
            const int64_t i = start + l;
            const bool in = m && i < n;
            mx[l] = in ? m[nm * i] : T(0);
            my[l] = in ? m[nm * i + 1] : T(0);
            mz[l] = in ? m[nm * i + 2] : T(0);
            if (bound) mb[l] = in ? m[nm * i + 3] : T(0);
        }
    }

    void store(T* m, int64_t start, int64_t n) const {
        for (int l = 0; l < B && start + l < n; ++l) {
            m[nm * (start + l)] = mx[l];
            m[nm * (start + l) + 1] = my[l];
            m[nm * (start + l) + 2] = mz[l];
            if (bound) m[nm * (start + l) + 3] = mb[l];
        }
    }

    // The exchange step of lane i over t with saturation rate w (exchange.z_exchange).
    ZStep<T> zstep(int i, T t, T w) const {
        const T kf = bk[i] * bf[i];
        return ZStep<T>(t, -(r1[i] + kf), bk[i], kf, -(br1[i] + bk[i] + w), r1[i] * pd[i], br1[i] * bf[i] * pd[i]);
    }

    // Bound-pool saturation rate of lane i in an RF segment: w1^2 G(2 pi (f_rf - offset - dw)).
    T saturation(int i, T a2, T f_rf, T bz) const {
        T g_t2, g_d;
        const T two_pi = T(2.0 * kPi);
        return two_pi * two_pi * (b1[i] * b1[i]) * a2 * line.eval(f_rf - bz - bdw[i], bt2[i], g_t2, g_d);
    }

    // advance() with a bound pool: the longitudinal relaxation becomes the exact exchange step
    // (saturated during RF); the transverse arithmetic is advance()'s.
    void advance_bound(const Segments<T>& seq, int64_t s) {
        const T w_per_s = T(-2.0 * kPi);
        const T dt = seq.dt[s];
        const T gx = seq.grad[3 * s], gy = seq.grad[3 * s + 1], gz = seq.grad[3 * s + 2];
        const T f = seq.offset(s);
        const T w = w_per_s * dt;
        bool fill;
        if (seq.flags[s] & kRF) {
            const T rx = seq.rf[2 * s], ry = seq.rf[2 * s + 1], a2 = rx * rx + ry * ry;
            const T wx0 = w * rx, wy0 = w * ry, h_dt = T(0.5) * dt, f_rf = seq.rf_freq[s];
            const T (*d)[B] = relaxation(h_dt);
            T (*r)[B] = rot.find({dt, rx, ry, gx, gy, gz, f}, fill);
            if (fill)
                for (int i = 0; i < B; ++i) rotation(r, i, w, wx0 * b1[i], wy0 * b1[i], gx, gy, gz, f);
            T (*e)[B] = zrf.find({dt, a2, gx, gy, gz, f, f_rf}, fill);
            if (fill)
                for (int i = 0; i < B; ++i) {
                    const ZStep<T> zs = zstep(i, h_dt, saturation(i, a2, f_rf, offset(i, gx, gy, gz, f)));
                    e[0][i] = zs.chm1, e[1][i] = zs.sh, e[2][i] = zs.z1, e[3][i] = zs.z2, e[4][i] = zs.h;
                }
            for (int i = 0; i < B; ++i) {
                const T chm1 = e[0][i], sh = e[1][i], hh = e[4][i], kf = bk[i] * bf[i];
                auto apply = [&](T& u, T& v) {  // ZStep::apply with the cached factors
                    const T d1 = u - e[2][i], d2 = v - e[3][i];
                    const T n1 = chm1 * d1 + sh * (hh * d1 + bk[i] * d2), n2 = chm1 * d2 + sh * (kf * d1 - hh * d2);
                    u = u + n1, v = v + n2;
                };
                const T wx = wx0 * b1[i], wy = wy0 * b1[i];
                const T x = mx[i] + mx[i] * d[1][i], y = my[i] + my[i] * d[1][i];
                T z = mz[i], b = mb[i];
                apply(z, b);
                const T c = r[0][i], s1 = r[1][i], s2 = r[2][i], wz = r[3][i];
                const T dot = wx * x + wy * y + wz * z;
                const T nx = c * x + s1 * (wy * z - wz * y) + s2 * wx * dot;
                const T ny = c * y + s1 * (wz * x - wx * z) + s2 * wy * dot;
                T nz = c * z + s1 * (wx * y - wy * x) + s2 * wz * dot;
                apply(nz, b);
                mx[i] = nx + nx * d[1][i];
                my[i] = ny + ny * d[1][i];
                mz[i] = nz, mb[i] = b;
            }
        } else {
            const T (*d)[B] = relaxation(dt);
            T (*p)[B] = precession(dt, w, gx, gy, gz, f);
            T (*e)[B] = zfree.find({dt}, fill);
            if (fill)
                for (int i = 0; i < B; ++i) {
                    const ZStep<T> zs = zstep(i, dt, T(0));
                    e[0][i] = zs.chm1, e[1][i] = zs.sh, e[2][i] = zs.z1, e[3][i] = zs.z2, e[4][i] = zs.h;
                }
#pragma omp simd
            for (int i = 0; i < B; ++i) {
                const T x = p[0][i] * mx[i] - p[1][i] * my[i];
                const T y = p[1][i] * mx[i] + p[0][i] * my[i];
                mx[i] = x + x * d[1][i];
                my[i] = y + y * d[1][i];
                const T chm1 = e[0][i], sh = e[1][i], hh = e[4][i], kf = bk[i] * bf[i];
                const T d1 = mz[i] - e[2][i], d2 = mb[i] - e[3][i];  // ZStep::apply with the cached factors
                const T n1 = chm1 * d1 + sh * (hh * d1 + bk[i] * d2), n2 = chm1 * d2 + sh * (kf * d1 - hh * d2);
                mz[i] = mz[i] + n1, mb[i] = mb[i] + n2;
            }
        }
    }

    // Raw coil sums: out[2c] += sum(mx * sr - my * si), out[2c + 1] += sum(mx * si + my * sr).
    // Four coils per pass give the vectorized loop 8 independent accumulators.
    void sample(T* out, int64_t n_coils) const {
        int64_t c = 0;
        for (; c + 4 <= n_coils; c += 4) {
            const T *a0 = &sr[c * B], *a1 = a0 + B, *a2 = a1 + B, *a3 = a2 + B;
            const T *b0 = &si[c * B], *b1 = b0 + B, *b2 = b1 + B, *b3 = b2 + B;
            T re0 = 0, re1 = 0, re2 = 0, re3 = 0, im0 = 0, im1 = 0, im2 = 0, im3 = 0;
#pragma omp simd reduction(+ : re0, re1, re2, re3, im0, im1, im2, im3)
            for (int i = 0; i < B; ++i) {
                const T x = mx[i], y = my[i];  // one FMA per product
                re0 += x * a0[i], re0 -= y * b0[i], im0 += x * b0[i], im0 += y * a0[i];
                re1 += x * a1[i], re1 -= y * b1[i], im1 += x * b1[i], im1 += y * a1[i];
                re2 += x * a2[i], re2 -= y * b2[i], im2 += x * b2[i], im2 += y * a2[i];
                re3 += x * a3[i], re3 -= y * b3[i], im3 += x * b3[i], im3 += y * a3[i];
            }
            out[2 * c] += re0, out[2 * c + 1] += im0, out[2 * c + 2] += re1, out[2 * c + 3] += im1;
            out[2 * c + 4] += re2, out[2 * c + 5] += im2, out[2 * c + 6] += re3, out[2 * c + 7] += im3;
        }
        for (; c < n_coils; ++c) {
            const T *a = &sr[c * B], *b = &si[c * B];
            T re = 0, im = 0;
#pragma omp simd reduction(+ : re, im)
            for (int i = 0; i < B; ++i)
                re += mx[i] * a[i], re -= my[i] * b[i], im += mx[i] * b[i], im += my[i] * a[i];
            out[2 * c] += re, out[2 * c + 1] += im;
        }
    }

    // d = expm1(-t * R1), expm1(-t * R2) for relaxation over t as m += (m - m0) * d.
    const T (*relaxation(T t))[B] {
        bool fill;
        T (*d)[B] = relax.find({t}, fill);
        if (fill)
            for (int i = 0; i < B; ++i) {
                d[0][i] = std::expm1(-t * r1[i]);
                d[1][i] = std::expm1(-t * r2[i]);
            }
        return d;
    }

    // Off-resonance [Hz] of lane i in a segment with gradient g and uniform offset f.
    T offset(int i, T gx, T gy, T gz, T f) const {
        const T lin = px[i] * gx + py[i] * gy + pz[i] * gz + db0[i] + f;
        if (mode == kConcNone) return lin;
        if (mode == kConcLeading)
            return lin + (cf[0][i] * (gx * gx) + cf[1][i] * (gy * gy) + cf[2][i] * (gz * gz) +
                          cf[3][i] * (gx * gy) + cf[4][i] * (gx * gz) + cf[5][i] * (gy * gz));
        const T tx = cf[0][i] * gx + cf[1][i] * gy + cf[2][i] * gz;
        const T ty = cf[3][i] * gx + cf[4][i] * gy + cf[5][i] * gz;
        const T t2 = tx * tx + ty * ty, u = cf[6][i] + lin;
        return lin + t2 / (std::sqrt(u * u + t2) + u);
    }

    // Rotation factors of an RF segment for lane i: cos(phi), sin(phi) / phi,
    // (1 - cos(phi)) / phi^2 and wz, for the rotation vector w = (wx, wy, wz) (wx, wy
    // already scaled by the lane's b1).
    void rotation(T (*r)[B], int i, T w, T wx, T wy, T gx, T gy, T gz, T f) const {
        const T wz = w * offset(i, gx, gy, gz, f);
        const T phi = std::sqrt(wx * wx + wy * wy + wz * wz + T(1e-30));
        const T hp = T(0.5) * phi;
        const T sh = std::sin(hp) / hp;
        r[0][i] = std::cos(phi);
        r[1][i] = std::sin(phi) / phi;  // sin(phi) / phi
        r[2][i] = T(0.5) * sh * sh;     // (1 - cos(phi)) / phi^2
        r[3][i] = wz;
    }

    // Step the block's magnetization through segment s (sampling and spoiling are up to
    // the caller).
    [[gnu::always_inline]] inline void advance(const Segments<T>& seq, int64_t s) {
        const T w_per_s = T(-2.0 * kPi);  // as in bloch.py: w = -TWO_PI * dt, rounded to T
        const T dt = seq.dt[s];
        const T gx = seq.grad[3 * s], gy = seq.grad[3 * s + 1], gz = seq.grad[3 * s + 2];
        const T f = seq.offset(s);
        const T w = w_per_s * dt;
        bool fill;
        if (seq.flags[s] & kRF) {
            // relax dt/2 -> rotate by w (Rodrigues) -> relax dt/2; lane i sees the RF times b1[i]
            // (exact for b1 = 1, so without a B1+ map the result is unchanged bit for bit)
            const T rx = seq.rf[2 * s], ry = seq.rf[2 * s + 1];
            const T wx0 = w * rx, wy0 = w * ry;
            const T (*d)[B] = relaxation(T(0.5) * dt);
            T (*r)[B] = rot.find({dt, rx, ry, gx, gy, gz, f}, fill);
            if (fill)
                for (int i = 0; i < B; ++i) rotation(r, i, w, wx0 * b1[i], wy0 * b1[i], gx, gy, gz, f);
            for (int i = 0; i < B; ++i) {
                const T wx = wx0 * b1[i], wy = wy0 * b1[i];
                const T x = mx[i] + mx[i] * d[1][i], y = my[i] + my[i] * d[1][i];
                const T z = mz[i] + (mz[i] - pd[i]) * d[0][i];
                const T c = r[0][i], s1 = r[1][i], s2 = r[2][i], wz = r[3][i];
                const T dot = wx * x + wy * y + wz * z;
                const T nx = c * x + s1 * (wy * z - wz * y) + s2 * wx * dot;
                const T ny = c * y + s1 * (wz * x - wx * z) + s2 * wy * dot;
                const T nz = c * z + s1 * (wx * y - wy * x) + s2 * wz * dot;
                mx[i] = nx + nx * d[1][i];
                my[i] = ny + ny * d[1][i];
                mz[i] = nz + (nz - pd[i]) * d[0][i];
            }
        } else {
            // exact precession about z, then exact relaxation (they commute)
            const T (*d)[B] = relaxation(dt);
            T (*p)[B] = precession(dt, w, gx, gy, gz, f);
            for (int i = 0; i < B; ++i) {
                const T x = p[0][i] * mx[i] - p[1][i] * my[i];
                const T y = p[1][i] * mx[i] + p[0][i] * my[i];
                mx[i] = x + x * d[1][i];
                my[i] = y + y * d[1][i];
                mz[i] = mz[i] + (mz[i] - pd[i]) * d[0][i];
            }
        }
    }

    // cos and sin of the free-precession phase w * offset for every lane. Except in exact
    // mode the uniform offset f only adds the same angle w f to every lane, so the cache is
    // keyed on the gradient alone and that rotation is applied on top (into `pf`): a
    // time-varying f then costs one sin / cos per segment, not one per lane.
    T (*precession(T dt, T w, T gx, T gy, T gz, T f))[B] {
        bool fill;
        const bool split = mode != kConcExact && f != T(0);
        T (*p)[B] = prec.find({dt, gx, gy, gz, split ? T(0) : f}, fill);
        if (fill)
            for (int i = 0; i < B; ++i) {
                const T phi = w * offset(i, gx, gy, gz, split ? T(0) : f);
                p[0][i] = std::cos(phi);
                p[1][i] = std::sin(phi);
            }
        if (!split) return p;
        const T cf_ = std::cos(w * f), sf_ = std::sin(w * f);
        for (int i = 0; i < B; ++i) {
            pf[0][i] = p[0][i] * cf_ - p[1][i] * sf_;
            pf[1][i] = p[1][i] * cf_ + p[0][i] * sf_;
        }
        return pf;
    }

    void spoil() {
        for (int i = 0; i < B; ++i) mx[i] = T(0), my[i] = T(0);
    }

    // Segments [s0, s1), whose first ADC sample is number k; returns the next sample number.
    int64_t run(const Segments<T>& seq, int64_t n_coils, T* acc, int64_t s0, int64_t s1, int64_t k) {
        for (int64_t s = s0; s < s1; ++s) {
            if (bound)
                advance_bound(seq, s);
            else
                advance(seq, s);
            const uint8_t f = seq.flags[s];
            if (f & kADC) sample(acc + 2 * n_coils * k++, n_coils);
            if (f & kSpoil) spoil();
        }
        return k;
    }
};

// Runs fn(t) on n_threads threads (t = 0 on the calling one) and rethrows the first error.
template <typename F>
void parallel(int n_threads, F&& fn) {
    std::vector<std::exception_ptr> errors(n_threads);
    auto worker = [&](int t) {
        try {
            fn(t);
        } catch (...) {
            errors[t] = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < n_threads; ++t) pool.emplace_back(worker, t);
    worker(0);
    for (auto& th : pool) th.join();
    for (auto& e : errors)
        if (e) std::rethrow_exception(e);
}

inline int clamp_threads(int n_threads, int64_t n_blocks) {
    if (n_threads <= 0) n_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(n_threads, n_blocks)));
}

template <typename T, int B>
void simulate_blocks(const Spins<T>& sp, const Segments<T>& seq, T* m, T* signal, int n_threads, T* ckpt,
                     int64_t every) {
    const int64_t nc = sp.n_coils;
    const int64_t n_adc = count_adc(seq.flags, seq.n);
    const int64_t n_blocks = (sp.n + B - 1) / B;
    const int64_t size = n_adc * nc * 2;
    if (!ckpt) every = std::max<int64_t>(seq.n, 1);
    n_threads = clamp_threads(n_threads, n_blocks);

    std::vector<std::vector<T>> acc(n_threads);
    std::atomic<int64_t> next{0};
    parallel(n_threads, [&](int t) {
        acc[t].assign(size, T(0));  // allocated by the thread that uses it
        auto blk = std::make_unique<Block<T, B>>(nc);
        for (int64_t b; (b = next.fetch_add(1)) < n_blocks;) {
            blk->load(sp, m, b * B);
            for (int64_t s0 = 0, k = 0; s0 < seq.n; s0 += every) {
                if (ckpt) blk->store(ckpt + sp.n_m() * sp.n * (s0 / every), b * B, sp.n);
                k = blk->run(seq, nc, acc[t].data(), s0, std::min(s0 + every, seq.n), k);
            }
            blk->store(m, b * B, sp.n);
        }
    });

    // reduce in thread order, then demodulate by exp(-i * adc_phase)
    std::fill(signal, signal + size, T(0));
    for (const auto& a : acc)
        for (int64_t j = 0; j < static_cast<int64_t>(a.size()); ++j) signal[j] += a[j];
    for (int64_t s = 0, k = 0; s < seq.n; ++s) {
        if (!(seq.flags[s] & kADC)) continue;
        const T c = std::cos(seq.adc_phase[s]), sn = std::sin(seq.adc_phase[s]);
        T* p = signal + 2 * nc * k++;
        for (int64_t j = 0; j < nc; ++j) {
            const T re = p[2 * j], im = p[2 * j + 1];
            p[2 * j] = re * c + im * sn;
            p[2 * j + 1] = im * c - re * sn;
        }
    }
}

// ---------------------------------------------------------------------------------------
// Adjoint (reverse mode).
//
// Per spin, segment s maps the magnetization affinely, m_s = A_s m_{s-1} + b_s, and each
// ADC sample is linear in m. With lambda_s = dL/dm_s (after segment s, spoil included),
// the reverse sweep is
//     lambda_{s-1} = A_s^T (P_s lambda_s + [ADC] dL/dm from the sample),
// where P_s zeroes the transverse part if s spoils. Parameter gradients of segment s
// follow from lambda_s and m_{s-1}. A_s itself:
//   free:  A = E(dt) Rz(phi), phi = -2 pi dt (pos.g + db0); E = diag(e2, e2, e1);
//   RF:    A = E(dt/2) R(w) E(dt/2), R = c I + s1 [w]x + s2 w w^T (Rodrigues), so
//          R^T = c I - s1 [w]x + s2 w w^T; w = -2 pi dt (b1 rf_x, b1 rf_y, pos.g + db0).
// With the field model, pos.g + db0 is the full offset (Block::offset); its partial
// derivatives come from Adjoint::field.
// m_{s-1} comes from checkpoints: the forward stores m before every `every` segments;
// the backward replays one interval at a time into a (every, 3, B) buffer and sweeps it
// in reverse. Relaxation and spoiling are never inverted. Work: one forward replay plus
// the sweep. Memory: the (S / every, N, 3) checkpoints, and per thread the replay buffer
// plus an (S, 6) double accumulator for the segment gradients (summed over spins, then
// over threads in thread order).

// Segment gradient slots, accumulated in double per thread.
enum { kGdt, kGrx, kGry, kGgx, kGgy, kGgz, kGoff, kGfrf, kNSeg };

// Per-spin gradient rows of Adjoint::total ahead of the sensitivities.
enum { kGpx, kGpy, kGpz, kGdb0, kGpd, kGr1, kGr2, kGb1, kGc0, kGbf = kGc0 + kConcMax, kGbk, kGbr1, kGbt2, kGbdw, kNSpin };

// Per-spin gradients are summed in T and moved into double after every segment whose
// index is a multiple of this: few enough terms for float32, and the same grouping for
// any checkpoint interval (so the interval does not change the result).
constexpr int64_t kFlush = 64;

// d/dphi of sin(phi)/phi and (1 - cos(phi))/phi^2, divided by phi:
//   f1 = (cos(phi) - sin(phi)/phi) / phi^2,  f2 = (sin(phi)/phi - 2 (1 - cos(phi))/phi^2) / phi^2.
// Both cancel catastrophically for small phi, so use their Taylor series there (to phi^8;
// below phi = 0.25 the first omitted term is < 1e-15 relative).
template <typename T>
inline void rotation_derivatives(T phi, T c, T s1, T s2, T& f1, T& f2) {
    const T p2 = phi * phi;
    if (p2 < T(0.0625)) {
        f1 = T(-1.0 / 3) + p2 * (T(1.0 / 30) + p2 * (T(-1.0 / 840) + p2 * (T(1.0 / 45360) + p2 * T(-1.0 / 3991680))));
        f2 = T(-1.0 / 12) + p2 * (T(1.0 / 180) + p2 * (T(-1.0 / 6720) + p2 * (T(1.0 / 453600) + p2 * T(-1.0 / 47900160))));
    } else {
        f1 = (c - s1) / p2;
        f2 = (s1 - T(2) * s2) / p2;
    }
}

template <typename T, int B>
struct Adjoint {
    Block<T, B> fw;  // spins, magnetization and forward factor caches, for the replay
    alignas(64) T lx[B], ly[B], lz[B], lb[B];  // lambda (lb: the bound pool's Mzb)
    // Per-spin gradients, summed in T over at most kFlush segments and then into `total`.
    alignas(64) T gpx[B], gpy[B], gpz[B], gdb0[B], gpd[B], gr1[B], gr2[B], gb1[B];
    alignas(64) T gcf[kConcMax][B];  // concomitant-field coefficients
    alignas(64) T gbf[B], gbk[B], gbr1[B], gbt2[B], gbdw[B];  // bound pool: f, k, 1/t1b, t2b, dw
    std::vector<double> total;  // (kNSpin + 2C, B): gpx, ..., gr2, gb1, gcf, gb*, d/d sens (re), (im)
    std::vector<T> buf;         // (every, nm, B): m before each segment of the current interval
    // Sensitivity gradients are sums over ADC samples of h (x) m_transverse; samples are
    // collected here and summed in register tiles by flush().
    std::vector<T> obuf;        // (kFlush, 2, B): (mx, my) at each ADC sample since the last flush
    std::vector<const T*> hbuf; // (kFlush,): its dL/d(raw coil sums)
    int64_t n_obuf = 0;
    // RF rotation: c, s1, s2, wz (as Block::rot) plus f1, f2. Key: dt, rf, grad, b0_offset.
    Cache<T, B, 7, 6, 4> rot;

    Adjoint(int64_t n_coils, int64_t every)
        : fw(n_coils), total((kNSpin + 2 * n_coils) * B), buf(4 * B * every), obuf(2 * B * kFlush), hbuf(kFlush) {}

    void load(const Spins<T>& sp, const T* g_m, int64_t start) {
        rot.clear();
        const int nm = sp.n_m();
        for (int l = 0; l < B; ++l) {
            const int64_t i = start + l;
            const bool in = g_m && i < sp.n;
            lx[l] = in ? g_m[nm * i] : T(0);
            ly[l] = in ? g_m[nm * i + 1] : T(0);
            lz[l] = in ? g_m[nm * i + 2] : T(0);
            lb[l] = in && nm == 4 ? g_m[nm * i + 3] : T(0);
            gpx[l] = gpy[l] = gpz[l] = gdb0[l] = gpd[l] = gr1[l] = gr2[l] = gb1[l] = T(0);
            gbf[l] = gbk[l] = gbr1[l] = gbt2[l] = gbdw[l] = T(0);
            for (int q = 0; q < kConcMax; ++q) gcf[q][l] = T(0);
        }
        std::fill(total.begin(), total.end(), 0.0);
        n_obuf = 0;
    }

    // Sensitivity gradients of coils [c0, c0 + NQ) from the collected samples:
    // d/d(sr) += hr mx + hi my, d/d(si) += hi mx - hr my.
    template <int NQ>
    void coil_tile(int64_t c0, int64_t n_coils) {
        constexpr int V = std::min<int>(B, 64 / sizeof(T));  // lanes per register tile
        for (int i0 = 0; i0 < B; i0 += V) {
            T ar[NQ][V] = {}, ai[NQ][V] = {};
            for (int64_t k = 0; k < n_obuf; ++k) {
                const T *x = &obuf[2 * B * k + i0], *y = x + B, *h = hbuf[k] + 2 * c0;
                for (int q = 0; q < NQ; ++q) {
                    const T hr = h[2 * q], hi = h[2 * q + 1];
#pragma omp simd
                    for (int v = 0; v < V; ++v) {
                        ar[q][v] += hr * x[v] + hi * y[v];
                        ai[q][v] += hi * x[v] - hr * y[v];
                    }
                }
            }
            for (int q = 0; q < NQ; ++q)
                for (int v = 0; v < V; ++v) {
                    total[(kNSpin + c0 + q) * B + i0 + v] += ar[q][v];
                    total[(kNSpin + n_coils + c0 + q) * B + i0 + v] += ai[q][v];
                }
        }
    }

    // Move the per-spin sums into `total`.
    void flush(int64_t n_coils) {
        T* part[kNSpin] = {gpx, gpy, gpz, gdb0, gpd, gr1, gr2, gb1};
        for (int q = 0; q < kConcMax; ++q) part[kGc0 + q] = gcf[q];
        part[kGbf] = gbf, part[kGbk] = gbk, part[kGbr1] = gbr1, part[kGbt2] = gbt2, part[kGbdw] = gbdw;
        for (int q = 0; q < kNSpin; ++q)
            for (int i = 0; i < B; ++i) total[q * B + i] += part[q][i], part[q][i] = T(0);
        if (n_obuf) {
            int64_t c = 0;
            for (; c + 4 <= n_coils; c += 4) coil_tile<4>(c, n_coils);
            for (; c < n_coils; ++c) coil_tile<1>(c, n_coils);
            n_obuf = 0;
        }
    }

    void store(const Spins<T>& sp, const Grads<T>& g, int64_t start) const {
        const int64_t nc = sp.n_coils;
        const double* t = total.data();
        for (int l = 0; l < B && start + l < sp.n; ++l) {
            const int64_t i = start + l;
            const double r1 = fw.r1[l], r2 = fw.r2[l];
            if (g.pos)
                g.pos[3 * i] = T(t[kGpx * B + l]), g.pos[3 * i + 1] = T(t[kGpy * B + l]),
                g.pos[3 * i + 2] = T(t[kGpz * B + l]);
            if (g.db0) g.db0[i] = T(t[kGdb0 * B + l]);
            if (g.pd) g.pd[i] = T(t[kGpd * B + l]);
            if (g.t1) g.t1[i] = T(-t[kGr1 * B + l] * r1 * r1);  // r1 = 1 / t1
            if (g.t2) g.t2[i] = T(-t[kGr2 * B + l] * r2 * r2);
            if (g.b1) g.b1[i] = T(t[kGb1 * B + l]);
            if (g.conc) {
                const int n_cf = fw.mode == kConcExact ? 7 : 6;
                for (int q = 0; q < n_cf; ++q) g.conc[n_cf * i + q] = T(t[(kGc0 + q) * B + l]);
            }
            if (g.sens)
                for (int64_t c = 0; c < nc; ++c) {
                    g.sens[2 * (i * nc + c)] = T(t[(kNSpin + c) * B + l]);
                    g.sens[2 * (i * nc + c) + 1] = T(t[(kNSpin + nc + c) * B + l]);
                }
            if (g.bound) {  // k f and the rates: f, k, t1b (r1b = 1 / t1b), t2b, dw
                T* q = g.bound + 5 * i;
                const double rb = fw.br1[l];
                q[0] = T(t[kGbf * B + l]), q[1] = T(t[kGbk * B + l]), q[2] = T(-t[kGbr1 * B + l] * rb * rb);
                q[3] = T(t[kGbt2 * B + l]), q[4] = T(t[kGbdw * B + l]);
            }
            const int nm = fw.nm;
            if (g.m) g.m[nm * i] = lx[l], g.m[nm * i + 1] = ly[l], g.m[nm * i + 2] = lz[l];
            if (g.m && nm == 4) g.m[nm * i + 3] = lb[l];
        }
    }

    // Replay segments [s0, s1) from fw's magnetization, keeping m before each in buf.
    void replay(const Segments<T>& seq, int64_t s0, int64_t s1) {
        for (int64_t s = s0; s < s1; ++s) {
            T* b = &buf[fw.nm * B * (s - s0)];
            std::copy(fw.mx, fw.mx + B, b);
            std::copy(fw.my, fw.my + B, b + B);
            std::copy(fw.mz, fw.mz + B, b + 2 * B);
            if (fw.bound) {
                std::copy(fw.mb, fw.mb + B, b + 3 * B);
                fw.advance_bound(seq, s);
            } else {
                fw.advance(seq, s);
            }
            if (seq.flags[s] & kSpoil) fw.spoil();
        }
    }

    // Slot for the transverse magnetization at an ADC sample with dL/d(raw coil sums) h.
    T* collect(const T* h) {
        hbuf[n_obuf] = h;
        return &obuf[2 * B * n_obuf++];
    }

    // ADC sample: lambda += dL/dm, with h = dL/d(raw coil sums) (C, 2):
    // dL/dmx = sum hr sr + hi si, dL/dmy = sum hi sr - hr si. Four coils per pass.
    void sample(const T* h, int64_t n_coils) {
        const T *sr = fw.sr.data(), *si = fw.si.data();
        int64_t c = 0;
        for (; c + 4 <= n_coils; c += 4) {
            const T *a0 = sr + c * B, *a1 = a0 + B, *a2 = a1 + B, *a3 = a2 + B;
            const T *b0 = si + c * B, *b1 = b0 + B, *b2 = b1 + B, *b3 = b2 + B;
            const T* q = h + 2 * c;
            const T hr0 = q[0], hi0 = q[1], hr1 = q[2], hi1 = q[3], hr2 = q[4], hi2 = q[5], hr3 = q[6], hi3 = q[7];
#pragma omp simd
            for (int i = 0; i < B; ++i) {
                lx[i] += hr0 * a0[i] + hi0 * b0[i] + hr1 * a1[i] + hi1 * b1[i] + hr2 * a2[i] + hi2 * b2[i] +
                         hr3 * a3[i] + hi3 * b3[i];
                ly[i] += hi0 * a0[i] - hr0 * b0[i] + hi1 * a1[i] - hr1 * b1[i] + hi2 * a2[i] - hr2 * b2[i] +
                         hi3 * a3[i] - hr3 * b3[i];
            }
        }
        for (; c < n_coils; ++c) {
            const T hr = h[2 * c], hi = h[2 * c + 1];
            const T *a = sr + c * B, *b = si + c * B;
#pragma omp simd
            for (int i = 0; i < B; ++i) {
                lx[i] += hr * a[i] + hi * b[i];
                ly[i] += hi * a[i] - hr * b[i];
            }
        }
    }

    // Apple clang cannot vectorize the RF-segment loop below (GCC does) and reports the
    // failed simd request at the function; harmless (one RF segment per TR), so keep
    // -Werror builds working.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif
    // Partial derivatives of lane i's offset (Block::offset) for gradient g and uniform
    // offset f: its value bz, d/d(lin) with lin = pos.g + db0 + f, and d/dg. Exact mode
    // also returns tx / S, ty / S and d/df0 = -t^2 / (S (S + u)), with S = |B|.
    template <int M>
    void field(int i, T gx, T gy, T gz, T f, T& bz, T& dlin, T& dgx, T& dgy, T& dgz, T& ex, T& ey,
               T& ef0) const {
        const T *px = fw.px, *py = fw.py, *pz = fw.pz;
        const T lin = px[i] * gx + py[i] * gy + pz[i] * gz + fw.db0[i] + f;
        const T(*c)[B] = fw.cf;
        if constexpr (M == kConcLeading) {
            bz = lin + (c[0][i] * (gx * gx) + c[1][i] * (gy * gy) + c[2][i] * (gz * gz) + c[3][i] * (gx * gy) +
                        c[4][i] * (gx * gz) + c[5][i] * (gy * gz));
            dlin = T(1);
            dgx = px[i] + T(2) * c[0][i] * gx + c[3][i] * gy + c[4][i] * gz;
            dgy = py[i] + T(2) * c[1][i] * gy + c[3][i] * gx + c[5][i] * gz;
            dgz = pz[i] + T(2) * c[2][i] * gz + c[4][i] * gx + c[5][i] * gy;
            ex = ey = ef0 = T(0);
        } else {
            const T tx = c[0][i] * gx + c[1][i] * gy + c[2][i] * gz;
            const T ty = c[3][i] * gx + c[4][i] * gy + c[5][i] * gz;
            const T t2 = tx * tx + ty * ty, u = c[6][i] + lin;
            const T S = std::sqrt(u * u + t2);
            bz = lin + t2 / (S + u);
            dlin = u / S, ex = tx / S, ey = ty / S;
            ef0 = -t2 / (S * (S + u));
            dgx = dlin * px[i] + ex * c[0][i] + ey * c[3][i];
            dgy = dlin * py[i] + ex * c[1][i] + ey * c[4][i];
            dgz = dlin * pz[i] + ex * c[2][i] + ey * c[5][i];
        }
    }

    // Lane i's gradient of the concomitant-field coefficients, given gb = dL/d(offset);
    // q: the gradient monomials (leading order).
    template <int M>
    void accumulate_field(int i, T gb, T gx, T gy, T gz, const T* q, T ex, T ey, T ef0) {
        if constexpr (M == kConcLeading) {
            for (int k = 0; k < 6; ++k) gcf[k][i] += gb * q[k];
        } else {
            const T a = gb * ex, b = gb * ey;
            gcf[0][i] += a * gx, gcf[1][i] += a * gy, gcf[2][i] += a * gz;
            gcf[3][i] += b * gx, gcf[4][i] += b * gy, gcf[5][i] += b * gz;
            gcf[6][i] += gb * ef0;
        }
    }

    // dL/d(offset) = gb of lane i into the per-spin gradients; returns the offset and
    // lane i's terms of the segment gradients (grad, b0_offset).
    template <int M>
    T offset_grads(int i, T gb, T gx, T gy, T gz, T f, const T* q, T& sgx, T& sgy, T& sgz, T& soff) {
        const T *px = fw.px, *py = fw.py, *pz = fw.pz;
        if constexpr (M == kConcNone) {
            gdb0[i] += gb;
            gpx[i] += gb * gx, gpy[i] += gb * gy, gpz[i] += gb * gz;
            sgx = gb * px[i], sgy = gb * py[i], sgz = gb * pz[i], soff = gb;
            return px[i] * gx + py[i] * gy + pz[i] * gz + fw.db0[i] + f;
        } else {
            T bz, dlin, dgx, dgy, dgz, ex, ey, ef0;
            field<M>(i, gx, gy, gz, f, bz, dlin, dgx, dgy, dgz, ex, ey, ef0);
            const T gl = gb * dlin;
            gdb0[i] += gl;
            gpx[i] += gl * gx, gpy[i] += gl * gy, gpz[i] += gl * gz;
            sgx = gb * dgx, sgy = gb * dgy, sgz = gb * dgz, soff = gl;
            accumulate_field<M>(i, gb, gx, gy, gz, q, ex, ey, ef0);
            return bz;
        }
    }

    // Reverse through segment s; m_{s-1} is at buf slot j, lambda holds dL/dm_s.
    // h: dL/d(raw coil sums) of this segment's ADC sample, or null.
    void reverse(const Segments<T>& seq, int64_t s, int64_t j, const T* h, int64_t n_coils, bool sens,
                 double* gseg) {
        if (fw.bound) {
            switch (fw.mode) {
                case kConcLeading: return reverse_bound<kConcLeading>(seq, s, j, h, n_coils, sens, gseg);
                case kConcExact: return reverse_bound<kConcExact>(seq, s, j, h, n_coils, sens, gseg);
                default: return reverse_bound<kConcNone>(seq, s, j, h, n_coils, sens, gseg);
            }
        }
        switch (fw.mode) {
            case kConcLeading: return reverse_mode<kConcLeading>(seq, s, j, h, n_coils, sens, gseg);
            case kConcExact: return reverse_mode<kConcExact>(seq, s, j, h, n_coils, sens, gseg);
            default: return reverse_mode<kConcNone>(seq, s, j, h, n_coils, sens, gseg);
        }
    }

    template <int M>
    void reverse_mode(const Segments<T>& seq, int64_t s, int64_t j, const T* h, int64_t n_coils, bool sens,
                      double* gseg) {
        const T w_per_s = T(-2.0 * kPi);
        const T* x0 = &buf[3 * B * j];
        const T *y0 = x0 + B, *z0 = x0 + 2 * B;
        const T *pd = fw.pd, *r1 = fw.r1, *r2 = fw.r2;
        const T* b1 = fw.b1;
        const T dt = seq.dt[s];
        const T gx = seq.grad[3 * s], gy = seq.grad[3 * s + 1], gz = seq.grad[3 * s + 2];
        const T f = seq.offset(s);
        const T q[6] = {gx * gx, gy * gy, gz * gz, gx * gy, gx * gz, gy * gz};
        const uint8_t fl = seq.flags[s];
        const T w = w_per_s * dt;
        if (fl & kSpoil)
            for (int i = 0; i < B; ++i) lx[i] = T(0), ly[i] = T(0);
        bool fill;
        T sgx = 0, sgy = 0, sgz = 0, soff = 0;
        // sums over lanes for the segment gradients
        T sdt = 0, swx = 0, swy = 0;
        if (fl & kRF) {
            const T h_dt = T(0.5) * dt;
            const T rx = seq.rf[2 * s], ry = seq.rf[2 * s + 1];
            const T wx0 = w * rx, wy0 = w * ry;  // times b1[i] per lane, as in Block::advance
            const T (*d)[B] = fw.relaxation(h_dt);
            T (*r)[B] = rot.find({dt, rx, ry, gx, gy, gz, f}, fill);
            if (fill)
                for (int i = 0; i < B; ++i) {
                    const T wx = wx0 * b1[i], wy = wy0 * b1[i];
                    fw.rotation(r, i, w, wx, wy, gx, gy, gz, f);
                    const T wz = r[3][i];
                    const T phi = std::sqrt(wx * wx + wy * wy + wz * wz + T(1e-30));
                    rotation_derivatives(phi, r[0][i], r[1][i], r[2][i], r[4][i], r[5][i]);
                }
            // forward intermediates: m1 = E m0, n = R m1, out = E n
            auto fwd = [&](int i, T& m1x, T& m1y, T& m1z, T& dot, T& nx, T& ny, T& nz) {
                const T wx = wx0 * b1[i], wy = wy0 * b1[i];
                m1x = x0[i] + x0[i] * d[1][i], m1y = y0[i] + y0[i] * d[1][i];
                m1z = z0[i] + (z0[i] - pd[i]) * d[0][i];
                const T c = r[0][i], s1 = r[1][i], s2 = r[2][i], wz = r[3][i];
                dot = wx * m1x + wy * m1y + wz * m1z;
                nx = c * m1x + s1 * (wy * m1z - wz * m1y) + s2 * wx * dot;
                ny = c * m1y + s1 * (wz * m1x - wx * m1z) + s2 * wy * dot;
                nz = c * m1z + s1 * (wx * m1y - wy * m1x) + s2 * wz * dot;
            };
            if (h) {
                if (sens) {
                    T* o = collect(h);
                    for (int i = 0; i < B; ++i) {
                        T m1x, m1y, m1z, dot, nx, ny, nz;
                        fwd(i, m1x, m1y, m1z, dot, nx, ny, nz);
                        o[i] = nx + nx * d[1][i], o[B + i] = ny + ny * d[1][i];
                    }
                }
                sample(h, n_coils);
            }
#pragma omp simd reduction(+ : sdt, sgx, sgy, sgz, soff, swx, swy)
            for (int i = 0; i < B; ++i) {
                T m1x, m1y, m1z, dot, nx, ny, nz;
                fwd(i, m1x, m1y, m1z, dot, nx, ny, nz);
                const T wx = wx0 * b1[i], wy = wy0 * b1[i];
                const T c = r[0][i], s1 = r[1][i], s2 = r[2][i], wz = r[3][i], f1 = r[4][i], f2 = r[5][i];
                const T e1 = T(1) + d[0][i], e2 = T(1) + d[1][i];
                // second half relaxation
                T gd2 = lx[i] * nx + ly[i] * ny, gd1 = lz[i] * (nz - pd[i]);
                T gpd_ = -lz[i] * d[0][i];
                // dL/dn (lambda + lambda * d, not (1 + d) * lambda: see the note on expm1 above)
                const T ax = lx[i] + lx[i] * d[1][i], ay = ly[i] + ly[i] * d[1][i], az = lz[i] + lz[i] * d[0][i];
                // rotation: dL/dm1 = R^T a, dL/dw
                const T wa = wx * ax + wy * ay + wz * az;
                const T bx = c * ax - s1 * (wy * az - wz * ay) + s2 * wx * wa;
                const T by = c * ay - s1 * (wz * ax - wx * az) + s2 * wy * wa;
                const T bz = c * az - s1 * (wx * ay - wy * ax) + s2 * wz * wa;
                const T am = ax * m1x + ay * m1y + az * m1z;
                const T awm = ax * (wy * m1z - wz * m1y) + ay * (wz * m1x - wx * m1z) + az * (wx * m1y - wy * m1x);
                const T k = -s1 * am + f1 * awm + f2 * wa * dot;
                const T gwx = k * wx + s1 * (m1y * az - m1z * ay) + s2 * (wa * m1x + dot * ax);
                const T gwy = k * wy + s1 * (m1z * ax - m1x * az) + s2 * (wa * m1y + dot * ay);
                const T gwz = k * wz + s1 * (m1x * ay - m1y * ax) + s2 * (wa * m1z + dot * az);
                // first half relaxation
                gd2 += bx * x0[i] + by * y0[i];
                gd1 += bz * (z0[i] - pd[i]);
                gpd_ -= bz * d[0][i];
                lx[i] = bx + bx * d[1][i], ly[i] = by + by * d[1][i], lz[i] = bz + bz * d[0][i];
                // d = expm1(-h_dt * r): dd/dh_dt = -r e, dd/dr = -h_dt e
                gr1[i] -= gd1 * h_dt * e1;
                gr2[i] -= gd2 * h_dt * e2;
                gpd[i] += gpd_;
                // wz = w * offset
                T tgx, tgy, tgz, toff;
                const T bz0 = offset_grads<M>(i, gwz * w, gx, gy, gz, f, q, tgx, tgy, tgz, toff);
                sgx += tgx, sgy += tgy, sgz += tgz, soff += toff;
                sdt += w_per_s * gwz * bz0 - T(0.5) * (gd1 * r1[i] * e1 + gd2 * r2[i] * e2);
                // (wx, wy) = b1 (wx0, wy0): the sequence sees dL/dw times b1, the spin dL/dw . w0
                swx += gwx * b1[i], swy += gwy * b1[i];
                gb1[i] += gwx * wx0 + gwy * wy0;
            }
            if (gseg) {
                gseg[kGdt] += double(sdt) + double(w_per_s) * (double(rx) * swx + double(ry) * swy);
                gseg[kGrx] += double(w) * swx;
                gseg[kGry] += double(w) * swy;
            }
        } else {
            const T (*d)[B] = fw.relaxation(dt);
            T (*p)[B] = fw.precession(dt, w, gx, gy, gz, f);
            if (h) {
                if (sens) {
                    T* o = collect(h);
                    for (int i = 0; i < B; ++i) {
                        const T ux = p[0][i] * x0[i] - p[1][i] * y0[i], uy = p[1][i] * x0[i] + p[0][i] * y0[i];
                        o[i] = ux + ux * d[1][i], o[B + i] = uy + uy * d[1][i];
                    }
                }
                sample(h, n_coils);
            }
#pragma omp simd reduction(+ : sdt, sgx, sgy, sgz, soff)
            for (int i = 0; i < B; ++i) {
                const T c = p[0][i], sn = p[1][i];
                const T e1 = T(1) + d[0][i], e2 = T(1) + d[1][i];
                const T ux = c * x0[i] - sn * y0[i], uy = sn * x0[i] + c * y0[i];
                const T gd2 = lx[i] * ux + ly[i] * uy, gd1 = lz[i] * (z0[i] - pd[i]);
                const T gphi = e2 * (ly[i] * ux - lx[i] * uy);
                gpd[i] -= lz[i] * d[0][i];
                const T ax = c * lx[i] + sn * ly[i], ay = c * ly[i] - sn * lx[i];
                lx[i] = ax + ax * d[1][i], ly[i] = ay + ay * d[1][i], lz[i] = lz[i] + lz[i] * d[0][i];
                gr1[i] -= gd1 * dt * e1;
                gr2[i] -= gd2 * dt * e2;
                // phi = w * offset
                T tgx, tgy, tgz, toff;
                const T bz0 = offset_grads<M>(i, gphi * w, gx, gy, gz, f, q, tgx, tgy, tgz, toff);
                sgx += tgx, sgy += tgy, sgz += tgz, soff += toff;
                sdt += w_per_s * gphi * bz0 - (gd1 * r1[i] * e1 + gd2 * r2[i] * e2);
            }
            if (gseg) gseg[kGdt] += double(sdt);
        }
        if (gseg) {
            gseg[kGgx] += double(sgx), gseg[kGgy] += double(sgy), gseg[kGgz] += double(sgz);
            gseg[kGoff] += double(soff);
        }
    }

    // dL/d(A, c, t) of lane i's exchange step into the pool parameters, given its saturation
    // rate's share yw = dL/dW (returned) and the step's time derivative into *yt.
    T exchange_grads(int i, T y11, T y12, T y21, T y22, T yc1, T yc2) {
        const T f = fw.bf[i], k = fw.bk[i], rb = fw.br1[i], pd = fw.pd[i];
        // a11 = -(r1 + k f), a12 = k, a21 = k f, a22 = -(r1b + k + W), c1 = r1 pd, c2 = r1b f pd
        gr1[i] += -y11 + pd * yc1;
        gpd[i] += fw.r1[i] * yc1 + rb * f * yc2;
        gbf[i] += k * (y21 - y11) + rb * pd * yc2;
        gbk[i] += f * (y21 - y11) + y12 - y22;
        gbr1[i] += -y22 + f * pd * yc2;
        return -y22;
    }

    // reverse_mode() with a bound pool: the longitudinal relaxation of advance_bound() is the
    // exchange step (ZStep::reverse); during RF its saturation rate W = w1^2 G(f_rf - offset - dw)
    // passes gradients to the RF amplitude, b1, t2b, dw, the RF frequency and the offset.
    template <int M>
    void reverse_bound(const Segments<T>& seq, int64_t s, int64_t j, const T* h, int64_t n_coils, bool sens,
                       double* gseg) {
        const T w_per_s = T(-2.0 * kPi), two_pi = T(2.0 * kPi);
        const T* x0 = &buf[4 * B * j];
        const T *y0 = x0 + B, *z0 = x0 + 2 * B, *v0 = x0 + 3 * B;
        const T *r2 = fw.r2, *b1 = fw.b1;
        const T dt = seq.dt[s];
        const T gx = seq.grad[3 * s], gy = seq.grad[3 * s + 1], gz = seq.grad[3 * s + 2];
        const T f = seq.offset(s);
        const T q[6] = {gx * gx, gy * gy, gz * gz, gx * gy, gx * gz, gy * gz};
        const uint8_t fl = seq.flags[s];
        const T w = w_per_s * dt;
        if (fl & kSpoil)
            for (int i = 0; i < B; ++i) lx[i] = T(0), ly[i] = T(0);
        bool fill;
        T sgx = 0, sgy = 0, sgz = 0, soff = 0, sdt = 0, swx = 0, swy = 0, srx = 0, sry = 0, sfrf = 0;
        if (fl & kRF) {
            const T h_dt = T(0.5) * dt;
            const T rx = seq.rf[2 * s], ry = seq.rf[2 * s + 1], a2 = rx * rx + ry * ry, f_rf = seq.rf_freq[s];
            const T wx0 = w * rx, wy0 = w * ry;
            const T (*d)[B] = fw.relaxation(h_dt);
            T (*r)[B] = rot.find({dt, rx, ry, gx, gy, gz, f}, fill);
            if (fill)
                for (int i = 0; i < B; ++i) {
                    const T wx = wx0 * b1[i], wy = wy0 * b1[i];
                    fw.rotation(r, i, w, wx, wy, gx, gy, gz, f);
                    const T wz = r[3][i];
                    const T phi = std::sqrt(wx * wx + wy * wy + wz * wz + T(1e-30));
                    rotation_derivatives(phi, r[0][i], r[1][i], r[2][i], r[4][i], r[5][i]);
                }
            // lane i's forward intermediates: m1 = (E2 x0, E2 y0, Z(z0, v0)), n = R m1,
            // out = (E2 n_xy, Z(nz, v1)), with the saturated exchange step Z
            struct Fwd {
                ZStep<T> zs;
                T G, g_t2, g_d, kw, m1x, m1y, m1z, v1, dot, nx, ny, nz;
            };
            auto fwd = [&](int i) {
                Fwd o;
                const T delta = f_rf - fw.offset(i, gx, gy, gz, f) - fw.bdw[i];
                o.G = fw.line.eval(delta, fw.bt2[i], o.g_t2, o.g_d);
                o.kw = two_pi * two_pi * (b1[i] * b1[i]) * a2;  // W = kw G
                o.zs = fw.zstep(i, h_dt, o.kw * o.G);
                const T wx = wx0 * b1[i], wy = wy0 * b1[i];
                o.m1x = x0[i] + x0[i] * d[1][i], o.m1y = y0[i] + y0[i] * d[1][i];
                o.m1z = z0[i], o.v1 = v0[i];
                o.zs.apply(o.m1z, o.v1);
                const T c = r[0][i], s1 = r[1][i], s2 = r[2][i], wz = r[3][i];
                o.dot = wx * o.m1x + wy * o.m1y + wz * o.m1z;
                o.nx = c * o.m1x + s1 * (wy * o.m1z - wz * o.m1y) + s2 * wx * o.dot;
                o.ny = c * o.m1y + s1 * (wz * o.m1x - wx * o.m1z) + s2 * wy * o.dot;
                o.nz = c * o.m1z + s1 * (wx * o.m1y - wy * o.m1x) + s2 * wz * o.dot;
                return o;
            };
            if (h) {
                if (sens) {
                    T* o = collect(h);
                    for (int i = 0; i < B; ++i) {
                        const Fwd v = fwd(i);
                        o[i] = v.nx + v.nx * d[1][i], o[B + i] = v.ny + v.ny * d[1][i];
                    }
                }
                sample(h, n_coils);
            }
            for (int i = 0; i < B; ++i) {
                const Fwd v = fwd(i);
                const ZStep<T>& zs = v.zs;
                const T G = v.G, g_t2 = v.g_t2, g_d = v.g_d, kw = v.kw;
                const T m1x = v.m1x, m1y = v.m1y, m1z = v.m1z, v1 = v.v1, dot = v.dot, nx = v.nx, ny = v.ny, nz = v.nz;
                const T wx = wx0 * b1[i], wy = wy0 * b1[i];
                const T c = r[0][i], s1 = r[1][i], s2 = r[2][i], wz = r[3][i], f1 = r[4][i], f2 = r[5][i];
                const T e2 = T(1) + d[1][i];
                // second half: transverse relaxation and the exchange step from (nz, v1)
                T gd2 = lx[i] * nx + ly[i] * ny;
                const T ax = lx[i] + lx[i] * d[1][i], ay = ly[i] + ly[i] * d[1][i];
                T az = lz[i], lv = lb[i];
                T yt, y11, y12, y21, y22, yc1, yc2;
                zs.reverse(h_dt, nz, v1, az, lv, yt, y11, y12, y21, y22, yc1, yc2);
                T yt_sum = yt;
                T ya[6] = {y11, y12, y21, y22, yc1, yc2};
                // rotation: dL/dm1 = R^T a, dL/dw
                const T wa = wx * ax + wy * ay + wz * az;
                const T bx = c * ax - s1 * (wy * az - wz * ay) + s2 * wx * wa;
                const T by = c * ay - s1 * (wz * ax - wx * az) + s2 * wy * wa;
                T bzl = c * az - s1 * (wx * ay - wy * ax) + s2 * wz * wa;
                const T am = ax * m1x + ay * m1y + az * m1z;
                const T awm = ax * (wy * m1z - wz * m1y) + ay * (wz * m1x - wx * m1z) + az * (wx * m1y - wy * m1x);
                const T kk = -s1 * am + f1 * awm + f2 * wa * dot;
                const T gwx = kk * wx + s1 * (m1y * az - m1z * ay) + s2 * (wa * m1x + dot * ax);
                const T gwy = kk * wy + s1 * (m1z * ax - m1x * az) + s2 * (wa * m1y + dot * ay);
                const T gwz = kk * wz + s1 * (m1x * ay - m1y * ax) + s2 * (wa * m1z + dot * az);
                // first half: transverse relaxation and the exchange step from (z0, v0)
                gd2 += bx * x0[i] + by * y0[i];
                lx[i] = bx + bx * d[1][i], ly[i] = by + by * d[1][i];
                zs.reverse(h_dt, z0[i], v0[i], bzl, lv, yt, y11, y12, y21, y22, yc1, yc2);
                lz[i] = bzl, lb[i] = lv;
                yt_sum += yt;
                ya[0] += y11, ya[1] += y12, ya[2] += y21, ya[3] += y22, ya[4] += yc1, ya[5] += yc2;
                const T yw = exchange_grads(i, ya[0], ya[1], ya[2], ya[3], ya[4], ya[5]);
                // W = kw G(delta, t2b), kw = (2 pi)^2 b1^2 |rf|^2
                const T yd = yw * kw * g_d;  // dL/d(delta), delta = f_rf - offset - dw
                gbt2[i] += yw * kw * g_t2;
                gbdw[i] -= yd;
                sfrf += yd;
                const T ykw = yw * G * two_pi * two_pi;
                gb1[i] += ykw * T(2) * b1[i] * a2 + gwx * wx0 + gwy * wy0;
                srx += ykw * (b1[i] * b1[i]) * T(2) * rx, sry += ykw * (b1[i] * b1[i]) * T(2) * ry;
                gr2[i] -= gd2 * h_dt * e2;
                // offset: wz = w * offset and delta = ... - offset
                T tgx, tgy, tgz, toff;
                const T bz0 = offset_grads<M>(i, gwz * w - yd, gx, gy, gz, f, q, tgx, tgy, tgz, toff);
                sgx += tgx, sgy += tgy, sgz += tgz, soff += toff;
                sdt += w_per_s * gwz * bz0 - T(0.5) * gd2 * r2[i] * e2 + T(0.5) * yt_sum;
                swx += gwx * b1[i], swy += gwy * b1[i];
            }
            if (gseg) {
                gseg[kGdt] += double(sdt) + double(w_per_s) * (double(rx) * swx + double(ry) * swy);
                gseg[kGrx] += double(w) * swx + double(srx);
                gseg[kGry] += double(w) * swy + double(sry);
                gseg[kGfrf] += double(sfrf);
            }
        } else {
            const T (*d)[B] = fw.relaxation(dt);
            T (*p)[B] = fw.precession(dt, w, gx, gy, gz, f);
            if (h) {
                if (sens) {
                    T* o = collect(h);
                    for (int i = 0; i < B; ++i) {
                        const T ux = p[0][i] * x0[i] - p[1][i] * y0[i], uy = p[1][i] * x0[i] + p[0][i] * y0[i];
                        o[i] = ux + ux * d[1][i], o[B + i] = uy + uy * d[1][i];
                    }
                }
                sample(h, n_coils);
            }
            for (int i = 0; i < B; ++i) {
                const T c = p[0][i], sn = p[1][i];
                const T e2 = T(1) + d[1][i];
                const T ux = c * x0[i] - sn * y0[i], uy = sn * x0[i] + c * y0[i];
                const T gd2 = lx[i] * ux + ly[i] * uy;
                const T gphi = e2 * (ly[i] * ux - lx[i] * uy);
                const T ax = c * lx[i] + sn * ly[i], ay = c * ly[i] - sn * lx[i];
                lx[i] = ax + ax * d[1][i], ly[i] = ay + ay * d[1][i];
                gr2[i] -= gd2 * dt * e2;
                T yt, y11, y12, y21, y22, yc1, yc2;
                fw.zstep(i, dt, T(0)).reverse(dt, z0[i], v0[i], lz[i], lb[i], yt, y11, y12, y21, y22, yc1, yc2);
                exchange_grads(i, y11, y12, y21, y22, yc1, yc2);
                T tgx, tgy, tgz, toff;
                const T bz0 = offset_grads<M>(i, gphi * w, gx, gy, gz, f, q, tgx, tgy, tgz, toff);
                sgx += tgx, sgy += tgy, sgz += tgz, soff += toff;
                sdt += w_per_s * gphi * bz0 - gd2 * r2[i] * e2 + yt;
            }
            if (gseg) gseg[kGdt] += double(sdt);
        }
        if (gseg) {
            gseg[kGgx] += double(sgx), gseg[kGgy] += double(sgy), gseg[kGgz] += double(sgz);
            gseg[kGoff] += double(soff);
        }
    }
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
};

template <typename T, int B>
void adjoint_blocks(const Spins<T>& sp, const Segments<T>& seq, const T* ckpt, int64_t every, const T* signal,
                    const T* g_signal, const T* g_m, const Grads<T>& g, int n_threads) {
    const int64_t nc = sp.n_coils, ns = seq.n;
    const int64_t n_adc = count_adc(seq.flags, ns);
    const int64_t n_blocks = (sp.n + B - 1) / B;
    const int64_t n_ckpt = count_checkpoints(ns, every);
    n_threads = clamp_threads(n_threads, n_blocks);
    const bool need_seg = g.dt || g.rf || g.grad || g.b0_offset || g.rf_freq;

    // dL/d(raw coil sums) = conj-free rotation of dL/dsignal by exp(+i * adc_phase); ADC sample
    // number at every checkpoint
    std::vector<T> h;
    std::vector<int64_t> k_at(n_ckpt + 1, 0);
    if (g_signal) h.resize(n_adc * nc * 2);
    for (int64_t s = 0, k = 0; s < ns; ++s) {
        if (s % every == 0) k_at[s / every] = k;
        if (!(seq.flags[s] & kADC)) continue;
        if (g_signal) {
            const T c = std::cos(seq.adc_phase[s]), sn = std::sin(seq.adc_phase[s]);
            for (int64_t j = 0; j < nc; ++j) {
                const T gr = g_signal[2 * (k * nc + j)], gi = g_signal[2 * (k * nc + j) + 1];
                h[2 * (k * nc + j)] = gr * c - gi * sn;
                h[2 * (k * nc + j) + 1] = gr * sn + gi * c;
            }
        }
        ++k;
    }
    if (n_ckpt) k_at[n_ckpt] = n_adc;

    std::vector<std::vector<double>> gseg(n_threads);
    std::atomic<int64_t> next{0};
    parallel(n_threads, [&](int t) {
        if (need_seg) gseg[t].assign(kNSeg * ns, 0.0);
        auto blk = std::make_unique<Adjoint<T, B>>(nc, std::min<int64_t>(every, std::max<int64_t>(ns, 1)));
        for (int64_t b; (b = next.fetch_add(1)) < n_blocks;) {
            blk->fw.load(sp, ckpt, b * B);  // magnetization: reloaded per interval below
            blk->load(sp, g_m, b * B);
            for (int64_t j = n_ckpt - 1; j >= 0; --j) {
                const int64_t s0 = j * every, s1 = std::min(s0 + every, ns);
                blk->fw.load_m(ckpt + sp.n_m() * sp.n * j, b * B, sp.n);
                blk->replay(seq, s0, s1);
                int64_t k = k_at[j + 1];
                for (int64_t s = s1 - 1; s >= s0; --s) {
                    const bool adc = (seq.flags[s] & kADC) != 0;
                    k -= adc;
                    const T* hk = adc && g_signal ? &h[2 * nc * k] : nullptr;
                    blk->reverse(seq, s, s - s0, hk, nc, g.sens != nullptr,
                                 need_seg ? &gseg[t][kNSeg * s] : nullptr);
                    if (s % kFlush == 0) blk->flush(nc);
                }
            }
            blk->store(sp, g, b * B);
        }
    });

    // segment gradients: reduce in thread order
    if (need_seg)
        for (int64_t s = 0; s < ns; ++s) {
            double v[kNSeg] = {};
            for (const auto& a : gseg)
                for (int q = 0; q < kNSeg; ++q) v[q] += a[kNSeg * s + q];
            if (g.dt) g.dt[s] = T(v[kGdt]);
            if (g.rf) g.rf[2 * s] = T(v[kGrx]), g.rf[2 * s + 1] = T(v[kGry]);
            if (g.grad) g.grad[3 * s] = T(v[kGgx]), g.grad[3 * s + 1] = T(v[kGgy]), g.grad[3 * s + 2] = T(v[kGgz]);
            if (g.b0_offset) g.b0_offset[s] = T(v[kGoff]);
            if (g.rf_freq) g.rf_freq[s] = T(v[kGfrf]);
        }
    // receiver phase: signal = raw * exp(-i psi), so dsignal/dpsi = -i signal
    if (g.adc_phase)
        for (int64_t s = 0, k = 0; s < ns; ++s) {
            T v = 0;
            if (seq.flags[s] & kADC) {
                for (int64_t j = 0; g_signal && j < nc; ++j) {
                    const int64_t q = 2 * (k * nc + j);
                    v += g_signal[q] * signal[q + 1] - g_signal[q + 1] * signal[q];
                }
                ++k;
            }
            g.adc_phase[s] = v;
        }
}

}  // namespace detail

constexpr int kBlockSizes[] = {16, 32, 64, 128, 256};

// Run the sequence on the spins.
//   m:      (n, 3) magnetization, read as the start state and overwritten with the final one
//   signal: (count_adc(seq), C, 2) output, (re, im) per ADC sample and coil
//   ckpt:   optional (count_checkpoints(seq.n, every), n, 3) output for adjoint(): the
//           magnetization before segments 0, every, 2 * every, ...
//   n_threads <= 0 uses every hardware thread; block must be one of kBlockSizes.
template <typename T>
void simulate(const Spins<T>& spins, const Segments<T>& seq, T* m, T* signal, int n_threads = 0, int block = 64,
              T* ckpt = nullptr, int64_t every = 0) {
    if (ckpt && every < 1) throw std::invalid_argument("checkpoint interval must be positive");
    switch (block) {
        case 16: return detail::simulate_blocks<T, 16>(spins, seq, m, signal, n_threads, ckpt, every);
        case 32: return detail::simulate_blocks<T, 32>(spins, seq, m, signal, n_threads, ckpt, every);
        case 64: return detail::simulate_blocks<T, 64>(spins, seq, m, signal, n_threads, ckpt, every);
        case 128: return detail::simulate_blocks<T, 128>(spins, seq, m, signal, n_threads, ckpt, every);
        case 256: return detail::simulate_blocks<T, 256>(spins, seq, m, signal, n_threads, ckpt, every);
        default: throw std::invalid_argument("block must be 16, 32, 64, 128 or 256");
    }
}

// Gradients of a loss L through simulate(), given dL/dsignal and dL/dm (final).
//   ckpt, every: as filled by simulate() on the same inputs
//   signal:      simulate()'s signal output (for the receiver-phase gradient)
//   g_signal:    (count_adc(seq), C, 2) dL/d(re, im), or null for zero
//   g_m:         (n, 3) dL/dm of the final magnetization, or null for zero
//   g:           outputs (see Grads); null members are skipped
// Any block size works (it need not match the forward's).
template <typename T>
void adjoint(const Spins<T>& spins, const Segments<T>& seq, const T* ckpt, int64_t every, const T* signal,
             const T* g_signal, const T* g_m, const Grads<T>& g, int n_threads = 0, int block = 64) {
    if (every < 1) throw std::invalid_argument("checkpoint interval must be positive");
    switch (block) {
        case 16: return detail::adjoint_blocks<T, 16>(spins, seq, ckpt, every, signal, g_signal, g_m, g, n_threads);
        case 32: return detail::adjoint_blocks<T, 32>(spins, seq, ckpt, every, signal, g_signal, g_m, g, n_threads);
        case 64: return detail::adjoint_blocks<T, 64>(spins, seq, ckpt, every, signal, g_signal, g_m, g, n_threads);
        case 128: return detail::adjoint_blocks<T, 128>(spins, seq, ckpt, every, signal, g_signal, g_m, g, n_threads);
        case 256: return detail::adjoint_blocks<T, 256>(spins, seq, ckpt, every, signal, g_signal, g_m, g, n_threads);
        default: throw std::invalid_argument("block must be 16, 32, 64, 128 or 256");
    }
}

}  // namespace blochparty::fused
