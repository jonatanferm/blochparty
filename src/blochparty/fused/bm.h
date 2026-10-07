// Exact Bloch-McConnell engine for free exchanging pools (exchange.py: ExchangePool, Phantom.pools),
// optionally with the rate-model bound pool: the math of bloch._simulate_exchange, per spin. Every
// segment is one matrix exponential of the augmented system (water and pools (x, y, z), Mzb, 1), as
// the PyTorch core and BMCTool; consecutive identical segments reuse it. The adjoint takes, per
// segment, dL/dM for M = A dt from the upper-right block of exp([[M^T, lambda x0^T], [0, M^T]])
// (the Frechet derivative of the exponential) and chains it to every parameter through the
// entries of A. Spins are independent: one thread per spin at a time, a shared counter.
//
// Cost: O(d^3) per spin and distinct segment (d = 3 (P + 1) + 2 with a bound pool), 8x that in the
// adjoint: meant for spectra and few-spin experiments, not large images.

#pragma once

#include "kernel.h"

namespace blochparty::fused::bm {

template <typename T>
struct Pools {
    int64_t n, nc;
    int np;
    const T *pos, *pd, *t1, *t2, *db0, *b1, *sens;  // as Spins; b1 may be null
    const T* pools;  // (n, np, 5): f, k [Hz] (pool -> water), t1, t2 [s], dw [Hz]
    const T* bound;  // (n, 5) rate-model bound pool (Spins::bound), or null
    Lineshape<T> line;
    const T* conc;   // (n, 6 or 7) field model, or null
    int conc_mode;

    int dim() const { return 3 * (np + 1) + (bound ? 1 : 0) + 1; }  // augmented: the last entry is 1
};

template <typename T>
struct Grads {
    T *dt = nullptr, *rf = nullptr, *grad = nullptr, *adc_phase = nullptr, *b0_offset = nullptr, *rf_freq = nullptr;
    T *pos = nullptr, *pd = nullptr, *t1 = nullptr, *t2 = nullptr, *db0 = nullptr, *b1 = nullptr, *sens = nullptr;
    T *pools = nullptr, *bound = nullptr, *conc = nullptr, *m = nullptr;  // m: (n, d - 1)
};

namespace detail {

constexpr double kPi = 3.14159265358979323846;

// C = A B for d x d row-major matrices.
template <typename T>
inline void matmul(int d, const T* A, const T* B, T* C) {
    std::fill(C, C + d * d, T(0));
    for (int i = 0; i < d; ++i)
        for (int k = 0; k < d; ++k) {
            const T a = A[i * d + k];
            if (a == T(0)) continue;
            for (int j = 0; j < d; ++j) C[i * d + j] += a * B[k * d + j];
        }
}

// Solve P X = Q (d x d, row-major) in place into Q by LU with partial pivoting; P is destroyed.
template <typename T>
inline void solve(int d, T* P, T* Q) {
    for (int c = 0; c < d; ++c) {
        int piv = c;
        for (int r = c + 1; r < d; ++r)
            if (std::abs(P[r * d + c]) > std::abs(P[piv * d + c])) piv = r;
        if (piv != c)
            for (int j = 0; j < d; ++j) std::swap(P[c * d + j], P[piv * d + j]), std::swap(Q[c * d + j], Q[piv * d + j]);
        for (int r = c + 1; r < d; ++r) {
            const T f = P[r * d + c] / P[c * d + c];
            if (f == T(0)) continue;
            for (int j = c; j < d; ++j) P[r * d + j] -= f * P[c * d + j];
            for (int j = 0; j < d; ++j) Q[r * d + j] -= f * Q[c * d + j];
        }
    }
    for (int c = d - 1; c >= 0; --c)
        for (int j = 0; j < d; ++j) {
            T v = Q[c * d + j];
            for (int k = c + 1; k < d; ++k) v -= P[c * d + k] * Q[k * d + j];
            Q[c * d + j] = v / P[c * d + c];
        }
}

// E = exp(A), d x d row-major: scaling and squaring with the [13/13] Pade approximant (Higham 2005).
template <typename T>
void expm(int d, const T* A_in, T* E, std::vector<T>& w) {
    static const double b[14] = {64764752532480000., 32382376266240000., 7771770303897600., 1187353796428800.,
                                 129060195264000., 10559470521600., 670442572800., 33522128640., 1323241920.,
                                 40840800., 960960., 16380., 182., 1.};
    const int dd = d * d;
    w.resize(9 * dd);
    T *A = &w[0], *A2 = A + dd, *A4 = A2 + dd, *A6 = A4 + dd, *U = A6 + dd, *V = U + dd, *t1 = V + dd, *t2 = t1 + dd,
      *P = t2 + dd;
    double norm = 0;
    for (int j = 0; j < d; ++j) {
        double c = 0;
        for (int i = 0; i < d; ++i) c += std::abs(double(A_in[i * d + j]));
        norm = std::max(norm, c);
    }
    const int s = norm > 5.371920351148152 ? static_cast<int>(std::ceil(std::log2(norm / 5.371920351148152))) : 0;
    const T scale = T(std::ldexp(1.0, -s));
    for (int i = 0; i < dd; ++i) A[i] = A_in[i] * scale;
    matmul(d, A, A, A2), matmul(d, A2, A2, A4), matmul(d, A4, A2, A6);
    // U = A [A6 (b13 A6 + b11 A4 + b9 A2) + b7 A6 + b5 A4 + b3 A2 + b1 I]
    for (int i = 0; i < dd; ++i) t1[i] = T(b[13]) * A6[i] + T(b[11]) * A4[i] + T(b[9]) * A2[i];
    matmul(d, A6, t1, t2);
    for (int i = 0; i < dd; ++i) t2[i] += T(b[7]) * A6[i] + T(b[5]) * A4[i] + T(b[3]) * A2[i];
    for (int i = 0; i < d; ++i) t2[i * d + i] += T(b[1]);
    matmul(d, A, t2, U);
    // V = A6 (b12 A6 + b10 A4 + b8 A2) + b6 A6 + b4 A4 + b2 A2 + b0 I
    for (int i = 0; i < dd; ++i) t1[i] = T(b[12]) * A6[i] + T(b[10]) * A4[i] + T(b[8]) * A2[i];
    matmul(d, A6, t1, V);
    for (int i = 0; i < dd; ++i) V[i] += T(b[6]) * A6[i] + T(b[4]) * A4[i] + T(b[2]) * A2[i];
    for (int i = 0; i < d; ++i) V[i * d + i] += T(b[0]);
    for (int i = 0; i < dd; ++i) P[i] = V[i] - U[i], E[i] = V[i] + U[i];
    solve(d, P, E);
    for (int k = 0; k < s; ++k) {
        matmul(d, E, E, t1);
        std::copy(t1, t1 + dd, E);
    }
}

// Off-resonance of spin i in a segment and its partial derivatives (fused/kernel.h: Block::offset,
// Adjoint::field): d/d(lin) for lin = pos.g + db0 + f, d/dg, and the field-model coefficients.
template <typename T>
struct Offset {
    T bz, dlin, dg[3], dc[7];
};

template <typename T>
Offset<T> offset(const Pools<T>& sp, int64_t i, T gx, T gy, T gz, T f) {
    Offset<T> o{};
    const T px = sp.pos[3 * i], py = sp.pos[3 * i + 1], pz = sp.pos[3 * i + 2];
    const T lin = px * gx + py * gy + pz * gz + sp.db0[i] + f;
    o.dlin = T(1), o.dg[0] = px, o.dg[1] = py, o.dg[2] = pz;
    if (!sp.conc || sp.conc_mode == 0) return o.bz = lin, o;
    if (sp.conc_mode == 1) {
        const T* c = sp.conc + 6 * i;
        const T q[6] = {gx * gx, gy * gy, gz * gz, gx * gy, gx * gz, gy * gz};
        o.bz = lin;
        for (int k = 0; k < 6; ++k) o.bz += c[k] * q[k], o.dc[k] = q[k];
        o.dg[0] += T(2) * c[0] * gx + c[3] * gy + c[4] * gz;
        o.dg[1] += T(2) * c[1] * gy + c[3] * gx + c[5] * gz;
        o.dg[2] += T(2) * c[2] * gz + c[4] * gx + c[5] * gy;
        return o;
    }
    const T* c = sp.conc + 7 * i;
    const T tx = c[0] * gx + c[1] * gy + c[2] * gz, ty = c[3] * gx + c[4] * gy + c[5] * gz;
    const T t2 = tx * tx + ty * ty, u = c[6] + lin, S = std::sqrt(u * u + t2);
    o.bz = lin + t2 / (S + u);
    o.dlin = u / S;
    const T ex = tx / S, ey = ty / S;
    o.dg[0] = o.dlin * px + ex * c[0] + ey * c[3];
    o.dg[1] = o.dlin * py + ex * c[1] + ey * c[4];
    o.dg[2] = o.dlin * pz + ex * c[2] + ey * c[5];
    o.dc[0] = ex * gx, o.dc[1] = ex * gy, o.dc[2] = ex * gz, o.dc[3] = ey * gx, o.dc[4] = ey * gy, o.dc[5] = ey * gz;
    o.dc[6] = -t2 / (S * (S + u));
    return o;
}

// The augmented matrix of spin i (exchange._bm_matrix) for RF (rx, ry), offset bz, saturation w.
template <typename T>
void build(const Pools<T>& sp, int64_t i, T rx, T ry, T bz, T w, T* A) {
    const int d = sp.dim(), one = d - 1, np = sp.np;
    std::fill(A, A + d * d, T(0));
    const T two_pi = T(2.0 * kPi), b1 = sp.b1 ? sp.b1[i] : T(1);
    const T wx = two_pi * rx * b1, wy = two_pi * ry * b1, pd = sp.pd[i];
    for (int p = 0; p <= np; ++p) {
        const T* q = p ? sp.pools + 5 * (i * np + p - 1) : nullptr;
        const T m0 = p ? q[0] * pd : pd, r1 = T(1) / (p ? q[2] : sp.t1[i]), r2 = T(1) / (p ? q[3] : sp.t2[i]);
        const T wz = two_pi * (p ? bz + q[4] : bz);
        const int x = 3 * p, y = x + 1, z = x + 2;
        A[x * d + y] += wz, A[x * d + z] -= wy, A[y * d + x] -= wz, A[y * d + z] += wx, A[z * d + x] += wy,
            A[z * d + y] -= wx;
        A[x * d + x] -= r2, A[y * d + y] -= r2, A[z * d + z] -= r1, A[z * d + one] += r1 * m0;
    }
    for (int p = 1; p <= np; ++p) {
        const T* q = sp.pools + 5 * (i * np + p - 1);
        const T k_out = q[1] * q[0], k_in = q[1];
        for (int c = 0; c < 3; ++c) {
            A[c * d + c] -= k_out, A[c * d + 3 * p + c] += k_in;
            A[(3 * p + c) * d + c] += k_out, A[(3 * p + c) * d + 3 * p + c] -= k_in;
        }
    }
    if (sp.bound) {
        const T* q = sp.bound + 5 * i;
        const int zb = 3 * (np + 1);
        A[2 * d + 2] -= q[1] * q[0], A[2 * d + zb] += q[1], A[zb * d + 2] += q[1] * q[0];
        A[zb * d + zb] -= T(1) / q[2] + q[1] + w, A[zb * d + one] += q[0] * pd / q[2];
    }
}

template <typename T>
struct Segment {
    T dt, rx, ry, gx, gy, gz, f, frf;
    bool rf;

    bool operator==(const Segment& o) const {
        return dt == o.dt && rx == o.rx && ry == o.ry && gx == o.gx && gy == o.gy && gz == o.gz && f == o.f &&
               frf == o.frf && rf == o.rf;
    }
};

template <typename T>
Segment<T> segment(const Segments<T>& seq, int64_t s) {
    const bool rf = (seq.flags[s] & kRF) != 0;
    return {seq.dt[s], rf ? seq.rf[2 * s] : T(0), rf ? seq.rf[2 * s + 1] : T(0), seq.grad[3 * s],
            seq.grad[3 * s + 1], seq.grad[3 * s + 2], seq.offset(s), rf && seq.rf_freq ? seq.rf_freq[s] : T(0), rf};
}

// Spin i's matrix and exponential for a segment (E only if `exp`).
template <typename T>
struct Step {
    std::vector<T> A, E, work;
    Offset<T> off;
    T w = 0, G = 0, g_t2 = 0, g_d = 0;  // bound-pool saturation and its lineshape partials

    void make(const Pools<T>& sp, int64_t i, const Segment<T>& sg) {
        const int d = sp.dim();
        A.resize(d * d), E.resize(d * d);
        off = offset(sp, i, sg.gx, sg.gy, sg.gz, sg.f);
        w = G = g_t2 = g_d = T(0);
        if (sp.bound && sg.rf) {  // exchange.saturation_rate
            const T* q = sp.bound + 5 * i;
            const T b1 = sp.b1 ? sp.b1[i] : T(1), two_pi = T(2.0 * kPi);
            G = sp.line.eval(sg.frf - off.bz - q[4], q[3], g_t2, g_d);
            w = two_pi * two_pi * (sg.rx * sg.rx + sg.ry * sg.ry) * b1 * b1 * G;
        }
        build(sp, i, sg.rx, sg.ry, off.bz, w, A.data());
        std::vector<T> M(A);
        for (auto& v : M) v *= sg.dt;
        expm(d, M.data(), E.data(), work);
    }
};

template <typename T>
void simulate(const Pools<T>& sp, const Segments<T>& seq, T* m, T* signal, int n_threads, T* ckpt, int64_t every) {
    const int d = sp.dim(), ds = d - 1, np = sp.np;
    const int64_t nc = sp.nc, n_adc = count_adc(seq.flags, seq.n), size = n_adc * nc * 2;
    if (!ckpt) every = std::max<int64_t>(seq.n, 1);
    n_threads = fused::detail::clamp_threads(n_threads, sp.n);
    std::vector<std::vector<T>> acc(n_threads);
    std::atomic<int64_t> next{0};
    fused::detail::parallel(n_threads, [&](int t) {
        acc[t].assign(size, T(0));
        Step<T> st;
        std::vector<T> v(d), nv(d);
        for (int64_t i; (i = next.fetch_add(1)) < sp.n;) {
            std::copy(m + ds * i, m + ds * (i + 1), v.begin());
            v[ds] = T(1);
            Segment<T> last{};
            bool have = false;
            for (int64_t s = 0, k = 0; s < seq.n; ++s) {
                if (ckpt && s % every == 0) std::copy(v.begin(), v.begin() + ds, ckpt + (s / every) * sp.n * ds + ds * i);
                const Segment<T> sg = segment(seq, s);
                if (!have || !(sg == last)) st.make(sp, i, sg), last = sg, have = true;
                for (int r = 0; r < d; ++r) {
                    T a = 0;
                    for (int c = 0; c < d; ++c) a += st.E[r * d + c] * v[c];
                    nv[r] = a;
                }
                std::swap(v, nv);
                v[ds] = T(1);
                if (seq.flags[s] & kADC) {
                    T X = 0, Y = 0;
                    for (int p = 0; p <= np; ++p) X += v[3 * p], Y += v[3 * p + 1];
                    T* o = acc[t].data() + 2 * nc * k++;
                    for (int64_t c = 0; c < nc; ++c) {
                        const T sr = sp.sens[2 * (i * nc + c)], si = sp.sens[2 * (i * nc + c) + 1];
                        o[2 * c] += X * sr - Y * si, o[2 * c + 1] += X * si + Y * sr;
                    }
                }
                if (seq.flags[s] & kSpoil)
                    for (int p = 0; p <= np; ++p) v[3 * p] = v[3 * p + 1] = T(0);
            }
            std::copy(v.begin(), v.begin() + ds, m + ds * i);
        }
    });
    std::fill(signal, signal + size, T(0));
    for (const auto& a : acc)
        for (int64_t j = 0; j < size; ++j) signal[j] += a[j];
    for (int64_t s = 0, k = 0; s < seq.n; ++s) {
        if (!(seq.flags[s] & kADC)) continue;
        const T c = std::cos(seq.adc_phase[s]), sn = std::sin(seq.adc_phase[s]);
        T* p = signal + 2 * nc * k++;
        for (int64_t j = 0; j < nc; ++j) {
            const T re = p[2 * j], im = p[2 * j + 1];
            p[2 * j] = re * c + im * sn, p[2 * j + 1] = im * c - re * sn;
        }
    }
}

// Segment gradient slots (per thread, double).
enum { kSdt, kSrx, kSry, kSgx, kSgy, kSgz, kSoff, kSfrf, kNS };

template <typename T>
void adjoint(const Pools<T>& sp, const Segments<T>& seq, const T* ckpt, int64_t every, const T* signal,
             const T* g_signal, const T* g_m, const Grads<T>& g, int n_threads) {
    const int d = sp.dim(), ds = d - 1, np = sp.np, d2 = 2 * d;
    const int64_t nc = sp.nc, ns = seq.n, n_adc = count_adc(seq.flags, ns);
    const int64_t n_ckpt = count_checkpoints(ns, every);
    const T two_pi = T(2.0 * kPi);
    // dL/d(raw coil sums), ADC numbers at checkpoints
    std::vector<T> h(g_signal ? n_adc * nc * 2 : 0);
    std::vector<int64_t> k_at(n_ckpt + 1, 0);
    for (int64_t s = 0, k = 0; s < ns; ++s) {
        if (s % every == 0) k_at[s / every] = k;
        if (!(seq.flags[s] & kADC)) continue;
        if (g_signal) {
            const T c = std::cos(seq.adc_phase[s]), sn = std::sin(seq.adc_phase[s]);
            for (int64_t j = 0; j < nc; ++j) {
                const T gr = g_signal[2 * (k * nc + j)], gi = g_signal[2 * (k * nc + j) + 1];
                h[2 * (k * nc + j)] = gr * c - gi * sn, h[2 * (k * nc + j) + 1] = gr * sn + gi * c;
            }
        }
        ++k;
    }
    if (n_ckpt) k_at[n_ckpt] = n_adc;
    n_threads = fused::detail::clamp_threads(n_threads, sp.n);
    std::vector<std::vector<double>> gseg(n_threads);
    std::atomic<int64_t> next{0};
    fused::detail::parallel(n_threads, [&](int t) {
        gseg[t].assign(kNS * ns, 0.0);
        Step<T> st;
        std::vector<T> lam(d), nl(d), buf(static_cast<size_t>(std::min(every, ns)) * d), blk(d2 * d2), bE(d2 * d2), work;
        for (int64_t i; (i = next.fetch_add(1)) < sp.n;) {
            // per-spin gradient accumulators
            double gpos[3] = {}, gpd = 0, gt1 = 0, gt2 = 0, gdb0 = 0, gb1 = 0, gconc[7] = {}, gbound[5] = {};
            std::vector<double> gpools(5 * np, 0.0), gsens(2 * nc, 0.0);
            for (int c = 0; c < ds; ++c) lam[c] = g_m ? g_m[ds * i + c] : T(0);
            lam[ds] = T(0);
            const T pd = sp.pd[i], b1 = sp.b1 ? sp.b1[i] : T(1);
            for (int64_t j = n_ckpt - 1; j >= 0; --j) {
                const int64_t s0 = j * every, s1 = std::min(s0 + every, ns);
                // replay: v before each segment of the interval
                std::vector<T> v(ckpt + j * sp.n * ds + ds * i, ckpt + j * sp.n * ds + ds * (i + 1));
                v.push_back(T(1));
                Segment<T> last{};
                bool have = false;
                for (int64_t s = s0; s < s1; ++s) {
                    std::copy(v.begin(), v.end(), &buf[(s - s0) * d]);
                    const Segment<T> sg = segment(seq, s);
                    if (!have || !(sg == last)) st.make(sp, i, sg), last = sg, have = true;
                    std::vector<T> nv(d);
                    for (int r = 0; r < d; ++r) {
                        T a = 0;
                        for (int c = 0; c < d; ++c) a += st.E[r * d + c] * v[c];
                        nv[r] = a;
                    }
                    nv[ds] = T(1);
                    if (seq.flags[s] & kSpoil)
                        for (int p = 0; p <= np; ++p) nv[3 * p] = nv[3 * p + 1] = T(0);
                    v.swap(nv);
                }
                int64_t k = k_at[j + 1];
                for (int64_t s = s1 - 1; s >= s0; --s) {
                    const uint8_t fl = seq.flags[s];
                    const T* x0 = &buf[(s - s0) * d];
                    const Segment<T> sg = segment(seq, s);
                    st.make(sp, i, sg);  // the replay's cache does not survive the reverse order
                    if (fl & kSpoil)
                        for (int p = 0; p <= np; ++p) lam[3 * p] = lam[3 * p + 1] = T(0);
                    if (fl & kADC) {
                        --k;
                        if (g_signal) {
                            // the transverse sum after the segment (before any spoil)
                            T X = 0, Y = 0;
                            for (int p = 0; p <= np; ++p)
                                for (int c = 0; c < d; ++c)
                                    X += st.E[(3 * p) * d + c] * x0[c], Y += st.E[(3 * p + 1) * d + c] * x0[c];
                            T lx = 0, ly = 0;
                            for (int64_t c = 0; c < nc; ++c) {
                                const T hr = h[2 * (k * nc + c)], hi = h[2 * (k * nc + c) + 1];
                                const T sr = sp.sens[2 * (i * nc + c)], si = sp.sens[2 * (i * nc + c) + 1];
                                lx += hr * sr + hi * si, ly += hi * sr - hr * si;
                                gsens[2 * c] += hr * X + hi * Y, gsens[2 * c + 1] += hi * X - hr * Y;
                            }
                            for (int p = 0; p <= np; ++p) lam[3 * p] += lx, lam[3 * p + 1] += ly;
                        }
                    }
                    lam[ds] = T(0);
                    // dL/dM, M = A dt: upper-right block of exp([[M^T, lam x0^T], [0, M^T]])
                    std::fill(blk.begin(), blk.end(), T(0));
                    for (int r = 0; r < d; ++r)
                        for (int c = 0; c < d; ++c) {
                            const T mt = st.A[c * d + r] * sg.dt;
                            blk[r * d2 + c] = mt, blk[(d + r) * d2 + d + c] = mt;
                            blk[r * d2 + d + c] = lam[r] * x0[c];
                        }
                    expm(d2, blk.data(), bE.data(), work);
                    // lambda <- E^T lambda
                    for (int c = 0; c < d; ++c) {
                        T a = 0;
                        for (int r = 0; r < d; ++r) a += st.E[r * d + c] * lam[r];
                        nl[c] = a;
                    }
                    std::swap(lam, nl);
                    // D = dL/dA = dt G; dL/ddt = sum G A
                    auto D = [&](int r, int c) { return bE[r * d2 + d + c] * sg.dt; };
                    double gdt = 0;
                    for (int r = 0; r < ds; ++r)
                        for (int c = 0; c < d; ++c) gdt += double(bE[r * d2 + d + c]) * double(st.A[r * d + c]);
                    // chain through the entries of A (build())
                    const int one = ds;
                    double gwx = 0, gwy = 0, gbz = 0;
                    for (int p = 0; p <= np; ++p) {
                        const T* q = p ? sp.pools + 5 * (i * np + p - 1) : nullptr;
                        const int x = 3 * p, y = x + 1, z = x + 2;
                        const T m0 = p ? q[0] * pd : pd, r1 = T(1) / (p ? q[2] : sp.t1[i]), r2 = T(1) / (p ? q[3] : sp.t2[i]);
                        const double gwz = double(D(x, y)) - D(y, x);
                        gwx += double(D(y, z)) - D(z, y), gwy += double(D(z, x)) - D(x, z);
                        const double gr2 = -(double(D(x, x)) + D(y, y)), gr1 = -double(D(z, z)) + double(m0) * D(z, one);
                        const double gm0 = double(r1) * D(z, one);
                        gbz += two_pi * gwz;
                        if (p) {
                            double* gp = &gpools[5 * (p - 1)];
                            gp[0] += pd * gm0, gpd += q[0] * gm0;
                            gp[2] -= gr1 * r1 * r1, gp[3] -= gr2 * r2 * r2;  // r = 1 / t
                            gp[4] += two_pi * gwz;
                        } else {
                            gpd += gm0, gt1 -= gr1 * r1 * r1, gt2 -= gr2 * r2 * r2;
                        }
                    }
                    for (int p = 1; p <= np; ++p) {
                        const T* q = sp.pools + 5 * (i * np + p - 1);
                        double gko = 0, gki = 0;
                        for (int c = 0; c < 3; ++c) {
                            gko += -double(D(c, c)) + D(3 * p + c, c);
                            gki += double(D(c, 3 * p + c)) - D(3 * p + c, 3 * p + c);
                        }
                        double* gp = &gpools[5 * (p - 1)];
                        gp[1] += q[0] * gko + gki, gp[0] += q[1] * gko;
                    }
                    double* sgv = &gseg[t][kNS * s];
                    if (sp.bound) {
                        const T* q = sp.bound + 5 * i;
                        const int zb = 3 * (np + 1);
                        const double gkf = -double(D(2, 2)) + D(zb, 2), gkb = double(D(2, zb)) - D(zb, zb);
                        const double gw = -double(D(zb, zb));
                        const double gr1b = -double(D(zb, zb)), gc = double(D(zb, one));  // c = f pd r1b
                        const double r1b = 1.0 / double(q[2]);
                        gbound[0] += q[1] * gkf + pd * r1b * gc, gbound[1] += q[0] * gkf + gkb;
                        gbound[2] -= (gr1b + double(q[0]) * pd * gc) * r1b * r1b;
                        gpd += q[0] * r1b * gc;
                        if (sg.rf) {  // w = (2 pi)^2 |rf|^2 b1^2 G(f_rf - bz - dw, t2b)
                            const double kw = double(two_pi) * two_pi * (double(sg.rx) * sg.rx + double(sg.ry) * sg.ry);
                            const double yw = gw;
                            gbound[3] += yw * kw * b1 * b1 * st.g_t2;
                            const double yd = yw * kw * b1 * b1 * st.g_d;
                            gbound[4] -= yd, gbz -= yd, sgv[kSfrf] += yd;
                            gb1 += yw * kw * 2 * b1 * st.G;
                            const double yk = yw * double(two_pi) * two_pi * b1 * b1 * st.G;
                            sgv[kSrx] += yk * 2 * sg.rx, sgv[kSry] += yk * 2 * sg.ry;
                        }
                    }
                    // wx = 2 pi rx b1, wy = 2 pi ry b1 (only RF segments carry RF)
                    if (sg.rf) {
                        sgv[kSrx] += two_pi * b1 * gwx, sgv[kSry] += two_pi * b1 * gwy;
                        gb1 += two_pi * (sg.rx * gwx + sg.ry * gwy);
                    }
                    // offset chain
                    const Offset<T>& o = st.off;
                    gdb0 += gbz * o.dlin;
                    gpos[0] += gbz * o.dlin * sg.gx, gpos[1] += gbz * o.dlin * sg.gy, gpos[2] += gbz * o.dlin * sg.gz;
                    sgv[kSgx] += gbz * o.dg[0], sgv[kSgy] += gbz * o.dg[1], sgv[kSgz] += gbz * o.dg[2];
                    sgv[kSoff] += gbz * o.dlin;
                    for (int c = 0; c < 7; ++c) gconc[c] += gbz * o.dc[c];
                    sgv[kSdt] += gdt;
                }
            }
            if (g.pos) for (int c = 0; c < 3; ++c) g.pos[3 * i + c] = T(gpos[c]);
            if (g.pd) g.pd[i] = T(gpd);
            if (g.t1) g.t1[i] = T(gt1);
            if (g.t2) g.t2[i] = T(gt2);
            if (g.db0) g.db0[i] = T(gdb0);
            if (g.b1) g.b1[i] = T(gb1);
            if (g.sens) for (int64_t c = 0; c < 2 * nc; ++c) g.sens[2 * nc * i + c] = T(gsens[c]);
            if (g.pools) for (int c = 0; c < 5 * np; ++c) g.pools[5 * np * i + c] = T(gpools[c]);
            if (g.bound) for (int c = 0; c < 5; ++c) g.bound[5 * i + c] = T(gbound[c]);
            if (g.conc) {
                const int ncf = sp.conc_mode == 2 ? 7 : 6;
                for (int c = 0; c < ncf; ++c) g.conc[ncf * i + c] = T(gconc[c]);
            }
            if (g.m) for (int c = 0; c < ds; ++c) g.m[ds * i + c] = lam[c];
        }
    });
    for (int64_t s = 0; s < ns; ++s) {
        double v[kNS] = {};
        for (const auto& a : gseg)
            for (int q = 0; q < kNS; ++q) v[q] += a[kNS * s + q];
        if (g.dt) g.dt[s] = T(v[kSdt]);
        if (g.rf) g.rf[2 * s] = T(v[kSrx]), g.rf[2 * s + 1] = T(v[kSry]);
        if (g.grad) g.grad[3 * s] = T(v[kSgx]), g.grad[3 * s + 1] = T(v[kSgy]), g.grad[3 * s + 2] = T(v[kSgz]);
        if (g.b0_offset) g.b0_offset[s] = T(v[kSoff]);
        if (g.rf_freq) g.rf_freq[s] = T(v[kSfrf]);
    }
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

using detail::adjoint;
using detail::simulate;

}  // namespace blochparty::fused::bm
