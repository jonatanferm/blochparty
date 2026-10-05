// Fused forward Bloch kernel: the math of bloch.py, but one pass over the sequence per
// block of spins, with the block's state in L1 instead of ~30 tensor ops over all spins
// per segment.
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
};

template <typename T>
struct Segments {
    int64_t n;             // number of segments
    const T* dt;           // (n,) [s]
    const T* rf;           // (n, 2) [Hz]
    const T* grad;         // (n, 3) [Hz/m]
    const T* adc_phase;    // (n,) [rad]
    const uint8_t* flags;  // (n,) kRF | kADC | kSpoil
};

inline int64_t count_adc(const uint8_t* flags, int64_t n) {
    int64_t k = 0;
    for (int64_t s = 0; s < n; ++s) k += (flags[s] & kADC) != 0;
    return k;
}

namespace detail {

constexpr double kPi = 3.14159265358979323846;

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
    alignas(64) T px[B], py[B], pz[B], db0[B], pd[B], r1[B], r2[B];
    alignas(64) T mx[B], my[B], mz[B];
    std::vector<T> sr, si;  // (C, B) sensitivities
    // Relaxation over a duration: expm1(-t * R1), expm1(-t * R2). Key: t.
    Cache<T, B, 1, 2, 8> relax;
    // Free precession: cos and sin of the phase. Key: dt, grad.
    Cache<T, B, 4, 2, 8> prec;
    // RF rotation: cos(phi), sin(phi) / phi, (1 - cos(phi)) / phi^2 and wz. Key: dt, rf, grad.
    Cache<T, B, 6, 4, 4> rot;

    explicit Block(int64_t n_coils) : sr(n_coils * B), si(n_coils * B) {}

    // Copy spins [start, start + B) in; lanes past the end get PD = 0, rates 0 and
    // zero sensitivity, so they stay at zero and add nothing to the signal.
    void load(const Spins<T>& sp, const T* m, int64_t start) {
        const int64_t nc = sp.n_coils;
        relax.clear(), prec.clear(), rot.clear();
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
            mx[l] = in ? m[3 * i] : T(0);
            my[l] = in ? m[3 * i + 1] : T(0);
            mz[l] = in ? m[3 * i + 2] : T(0);
            for (int64_t c = 0; c < nc; ++c) {
                sr[c * B + l] = in ? sp.sens[2 * (i * nc + c)] : T(0);
                si[c * B + l] = in ? sp.sens[2 * (i * nc + c) + 1] : T(0);
            }
        }
    }

    void store(T* m, int64_t start, int64_t n) const {
        for (int l = 0; l < B && start + l < n; ++l) {
            m[3 * (start + l)] = mx[l];
            m[3 * (start + l) + 1] = my[l];
            m[3 * (start + l) + 2] = mz[l];
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

    void run(const Segments<T>& seq, int64_t n_coils, T* acc) {
        const T w_per_s = T(-2.0 * kPi);  // as in bloch.py: w = -TWO_PI * dt, rounded to T
        const T tiny = T(1e-30);
        int64_t k = 0;  // ADC sample index
        for (int64_t s = 0; s < seq.n; ++s) {
            const T dt = seq.dt[s];
            const T gx = seq.grad[3 * s], gy = seq.grad[3 * s + 1], gz = seq.grad[3 * s + 2];
            const uint8_t f = seq.flags[s];
            const T w = w_per_s * dt;
            bool fill;
            if (f & kRF) {
                // relax dt/2 -> rotate by w (Rodrigues) -> relax dt/2
                const T rx = seq.rf[2 * s], ry = seq.rf[2 * s + 1];
                const T wx = w * rx, wy = w * ry;
                const T (*d)[B] = relaxation(T(0.5) * dt);
                T (*r)[B] = rot.find({dt, rx, ry, gx, gy, gz}, fill);
                if (fill)
                    for (int i = 0; i < B; ++i) {
                        const T wz = w * (px[i] * gx + py[i] * gy + pz[i] * gz + db0[i]);
                        const T phi = std::sqrt(wx * wx + wy * wy + wz * wz + tiny);
                        const T hp = T(0.5) * phi;
                        const T sh = std::sin(hp) / hp;
                        r[0][i] = std::cos(phi);
                        r[1][i] = std::sin(phi) / phi;  // sin(phi) / phi
                        r[2][i] = T(0.5) * sh * sh;     // (1 - cos(phi)) / phi^2
                        r[3][i] = wz;
                    }
                for (int i = 0; i < B; ++i) {
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
                T (*p)[B] = prec.find({dt, gx, gy, gz}, fill);
                if (fill)
                    for (int i = 0; i < B; ++i) {
                        const T phi = w * (px[i] * gx + py[i] * gy + pz[i] * gz + db0[i]);
                        p[0][i] = std::cos(phi);
                        p[1][i] = std::sin(phi);
                    }
                for (int i = 0; i < B; ++i) {
                    const T x = p[0][i] * mx[i] - p[1][i] * my[i];
                    const T y = p[1][i] * mx[i] + p[0][i] * my[i];
                    mx[i] = x + x * d[1][i];
                    my[i] = y + y * d[1][i];
                    mz[i] = mz[i] + (mz[i] - pd[i]) * d[0][i];
                }
            }
            if (f & kADC) sample(acc + 2 * n_coils * k++, n_coils);
            if (f & kSpoil)
                for (int i = 0; i < B; ++i) mx[i] = T(0), my[i] = T(0);
        }
    }
};

template <typename T, int B>
void simulate_blocks(const Spins<T>& sp, const Segments<T>& seq, T* m, T* signal, int n_threads) {
    const int64_t nc = sp.n_coils;
    const int64_t n_adc = count_adc(seq.flags, seq.n);
    const int64_t n_blocks = (sp.n + B - 1) / B;
    const int64_t size = n_adc * nc * 2;
    if (n_threads <= 0) n_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    n_threads = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(n_threads, n_blocks)));

    std::vector<std::vector<T>> acc(n_threads);
    std::vector<std::exception_ptr> errors(n_threads);
    std::atomic<int64_t> next{0};
    auto worker = [&](int t) {
        try {
            acc[t].assign(size, T(0));  // allocated by the thread that uses it
            auto blk = std::make_unique<Block<T, B>>(nc);
            for (int64_t b; (b = next.fetch_add(1)) < n_blocks;) {
                blk->load(sp, m, b * B);
                blk->run(seq, nc, acc[t].data());
                blk->store(m, b * B, sp.n);
            }
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

}  // namespace detail

constexpr int kBlockSizes[] = {16, 32, 64, 128, 256};

// Run the sequence on the spins.
//   m:      (n, 3) magnetization, read as the start state and overwritten with the final one
//   signal: (count_adc(seq), C, 2) output, (re, im) per ADC sample and coil
//   n_threads <= 0 uses every hardware thread; block must be one of kBlockSizes.
template <typename T>
void simulate(const Spins<T>& spins, const Segments<T>& seq, T* m, T* signal, int n_threads = 0,
              int block = 64) {
    switch (block) {
        case 16: return detail::simulate_blocks<T, 16>(spins, seq, m, signal, n_threads);
        case 32: return detail::simulate_blocks<T, 32>(spins, seq, m, signal, n_threads);
        case 64: return detail::simulate_blocks<T, 64>(spins, seq, m, signal, n_threads);
        case 128: return detail::simulate_blocks<T, 128>(spins, seq, m, signal, n_threads);
        case 256: return detail::simulate_blocks<T, 256>(spins, seq, m, signal, n_threads);
        default: throw std::invalid_argument("block must be 16, 32, 64, 128 or 256");
    }
}

}  // namespace blochparty::fused
