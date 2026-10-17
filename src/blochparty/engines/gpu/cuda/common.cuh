
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
