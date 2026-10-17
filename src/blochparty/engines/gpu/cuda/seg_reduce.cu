
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
