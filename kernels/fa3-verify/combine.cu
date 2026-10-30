#include <cuda_bf16.h>
#include "merge_fp8.cuh"
#include "verify.h"
#include "cuda_check.h"
namespace apac_fa3_verify {
// Eight lanes cooperate on each D128 row; all partial arithmetic stays FP32.
template<int S, bool FP8>
__global__ void combine_fixed(const float* __restrict__ partial, const float* __restrict__ plse, __nv_bfloat16* __restrict__ out, float* __restrict__ lse, int tq, int* semaphore, uint8_t* fp8, float* scales, int64_t scale_stride, const int* dynamic, const int* order, int batch) {
    int lane = threadIdx.x % 8;
    int row = blockIdx.x * 16 + threadIdx.x / 8;
    int rows = tq * 64;
    if (blockIdx.x == 0 && threadIdx.x == 0) { *semaphore = 0; }
    if (row >= rows) { return; }
    // Partial storage is head-major; final output is token-major.
    int head = row / tq, token = row % tq;
    int active = S;
    if (S == 4) {
        // Dynamic split counts are indexed by virtual (sorted) batch id.
        for (int i = 0; i < batch; ++i) { if (order[i] == token / 4) { active = dynamic[i]; break; } }
    }
    // Stream split (four KV heads per CTA): the main kernel stored each request's slot count.
    if (S == 6) { active = dynamic[token / 4]; }
    float weight[S] = {};
    if (lane == 0) {
        float local[S], maximum = -INFINITY;
        #pragma unroll
        for (int s = 0; s < S; ++s) { local[s] = s < active ? plse[s * rows + row] : -INFINITY; maximum = fmaxf(maximum, local[s]); }
        float m = maximum == -INFINITY ? 0.f : maximum, sum = 0.f;
        #pragma unroll
        for (int s = 0; s < S; ++s) { weight[s] = expf(local[s] - m); sum += weight[s]; }
        float inv = sum == 0.f || isnan(sum) ? 0.f : 1.f / sum;
        #pragma unroll
        for (int s = 0; s < S; ++s) { weight[s] *= inv; }
        lse[row] = logf(sum) + maximum;
    }
    #pragma unroll
    for (int s = 0; s < S; ++s) { weight[s] = __shfl_sync(0xffffffff, weight[s], 0, 8); }
    float4 values[4];
    #pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        int d = lane * 4 + chunk * 32;
        float4 sum = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int s = 0; s < S; ++s) {
            if (weight[s] == 0.f) { continue; }
            float4 x = *reinterpret_cast<const float4*>(partial + (s * rows + row) * 128 + d);
            sum.x += weight[s] * x.x; sum.y += weight[s] * x.y; sum.z += weight[s] * x.z; sum.w += weight[s] * x.w;
        }
        if constexpr (FP8) { values[chunk] = sum; }
        else {
            auto target = out + (token * 64 + head) * 128 + d;
            *reinterpret_cast<__nv_bfloat162*>(target) = __floats2bfloat162_rn(sum.x, sum.y);
            *reinterpret_cast<__nv_bfloat162*>(target + 2) = __floats2bfloat162_rn(sum.z, sum.w);
        }
    }
    if constexpr (FP8) { merge_row_to_fp8(values, fp8 + (token * 64 + head) * 128, scales + int64_t(head) * scale_stride + token); }
}
template<int S, bool FP8>
void launch_combine(Flash_fwd_params& p, cudaStream_t stream) {
    combine_fixed<S, FP8><<<(p.total_q * 64 + 15) / 16, 128, 0, stream>>>((float*)p.oaccum_ptr, (float*)p.softmax_lseaccum_ptr, (__nv_bfloat16*)p.o_ptr, (float*)p.softmax_lse_ptr, p.total_q, p.tile_count_semaphore, (uint8_t*)p.o_fp8_ptr, p.o_scale_ptr, p.o_scale_stride, p.num_splits_dynamic_ptr, p.varlen_batch_idx_ptr, p.b);
}
void combine(Flash_fwd_params& p, cudaStream_t stream) {
    if (p.o_fp8_ptr) {
        if (p.num_splits == 2) { launch_combine<2, true>(p, stream); }
        else if (p.num_splits == 4) { launch_combine<4, true>(p, stream); }
        else if (p.num_splits == 6) { launch_combine<6, true>(p, stream); }
        else { launch_combine<3, true>(p, stream); }
    } else {
        if (p.num_splits == 2) { launch_combine<2, false>(p, stream); }
        else if (p.num_splits == 4) { launch_combine<4, false>(p, stream); }
        else { launch_combine<3, false>(p, stream); }
    }
    CHECK_CUDA_KERNEL_LAUNCH();
}
}
