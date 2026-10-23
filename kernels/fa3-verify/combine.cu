#include <cuda_bf16.h>
#include "verify.h"
#include "cuda_check.h"
namespace apac_fa3_verify {
// Eight lanes cooperate on each D128 row; all partial arithmetic stays FP32.
template<int S>
__global__ void combine_fixed(const float* __restrict__ partial, const float* __restrict__ plse, __nv_bfloat16* __restrict__ out, float* __restrict__ lse, int tq, int* semaphore) {
    int lane = threadIdx.x % 8;
    int row = blockIdx.x * 16 + threadIdx.x / 8;
    int rows = tq * 64;
    if (blockIdx.x == 0 && threadIdx.x == 0) { *semaphore = 0; }
    if (row >= rows) { return; }
    // Partial storage is head-major; final output is token-major.
    int head = row / tq, token = row % tq;
    float weight[S] = {};
    if (lane == 0) {
        float local[S], maximum = -INFINITY;
        #pragma unroll
        for (int s = 0; s < S; ++s) { local[s] = plse[s * rows + row]; maximum = fmaxf(maximum, local[s]); }
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
    #pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        int d = lane * 4 + chunk * 32;
        float4 sum = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int s = 0; s < S; ++s) {
            float4 x = *reinterpret_cast<const float4*>(partial + (s * rows + row) * 128 + d);
            sum.x += weight[s] * x.x; sum.y += weight[s] * x.y; sum.z += weight[s] * x.z; sum.w += weight[s] * x.w;
        }
        auto target = out + (token * 64 + head) * 128 + d;
        *reinterpret_cast<__nv_bfloat162*>(target) = __floats2bfloat162_rn(sum.x, sum.y);
        *reinterpret_cast<__nv_bfloat162*>(target + 2) = __floats2bfloat162_rn(sum.z, sum.w);
    }
}
void combine(Flash_fwd_params& p, cudaStream_t stream) {
    int grid = (p.total_q * 64 + 15) / 16;
    if (p.num_splits == 2) { combine_fixed<2><<<grid, 128, 0, stream>>>((float*)p.oaccum_ptr, (float*)p.softmax_lseaccum_ptr, (__nv_bfloat16*)p.o_ptr, (float*)p.softmax_lse_ptr, p.total_q, p.tile_count_semaphore); }
    else { combine_fixed<3><<<grid, 128, 0, stream>>>((float*)p.oaccum_ptr, (float*)p.softmax_lseaccum_ptr, (__nv_bfloat16*)p.o_ptr, (float*)p.softmax_lse_ptr, p.total_q, p.tile_count_semaphore); }
    CHECK_CUDA_KERNEL_LAUNCH();
}
}
