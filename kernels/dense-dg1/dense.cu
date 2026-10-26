// Decode-only SwapAB FP8 block128 GEMM. No atomics or grid-wide spin waits.
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cute/numeric/integer_sequence.hpp>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace {
constexpr int sms = 132;
// Both math WGs consume the same staged weight tile and token tile. Each
// computes 64 weight rows; tokens occupy WGMMA N rather than padded M.
template<int BN>
__device__ void load_stage(unsigned char* dst, const unsigned char* x, const unsigned char* w, int rows, int k, int tile, int kb) {
    for (int v = threadIdx.x; v < (128 + BN) * 8; v += 256) {
        const int row = v / 8, col = (v % 8) * 16;
        auto target = dst + row * 128 + (col ^ ((row % 8) * 16));
        const bool valid = row < 128 || row - 128 < rows;
        auto source = row < 128 ? w + (tile * 128 + row) * k + kb * 128 + col : x + (valid ? (row - 128) * k + kb * 128 + col : 0);
        // Predicated 16-byte async copy zero-fills the token tail.
        __pipeline_memcpy_async(target, source, 16, valid ? 0 : 16);
    }
    __pipeline_commit();
}

template<int BN>
__global__ __launch_bounds__(256, 1) void stream_k(const unsigned char* x, const unsigned char* w, const float* sx, const float* sw, float* partial, int rows, int n, int k, int sx0, int sx1) {
    using MMA = typename deep_gemm::mma::sm90::FP8MMASelector<BN>::type;
    constexpr int stage_bytes = (128 + BN) * 128;
    extern __shared__ __align__(1024) unsigned char stages[];
    const int blocks = k / 128, total = (n / 128) * blocks;
    const int begin = total * blockIdx.x / sms, end = total * (blockIdx.x + 1) / sms;
    const int lane = threadIdx.x % 32, warp = (threadIdx.x % 128) / 32, wg = threadIdx.x / 128;
    const int r0 = wg * 64 + warp * 16 + lane / 4;
    int unit = begin;
    load_stage<BN>(stages, x, w, rows, k, unit / blocks, unit % blocks);
    __pipeline_wait_prior(0);
    __syncthreads();
    while (unit < end) {
        const int tile = unit / blocks, stop = min(end, (tile + 1) * blocks);
        float sum[BN / 2] = {};
        for (; unit < stop; ++unit) {
            const int stage = (unit - begin) % 2, kb = unit % blocks;
            if (unit + 1 < end)
                load_stage<BN>(stages + (1 - stage) * stage_bytes, x, w, rows, k, (unit + 1) / blocks, (unit + 1) % blocks);
            // Generic writes must be visible to WGMMA's async proxy.
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            float acc[BN / 2];
            #pragma unroll
            for (int i = 0; i < BN / 2; ++i) deep_gemm::ptx::warpgroup_fence_operand(acc[i]);
            deep_gemm::ptx::warpgroup_arrive();
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const auto a = deep_gemm::mma::sm90::make_smem_desc(stages + stage * stage_bytes + wg * 64 * 128 + kk * 32, 1);
                const auto b = deep_gemm::mma::sm90::make_smem_desc(stages + stage * stage_bytes + 128 * 128 + kk * 32, 1);
                MMA::wgmma(a, b, acc, kk != 0);
            }
            deep_gemm::ptx::warpgroup_commit_batch();
            deep_gemm::ptx::warpgroup_wait<0>();
            #pragma unroll
            for (int i = 0; i < BN / 2; ++i) deep_gemm::ptx::warpgroup_fence_operand(acc[i]);
            const float weight_scale = sw[tile * blocks + kb];
            #pragma unroll
            for (int i = 0; i < BN / 2; ++i) {
                const int token = (i / 4) * 8 + (lane % 4) * 2 + i % 2;
                const float token_scale = token < rows ? sx[token * sx0 + kb * sx1] : 0.f;
                sum[i] += acc[i] * (weight_scale * token_scale);
            }
            if (unit + 1 < end) __pipeline_wait_prior(0);
            // Both WGs must release the old shared stage before it is reused.
            __syncthreads();
        }
        const int slot = 2 * blockIdx.x + tile - begin / blocks;
        #pragma unroll
        for (int i = 0; i < BN / 2; ++i) {
            const int token = (i / 4) * 8 + (lane % 4) * 2 + i % 2;
            const int weight = r0 + ((i % 4) / 2) * 8;
            partial[(slot * BN + token) * 128 + weight] = sum[i];
        }
    }
}

template<int BN>
__global__ void fixup(const float* partial, __nv_bfloat16* out, int rows, int n, int k) {
    const int tile = blockIdx.x, blocks = k / 128, total = (n / 128) * blocks;
    // Inverse of floor(total * CTA / 132), including non-divisible endpoints.
    const int first = ((tile * blocks + 1) * sms + total - 1) / total - 1;
    const int last = (((tile + 1) * blocks) * sms + total - 1) / total - 1;
    for (int i = threadIdx.x; i < rows * 128; i += blockDim.x) {
        float value = 0.f;
        for (int cta = first; cta <= last; ++cta) {
            const int slot = 2 * cta + tile - (total * cta / sms) / blocks;
            value += partial[(slot * BN + i / 128) * 128 + i % 128];
        }
        out[(i / 128) * n + tile * 128 + i % 128] = __float2bfloat16_rn(value);
    }
}

template<int BN>
cudaError_t launch(const unsigned char* x, const unsigned char* w, const float* sx, const float* sw, float* partial, __nv_bfloat16* out, int rows, int n, int k, int sx0, int sx1, cudaStream_t stream) {
    auto status = cudaFuncSetAttribute(stream_k<BN>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * (128 + BN) * 128);
    if (status != cudaSuccess) return status;
    stream_k<BN><<<sms, 256, 2 * (128 + BN) * 128, stream>>>(x, w, sx, sw, partial, rows, n, k, sx0, sx1);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    fixup<BN><<<n / 128, 256, 0, stream>>>(partial, out, rows, n, k);
    return cudaGetLastError();
}
} // namespace

extern "C" int dense_dg1(const unsigned char* x, const unsigned char* w, const float* sx, const float* sw, float* partial, __nv_bfloat16* out, int rows, int n, int k, int sx0, int sx1, cudaStream_t stream) {
    if (rows <= 0 || rows > 176 || !((n == 9216 && k == 4096) || (n == 4096 && k == 8192))) return cudaErrorInvalidValue;
    if (rows <= 128) return launch<128>(x, w, sx, sw, partial, out, rows, n, k, sx0, sx1, stream);
    if (rows <= 160) return launch<160>(x, w, sx, sw, partial, out, rows, n, k, sx0, sx1, stream);
    return launch<176>(x, w, sx, sw, partial, out, rows, n, k, sx0, sx1, stream);
}
