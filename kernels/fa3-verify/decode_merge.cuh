#pragma once
#include <cstdint>
// Private companion ABI; carried only to Decode's independent epilogue hook.
struct DecodeMergeArgs {
    int* counters = nullptr;
    const float* partial = nullptr;
    const float* partial_lse = nullptr;
    void* output = nullptr;
    float* lse = nullptr;
    uint8_t* fp8 = nullptr;
    float* scales = nullptr;
    int64_t scale_stride = 0;
    int total_q = 0;
};
#ifdef __CUDACC__
#include <cuda/atomic>
#include <cuda_bf16.h>
#include "merge_fp8.cuh"
// Split partials are produced by other CTAs. Read through L2, not a consumer's
// non-coherent L1 (the workspace is reused across all layers and graph replays).
__device__ __forceinline__ float merge_load_l2(const float* ptr) {
    float x;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(x) : "l"(ptr) : "memory");
    return x;
}
__device__ __forceinline__ float4 merge_load_l2x4(const float* ptr) {
    float4 x;
    asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(x.x), "=f"(x.y), "=f"(x.z), "=f"(x.w) : "l"(ptr) : "memory");
    return x;
}
// Only the 128 consumer threads enter this function. Producers may already
// be loading the next unit, so a CTA-wide __syncthreads would deadlock.
__device__ __forceinline__ void decode_merge_finish(const DecodeMergeArgs& p, int tid, int batch, int kv_head, int splits) {
    // Reuse reserved EpilogueBarrier (1) after its split stores have finished.
    // Split varlen uses direct global stores; no producer participates here.
    __shared__ int last;
    // Every partial/LSE writer publishes before thread 0 signals arrival.
    __threadfence();
    asm volatile("bar.sync 1, 128;" ::: "memory");
    cuda::atomic_ref<int, cuda::thread_scope_device> arrived(p.counters[batch * 4 + kv_head]);
    if (tid == 0) { last = arrived.fetch_add(1, cuda::memory_order_acq_rel) == splits - 1; }
    asm volatile("bar.sync 1, 128;" ::: "memory");
    if (last) {
        // The elected thread acquired the arrival chain; every reader must
        // acquire it too before reading another CTA's published partials.
        (void)arrived.load(cuda::memory_order_acquire);
        int lane = tid % 8;
        int rows = p.total_q * 64;
        // Four iterations cover all Q4 x GQA16 rows for this KV head.
        for (int r = tid / 8; r < 64; r += 16) {
            int token = batch * 4 + r / 16, head = kv_head * 16 + r % 16;
            int row = head * p.total_q + token;
            float w[4] = {}, maximum = -INFINITY;
            if (lane == 0) {
                for (int s = 0; s < splits; ++s) { maximum = fmaxf(maximum, merge_load_l2(p.partial_lse + s * rows + row)); }
                float m = maximum == -INFINITY ? 0.f : maximum, sum = 0.f;
                for (int s = 0; s < splits; ++s) { w[s] = expf(merge_load_l2(p.partial_lse + s * rows + row) - m); sum += w[s]; }
                float inv = sum == 0.f || isnan(sum) ? 0.f : 1.f / sum;
                for (int s = 0; s < splits; ++s) { w[s] *= inv; }
                p.lse[row] = logf(sum) + maximum;
            }
            for (int s = 0; s < splits; ++s) { w[s] = __shfl_sync(0xffffffff, w[s], 0, 8); }
            float4 values[4];
            #pragma unroll
            for (int chunk = 0; chunk < 4; ++chunk) {
                int d = lane * 4 + chunk * 32;
                float4 sum = {0.f, 0.f, 0.f, 0.f};
                // Fixed ascending split order, independent of arrival order.
                for (int s = 0; s < splits; ++s) {
                    if (w[s] == 0.f) { continue; } // Never read poisoned empty partials.
                    float4 x = merge_load_l2x4(p.partial + (s * rows + row) * 128 + d);
                    sum.x += w[s] * x.x; sum.y += w[s] * x.y; sum.z += w[s] * x.z; sum.w += w[s] * x.w;
                }
                values[chunk] = sum;
                if (!p.fp8) {
                    auto out = static_cast<__nv_bfloat16*>(p.output) + (token * 64 + head) * 128 + d;
                    *reinterpret_cast<__nv_bfloat162*>(out) = __floats2bfloat162_rn(sum.x, sum.y);
                    *reinterpret_cast<__nv_bfloat162*>(out + 2) = __floats2bfloat162_rn(sum.z, sum.w);
                }
            }
            if (p.fp8) { merge_row_to_fp8(values, p.fp8 + (token * 64 + head) * 128, p.scales + int64_t(head) * p.scale_stride + token); }
        }
        // No future unit in this invocation uses this request/head. The next
        // invocation is ordered on the same stream, including graph replay.
        asm volatile("bar.sync 1, 128;" ::: "memory");
        if (tid == 0) { arrived.store(0, cuda::memory_order_release); }
    }
    // Keep all consumers together before reusing the shared arrival flag.
    asm volatile("bar.sync 1, 128;" ::: "memory");
}
#endif
