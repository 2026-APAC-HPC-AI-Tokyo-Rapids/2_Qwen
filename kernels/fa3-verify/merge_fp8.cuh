#pragma once
#include <cstdint>
#include "decode_probe.cuh"
#include <cuda_fp8.h>
// Eight lanes hold four float4 chunks of one D128 group. Reusable by the
// future last-arriving split CTA; callers merge in the original split order.
__device__ __forceinline__ void merge_row_to_fp8(const float4 (&values)[4], uint8_t* out, float* scale
#ifdef FA3_DECODE_MERGE_PROBE
    , const DecodeProbe& probe, int out_row, int scale_row
#endif
) {
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(values[i].x), fabsf(values[i].y)), fmaxf(fabsf(values[i].z), fabsf(values[i].w))));
    }
    #pragma unroll
    for (int mask = 4; mask >= 1; mask >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 8)); }
    amax = fmaxf(amax, 1e-10f);
    float gain = 448.f / amax;
    int lane = threadIdx.x % 8;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        auto v = values[i];
#ifdef FA3_DECODE_MERGE_PROBE
        *reinterpret_cast<float4*>(probe.merged + out_row * 128 + lane * 4 + i * 32) = v;
#endif
        uint16_t lo = __nv_cvt_float2_to_fp8x2(make_float2(fminf(v.x * gain, 448.f), fminf(v.y * gain, 448.f)), __NV_SATFINITE, __NV_E4M3);
        uint16_t hi = __nv_cvt_float2_to_fp8x2(make_float2(fminf(v.z * gain, 448.f), fminf(v.w * gain, 448.f)), __NV_SATFINITE, __NV_E4M3);
        *reinterpret_cast<uint32_t*>(out + lane * 4 + i * 32) = uint32_t(lo) | (uint32_t(hi) << 16);
#ifdef FA3_DECODE_MERGE_PROBE
        *reinterpret_cast<uint32_t*>(probe.codes + out_row * 128 + lane * 4 + i * 32) = uint32_t(lo) | (uint32_t(hi) << 16);
#endif
    }
#ifdef FA3_DECODE_MERGE_PROBE
    if (lane == 0) { probe.amax[scale_row] = amax; probe.scales[scale_row] = amax * (1.f / 448.f); }
#endif
    if (lane == 0) { *scale = amax * (1.f / 448.f); }
}
