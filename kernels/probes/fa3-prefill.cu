// Pinned FA3 with optional group-128 FP8 output; BF16 controls use the same launch.
#include <cstdio>
#ifndef FA3_PROBE_E4
#define FA3_PROBE_E4 0
#endif
#ifndef FA3_PROBE_M
#define FA3_PROBE_M 0
#endif
#ifndef FA3_PROBE_NOBARRIER
#define FA3_PROBE_NOBARRIER 0
#endif
#ifndef FA3_PROBE_REGS232
#define FA3_PROBE_REGS232 0
#endif
#ifndef FA3_PROBE_LAZY
#define FA3_PROBE_LAZY 0
#endif
#ifndef FA3_PROBE_COUNTER
#define FA3_PROBE_COUNTER 0
#endif
#ifndef FA3_PROBE_VCOL
#define FA3_PROBE_VCOL 0
#endif
#if FA3_PROBE_COUNTER
__device__ unsigned long long fa3_probe_counts[2];
#endif
#include "flash_fwd_launch_template.h"
#if FA3_PROBE_COUNTER
extern "C" int prefill_counts(void* output, void* stream) {
    return int(cudaMemcpyFromSymbolAsync(output, fa3_probe_counts, sizeof(fa3_probe_counts), 0, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)));
}
extern "C" int prefill_counts_reset() {
    unsigned long long zero[2] = {};
    return int(cudaMemcpyToSymbol(fa3_probe_counts, zero, sizeof(zero)));
}
#endif

template<int N>
void launch(Flash_fwd_params& p, cudaStream_t stream, int* resource=nullptr) {
    run_flash_fwd<90,128,128,1,cutlass::float_e4m3_t,cutlass::bfloat16_t,
        true,false,false,true,false,false,false,false,true,false,FA3_PROBE_VCOL,false,false,N,2>(p,stream,resource);
}

void dispatch(int id, Flash_fwd_params& p, cudaStream_t stream, int* resource=nullptr) {
#ifdef FA3_PROBE_SINGLE_N
    launch<FA3_PROBE_SINGLE_N>(p, stream, resource);
#else
    if(id) launch<192>(p, stream, resource); else launch<224>(p, stream, resource);
#endif
}

extern "C" int prefill_init() {
    int sms=0, limit=0;
    cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0);
    cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,0);
    if(sms!=132) return -1;
#ifdef FA3_PROBE_SINGLE_N
    constexpr int variants = 1;
#else
    constexpr int variants = 2;
#endif
    for(int id=0;id<variants;++id) {
        Flash_fwd_params p{}; int bytes=0;
        dispatch(id,p,nullptr,&bytes);
#ifdef FA3_PROBE_SINGLE_N
        constexpr int tile_n = FA3_PROBE_SINGLE_N;
#else
        int tile_n = id ? 192 : 224;
#endif
        std::printf("fa3_resources tile_n=%d smem=%d limit=%d\n",tile_n,bytes,limit);
        if(bytes>limit) return -2;
    }
    return 0;
}

extern "C" int prefill_run(int id,int rows,int batch,int maxlen,void* q,void* k,void* v,void* cu,void* out,void* lse,void* metadata,void* fp8,void* scale,int scale_stride,void* stream_ptr) {
    if(id<0 || id>1 || batch<1 || batch>4 || rows<1) return -1;
    Flash_fwd_params p{};
    p.arch=90; p.num_sm=132; p.b=p.b_k=batch;
    p.total_q=p.total_k=rows; p.seqlen_q=p.seqlen_k=maxlen;
    p.seqlen_q_rounded=p.seqlen_k_rounded=(maxlen+127)/128*128;
    p.h=64; p.h_k=4; p.d=p.dv=p.d_rounded=p.dv_rounded=128;
    p.q_ptr=q; p.k_ptr=k; p.v_ptr=v; p.o_ptr=out; p.softmax_lse_ptr=lse;
    p.o_fp8_ptr=fp8; p.o_scale_ptr=static_cast<float*>(scale); p.o_scale_stride=scale_stride;
    p.q_row_stride=p.o_row_stride=64*128; p.k_row_stride=p.v_row_stride=4*128;
    p.q_head_stride=p.k_head_stride=p.v_head_stride=p.o_head_stride=128;
    p.v_dim_stride=1; p.cu_seqlens_q=p.cu_seqlens_k=static_cast<int*>(cu);
    p.is_e4m3=true; p.is_causal=true; p.pack_gqa=true;
    p.varlen_sort_batches=true; p.head_swizzle=true; p.num_splits=1;
    p.window_size_left=maxlen-1; p.window_size_right=0; p.scale_softmax=0.08838834764831845f;
    p.num_splits_dynamic_ptr=static_cast<int*>(metadata);
    p.num_m_blocks_ptr=p.num_splits_dynamic_ptr+4;
    p.varlen_batch_idx_ptr=p.num_splits_dynamic_ptr+8;
    p.num_nheads_in_l2_ptr=p.num_splits_dynamic_ptr+12;
    p.tile_count_semaphore=p.num_splits_dynamic_ptr+16;
    auto stream=static_cast<cudaStream_t>(stream_ptr);
    dispatch(id,p,stream);
    return int(cudaGetLastError());
}

#ifdef FA3_PROBE_RESOURCE_MAIN
int main() {
    Flash_fwd_params p{};
    int bytes = 0;
    launch<FA3_PROBE_RESOURCE_MAIN>(p, nullptr, &bytes);
    std::printf("probe_resource M=%d N=%d V_colmajor=%d smem=%d\n", FA3_PROBE_M ? FA3_PROBE_M : 128, FA3_PROBE_RESOURCE_MAIN, FA3_PROBE_VCOL, bytes);
    return 0;
}
#endif
