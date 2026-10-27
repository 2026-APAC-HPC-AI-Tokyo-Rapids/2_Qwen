// Fixed serving Prefill specialization; the two DSOs differ only in exp macro.
#include <cstdio>
#include "flash_fwd_launch_template.h"

namespace {
void launch(Flash_fwd_params& p, cudaStream_t stream, int* resource=nullptr) {
    run_flash_fwd<90,128,128,1,cutlass::float_e4m3_t,cutlass::bfloat16_t,
        true,false,false,true,false,false,false,false,true,false,false,false,false,224,2,true,0>(p,stream,resource);
}
}

extern "C" int prefill_init() {
    int sms=0, limit=0;
    auto status=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0);
    if(status!=cudaSuccess) return int(status);
    status=cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,0);
    if(status!=cudaSuccess) return int(status);
    if(sms!=132) return -1;
    Flash_fwd_params p{};
    int bytes=0;
    launch(p,nullptr,&bytes);
    std::printf("fa3_exp_resources tile_n=224 fixed=1 variant=0 smem=%d limit=%d\n",bytes,limit);
    return bytes>limit ? -2 : 0;
}

extern "C" int prefill_run(int id,int rows,int batch,int maxlen,void* q,void* k,void* v,void* cu,void* out,void* lse,void* metadata,void* fp8,void* scale,int scale_stride,void* stream_ptr) {
    if(id!=0 || batch!=1 || rows!=maxlen || rows<3584 || rows>4096) return -1;
    if(!q || !k || !v || !cu || !out || !metadata || (fp8 && (!scale || scale_stride<rows || scale_stride%4))) return -2;
    (void)lse;  // FixedPrefill never stores LSE, matching the serving wrapper.
    Flash_fwd_params p{};
    p.arch=90; p.num_sm=132; p.b=p.b_k=1;
    p.total_q=p.total_k=p.seqlen_q=p.seqlen_k=rows;
    p.seqlen_q_rounded=p.seqlen_k_rounded=(rows+127)/128*128;
    p.h=64; p.h_k=4; p.d=p.dv=p.d_rounded=p.dv_rounded=128;
    p.q_ptr=q; p.k_ptr=k; p.v_ptr=v; p.o_ptr=out;
    p.o_fp8_ptr=fp8; p.o_scale_ptr=static_cast<float*>(scale); p.o_scale_stride=scale_stride;
    p.softmax_lse_ptr=nullptr;
    p.q_row_stride=p.o_row_stride=8192; p.k_row_stride=p.v_row_stride=512;
    p.q_head_stride=p.k_head_stride=p.v_head_stride=p.o_head_stride=128;
    p.v_dim_stride=1; p.cu_seqlens_q=p.cu_seqlens_k=static_cast<int*>(cu);
    p.is_e4m3=true; p.is_causal=true; p.pack_gqa=true;
    p.varlen_sort_batches=true; p.head_swizzle=true; p.num_splits=1;
    p.window_size_left=rows-1; p.window_size_right=0; p.scale_softmax=0.08838834764831845f;
    p.num_splits_dynamic_ptr=static_cast<int*>(metadata);
    p.num_m_blocks_ptr=p.num_splits_dynamic_ptr+4;
    p.varlen_batch_idx_ptr=p.num_splits_dynamic_ptr+8;
    p.num_nheads_in_l2_ptr=p.num_splits_dynamic_ptr+12;
    p.tile_count_semaphore=p.num_splits_dynamic_ptr+16;
    p.skip_scheduler_metadata_computation=true;
    auto stream=static_cast<cudaStream_t>(stream_ptr);
    auto status=cudaMemsetAsync(p.tile_count_semaphore,0,sizeof(int),stream);
    if(status!=cudaSuccess) return int(status);
    launch(p,stream);
    return int(cudaGetLastError());
}

#ifdef FA3_EXP_RESOURCE_MAIN
int main() {
    Flash_fwd_params p{};
    int bytes=0;
    launch(p,nullptr,&bytes);
    std::printf("fa3_exp_resource M=128 N=224 fixed=1 variant=0 smem=%d\n",bytes);
    return bytes>0 && bytes<=227*1024 ? 0 : 1;
}
#endif
