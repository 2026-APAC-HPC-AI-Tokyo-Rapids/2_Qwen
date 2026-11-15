// Serving fixed Prefill FA3 rows (N192 varlen) for the one-GPU paired probe.
// pf-variants.h defines PF_VARIANTS, the PrefillVariant values compiled into this library.
#include <cstdio>
#include "pf-variants.h"
#include <cstring>
#include "flash_fwd_launch_template.h"

template<int Variant>
void launch(Flash_fwd_params& p, cudaStream_t stream, int* resource) {
    run_flash_fwd<90,128,128,1,cutlass::float_e4m3_t,cutlass::bfloat16_t,
        true,false,false,true,false,false,false,false,true,false,false,false,false,96,2,true,Variant>(p,stream,resource);
}

template<int... Vs> struct Table {
    static int count() { return sizeof...(Vs); }
    static void run(int id, Flash_fwd_params& p, cudaStream_t stream, int* resource) {
        int i = 0;
        ((i++ == id ? launch<Vs>(p, stream, resource) : void()), ...);
    }
};
using Variants = Table<PF_VARIANTS>;

extern "C" int pf_count() { return Variants::count(); }

extern "C" int pf_resources(int id) {
    Flash_fwd_params p{};
    int bytes = 0;
    Variants::run(id, p, nullptr, &bytes);
    return bytes;
}

// Same parameters as prefill_fwd_impl(fixed=true) in kernels/fa3-verify/wrapper.cpp.
extern "C" int pf_run(int id, int rows, int batch, int maxlen, void* q, void* k, void* v, void* cu, void* out, void* scales, int scale_stride, void* metadata, float scale, void* stream_ptr) {
    if (id < 0 || id >= Variants::count() || batch < 1 || batch > 4) return -1;
    Flash_fwd_params p{};
    p.arch = 90; p.num_sm = 132; p.b = p.b_k = batch;
    p.total_q = p.total_k = rows; p.seqlen_q = p.seqlen_k = maxlen;
    p.seqlen_q_rounded = p.seqlen_k_rounded = (maxlen + 127) / 128 * 128;
    p.h = 64; p.h_k = 4; p.d = p.dv = p.d_rounded = p.dv_rounded = 128;
    p.q_ptr = q; p.k_ptr = k; p.v_ptr = v; p.o_ptr = out;
    p.o_fp8_ptr = out; p.o_scale_ptr = static_cast<float*>(scales); p.o_scale_stride = scale_stride;
    p.softmax_lse_ptr = nullptr;
    p.q_row_stride = p.o_row_stride = 8192; p.k_row_stride = p.v_row_stride = 512;
    p.q_head_stride = p.k_head_stride = p.v_head_stride = p.o_head_stride = 128; p.v_dim_stride = 1;
    p.cu_seqlens_q = p.cu_seqlens_k = static_cast<int*>(cu);
    p.is_e4m3 = true; p.is_causal = true; p.pack_gqa = true;
    p.varlen_sort_batches = true; p.head_swizzle = true; p.num_splits = 1;
    p.window_size_left = maxlen - 1; p.window_size_right = 0; p.scale_softmax = scale;
    p.num_splits_dynamic_ptr = static_cast<int*>(metadata);
    p.num_m_blocks_ptr = p.num_splits_dynamic_ptr + 4; p.varlen_batch_idx_ptr = p.num_splits_dynamic_ptr + 8;
    p.num_nheads_in_l2_ptr = p.num_splits_dynamic_ptr + 12; p.tile_count_semaphore = p.num_splits_dynamic_ptr + 16;
    p.skip_scheduler_metadata_computation = true;
    auto stream = static_cast<cudaStream_t>(stream_ptr);
    cudaMemsetAsync(p.tile_count_semaphore, 0, sizeof(int), stream);
    Variants::run(id, p, stream, nullptr);
    return int(cudaGetLastError());
}

#ifdef PF_STAMP
extern "C" int pf_stamp_reset() {
    unsigned int zero = 0;
    void* buffer = nullptr;
    cudaError_t e = cudaGetSymbolAddress(&buffer, apac_stamp_buf);
    if (e == cudaSuccess) e = cudaMemset(buffer, 0, sizeof(apac_stamp_buf));
    if (e != cudaSuccess) return int(e);
    return int(cudaMemcpyToSymbol(apac_stamp_count, &zero, sizeof(zero)));
}
extern "C" int pf_stamp_read(void* host, unsigned int* count) {
    cudaError_t e = cudaMemcpyFromSymbol(count, apac_stamp_count, sizeof(unsigned int));
    if (e != cudaSuccess) return int(e);
    return int(cudaMemcpyFromSymbol(host, apac_stamp_buf, sizeof(apac_stamp_buf)));
}
#endif

#ifdef PF_RESOURCE_MAIN
int main() {
    for (int i = 0; i < Variants::count(); ++i) std::printf("pf_resource index=%d smem=%d\n", i, pf_resources(i));
    return 0;
}
#endif
