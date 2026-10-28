#include "verify.h"
#include "cuda_check.h"

namespace apac_fa3_verify {
void verify_n128_s3_split1_merge0(Flash_fwd_params&, cudaStream_t, int*);
void verify_n128_s3_split1_merge1(Flash_fwd_params&, cudaStream_t, int*);

// Fixed B40/Q4/GQA16/D128 diagnostic. Buffer order is shared with bench.py.
Flash_fwd_params probe_params(void** x, int columns) {
    Flash_fwd_params p{};
    cudaDeviceProp props{};
    CHECK_CUDA(cudaGetDeviceProperties(&props, 0));
    p.arch = 90; p.num_sm = props.multiProcessorCount;
    p.b = p.b_k = 40; p.total_q = 160; p.seqlen_q = 4;
    p.h = 64; p.h_k = 4; p.d = p.dv = p.d_rounded = p.dv_rounded = 128;
    p.seqlen_q_rounded = 128; p.num_splits = 3;
    p.q_ptr = x[0]; p.k_ptr = x[1]; p.v_ptr = x[2];
    p.page_table = static_cast<int*>(x[3]); p.seqused_k = static_cast<int*>(x[4]); p.cu_seqlens_q = static_cast<int*>(x[5]);
    p.o_ptr = x[6]; p.softmax_lse_ptr = x[7]; p.oaccum_ptr = x[8]; p.softmax_lseaccum_ptr = x[9];
    p.num_splits_dynamic_ptr = static_cast<int*>(x[10]);
    p.num_m_blocks_ptr = p.num_splits_dynamic_ptr + 40; p.varlen_batch_idx_ptr = p.num_splits_dynamic_ptr + 80;
    p.num_nheads_in_l2_ptr = p.num_splits_dynamic_ptr + 120; p.tile_count_semaphore = p.num_splits_dynamic_ptr + 160;
    p.o_scale_ptr = static_cast<float*>(x[11]); p.o_scale_stride = 160;
    p.q_row_stride = p.o_row_stride = 8192; p.q_head_stride = p.o_head_stride = 128;
    p.k_batch_stride = p.v_batch_stride = 32768; p.k_row_stride = p.v_row_stride = 512;
    p.k_head_stride = p.v_head_stride = 128; p.v_dim_stride = 1;
    p.page_size = 64; p.num_pages = 40 * columns; p.page_table_batch_stride = columns;
    p.seqlen_k = p.seqlen_k_rounded = columns * 64;
    p.window_size_left = p.seqlen_k - 1; p.window_size_right = 0; p.scale_softmax = 0.08838834764831845f;
    p.is_e4m3 = true; p.is_causal = true; p.pack_gqa = true;
    p.varlen_sort_batches = true; p.head_swizzle = true; p.skip_scheduler_metadata_computation = true;
    p.oaccum_split_stride = 64 * 160 * 128; p.oaccum_row_stride = 128; p.oaccum_head_stride = 160 * 128;
    p.lseaccum_split_stride = 64 * 160; p.lseaccum_head_stride = 160;
    p.verify_merge = {p.tile_count_semaphore + 1, static_cast<float*>(x[8]), static_cast<float*>(x[9]), x[6], static_cast<float*>(x[7]), static_cast<uint8_t*>(x[6]), p.o_scale_ptr, 160, 160};
    p.verify_merge.probe = {static_cast<int*>(x[12]), static_cast<int64_t*>(x[13]), static_cast<float*>(x[14]), static_cast<float*>(x[15]), static_cast<float*>(x[16]), static_cast<float*>(x[17]), static_cast<uint8_t*>(x[18]), static_cast<float*>(x[19])};
    return p;
}
}

extern "C" __attribute__((visibility("default"))) int decode_probe_prepare(void** x, int columns, void* stream) {
    auto p = apac_fa3_verify::probe_params(x, columns);
    prepare_varlen_num_blocks(p, static_cast<cudaStream_t>(stream), true, 64, 128, false);
    CHECK_CUDA(cudaMemsetAsync(p.tile_count_semaphore + 1, 0, 160 * sizeof(int), static_cast<cudaStream_t>(stream)));
    return int(cudaGetLastError());
}

extern "C" __attribute__((visibility("default"))) int decode_probe_run(int merge, void** x, int columns, void* raw_stream) {
    using namespace apac_fa3_verify;
    auto p = probe_params(x, columns);
    auto stream = static_cast<cudaStream_t>(raw_stream);
    if (merge) {
        CHECK_CUDA(cudaMemsetAsync(p.tile_count_semaphore, 0, sizeof(int), stream));
        verify_n128_s3_split1_merge1(p, stream, nullptr);
    } else {
        verify_n128_s3_split1_merge0(p, stream, nullptr);
        p.o_fp8_ptr = p.o_ptr;
        combine(p, stream);
    }
    return int(cudaGetLastError());
}
