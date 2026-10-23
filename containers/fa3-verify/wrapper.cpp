#include <ATen/ATen.h>
#include <cmath>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include "verify.h"

namespace apac_fa3_verify {
static Launch launches[] = {control96, t64s2, t64s3, p128s2, t64split, t64split};
static int splits(int64_t id) { TORCH_CHECK(id >= 0 && id < 6, "invalid FA3 verify candidate"); return id < 4 ? 1 : id - 2; }
static int block_n(int64_t id) { return id == 0 ? 96 : id == 3 ? 128 : 64; }
static void tensor(const at::Tensor& t, const at::Tensor& ref, at::ScalarType dtype) {
    TORCH_CHECK(t.is_cuda() && t.device() == ref.device() && t.scalar_type() == dtype && t.is_contiguous(), "FA3 verify tensor dtype/device/stride mismatch");
}
static Flash_fwd_params base(const at::Tensor& lengths, const at::Tensor& cu, const at::Tensor& metadata, int64_t id) {
    int s = splits(id), b = lengths.numel(), rounded = (b + 3) / 4 * 4;
    tensor(lengths, lengths, at::kInt); tensor(cu, lengths, at::kInt); tensor(metadata, lengths, at::kInt);
    TORCH_CHECK(lengths.dim() == 1 && b >= 32 && b <= 36 && cu.dim() == 1 && cu.numel() == b + 1, "FA3 verify requires B32..36 and Q4");
    TORCH_CHECK(metadata.dim() == 1 && metadata.numel() == rounded * 4 + 1, "FA3 verify metadata size");
    Flash_fwd_params p{};
    p.b = b; p.b_k = b; p.seqlen_q = 4; p.total_q = 4 * b; p.h = 64; p.h_k = 4;
    p.d = p.dv = p.d_rounded = p.dv_rounded = 128; p.seqlen_q_rounded = 128;
    p.cu_seqlens_q = cu.data_ptr<int>(); p.seqused_k = lengths.data_ptr<int>();
    p.num_splits = s; p.is_e4m3 = true; p.is_bf16 = false; p.is_causal = true;
    p.pack_gqa = true; p.varlen_sort_batches = true; p.head_swizzle = true;
    p.num_splits_dynamic_ptr = metadata.data_ptr<int>(); p.num_m_blocks_ptr = p.num_splits_dynamic_ptr + rounded;
    p.varlen_batch_idx_ptr = p.num_splits_dynamic_ptr + rounded * 2; p.num_nheads_in_l2_ptr = p.num_splits_dynamic_ptr + rounded * 3;
    p.tile_count_semaphore = p.num_splits_dynamic_ptr + rounded * 4;
    p.skip_scheduler_metadata_computation = true; p.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 9, "FA3 verify requires SM90");
    return p;
}
void prepare(const at::Tensor& lengths, const at::Tensor& cu, const at::Tensor& metadata, int64_t id) {
    c10::cuda::CUDAGuard guard(lengths.device());
    auto p = base(lengths, cu, metadata, id);
    prepare_varlen_num_blocks(p, at::cuda::getCurrentCUDAStream(), true, 64, block_n(id), false);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void fwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& pages, const at::Tensor& lengths, const at::Tensor& cu, const at::Tensor& out, const at::Tensor& lse, const at::Tensor& partial, const at::Tensor& partial_lse, const at::Tensor& metadata, int64_t id, double scale) {
    c10::cuda::CUDAGuard guard(q.device());
    auto p = base(lengths, cu, metadata, id);
    tensor(q, lengths, at::ScalarType::Float8_e4m3fn); tensor(k, q, q.scalar_type()); tensor(v, q, q.scalar_type());
    tensor(pages, q, at::kInt); tensor(out, q, at::kBFloat16); tensor(lse, q, at::kFloat);
    tensor(partial, q, at::kFloat); tensor(partial_lse, q, at::kFloat);
    TORCH_CHECK(q.sizes() == at::IntArrayRef({p.total_q, 64, 128}) && out.sizes() == q.sizes(), "FA3 verify Q/O shape");
    TORCH_CHECK(k.dim() == 4 && k.size(1) == 64 && k.size(2) == 4 && k.size(3) == 128 && v.sizes() == k.sizes(), "FA3 verify paged KV shape");
    TORCH_CHECK(pages.dim() == 2 && pages.size(0) == p.b && pages.size(1) > 0 && lse.sizes() == at::IntArrayRef({64, p.total_q}), "FA3 verify pages/LSE shape");
    TORCH_CHECK(partial.numel() == (p.num_splits > 1 ? int64_t(p.num_splits) * 64 * p.total_q * 128 : 0) && partial_lse.numel() == (p.num_splits > 1 ? int64_t(p.num_splits) * 64 * p.total_q : 0), "FA3 verify FP32 workspace size");
    TORCH_CHECK(std::isfinite(scale) && scale > 0, "FA3 verify scale");
    p.q_ptr = q.data_ptr(); p.k_ptr = k.data_ptr(); p.v_ptr = v.data_ptr(); p.o_ptr = out.data_ptr();
    p.q_row_stride = q.stride(0); p.q_head_stride = q.stride(1);
    p.k_batch_stride = k.stride(0); p.k_row_stride = k.stride(1); p.k_head_stride = k.stride(2);
    p.v_batch_stride = v.stride(0); p.v_row_stride = v.stride(1); p.v_head_stride = v.stride(2); p.v_dim_stride = v.stride(3);
    p.o_row_stride = out.stride(0); p.o_head_stride = out.stride(1);
    p.page_table = pages.data_ptr<int>(); p.page_table_batch_stride = pages.stride(0); p.page_size = 64; p.num_pages = k.size(0);
    p.seqlen_k = pages.size(1) * 64; p.seqlen_k_rounded = p.seqlen_k;
    p.window_size_left = p.seqlen_k - 1; p.window_size_right = 0; p.scale_softmax = scale;
    p.softmax_lse_ptr = lse.data_ptr(); p.pagedkv_tma = block_n(id) == 64;
    if (p.num_splits > 1) {
        p.oaccum_ptr = partial.data_ptr(); p.softmax_lseaccum_ptr = partial_lse.data_ptr();
        p.oaccum_split_stride = int64_t(64) * p.total_q * 128; p.oaccum_row_stride = 128; p.oaccum_head_stride = int64_t(p.total_q) * 128;
        p.lseaccum_split_stride = 64 * p.total_q; p.lseaccum_head_stride = p.total_q;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    launches[id](p, stream, nullptr);
    if (p.num_splits > 1) { combine(p, stream); }
    else { C10_CUDA_CHECK(cudaMemsetAsync(p.tile_count_semaphore, 0, sizeof(int), stream)); }
}
std::vector<int64_t> resources() {
    std::vector<int64_t> result;
    for (auto launch : launches) { Flash_fwd_params p{}; int bytes = 0; launch(p, nullptr, &bytes); result.push_back(bytes); }
    return result;
}
}
TORCH_LIBRARY(apac_fa3_verify, m) {
    m.def("prepare(Tensor lengths, Tensor cu_q, Tensor(a!) metadata, int candidate) -> ()");
    m.def("fwd_out(Tensor q, Tensor k, Tensor v, Tensor pages, Tensor lengths, Tensor cu_q, Tensor(a!) out, Tensor(b!) lse, Tensor(c!) partial, Tensor(d!) partial_lse, Tensor(e!) metadata, int candidate, float scale) -> ()");
    m.def("resources() -> int[]", &apac_fa3_verify::resources);
}
TORCH_LIBRARY_IMPL(apac_fa3_verify, CUDA, m) {
    m.impl("prepare", &apac_fa3_verify::prepare);
    m.impl("fwd_out", &apac_fa3_verify::fwd);
}
