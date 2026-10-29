#include <ATen/ATen.h>
#include <cmath>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include "verify.h"

namespace apac_fa3_verify {
struct Candidate { const char* name; int block_n, stages, splits; bool fp8, merge, uneven; Launch launch; };
#include "candidates.h"
static const Candidate& candidate(int64_t id) {
    TORCH_CHECK(id >= 0 && id < int64_t(sizeof(candidates) / sizeof(candidates[0])), "invalid FA3 verify candidate");
    return candidates[id];
}
static int splits(int64_t id) { return candidate(id).splits; }
static int block_n(int64_t id) { return candidate(id).block_n; }
static void tensor(const at::Tensor& t, const at::Tensor& ref, at::ScalarType dtype) {
    TORCH_CHECK(t.is_cuda() && t.device() == ref.device() && t.scalar_type() == dtype && t.is_contiguous(), "FA3 verify tensor dtype/device/stride mismatch");
}
static Flash_fwd_params base(const at::Tensor& lengths, const at::Tensor& cu, const at::Tensor& metadata, int64_t id) {
    int s = splits(id), b = lengths.numel(), rounded = (b + 3) / 4 * 4;
    tensor(lengths, lengths, at::kInt); tensor(cu, lengths, at::kInt); tensor(metadata, lengths, at::kInt);
    TORCH_CHECK(lengths.dim() == 1 && b >= 32 && b <= 48 && cu.dim() == 1 && cu.numel() == b + 1, "FA3 verify requires B32..48 and Q4");
    TORCH_CHECK(metadata.dim() == 1 && metadata.numel() == rounded * 4 + 1 + (candidate(id).merge ? b * 4 : 0), "FA3 verify metadata size");
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
    auto stream = at::cuda::getCurrentCUDAStream();
    if (candidate(id).uneven) { decode_prepare(p, stream); }
    else { prepare_varlen_num_blocks(p, stream, true, 64, block_n(id), false); }
    if (candidate(id).merge) { C10_CUDA_CHECK(cudaMemsetAsync(p.tile_count_semaphore + 1, 0, p.b * 4 * sizeof(int), stream)); }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void fwd_impl(const at::Tensor& scales, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& pages, const at::Tensor& lengths, const at::Tensor& cu, const at::Tensor& out, const at::Tensor& lse, const at::Tensor& partial, const at::Tensor& partial_lse, const at::Tensor& metadata, int64_t id, double scale) {
    c10::cuda::CUDAGuard guard(q.device());
    auto p = base(lengths, cu, metadata, id);
    tensor(q, lengths, at::ScalarType::Float8_e4m3fn); tensor(k, q, q.scalar_type()); tensor(v, q, q.scalar_type());
    tensor(pages, q, at::kInt); tensor(out, q, candidate(id).fp8 ? q.scalar_type() : at::kBFloat16); tensor(lse, q, at::kFloat);
    if (candidate(id).fp8) {
        TORCH_CHECK(scales.is_cuda() && scales.device()==q.device() && scales.scalar_type()==at::kFloat && scales.sizes()==at::IntArrayRef({p.total_q,64}) && scales.stride(0)==1 && scales.stride(1)>=p.total_q && scales.stride(1)%4==0, "FA3 verify scale layout");
        p.o_scale_ptr=scales.data_ptr<float>(); p.o_scale_stride=scales.stride(1);
    }
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
    p.softmax_lse_ptr = lse.data_ptr(); p.pagedkv_tma = false;
    // Split epilogues must never see the FP8 pointer; only combine consumes it.
    p.o_fp8_ptr = nullptr;
    if (p.num_splits > 1) {
        p.oaccum_ptr = partial.data_ptr(); p.softmax_lseaccum_ptr = partial_lse.data_ptr();
        p.oaccum_split_stride = int64_t(64) * p.total_q * 128; p.oaccum_row_stride = 128; p.oaccum_head_stride = int64_t(p.total_q) * 128;
        p.lseaccum_split_stride = 64 * p.total_q; p.lseaccum_head_stride = p.total_q;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    if (candidate(id).merge) {
        p.verify_merge = {p.tile_count_semaphore + 1, (const float*)p.oaccum_ptr, (const float*)p.softmax_lseaccum_ptr, p.o_ptr, (float*)p.softmax_lse_ptr, candidate(id).fp8 ? (uint8_t*)p.o_ptr : nullptr, p.o_scale_ptr, p.o_scale_stride, p.total_q};
        // Persistent producer CTAs can still fetch work after the last group
        // merges. Reset only at the next stream-ordered launch, never in merge.
        C10_CUDA_CHECK(cudaMemsetAsync(p.tile_count_semaphore, 0, sizeof(int), stream));
    }
    candidate(id).launch(p, stream, nullptr);
    if (p.num_splits > 1 && !candidate(id).merge) { p.o_fp8_ptr = candidate(id).fp8 ? out.data_ptr() : nullptr; combine(p, stream); }
    else if (!candidate(id).merge) { C10_CUDA_CHECK(cudaMemsetAsync(p.tile_count_semaphore, 0, sizeof(int), stream)); }
}
void fwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& pages, const at::Tensor& lengths, const at::Tensor& cu, const at::Tensor& out, const at::Tensor& lse, const at::Tensor& partial, const at::Tensor& partial_lse, const at::Tensor& metadata, int64_t id, double scale) {
    TORCH_CHECK(!candidate(id).fp8, "FP8 candidate requires scales");
    fwd_impl(at::Tensor(), q,k,v,pages,lengths,cu,out,lse,partial,partial_lse,metadata,id,scale);
}
void fwd_fp8(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& pages, const at::Tensor& lengths, const at::Tensor& cu, const at::Tensor& out, const at::Tensor& lse, const at::Tensor& partial, const at::Tensor& partial_lse, const at::Tensor& metadata, int64_t id, double scale, const at::Tensor& scales) {
    TORCH_CHECK(candidate(id).fp8, "BF16 candidate requires fwd_out");
    fwd_impl(scales,q,k,v,pages,lengths,cu,out,lse,partial,partial_lse,metadata,id,scale);
}
std::vector<int64_t> resources() {
    std::vector<int64_t> result;
    for (auto c : candidates) { Flash_fwd_params p{}; int bytes = 0; c.launch(p, nullptr, &bytes); result.push_back(bytes); }
    return result;
}
int64_t prefill_resources() {
    Flash_fwd_params p{}; int bytes=0; prefill(p, nullptr, &bytes); return bytes;
}
// Prefill candidates are independent of the Decode table above.
struct PrefillCandidate { const char* name; bool byte_image; bool varlen; Launch launch; };
#include "prefill-candidates.h"
static const PrefillCandidate& prefill_candidate(int64_t id) {
    TORCH_CHECK(id >= 0 && id < int64_t(sizeof(prefill_candidates)/sizeof(prefill_candidates[0])), "FA3 Prefill candidate index");
    return prefill_candidates[id];
}
std::vector<int64_t> prefill_variant_resources() {
    std::vector<int64_t> result;
    for (auto c : prefill_candidates) { Flash_fwd_params p{}; int bytes=0; c.launch(p,nullptr,&bytes); result.push_back(bytes); }
    return result;
}
void prefill_fwd_impl(bool fixed, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                 const at::Tensor& cu, const at::Tensor& out, const at::Tensor& scales,
                 const at::Tensor& lse, const at::Tensor& metadata, int64_t maxlen, double scale, int64_t variant=-1, const at::Tensor& image=at::Tensor()) {
    c10::cuda::CUDAGuard guard(q.device());
    tensor(q,q,at::ScalarType::Float8_e4m3fn); tensor(k,q,q.scalar_type()); tensor(v,q,q.scalar_type());
    tensor(cu,q,at::kInt); tensor(out,q,q.scalar_type()); tensor(lse,q,at::kFloat); tensor(metadata,q,at::kInt);
    int64_t rows=q.size(0), batch=cu.numel()-1;
    TORCH_CHECK(rows>0 && batch>=1 && batch<=4 && maxlen>0 && maxlen<=rows, "FA3 Prefill lengths");
    TORCH_CHECK(q.sizes()==at::IntArrayRef({rows,64,128}) && k.sizes()==at::IntArrayRef({rows,4,128}) && v.sizes()==k.sizes(), "FA3 Prefill QKV shape");
    TORCH_CHECK(out.dim()==3 && out.size(0)>=rows && out.size(1)==64 && out.size(2)==128, "FA3 Prefill output shape");
    TORCH_CHECK(scales.is_cuda() && scales.device()==q.device() && scales.scalar_type()==at::kFloat && scales.sizes()==at::IntArrayRef({out.size(0),64}) && scales.stride(0)==1 && scales.stride(1)>=out.size(0) && scales.stride(1)%4==0, "FA3 Prefill scale layout");
    TORCH_CHECK(cu.dim()==1 && metadata.numel()==17 && lse.sizes()==at::IntArrayRef({64,rows}), "FA3 Prefill metadata shape");
    TORCH_CHECK(std::isfinite(scale) && scale>0, "FA3 Prefill scale");
    auto props=at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major==9 && props->minor==0 && prefill_resources()<=props->sharedMemPerBlockOptin, "FA3 Prefill requires SM90 with sufficient shared memory");
    Flash_fwd_params p{};
    p.arch=90; p.num_sm=props->multiProcessorCount; p.b=p.b_k=batch;
    p.total_q=p.total_k=rows; p.seqlen_q=p.seqlen_k=maxlen;
    p.seqlen_q_rounded=p.seqlen_k_rounded=(maxlen+127)/128*128;
    p.h=64; p.h_k=4; p.d=p.dv=p.d_rounded=p.dv_rounded=128;
    p.q_ptr=q.data_ptr(); p.k_ptr=k.data_ptr(); p.v_ptr=v.data_ptr(); p.o_ptr=out.data_ptr();
    p.o_fp8_ptr=out.data_ptr(); p.o_scale_ptr=scales.data_ptr<float>(); p.o_scale_stride=scales.stride(1);
    p.softmax_lse_ptr=lse.data_ptr();
    p.q_row_stride=p.o_row_stride=8192; p.k_row_stride=p.v_row_stride=512;
    p.q_head_stride=p.k_head_stride=p.v_head_stride=p.o_head_stride=128; p.v_dim_stride=1;
    p.cu_seqlens_q=p.cu_seqlens_k=cu.data_ptr<int>();
    p.is_e4m3=true; p.is_causal=true; p.pack_gqa=true;
    p.varlen_sort_batches=true; p.head_swizzle=true; p.num_splits=1;
    p.window_size_left=maxlen-1; p.window_size_right=0; p.scale_softmax=scale;
    p.num_splits_dynamic_ptr=metadata.data_ptr<int>();
    p.num_m_blocks_ptr=p.num_splits_dynamic_ptr+4; p.varlen_batch_idx_ptr=p.num_splits_dynamic_ptr+8;
    p.num_nheads_in_l2_ptr=p.num_splits_dynamic_ptr+12; p.tile_count_semaphore=p.num_splits_dynamic_ptr+16;
    auto stream=at::cuda::getCurrentCUDAStream();
    if (fixed) {
        // Varlen candidates take 1..4 packed sequences; the host caller bounds each to 3584..4096.
        bool varlen = variant >= 0 && prefill_candidate(variant).varlen;
        TORCH_CHECK(props->multiProcessorCount==132 && (varlen ? maxlen>=3584 && maxlen<=4096 && rows>=batch*3584 && rows<=batch*maxlen : batch==1 && rows==maxlen && rows>=3584 && rows<=4096), "FA3 fixed Prefill requires 3584..4096-row sequences (one unless varlen) and 132 SMs");
        p.skip_scheduler_metadata_computation=true;
        p.softmax_lse_ptr=nullptr;
        // No preparation kernel: only the dynamic tile counter remains.
        C10_CUDA_CHECK(cudaMemsetAsync(p.tile_count_semaphore,0,sizeof(int),stream));
        if (variant >= 0) {
            auto c = prefill_candidate(variant);
            if (c.byte_image) {
                tensor(image,q,at::kByte);
                TORCH_CHECK(image.sizes()==at::IntArrayRef({4,(rows+223)/224,224*128}), "FA3 Prefill byte image shape");
                p.v_ptr=image.data_ptr();
            }
            int bytes=0; c.launch(p,nullptr,&bytes);
            TORCH_CHECK(bytes<=props->sharedMemPerBlockOptin, "FA3 Prefill candidate shared memory");
            c.launch(p,stream,nullptr);
        } else { prefill_fixed(p,stream,nullptr); }
    } else { prefill(p,stream,nullptr); }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void prefill_fwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& cu, const at::Tensor& out, const at::Tensor& scales, const at::Tensor& lse, const at::Tensor& metadata, int64_t maxlen, double scale) { prefill_fwd_impl(false,q,k,v,cu,out,scales,lse,metadata,maxlen,scale); }
void prefill_fixed_fwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& cu, const at::Tensor& out, const at::Tensor& scales, const at::Tensor& lse, const at::Tensor& metadata, int64_t maxlen, double scale) { prefill_fwd_impl(true,q,k,v,cu,out,scales,lse,metadata,maxlen,scale); }
void prefill_variant_fwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& image, const at::Tensor& cu, const at::Tensor& out, const at::Tensor& scales, const at::Tensor& lse, const at::Tensor& metadata, int64_t maxlen, double scale, int64_t variant) {
    prefill_candidate(variant);
    prefill_fwd_impl(true,q,k,v,cu,out,scales,lse,metadata,maxlen,scale,variant,image);
}
void prefill_dump_v(const at::Tensor& v, const at::Tensor& out) {
    c10::cuda::CUDAGuard guard(v.device());
    tensor(v,v,at::ScalarType::Float8_e4m3fn); tensor(out,v,at::kByte);
    TORCH_CHECK(v.dim()==3 && v.size(1)==4 && v.size(2)==128 && v.size(0)>=3584 && v.size(0)<=4096, "FA3 Prefill dump V shape");
    TORCH_CHECK(out.sizes()==at::IntArrayRef({4,(v.size(0)+223)/224,224*128}), "FA3 Prefill dump output shape");
    prefill_dump_v_launch(v.data_ptr(),out.data_ptr(),v.size(0),at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
int64_t prefill_fixed_resources() { Flash_fwd_params p{}; int bytes=0; prefill_fixed(p,nullptr,&bytes); return bytes; }

}
TORCH_LIBRARY(apac_fa3_verify, m) {
    m.def("prepare(Tensor lengths, Tensor cu_q, Tensor(a!) metadata, int candidate) -> ()");
    m.def("fwd_out(Tensor q, Tensor k, Tensor v, Tensor pages, Tensor lengths, Tensor cu_q, Tensor(a!) out, Tensor(b!) lse, Tensor(c!) partial, Tensor(d!) partial_lse, Tensor(e!) metadata, int candidate, float scale) -> ()");
    m.def("fwd_fp8_out(Tensor q, Tensor k, Tensor v, Tensor pages, Tensor lengths, Tensor cu_q, Tensor(a!) out, Tensor(b!) lse, Tensor(c!) partial, Tensor(d!) partial_lse, Tensor(e!) metadata, int candidate, float scale, Tensor(f!) scales) -> ()");
    m.def("resources() -> int[]", &apac_fa3_verify::resources);
    m.def("prefill_variant_resources() -> int[]", &apac_fa3_verify::prefill_variant_resources);
    m.def("prefill_dump_v(Tensor v, Tensor(a!) out) -> ()");
    m.def("prefill_variant_fwd_out(Tensor q, Tensor k, Tensor v, Tensor image, Tensor cu, Tensor(a!) out, Tensor(b!) scales, Tensor(c!) lse, Tensor(d!) metadata, int maxlen, float scale, int candidate) -> ()");
    m.def("prefill_fixed_resources() -> int", &apac_fa3_verify::prefill_fixed_resources);
    m.def("prefill_resources() -> int", &apac_fa3_verify::prefill_resources);
    m.def("prefill_fixed_fwd_out(Tensor q, Tensor k, Tensor v, Tensor cu, Tensor(a!) out, Tensor(b!) scales, Tensor(c!) lse, Tensor(d!) metadata, int maxlen, float scale) -> ()");
    m.def("prefill_fwd_out(Tensor q, Tensor k, Tensor v, Tensor cu, Tensor(a!) out, Tensor(b!) scales, Tensor(c!) lse, Tensor(d!) metadata, int maxlen, float scale) -> ()");
}
TORCH_LIBRARY_IMPL(apac_fa3_verify, CUDA, m) {
    m.impl("prepare", &apac_fa3_verify::prepare);
    m.impl("fwd_out", &apac_fa3_verify::fwd);
    m.impl("fwd_fp8_out", &apac_fa3_verify::fwd_fp8);
    m.impl("prefill_variant_fwd_out", &apac_fa3_verify::prefill_variant_fwd);
    m.impl("prefill_dump_v", &apac_fa3_verify::prefill_dump_v);
    m.impl("prefill_fixed_fwd_out", &apac_fa3_verify::prefill_fixed_fwd);
    m.impl("prefill_fwd_out", &apac_fa3_verify::prefill_fwd);
}
