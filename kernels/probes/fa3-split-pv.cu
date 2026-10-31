// Research-only SM90a Prefill kernel. Not selected by the serving dispatcher.
// QK/softmax own two warpgroups; a third owns PV and its output accumulators.
// A two-slot FP8 P mailbox replaces the cross-iteration RS operand dependency.
// The Q/K/V representation, causal mask, FP32 softmax and FP8 rounding are kept.
#include "verify.h"
#include "utils.h"
#include "cutlass/arch/reg_reconfig.h"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "softmax.h"
#include "cuda_check.h"
#include <stdexcept>

namespace apac_split_pv {
using namespace cute;
using E = cutlass::float_e4m3_t;
using ShapeQK = Shape<_128, _192, _128>;
using ShapePV = Shape<_128, _128, _192>;
using ML = flash::CollectiveMainloopFwdSm90<2, Shape<_1, _1, _1>, ShapeQK, 128,
    E, float, cutlass::arch::Sm90, true, false, false, true, false, false,
    false, false, true, true, true, false, false, cutlass::bfloat16_t>;
using PV = decltype(make_tiled_mma(GMMA::ss_op_selector<E, E, float, ShapePV,
    GMMA::Major::K, GMMA::Major::K>(), Layout<Shape<_1, _1, _1>>{}));
using PLayout = decltype(tile_to_shape(ML::SmemLayoutAtomP{}, Shape<_128, _192, _2>{}));
using PPipe = cutlass::PipelineAsync<2>;

struct Shared : aligned_struct<1024, _1> {
    struct {
        ML::TensorStorage mainloop;
        array_aligned<E, cosize_v<PLayout>, 1024> p;
        float alpha[2][128];
        float final_scale[128];
    } tensors;
    struct {
        alignas(16) cutlass::arch::ClusterBarrier barrier_Q;
        alignas(16) cutlass::arch::ClusterBarrier barrier_Qv;
        alignas(16) cutlass::arch::ClusterBarrier barrier_O;
        alignas(16) ML::MainloopPipelineK::SharedStorage pipeline_k;
        alignas(16) ML::MainloopPipelineV::SharedStorage pipeline_v;
        alignas(16) ML::MainloopPipelineVt::SharedStorage pipeline_vt;
        alignas(16) ML::MainloopPipelineMask::SharedStorage pipeline_mask;
        alignas(16) PPipe::SharedStorage p;
    } pipelines;
};
static_assert(sizeof(Shared) <= 227 * 1024, "P mailbox must fit beside the unchanged KV stages");
static_assert(128 * (24 + 2 * 152 + 176) <= 65536, "role register budgets exceed one SM");

#ifndef APAC_SPLIT_PV_LAYOUT_ONLY
struct Params {
    ML::Params mainloop;
    uint8_t* out;
    float* scale;
    int64_t scale_stride;
    int max_m_blocks;
};

// This is the deployed direct-O mapping with an extra M-wave loop: one PV WG
// owns 128 rows rather than 64. It does not stage another full O tensor.
template<class Acc>
CUTLASS_DEVICE void store_output(Params const& p, Acc const& acc, int tid,
                                 int m_block, int kv_head, int len, int offset) {
    auto values = make_tensor_like<float>(group_modes<1, 3>(acc));
    CUTE_UNROLL
    for (int i = 0; i < size(values); ++i) {
        uint32_t bits;
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(bits)
            : "f"(group_modes<1, 3>(acc)(i)), "f"(0.f));
        values(i) = __uint_as_float(bits);
    }
    auto coords = group_modes<1, 3>(PV{}.get_thread_slice(tid).partition_C(
        make_identity_tensor(select<0, 1>(ShapePV{}))));
    cutlass::NumericArrayConverter<E, float, 4> convert;
    CUTE_UNROLL
    for (int wave = 0; wave < size<1>(values); ++wave) {
        int const col = 2 * get<1>(coords(_0{}, wave));
        CUTE_UNROLL
        for (int r = 0; r < 2; ++r) {
            float amax = 0.f;
            CUTE_UNROLL
            for (int n = 0; n < 16; ++n) {
                CUTE_UNROLL
                for (int c = 0; c < 2; ++c) {
                    amax = fmaxf(amax, fabsf(values(make_coord(c, r, n), wave)));
                }
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
            amax = fmaxf(amax, 1e-10f);
            float const qs = 448.f / amax;
            int const row = m_block * 128 + get<0>(coords(make_coord(_0{}, r, _0{}), wave));
            if (row < len * 16) {
                int const token = offset + row / 16;
                int const head = kv_head * 16 + row % 16;
                uint8_t* dst = p.out + (int64_t(token) * 64 + head) * 128 + col;
                CUTE_UNROLL
                for (int i = 0; i < 8; ++i) {
                    cutlass::Array<float, 4> x;
                    x[0] = values(make_coord(0, r, 2 * i), wave) * qs;
                    x[1] = values(make_coord(1, r, 2 * i), wave) * qs;
                    x[2] = values(make_coord(0, r, 2 * i + 1), wave) * qs;
                    x[3] = values(make_coord(1, r, 2 * i + 1), wave) * qs;
                    auto q = convert(x);
                    *reinterpret_cast<uint32_t*>(dst + 16 * i) = reinterpret_cast<uint32_t const&>(q);
                }
                if (col == 0) { p.scale[int64_t(head) * p.scale_stride + token] = amax * (1.f / 448.f); }
            }
        }
    }
}

__global__ __launch_bounds__(512, 1) void kernel(__grid_constant__ Params const p) {
    extern __shared__ __align__(1024) char storage[];
    auto& s = *reinterpret_cast<Shared*>(storage);
    int const wg = __shfl_sync(0xffffffff, int(threadIdx.x) / 128, 0);
    int const tid = int(threadIdx.x) % 128;
    int const bidb = blockIdx.z;
    int const head = blockIdx.y;
    ML::SeqlenInfo_t info{bidb, get<0>(p.mainloop.shape_Q), get<0>(p.mainloop.shape_K),
        0, p.mainloop.cu_seqlens_q, p.mainloop.cu_seqlens_k, nullptr,
        nullptr, nullptr, nullptr, nullptr};
    int const m_block = p.max_m_blocks - 1 - int(blockIdx.x);
    if (m_block * 8 >= info.seqlen_q) { return; }
    auto coord = make_tuple(m_block, head, bidb, 0);
    auto bounds = ML::BlockMN_t::get_n_block_min_max(
        info, m_block, bidb, 0, 1, -1, -1, p.mainloop.attention_chunk_divmod,
        p.mainloop.qhead_per_khead_divmod);
    int const n_min = get<0>(bounds);
    int const n_max = get<1>(bounds);
    if (threadIdx.x == 0) {
        ML::prefetch_tma_descriptors(p.mainloop);
        s.pipelines.barrier_Q.init(128);
        s.pipelines.barrier_O.init(128);
    }
    ML::MainloopPipelineK::Params kp;
    kp.role = wg == 0 ? ML::MainloopPipelineK::ThreadCategory::Producer : ML::MainloopPipelineK::ThreadCategory::Consumer;
    kp.transaction_bytes = ML::TmaTransactionBytesK;
    kp.is_leader = tid == 0;
    kp.num_consumers = 256;
    ML::MainloopPipelineK pk(s.pipelines.pipeline_k, kp, Shape<_1, _1, _1>{});
    auto vtp = kp;
    vtp.transaction_bytes = ML::TmaTransactionBytesV;
    vtp.num_consumers = 128;
    ML::MainloopPipelineVt pvt(s.pipelines.pipeline_vt, vtp, Shape<_1, _1, _1>{});
    ML::MainloopPipelineV::Params vp;
    vp.role = wg == 0 ? ML::MainloopPipelineV::ThreadCategory::Producer : ML::MainloopPipelineV::ThreadCategory::Consumer;
    vp.producer_arv_count = 128;
    vp.consumer_arv_count = 128;
    ML::MainloopPipelineV pv(s.pipelines.pipeline_v, vp);
    PPipe::Params pp;
    pp.role = wg == 1 || wg == 2 ? PPipe::ThreadCategory::Producer : PPipe::ThreadCategory::Consumer;
    pp.producer_arv_count = 256;
    pp.consumer_arv_count = 128;
    PPipe mailbox(s.pipelines.p, pp);
    ML::MainloopPipelineMask::Params mp;
    mp.role = ML::MainloopPipelineMask::ThreadCategory::Producer;
    mp.producer_arv_count = 128;
    mp.consumer_arv_count = 256;
    ML::MainloopPipelineMask mask_pipe(s.pipelines.pipeline_mask, mp);
    __syncthreads();

    if (wg == 0) {
        cutlass::arch::warpgroup_reg_dealloc<24>();
        auto write = cutlass::make_producer_start_state<ML::MainloopPipelineK>();
        int work = 0;
        auto no_prefetch = []() {};
        ML{}.load(p.mainloop, pk, pv, pvt, mask_pipe, write, s, no_prefetch, info, coord, work);
        ML{}.load_tail(pk, pv, pvt, mask_pipe, write, s, work);
    } else if (wg < 3) {
        cutlass::arch::warpgroup_reg_alloc<152>();
        int const qtid = int(threadIdx.x) - 128;
        ML::TiledMmaQK qk;
        auto qk_wg = qk.get_slice((wg - 1) * 128);
        auto sq = make_tensor(make_smem_ptr(s.tensors.mainloop.smem_q.data()), ML::SmemLayoutQ{});
        auto sk = make_tensor(make_smem_ptr(s.tensors.mainloop.smem_k.data()), ML::SmemLayoutK{});
        auto q = qk_wg.partition_fragment_A(sq);
        auto k = qk_wg.partition_fragment_B(sk);
        auto sp = make_tensor(make_smem_ptr(s.tensors.p.data()), PLayout{});
        // Use the RS operand's own coordinate mapping, rather than guessing
        // the cross-lane FP8 layout from the QK accumulator's layout.
        auto rs_thread = ML::TiledMmaPV{}.get_thread_slice(qtid);
        auto a_coords = qk.get_thread_slice(qtid).partition_C(make_identity_tensor(select<0, 1>(ShapeQK{})));
        auto row_coords = make_tensor(a_coords.data(), flash::convert_layout_acc_rowcol(a_coords.layout()))(_, _0{});
        float const qscale = p.mainloop.ptr_q_descale ? p.mainloop.ptr_q_descale[bidb * get<0>(p.mainloop.stride_q_descale) + head * get<1>(p.mainloop.stride_q_descale)] : 1.f;
        float const kscale = p.mainloop.ptr_k_descale ? p.mainloop.ptr_k_descale[bidb * get<0>(p.mainloop.stride_k_descale) + head * get<1>(p.mainloop.stride_k_descale)] : 1.f;
        float const vscale = p.mainloop.ptr_v_descale ? p.mainloop.ptr_v_descale[bidb * get<0>(p.mainloop.stride_v_descale) + head * get<1>(p.mainloop.stride_v_descale)] : 1.f;
        flash::Softmax<2, 8> softmax(p.mainloop.softmax_scale_log2 * (qscale * kscale));
        flash::Mask<128, 192, true, ML::TiledMmaQK> mask(qtid, info.seqlen_q, info.seqlen_k,
            -1, -1, 0, p.mainloop.attention_chunk_divmod, p.mainloop.qhead_per_khead_divmod);
        cutlass::arch::NamedBarrier::arrive(384, uint32_t(flash::FwdNamedBarriers::QueryEmpty));
        s.pipelines.barrier_Q.wait(0);
        ML::PipelineState read;
        auto write = cutlass::make_producer_start_state<PPipe>();
        auto step = [&](int n, auto first) {
            constexpr bool First = decltype(first)::value;
            auto scores = partition_fragment_C(qk, select<0, 1>(ShapeQK{}));
            pk.consumer_wait(read);
            flash::gemm<true, -1>(qk, q, k(_, _, _, read.index()), scores);
            warpgroup_wait<0>();
            pk.consumer_release(read);
            if constexpr (First) { mask.template apply<true, true, false>(scores, m_block, n); }
            auto alpha = softmax.template max_get_scale<First, First>(scores);
            softmax.template online_softmax<First, First>(scores);
            flash::permute_Cregs_fp8(scores);
            auto pa = make_tensor(scores.data(), flash::convert_layout_acc_Aregs<ML::TiledMmaPV>(scores.layout()));
            auto packed = make_tensor_like<E>(pa);
            flash::convert_type_out(pa, packed);
            mailbox.producer_acquire(write);
            auto dest = rs_thread.partition_A(sp(_, _, write.index()));
            auto words = recast<uint32_t>(packed);
            CUTE_UNROLL
            for (int i = 0; i < size(words); ++i) {
                uint32_t address = cast_smem_ptr_to_uint(&dest(4 * i));
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(address), "r"(words(i)) : "memory");
            }
            CUTE_UNROLL
            for (int r = 0; r < size(alpha); ++r) {
                if (get<1>(row_coords(_0{})) == 0) { s.tensors.alpha[write.index()][get<0>(row_coords(r))] = alpha(r); }
            }
            if (n == n_min) {
                auto final = softmax.finalize(vscale);
                CUTE_UNROLL
                for (int r = 0; r < size(final); ++r) {
                    if (get<1>(row_coords(_0{})) == 0) { s.tensors.final_scale[get<0>(row_coords(r))] = final(r); }
                }
            }
            cutlass::arch::fence_view_async_shared();
            mailbox.producer_commit(write);
            ++read;
            ++write;
        };
        step(n_max - 1, true_type{});
        for (int n = n_max - 2; n >= n_min; --n) { step(n, false_type{}); }
        mailbox.producer_tail(write);
    } else {
        cutlass::arch::warpgroup_reg_alloc<176>();
        PV mma;
        auto wg_mma = mma.get_slice(0);
        auto sv = make_tensor(make_smem_ptr(s.tensors.mainloop.smem_v.data()), ML::SmemLayoutVtMma{});
        auto sp = make_tensor(make_smem_ptr(s.tensors.p.data()), PLayout{});
        auto v = wg_mma.partition_fragment_B(sv);
        auto a = wg_mma.partition_fragment_A(sp);
        auto out = partition_fragment_C(mma, select<0, 1>(ShapePV{}));
        clear(out);
        auto coords = mma.get_thread_slice(tid).partition_C(make_identity_tensor(select<0, 1>(ShapePV{})));
        auto rows = make_tensor(coords.data(), flash::convert_layout_acc_rowcol(coords.layout()))(_, _0{});
        auto o_rc = make_tensor(out.data(), flash::convert_layout_acc_rowcol(out.layout()));
        ML::PipelineState read;
        for (int n = n_max - 1; n >= n_min; --n) {
            mailbox.consumer_wait(read);
            pv.consumer_wait(read);
            CUTE_UNROLL
            for (int r = 0; r < size<0>(o_rc); ++r) {
                float const alpha = s.tensors.alpha[read.index()][get<0>(rows(r))];
                CUTE_UNROLL
                for (int c = 0; c < size<1>(o_rc); ++c) { o_rc(r, c) *= alpha; }
            }
            flash::gemm<false, -1>(mma, a(_, _, _, read.index()), v(_, _, _, read.index()), out);
            warpgroup_wait<0>();
            pv.consumer_release(read);
            mailbox.consumer_release(read);
            ++read;
        }
        CUTE_UNROLL
        for (int r = 0; r < size<0>(o_rc); ++r) {
            float const f = s.tensors.final_scale[get<0>(rows(r))];
            CUTE_UNROLL
            for (int c = 0; c < size<1>(o_rc); ++c) { o_rc(r, c) *= f; }
        }
        flash::permute_output_fp8(out);
        store_output(p, out, tid, m_block, head, info.seqlen_q, info.offset_q);
        s.pipelines.barrier_O.arrive();
    }
}


// Match the fixed-prefill probe ABI; tensor maps are built from the actual
// packed request buffer, while each CTA uses its own cu_seqlens offsets.
extern "C" int pf_count() { return 1; }
extern "C" int pf_resources(int) { return sizeof(Shared); }
extern "C" int pf_init() {
    return int(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared)));
}
extern "C" int pf_run(int id, int rows, int batch, int maxlen,
                     void* q, void* k, void* v, void* cu, void* out, void* scales,
                     int scale_stride, void*, float scale, void* stream_ptr) {
    if (id != 0 || rows < 1 || batch < 1 || batch > 4 || maxlen < 1 || scale_stride < rows) { return -1; }
    ML::Arguments args{
        static_cast<E const*>(q), {rows, 128, 64, 1}, {8192, _1{}, 128, 0},
        static_cast<E*>(k), {rows, 128, 4, 1}, {512, _1{}, 128, 0},
        static_cast<E*>(v), 128, {512, _1{}, 128, 0},
        nullptr, {0, 128, 4, 1}, {512, _1{}, 128, 0},
        nullptr, {512, _1{}, 128, 0}, nullptr, {8192, _1{}, 128, 0},
        nullptr, {0, 0}, {0, _1{}}, nullptr, {0, _1{}}, false,
        nullptr, {batch, 0}, {0, _1{}}, scale,
        nullptr, nullptr, nullptr, {0, 0}, {0, 0}, {0, 0},
        maxlen - 1, 0, 0, 0.f, 1, nullptr,
        static_cast<int*>(cu), static_cast<int*>(cu), nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr};
    Params params{ML::to_underlying_arguments(args), static_cast<uint8_t*>(out),
        static_cast<float*>(scales), scale_stride, (maxlen * 16 + 127) / 128};
    kernel<<<dim3(params.max_m_blocks, 4, batch), 512, sizeof(Shared), static_cast<cudaStream_t>(stream_ptr)>>>(params);
    return int(cudaGetLastError());
}

#endif
}  // namespace apac_split_pv

extern "C" int apac_split_pv_shared_bytes() { return sizeof(apac_split_pv::Shared); }
