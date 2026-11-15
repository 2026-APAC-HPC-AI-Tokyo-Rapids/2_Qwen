// Research-only SM90a Prefill kernel. Not selected by the serving dispatcher.
// Two QK and two PV warpgroups; one 128-thread producer warpgroup. V is transposed once.
// A two-slot FP8 P mailbox replaces the cross-iteration RS operand dependency.
// The Q/K/V representation, causal mask, FP32 softmax and FP8 rounding are kept.
#include "verify.h"
#include "utils.h"
#include "cutlass/arch/reg_reconfig.h"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "softmax.h"
#include "cuda_check.h"
#include <stdexcept>
#include <cuda.h>

namespace apac_split_pv {
using namespace cute;
using E = cutlass::float_e4m3_t;
using ShapeQK = Shape<_128, _192, _128>;
using ShapePV = Shape<_128, _128, _192>;
using ML = flash::CollectiveMainloopFwdSm90<2, Shape<_1, _1, _1>, ShapeQK, 128,
    E, float, cutlass::arch::Sm90, true, false, false, true, false, false,
    false, false, true, true, true, false, false, cutlass::bfloat16_t>;
// Two N96 atoms retain the N192 mathematical tile and the reduction order.
// Keep the exact N192 mathematics with smaller contiguous accumulator fragments.
using QK = decltype(make_tiled_mma(GMMA::ss_op_selector<E, E, float,
    Shape<_128, _96, _128>>(), Layout<Shape<_2, _1, _1>>{}));
using PV = decltype(make_tiled_mma(GMMA::ss_op_selector<E, E, float, ShapePV,
    GMMA::Major::K, GMMA::Major::K>(), Layout<Shape<_2, _1, _1>>{}));
// Native N128 C layout for the full-row amax and output permutation.
using PVOutput = decltype(make_tiled_mma(GMMA::ss_op_selector<E, E, float, ShapePV,
    GMMA::Major::K, GMMA::Major::K>(), Layout<Shape<_2, _1, _1>>{}));
using PLayout = decltype(tile_to_shape(ML::SmemLayoutAtomP{}, Shape<_128, _192, _2>{}));
using PPipe = cutlass::PipelineAsync<2>;

struct Shared : aligned_struct<1024, _1> {
    struct {
        struct {
            array_aligned<E, cosize_v<ML::SmemLayoutQ>, 1024> smem_q;
            array_aligned<E, cosize_v<ML::SmemLayoutK>, 1024> smem_k;
            array_aligned<E, cosize_v<ML::SmemLayoutVtMma>, 1024> smem_v;
        } mainloop;
        array_aligned<E, cosize_v<PLayout>, 1024> p;
        float alpha[2][128];
        float final_scale[128];
    } tensors;
    struct {
        alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_Q;
        alignas(16) ML::MainloopPipelineK::SharedStorage pipeline_k;
        alignas(16) ML::MainloopPipelineK::SharedStorage pipeline_v;
        alignas(16) PPipe::SharedStorage p[2];
    } pipelines;
};
static_assert(sizeof(Shared) <= 227 * 1024, "P mailbox must fit beside the unchanged KV stages");
// Five complete warpgroups keep register redistribution valid on each SM partition.
// Native assembly must report 96 initial registers (61,440 per CTA).
static_assert(128 * (2 * 128 + 2 * 96 + 32) <= 640 * 96);
static_assert(128 * (2 * 24 + 2 * 176 + 32) <= 640 * 96);

#ifndef APAC_SPLIT_PV_LAYOUT_ONLY
struct Params {
    CUtensorMap q_map, k_map;
    uint8_t* out;
    float* scale;
    uint8_t const* packed_v;
    int const* cu;
    int64_t scale_stride;
    int max_m_blocks, max_n_blocks;
    float qk_scale_log2;
    decltype(ML::Params{}.qhead_per_khead_divmod) qdiv;
    decltype(ML::Params{}.attention_chunk_divmod) chunkdiv;
};

// Same LDSM.T / PRMT / STSM.N transpose as the native producer, once per
// unique V tile. Padded rows are zero; no new precision conversion occurs.
__global__ void pack_v(uint8_t const* input, uint8_t* output, int const* cu) {
    __shared__ __align__(1024) uint8_t vt[192*128], vv[192*128];
    int const batch = blockIdx.z, head = blockIdx.y;
    int const offset = cu[batch], len = cu[batch+1]-offset;
    auto st = as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(vt), ML::SmemLayoutVt{}));
    auto sv = as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(vv), ML::SmemLayoutVtMma{}));
    for (int x=threadIdx.x; x<192*128; x+=128) {
        int const token = int(blockIdx.x)*192+x/128;
        st(x%128, x/128, 0) = token<len ? input[((int64_t(offset)+token)*4+head)*128+x%128] : 0;
    }
    __syncthreads();
    auto ts = ML::S2RTiledCopyVt{}.get_thread_slice(threadIdx.x).partition_S(flat_divide(st,ML::LDSM_divide_shape{}));
    auto td = ML::R2STiledCopyV{}.get_thread_slice(threadIdx.x).partition_D(flat_divide(sv,ML::STSM_divide_shape{}));
    auto ss = logical_divide(group_modes<1,rank(ts)-1>(ts),Shape<Underscore,_2>{});
    auto dd = logical_divide(group_modes<1,rank(td)-1>(td),Shape<Underscore,_2>{});
    CUTE_UNROLL
    for (int i=0;i<size<1,1>(ss);++i) {
        auto reg=make_fragment_like(dd(_,make_coord(_,_0{}),_0{}));
        copy(ML::S2RTiledCopyVt{},ss(_,make_coord(_,i),0),reg);
        auto r64=recast<uint2>(reg);
        CUTE_UNROLL
        for (int j=0;j<size(r64);++j) {
            uint32_t a=r64[j].x,b=r64[j].y;
            r64[j].x=__byte_perm(a,b,0x6420);r64[j].y=__byte_perm(a,b,0x7531);
        }
        copy(ML::R2STiledCopyV{},reg,dd(_,make_coord(_,i),0));
    }
    __syncthreads();
    auto dst=output+((batch*4+head)*gridDim.x+blockIdx.x)*192*128;
    for (int x=threadIdx.x;x<192*128;x+=128) dst[x]=vv[x];
}

// Each PV warpgroup owns the same 64 rows as its corresponding QK warpgroup.
// The native direct-O mapping does not stage another full O tensor.
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
    auto coords = group_modes<1, 3>(PVOutput{}.get_thread_slice(tid).partition_C(
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

__global__ __launch_bounds__(640, 1) void kernel(__grid_constant__ Params const p) {
    extern __shared__ __align__(1024) char storage[];
    auto& s = *reinterpret_cast<Shared*>(storage);
    int const wg = __shfl_sync(0xffffffff, int(threadIdx.x) / 128, 0);
    int const bidb = blockIdx.z;
    int const head = blockIdx.y;
    int const offset = p.cu[bidb], len = p.cu[bidb+1]-offset;
    int const m_block = p.max_m_blocks - 1 - int(blockIdx.x);
    if (m_block * 8 >= len) { return; }
    int const n_min = 0;
    int const n_max = (min(len, (m_block+1)*8)+191)/192;
    if (threadIdx.x == 0) { s.pipelines.barrier_Q.init(1); }
    ML::MainloopPipelineK::Params kp;
    kp.role = wg == 4 ? ML::MainloopPipelineK::ThreadCategory::Producer : ML::MainloopPipelineK::ThreadCategory::Consumer;
    kp.transaction_bytes = 192*128;
    kp.is_leader = threadIdx.x == 512;
    kp.num_consumers = 256;
    ML::MainloopPipelineK pk(s.pipelines.pipeline_k, kp, Shape<_1, _1, _1>{});
    ML::MainloopPipelineK pv(s.pipelines.pipeline_v, kp, Shape<_1, _1, _1>{});
    PPipe::Params pp;
    pp.role = wg < 2 ? PPipe::ThreadCategory::Producer : PPipe::ThreadCategory::Consumer;
    pp.producer_arv_count = 128;
    pp.consumer_arv_count = 128;
    PPipe mailbox0(s.pipelines.p[0], pp);
    PPipe mailbox1(s.pipelines.p[1], pp);
    auto mailbox = wg == 1 || wg == 3 ? mailbox1 : mailbox0;
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    if (wg == 4) {
        cutlass::arch::warpgroup_reg_dealloc<32>();
        if (threadIdx.x == 512) {
            s.pipelines.barrier_Q.arrive_and_expect_tx(128*128);
            SM90_TMA_LOAD_3D::copy(&p.q_map, reinterpret_cast<uint64_t*>(&s.pipelines.barrier_Q), uint64_t(TMA::CacheHintSm90::EVICT_NORMAL), s.tensors.mainloop.smem_q.data(), 0, head*16, offset+m_block*8);
            auto write = cutlass::make_producer_start_state<ML::MainloopPipelineK>();
            for (int n=n_max-1;n>=0;--n) {
                pk.producer_acquire(write);
                SM90_TMA_LOAD_3D::copy(&p.k_map, pk.producer_get_barrier(write), uint64_t(TMA::CacheHintSm90::EVICT_LAST), s.tensors.mainloop.smem_k.data()+write.index()*192*128, 0, head, offset+n*192);
                pv.producer_acquire(write);
                SM90_BULK_COPY_G2S::copy(p.packed_v+((bidb*4+head)*p.max_n_blocks+n)*192*128, pv.producer_get_barrier(write), s.tensors.mainloop.smem_v.data()+write.index()*192*128, 192*128);
                ++write;
            }
            pk.producer_tail(write);
            pv.producer_tail(write);
        }
    } else if (wg < 2) {
        cutlass::arch::warpgroup_reg_alloc<128>();
        int const qtid = int(threadIdx.x);
        QK qk;
        auto qk_wg = qk.get_slice(wg * 128);
        auto sq = make_tensor(make_smem_ptr(s.tensors.mainloop.smem_q.data()), ML::SmemLayoutQ{});
        auto sk = make_tensor(make_smem_ptr(s.tensors.mainloop.smem_k.data()), ML::SmemLayoutK{});
        auto q = qk_wg.partition_fragment_A(sq);
        auto k = qk_wg.partition_fragment_B(sk);
        auto p_layout = make_composed_layout(PLayout{}.layout_a(), _0{}, PLayout{}.layout_b());
        auto sp = make_tensor(counting_iterator<uint32_t>{}, p_layout);
        uint32_t const p_base = cast_smem_ptr_to_uint(s.tensors.p.data());
        // Use the RS operand's own coordinate mapping, rather than guessing
        // the cross-lane FP8 layout from the QK accumulator's layout.
        auto rs_thread = ML::TiledMmaPV{}.get_thread_slice(qtid);
        flash::Softmax<2, 8> softmax(p.qk_scale_log2);
        flash::Mask<128, 192, true, QK> mask(qtid, len, len, -1, -1, 0, p.chunkdiv, p.qdiv);
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
                uint32_t address = p_base + dest(4 * i);
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(address), "r"(words(i)) : "memory");
            }
            // Reload the lane identity after the score fragment dies, rather
            // than spilling a row coordinate across every EX2/P conversion.
            uint32_t lane_thread;
            asm volatile("mov.u32 %0, %%tid.x;" : "=r"(lane_thread));
            int const row_base = (lane_thread / 128) * 64 + ((lane_thread / 32) % 4) * 16 + (lane_thread % 32) / 4;
            CUTE_UNROLL
            for (int r = 0; r < size(alpha); ++r) {
                if (lane_thread % 4 == 0) { s.tensors.alpha[write.index()][row_base + 8*r] = alpha(r); }
            }
            if (n == n_min) {
                auto final = softmax.finalize(1.f);
                CUTE_UNROLL
                for (int r = 0; r < size(final); ++r) {
                    if (lane_thread % 4 == 0) { s.tensors.final_scale[row_base + 8*r] = final(r); }
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
        cutlass::arch::NamedBarrier::sync(128, 13+wg);
        cutlass::arch::warpgroup_reg_dealloc<24>();
    } else {
        cutlass::arch::warpgroup_reg_dealloc<96>();
        PV mma;
        int const pv_tid = int(threadIdx.x) - 256;
        auto wg_mma = mma.get_slice((wg - 2) * 128);
        auto sv = make_tensor(make_smem_ptr(s.tensors.mainloop.smem_v.data()), ML::SmemLayoutVtMma{});
        auto sp = make_tensor(make_smem_ptr(s.tensors.p.data()), PLayout{});
        auto v = wg_mma.partition_fragment_B(sv);
        auto a = wg_mma.partition_fragment_A(sp);
        auto out = partition_fragment_C(mma, select<0, 1>(ShapePV{}));
        clear(out);
        auto coords = mma.get_thread_slice(pv_tid).partition_C(make_identity_tensor(select<0, 1>(ShapePV{})));
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
        cutlass::arch::NamedBarrier::sync(128, 11+wg-2);
        cutlass::arch::warpgroup_reg_alloc<176>();
        CUTE_UNROLL
        for (int r = 0; r < size<0>(o_rc); ++r) {
            float const f = s.tensors.final_scale[get<0>(rows(r))];
            CUTE_UNROLL
            for (int c = 0; c < size<1>(o_rc); ++c) { o_rc(r, c) *= f; }
        }
        auto output_view = make_tensor(out.data(), partition_fragment_C(PVOutput{}, select<0, 1>(ShapePV{})).layout());
        flash::permute_output_fp8(output_view);
        store_output(p, output_view, pv_tid, m_block, head, len, offset);
    }
}


// Match the fixed-prefill probe ABI; tensor maps are built from the actual
// packed request buffer, while each CTA uses its own cu_seqlens offsets.
extern "C" int pf_count() { return 1; }
extern "C" int pf_resources(int) { return sizeof(Shared); }
// Probe-owned scratch is allocated before graph capture and released explicitly.
static uint8_t* packed_workspace = nullptr;
extern "C" int pf_init() {
    auto error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared));
    if (error != cudaSuccess) return int(error);
    if (packed_workspace) return 0;
    return int(cudaMalloc(&packed_workspace, 4*4*22*192*128));
}
extern "C" int pf_fini() {
    auto error = cudaFree(packed_workspace); packed_workspace = nullptr; return int(error);
}
static CUresult map_tensor(CUtensorMap* map, void* ptr, int rows, int heads, int box_heads, int box_tokens) {
    uint64_t dims[3] = {128, uint64_t(heads), uint64_t(rows)};
    uint64_t strides[2] = {128, uint64_t(heads)*128};
    uint32_t box[3] = {128, uint32_t(box_heads), uint32_t(box_tokens)}, elem[3] = {1,1,1};
    return CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, ptr, dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
extern "C" int pf_run(int id, int rows, int batch, int maxlen,
                     void* q, void* k, void* v, void* cu, void* out, void* scales,
                     int scale_stride, void*, float scale, void* stream_ptr) {
    if (id != 0 || rows < 1 || batch < 1 || batch > 4 || maxlen < 1 || maxlen > 4096 || scale_stride < rows || !packed_workspace) { return -1; }
    Params p{};
    auto err = map_tensor(&p.q_map, q, rows, 64, 16, 8);
    if (err != CUDA_SUCCESS) return 1000+int(err);
    err = map_tensor(&p.k_map, k, rows, 4, 1, 192);
    if (err != CUDA_SUCCESS) return 1000+int(err);
    p.out=static_cast<uint8_t*>(out);p.scale=static_cast<float*>(scales);p.scale_stride=scale_stride;
    p.packed_v=packed_workspace;p.cu=static_cast<int*>(cu);
    p.max_m_blocks=(maxlen*16+127)/128;p.max_n_blocks=(maxlen+191)/192;
    p.qk_scale_log2=scale*float(M_LOG2E);
    p.qdiv=decltype(p.qdiv)(16);p.chunkdiv=decltype(p.chunkdiv)(1);
    auto stream=static_cast<cudaStream_t>(stream_ptr);
    pack_v<<<dim3(p.max_n_blocks,4,batch),128,0,stream>>>(static_cast<uint8_t const*>(v),packed_workspace,p.cu);
    auto result=cudaGetLastError();if (result!=cudaSuccess)return int(result);
    kernel<<<dim3(p.max_m_blocks,4,batch),640,sizeof(Shared),stream>>>(p);
    return int(cudaGetLastError());
}

#endif
}  // namespace apac_split_pv

extern "C" int apac_split_pv_shared_bytes() { return sizeof(apac_split_pv::Shared); }
