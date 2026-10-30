// Prefill EP1 MegaMoE FC1 mainloop model: which resource binds the 128 x 256 x 128 K block?
//
// One persistent CTA per SM runs the serving tile's FC1 K loop (four 48 KiB A/B stages, an A+SFA
// TMA warp and a B TMA warp, two math warpgroups with one m64n256k32 FP8 accumulator each and the
// carried-scale rescale between K blocks, as in sm90_fp8_mega_moe_impl's kPrefill2WG loop) over
// 44 experts x 4 pool blocks x 12 N blocks with K = 4096 (16 tiles of 32 K blocks per SM, the FC1
// tile order of the serving scheduler: N fastest). Variants change one resource at a time:
//   ss          the serving loop (TMA A 16 + B 32 KiB, WGMMA reads A 2 x 8 + B 2 x 32 KiB per K block)
//   rs-ldg      A straight from global into the WGMMA register fragment (RS form), prefetched two K
//               blocks deep from a fragment-ordered copy of A; TMA moves B and SFA only
//   ss-mcb      2-CTA clusters on the same N block, B multicast: each CTA's TMA fetches half of B
//               (L2 -> SM 48 -> 32 KiB per CTA and K block, shared-memory bytes unchanged)
//   ss-norescale  the serving loop without the 128 FMUL rescale (wrong numbers, timing only)
//   ss-ares     A written once per stage at start; TMA moves B and SFA only (timing only)
//   rs-const    RS form with constant A registers, no A traffic at all (timing only)
//   splitP-uU   the serving loop with each warpgroup's m64n256 accumulator issued as P parts
//               (m64n128 or m64n64) on the same stage: the rescale of part p of a K block runs while
//               its later parts and the previous K block's are still queued (same products, K order
//               and multipliers per element); U K blocks per straight-line chunk, drained at its end
//   rs-nat      RS form with A in its natural row-major layout (as the pool and the FC1 output are):
//               per K block each thread loads bytes 16c..16c+15 and 64+16c..64+16c+15 (c = lane % 4)
//               of its two rows with coherent ld.global.cg (the pool is written inside the kernel),
//               and B's K is permuted within each 128-K block to the same order (weights are
//               transformed once at load); only the summation order inside the tensor core differs
// ss, rs-ldg, ss-mcb and the split variants compute the same values: their per-thread output hashes must match bit
// for bit, and tile 0 must match an FP64 host reference.
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("FAIL cuda %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define CHECK_CU(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { printf("FAIL driver %d at %s:%d\n", int(r_), __FILE__, __LINE__); exit(1); } } while (0)

enum Variant : int { SS = 0, RS_LDG = 1, SS_MCB = 2, SS_NORESCALE = 3, SS_ARES = 4, RS_CONST = 5, RS_NAT = 6, SPLIT2_U1 = 7, SPLIT2_U4 = 8, SPLIT2_U8 = 9, SPLIT4_U1 = 10, SPLIT4_U4 = 11, NUM_VARIANTS = 12 };
static const char* kVariantNames[NUM_VARIANTS] = {"ss", "rs-ldg", "ss-mcb", "ss-norescale", "ss-ares", "rs-const", "rs-nat", "split2-u1", "split2-u4", "split2-u8", "split4-u1", "split4-u4"};

constexpr uint32_t K = 4096, KB = K / 128, NUM_N_BLOCKS = 12, M_BLOCKS_PER_EXPERT = 4, EXPERTS = 44;
constexpr uint32_t BM = 128, BN = 256, STAGES = 4;
constexpr uint32_t A_STAGE = BM * 128, B_STAGE = BN * 128, SFA_STAGE = BM * 4;
constexpr uint32_t NUM_TILES = EXPERTS * M_BLOCKS_PER_EXPERT * NUM_N_BLOCKS;
constexpr uint32_t M_TOTAL = EXPERTS * M_BLOCKS_PER_EXPERT * BM, N_TOTAL = EXPERTS * NUM_N_BLOCKS * BN;
constexpr uint32_t SMEM_BYTES = 1024 + STAGES * (A_STAGE + B_STAGE + SFA_STAGE) + 2 * STAGES * 8 + 8;
constexpr uint32_t THREADS = 384;

using Barrier = cutlass::arch::ClusterTransactionBarrier;
using SS_MMA = cute::SM90::GMMA::MMA_64x256x32_F32E4M3E4M3_SS_TN<>;
using SS_HALF_MMA = cute::SM90::GMMA::MMA_64x128x32_F32E4M3E4M3_SS_TN<>;
using RS_MMA = cute::SM90::GMMA::MMA_64x256x32_F32E4M3E4M3_RS_TN<>;

template <size_t... I>
__device__ __forceinline__ void wgmma_ss(uint64_t da, uint64_t db, float* d, std::index_sequence<I...>) {
    SS_MMA::fma(da, db, d[I]..., cute::SM90::GMMA::ScaleOut::One);
}
template <size_t... I>
__device__ __forceinline__ void wgmma_ss_half(uint64_t da, uint64_t db, float* d, bool accumulate, std::index_sequence<I...>) {
    SS_HALF_MMA::fma(da, db, d[I]..., accumulate ? cute::SM90::GMMA::ScaleOut::One : cute::SM90::GMMA::ScaleOut::Zero);
}
template <size_t... I>
__device__ __forceinline__ void wgmma_rs(const uint32_t* a, uint64_t db, float* d, std::index_sequence<I...>) {
    RS_MMA::fma(a[0], a[1], a[2], a[3], db, d[I]..., cute::SM90::GMMA::ScaleOut::One);
}

__device__ __forceinline__ uint4 ldg_cg(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ uint4 ldg_na(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

struct Stats { unsigned long long cycles, ns, loop_cycles; };

template <int V>
__global__ void __launch_bounds__(THREADS, 1)
mainloop(const __grid_constant__ CUtensorMap tma_a, const __grid_constant__ CUtensorMap tma_b,
         const __grid_constant__ CUtensorMap tma_sfa, const float* __restrict__ w_sf,
         const uint4* __restrict__ a_frag, const uint8_t* __restrict__ a_nat, uint32_t* __restrict__ hashes, float* __restrict__ tile0, Stats* __restrict__ stats) {
    constexpr bool kRS = V == RS_LDG or V == RS_CONST or V == RS_NAT;
    constexpr bool kMC = V == SS_MCB;
    constexpr uint32_t kSplit = (V == SPLIT2_U1 or V == SPLIT2_U4 or V == SPLIT2_U8) ? 2 : ((V == SPLIT4_U1 or V == SPLIT4_U4) ? 4 : 1);
    constexpr uint32_t kUnroll = (V == SPLIT2_U4 or V == SPLIT4_U4) ? 4 : (V == SPLIT2_U8 ? 8 : 1);
    constexpr bool kLoadA = not kRS and V != SS_ARES;
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
    auto smem_a = [&](uint32_t s) { return smem + s * A_STAGE; };
    auto smem_b = [&](uint32_t s) { return smem + STAGES * A_STAGE + s * B_STAGE; };
    auto smem_sfa = [&](uint32_t s) { return reinterpret_cast<float*>(smem + STAGES * (A_STAGE + B_STAGE) + s * SFA_STAGE); };
    Barrier* full = reinterpret_cast<Barrier*>(smem + STAGES * (A_STAGE + B_STAGE + SFA_STAGE));
    Barrier* empty = full + STAGES;
    Barrier* init_bar = empty + STAGES;

    const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const uint32_t rank = kMC ? cute::block_rank_in_cluster() : 0;
    const uint32_t group = kMC ? blockIdx.x / 2 : blockIdx.x, num_groups = kMC ? gridDim.x / 2 : gridDim.x;
    const uint32_t tiles_per_cta = NUM_TILES / gridDim.x;
    // Tile i of this CTA: expert, pool block and N block (N fastest; with multicast both CTAs of a
    // cluster take the same N block on neighbouring pool blocks).
    auto tile_of = [&](uint32_t i, uint32_t& m_glob, uint32_t& n_glob, uint32_t& id) {
        if constexpr (kMC) {
            const uint32_t p = group + i * num_groups;
            const uint32_t e = p / (NUM_N_BLOCKS * M_BLOCKS_PER_EXPERT / 2), m = ((p / NUM_N_BLOCKS) % (M_BLOCKS_PER_EXPERT / 2)) * 2 + rank, n = p % NUM_N_BLOCKS;
            m_glob = e * M_BLOCKS_PER_EXPERT + m, n_glob = e * NUM_N_BLOCKS + n;
        } else {
            const uint32_t t = blockIdx.x + i * gridDim.x;
            const uint32_t e = t / (NUM_N_BLOCKS * M_BLOCKS_PER_EXPERT), m = (t / NUM_N_BLOCKS) % M_BLOCKS_PER_EXPERT, n = t % NUM_N_BLOCKS;
            m_glob = e * M_BLOCKS_PER_EXPERT + m, n_glob = e * NUM_N_BLOCKS + n;
        }
        id = m_glob * NUM_N_BLOCKS + n_glob % NUM_N_BLOCKS;
    };

    if (threadIdx.x == 0) {
        for (uint32_t s = 0; s < STAGES; ++s) {
            full[s].init(2);
            empty[s].init(kMC ? 16 : 8);
        }
        init_bar->init(1);
        cutlass::arch::fence_barrier_init();
    }
    if constexpr (kMC) cute::cluster_sync(); else __syncthreads();

    if (warp < 4) {
        cutlass::arch::warpgroup_reg_dealloc<40>();
        const bool leader = cute::elect_one_sync();
        if (warp == 2 and leader) {
            // A + SFA loader (one lane).
            if constexpr (V == SS_ARES) {
                init_bar->arrive_and_expect_tx(STAGES * A_STAGE);
                for (uint32_t s = 0; s < STAGES; ++s)
                    cute::SM90_TMA_LOAD_2D::copy(&tma_a, reinterpret_cast<uint64_t*>(init_bar), 0, smem_a(s), s * 128, 0);
                init_bar->wait(0);
            }
            uint32_t stage = 0, phase = 0;
            for (uint32_t i = 0; i < tiles_per_cta; ++i) {
                uint32_t m_glob, n_glob, id;
                tile_of(i, m_glob, n_glob, id);
                for (uint32_t kb = 0; kb < KB; ++kb) {
                    empty[stage].wait(phase ^ 1);
                    full[stage].arrive_and_expect_tx((kLoadA ? A_STAGE : 0) + SFA_STAGE);
                    if constexpr (kLoadA)
                        cute::SM90_TMA_LOAD_2D::copy(&tma_a, reinterpret_cast<uint64_t*>(&full[stage]), 0, smem_a(stage), kb * 128, m_glob * BM);
                    cute::SM90_TMA_LOAD_2D::copy(&tma_sfa, reinterpret_cast<uint64_t*>(&full[stage]), 0, smem_sfa(stage), m_glob * BM, kb);
                    if (++stage == STAGES) stage = 0, phase ^= 1;
                }
            }
        } else if (warp == 3 and leader) {
            // B loader (one lane).
            uint32_t stage = 0, phase = 0;
            for (uint32_t i = 0; i < tiles_per_cta; ++i) {
                uint32_t m_glob, n_glob, id;
                tile_of(i, m_glob, n_glob, id);
                for (uint32_t kb = 0; kb < KB; ++kb) {
                    empty[stage].wait(phase ^ 1);
                    full[stage].arrive_and_expect_tx(B_STAGE);
                    if constexpr (kMC)
                        cute::SM90_TMA_LOAD_MULTICAST_2D::copy(&tma_b, reinterpret_cast<uint64_t*>(&full[stage]), 3, 0, smem_b(stage) + rank * (B_STAGE / 2), kb * 128, n_glob * BN + rank * (BN / 2));
                    else
                        cute::SM90_TMA_LOAD_2D::copy(&tma_b, reinterpret_cast<uint64_t*>(&full[stage]), 0, smem_b(stage), kb * 128, n_glob * BN);
                    if (++stage == STAGES) stage = 0, phase ^= 1;
                }
            }
        }
        __syncwarp();
    } else {
        cutlass::arch::warpgroup_reg_alloc<(kRS or kUnroll > 4) ? 232 : 168>();
        const uint32_t wg = (warp - 4) / 4, warp_in_wg = (warp - 4) % 4, t = threadIdx.x - 128 - wg * 128;
        const uint32_t r0 = wg * 64 + warp_in_wg * 16 + lane / 4, r1 = r0 + 8;
        float accum[128];
        uint32_t a_buf[2][16];
        const auto desc_a = deep_gemm::mma::sm90::make_smem_desc(smem_a(0) + wg * 64 * 128, 1);
        const auto desc_b = deep_gemm::mma::sm90::make_smem_desc(smem_b(0), 1);
        const uint32_t a_lo = desc_a.reg32_[0], b_lo = desc_b.reg32_[0];
        const uint64_t a_hi = uint64_t(desc_a.reg32_[1]) << 32, b_hi = uint64_t(desc_b.reg32_[1]) << 32;
        // Fragment-ordered A: [m block][warpgroup][K block][k32 step][thread] x 16 bytes.
        auto load_a = [&](uint32_t* a, uint32_t i, uint32_t kb) {
            if (i >= tiles_per_cta) return;
            uint32_t m_glob, n_glob, id;
            tile_of(i, m_glob, n_glob, id);
            const uint4* p = a_frag + ((size_t(m_glob * 2 + wg) * KB + kb) * 4) * 128 + t;
            #pragma unroll
            for (uint32_t j = 0; j < 4; ++j) {
                const uint4 v = ldg_na(p + j * 128);
                a[j * 4 + 0] = v.x, a[j * 4 + 1] = v.y, a[j * 4 + 2] = v.z, a[j * 4 + 3] = v.w;
            }
        };
        // Natural A: register j * 4 + {0, 1, 2, 3} = (row r, row r + 8) x (low, high) 4 bytes of step j;
        // step j takes bytes 8 * (j % 2) .. +7 of chunk j / 2.
        auto load_a_nat = [&](uint32_t* a, uint32_t i, uint32_t kb) {
            if (i >= tiles_per_cta) return;
            uint32_t m_glob, n_glob, id;
            tile_of(i, m_glob, n_glob, id);
            const uint8_t* p = a_nat + size_t(m_glob * BM + wg * 64 + warp_in_wg * 16 + lane / 4) * K + kb * 128 + 16 * (lane % 4);
            #pragma unroll
            for (uint32_t h = 0; h < 2; ++h) {
                const uint4 lo = ldg_cg(reinterpret_cast<const uint4*>(p + 64 * h)), hi = ldg_cg(reinterpret_cast<const uint4*>(p + 8 * K + 64 * h));
                a[h * 8 + 0] = lo.x, a[h * 8 + 1] = hi.x, a[h * 8 + 2] = lo.y, a[h * 8 + 3] = hi.y;
                a[h * 8 + 4] = lo.z, a[h * 8 + 5] = hi.z, a[h * 8 + 6] = lo.w, a[h * 8 + 7] = hi.w;
            }
        };
        if constexpr (V == RS_LDG) {
            load_a(a_buf[0], 0, 0);
            load_a(a_buf[1], 0, 1);
        }
        if constexpr (V == RS_NAT) {
            load_a_nat(a_buf[0], 0, 0);
            load_a_nat(a_buf[1], 0, 1);
        }
        if constexpr (V == RS_CONST) {
            #pragma unroll
            for (uint32_t j = 0; j < 16; ++j) a_buf[0][j] = a_buf[1][j] = 0x38383838u;
        }
        const unsigned long long start_cycles = clock64();
        unsigned long long start_ns;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start_ns));
        unsigned long long loop_cycles = 0;
        uint32_t stage = 0, phase = 0;
        for (uint32_t i = 0; i < tiles_per_cta; ++i) {
            uint32_t m_glob, n_glob, id;
            tile_of(i, m_glob, n_glob, id);
            const uint32_t expert = m_glob / M_BLOCKS_PER_EXPERT, n = n_glob % NUM_N_BLOCKS;
            const float* gate_sf = w_sf + (expert * 2 * NUM_N_BLOCKS + n) * KB;
            const float* up_sf = gate_sf + NUM_N_BLOCKS * KB;
            if constexpr (kSplit == 1) {
                #pragma unroll
                for (uint32_t j = 0; j < 128; ++j) accum[j] = 0.0f;
            }
            float prev_sa0 = 1.0f, prev_sa1 = 1.0f, prev_g = 1.0f, prev_u = 1.0f;
            float r0g = 1.0f, r1g = 1.0f, r0u = 1.0f, r1u = 1.0f;
            const unsigned long long tile_start = clock64();
            auto k_block = [&](uint32_t kb, uint32_t* a) __attribute__((always_inline)) {
                if (kb == 0) {
                    full[stage].wait(phase);
                } else if constexpr (V != SS_NORESCALE) {
                    #pragma unroll
                    for (uint32_t j = 0; j < 32; ++j) {
                        const float x0 = (j & 1) ? r0u : r0g, x1 = (j & 1) ? r1u : r1g;
                        accum[j * 4 + 0] *= x0, accum[j * 4 + 1] *= x0, accum[j * 4 + 2] *= x1, accum[j * 4 + 3] *= x1;
                    }
                }
                #pragma unroll
                for (uint32_t j = 0; j < 128; ++j) deep_gemm::ptx::warpgroup_fence_operand(accum[j]);
                if constexpr (kRS) {
                    #pragma unroll
                    for (uint32_t j = 0; j < 16; ++j) asm volatile("" : "+r"(a[j]) :: "memory");
                }
                deep_gemm::ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < 4; ++k) {
                    const uint64_t db = b_hi | (b_lo + stage * (B_STAGE / 16) + k * 2);
                    if constexpr (kRS)
                        wgmma_rs(a + k * 4, db, accum, std::make_index_sequence<128>{});
                    else
                        wgmma_ss(a_hi | (a_lo + stage * (A_STAGE / 16) + k * 2), db, accum, std::make_index_sequence<128>{});
                }
                deep_gemm::ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t j = 0; j < 128; ++j) deep_gemm::ptx::warpgroup_fence_operand(accum[j]);
                if (kb == 0) {
                    prev_sa0 = smem_sfa(stage)[r0], prev_sa1 = smem_sfa(stage)[r1];
                    prev_g = __ldg(gate_sf), prev_u = __ldg(up_sf);
                }
                if (kb + 1 < KB) {
                    const uint32_t next_stage = stage == STAGES - 1 ? 0 : stage + 1;
                    const uint32_t next_phase = phase ^ (next_stage == 0 ? 1u : 0u);
                    const float g = __ldg(gate_sf + kb + 1), u = __ldg(up_sf + kb + 1);
                    full[next_stage].wait(next_phase);
                    const float sa0 = smem_sfa(next_stage)[r0], sa1 = smem_sfa(next_stage)[r1];
                    r0g = (prev_sa0 * prev_g) * __frcp_rn(sa0 * g);
                    r1g = (prev_sa1 * prev_g) * __frcp_rn(sa1 * g);
                    r0u = (prev_sa0 * prev_u) * __frcp_rn(sa0 * u);
                    r1u = (prev_sa1 * prev_u) * __frcp_rn(sa1 * u);
                    prev_sa0 = sa0, prev_sa1 = sa1, prev_g = g, prev_u = u;
                }
                deep_gemm::ptx::warpgroup_wait<0>();
                if constexpr (kRS) {
                    #pragma unroll
                    for (uint32_t j = 0; j < 16; ++j) asm volatile("" : "+r"(a[j]) :: "memory");
                }
                if (lane == 0) {
                    if constexpr (kMC) {
                        empty[stage].arrive(0u);
                        empty[stage].arrive(1u);
                    } else {
                        empty[stage].arrive();
                    }
                }
                if constexpr (V == RS_LDG)
                    load_a(a, i + (kb + 2) / KB, (kb + 2) % KB);
                if constexpr (V == RS_NAT)
                    load_a_nat(a, i + (kb + 2) / KB, (kb + 2) % KB);
                if (++stage == STAGES) stage = 0, phase ^= 1;
            };
            if constexpr (kSplit > 1) {
                // Part p: accumulator values p * 128 / P .. (columns p * 256 / P ..), B rows p * 256 / P ..
                // K blocks run in straight-line chunks of kUnroll (ptxas tracks the WGMMA groups only
                // without a loop edge in between; with one it serializes every WGMMA): inside a chunk,
                // the rescale of part p waits only for part p of the previous K block, while the later
                // parts are still queued; at a chunk end every group is drained.
                using PartMMA = typename deep_gemm::mma::sm90::FP8MMASelector<BN / kSplit>::type;
                constexpr uint32_t kPartAccum = 128 / kSplit, kPartChunks = 32 / kSplit;
                auto issue_part = [&](const uint32_t p, const uint32_t s, const bool accumulate) __attribute__((always_inline)) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kPartAccum; ++j) deep_gemm::ptx::warpgroup_fence_operand(accum[p * kPartAccum + j]);
                    deep_gemm::ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t k = 0; k < 4; ++k)
                        PartMMA::wgmma(a_hi | (a_lo + s * (A_STAGE / 16) + k * 2), b_hi | (b_lo + s * (B_STAGE / 16) + p * (BN / kSplit) * 8 + k * 2), accum + p * kPartAccum, accumulate or k > 0);
                    deep_gemm::ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t j = 0; j < kPartAccum; ++j) deep_gemm::ptx::warpgroup_fence_operand(accum[p * kPartAccum + j]);
                };
                auto rescale_part = [&](const uint32_t p) __attribute__((always_inline)) {
                    #pragma unroll
                    for (uint32_t j = p * kPartChunks; j < (p + 1) * kPartChunks; ++j) {
                        const float x0 = (j & 1) ? r0u : r0g, x1 = (j & 1) ? r1u : r1g;
                        accum[j * 4 + 0] *= x0, accum[j * 4 + 1] *= x0, accum[j * 4 + 2] *= x1, accum[j * 4 + 3] *= x1;
                    }
                };
                auto release = [&](const uint32_t s) __attribute__((always_inline)) {
                    if (lane == 0) empty[s].arrive();
                };
                uint32_t prev_stage = 0;
                for (uint32_t kc = 0; kc < KB; kc += kUnroll) {
                    #pragma unroll
                    for (uint32_t u = 0; u < kUnroll; ++u) {
                        const uint32_t kb = kc + u;
                        // Only the chunk start can be the tile's first K block (compile-time false
                        // otherwise: no branch between pending WGMMA groups and the rescale).
                        const bool first = u == 0 and kc == 0;
                        full[stage].wait(phase);
                        if (first) {
                            prev_sa0 = smem_sfa(stage)[r0], prev_sa1 = smem_sfa(stage)[r1];
                            prev_g = __ldg(gate_sf), prev_u = __ldg(up_sf);
                        } else {
                            const float g = __ldg(gate_sf + kb), uu = __ldg(up_sf + kb);
                            const float sa0 = smem_sfa(stage)[r0], sa1 = smem_sfa(stage)[r1];
                            r0g = (prev_sa0 * prev_g) * __frcp_rn(sa0 * g);
                            r1g = (prev_sa1 * prev_g) * __frcp_rn(sa1 * g);
                            r0u = (prev_sa0 * prev_u) * __frcp_rn(sa0 * uu);
                            r1u = (prev_sa1 * prev_u) * __frcp_rn(sa1 * uu);
                            prev_sa0 = sa0, prev_sa1 = sa1, prev_g = g, prev_u = uu;
                        }
                        #pragma unroll
                        for (uint32_t p = 0; p < kSplit; ++p) {
                            if (u > 0) {
                                // Part p of the previous K block is the oldest pending group.
                                deep_gemm::ptx::warpgroup_wait<kSplit - 1>();
                                if (p == kSplit - 1) release(prev_stage);
                            } else if (p > 0) {
                                // Chunk start: exactly parts 0..p-1 are pending, so this waits for
                                // nothing; it tells ptxas that part p is not in flight (without it
                                // ptxas injects a full wait here).
                                if (p == 1) deep_gemm::ptx::warpgroup_wait<1>();
                                if (p == 2) deep_gemm::ptx::warpgroup_wait<2>();
                                if (p == 3) deep_gemm::ptx::warpgroup_wait<3>();
                            }
                            // On the first K block the ratios are 1 and the WGMMA overwrites the part.
                            rescale_part(p);
                            issue_part(p, stage, not first);
                        }
                        prev_stage = stage;
                        if (++stage == STAGES) stage = 0, phase ^= 1;
                    }
                    deep_gemm::ptx::warpgroup_wait<0>();
                    release(prev_stage);
                }
            } else {
                for (uint32_t kb = 0; kb < KB; kb += 2) {
                    k_block(kb, a_buf[0]);
                    k_block(kb + 1, a_buf[1]);
                }
            }
            loop_cycles += clock64() - tile_start;
            // Final scale (the serving loop's postscale) and an order-sensitive hash per thread.
            uint32_t h = 0;
            #pragma unroll
            for (uint32_t j = 0; j < 32; ++j) {
                const float s0 = prev_sa0 * ((j & 1) ? prev_u : prev_g), s1 = prev_sa1 * ((j & 1) ? prev_u : prev_g);
                accum[j * 4 + 0] *= s0, accum[j * 4 + 1] *= s0, accum[j * 4 + 2] *= s1, accum[j * 4 + 3] *= s1;
                #pragma unroll
                for (uint32_t q = 0; q < 4; ++q) h += __float_as_uint(accum[j * 4 + q]) * (2u * (j * 4 + q) + 1u);
            }
            hashes[size_t(id) * 256 + wg * 128 + t] = h;
            if (id == 0) {
                #pragma unroll
                for (uint32_t j = 0; j < 32; ++j) {
                    #pragma unroll
                    for (uint32_t q = 0; q < 4; ++q) {
                        const uint32_t row = (q & 2) ? r1 : r0, col = j * 8 + (lane % 4) * 2 + (q & 1);
                        tile0[row * BN + col] = accum[j * 4 + q];
                    }
                }
            }
        }
        if (t == 0 and wg == 0) {
            unsigned long long end_ns;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end_ns));
            stats[blockIdx.x] = {clock64() - start_cycles, end_ns - start_ns, loop_cycles};
        }
    }
    if constexpr (kMC) cute::cluster_sync(); else __syncthreads();
}

// ---------------------------------------------------------------------------------------------
static float e4m3_to_float(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    const float v = e == 0 ? std::ldexp(m / 8.0f, -6) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return s ? -v : v;
}

static uint32_t mix(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
    return uint32_t(x);
}

static uint8_t random_fp8(uint64_t i) {
    const uint32_t r = mix(i);
    return uint8_t((r & 0x80) | ((4 + (r >> 8) % 7) << 3) | ((r >> 16) & 7));
}

static CUtensorMap make_map(void* base, CUtensorMapDataType type, uint32_t elem, uint64_t inner, uint64_t outer, uint32_t box_inner, uint32_t box_outer, CUtensorMapSwizzle swizzle) {
    CUtensorMap map;
    const cuuint64_t dims[2] = {inner, outer}, strides[1] = {inner * elem};
    const cuuint32_t box[2] = {box_inner, box_outer}, element_strides[2] = {1, 1};
    CHECK_CU(cuTensorMapEncodeTiled(&map, type, 2, base, dims, strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

int main() {
    int sms = 0;
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    const uint32_t grid = 132;
    if (sms < int(grid)) { printf("FAIL need %u SMs, have %d\n", grid, sms); return 1; }
    static_assert(NUM_TILES % 132 == 0 and (NUM_TILES / 132) % 1 == 0, "tiles per CTA");

    float fp8[256];
    for (int i = 0; i < 256; ++i) fp8[i] = e4m3_to_float(uint8_t(i));
    std::vector<uint8_t> a(size_t(M_TOTAL) * K), b(size_t(N_TOTAL) * K);
    for (size_t i = 0; i < a.size(); ++i) a[i] = random_fp8(i);
    for (size_t i = 0; i < b.size(); ++i) b[i] = random_fp8(i + (1ull << 40));
    std::vector<float> sfa(size_t(KB) * M_TOTAL), wsf(size_t(EXPERTS) * 2 * NUM_N_BLOCKS * KB);
    for (size_t i = 0; i < sfa.size(); ++i) sfa[i] = 0.5f + (mix(i + (2ull << 40)) % 1000) / 1000.0f;
    for (size_t i = 0; i < wsf.size(); ++i) wsf[i] = 0.5f + (mix(i + (3ull << 40)) % 1000) / 1000.0f;
    // Fragment order of A for the RS form (ALayout_64x32): thread t of warpgroup g holds, per k32
    // step, row 16 * (t / 32) + (t % 32) / 4 (+8) and bytes 4 * (t % 4) .. +3 (+16) of the step.
    std::vector<uint8_t> frag(a.size());
    for (uint32_t mb = 0; mb < M_TOTAL / BM; ++mb)
        for (uint32_t g = 0; g < 2; ++g)
            for (uint32_t kb = 0; kb < KB; ++kb)
                for (uint32_t j = 0; j < 4; ++j)
                    for (uint32_t t = 0; t < 128; ++t) {
                        uint8_t* dst = &frag[((((size_t(mb) * 2 + g) * KB + kb) * 4 + j) * 128 + t) * 16];
                        const uint32_t row = mb * BM + g * 64 + 16 * (t / 32) + (t % 32) / 4;
                        const uint32_t k0 = kb * 128 + j * 32 + 4 * (t % 4);
                        for (uint32_t reg = 0; reg < 4; ++reg)
                            for (uint32_t byte = 0; byte < 4; ++byte)
                                dst[reg * 4 + byte] = a[size_t(row + 8 * (reg & 1)) * K + k0 + 16 * (reg >> 1) + byte];
                    }

    // B with K permuted within each 128-K block to rs-nat's register order: tensor-core position
    // 32j + 4c + q (q < 4) holds natural byte 64(j/2) + 16c + 8(j%2) + q, position +16 holds +4.
    std::vector<uint8_t> b_perm(b.size());
    for (size_t row = 0; row < N_TOTAL; ++row)
        for (uint32_t kb = 0; kb < KB; ++kb)
            for (uint32_t j = 0; j < 4; ++j)
                for (uint32_t c = 0; c < 4; ++c)
                    for (uint32_t q = 0; q < 8; ++q) {
                        const uint32_t pos = 32 * j + 4 * c + (q < 4 ? q : 12 + q), nat = 64 * (j / 2) + 16 * c + 8 * (j % 2) + q;
                        b_perm[row * K + kb * 128 + pos] = b[row * K + kb * 128 + nat];
                    }

    uint8_t *d_a, *d_b, *d_frag, *d_b_perm;
    float *d_sfa, *d_wsf, *d_tile0;
    uint32_t* d_hash;
    Stats* d_stats;
    CHECK(cudaMalloc(&d_a, a.size()));
    CHECK(cudaMalloc(&d_b, b.size()));
    CHECK(cudaMalloc(&d_frag, frag.size()));
    CHECK(cudaMalloc(&d_b_perm, b_perm.size()));
    CHECK(cudaMemcpy(d_b_perm, b_perm.data(), b_perm.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_sfa, sfa.size() * 4));
    CHECK(cudaMalloc(&d_wsf, wsf.size() * 4));
    CHECK(cudaMalloc(&d_tile0, BM * BN * 4));
    CHECK(cudaMalloc(&d_hash, size_t(NUM_TILES) * 256 * 4));
    CHECK(cudaMalloc(&d_stats, grid * sizeof(Stats)));
    CHECK(cudaMemcpy(d_a, a.data(), a.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b.data(), b.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_frag, frag.data(), frag.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_sfa, sfa.data(), sfa.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_wsf, wsf.data(), wsf.size() * 4, cudaMemcpyHostToDevice));
    const CUtensorMap tma_a = make_map(d_a, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, K, M_TOTAL, 128, BM, CU_TENSOR_MAP_SWIZZLE_128B);
    const CUtensorMap tma_b = make_map(d_b, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, K, N_TOTAL, 128, BN, CU_TENSOR_MAP_SWIZZLE_128B);
    const CUtensorMap tma_b_perm = make_map(d_b_perm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, K, N_TOTAL, 128, BN, CU_TENSOR_MAP_SWIZZLE_128B);
    const CUtensorMap tma_b_half = make_map(d_b, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, K, N_TOTAL, 128, BN / 2, CU_TENSOR_MAP_SWIZZLE_128B);
    const CUtensorMap tma_sfa = make_map(d_sfa, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, M_TOTAL, KB, BM, 1, CU_TENSOR_MAP_SWIZZLE_NONE);

    void (*kernels[NUM_VARIANTS])(CUtensorMap, CUtensorMap, CUtensorMap, const float*, const uint4*, const uint8_t*, uint32_t*, float*, Stats*) = {
        mainloop<SS>, mainloop<RS_LDG>, mainloop<SS_MCB>, mainloop<SS_NORESCALE>, mainloop<SS_ARES>, mainloop<RS_CONST>, mainloop<RS_NAT>,
        mainloop<SPLIT2_U1>, mainloop<SPLIT2_U4>, mainloop<SPLIT2_U8>, mainloop<SPLIT4_U1>, mainloop<SPLIT4_U4>};
    for (int v = 0; v < NUM_VARIANTS; ++v) {
        CHECK(cudaFuncSetAttribute(kernels[v], cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
        cudaFuncAttributes attr;
        CHECK(cudaFuncGetAttributes(&attr, kernels[v]));
        printf("RESOURCE variant=%s registers=%d local_bytes=%zu smem=%u\n", kVariantNames[v], attr.numRegs, attr.localSizeBytes, SMEM_BYTES);
    }
    {
        cudaLaunchConfig_t config = {};
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2, attribute[0].val.clusterDim.y = 1, attribute[0].val.clusterDim.z = 1;
        config.gridDim = dim3(grid), config.blockDim = dim3(THREADS), config.dynamicSmemBytes = SMEM_BYTES, config.attrs = attribute, config.numAttrs = 1;
        int clusters = 0;
        CHECK(cudaOccupancyMaxActiveClusters(&clusters, kernels[SS_MCB], &config));
        printf("DEVICE name=%s sms=%d max_active_2cta_clusters=%d\n", prop.name, sms, clusters);
    }

    auto launch = [&](int v, cudaStream_t stream) {
        cudaLaunchConfig_t config = {};
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = v == SS_MCB ? 2 : 1, attribute[0].val.clusterDim.y = 1, attribute[0].val.clusterDim.z = 1;
        config.gridDim = dim3(grid), config.blockDim = dim3(THREADS), config.dynamicSmemBytes = SMEM_BYTES, config.stream = stream, config.attrs = attribute, config.numAttrs = 1;
        CHECK(cudaLaunchKernelEx(&config, kernels[v], tma_a, v == SS_MCB ? tma_b_half : (v == RS_NAT ? tma_b_perm : tma_b), tma_sfa, (const float*)d_wsf, (const uint4*)d_frag, (const uint8_t*)d_a, d_hash, d_tile0, d_stats));
    };

    // Correctness: ss and rs-nat against an FP64 reference on tile 0 (the carried-scale FP8
    // accumulation over 4096 K is within about 2e-3; the serving gate allows 0.025); rs-ldg and
    // ss-mcb and every split variant bit for bit against ss.
    std::vector<uint32_t> hash_ss(size_t(NUM_TILES) * 256), hash_v(hash_ss.size());
    std::vector<float> tile_ss(BM * BN), tile_v(BM * BN);
    bool ok = true;
    for (int v : {SS, RS_LDG, SS_MCB, RS_NAT, SPLIT2_U1, SPLIT2_U4, SPLIT2_U8, SPLIT4_U1, SPLIT4_U4}) {
        CHECK(cudaMemset(d_hash, 0, hash_ss.size() * 4));
        launch(v, 0);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(v == SS ? hash_ss.data() : hash_v.data(), d_hash, hash_ss.size() * 4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(v == SS ? tile_ss.data() : tile_v.data(), d_tile0, BM * BN * 4, cudaMemcpyDeviceToHost));
        if (v == SS or v == RS_NAT) {
            const std::vector<float>& tile = v == SS ? tile_ss : tile_v;
            double num = 0, den = 0, diff = 0, norm = 0;
            for (uint32_t r = 0; r < BM; ++r)
                for (uint32_t c = 0; c < BN; ++c) {
                    double ref = 0;
                    const float* w = &wsf[(((c / 8) & 1) ? NUM_N_BLOCKS : 0) * KB];
                    for (uint32_t kb = 0; kb < KB; ++kb) {
                        double dot = 0;
                        for (uint32_t k = kb * 128; k < kb * 128 + 128; ++k)
                            dot += double(fp8[a[size_t(r) * K + k]]) * fp8[b[size_t(c) * K + k]];
                        ref += dot * sfa[size_t(kb) * M_TOTAL + r] * w[kb];
                    }
                    num += (tile[r * BN + c] - ref) * (tile[r * BN + c] - ref), den += ref * ref;
                    diff += double(tile[r * BN + c] - tile_ss[r * BN + c]) * (tile[r * BN + c] - tile_ss[r * BN + c]), norm += double(tile_ss[r * BN + c]) * tile_ss[r * BN + c];
                }
            const double rel = std::sqrt(num / den);
            printf("CHECK %s tile0 relative_l2=%.3e relative_l2_to_ss=%.3e\n", kVariantNames[v], rel, std::sqrt(diff / norm));
            ok = ok and rel < 1e-2;
        } else {
            const bool same = hash_v == hash_ss and tile_v == tile_ss;
            printf("CHECK %s bitwise_equal_to_ss=%d\n", kVariantNames[v], int(same));
            ok = ok and same;
        }
    }
    if (not ok) { printf("FAIL correctness\n"); return 1; }

    // Timing: rounds of all variants in rotating order; per-round ratios to ss.
    constexpr int ROUNDS = 40;
    std::vector<std::vector<float>> us(NUM_VARIANTS), loop_clk(NUM_VARIANTS), mhz(NUM_VARIANTS);
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    std::vector<Stats> stats(grid);
    for (int v = 0; v < NUM_VARIANTS; ++v) launch(v, 0);
    CHECK(cudaDeviceSynchronize());
    for (int round = 0; round < ROUNDS; ++round) {
        for (int o = 0; o < NUM_VARIANTS; ++o) {
            const int v = (round + o) % NUM_VARIANTS;
            CHECK(cudaEventRecord(e0));
            launch(v, 0);
            CHECK(cudaEventRecord(e1));
            CHECK(cudaEventSynchronize(e1));
            float ms = 0;
            CHECK(cudaEventElapsedTime(&ms, e0, e1));
            us[v].push_back(ms * 1000.0f);
            CHECK(cudaMemcpy(stats.data(), d_stats, grid * sizeof(Stats), cudaMemcpyDeviceToHost));
            double cyc = 0, ns = 0, loop = 0;
            for (const auto& s : stats) cyc += s.cycles, ns += s.ns, loop += s.loop_cycles;
            loop_clk[v].push_back(float(loop / (double(grid) * (NUM_TILES / grid) * KB)));
            mhz[v].push_back(float(cyc / ns * 1000.0));
        }
    }
    auto median = [](std::vector<float> x) { std::sort(x.begin(), x.end()); return x[x.size() / 2]; };
    for (int v = 0; v < NUM_VARIANTS; ++v) {
        std::vector<float> ratio;
        for (int round = 0; round < ROUNDS; ++round) ratio.push_back(us[v][round] / us[SS][round]);
        printf("PASS variant=%s median_us=%.1f ratio_to_ss=%.4f loop_clk_per_kblock=%.1f sm_mhz=%.0f\n", kVariantNames[v], median(us[v]), median(ratio), median(loop_clk[v]), median(mhz[v]));
    }
    return 0;
}
