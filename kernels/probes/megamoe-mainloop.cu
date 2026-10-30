// Prefill EP1 MegaMoE FC1 mainloop model: what binds the split-accumulator K block?
//
// One persistent CTA per SM runs the serving tile's FC1 K loop (four 48 KiB A/B stages, an A+SFA
// TMA warp and a B TMA warp, two math warpgroups with a m64n256 FP8 accumulator each and the
// carried-scale rescale between K blocks, as in sm90_fp8_mega_moe_impl's kPrefill2WG loop) over
// 44 experts x 4 pool blocks x 12 N blocks with K = 4096 (16 tiles of 32 K blocks per SM, the FC1
// tile order of the serving scheduler: N fastest); the -k12 variants use 12 K blocks per tile (the
// FC2 length). Round 4 (commit fa0b244, runs 40978170082 / 40979704662 / 40980472965) measured the
// serving loop's rescale as the binding term and the split accumulator (each warpgroup's
// accumulator issued as two m64n128 halves; the rescale of half h waits only for half h of the
// previous K block) as the fix, now served as DG_SM90_PREFILL_SPLIT_ACCUM. This version asks what
// binds the split loop:
//   ss, ss-k12             the serving loop (control)
//   ss-norescale           without the rescale (wrong numbers, timing only)
//   split-uU               the served split loop, U K blocks per straight-line chunk, drained at its
//                          end (ptxas tracks WGMMA groups only in straight-line code)
//   split-norescale-uU     the split loop without the rescale (timing only): its floor
//   split-pre-uU           at a chunk end, half 0 of the next K block is rescaled after half 0 of the
//                          last K block completes and before the drain waits for half 1, so the next
//                          chunk starts by issuing half 0 (ratio 1 at the tile end: exact)
//   split-stag-pre-uU      as pre, with warpgroup 1's chunk boundaries U/2 K blocks after warpgroup
//                          0's, so one warpgroup's drain overlaps the other's queued WGMMAs
//   ldsm-*                 the split loop reads A once per K block into registers (ldmatrix from
//                          the TMA-swizzled stage, two K blocks of 16 registers) and issues both
//                          halves in the RS form: per K block the WGMMAs read A 16 KiB instead of the
//                          SS halves' 32 KiB (shared memory per K block 144 -> 128 KiB)
// Every variant except the norescale ones computes the same values as ss (or ss-k12): per-thread
// output hashes must match bit for bit, and ss's tile 0 must match an FP64 host reference.
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("FAIL cuda %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define CHECK_CU(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { printf("FAIL driver %d at %s:%d\n", int(r_), __FILE__, __LINE__); exit(1); } } while (0)

struct Cfg { const char* name; uint32_t kb, unroll; bool split, rescale, pre, stagger, ldsm; };
constexpr Cfg kCfgs[] = {
    {"ss", 32, 1, false, true, false, false, false},
    {"ss-norescale", 32, 1, false, false, false, false, false},
    {"split-u8", 32, 8, true, true, false, false, false},
    {"split-u16", 32, 16, true, true, false, false, false},
    {"split-norescale-u8", 32, 8, true, false, false, false, false},
    {"split-pre-u8", 32, 8, true, true, true, false, false},
    {"split-pre-u16", 32, 16, true, true, true, false, false},
    {"split-stag-pre-u8", 32, 8, true, true, true, true, false},
    {"ldsm-u8", 32, 8, true, true, false, false, true},
    {"ldsm-pre-u8", 32, 8, true, true, true, false, true},
    {"ldsm-pre-u16", 32, 16, true, true, true, false, true},
    {"ldsm-stag-pre-u8", 32, 8, true, true, true, true, true},
    {"ldsm-norescale-u8", 32, 8, true, false, false, false, true},
    {"ss-k12", 12, 1, false, true, false, false, false},
    {"split-k12-u12", 12, 12, true, true, false, false, false},
    {"split-norescale-k12-u12", 12, 12, true, false, false, false, false},
    {"ldsm-k12-u12", 12, 12, true, true, false, false, true},
    {"ldsm-pre-k12-u6", 12, 6, true, true, true, false, true},
};
constexpr int NUM_VARIANTS = sizeof(kCfgs) / sizeof(kCfgs[0]);
constexpr int SS = 0, SS_K12 = 13;
static_assert(kCfgs[SS_K12].kb == 12 and not kCfgs[SS_K12].split, "k12 control");

constexpr uint32_t K = 4096, KB = K / 128, NUM_N_BLOCKS = 12, M_BLOCKS_PER_EXPERT = 4, EXPERTS = 44;
constexpr uint32_t BM = 128, BN = 256, STAGES = 4;
constexpr uint32_t A_STAGE = BM * 128, B_STAGE = BN * 128, SFA_STAGE = BM * 4;
constexpr uint32_t NUM_TILES = EXPERTS * M_BLOCKS_PER_EXPERT * NUM_N_BLOCKS;
constexpr uint32_t M_TOTAL = EXPERTS * M_BLOCKS_PER_EXPERT * BM, N_TOTAL = EXPERTS * NUM_N_BLOCKS * BN;
constexpr uint32_t SMEM_BYTES = 1024 + STAGES * (A_STAGE + B_STAGE + SFA_STAGE) + 2 * STAGES * 8;
constexpr uint32_t THREADS = 384;

using Barrier = cutlass::arch::ClusterTransactionBarrier;
using SS_MMA = cute::SM90::GMMA::MMA_64x256x32_F32E4M3E4M3_SS_TN<>;
using HALF_SS_MMA = deep_gemm::mma::sm90::FP8MMASelector<BN / 2>::type;
using HALF_RS_MMA = cute::SM90::GMMA::MMA_64x128x32_F32E4M3E4M3_RS_TN<>;

template <size_t... I>
__device__ __forceinline__ void wgmma_ss(uint64_t da, uint64_t db, float* d, std::index_sequence<I...>) {
    SS_MMA::fma(da, db, d[I]..., cute::SM90::GMMA::ScaleOut::One);
}
template <size_t... I>
__device__ __forceinline__ void wgmma_rs_half(const uint32_t* a, uint64_t db, float* d, std::index_sequence<I...>) {
    HALF_RS_MMA::fma(a[0], a[1], a[2], a[3], db, d[I]..., cute::SM90::GMMA::ScaleOut::One);
}

__device__ __forceinline__ void ldsm_x4(uint32_t* r, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}

struct Stats { unsigned long long cycles, ns, loop_cycles; };

template <int V>
__global__ void __launch_bounds__(THREADS, 1)
mainloop(const __grid_constant__ CUtensorMap tma_a, const __grid_constant__ CUtensorMap tma_b, const __grid_constant__ CUtensorMap tma_sfa,
         const float* __restrict__ w_sf, uint32_t* __restrict__ hashes, float* __restrict__ tile0, Stats* __restrict__ stats) {
    constexpr uint32_t kTileKB = kCfgs[V].kb, kUnroll = kCfgs[V].unroll;
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
    auto smem_a = [&](uint32_t s) { return smem + s * A_STAGE; };
    auto smem_b = [&](uint32_t s) { return smem + STAGES * A_STAGE + s * B_STAGE; };
    auto smem_sfa = [&](uint32_t s) { return reinterpret_cast<float*>(smem + STAGES * (A_STAGE + B_STAGE) + s * SFA_STAGE); };
    Barrier* full = reinterpret_cast<Barrier*>(smem + STAGES * (A_STAGE + B_STAGE + SFA_STAGE));
    Barrier* empty = full + STAGES;

    const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const uint32_t tiles_per_cta = NUM_TILES / gridDim.x;
    // Tile i of this CTA: expert, pool block and N block (N fastest).
    auto tile_of = [&](uint32_t i, uint32_t& m_glob, uint32_t& n_glob, uint32_t& id) {
        const uint32_t t = blockIdx.x + i * gridDim.x;
        const uint32_t e = t / (NUM_N_BLOCKS * M_BLOCKS_PER_EXPERT), m = (t / NUM_N_BLOCKS) % M_BLOCKS_PER_EXPERT, n = t % NUM_N_BLOCKS;
        m_glob = e * M_BLOCKS_PER_EXPERT + m, n_glob = e * NUM_N_BLOCKS + n;
        id = m_glob * NUM_N_BLOCKS + n;
    };

    if (threadIdx.x == 0) {
        for (uint32_t s = 0; s < STAGES; ++s) {
            full[s].init(2);
            empty[s].init(8);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    if (warp < 4) {
        cutlass::arch::warpgroup_reg_dealloc<40>();
        const bool leader = cute::elect_one_sync();
        if (warp == 2 and leader) {
            // A + SFA loader (one lane).
            uint32_t stage = 0, phase = 0;
            for (uint32_t i = 0; i < tiles_per_cta; ++i) {
                uint32_t m_glob, n_glob, id;
                tile_of(i, m_glob, n_glob, id);
                for (uint32_t kb = 0; kb < kTileKB; ++kb) {
                    empty[stage].wait(phase ^ 1);
                    full[stage].arrive_and_expect_tx(A_STAGE + SFA_STAGE);
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
                for (uint32_t kb = 0; kb < kTileKB; ++kb) {
                    empty[stage].wait(phase ^ 1);
                    full[stage].arrive_and_expect_tx(B_STAGE);
                    cute::SM90_TMA_LOAD_2D::copy(&tma_b, reinterpret_cast<uint64_t*>(&full[stage]), 0, smem_b(stage), kb * 128, n_glob * BN);
                    if (++stage == STAGES) stage = 0, phase ^= 1;
                }
            }
        }
        __syncwarp();
    } else {
        cutlass::arch::warpgroup_reg_alloc<kCfgs[V].split ? 232 : 168>();
        const uint32_t wg = (warp - 4) / 4, warp_in_wg = (warp - 4) % 4, t = threadIdx.x - 128 - wg * 128;
        const uint32_t r0 = wg * 64 + warp_in_wg * 16 + lane / 4, r1 = r0 + 8;
        float accum[128];
        uint32_t a_buf[2][16];
        const auto desc_a = deep_gemm::mma::sm90::make_smem_desc(smem_a(0) + wg * 64 * 128, 1);
        const auto desc_b = deep_gemm::mma::sm90::make_smem_desc(smem_b(0), 1);
        const uint32_t a_lo = desc_a.reg32_[0], b_lo = desc_b.reg32_[0];
        const uint64_t a_hi = uint64_t(desc_a.reg32_[1]) << 32, b_hi = uint64_t(desc_b.reg32_[1]) << 32;
        // ldmatrix.x4 per k32 step: lanes 8i..8i+7 address matrix i = (rows +8 if i is odd) x (16-byte
        // chunk 2j + i / 2) of this warp's 16 rows; thread t then holds rows t / 4 (+8) and bytes
        // 4 (t % 4) .. +3 (+16) of the step, the m64k32 FP8 A fragment. The TMA 128B swizzle puts
        // chunk c of row r at c ^ (r % 8), and r % 8 = lane % 8.
        const uint32_t ldsm_row = wg * 64 + warp_in_wg * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
        const uint32_t ldsm_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem_a(0))) + ldsm_row * 128;
        auto load_a = [&](uint32_t* a, uint32_t s) __attribute__((always_inline)) {
            #pragma unroll
            for (uint32_t j = 0; j < 4; ++j)
                ldsm_x4(a + j * 4, ldsm_base + s * A_STAGE + (((2 * j + (lane >> 4)) ^ (lane & 7)) << 4));
        };
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
            #pragma unroll
            for (uint32_t j = 0; j < 128; ++j) accum[j] = 0.0f;
            float prev_sa0 = 1.0f, prev_sa1 = 1.0f, prev_g = 1.0f, prev_u = 1.0f;
            float r0g = 1.0f, r1g = 1.0f, r0u = 1.0f, r1u = 1.0f;
            // The next K block's ratios (previous over next s_a * s_w), after waiting for its stage.
            auto next_ratios = [&](uint32_t kb_next, uint32_t s, uint32_t p) __attribute__((always_inline)) {
                const float g = __ldg(gate_sf + kb_next), u = __ldg(up_sf + kb_next);
                full[s].wait(p);
                const float sa0 = smem_sfa(s)[r0], sa1 = smem_sfa(s)[r1];
                r0g = (prev_sa0 * prev_g) * __frcp_rn(sa0 * g);
                r1g = (prev_sa1 * prev_g) * __frcp_rn(sa1 * g);
                r0u = (prev_sa0 * prev_u) * __frcp_rn(sa0 * u);
                r1u = (prev_sa1 * prev_u) * __frcp_rn(sa1 * u);
                prev_sa0 = sa0, prev_sa1 = sa1, prev_g = g, prev_u = u;
            };
            const unsigned long long tile_start = clock64();
            if constexpr (kCfgs[V].split) {
                // Half h: accumulator values 64h.. (columns 128h..), B rows 128h.. of the stage.
                auto issue_half = [&](const uint32_t h, const uint32_t s, const uint32_t* a) __attribute__((always_inline)) {
                    #pragma unroll
                    for (uint32_t j = 0; j < 64; ++j) deep_gemm::ptx::warpgroup_fence_operand(accum[h * 64 + j]);
                    deep_gemm::ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t k = 0; k < 4; ++k) {
                        const uint64_t db = b_hi | (b_lo + s * (B_STAGE / 16) + h * 128 * 8 + k * 2);
                        if constexpr (kCfgs[V].ldsm)
                            wgmma_rs_half(a + k * 4, db, accum + h * 64, std::make_index_sequence<64>{});
                        else
                            HALF_SS_MMA::wgmma(a_hi | (a_lo + s * (A_STAGE / 16) + k * 2), db, accum + h * 64, true);
                    }
                    deep_gemm::ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t j = 0; j < 64; ++j) deep_gemm::ptx::warpgroup_fence_operand(accum[h * 64 + j]);
                };
                auto rescale_half = [&](const uint32_t h) __attribute__((always_inline)) {
                    if constexpr (kCfgs[V].rescale) {
                        #pragma unroll
                        for (uint32_t j = h * 16; j < (h + 1) * 16; ++j) {
                            const float x0 = (j & 1) ? r0u : r0g, x1 = (j & 1) ? r1u : r1g;
                            accum[j * 4 + 0] *= x0, accum[j * 4 + 1] *= x0, accum[j * 4 + 2] *= x1, accum[j * 4 + 3] *= x1;
                        }
                    }
                };
                uint32_t prev_stage = stage;
                // One straight-line chunk of L K blocks from kc, drained at its end. Per K block and
                // half h: wait for half h of the previous K block (the oldest pending group), release
                // the previous stage after both halves, rescale half h, issue half h; then the next K
                // block's ratios (and A registers), as the serving loop does.
                auto chunk = [&](const uint32_t kc, auto len_c) __attribute__((always_inline)) {
                    constexpr uint32_t L = decltype(len_c)::value;
                    #pragma unroll
                    for (uint32_t u = 0; u < L; ++u) {
                        #pragma unroll
                        for (uint32_t h = 0; h < 2; ++h) {
                            if (u > 0) {
                                deep_gemm::ptx::warpgroup_wait<1>();
                                if (h == 1 and lane == 0) empty[prev_stage].arrive();
                            } else if (h == 1) {
                                // Chunk start: only half 0 is pending, so this waits for nothing;
                                // it tells ptxas that half 1 is not in flight.
                                deep_gemm::ptx::warpgroup_wait<1>();
                            }
                            // With kCfgs[V].pre the chunk's first half 0 was rescaled before the drain.
                            if (not (kCfgs[V].pre and u == 0 and h == 0)) rescale_half(h);
                            issue_half(h, stage, a_buf[u % 2]);
                        }
                        const uint32_t next_stage = stage == STAGES - 1 ? 0 : stage + 1;
                        const uint32_t next_phase = phase ^ (next_stage == 0 ? 1u : 0u);
                        if (u + 1 < L or kc + L < kTileKB) {
                            next_ratios(kc + u + 1, next_stage, next_phase);
                            if constexpr (kCfgs[V].ldsm) load_a(a_buf[(u + 1) % 2], next_stage);
                        } else if constexpr (kCfgs[V].pre) {
                            // Tile end: the pre-rescale below multiplies by 1 (exact).
                            r0g = r1g = r0u = r1u = 1.0f;
                        }
                        prev_stage = stage;
                        stage = next_stage, phase = next_phase;
                    }
                    if constexpr (kCfgs[V].pre) {
                        // Half 0 of the last K block is done; rescale it for the next K block while
                        // half 1 runs.
                        deep_gemm::ptx::warpgroup_wait<1>();
                        rescale_half(0);
                    }
                    deep_gemm::ptx::warpgroup_wait<0>();
                    if (lane == 0) empty[prev_stage].arrive();
                };
                // Zeroed accumulators are pinned before the first WGMMA (no pending group yet).
                #pragma unroll
                for (uint32_t j = 0; j < 128; ++j) deep_gemm::ptx::warpgroup_fence_operand(accum[j]);
                full[stage].wait(phase);
                prev_sa0 = smem_sfa(stage)[r0], prev_sa1 = smem_sfa(stage)[r1];
                prev_g = __ldg(gate_sf), prev_u = __ldg(up_sf);
                if constexpr (kCfgs[V].ldsm) load_a(a_buf[0], stage);
                if constexpr (kCfgs[V].stagger) {
                    // Warpgroup 1 runs a half chunk first and last, so its drains fall in the middle
                    // of warpgroup 0's chunks.
                    constexpr uint32_t H = kUnroll / 2;
                    static_assert(H % 2 == 0 and kTileKB % kUnroll == 0, "even half chunks");
                    for (uint32_t kc = 0; kc < kTileKB;) {
                        if (wg == 1 and (kc == 0 or kc + H == kTileKB)) {
                            chunk(kc, std::integral_constant<uint32_t, H>{});
                            kc += H;
                        } else {
                            chunk(kc, std::integral_constant<uint32_t, kUnroll>{});
                            kc += kUnroll;
                        }
                    }
                } else {
                    static_assert(kTileKB % kUnroll == 0 and kUnroll % 2 == 0, "whole even chunks");
                    for (uint32_t kc = 0; kc < kTileKB; kc += kUnroll)
                        chunk(kc, std::integral_constant<uint32_t, kUnroll>{});
                }
            } else {
                // The serving loop: wait (group k) -> rescale all 128 -> issue (group k + 1).
                for (uint32_t kb = 0; kb < kTileKB; ++kb) {
                    if (kb == 0) {
                        full[stage].wait(phase);
                    } else if constexpr (kCfgs[V].rescale) {
                        #pragma unroll
                        for (uint32_t j = 0; j < 32; ++j) {
                            const float x0 = (j & 1) ? r0u : r0g, x1 = (j & 1) ? r1u : r1g;
                            accum[j * 4 + 0] *= x0, accum[j * 4 + 1] *= x0, accum[j * 4 + 2] *= x1, accum[j * 4 + 3] *= x1;
                        }
                    }
                    #pragma unroll
                    for (uint32_t j = 0; j < 128; ++j) deep_gemm::ptx::warpgroup_fence_operand(accum[j]);
                    deep_gemm::ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t k = 0; k < 4; ++k)
                        wgmma_ss(a_hi | (a_lo + stage * (A_STAGE / 16) + k * 2), b_hi | (b_lo + stage * (B_STAGE / 16) + k * 2), accum, std::make_index_sequence<128>{});
                    deep_gemm::ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t j = 0; j < 128; ++j) deep_gemm::ptx::warpgroup_fence_operand(accum[j]);
                    if (kb == 0) {
                        prev_sa0 = smem_sfa(stage)[r0], prev_sa1 = smem_sfa(stage)[r1];
                        prev_g = __ldg(gate_sf), prev_u = __ldg(up_sf);
                    }
                    const uint32_t next_stage = stage == STAGES - 1 ? 0 : stage + 1;
                    const uint32_t next_phase = phase ^ (next_stage == 0 ? 1u : 0u);
                    if (kb + 1 < kTileKB) next_ratios(kb + 1, next_stage, next_phase);
                    deep_gemm::ptx::warpgroup_wait<0>();
                    if (lane == 0) empty[stage].arrive();
                    stage = next_stage, phase = next_phase;
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
    __syncthreads();
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

using Kernel = void (*)(CUtensorMap, CUtensorMap, CUtensorMap, const float*, uint32_t*, float*, Stats*);
template <size_t... I>
static std::vector<Kernel> all_kernels(std::index_sequence<I...>) { return {mainloop<int(I)>...}; }

int main() {
    int sms = 0;
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    const uint32_t grid = 132;
    if (sms < int(grid)) { printf("FAIL need %u SMs, have %d\n", grid, sms); return 1; }
    static_assert(NUM_TILES % 132 == 0, "tiles per CTA");
    printf("DEVICE name=%s sms=%d\n", prop.name, sms);

    float fp8[256];
    for (int i = 0; i < 256; ++i) fp8[i] = e4m3_to_float(uint8_t(i));
    std::vector<uint8_t> a(size_t(M_TOTAL) * K), b(size_t(N_TOTAL) * K);
    for (size_t i = 0; i < a.size(); ++i) a[i] = random_fp8(i);
    for (size_t i = 0; i < b.size(); ++i) b[i] = random_fp8(i + (1ull << 40));
    std::vector<float> sfa(size_t(KB) * M_TOTAL), wsf(size_t(EXPERTS) * 2 * NUM_N_BLOCKS * KB);
    for (size_t i = 0; i < sfa.size(); ++i) sfa[i] = 0.5f + (mix(i + (2ull << 40)) % 1000) / 1000.0f;
    for (size_t i = 0; i < wsf.size(); ++i) wsf[i] = 0.5f + (mix(i + (3ull << 40)) % 1000) / 1000.0f;

    uint8_t *d_a, *d_b;
    float *d_sfa, *d_wsf, *d_tile0;
    uint32_t* d_hash;
    Stats* d_stats;
    CHECK(cudaMalloc(&d_a, a.size()));
    CHECK(cudaMalloc(&d_b, b.size()));
    CHECK(cudaMalloc(&d_sfa, sfa.size() * 4));
    CHECK(cudaMalloc(&d_wsf, wsf.size() * 4));
    CHECK(cudaMalloc(&d_tile0, BM * BN * 4));
    CHECK(cudaMalloc(&d_hash, size_t(NUM_TILES) * 256 * 4));
    CHECK(cudaMalloc(&d_stats, grid * sizeof(Stats)));
    CHECK(cudaMemcpy(d_a, a.data(), a.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b.data(), b.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_sfa, sfa.data(), sfa.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_wsf, wsf.data(), wsf.size() * 4, cudaMemcpyHostToDevice));
    const CUtensorMap tma_a = make_map(d_a, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, K, M_TOTAL, 128, BM, CU_TENSOR_MAP_SWIZZLE_128B);
    const CUtensorMap tma_b = make_map(d_b, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, K, N_TOTAL, 128, BN, CU_TENSOR_MAP_SWIZZLE_128B);
    const CUtensorMap tma_sfa = make_map(d_sfa, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, M_TOTAL, KB, BM, 1, CU_TENSOR_MAP_SWIZZLE_NONE);

    const std::vector<Kernel> kernels = all_kernels(std::make_index_sequence<NUM_VARIANTS>{});
    for (int v = 0; v < NUM_VARIANTS; ++v) {
        CHECK(cudaFuncSetAttribute(kernels[v], cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
        cudaFuncAttributes attr;
        CHECK(cudaFuncGetAttributes(&attr, kernels[v]));
        printf("RESOURCE variant=%s registers=%d local_bytes=%zu smem=%u\n", kCfgs[v].name, attr.numRegs, attr.localSizeBytes, SMEM_BYTES);
    }
    auto launch = [&](int v) {
        kernels[v]<<<grid, THREADS, SMEM_BYTES>>>(tma_a, tma_b, tma_sfa, d_wsf, d_hash, d_tile0, d_stats);
        CHECK(cudaGetLastError());
    };
    auto control_of = [](int v) { return kCfgs[v].kb == 12 ? SS_K12 : SS; };

    // Correctness: ss against an FP64 reference on tile 0 (the carried-scale FP8 accumulation over
    // 4096 K is within about 2e-3); every exact variant bit for bit against its control.
    std::vector<uint32_t> hash_ctl[2], hash_v(size_t(NUM_TILES) * 256);
    std::vector<float> tile_ctl[2], tile_v(BM * BN);
    bool ok = true;
    for (int v : {SS, SS_K12}) {
        const int c = v == SS ? 0 : 1;
        CHECK(cudaMemset(d_hash, 0, hash_v.size() * 4));
        launch(v);
        CHECK(cudaDeviceSynchronize());
        hash_ctl[c].resize(hash_v.size()), tile_ctl[c].resize(BM * BN);
        CHECK(cudaMemcpy(hash_ctl[c].data(), d_hash, hash_v.size() * 4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(tile_ctl[c].data(), d_tile0, BM * BN * 4, cudaMemcpyDeviceToHost));
    }
    {
        double num = 0, den = 0;
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
                num += (tile_ctl[0][r * BN + c] - ref) * (tile_ctl[0][r * BN + c] - ref), den += ref * ref;
            }
        const double rel = std::sqrt(num / den);
        printf("CHECK ss tile0 relative_l2=%.3e\n", rel);
        ok = ok and rel < 1e-2;
    }
    for (int v = 0; v < NUM_VARIANTS; ++v) {
        if (v == SS or v == SS_K12 or not kCfgs[v].rescale) continue;
        const int c = control_of(v) == SS ? 0 : 1;
        CHECK(cudaMemset(d_hash, 0, hash_v.size() * 4));
        launch(v);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hash_v.data(), d_hash, hash_v.size() * 4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(tile_v.data(), d_tile0, BM * BN * 4, cudaMemcpyDeviceToHost));
        const bool same = hash_v == hash_ctl[c] and tile_v == tile_ctl[c];
        printf("CHECK %s bitwise_equal_to_%s=%d\n", kCfgs[v].name, kCfgs[control_of(v)].name, int(same));
        ok = ok and same;
    }
    if (not ok) { printf("FAIL correctness\n"); return 1; }

    // Timing: rounds of all variants in rotating order; per-round ratios to the control.
    constexpr int ROUNDS = 40;
    std::vector<std::vector<float>> us(NUM_VARIANTS), loop_clk(NUM_VARIANTS), mhz(NUM_VARIANTS);
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    std::vector<Stats> stats(grid);
    for (int v = 0; v < NUM_VARIANTS; ++v) launch(v);
    CHECK(cudaDeviceSynchronize());
    for (int round = 0; round < ROUNDS; ++round) {
        for (int o = 0; o < NUM_VARIANTS; ++o) {
            const int v = (round + o) % NUM_VARIANTS;
            CHECK(cudaEventRecord(e0));
            launch(v);
            CHECK(cudaEventRecord(e1));
            CHECK(cudaEventSynchronize(e1));
            float ms = 0;
            CHECK(cudaEventElapsedTime(&ms, e0, e1));
            us[v].push_back(ms * 1000.0f);
            CHECK(cudaMemcpy(stats.data(), d_stats, grid * sizeof(Stats), cudaMemcpyDeviceToHost));
            double cyc = 0, ns = 0, loop = 0;
            for (const auto& s : stats) cyc += s.cycles, ns += s.ns, loop += s.loop_cycles;
            loop_clk[v].push_back(float(loop / (double(grid) * (NUM_TILES / grid) * kCfgs[v].kb)));
            mhz[v].push_back(float(cyc / ns * 1000.0));
        }
    }
    auto median = [](std::vector<float> x) { std::sort(x.begin(), x.end()); return x[x.size() / 2]; };
    for (int v = 0; v < NUM_VARIANTS; ++v) {
        std::vector<float> ratio;
        const int control = control_of(v);
        for (int round = 0; round < ROUNDS; ++round) ratio.push_back(us[v][round] / us[control][round]);
        printf("PASS variant=%s median_us=%.1f ratio_to_%s=%.4f loop_clk_per_kblock=%.1f sm_mhz=%.0f\n", kCfgs[v].name, median(us[v]), kCfgs[control].name, median(ratio), median(loop_clk[v]), median(mhz[v]));
    }
    return 0;
}
