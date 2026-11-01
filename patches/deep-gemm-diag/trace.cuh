#pragma once
#include <cstdint>
#include <cutlass/cutlass.h>

// Private ABI. No serving header includes this file. Globaltimer is in ns;
// no cycle values from different SMs are stitched together.
namespace deep_gemm::r18 {
constexpr uint64_t magic = 0x18314d4f45444941ull;
constexpr uint32_t capacity = 2048, words = 32 + capacity * 8, warps = 12;
CUTLASS_DEVICE uint64_t now() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    return t;
}
struct Trace {
    volatile uint64_t* p;
    bool lead;
    CUTLASS_DEVICE Trace(const void* end, uint32_t tokens) {
        auto base = reinterpret_cast<uint64_t*>((reinterpret_cast<uint64_t>(end) + 4095) / 4096 * 4096);
        p = base + (blockIdx.x * warps + threadIdx.x / 32) * words;
        lead = threadIdx.x % 32 == 0;
        if (lead) {
            uint32_t sm;
            asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
            p[0] = magic; p[1] = 0; p[2] = sm; p[3] = tokens; p[4] = 0;
            // Context: phase, tile serial, valid M, expert, pool block, N block.
            for (int i = 5; i < 32; ++i) p[i] = 0;
            event(1);
        }
    }
    CUTLASS_DEVICE bool math_lead() const { return (threadIdx.x / 32 == 4 or threadIdx.x / 32 == 8); }
    CUTLASS_DEVICE bool detail() const { return lead and math_lead() and p[11]; }
    CUTLASS_DEVICE void event(uint32_t id, uint64_t arg = 0) const {
        if (lead) {
            const auto t = now();
            const uint64_t i = p[1];
            if (i < capacity) {
                auto q = p + 32 + i * 8;
                q[0] = t; q[1] = id; q[2] = p[5]; q[3] = p[6];
                q[4] = p[7]; q[5] = p[8]; q[6] = p[9]; q[7] = arg;
                p[1] = i + 1;
            } else p[4] = 1;
        }
    }
    CUTLASS_DEVICE void sample(uint32_t id, uint64_t arg = 0) const {
        if (detail()) { p[13] = arg; event(id, arg); }
    }
    CUTLASS_DEVICE void tile(uint32_t phase, uint32_t rows, uint32_t expert, uint32_t pool, uint32_t n) const {
        if (lead) {
            p[5] = phase; p[6] += 1; p[7] = rows; p[8] = expert; p[9] = pool; p[10] = n;
            const uint32_t bit = (phase - 1) * 2 + (rows < 128);
            p[11] = (p[12] & (1u << bit)) == 0;
            p[12] |= 1u << bit;
            if (math_lead()) event(2, n);
        }
    }
    // Full/empty sums are nested within their role's mainloop, never additive
    // components of kernel latency. Every original wait still executes.
    template <typename Barrier>
    CUTLASS_DEVICE void wait(Barrier barrier, uint32_t phase, uint32_t kind) const {
        if (kind == 0 and detail()) event(19, p[13]);
        const auto t = now();
        barrier->wait(phase);
        const auto end = now();
        if (kind == 0 and detail()) event(20, p[13]);
        if (lead) {
            const uint32_t slot = 16 + (p[5] == 2 ? 4 : 0) + kind;
            p[slot] += end - t;
            p[slot + 8] += 1;
        }
    }
    CUTLASS_DEVICE ~Trace() { event(15); }
};
}
