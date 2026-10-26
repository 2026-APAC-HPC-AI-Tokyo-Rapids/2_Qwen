"""Apply probe-only switches to an extracted, pinned FA3 source tree."""
import sys
from pathlib import Path

hopper = Path(sys.argv[1])


def replace(file, old, new, count=1):
    path = hopper / file
    text = path.read_text()
    assert text.count(old) == count, (file, old)
    path.write_text(text.replace(old, new))


replace('flash_fwd_launch_template.h', 'static constexpr int kBlockM = Arch >= 90 ?', 'static constexpr int kBlockM = FA3_PROBE_M ? FA3_PROBE_M : Arch >= 90 ?', count=2)
replace('mainloop_fwd_sm90_tma_gmma_ws.hpp', 'static constexpr bool UseSchedulerBarrier = (IntraWGOverlap', 'static constexpr bool UseSchedulerBarrier = !FA3_PROBE_NOBARRIER && (IntraWGOverlap')
replace('softmax.h', '    template<typename Tensor1>\n    __forceinline__ __device__ void rescale_o', '''    template<typename Tensor1>
    __forceinline__ __device__ void probe_rescale_o(Tensor1 &acc_o, TensorT const &scores_scale) {
#if FA3_PROBE_LAZY
        bool unit = true;
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) { unit = unit && scores_scale(mi) == 1.0f; }
        bool skip = __all_sync(0xffffffff, unit);
#if FA3_PROBE_COUNTER
        if ((threadIdx.x & 31) == 0) {
            atomicAdd(fa3_probe_counts + 1, 1ULL);
            if (skip) { atomicAdd(fa3_probe_counts, 1ULL); }
        }
#endif
        if (skip) { return; }
#endif
        rescale_o(acc_o, scores_scale);
    }

    template<typename Tensor1>
    __forceinline__ __device__ void rescale_o''')
# The shared Prefill rescale wrapper is used at all four online-update sites.
replace('mainloop_fwd_sm90_tma_gmma_ws.hpp', 'softmax.rescale_o(acc, scales);', 'softmax.probe_rescale_o(acc, scales);')
