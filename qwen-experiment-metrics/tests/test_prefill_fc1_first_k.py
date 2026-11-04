"""Execute the actual overlaid K-loop bodies on CPU, without git history or CUDA."""
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HEADER = "deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh"


def loop(source, name):
    start = source.index(f"                    [[maybe_unused]] auto {name} =")
    end = source.index("\n                    };", start) + len("\n                    };")
    return source[start:end]


def harness(control, candidate):
    prefix = r'''
#include <cassert>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>
#define CUTLASS_LAMBDA_FUNC_INLINE
#define DG_STATIC_ASSERT static_assert
std::vector<std::string> events;
uint32_t current_k;
float* base;
void event(std::string name, uint32_t a, uint32_t b = 0, uint32_t c = 0) { events.push_back(name + ":" + std::to_string(a) + ":" + std::to_string(b) + ":" + std::to_string(c)); }
namespace ptx {
void warpgroup_fence_operand(float&) {}
void warpgroup_arrive() { event("issue", current_k); }
void warpgroup_commit_batch() { event("commit", current_k); }
template <int N> void warpgroup_wait() { event("wait", N); }
}
struct MMA {
    static constexpr int kNumAccum = 64, K = 32;
    static void wgmma(uint64_t a, uint64_t b, float* out, bool accumulate) { event("mma", current_k, (out - base) / 64, accumulate); event("desc", a, b); }
};
namespace mma::sm90 { template <int N> struct FP8MMASelector { using type = MMA; }; }
struct Barrier { uint32_t idx; void arrive() { event("release", idx); } };
template <bool Candidate> std::vector<std::string> run(uint32_t initial_stage, uint32_t phase, uint32_t lane_idx) {
    using WGMMA = MMA;
    using a_dtype_t = uint8_t;
    using b_dtype_t = uint8_t;
    constexpr uint32_t WG_BLOCK_N = 256, kAccumPerThread = 128, BLOCK_K = 128, kNumStages = 4;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = 16384, SMEM_B_SIZE_PER_STAGE = 32768;
    uint32_t stage_idx = initial_stage;
    uint32_t desc_a_lo = 0x1000, desc_b_lo = 0x10000;
    uint64_t desc_a_hi = 0, desc_b_hi = 0;
    float final_accum[128] = {};
    base = final_accum;
    events.clear(); current_k = 0;
    Barrier barriers[4] = {{0}, {1}, {2}, {3}};
    Barrier* empty_barriers[4] = {barriers, barriers + 1, barriers + 2, barriers + 3};
    auto next_stage_and_phase = [&](uint32_t& next_stage, uint32_t& next_phase) { next_stage = (stage_idx + 1) % 4; next_phase = phase ^ (next_stage == 0); };
    auto advance_pipeline = [&](uint32_t& k) { stage_idx = (stage_idx + 1) % 4; phase ^= stage_idx == 0; ++k; current_k = k; };
'''
    suffix = r'''
    auto rescale = [&](uint32_t h) { event("rescale", current_k, h); };
    auto after = [&](uint32_t k) { event("after", k); if (k == 2) event("publish", k); };
    auto prefetch = [&](uint32_t k, uint32_t stage, uint32_t phase) { event("prefetch", k, stage, phase); };
    if constexpr (Candidate) first_k_split_loop(std::integral_constant<uint32_t, 32>{}, std::integral_constant<uint32_t, 8>{}, rescale, after, prefetch);
    else split_k_loop(std::integral_constant<uint32_t, 32>{}, std::integral_constant<uint32_t, 8>{}, rescale, after, prefetch);
    assert(current_k == 32 && stage_idx == initial_stage);
    return events;
}
int main() {
    for (uint32_t stage = 0; stage < 4; ++stage) for (uint32_t phase = 0; phase < 2; ++phase) for (uint32_t lane : {0, 1}) {
        auto control = run<false>(stage, phase, lane), candidate = run<true>(stage, phase, lane);
        int rescale = 0, mma = 0, init = 0, publish = 0, release = 0, prefetch = 0;
        for (auto& e : candidate) {
            rescale += e.starts_with("rescale:"); mma += e.starts_with("mma:");
            init += e.starts_with("mma:0:") && e.ends_with(":0");
            publish += e.starts_with("publish:"); release += e.starts_with("release:"); prefetch += e.starts_with("prefetch:");
        }
        assert(rescale == 62 && mma == 256 && init == 2 && publish == 1 && release == (lane == 0 ? 32 : 0) && prefetch == 31);
        std::vector<std::string> expected;
        for (auto e : control) {
            if (e.starts_with("rescale:0:")) continue;
            if (e.starts_with("mma:0:")) {
                // Only the first K32 in each half changes scale-d.
                static_cast<void>(e);
            }
            expected.push_back(e);
        }
        for (auto& e : candidate) if (e.starts_with("mma:0:") && e.ends_with(":0")) e.back() = '1';
        assert(expected == candidate);
    }
}
'''
    return prefix + loop(control, "split_k_loop") + "\n" + loop(candidate, "first_k_split_loop") + suffix


def test_actual_overlay_stage_publish_and_half_initialization(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native C++ compiler unavailable")
    fixture = ROOT / "qwen-experiment-metrics/tests/fixtures/prefill-r32-control.cuh.gz"
    control = gzip.decompress(fixture.read_bytes()).decode()
    assert hashlib.sha256(control.encode()).hexdigest() == "fbfaa6be207b26dd7127aae406204b44668b4341ff5ce9e2928510c932e57852"
    header = tmp_path / HEADER
    header.parent.mkdir(parents=True)
    header.write_text(control)
    patch = ROOT / "patches/deep-gemm-overlay/sm90-prefill-fc1-first-k.patch"
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "apply", str(patch)], cwd=tmp_path, check=True, capture_output=True)
    candidate = header.read_text()
    # FC2, the epilogues and all arrival publication remain the exact original source.
    tail = "                            postscale_l1_final(prev_scale_a_0, prev_scale_a_1, prev_gate_sf, prev_up_sf);"
    assert control[control.index(tail):] == candidate[candidate.index(tail):]
    cpp = tmp_path / "schedule.cpp"
    cpp.write_text(harness(control, candidate))
    subprocess.run([compiler, "-std=c++20", "-Wno-unknown-pragmas", str(cpp), "-o", str(tmp_path / "schedule")], check=True, capture_output=True)
    subprocess.run([str(tmp_path / "schedule")], check=True, capture_output=True)


def test_fc1_overlay_remains_and_runtime_is_pinned():
    config = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    assert config["deep_gemm_overlay_patches"].index("patches/deep-gemm-overlay/sm90-prefill-fc1-first-k.patch") < config["deep_gemm_overlay_patches"].index("patches/deep-gemm-overlay/sm90-prefill-dense-first-k-r37.patch")
    assert "megamoe_fc1_first_k_gate" not in config["prefill"]
    assert "sm90-prefill-fc1-first-k.patch" in (ROOT / "patches/deep-gemm-host/series").read_text().split()
    assert '"DG_SM90_PREFILL_FC1_FIRST_K": "1"' in (ROOT / "scripts/prepare-firmus-pp4-settings.py").read_text()
