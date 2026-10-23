"""CPU checks of the emitted prefetch coordinates, JIT switch and probe policy."""
from __future__ import annotations

import ast
import json
import shutil
import statistics
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OVERLAY = ROOT / "patches/deep-gemm-overlay/sm90-decode-swapab-l2-prefetch.patch"
HOST = ROOT / "patches/deep-gemm-host/sm90-decode-l2-prefetch.patch"
BENCH = ROOT / "scripts/bench-firmus-megamoe.py"


def additions(path):
    return "\n".join(line[1:] for line in path.read_text().splitlines() if line.startswith("+") and not line.startswith("+++"))


def bench_functions():
    source = ast.parse(BENCH.read_text())
    names = {"sample_summary", "select_l2_prefetch", "load_decode_routing"}
    context = {"statistics": statistics, "json": json, "NUM_EXPERTS": 128, "TOPK": 8}
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(BENCH), "exec"), context)  # noqa: S102 - trusted CPU helpers; omit CUDA module initialization
    return context


@pytest.fixture(scope="module")
def coordinate_runner(tmp_path_factory):
    """Compile the actual overlay's predicates/loop, replacing only the PTX sink."""
    cxx = shutil.which("c++")
    assert cxx, "a CPU C++17 compiler is required for the coordinate fixture"
    directory = tmp_path_factory.mktemp("l2-prefetch-coordinates")
    added = additions(OVERLAY)
    start = added.index("                if constexpr (kSwapABActive and")
    end = added.index("            if constexpr (kSwapABActive)\n", start)
    body = added[start:end]
    instruction = next(line for line in body.splitlines() if "asm volatile" in line)
    assert 'cp.async.bulk.prefetch.tensor.2d.L2.global [%0, {%1, %2}]' in instruction
    body = body.replace(instruction, "                            emit(local_expert_idx, n_block_idx, prefetch_block, prefetch_n, prefetch_k);")
    release = added[end:]
    source = directory / "coordinates.cpp"
    source.write_text("""#include <algorithm>
#include <cstdint>
#include <iostream>
namespace cute { using std::min; bool elect_one_sync() { return true; } }
namespace sched { enum class BlockPhase { Linear1=1, Linear2=2 }; }
void emit(uint32_t e, uint32_t n, uint32_t k, uint32_t nc, uint32_t kc) { std::cout << e << ' ' << n << ' ' << k << ' ' << nc << ' ' << kc << '\\n'; }
template <bool kSwapABActive, uint32_t kDecodeL2PrefetchStages> void run() {
constexpr uint32_t kNumStages=8, BLOCK_N=128, BLOCK_K=128;
uint32_t ph, local_expert_idx, m_block_idx, n_block_idx;
while (std::cin >> ph >> local_expert_idx >> m_block_idx >> n_block_idx) {
bool decode_l2_prefetch_first_block = true;
// A later block for the same CTA must not emit a second set of hints.
for (unsigned later=0; later<2; ++later) {
const auto block_phase = static_cast<sched::BlockPhase>(ph);
const uint32_t num_k_blocks = ph==1 ? 32 : 12;
const uint32_t shape_n = ph==1 ? 3072 : 4096;
for (uint32_t k_block_idx=0; k_block_idx<num_k_blocks; ++k_block_idx) {
""" + body + "\n}\n" + release + """
}
}
}
int main(int argc, char** argv) {
const int stages=std::stoi(argv[1]);
if (argc>2) run<false,16>();
else if (stages==0) run<true,0>();
else if (stages==8) run<true,8>();
else if (stages==16) run<true,16>();
else return 2;
}
""")
    executable = directory / "coordinates"
    subprocess.run([cxx, "-std=c++17", "-O2", str(source), "-o", str(executable)], check=True, capture_output=True, text=True)
    return executable


@pytest.mark.parametrize("counts", [[0] * 32, [1] * 32, [64] * 32, [65] * 32, [128] * 32, [0, 1, 64, 65, 128] + [0] * 27, [128] + [0] * 31, [0] * 29 + [128, 65, 1]])
@pytest.mark.parametrize("stages", [0, 8, 16])
def test_first_cta_tiles_are_in_bounds_unique_and_not_repeated(coordinate_runner, counts, stages):
    # Independent task-space oracle: each wave's FC1 then FC2, expert/M/N order.
    fc1 = [(1, expert, m, n) for expert, count in enumerate(counts) for m in range((count + 63) // 64) for n in range(24)]
    fc2 = [(2, expert, m, n) for expert, count in enumerate(counts) for m in range((count + 63) // 64) for n in range(32)]
    initial = (fc1 + fc2)[:132]
    data = "".join(" ".join(map(str, block)) + "\n" for block in initial)
    result = subprocess.run([str(coordinate_runner), str(stages)], input=data, capture_output=True, text=True, check=True)
    actual = [tuple(map(int, line.split())) for line in result.stdout.splitlines()]
    expected = [(e, n, k, e * 3072 + n * 128, k * 128) for ph, e, m, n in initial if ph == 1 and m == 0 for k in range(8, min(32, 8 + stages))]
    assert actual == expected
    assert len(actual) == len(set(actual)), "M blocks may share weights; hint each tile once"
    for e, n, k, nc, kc in actual:
        assert 0 <= e < 32 and 0 <= n < 24 and 8 <= k < 32
        assert e * 3072 <= nc and nc + 128 <= (e + 1) * 3072
        assert kc >= 0 and kc + 128 <= 4096
    inactive = subprocess.run([str(coordinate_runner), "16", "nonswap"], input=data, capture_output=True, text=True, check=True)
    assert inactive.stdout == ""


def test_fc2_first_block_never_prefetches(coordinate_runner):
    result = subprocess.run([str(coordinate_runner), "16"], input="2 0 0 0\n2 31 0 31\n", capture_output=True, text=True, check=True)
    assert not result.stdout


def test_overlay_keeps_existing_barrier_code_and_defaults_to_zero():
    text = OVERLAY.read_text()
    assert not [line for line in text.splitlines() if line.startswith("-") and not line.startswith("---")]
    added = additions(OVERLAY)
    assert "uint32_t kDecodeL2PrefetchStages = 0," in added
    assert not any(word in added for word in ("arrive_and_expect_tx", "->arrive", "->wait", "atomicAdd"))
    assert not (ROOT / "patches/deep-gemm-bench").exists()
    config = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    assert config["deep_gemm_overlay_patches"][-1] == str(OVERLAY.relative_to(ROOT))


def test_host_env_is_read_per_generate_and_zero_preserves_argument(tmp_path):
    added = additions(HOST)
    logic = added[:added.index("    swap_ab_argument);")]
    cxx = shutil.which("c++")
    assert cxx
    source = tmp_path / "host-env.cpp"
    source.write_text("""#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
template <typename T> T get_env(const std::string& name, const T& fallback) { auto p=std::getenv(name.c_str()); return p ? T(p) : fallback; }
#define DG_HOST_ASSERT(x) do { if (!(x)) throw std::runtime_error("invalid stages"); } while(0)
namespace fmt { std::string format(const char*, const std::string& value) { return ", " + value; } }
struct Args { bool use_swap_ab; };
std::string generate(const Args& args) {
""" + logic + """
return swap_ab_argument;
}
int main() {
try {
unsetenv("DG_SM90_DECODE_L2_PREFETCH_STAGES"); std::cout << generate({true}) << '\\n';
for (auto value : {"0", "8", "16", "0"}) { setenv("DG_SM90_DECODE_L2_PREFETCH_STAGES",value,1); std::cout << generate({true}) << '\\n'; }
for (auto value : {"0", "8", "16", "invalid"}) { setenv("DG_SM90_DECODE_L2_PREFETCH_STAGES",value,1); std::cout << generate({false}) << '\\n'; }
setenv("DG_SM90_DECODE_L2_PREFETCH_STAGES","4",1);
try { generate({true}); return 3; } catch (const std::runtime_error&) {}
} catch (...) { return 2; }
}
""")
    executable = tmp_path / "host-env"
    subprocess.run([cxx, "-std=c++17", str(source), "-o", str(executable)], check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == ["true", "true", "true, 8", "true, 16", "true", *["false"] * 4]
    removed = [line[1:] for line in HOST.read_text().splitlines() if line.startswith("-") and not line.startswith("---")]
    assert removed == ['    args.use_swap_ab ? "true" : "false");']


def probe_case(control=(200, 200), eight=(196, 196), sixteen=(194, 194)):
    summary = bench_functions()["sample_summary"]
    return {"measurements": [{"stages": stage, "cache": "cold", **summary(list(values)), "blocks": [summary([value] * 30) for value in values]} for stage, values in [(0, control), (8, eight), (16, sixteen)]]}


def test_selection_requires_threshold_every_case_and_both_blocks():
    choose = bench_functions()["select_l2_prefetch"]
    assert choose([probe_case()])["selected_stages"] == 16
    assert choose([probe_case(eight=(196, 196), sixteen=(195, 201))])["selected_stages"] == 8
    assert choose([probe_case(eight=(198, 198), sixteen=(199, 199))])["selected_stages"] == 0
    assert choose([probe_case(), probe_case(eight=(200, 201), sixteen=(201, 201))])["selected_stages"] == 0
    assert choose([probe_case(eight=(196, 196), sixteen=(196, 196))])["selected_stages"] == 8
    assert choose([])["selected_stages"] == 0


def test_decode_routing_requires_decode_provenance_and_unreplicated_map(tmp_path):
    load = bench_functions()["load_decode_routing"]
    data = {"source": {"phase": "decode", "run_id": "test-fixture"}, "layers": [{"layer": 43, "counts": [1] * 128, "physical_to_logical_map": list(range(128))}]}
    path = tmp_path / "routing.json"
    path.write_text(json.dumps(data))
    assert load(path) == data
    data["source"]["phase"] = "prefill-proxy"
    path.write_text(json.dumps(data))
    assert load(path) == data
    data["source"]["phase"] = "prefill"
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        load(path)
    data["source"]["phase"] = "decode"
    data["layers"][0]["physical_to_logical_map"] = list(range(140))
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        load(path)


def test_probe_capture_timing_and_interleaving_contract():
    tree = ast.parse(BENCH.read_text())
    tune = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "tune_l2_prefetch")
    source = ast.get_source_segment(BENCH.read_text(), tune)
    cold_capture = source[source.index('with torch.cuda.graph(graph):'):source.index('graphs[stages, cache] =')]
    assert cold_capture.index("read()") < cold_capture.index("pre_dispatch()") < cold_capture.index("start.record()") < cold_capture.index("moe()") < cold_capture.index("end.record()")
    assert "external=True" in source
    assert "for block in range(2)" in source
    cold_loop = source[source.index("# Capture fixes"):source.index("for stages in L2_PREFETCH_STAGES:\n                        durations")]
    assert cold_loop.index("for _ in range(ITERATIONS)") < cold_loop.index("for stages in L2_PREFETCH_STAGES")
    assert cold_loop.index("end.synchronize()") < cold_loop.index("start.elapsed_time(end)")
    assert "block_median_spread_us" in source
