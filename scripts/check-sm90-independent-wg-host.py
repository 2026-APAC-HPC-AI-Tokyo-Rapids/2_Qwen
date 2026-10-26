"""Run the pinned, patched host selector and kernel layout on CPU (no CUDA).

Only Tensor metadata, device attributes, CUDA descriptor encoding and JIT
build/launch are substituted. Selector functions, descriptor construction,
Args population, JIT text generation and kernel constants come from source.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "1c8138a59b3df5dac50b76b730834ed57982fdfb"
TOKENS = (0, 1, 4, 7, 8, 9, 16, 17, 25, 32, 33, 41, 49, 57, 64, 65, 96, 120, 127, 128, 129, 132, 136, 140, 144, 145, 160, 161, 192, 193, 255, 256, 279, 280, 281, 303, 304, 1024, 2048, 4096)
IMAGE_SHA256 = "ff7a58cca00aa42a90386d12b0437b5d8caf35eb0ad082a8a1a0b4b74001a2a1"
BASELINE_SHA256 = "77ee7caea6e5736803b6034f83575f9a846d30c54ce4970ced7c65450a9be497"
KERNEL = Path("deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh")
SOURCE_FILES = [
    "csrc/jit_kernels/heuristics/sm90_mega_moe.hpp", "csrc/jit_kernels/heuristics/mega_moe.hpp", "csrc/jit_kernels/heuristics/sm90.hpp",
    "csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp", "csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp", "csrc/jit_kernels/impls/runtime_utils.hpp",
    "csrc/utils/math.hpp", "csrc/utils/system.hpp", "deep_gemm/include/deep_gemm/common/math.cuh", "deep_gemm/include/deep_gemm/layout/mega_moe.cuh", "deep_gemm/include/deep_gemm/layout/sym_buffer.cuh",
]


def strip_includes(source):
    return re.sub(r"^#(?:include|pragma once).*\n", "", source, flags=re.MULTILINE)


def function(source, start):
    """Extract an unchanged function (including template line if requested)."""
    begin = source.index(start)
    opening = source.index("{", begin)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end] + "\n"


STUBS = r'''
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <vector>
#define FMT_HEADER_ONLY
#include <fmt/format.h>
#define CUTLASS_HOST_DEVICE
#define CUTLASS_DEVICE
#define DG_STATIC_ASSERT static_assert
#define DG_HOST_ASSERT(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
#define DG_UNIFIED_ASSERT DG_HOST_ASSERT
#define DG_DEVICE_ASSERT DG_HOST_ASSERT
#define DG_HOST_UNREACHABLE(x) throw std::runtime_error(x)
#define DG_CUDA_DRIVER_CHECK(x) DG_HOST_ASSERT((x) == 0)
#define DG_CUDA_UNIFIED_CHECK(x) DG_HOST_ASSERT((x) == 0)
#define CUDA_VERSION 13000
using nv_bfloat16 = uint16_t;
namespace cutlass { using float_e4m3_t = uint8_t; }
namespace cute { template<class T> using numeric_limits = std::numeric_limits<T>; namespace UMMA { enum class Major { MN, K }; } }
namespace at { enum ScalarType { Int, Float, BF16, FP8, Int8 }; }
namespace torch {
constexpr auto kInt=at::Int, kFloat=at::Float, kBFloat16=at::BF16, kFloat8_e4m3fn=at::FP8, kInt8=at::Int8;
struct Tensor {
    int64_t rows=0, columns=0; at::ScalarType dtype=at::FP8;
    int64_t size(int index) const { DG_HOST_ASSERT(index == 0); return rows; }
    int64_t stride(int index) const { DG_HOST_ASSERT(index == -2); return columns; }
    int64_t element_size() const { return dtype == at::Float ? 4 : dtype == at::BF16 ? 2 : 1; }
    at::ScalarType scalar_type() const { return dtype; }
    template<class T=void> T* data_ptr() const { return nullptr; }
};
}
using cuuint64_t=uint64_t; using cuuint32_t=uint32_t;
enum CUtensorMapDataType { CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, CU_TENSOR_MAP_DATA_TYPE_INT32, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_DATA_TYPE_UINT8, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B };
enum CUtensorMapSwizzle { CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B };
constexpr int CU_TENSOR_MAP_INTERLEAVE_NONE=0, CU_TENSOR_MAP_L2_PROMOTION_L2_256B=0, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE=0;
struct CUtensorMap { uint64_t global[2], stride; uint32_t box[2]; int dtype, swizzle; };
int lazy_cuTensorMapEncodeTiled(CUtensorMap* out, CUtensorMapDataType dtype, int dimensions, void*, const cuuint64_t* global, const cuuint64_t* strides, const cuuint32_t* box, const cuuint32_t* element, int, CUtensorMapSwizzle swizzle, int, int) {
    DG_HOST_ASSERT(dimensions == 2 && element[0] == 1 && element[1] == 1);
    *out = {{global[0],global[1]},strides[0],{box[0],box[1]},dtype,swizzle}; return 0;
}
namespace deep_gemm {
struct DeviceRuntime { int get_num_sms() const { return 132; } };
DeviceRuntime device; DeviceRuntime* device_runtime=&device;
struct LaunchArgs { std::pair<int,int> grid_dim; int threads, smem, cluster; LaunchArgs(int sms,int threads,int smem,int cluster):grid_dim(sms,1),threads(threads),smem(smem),cluster(cluster){} };
using KernelHandle=int; using LaunchConfigHandle=int;
template<class... T> int launch_kernel(T...) { throw std::runtime_error("GPU launch forbidden"); }
std::string emitted_code;
struct Compiler { int build(const char*, const std::string& code) { emitted_code=code; return 0; } };
Compiler instance; Compiler* compiler=&instance;
template<class Args> void record(const Args& args);
template<class Derived> struct LaunchRuntime {
    template<class Args> static std::string generate(const Args& args) { return Derived::generate_impl(args); }
    template<class Args> static void launch(int, const Args& args) { record(args); }
};
}
'''


HOST_MAIN = r'''
namespace deep_gemm {
int prefill_env, independent_env;
template<class Args> void record(const Args& a) {
    const auto& c=a.config;
    std::cout << a.num_tokens << '\t' << a.num_experts << '\t' << prefill_env << '\t' << independent_env;
    for (auto v : {c.block_m,c.block_n,c.block_k,c.num_stages,c.smem_size,c.num_dispatch_threads,c.num_non_epilogue_threads,c.num_epilogue_threads,c.num_experts_per_wave,c.num_max_pool_tokens,c.num_padded_sf_pool_tokens,int(a.use_swap_ab),int(a.reuse_accum_as_final),int(a.l2_arrival_counter),int(a.l2_epilogue_requires_full_sync),int(a.split_phase_hot_path),a.epilogue_registers,a.launch_args.threads,a.launch_args.smem}) std::cout << '\t' << v;
    for (const auto& map : {a.tensor_map_l1_acts,a.tensor_map_l1_acts_sf,a.tensor_map_l1_weights,a.tensor_map_l1_output,a.tensor_map_l2_acts,a.tensor_map_l2_acts_sf,a.tensor_map_l2_weights}) {
        for (auto v : {map.global[0],map.global[1],map.stride,uint64_t(map.box[0]),uint64_t(map.box[1]),uint64_t(map.dtype),uint64_t(map.swizzle)}) std::cout << '\t' << v;
    }
    const std::string needle="sm90_fp8_mega_moe_impl<";
    auto start=emitted_code.find(needle)+needle.size();
    auto text=emitted_code.substr(start,emitted_code.rfind(">);")-start);
    text.erase(std::remove_if(text.begin(),text.end(),[](char c){return std::isspace(static_cast<unsigned char>(c));}),text.end());
    std::cout << '\t' << text << '\n';
}
}
int main() {
    using namespace deep_gemm;
    unsetenv("DG_JIT_DEBUG"); unsetenv("DG_PRINT_CONFIGS"); setenv("DG_SM90_FP8_SWAP_AB","1",1);
    for(int experts : {128,140,152}) for(int tokens : {HOST_TOKEN_CASES}) for(int prefill : {0,1}) for(int independent : {0,1}) {
        prefill_env=prefill; independent_env=independent;
        setenv("DG_SM90_PREFILL_2WG",prefill ? "1":"0",1); setenv("DG_SM90_DECODE_INDEPENDENT_WG",independent ? "1":"0",1);
        int pool=layout::get_num_max_pool_tokens(4,4224,8,experts/4), sf=layout::get_num_sf_ring_tokens(pool,layout::kMinCandidateBlockM);
        torch::Tensor y{tokens,4096,torch::kBFloat16}, l1{pool,4096,torch::kFloat8_e4m3fn}, l1sf{sf,4096/128,torch::kFloat}, l2{pool,1536,torch::kFloat8_e4m3fn}, l2sf{sf,1536/64,torch::kFloat}, w1{experts/4*3072,4096,torch::kFloat8_e4m3fn}, w2{experts/4*4096,1536,torch::kFloat8_e4m3fn}, wsf{0,0,torch::kFloat};
        sm90_fp8_mega_moe(y,l1,l1sf,l2,l2sf,w1,w2,wsf,wsf,std::nullopt,std::vector<int64_t>(4),0,4224,experts/4,tokens,8,4096,1536,std::numeric_limits<float>::infinity(),true);
    }
}
'''


def host_translation_unit(source):
    text = {name: (source / name).read_text() for name in SOURCE_FILES}
    math = strip_includes(text["deep_gemm/include/deep_gemm/common/math.cuh"].split("#ifdef DG_IN_CUDA_COMPILATION")[0]) + "}\n"
    host_math = text["csrc/utils/math.hpp"]
    utilities = function(host_math, "template <typename T>\nstatic T ceil_div") + function(host_math, "template <typename T>\nstatic constexpr T align") + function(host_math, "static int get_tma_aligned_size")
    utilities += function(text["csrc/utils/system.hpp"], "template <typename dtype_t>\nstatic dtype_t get_env")
    utils = text["csrc/jit_kernels/impls/runtime_utils.hpp"]
    for start in ("static std::string to_string(const float&", "static CUtensorMapDataType aten_dtype", "static CUtensorMapSwizzle mode_into", "static CUtensorMap make_tma_2d_desc", "static CUtensorMap make_tma_sf_desc"):
        utilities += function(utils, start)
    common = text["csrc/jit_kernels/heuristics/mega_moe.hpp"]
    utilities += function(common, "static int get_num_wave_pool_tokens") + function(common, "static int get_num_experts_per_wave_for_mega_moe(")
    capacity = re.search(r"static constexpr int smem_capacity = \d+;", text["csrc/jit_kernels/heuristics/sm90.hpp"]).group()
    return STUBS + math + strip_includes(text["deep_gemm/include/deep_gemm/layout/mega_moe.cuh"]) + strip_includes(text["deep_gemm/include/deep_gemm/layout/sym_buffer.cuh"]) + "namespace deep_gemm { constexpr auto kPackedFP4=torch::kInt8; struct SM90ArchSpec {" + capacity + "};\n" + utilities + "}\n" + strip_includes(text["csrc/jit_kernels/heuristics/sm90_mega_moe.hpp"]) + strip_includes(text["csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp"]) + HOST_MAIN.replace("HOST_TOKEN_CASES", ",".join(map(str, TOKENS)))


def prepare_source(repo, destination):
    for name in SOURCE_FILES:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.check_output(["git", "-C", str(repo), "show", f"{REVISION}:{name}"]))
    patches = (ROOT / "patches/deep-gemm-host/series").read_text().split()
    for patch in patches:
        path = ROOT / "patches/deep-gemm-host" / patch
        subprocess.run(["git", "apply", "--check", str(path)], cwd=destination, check=True)
        subprocess.run(["git", "apply", str(path)], cwd=destination, check=True)
    return patches


HOST_FIELDS = ("block_m", "block_n", "block_k", "stages", "smem", "dispatch_threads", "loader_threads", "math_threads", "experts_per_wave", "pool_tokens", "sf_pool_tokens", "swap_ab", "reuse_accum", "arrival_counter", "epilogue_full_sync", "split_phase", "epilogue_registers", "launch_threads", "launch_smem")
MAP_NAMES = ("l1_acts", "l1_sf", "l1_weights", "l1_output", "l2_acts", "l2_sf", "l2_weights")
KERNEL_NAMES = ("fp8_token_layout", "kPrefill2WG", "kIndependentWG", "kTaskQueues", "kQueueStages", "kSplitNWarpgroups", "kSplitMNWarpgroups", "kWarpgroupSplitM", "kWarpgroupSplitN", "WG_BLOCK_M", "WG_BLOCK_N", "kNumCombineWarps", "LOAD_BLOCK_M", "LOAD_BLOCK_N", "L1_OUT_BLOCK_N", "WG_L1_OUT_BLOCK_N", "kSplitNSharesSF", "kSwapABEligible", "kSwapABActive", "kSharedMemoryAlignment", "SMEM_EXPERT_COUNT_SIZE", "SMEM_SEND_BUFFER_SIZE", "SMEM_A_SIZE_PER_STAGE", "SMEM_B_SIZE_PER_STAGE", "SMEM_SFA_SIZE_PER_STAGE", "SMEM_CD_L1_SIZE", "SMEM_CD_L2_SIZE", "SMEM_CD_SWAP_L1_FP32_SIZE", "SMEM_CD_SWAP_L1_FP8_SIZE", "SMEM_CD_SWAP_L1_SIZE", "SMEM_CD_BASE_SIZE", "SMEM_CD_PER_QUEUE", "SMEM_CD_SIZE", "SMEM_BEFORE_BARRIER_SIZE")


def kernel_translation_unit(source, header, templates):
    # Compile the declarations from the candidate itself, instantiated with the
    # actual host-generated argument text, rather than a Python copy of formulas.
    math = strip_includes((source / "deep_gemm/include/deep_gemm/common/math.cuh").read_text().split("#ifdef DG_IN_CUDA_COMPILATION")[0]) + "}\n"
    layout = strip_includes((source / "deep_gemm/include/deep_gemm/layout/mega_moe.cuh").read_text())
    begin = header.index("template <\n    uint32_t kNumMaxTokensPerRank,")
    declaration = header[begin:header.index("CUTLASS_GLOBAL", begin)]
    constants = []
    for name in KERNEL_NAMES:
        found = re.findall(r"constexpr\s+(?:auto|bool|uint32_t)\s+" + name + r"\s*=[^;]+;", header)
        assert len(found) == 1, name
        constants.append(found[0])
    guard = re.search(r'DG_STATIC_ASSERT\(not kIndependentWG or \(not kSplitPhaseHotPath[^;]+;', header).group()
    body = declaration + "void inspect_kernel() { using a_dtype_t=uint8_t; using b_dtype_t=uint8_t;\n" + "\n".join(constants) + guard
    # The final allocation covers the SF region and all barriers after it.
    body += r'''
    const uint32_t end = SMEM_BEFORE_BARRIER_SIZE + kNumStages * SMEM_SFA_SIZE_PER_STAGE + (kNumDispatchWarps + 2 * kNumStages + 2 * kNumCombineWarps) * sizeof(uint64_t);
    const uint32_t output_n = (kIndependentWG or kSplitNSharesSF) ? L1_OUT_BLOCK_N : WG_L1_OUT_BLOCK_N;
    const uint32_t output_m = (kIndependentWG or kSplitNSharesSF) ? BLOCK_M : WG_BLOCK_M;
    for (auto value : {end,uint32_t(kIndependentWG),kTaskQueues,kQueueStages,WG_BLOCK_M,WG_BLOCK_N,output_n,output_m,SMEM_CD_PER_QUEUE,SMEM_A_SIZE_PER_STAGE,SMEM_B_SIZE_PER_STAGE,SMEM_SFA_SIZE_PER_STAGE,LOAD_BLOCK_M,LOAD_BLOCK_N,BLOCK_K}) std::cout << value << " ";
    std::cout << "\n";
}
'''
    main = "int main() {\n" + "\n".join("deep_gemm::inspect_kernel<" + arguments + ">();" for arguments in templates) + "\n}\n"
    return STUBS + math + layout + "namespace deep_gemm {\n" + body + "}\n" + main


def decode_rows(output):
    rows = []
    for line in output.splitlines():
        fields = line.split("\t")
        numbers = list(map(int, fields[:-1]))
        assert len(numbers) == 4 + len(HOST_FIELDS) + 7 * len(MAP_NAMES)
        row = {"input": dict(zip(("tokens", "experts", "prefill_2wg_env", "independent_wg_env"), numbers[:4], strict=True)), "host": dict(zip(HOST_FIELDS, numbers[4:23], strict=True)), "tma": {}, "jit_template_args": fields[-1]}
        for i, name in enumerate(MAP_NAMES):
            values = numbers[23 + i * 7:23 + (i + 1) * 7]
            row["tma"][name] = {"global": values[:2], "stride_bytes": values[2], "box": values[3:5], "dtype": ("TFLOAT32", "INT32", "FLOAT32", "BFLOAT16", "UINT8", "16U4_ALIGN16B", "16U4_ALIGN8B")[values[5]], "swizzle": ("NONE", "32B", "64B", "128B", "128B_ATOM_32B")[values[6]]}
        rows.append(row)
    assert len(rows) == len(TOKENS) * 3 * 4
    return rows


def check_jit_arguments(row, header):
    start = header.index("template <\n    uint32_t kNumMaxTokensPerRank,")
    declaration = header[start:header.index("CUTLASS_GLOBAL", start)]
    names = re.findall(r"(?:uint32_t|float|bool)\s+(\w+)", declaration)
    values = row["jit_template_args"].split(",")
    actual = dict(zip(names[:len(values)], values, strict=True))
    h = row["host"]
    expected = {"kNumMaxTokensPerRank": 4224, "kHidden": 4096, "kIntermediateHidden": 1536, "kNumExperts": row["input"]["experts"], "kNumTopk": 8, "kNumExpertsPerWave": h["experts_per_wave"], "BLOCK_M": h["block_m"], "BLOCK_N": h["block_n"], "BLOCK_K": h["block_k"], "kNumMaxPoolTokens": h["pool_tokens"], "kNumPaddedSFPoolTokens": h["sf_pool_tokens"], "kNumStages": h["stages"], "kNumDispatchThreads": h["dispatch_threads"], "kNumNonEpilogueThreads": h["loader_threads"], "kNumEpilogueThreads": h["math_threads"], "kNumSMs": 132, "kNumRanks": 4, "kActivationClamp": "cute::numeric_limits<float>::infinity()", "kFastMath": "true", "kEpilogueRegisterBudget": h["epilogue_registers"]}
    for parameter, field in (("kReuseAccumAsFinal", "reuse_accum"), ("kL2ArrivalCounter", "arrival_counter"), ("kL2EpilogueRequiresFullSync", "epilogue_full_sync"), ("kSplitPhaseHotPath", "split_phase"), ("kFP8SwapAB", "swap_ab")):
        expected[parameter] = "true" if h[field] else "false"
    assert actual == {name: str(value) for name, value in expected.items()}, (row["input"], actual, expected)


def compare_contract(row, values):
    names = ("smem", "independent_wg", "queues", "queue_stages", "wg_m", "wg_n", "output_n", "output_m", "cd_bytes_per_queue", "a_bytes_per_stage", "b_bytes_per_stage", "sf_bytes_per_stage", "load_m", "load_n", "block_k")
    kernel = dict(zip(names, values, strict=True))
    host, maps, inputs = row["host"], row["tma"], row["input"]
    assert host["smem"] == host["launch_smem"] == kernel["smem"], row
    assert host["launch_threads"] == host["dispatch_threads"] + host["loader_threads"] + host["math_threads"]
    assert host["smem"] <= 232448
    assert maps["l1_output"]["box"] == [kernel["output_n"], kernel["output_m"]], row
    for key in ("l1_acts", "l2_acts"):
        assert maps[key]["box"] == [kernel["block_k"], kernel["load_m"]], row
    for key in ("l1_weights", "l2_weights"):
        assert maps[key]["box"] == [kernel["block_k"], min(kernel["load_n"], 256)], row
    for key in ("l1_sf", "l2_sf"):
        assert maps[key]["box"] == [host["block_m"], 1], row
        assert maps[key]["stride_bytes"] == host["sf_pool_tokens"] * 4, row
    expected_independent = inputs["experts"] == 128 and 0 < inputs["tokens"] <= 192 and inputs["independent_wg_env"] == 1
    assert bool(kernel["independent_wg"]) == expected_independent, row
    if kernel["independent_wg"]:
        assert not any(host[name] for name in ("reuse_accum", "arrival_counter", "split_phase")), row
        assert host["stages"] == 6 and kernel["queues"] == 2 and kernel["queue_stages"] == 3, row
    return kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--fmt-include", type=Path, required=True)
    parser.add_argument("--image-header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compiler = shutil.which("g++")
    assert compiler
    with tempfile.TemporaryDirectory(prefix="independent-wg-host-") as directory:
        temp = Path(directory)
        source = temp / "source"
        patches = prepare_source(args.source_repo, source)
        cpp = temp / "host.cpp"
        cpp.write_text(host_translation_unit(source))
        subprocess.run([compiler, "-std=c++20", "-O0", "-I" + str(args.fmt_include), str(cpp), "-o", str(temp / "host")], check=True)
        output = subprocess.check_output([str(temp / "host")], text=True)
        rows = decode_rows(output)
        overlay = temp / "overlay"
        (overlay / KERNEL).parent.mkdir(parents=True)
        assert hashlib.sha256(args.image_header.read_bytes()).hexdigest() == IMAGE_SHA256
        shutil.copy2(args.image_header, overlay / KERNEL)
        overlay_patches = json.loads((ROOT / "configs/firmus-framework.json").read_text())["deep_gemm_overlay_patches"]
        for patch in overlay_patches:
            # This hash identifies the input to Decode IWG, even when later overlays follow.
            if Path(patch).name == "sm90-decode-independent-wg.patch":
                assert hashlib.sha256((overlay / KERNEL).read_bytes()).hexdigest() == BASELINE_SHA256
            subprocess.run(["git", "apply", "--check", str(ROOT / patch)], cwd=overlay, check=True)
            subprocess.run(["git", "apply", str(ROOT / patch)], cwd=overlay, check=True)
        header = (overlay / KERNEL).read_text()
        templates = list(dict.fromkeys(row["jit_template_args"] for row in rows))
        cpp.write_text(kernel_translation_unit(source, header, templates))
        subprocess.run([compiler, "-std=c++20", "-O0", "-I" + str(args.fmt_include), str(cpp), "-o", str(temp / "kernel")], check=True)
        contracts = subprocess.check_output([str(temp / "kernel")], text=True).splitlines()
        assert len(contracts) == len(templates)
        lookup = {template: list(map(int, line.split())) for template, line in zip(templates, contracts, strict=True)}
        kernels, cases = [], []
        for row in rows:
            check_jit_arguments(row, header)
            kernel = compare_contract(row, lookup[row["jit_template_args"]])
            key = templates.index(row["jit_template_args"])
            contract = {"id": key, "host": row["host"], "tma": row["tma"], "kernel": kernel, "jit_template_args": row["jit_template_args"]}
            if key == len(kernels):
                kernels.append(contract)
            else:
                assert kernels[key] == contract
            cases.append({**row["input"], "kernel_id": key})
        hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
        result = {"compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0], "compile_flags": ["-std=c++20", "-O0"], "fmt_format_header_sha256": hashlib.sha256((args.fmt_include / "fmt/format.h").read_bytes()).hexdigest(), "source_commit": REVISION, "host_patches": {name: hashlib.sha256((ROOT / "patches/deep-gemm-host" / name).read_bytes()).hexdigest() for name in patches}, "patched_source_sha256": hashes, "candidate_header_sha256": hashlib.sha256(header.encode()).hexdigest(), "gpu_executed": False, "driver_descriptor_encoding_executed": False, "host_extension_loaded": False, "cases": cases, "kernels": kernels}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"PASS: actual host selector/descriptors/Args/JIT and candidate kernel layout agree for {len(rows)} configurations, {len(templates)} distinct template instantiations")


if __name__ == "__main__":
    main()
