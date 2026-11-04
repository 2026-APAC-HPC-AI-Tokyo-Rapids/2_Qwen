"""History-independent CPU contracts for the bounded dense gate and overlay."""
import ast
import gzip
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/probe-firmus-prefill-dense.py"
spec = importlib.util.spec_from_file_location("dense_r44", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture
def source(tmp_path):
    files = json.loads(gzip.decompress((ROOT / "qwen-experiment-metrics/tests/fixtures/prefill-dense-r44-source.json.gz").read_bytes()))
    for name, text in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    for kind in ("host", "overlay"):
        subprocess.run(["git", "apply", "--check", str(ROOT / f"patches/deep-gemm-{kind}/sm90-prefill-dense-n64-r44.patch")], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "apply", str(ROOT / f"patches/deep-gemm-{kind}/sm90-prefill-dense-n64-r44.patch")], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path, files


def generator(source):
    start = source.index("    static std::string generate_impl(")
    return source[start:source.index("    static void launch_impl(", start)]


def test_actual_generator_off_byte_identity_and_exact_on_guard(source):
    directory, files = source
    name = "csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp"
    prefix = r'''
#include <cassert>
#include <optional>
#include <sstream>
#include <string>
namespace cute { namespace UMMA { enum class Major { K, MN }; } }
enum class GemmType {Normal, Other};
int first=0,n64=0;
template<class T> T get_env(const char* key,T) { return std::string(key)=="DG_SM90_PREFILL_DENSE_FIRST_K" ? first : std::string(key)=="DG_SM90_PREFILL_DENSE_N64" ? n64 : 0; }
namespace fmt { template<class... T> std::string format(const char* s, T... args) { std::ostringstream o; o<<s; ((o<<args<<";"),...); return o.str(); } }
std::string to_string(int x) { return std::to_string(x); }
std::string to_string(GemmType x) { return std::to_string(int(x)); }
std::string to_string(cute::UMMA::Major x) { return std::to_string(int(x)); }
int get_compiled_dim(int n,char dim,const std::string& dims) { return dims.find(dim)==std::string::npos ? 0 : n; }
std::string get_default_epilogue_type(const std::optional<std::string>& x) { return x.value_or("identity"); }
struct Desc { int m=7988,n=9216,k=4096,num_groups=1,cd_dtype=1; GemmType gemm_type=GemmType::Normal; std::string compiled_dims="nk"; };
struct Layout { int block_m=256,block_n=128,block_k=128,cluster_n=2,cluster_size=2; int get_cluster_size() const { return cluster_size; } };
struct Storage { int swizzle_a_mode=128,swizzle_b_mode=128,swizzle_cd_mode=128; };
struct Pipeline { int num_stages=3; };
struct Launch { int num_tma_threads=128,num_math_threads=256,num_sms=132; };
struct Config { Layout layout; Storage storage_config; Pipeline pipeline_config; Launch launch_config; };
struct Args { Desc gemm_desc; Config gemm_config; cute::UMMA::Major major_sfb=cute::UMMA::Major::K; std::optional<std::string> epilogue_type; bool diagnostic_no_promotion=false; };
'''
    code = prefix + "struct Before {\n" + generator(files[name]) + "};\nstruct After {\n" + generator((directory / name).read_text()) + "};\n" + r'''
int main() {
 // Runtime M must not change the source cache key, in either flag state.
 for (n64=0;n64<2;++n64) for (int n:{4096,9216}) {
  Args a; a.gemm_desc.n=n;a.gemm_desc.k=n==9216?4096:8192;
  auto expected=After::generate_impl(a);
  for (int m:{1,255,256,257,3994,3995,4001,4002,7988,7989,7990,8001,8002,8003,8004,8064}) {
   a.gemm_desc.m=m;assert(After::generate_impl(a)==expected);
  }
 }
 for (first=0;first<2;++first) for (int m:{1,192,255,256,257,3994,3995,4001,4002,7988,7989,7990,8001,8002,8003,8004,8064,8192}) for (int n:{4096,9216,12288}) for (int k:{4096,8192}) {
  Args a; a.gemm_desc.m=m;a.gemm_desc.n=n;a.gemm_desc.k=k;
  n64=0;auto prod=Before::generate_impl(a); assert(After::generate_impl(a)==prod);
  n64=1;bool on=((n==9216&&k==4096)||(n==4096&&k==8192));
  assert(After::generate_impl(a)==(on?"#define DG_SM90_PREFILL_DENSE_N64 1\n"+prod:prod));
  a.diagnostic_no_promotion=true;
  assert(After::generate_impl(a)=="#define DG_R44_DIAGNOSTIC_NO_PROMOTION 1\n#define sm90_fp8_gemm_1d2d_impl sm90_fp8_gemm_promotion_diagnostic_impl\n"+prod);
 }
 for (int axis=0;axis<17;++axis) {
  Args a;n64=1;auto& d=a.gemm_desc;auto& c=a.gemm_config;
  switch(axis) {
   case 0:d.num_groups=2;break;case 1:d.gemm_type=GemmType::Other;break;case 2:d.compiled_dims="mnk";break;
   case 3:a.major_sfb=cute::UMMA::Major::MN;break;case 4:c.layout.block_m=128;break;case 5:c.layout.block_n=64;break;
   case 6:c.layout.block_k=64;break;case 7:c.storage_config.swizzle_a_mode=64;break;case 8:c.storage_config.swizzle_b_mode=64;break;
   case 9:c.storage_config.swizzle_cd_mode=64;break;case 10:c.pipeline_config.num_stages=4;break;case 11:c.launch_config.num_tma_threads=64;break;
   case 12:c.launch_config.num_math_threads=128;break;case 13:c.layout.cluster_size=1;break;case 14:c.layout.cluster_n=1;break;
   case 15:c.launch_config.num_sms=128;break;case 16:a.epilogue_type="other";break;
  }
  assert(After::generate_impl(a)==Before::generate_impl(a));
 }
}
'''
    cpp = directory / "generator.cpp"
    cpp.write_text(code)
    subprocess.run(["g++", "-std=c++17", "-O2", str(cpp), "-o", str(directory / "generator")], check=True, capture_output=True)
    subprocess.run([str(directory / "generator")], check=True)


def test_diagnostic_serving_isolation_and_device_guards(source):
    directory, files = source
    api = (directory / "csrc/apis/gemm.hpp").read_text()
    normal = api[api.index("static void fp8_fp4_gemm_nt("):api.index("// Separate timing-only API.")]
    assert "sm90_fp8_gemm_1d2d<false>" in normal and "<true>" not in normal
    assert "sm90_fp8_gemm_1d2d<true>" in api[api.index("// Separate timing-only API."):]
    ffi = (directory / "csrc/tvm_ffi_api.cpp").read_text()
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_gemm_nt_promotion_diagnostic, dg_fp8_gemm_nt_promotion_diagnostic)" in ffi
    assert "promotion_diagnostic" not in files["sgl_deep_gemm/__init__.py"]
    host = (directory / "csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp").read_text()
    assert 'get_env<int>("DG_R44' not in host
    assert ".diagnostic_no_promotion = DiagnosticNoPromotion" in host
    device = (directory / "deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh").read_text()
    original = files["deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh"]
    begin = original.index("#if defined(DG_SM90_PREFILL_DENSE_FIRST_K)")
    assert original[begin:original.index("\n                });", begin)] in device
    assert "float t0[32], t1[32]" in device
    assert "if constexpr (decltype(next_c)::value)" in device
    assert "chunk(std::true_type{}, 0)" in device
    assert "begin += 8" in device and "release(old_stage)" in device
    assert "half * 64 * BLOCK_K" in device and "wave * 64 + half * 32" in device
    assert "first ? 0.0f" in device
    diag = device[device.index("// TIMING ONLY"):device.index("#else\n                        using MMA64")]
    loop = diag[diag.index("for (uint32_t kb"):diag.index("// The sole wait0")]
    assert "warpgroup_wait<0>" not in loop and "__fmaf_rn" not in loop
    assert "warpgroup_wait<1>" in loop and "release(previous_stage)" in loop
    assert "kb != 0 || kk != 0" in diag


def source_triple(shape):
    _, n, k = probe.SHAPES[shape]
    template = f"cute::UMMA::Major::K,0,{n},{k},1,256,128,128,128,128,128,3,128,256,2,true,132,GemmType::Normal,cutlass::bfloat16_t,epilogue::transform::EpilogueIdentity"
    body = "#define DG_SM90_PREFILL_DENSE_FIRST_K 1\nvoid instantiate() { auto ptr = reinterpret_cast<void*>(&sm90_fp8_gemm_1d2d_impl<" + template + ">); };\n"
    header = "// Includes' hash value: same\n"
    return header + body, header + f"#define {probe.FLAG} 1\n" + body, header + probe.PREFIX + body


@pytest.mark.parametrize("shape", probe.SHAPES)
def test_real_jit_contract_and_mutations(shape):
    p, c, d = source_triple(shape)
    probe.assert_jit_sources(p, c, d, shape)
    for changed in (p.replace("256,128,128", "128,128,128"), p.replace("FIRST_K 1", "FIRST_K 0"), p.replace("same", "different")):
        with pytest.raises(AssertionError):
            probe.assert_jit_sources(changed, c, d, shape)
    with pytest.raises(AssertionError):
        probe.assert_jit_sources(p, c, d + "other", shape)


def test_cache_identity_from_two_shapes_and_three_variants(tmp_path, monkeypatch):
    for shape in probe.SHAPES:
        for variant, text in zip(("production", "candidate", "diagnostic"), source_triple(shape)):
            folder = tmp_path / f"cache/kernel.sm90_fp8_gemm_1d2d.{shape}-{variant}"
            folder.mkdir(parents=True)
            (folder / "kernel.cu").write_text(text)
    monkeypatch.setattr(subprocess, "check_output", lambda cmd, **kwargs: "STT_FUNC STO_ENTRY " + (probe.SYMBOL if "diagnostic" in str(cmd[-1]) else "sm90_fp8_gemm_1d2d_impl") + "\n")
    result = probe.cache_identity(tmp_path, Path("/cuda/bin/cuobjdump"))
    assert set(result) == {"qkv", "o"}
    assert all(set(row) == {"production", "candidate", "diagnostic"} for row in result.values())


def samples(delta=10, alternative="candidate"):
    return [{"round": r, "order": order, **{s: {"production_us": 500.0, alternative + "_us": 500.0 - delta / 2} for s in probe.SHAPES}} for order in ("forward", "reverse") for r in range(probe.ROUNDS)]


def populate(directory, delta=10):
    for worker in range(3):
        for rank in range(4):
            probe.write(directory / f"worker-{worker}/rank-{rank}.json", {"schema_version": probe.SCHEMA, "status": "complete", "worker": worker, "rank": rank, "run_id": "run", "image_sha256": "image", "started_unix": 101, "published_unix": 150, "sequence_lengths": [3994, 3994], "timing_m": probe.PAIR_M, "numeric_ms": list(probe.NUMERIC_MS), "serving_environment": {"DG_SM90_PREFILL_DENSE_FIRST_K": "1", probe.FLAG: "0"}, "weight_sha256": "weights", "runtime_header_sha256": "header", "gpu_uuid": f"GPU-{worker}-{rank}", "jit": {s: {v: {"jit_cache_key": s + v, "generated_source_sha256": s + v, "kernel_name": probe.SYMBOL if v == "diagnostic" else "sm90_fp8_gemm_1d2d_impl"} for v in ("production", "candidate", "diagnostic")} for s in probe.SHAPES}, "numerics": {str(m): {s: {"passed": True, "comparison_mode": "bitwise", "production_comparison_passed": True, "production_reference_passed": True, "bitwise_equal_production": True, "reference_passed": True, "reference_kind": "independent_fp32_k128_no_tf32", "production_max_abs": 0, "production_max_rel": 0, "reference_max_abs": .001, "reference_max_rel": .01} for s in probe.SHAPES} for m in probe.NUMERIC_MS}, "endurance": {s: {"calls": 500, "passed": True, "compared_every_call": True, "reference": "own_first_output", "rows": probe.PAIR_M} for s in probe.SHAPES}, "sm_clock_samples_mhz": [1400, 1420], "samples": samples(delta), "diagnostic_samples": samples(100, "diagnostic"), "exit_mode": "os._exit(0)", "teardown": "cuda_synchronized_graphs_reset_process_group_destroyed"})


@pytest.mark.parametrize("delta,adopted", [(9.999, False), (10, True), (10.001, True), (-3, False)])
def test_twelve_gpu_pair_median_threshold(tmp_path, delta, adopted):
    populate(tmp_path, delta)
    result = probe.aggregate(tmp_path, "run", "image", [0, 0, 0], 100, now=200)
    assert result["status"] == "complete" and result["adopted"] is adopted
    assert result["median_12gpu_pair_saving_us"] == pytest.approx(delta)
    assert result["median_12gpu_diagnostic_bound_us"] == 100
    assert len(result["gpus"]) == 12 and "samples" in result["gpus"][0]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("failure", ["missing", "failed", "clock", "round", "nan", "run", "image", "uuid", "late", "negative_error", "reference", "production_reference", "comparison", "byte", "endurance", "count", "every", "own", "weights", "source", "diag_symbol", "environment", "deadline"])
def test_any_unhealthy_gpu_or_missing_proof_keeps_production(tmp_path, failure):
    populate(tmp_path)
    path = tmp_path / "worker-1/rank-2.json"
    row = json.loads(path.read_text())
    if failure == "missing": path.unlink()
    if failure == "failed": row["status"] = "incomplete"
    if failure == "clock": row["sm_clock_samples_mhz"] = []
    if failure == "round": row["samples"].pop()
    if failure == "nan": row["samples"][0]["qkv"]["candidate_us"] = float("nan")
    if failure == "run": row["run_id"] = "other"
    if failure == "image": row["image_sha256"] = "other"
    if failure == "uuid": row["gpu_uuid"] = "GPU-0-0"
    if failure == "late": row["published_unix"] = 210.001
    if failure == "negative_error": row["numerics"][str(probe.PAIR_M)]["qkv"]["reference_max_abs"] = -1
    if failure == "reference": row["numerics"][str(probe.PAIR_M)]["o"]["reference_passed"] = False
    if failure == "production_reference": row["numerics"][str(probe.PAIR_M)]["qkv"]["production_reference_passed"] = False
    if failure == "comparison": row["numerics"][str(probe.PAIR_M)]["o"]["production_comparison_passed"] = False
    if failure == "byte": row["numerics"][str(probe.PAIR_M)]["qkv"]["bitwise_equal_production"] = False
    if failure == "endurance": row["endurance"]["o"]["passed"] = False
    if failure == "count": row["endurance"]["qkv"]["calls"] = 499
    if failure == "every": row["endurance"]["o"]["compared_every_call"] = False
    if failure == "own": row["endurance"]["o"]["reference"] = "production"
    if failure == "weights": row["weight_sha256"] = "other"
    if failure == "source": row["jit"]["o"]["candidate"]["generated_source_sha256"] = "other"
    if failure == "diag_symbol": row["jit"]["o"]["production"]["kernel_name"] = probe.SYMBOL
    if failure == "environment": row["serving_environment"][probe.DIAGNOSTIC] = "0"
    if failure != "missing": path.write_text(json.dumps(row))
    result = probe.aggregate(tmp_path, "run", "image", [0, 0, 0], 100, now=220.001 if failure == "deadline" else 200)
    assert result["adopted"] is False and result["status"] == "incomplete"
    assert len(result["gpus"]) == 12
    if failure in ("byte", "reference", "count", "endurance"):
        assert result["median_12gpu_diagnostic_bound_us"] == 100


def test_exact_deadlines_and_failed_worker_cannot_be_adopted(tmp_path):
    populate(tmp_path)
    for path in tmp_path.glob("worker-*/rank-*.json"):
        row = json.loads(path.read_text())
        row["published_unix"] = 210
        probe.write(path, row)
    assert probe.aggregate(tmp_path, "run", "image", [0, 0, 0], 100, now=220)["adopted"]
    for codes in ([124, 0, 0], [0, 137, 0], [0, 0, 1]):
        result = probe.aggregate(tmp_path, "run", "image", codes, 100, now=220)
        assert not result["adopted"] and result["worker_exit_codes"] == codes


@pytest.mark.parametrize("seconds,allowed", [(246, True), (249, True), (257, True), (270, True), (270.001, False), (271, False), (285, False)])
def test_budget_boundary(seconds, allowed):
    assert probe.budget_allows_start(seconds) is allowed


def test_diagnostic_env_rejected_before_any_import(monkeypatch):
    monkeypatch.setenv("DG_SM90_PREFILL_DENSE_FIRST_K", "1")
    monkeypatch.setenv(probe.DIAGNOSTIC, "0")
    with pytest.raises(AssertionError, match="never enter"):
        probe.run_worker(None, {})


@pytest.mark.parametrize("k", [4096, 8192])
@pytest.mark.parametrize("m", [1, 255, 256, 257, 3994, 3995, 4001, 4002, 7988, 7989, 7990, 8001, 8002, 8003, 8004])
def test_cpu_emulated_accumulator_layout_and_full_k_element_order(k, m):
    # An M256 CTA has two WGs and two M64 waves. Ordinary FP32 products of
    # these exactly representable E4M3 numbers are summed in identical K32
    # order. N64 changes only the fragment mapping; this is not GPU numerics.
    rng = np.random.default_rng(44 + k)
    a = rng.integers(-16, 17, (256, k)).astype(np.float32) / 8
    b = rng.integers(-16, 17, (128, k)).astype(np.float32) / 8
    sa = rng.uniform(.001, .01, (256, k // 128)).astype(np.float32)
    sb = rng.uniform(.001, .01, (k // 128,)).astype(np.float32)
    # Emulate the last CTA: TMA zero fills A outside runtime M. The
    # unchanged TMA epilogue writes only the valid rows of this full fragment.
    valid = (m - 1) % 256 + 1
    a[valid:] = 0
    production = np.zeros((256, 128), np.float32)
    candidate = production.copy()
    for kb in range(k // 128):
        product = np.zeros_like(production)
        for kk in range(4):
            sl = slice(kb * 128 + kk * 32, kb * 128 + (kk + 1) * 32)
            product += a[:, sl] @ b[:, sl].T
        scale = (sa[:, kb] * sb[kb])[:, None]
        production[:] = (scale.astype(np.float64) * product + production.astype(np.float64)).astype(np.float32)
        for wg in range(2):
            for wave in range(2):
                rows = slice(wg * 64 + wave * 128, wg * 64 + wave * 128 + 64)
                for half in range(2):
                    columns = slice(half * 64, half * 64 + 64)
                    block = np.zeros((64, 64), np.float32)
                    for kk in range(4):
                        sl = slice(kb * 128 + kk * 32, kb * 128 + (kk + 1) * 32)
                        block += a[rows, sl] @ b[columns, sl].T
                    candidate[rows, columns] = (scale[rows].astype(np.float64) * block + candidate[rows, columns].astype(np.float64)).astype(np.float32)
    assert np.array_equal(production.view(np.uint32), candidate.view(np.uint32))
    destination = np.full((258, 128), np.nan, np.float32)
    destination[1:valid+1] = candidate[:valid]
    assert np.array_equal(destination[1:valid+1], production[:valid])
    assert np.isnan(destination[:1]).all() and np.isnan(destination[valid+1:]).all()


def test_dense_snapshot_production_copy_without_torch(monkeypatch):
    from types import SimpleNamespace

    class Tensor:
        def __init__(self, array, dtype):
            self.array = np.asarray(array)
            self.shape = self.array.shape
            self.dtype = dtype
            self.device = "cuda"

        def copy_(self, raw):
            self.array[:] = raw.array
            return self

    def empty(shape, dtype, device):
        return Tensor(np.empty(shape, np.float32), dtype)

    monkeypatch.setitem(__import__("sys").modules, "torch", SimpleNamespace(float32="fp32", float8_e4m3fn="e4m3", empty=empty, cat=lambda values, dim: Tensor(np.concatenate([v.array for v in values], axis=dim), values[0].dtype)))
    tensors = {}
    for i, projection in enumerate(("q_proj", "k_proj", "v_proj", "o_proj")):
        for suffix in ("weight", "weight_scale_inv"):
            tensors[f"self_attn.{projection}.{suffix}"] = Tensor(np.full((2, 3), i + .25), "bf16" if suffix.endswith("inv") else "e4m3")
    qkv, o = probe.dense_snapshot(tensors)
    assert qkv[1].dtype == o[1].dtype == "fp32"
    assert qkv[0].dtype == o[0].dtype == "e4m3"
    assert np.array_equal(qkv[1].array[:, 0], [.25, .25, 1.25, 1.25, 2.25, 2.25])
    assert not np.shares_memory(qkv[1].array, tensors["self_attn.q_proj.weight_scale_inv"].array)


def test_sbatch_budget_staging_collection_and_gate_only_env():
    batch = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    start = batch.index("# Dense N64 gate:")
    end = batch.index("# FC1 first-K is pinned", start)
    block = batch[start:end]
    assert batch.index("# Decode greedy gate end.") < start < batch.index('bash "${ROOT}/source/worker.sh" prefill-0')
    assert "SECONDS + 120 + 630 <= 1020" in block and "timeout --signal=KILL 120 bash" in block
    assert "--foreground --kill-after=2 110 srun" in (ROOT / "scripts/launch-firmus-prefill-dense-probe.sh").read_text()
    assert block.index('elif jq -e') < block.index("sed -i")
    assert "DG_SM90_PREFILL_DENSE_N64=0$/DG_SM90_PREFILL_DENSE_N64=1" in block
    assert '"adopted":false' in block and probe.DIAGNOSTIC not in block
    assert "results/prefill-dense-r44/dispatch-${worker}.txt" in batch
    provision = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    for name in (SCRIPT.name, "launch-firmus-prefill-dense-probe.sh", "run-firmus-prefill-dense-probe.sh"):
        assert provision.count(name) == 2
    assert "megamoe-local-ep1-routing.npz" not in provision
    workflow = (ROOT / ".github/workflows/firmus.yml").read_text()
    assert "for part in logs profile results source; do" in workflow
    wrapper = (ROOT / "scripts/run-firmus-prefill-dense-probe.sh").read_text()
    assert "--nproc-per-node=4" in wrapper and 'source "${root}/source/prefill-kernels.env"' in wrapper
    source = SCRIPT.read_text()
    assert source.index('compile_variants(("production", "candidate"))') < source.index("start.record()")
    assert 'torch.backends.cuda.matmul.allow_tf32 = False' in source
    assert 'for _ in range(ENDURANCE)' in source and 'ok.logical_and_' in source
    assert "/opt/sglang/venv/bin/torchrun" in wrapper
    assert 'mktemp -d /tmp/prefill-dense-r44.XXXXXXXX' in wrapper
    assert '--cache "${cache}"' in wrapper and '--torchrun-started "$(date +%s.%N)"' in wrapper
    assert 'trap \'rm -rf "${cache}"\' EXIT' in wrapper
    assert "DG_SM90_PREFILL_DENSE_N64\": \"0\"" in (ROOT / "scripts/prepare-firmus-pp4-settings.py").read_text()


@pytest.mark.parametrize("fail_worker", [False, True])
def test_supervisor_three_workers_concurrent_and_aggregates(tmp_path, fail_worker):
    import sys

    runtime = tmp_path / "runtime"
    (runtime / "scripts").mkdir(parents=True)
    (runtime / "logs").mkdir()
    shutil.copy2(SCRIPT, runtime / "scripts" / SCRIPT.name)
    populate(runtime / "results/prefill-dense-r44")
    # Actual supervisor passes a real start epoch; fixture rows use epoch 100.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    srun = bindir / "srun"
    srun.write_text(f'#!/usr/bin/env bash\ntouch "{tmp_path}/started-${{BASHPID}}"\nfor i in $(seq 1 100); do\n  count=$(find "{tmp_path}" -maxdepth 1 -name "started-*" | wc -l)\n  if (( count == 3 )); then exit {1 if fail_worker else 0}; fi\n  sleep .01\ndone\nexit 99\n')
    srun.chmod(0o755)
    uv = bindir / "uv"
    uv.write_text(f'#!/usr/bin/env bash\nshift 5\nexec "{sys.executable}" "$@"\n')
    uv.chmod(0o755)
    # Main aggregator uses wall clock; choose a recent fixture epoch.
    import time
    now = int(time.time())
    for path in (runtime / "results/prefill-dense-r44").glob("worker-*/rank-*.json"):
        row = json.loads(path.read_text()); row.update(started_unix=now, published_unix=now)
        probe.write(path, row)
    result = subprocess.run(["bash", str(ROOT / "scripts/launch-firmus-prefill-dense-probe.sh"), str(runtime), "head", "tail", "container", "run", "image", "model", str(now), "mount0", "mount1", "mount2"], env={**os.environ, "PATH": str(bindir) + ":" + os.environ["PATH"]}, capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    summary = json.loads((runtime / "results/prefill-dense-r44/summary.json").read_text())
    assert summary["adopted"] is (not fail_worker)
    assert summary["worker_exit_codes"] == ([1, 1, 1] if fail_worker else [0, 0, 0])
    assert len(list(tmp_path.glob("started-*"))) == 3


def test_prewarm_dispatch_wiring():
    config = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    assert "patches/sglang/prefill-dense-n64-r44.patch" in config["sglang_overlay_patches"]
    patch = (ROOT / "patches/sglang/prefill-dense-n64-r44.patch").read_text()
    assert "prewarm_dense_r44(self.model)" in patch
    assert 'case.step("candidate", 0)' in patch and 'case.step("candidate", 1)' in patch
    assert patch.index('+            torch.cuda.synchronize()') < patch.index("prewarm done") < patch.index("_prewarmed = True")
    assert 'if not _r44_logged and os.environ.get(' in patch
    assert 'dispatch_dense_r44(a.shape[0])' in patch
    assert 'a.shape[0] ==' not in patch
    assert 'for rows in (7988, 3994):' in patch
    for name in ("probe-firmus-megamoe-rescale.py", "run-firmus-megamoe-rescale-probe.sh", "launch-firmus-megamoe-rescale-probe.sh"):
        assert not (ROOT / "scripts" / name).exists()
    assert not (ROOT / "patches/deep-gemm-host/sm90-rescale-exposure-probe-r41.patch").exists()
    builder = (ROOT / "containers/rebuild-deepgemm-host.py").read_text()
    assert "deep_gemm._C.fp8_gemm_nt_promotion_diagnostic" in builder
    assert "rescale_exposure_diagnostic" not in builder


def test_prewarm_both_shapes_once_with_cpu_doubles(monkeypatch, caplog):
    import logging
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace as NS

    patch = (ROOT / "patches/sglang/prefill-dense-n64-r44.patch").read_text()
    start = patch.index('+++ b/python/sglang/srt/layers/prefill_dense_r44.py')
    text = "\n".join(line[1:] for line in patch[start:].splitlines() if line.startswith("+") and not line.startswith("+++"))
    tree = ast.parse(text)
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    calls = []
    class Layer:
        def __init__(self):
            weight = NS(weight=object(), weight_scale_inv=object())
            self.self_attn = NS(qkv_proj=weight, o_proj=weight)
    class Case:
        def __init__(self, rows, weights): calls.append(("allocate", rows))
        def step(self, mode, projection): calls.append((mode, projection))
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.prefill_r37", NS(Dense=Case))
    monkeypatch.setitem(sys.modules, "sglang.srt.models.qwen3_moe", NS(Qwen3MoeDecoderLayer=Layer))
    monkeypatch.setenv(probe.FLAG, "1")
    ns = {"os": os, "torch": NS(inference_mode=nullcontext, random=NS(fork_rng=lambda **kwargs: nullcontext()), cuda=NS(current_device=lambda: 0, synchronize=lambda: calls.append(("sync",)))), "logging": logging, "__name__": "r44-test", "FLAG": probe.FLAG, "_prewarmed": False, "_logged": set()}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "r44-prewarm", "exec"), ns)  # noqa: S102 - actual overlaid functions with CPU doubles
    model = NS(modules=lambda: [Layer()])
    ns["prewarm"](model); ns["prewarm"](model)
    ns["dispatch"](4001); ns["dispatch"](8002)
    assert calls == [entry for m in (7988, 3994) for entry in (("allocate", m), ("candidate", 0), ("candidate", 1), ("sync",))]
    assert sum("using candidate" in row.message for row in caplog.records) == 1
    assert sum("prewarm done" in row.message for row in caplog.records) == 2
    assert any("using candidate rows=4001" in row.message for row in caplog.records)


def test_diagnostic_host_api_compiles_against_real_scale_transform_signature(source):
    directory, files = source
    api = (directory / "csrc/apis/gemm.hpp").read_text()
    begin = api.index("static void fp8_gemm_nt_promotion_diagnostic(")
    function = api[begin:api.index("\nstatic void fp8_fp4_gemm_nn(", begin)]
    layout = files["csrc/apis/layout.hpp"]
    begin = layout.index("static std::tuple<torch::Tensor, torch::Tensor, int, int> transform_sf_pair_into_required_layout(")
    signature = layout[begin:layout.index(" {", begin)]
    code = r'''
#include <cassert>
#include <optional>
#include <tuple>
#include <utility>
#include <string>
#define DG_HOST_ASSERT(x) assert(x)
namespace cute { namespace UMMA { enum class Major {K,MN}; } }
namespace torch {
 const int kBFloat16=1,kFloat8_e4m3fn=2,kFloat=3;
 struct Tensor { int type,m,n; int scalar_type() const {return type;} };
}
struct Device {int get_arch_major(){return 9;}} device;
auto device_runtime=&device;
auto get_major_type_ab(const torch::Tensor&) {return cute::UMMA::Major::K;}
void check_major_type_cd(const torch::Tensor&) {}
auto check_ab_fp8_fp4(const torch::Tensor& t,cute::UMMA::Major,int) {return std::make_pair(t.m,t.n);}
template<int N> auto get_shape(const torch::Tensor& t) {return std::make_pair(t.m,t.n);}
bool diagnostic_called=false;
template<bool DiagnosticNoPromotion> void sm90_fp8_gemm_1d2d(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,std::optional<torch::Tensor>,const torch::Tensor&,int,int,int,cute::UMMA::Major,cute::UMMA::Major,cute::UMMA::Major,const std::string&) {static_assert(DiagnosticNoPromotion);diagnostic_called=true;}
namespace layout {
'''
    code += signature + " { return {sfa,sfb,128,128}; }\n}\n" + function + r'''
int main() {
 torch::Tensor a{torch::kFloat8_e4m3fn,8002,4096},b{torch::kFloat8_e4m3fn,9216,4096},sa{torch::kFloat,8002,32},sb{torch::kFloat,72,32},out{torch::kBFloat16,8002,9216};
 fp8_gemm_nt_promotion_diagnostic({a,sa},{b,sb},out);assert(diagnostic_called);
}
'''
    cpp = directory / "diagnostic-api.cpp"
    cpp.write_text(code)
    subprocess.run(["g++", "-std=c++17", "-O2", str(cpp), "-o", str(directory / "diagnostic-api")], check=True, capture_output=True)
    subprocess.run([str(directory / "diagnostic-api")], check=True)


def test_actual_independent_reference_and_failure_metrics_with_numpy_doubles(monkeypatch):
    import sys
    from types import SimpleNamespace

    class Array(np.ndarray):
        def float(self): return self.astype(np.float32)
        def abs(self): return np.abs(self)
        def clamp_min(self, value): return np.maximum(self, value)
        def repeat_interleave(self, count): return np.repeat(self, count).view(Array)
        def addcmul_(self, x, y):
            self[:] = (self.astype(np.float64) + x.astype(np.float64) * y.astype(np.float64)).astype(np.float32)
            return self

    def array(value): return np.asarray(value, np.float32).view(Array)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32=np.float32, uint8=np.uint8, zeros=lambda shape, **kwargs: np.zeros(shape, np.float32).view(Array), zeros_like=lambda x: np.zeros_like(x), equal=np.array_equal, all=np.all, isfinite=np.isfinite))
    rng = np.random.default_rng(44)
    a = array(rng.integers(-4, 5, (4, 256)) / 8)
    b = array(rng.integers(-4, 5, (128, 256)) / 8)
    sa, sb = array(rng.uniform(.01, .1, (4, 2))), array([[.01, .03]])
    out = array(np.zeros((4, 128)))
    for kb in range(2):
        scale = sa[:, kb, None] * sb[:, kb].repeat_interleave(128)[None, :]
        out.addcmul_(a[:, kb * 128:(kb + 1) * 128] @ b[:, kb * 128:(kb + 1) * 128].T, scale)
    result = probe.numerical_check((a, sa), (b, sb), out, out.copy())
    assert result["passed"] and result["reference_max_abs"] == 0 and result["production_max_abs"] == 0
    near = out.copy(); near[0, 0] = np.nextafter(near[0, 0], np.float32(np.inf))
    reordered = probe.numerical_check((a, sa), (b, sb), out, near)
    assert not reordered["passed"] and reordered["comparison_mode"] == "fp32_reordering"
    wrong = out.copy(); wrong[0, 0] += 1
    result = probe.numerical_check((a, sa), (b, sb), out, wrong)
    assert not result["passed"] and not result["reference_passed"] and result["production_max_abs"] == pytest.approx(1)
    wrong[0, 0] = float("nan")
    result = probe.numerical_check((a, sa), (b, sb), out, wrong)
    assert not result["passed"] and result["production_max_abs"] is None
    json.dumps(result, allow_nan=False)


def test_nonbitwise_comparison_cannot_be_adopted_even_with_fp32_reference(tmp_path):
    populate(tmp_path)
    path = tmp_path / "worker-0/rank-0.json"
    row = json.loads(path.read_text())
    row["numerics"][str(probe.PAIR_M)]["qkv"].update(comparison_mode="fp32_reordering", bitwise_equal_production=False, production_max_abs=1e-7, production_max_rel=1e-6)
    probe.write(path, row)
    assert not probe.aggregate(tmp_path, "run", "image", [0, 0, 0], 100, now=200)["adopted"]
    row["numerics"][str(probe.PAIR_M)]["qkv"]["production_comparison_passed"] = False
    probe.write(path, row)
    result = probe.aggregate(tmp_path, "run", "image", [0, 0, 0], 100, now=200)
    assert not result["adopted"] and result["median_12gpu_diagnostic_bound_us"] == 100


@pytest.mark.parametrize("diagnostic", [False, True])
def test_actual_device_loop_full_k_scales_and_stage_lifecycle(source, diagnostic):
    directory, files = source
    name = "deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh"
    header = (directory / name).read_text()
    start = header.index("                        // Release the OLD stage")
    body = header[start:header.index("\n                    } else\n#endif", start)]
    original = files[name]
    start = original.index("    auto advance_pipeline =")
    advance = original[start:original.index("\n    };", start) + len("\n    };")]
    harness = (ROOT / "qwen-experiment-metrics/tests/fixtures/prefill-dense-r44-pipeline.cpp").read_text().replace("// ADVANCE_PIPELINE", advance).replace("// R44_BODY", body)
    cpp = directory / "pipeline.cpp"
    cpp.write_text(harness)
    executable = directory / "pipeline"
    subprocess.run(["g++", "-std=c++20", "-O2", "-DDG_R44_DIAGNOSTIC_NO_PROMOTION=" + str(int(diagnostic)), str(cpp), "-o", str(executable)], check=True, capture_output=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    assert "blocks=32 releases=32" in result.stdout and "blocks=64 releases=64" in result.stdout


def test_n64_keeps_runtime_m_tma_producer_and_epilogue_unchanged(source):
    directory, files = source
    name = "deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh"
    device = (directory / name).read_text()
    start = device.index("#if (defined(DG_SM90_PREFILL_DENSE_N64)")
    end = device.index("                    } else\n#endif", start) + len("                    } else\n#endif\n")
    # The entire producer, scheduler and epilogue are byte-identical.
    original = files[name]
    assert device[:start] == original[:start]
    tail = original.index("                });", original.index("// Dispatch `num_former_iters`"))
    assert device[device.index("                });", end):] == original[tail:]
    assert "shape_m" not in device[start:end]
    host = (directory / "csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp").read_text()
    assert "make_tma_a_desc(major_a, a, m, k," in host
    assert "make_tma_cd_desc(d, m, static_cast<int>(d.size(-1))," in host
    assert "make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k," in host


@pytest.mark.parametrize("rows", probe.NUMERIC_MS)
def test_every_runtime_m_numeric_proof_is_required(tmp_path, rows):
    populate(tmp_path)
    path = tmp_path / "worker-2/rank-3.json"
    row = json.loads(path.read_text())
    del row["numerics"][str(rows)]
    probe.write(path, row)
    result = probe.aggregate(tmp_path, "run", "image", [0, 0, 0], 100, now=200)
    assert not result["adopted"] and result["status"] == "incomplete"
    assert result["median_12gpu_diagnostic_bound_us"] == 100


def test_only_real_pair_m_timing_controls_adoption(tmp_path):
    populate(tmp_path, 9.999)
    path = tmp_path / "worker-0/rank-0.json"
    row = json.loads(path.read_text())
    row["diagnostic_samples"] = samples(200, "diagnostic")
    probe.write(path, row)
    result = probe.aggregate(tmp_path, "run", "image", [0, 0, 0], 100, now=200)
    assert not result["adopted"] and result["timing_m"] == 7988
    assert result["additional_timing_ms"] == []
    assert result["gpus"][0]["median_diagnostic_bound_us"] == 200


@pytest.mark.parametrize("rows", [4001, 4002, 8001, 8002, 8003, 8004])
@pytest.mark.parametrize("enabled", [False, True])
def test_dispatch_actual_condition_logs_first_eligible_runtime_m(rows, enabled, monkeypatch):
    from types import SimpleNamespace as NS

    patch = (ROOT / "patches/sglang/prefill-dense-n64-r44.patch").read_text()
    condition = next(line[1:] for line in patch.splitlines() if line.startswith("+    if not _r44_logged"))
    body = condition.strip() + "\n    dispatch_dense_r44(a.shape[0])\n"
    seen, logs = set(), []
    def dispatch(rows):
        logs.append(rows)
        seen.add("dense-n64")
    monkeypatch.setenv(probe.FLAG, "1" if enabled else "0")
    ns = {"os": os, "_r44_logged": seen, "a": NS(shape=(rows, 4096)), "layer": NS(weight=NS(shape=(9216, 4096))), "dispatch_dense_r44": dispatch}
    for _ in range(2):
        exec(compile(body, "r44-dispatch", "exec"), ns)  # noqa: S102 - actual patched condition
    assert logs == ([rows] if enabled else [])
    seen.clear(); logs.clear()
    ns["layer"].weight.shape = (12288, 4096)
    exec(compile(body, "r44-dispatch", "exec"), ns)  # noqa: S102 - ineligible N
    assert not logs


@pytest.mark.parametrize("fail_cleanup", [False, True])
def test_teardown_releases_graphs_before_group_and_never_publishes(tmp_path, monkeypatch, fail_cleanup):
    import sys
    from types import ModuleType, SimpleNamespace

    calls = []
    dist = ModuleType("torch.distributed")
    dist.barrier = lambda: calls.append("barrier")
    def destroy():
        calls.append("destroy")
        if fail_cleanup:
            raise RuntimeError("cleanup failed")
    dist.destroy_process_group = destroy
    torch = ModuleType("torch")
    torch.distributed = dist
    torch.cuda = SimpleNamespace(synchronize=lambda: calls.append("sync"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    monkeypatch.setattr(probe.shutil, "copytree", lambda src, dst: calls.append("archive"))
    graphs = {"qkv": {"candidate": SimpleNamespace(reset=lambda: calls.append("reset"))}}
    if fail_cleanup:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            probe.teardown(graphs, tmp_path / "cache", tmp_path, 0)
    else:
        probe.teardown(graphs, tmp_path / "cache", tmp_path, 0)
    assert calls == ["sync", "reset", "barrier", "archive", "barrier", "destroy"]
    assert not list(tmp_path.glob("rank-*.json"))


def test_normal_fast_exit_publishes_and_flushes_without_running_atexit(tmp_path):
    import sys
    import time

    code = '''import atexit, importlib.util, os, sys
from pathlib import Path
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location("probe", sys.argv[1])
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
atexit.register(lambda: Path(sys.argv[2], "unwanted-destructor").touch())
print("flushed-success", end="")
args = SimpleNamespace(output=Path(sys.argv[2]), started=float(sys.argv[3]))
probe.publish_and_exit(args, {"rank": 0, "status": "complete"})
raise RuntimeError("fast exit returned")
'''
    result = subprocess.run([sys.executable, "-c", code, str(SCRIPT), str(tmp_path), str(time.time())], capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 0 and result.stdout == "flushed-success"
    assert json.loads((tmp_path / "rank-0.json").read_text())["status"] == "complete"
    assert not (tmp_path / "unwanted-destructor").exists()


@pytest.mark.parametrize("failure", ["exit_mode", "teardown", "diagnostic_missing", "diagnostic_round"])
def test_fast_exit_and_separate_bound_contract_is_required(tmp_path, failure):
    populate(tmp_path)
    path = tmp_path / "worker-2/rank-3.json"
    row = json.loads(path.read_text())
    if failure in ("exit_mode", "teardown"):
        del row[failure]
    if failure == "diagnostic_missing":
        del row["diagnostic_samples"]
    if failure == "diagnostic_round":
        row["diagnostic_samples"].pop()
    probe.write(path, row)
    result = probe.aggregate(tmp_path, "run", "image", [0, 0, 0], 100, now=200)
    assert result["status"] == "incomplete" and not result["adopted"]


def test_cold_jit_has_one_owner_per_shape_and_barrier_before_reuse():
    from types import SimpleNamespace

    tree = ast.parse(SCRIPT.read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_worker")
    compile_func = next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == "compile_variants")
    calls = []
    ns = {"SHAPES": probe.SHAPES, "call": lambda variant, shape: calls.append((shape, variant)), "torch": SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: calls.append("sync"))), "dist": SimpleNamespace(barrier=lambda: calls.append("barrier"))}
    exec(compile(ast.Module(body=[compile_func], type_ignores=[]), "cold-jit", "exec"), ns)  # noqa: S102 - actual worker function
    for variants in (("production", "candidate"), ("diagnostic",)):
        for rank in range(4):
            ns["rank"] = rank
            ns["compile_variants"](variants)
            assert calls[-1] == "barrier"
    assert sorted(c for c in calls if isinstance(c, tuple)) == sorted((s, v) for s in range(2) for v in ("production", "candidate", "diagnostic"))
    text = SCRIPT.read_text()
    assert text.index('compile_variants(("production", "candidate"))') < text.index('checkpoint("correctness_done")') < text.index('checkpoint("endurance_done")') < text.index('timing(graphs, "candidate", samples)') < text.index('compile_variants(("diagnostic",))')
    assert probe.NUMERIC_MS == (7988, 8003, 3994)
    assert probe.ROUNDS == 4 and probe.CALLS == 4 and probe.ENDURANCE == 500
    assert 'single_graphs' not in text
    assert text.index('teardown(graphs, cache, args.output, rank)') < text.index('record.update(status="complete"')
    compiler = json.loads(gzip.decompress((ROOT / "qwen-experiment-metrics/tests/fixtures/prefill-r37-host-contract.json.gz").read_bytes()))["files"]["csrc/jit/compiler.hpp"]
    assert 'get_env<std::string>("DG_JIT_CACHE_DIR")' in compiler
    assert 'std::filesystem::rename(tmp_dir_path, dir_path, error_code)' in compiler
    assert compiler.index('kernel_runtime_cache->get(dir_path)') < compiler.index('compile(code, tmp_dir_path')
