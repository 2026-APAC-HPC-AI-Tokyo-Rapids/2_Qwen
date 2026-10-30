"""CPU checks for the Prefill-only tail skip and its fail-closed job gate."""

import json
import subprocess
import time
from pathlib import Path
from runpy import run_path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SELECTOR = ROOT / "scripts/select-firmus-prefill-dense.py"
KEY = "DG_SM90_1D2D_PREFILL_TAIL_SKIP"


@pytest.fixture
def gate(tmp_path):
    ns = run_path(str(SELECTOR))
    deadline = time.time() + 100
    for worker in range(3):
        directory = tmp_path / f"prefill-{worker}"
        directory.mkdir()
        for rank in range(4):
            record = {"schema": ns["SCHEMA"], "worker": directory.name, "rank": rank, "run_id": "test", "correctness": True, "endurance": True, "distinct_kernels": True, "endurance_calls": 200, "rounds": 15, "calls_per_sample": 20, "finished_at": deadline - 10, "pair_us": {"stock": [950.0] * 15, "skip": [948.0] * 15}}
            (directory / f"rank-{rank}.json").write_text(json.dumps(record))
    env = tmp_path / "prefill.env"
    env.write_text("SGLANG_PREFILL_FA3_VERIFY=pin\nOTHER='keep me'\n")
    return ns, tmp_path, env, deadline


def test_aggregate_adopts_and_preserves_every_pinned_line(gate):
    ns, directory, env, deadline = gate
    original = env.read_text()
    result = ns["aggregate"](directory, env, "test", deadline)
    assert result["adopted"]
    assert env.read_text() == original + KEY + "=1\n"


def test_failed_env_publish_keeps_pins_and_removes_temporary_file(gate, monkeypatch):
    ns, directory, env, deadline = gate
    original = env.read_bytes()

    def failed_replace(self, target):
        raise OSError("cannot publish env")

    monkeypatch.setattr(Path, "replace", failed_replace)
    result = ns["aggregate"](directory, env, "test", deadline)
    assert not result["adopted"] and "cannot publish env" in result["reason"]
    assert env.read_bytes() == original
    assert not env.with_name(env.name + ".tail-skip.tmp").exists()


@pytest.mark.parametrize("failure", ["missing", "correctness", "endurance", "margin", "distinct_kernels", "late", "stale", "nan", "short", "exception", "timeout", "deadline"])
def test_every_failed_gate_keeps_stock(gate, failure):
    ns, directory, env, deadline = gate
    path = directory / "prefill-2/rank-3.json"
    record = json.loads(path.read_text())
    if failure in ("correctness", "endurance", "distinct_kernels"):
        record[failure] = False
    elif failure == "margin":
        record["pair_us"]["skip"] = [948.001] * 15
    elif failure == "late":
        record["finished_at"] = deadline + 1
    elif failure == "stale":
        record["run_id"] = "previous-job"
    elif failure == "nan":
        record["pair_us"]["skip"][0] = float("nan")
    elif failure == "short":
        record["pair_us"]["skip"].pop()
    path.write_text(json.dumps(record))
    if failure == "missing":
        path.unlink()
    elif failure == "exception":
        path.write_text("broken json")
    original = env.read_bytes()
    result = ns["aggregate"](directory, env, "test", time.time() - 1 if failure == "deadline" else deadline, failure != "timeout")
    assert not result["adopted"] and result["reason"]
    assert env.read_bytes() == original


def test_real_host_predicate_excludes_decode_and_keeps_stock_suffix(tmp_path):
    patch = (ROOT / "patches/deep-gemm-host/sm90-1d2d-prefill-tail-skip.patch").read_text()
    added = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    helper = added[added.index("static bool get_sm90"):added.index("\n        {}{}")].strip()
    # Compile the actual added host predicate with only the metadata/env/print types stubbed.
    source = r'''
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_set>
enum class GemmType { Normal, Grouped };
struct GemmDesc { int m,n,k; GemmType gemm_type = GemmType::Normal; };
struct GemmConfig { struct { int block_m; } layout; };
std::ostream& operator<<(std::ostream& o, const GemmDesc&) { return o; }
std::ostream& operator<<(std::ostream& o, const GemmConfig&) { return o; }
template<typename T> T get_env(const char* name, T default_value = 0) { auto s = std::getenv(name); return s ? std::stoi(s) : default_value; }
'''
    source += helper
    source += r'''
int main() {
  unsetenv("DG_PRINT_CONFIGS"); unsetenv("DG_JIT_DEBUG");
  for (auto env : {"0", "1", "-1"}) {
    setenv("DG_SM90_1D2D_PREFILL_TAIL_SKIP", env, 1);
    for (int m : {1, 48, 128, 160, 192, 256, 3994, 4096, 7988, 8192})
      for (int bm : {64, 128, 192, 256})
        for (auto shape : {std::pair{9216,4096}, std::pair{4096,8192}, std::pair{4096,4096}}) {
          GemmDesc d{m,shape.first,shape.second}; GemmConfig c{{bm}};
          bool expected = std::stoi(env) != 0 && m > 192 && bm == 256 && m % 256 != 0 && shape.first != shape.second;
          bool selected = get_sm90_1d2d_prefill_tail_skip(d,c);
          if (selected != expected) return 1;
          auto suffix = selected ? ", true" : "";
          if (!expected && std::string(suffix) != "") return 2;
        }
  }
  unsetenv("DG_SM90_1D2D_PREFILL_TAIL_SKIP");
  if (get_sm90_1d2d_prefill_tail_skip({7988,9216,4096},{{256}})) return 3;
}
'''
    cpp = tmp_path / "host.cpp"
    cpp.write_text(source)
    executable = tmp_path / "host"
    subprocess.run(["g++", "-std=c++20", str(cpp), "-o", str(executable)], check=True, capture_output=True)
    subprocess.run([str(executable)], check=True)
    assert 'get_sm90_1d2d_prefill_tail_skip(args.gemm_desc, args.gemm_config) ? ", true" : ""' in added
    # No changed existing template argument or compiled-dimension key.
    removed = [line[1:] for line in patch.splitlines() if line.startswith("-") and not line.startswith("---")]
    assert removed == ["        {}", "        get_default_epilogue_type(args.epilogue_type));"]


def test_overlay_order_and_default_false_keep_existing_instantiations():
    host = (ROOT / "patches/deep-gemm-host/series").read_text().split()
    assert host[-2:] == ["sm90-1d2d-decode-bm192.patch", "sm90-1d2d-prefill-tail-skip.patch"]
    overlays = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())["deep_gemm_overlay_patches"]
    assert overlays[-2:] == ["patches/deep-gemm-bench/decode-dense-bm192/sm90-1d2d-decode-bm192.patch", "patches/deep-gemm-overlay/sm90-1d2d-prefill-tail-skip.patch"]
    patch = (ROOT / overlays[-1]).read_text()
    assert "bool kSkipTailWaves = false" in patch
    assert "if constexpr (kSkipTailWaves and BLOCK_M == 256)" in patch
    assert "if (local_idx == BLOCK_M / WAVE_BLOCK_M - 1)\n+                                        empty_barrier_arrive();\n+                                    continue;" in patch
    assert "NamedBarrier" not in patch and "advance_pipeline" not in patch and "->wait" not in patch


def test_pp4_parallel_timeout_merge_and_prefill_only_environment():
    sbatch = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    pp4 = sbatch.split('if [[ -s "${ROOT}/source/pp4-kernel-settings.json" ]]; then', 1)[1].split('elif [[ -s "${ROOT}/source/reused-kernel-selection.json"', 1)[0]
    assert "for worker in 0 1 2; do" in pp4
    assert "timeout --kill-after=2 110 srun" in pp4 and "--nproc-per-node=4" in pp4
    assert "select-firmus-prefill-dense.py" in pp4 and "--aggregate" in pp4
    assert 'prefill_dense_selections+=("$!")' in pp4
    assert 'wait "${pid}" || prefill_dense_ok=0' in pp4
    assert '--workers-ok "${prefill_dense_ok}"' in pp4
    assert '--env-output "${ROOT}/source/prefill-kernels.env"' in pp4
    worker = sbatch.split("cat >", 1)[1].split("\nEOF", 1)[0]
    assert "unset " + KEY in worker
    assert 'if [[ "$role" == prefill ]]; then set -a; source "$(dirname "$0")/prefill-kernels.env"; set +a; fi' in worker
    assert "tail_skip=${" + KEY + ":-0}" in worker
    assert KEY + "=1" not in sbatch
    provision = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    assert '"${GITHUB_WORKSPACE}/scripts/select-firmus-prefill-dense.py"' in provision
    assert '"${GITHUB_WORKSPACE}/patches/deep-gemm-host/sm90-1d2d-prefill-tail-skip.patch"' in provision


def test_selector_uses_serving_call_and_requires_native_host_evidence():
    source = SELECTOR.read_text()
    assert 'deep_gemm.fp8_gemm_nt(lhs, rhs, out, compiled_dims="nk")' in source
    assert "column_major_scales=True, scale_tma_aligned=True" in source
    assert 'KEY + "=1:" in log.read_text()' in source
    assert "view(torch.int16)" in source
    assert "torch.backends.cuda.matmul.allow_tf32 = False" in source
    assert "for index in range(ENDURANCE)" in source
    assert 'record["patches"][patch.name] == digest' in source
