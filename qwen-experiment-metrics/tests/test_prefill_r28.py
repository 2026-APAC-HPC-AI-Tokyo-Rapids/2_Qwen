"""Independent all-UUID decisions, real overlay dispatch and allocation budget."""
import ast
import copy
import importlib.util
import json
import os
import runpy
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("firmus_prefill_r28", ROOT / "scripts/firmus_prefill_r28.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
sys.modules["firmus_prefill_r28"] = G
SELECTOR = runpy.run_path(str(ROOT / "scripts/select-firmus-prefill-r28.py"))


@pytest.fixture
def screen(tmp_path):
    now = time.time()
    env = tmp_path / "prefill.env"
    env.write_text("KEEP=unchanged\n" + "".join(s["key"] + "=0\n" for s in G.SPECS.values()))
    identities = []
    shape = {"bytes_equal": True, "input_versions": 2, "restored": True, "samples_us": {order: {"control": [1000.] * G.ROUNDS, "candidate": [990.] * G.ROUNDS} for order in ("forward", "reverse")}, "endurance": {"calls": 1200, "buffers": 2, "input_versions": 2, "bytes_equal": True, "restored": True}}
    for worker in range(3):
        for rank in range(4):
            identity = {"uuid": f"gpu-{worker}-{rank}", "run_id": "123-pair"}
            identities.append(identity)
            for name, settings in G.SPECS.items():
                path = tmp_path / name / f"prefill-{worker}"
                path.mkdir(parents=True, exist_ok=True)
                record = {**identity, "run_id": "123-r28", "worker": path.name, "rank": rank, "schema": G.SCHEMA, "candidate": name, "device": "NVIDIA H200", "healthy": True, "image_sha256": "a" * 64, "source_sha256": "b" * 64, "fa3_pin": "P3", "elapsed_seconds": 50, "finished_at": now, "shapes": {s: copy.deepcopy(shape) for s in settings["shapes"]}}
                (path / f"rank-{rank}.json").write_text(json.dumps(record))
    pair = tmp_path / "pair.json"
    pair.write_text(json.dumps({"schema": "firmus-prefill-fa3-pair/v1", "gpus": identities, "adopted": False}))
    return SimpleNamespace(aggregate=tmp_path, env_output=env, pair_selection=pair, run_id="123-r28", deadline=now + 100, workers_ok=1, image_sha256="a" * 64, fa3_pin="P3")


@pytest.mark.parametrize("name", G.SPECS)
def test_twelve_exact_ranks_adopt_independently(screen, name):
    screen.workers_ok = 0  # A later candidate's timeout cannot invalidate this evidence.
    before = screen.env_output.read_text()
    assert SELECTOR["aggregate_candidate"](screen, name)["adopted"]
    key = G.SPECS[name]["key"]
    assert screen.env_output.read_text() == before.replace(key + "=0", key + "=1")
    assert not list(screen.aggregate.glob("*.tmp"))


@pytest.mark.parametrize("name", G.SPECS)
@pytest.mark.parametrize("bad", ("missing", "stale", "uuid", "allocation", "image", "source", "pin", "late", "elapsed", "shape", "bytes", "restore", "endurance", "timing", "reverse", "relative", "nan", "overflow"))
def test_one_failed_cell_leaves_flags_unchanged(screen, name, bad):
    before = screen.env_output.read_bytes()
    path = screen.aggregate / name / "prefill-2/rank-3.json"
    record = json.loads(path.read_text())
    shape = next(iter(record["shapes"].values()))
    if bad == "missing": path.unlink()
    elif bad == "overflow": path.write_text(path.read_text().replace('"elapsed_seconds": 50', '"elapsed_seconds": 1e309'))
    elif bad == "allocation":
        pair = json.loads(screen.pair_selection.read_text()); pair["gpus"][-1]["run_id"] = "999-old"; screen.pair_selection.write_text(json.dumps(pair))
    else:
        if bad == "stale": record["run_id"] = "123-old"
        elif bad == "uuid": record["uuid"] = "gpu-0-0"
        elif bad == "image": record["image_sha256"] = "c" * 64
        elif bad == "source": record["source_sha256"] = "c" * 64
        elif bad == "pin": record["fa3_pin"] = "P"
        elif bad == "late": record["finished_at"] = screen.deadline + 1
        elif bad == "elapsed": record["elapsed_seconds"] = 90
        elif bad == "shape": record["shapes"].pop(next(iter(record["shapes"])))
        elif bad == "bytes": shape["bytes_equal"] = False
        elif bad == "restore": shape["restored"] = False
        elif bad == "endurance": shape["endurance"]["bytes_equal"] = False
        elif bad == "timing": shape["samples_us"]["forward"]["candidate"].pop()
        elif bad == "reverse": shape["samples_us"]["reverse"]["candidate"] = [1001.] * G.ROUNDS
        elif bad == "relative": shape["samples_us"]["forward"]["candidate"] = [996.] * G.ROUNDS
        elif bad == "nan": shape["samples_us"]["forward"]["candidate"][0] = float("nan")
        path.write_text(json.dumps(record))
    assert not SELECTOR["aggregate_candidate"](screen, name)["adopted"]
    assert screen.env_output.read_bytes() == before


def test_later_failure_does_not_revert_an_adoption(screen):
    assert SELECTOR["aggregate_candidate"](screen, "norm-store")["adopted"]
    saved = screen.env_output.read_bytes()
    (screen.aggregate / "gemm-tail/prefill-1/rank-2.json").unlink()
    assert not SELECTOR["aggregate_candidate"](screen, "gemm-tail")["adopted"]
    assert screen.env_output.read_bytes() == saved


@pytest.fixture
def helper(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    rel = "python/sglang/srt/layers/attention/prefill_fp8.py"
    for patch in ("prefill-fp8-integration.patch", "prefill-fa3-pair.patch", "prefill-fa3-vsum.patch", "prefill-small-bundle-r28.patch"):
        subprocess.run(["git", "apply", "--include=" + rel, str(ROOT / "patches/sglang" / patch)], cwd=tmp_path, check=True)
    tree = ast.parse((tmp_path / rel).read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("dense_gemm", "prefill_candidate", "pair_candidate", "r28_log")]
    class Table:
        def __init__(self, _): pass
        def read_text(self): return (ROOT / "kernels/fa3-verify/prefill-candidates.json").read_text()
    namespace = {"json": json, "os": os, "Path": Table, "lru_cache": lru_cache, "_dispatch_seen": set()}
    exec(compile(tree, "<actual r28 host overlay>", "exec"), namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize("suffix", ("", "-prepack", "-prepack-v3"))
def test_actual_pair_dispatch_preserves_baseline_and_single(helper, monkeypatch, suffix):
    monkeypatch.setenv("SGLANG_PREFILL_FA3_VERIFY", "n192-direct-o-varlen-pipe+m192n96-pair" + suffix)
    monkeypatch.delenv(G.SPECS["fa3-pair-order"]["key"], raising=False)
    baseline = helper["pair_candidate"]((3994, 3994))
    single = helper["pair_candidate"]((3994,))
    monkeypatch.setenv(G.SPECS["fa3-pair-order"]["key"], "1")
    candidate = helper["pair_candidate"]((3994, 3994))
    assert candidate["variant"] == baseline["variant"] | 1048576
    assert candidate.get("native_v_n") == baseline.get("native_v_n")
    assert helper["pair_candidate"]((3994,)) == single
    assert helper["pair_candidate"]((3899, 4096)) == single


@pytest.mark.parametrize("enabled,rows,count", ((False, 7988, 1), (True, 7988, 2), (True, 7989, 1), (True, 3994, 1)))
def test_actual_split_launches_only_the_last_m_row(helper, monkeypatch, enabled, rows, count):
    calls = []
    monkeypatch.setitem(sys.modules, "deep_gemm", SimpleNamespace(fp8_gemm_nt=lambda *a, **kw: calls.append((a, kw))))
    class Tensor:
        def __init__(self, name, shape): self.name, self.shape = name, shape
        def __getitem__(self, key): return self.name, key.start, key.stop
    a, sa, w, s, out = (Tensor("a", (rows, 4096)), Tensor("sa", (rows, 32)), Tensor("w", (9216, 4096)), object(), Tensor("out", (rows, 9216)))
    helper["dense_gemm"](a, sa, w, s, out, tail_split=enabled)
    assert len(calls) == count and all(k == {"compiled_dims": "nk"} for _, k in calls)
    if count == 2:
        assert calls[0][0][0] == (("a", None, 7936), ("sa", None, 7936))
        assert calls[1][0][0] == (("a", 7936, None), ("sa", 7936, None))
        assert calls[0][0][2] == ("out", None, 7936) and calls[1][0][2] == ("out", 7936, None)
    else:
        assert calls[0][0] == ((a, sa), (w, s), out)


def test_overlay_norm_applies_to_real_base_and_keeps_math(tmp_path):
    rel = "python/sglang/kernels/ops/layernorm/add_rmsnorm_group_quant.py"
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for patch in ("add-rmsnorm-group-quant.patch", "prefill-fp8-integration.patch", "prefill-norm-store-r28.patch"):
        subprocess.run(["git", "apply", "--include=" + rel, str(ROOT / "patches/sglang" / patch)], cwd=tmp_path, check=True)
    source = (tmp_path / rel).read_text()
    ast.parse(source)
    assert "STORE_PAIR: tl.constexpr = False" in source and "enable_fp_fusion=False" in source
    assert "shfl.sync.bfly.b32" in source and "@even st.global.b32" in source
    assert "rstd = tl.rsqrt(tl.sum(h * h, 0) / 4096.0 + eps)" in source
    assert "SQUARE_FMA" not in source


def test_gate_order_budget_node_local_and_dispatch():
    source = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    block = source.split("# Round 28 gate begin:")[1].split("# Round 28 gate end.")[0]
    assert source.index("# Decode greedy gate end.") < source.index("# Round 28 gate begin:") < source.index('"$(mounts prefill-0)"')
    assert "SECONDS + 100 + 660 <= 1020" in block
    assert "prefill_r28_started + 90" in block and "prefill_r28_started + 100" in block
    assert 'TRITON_CACHE_DIR="/tmp/qwen-prefill-r28-${SLURM_JOB_ID}-${worker}/triton"' in block
    assert 'for worker in 0 1 2' in block and 'prefill_r28_pids+=("$!")' in block
    assert source.index("test -s") < source.index("prefill-r28-dispatch-${worker}.txt")
    assert 'grep -a \'Prefill r28 .* using \'' in source
    assert "unset SGLANG_PREFILL_GEMM_TAIL_SPLIT SGLANG_PREFILL_FA3_PAIR_GLOBAL_ORDER SGLANG_PREFILL_NORM_STORE_PAIR" in source
    subprocess.run(["bash", "-n", str(ROOT / "platforms/firmus/framework-comparison/sglang.sbatch")], check=True)


def test_config_staging_and_image_pin():
    config = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    base = json.loads(subprocess.check_output(["git", "show", "8d7848f:configs/firmus-prefill-pp4.json"], cwd=ROOT, text=True))
    assert config["images"] == base["images"]
    assert all(config["prefill"][k] is True for k in ("gemm_tail_split_gate", "fa3_pair_global_order_gate", "norm_store_pair_gate"))
    assert config["sglang_overlay_patches"][:len(base["sglang_overlay_patches"])] == base["sglang_overlay_patches"]
    assert config["deep_gemm_overlay_patches"] == base["deep_gemm_overlay_patches"]
    settings = (ROOT / "scripts/prepare-firmus-pp4-settings.py").read_text()
    for s in G.SPECS.values(): assert '"' + s["key"] + '": "0"' in settings
    provision = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    assert 'scripts/select-firmus-prefill-r28.py' in provision and 'scripts/firmus_prefill_r28.py' in provision
    assert 'prefill-pair-global-order.patch' in (ROOT / "kernels/fa3-verify/build.py").read_text()


@pytest.mark.parametrize("elapsed,starts", ((260, True), (261, False)))
def test_actual_shell_budget_and_worker_timeout_keep_control(tmp_path, elapsed, starts):
    source = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    block = "# Round 28 gate begin:" + source.split("# Round 28 gate begin:")[1].split("# Round 28 gate end.")[0]
    for directory in ("source", "results", "logs"):
        (tmp_path / directory).mkdir()
    (tmp_path / "source/firmus-framework.json").write_text(json.dumps({"prefill": {"gemm_tail_split_gate": True}, "images": {"sglang": {"sha256": "a" * 64}}}))
    env = tmp_path / "source/prefill-kernels.env"
    pin = "SGLANG_PREFILL_FA3_VERIFY=control\nSGLANG_PREFILL_GEMM_TAIL_SPLIT=0\n"
    env.write_text(pin)
    program = '''set -euo pipefail
ROOT=$1; ART=$1/logs; HEAD=head; TAIL=tail; SGLANG=unused; SLURM_JOB_ID=123; prefill_pp=4
mounts() { echo unused; }
date() { echo 1000; }
timeout() { printf 'timeout %s\n' "$*" >>"$ROOT/timeout-calls"; return 124; }
SECONDS=$2
''' + block
    subprocess.run(["bash", "-c", program, "r28", str(tmp_path), str(elapsed)], check=True)
    assert env.read_text() == pin
    assert (tmp_path / "timeout-calls").exists() is starts
    if starts:
        calls = (tmp_path / "timeout-calls").read_text().splitlines()
        assert len(calls) == 4 and sum("--nproc-per-node=4" in c for c in calls) == 3
        assert all('TRITON_CACHE_DIR=/tmp/qwen-prefill-r28-123-' in c for c in calls[:3])
    else:
        assert json.loads((tmp_path / "results/prefill-r28/selection.json").read_text())["adopted"] is False


def test_pair_assignment_actual_cuda_body_on_cpu(tmp_path):
    patch = (ROOT / "kernels/fa3-verify/prefill-pair-global-order.patch").read_text()
    added = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    body = added[added.index("            if constexpr (PairGlobalOrder)"):added.index("---") if "---" in added else added.index("        flash::")]
    # Trim the scheduler body before the separate launch-template hunk.
    body = body[:body.rfind("}") + 1]
    program = '''#include <algorithm>
#include <cassert>
#include <iostream>
#include <set>
#include <tuple>
using namespace std;
namespace cute {int ceil_div(int a,int b){return (a+b-1)/b;}}
struct Params {const int* cu_seqlens; int num_batch=2,num_head=4,qhead_per_khead=16;};
struct Work {int unused,block,head,batch;};
constexpr bool PairGlobalOrder=true; constexpr int kBlockM=192;
Work order(Params const& params,int next_tile_idx) {
''' + body + '''return {0,0,0,2};}
int main() { for (int x=3900;x<=4096;x++) for(int y : {3900,3994,4096}) {
int cu[3]={0,x,x+y}; Params p{cu};set<tuple<int,int,int>> seen;int previous=100000;
int blocks[2]={(x*16+191)/192,(y*16+191)/192};int total=(blocks[0]+blocks[1])*4;
for(int i=0;i<total;i++){auto w=order(p,i);assert(w.batch>=0&&w.batch<2&&w.block>=0&&w.block<blocks[w.batch]&&w.head>=0&&w.head<4);assert(seen.insert({w.batch,w.block,w.head}).second);
int cost=(min(w.batch?y:x,(w.block+1)*12)+95)/96;assert(cost<=previous);previous=cost;}
assert(int(seen.size())==total&&order(p,total).batch==2);
} }
'''
    cpp = tmp_path / "order.cpp"
    cpp.write_text(program)
    subprocess.run(["g++", "-std=c++17", "-O2", str(cpp), "-o", str(tmp_path / "order")], check=True)
    subprocess.run([str(tmp_path / "order")], check=True)
