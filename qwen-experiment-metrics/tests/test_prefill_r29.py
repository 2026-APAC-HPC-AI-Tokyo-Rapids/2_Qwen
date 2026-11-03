"""QKV-only all-UUID decision, real serving dispatch and allocation budget."""
import ast
import copy
import importlib.util
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("firmus_prefill_r29", ROOT / "scripts/firmus_prefill_r29.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
sys.modules["firmus_prefill_r29"] = G
SELECTOR = runpy.run_path(str(ROOT / "scripts/select-firmus-prefill-r29.py"))


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
                record = {**identity, "run_id": "123-r29", "worker": path.name, "rank": rank, "schema": G.SCHEMA, "candidate": name, "device": "NVIDIA H200", "healthy": True, "image_sha256": "a" * 64, "source_sha256": "b" * 64, "fa3_pin": "P3", "elapsed_seconds": 50, "finished_at": now, "shapes": {s: copy.deepcopy(shape) for s in settings["shapes"]}}
                (path / f"rank-{rank}.json").write_text(json.dumps(record))
    pair = tmp_path / "pair.json"
    pair.write_text(json.dumps({"schema": "firmus-prefill-fa3-pair/v1", "gpus": identities, "adopted": False}))
    return SimpleNamespace(aggregate=tmp_path, env_output=env, pair_selection=pair, run_id="123-r29", deadline=now + 100, workers_ok=1, image_sha256="a" * 64, fa3_pin="P3")


@pytest.mark.parametrize("name", G.SPECS)
def test_twelve_exact_ranks_adopt_independently(screen, name):
    screen.workers_ok = 0  # A later candidate's timeout cannot invalidate this evidence.
    before = screen.env_output.read_text()
    assert SELECTOR["aggregate_candidate"](screen, name)["adopted"]
    key = G.SPECS[name]["key"]
    assert screen.env_output.read_text() == before.replace(key + "=0", key + "=1")
    assert not list(screen.aggregate.glob("*.tmp"))


@pytest.mark.parametrize("name", G.SPECS)
@pytest.mark.parametrize("bad", ("missing", "stale", "uuid", "allocation", "image", "source", "pin", "late", "worker-late", "elapsed", "shape", "extra-shape", "bytes", "restore", "endurance", "timing", "reverse", "relative", "absolute", "nan", "overflow"))
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
        elif bad == "worker-late": record["finished_at"] = screen.deadline - 9
        elif bad == "elapsed": record["elapsed_seconds"] = 90
        elif bad == "shape": record["shapes"].pop(next(iter(record["shapes"])))
        elif bad == "extra-shape": record["shapes"]["O"] = copy.deepcopy(shape)
        elif bad == "bytes": shape["bytes_equal"] = False
        elif bad == "restore": shape["restored"] = False
        elif bad == "endurance": shape["endurance"]["bytes_equal"] = False
        elif bad == "timing": shape["samples_us"]["forward"]["candidate"].pop()
        elif bad == "reverse": shape["samples_us"]["reverse"]["candidate"] = [1001.] * G.ROUNDS
        elif bad == "relative": shape["samples_us"]["forward"]["candidate"] = [996.] * G.ROUNDS
        elif bad == "absolute": shape["samples_us"]["forward"] = {"control": [100.] * G.ROUNDS, "candidate": [99.] * G.ROUNDS}
        elif bad == "nan": shape["samples_us"]["forward"]["candidate"][0] = float("nan")
        path.write_text(json.dumps(record))
    assert not SELECTOR["aggregate_candidate"](screen, name)["adopted"]
    assert screen.env_output.read_bytes() == before

@pytest.fixture
def helper(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    rel = "python/sglang/srt/layers/attention/prefill_fp8.py"
    config = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    for patch in config["sglang_overlay_patches"]:
        if "+++ b/" + rel in (ROOT / patch).read_text():
            subprocess.run(["git", "apply", "--include=" + rel, str(ROOT / patch)], cwd=tmp_path, check=True)
    tree = ast.parse((tmp_path / rel).read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("dense_gemm", "_log_qkv_tail_dispatch", "linear")]
    namespace = {"os": os, "_qkv_tail_dispatch_logged": False}
    exec(compile(tree, "<actual r29 host overlay>", "exec"), namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize("enabled,rows,n,k,count", ((False, 7988, 9216, 4096, 1), (True, 7988, 9216, 4096, 2), (True, 7989, 9216, 4096, 1), (True, 3994, 9216, 4096, 1), (True, 7988, 4096, 8192, 1), (True, 7988, 9216, 8192, 1)))
def test_actual_split_launches_only_the_last_m_row(helper, monkeypatch, enabled, rows, n, k, count):
    calls = []
    monkeypatch.setitem(sys.modules, "deep_gemm", SimpleNamespace(fp8_gemm_nt=lambda *a, **kw: calls.append((a, kw))))
    class Tensor:
        def __init__(self, name, shape): self.name, self.shape = name, shape
        def __getitem__(self, key): return self.name, key.start, key.stop
    a, sa, w, s, out = (Tensor("a", (rows, k)), Tensor("sa", (rows, k // 128)), Tensor("w", (n, k)), object(), Tensor("out", (rows, n)))
    helper["dense_gemm"](a, sa, w, s, out, tail_split=enabled)
    assert len(calls) == count and all(k == {"compiled_dims": "nk"} for _, k in calls)
    if count == 2:
        assert calls[0][0][0] == (("a", None, 7936), ("sa", None, 7936))
        assert calls[1][0][0] == (("a", 7936, None), ("sa", 7936, None))
        assert calls[0][0][2] == ("out", None, 7936) and calls[1][0][2] == ("out", 7936, None)
    else:
        assert calls[0][0] == ((a, sa), (w, s), out)


@pytest.mark.parametrize("projection,enabled,rows,count", (("QKV", True, 7988, 2), ("QKV", False, 7988, 1), ("QKV", True, 7989, 1), ("O", True, 7988, 1)))
def test_serving_linear_dispatch_logs_only_actual_qkv_once(helper, monkeypatch, caplog, projection, enabled, rows, count):
    calls = []
    monkeypatch.setitem(sys.modules, "deep_gemm", SimpleNamespace(fp8_gemm_nt=lambda *a, **kw: calls.append((a, kw))))
    monkeypatch.setenv("SGLANG_PREFILL_QKV_TAIL_SPLIT", "1" if enabled else "0")
    class Tensor:
        def __init__(self, shape): self.shape, self.ndim, self.device = shape, 2, "cuda"
        def __getitem__(self, key): return (self, key.start, key.stop)
    n, k = (9216, 4096) if projection == "QKV" else (4096, 8192)
    a, sa = Tensor((rows, k)), Tensor((rows, k // 128))
    layer = SimpleNamespace(weight=Tensor((n, k)), weight_scale_inv=object())
    helper["linear_supported"] = lambda *args: True
    helper["torch"] = SimpleNamespace(empty=lambda shape, **kw: Tensor(shape), bfloat16=object())
    for _ in range(2):
        out, bias = helper["linear"](object(), projection, layer, (a, sa))
        assert out.shape == (rows, n) and bias is None
    assert len(calls) == count * 2
    logs = [r.getMessage() for r in caplog.records if "Prefill r29 qkv-tail using " in r.getMessage()]
    assert len(logs) == (1 if count == 2 else 0)
    if logs:
        assert "projection=QKV M=7988 N=9216 K=4096" in logs[0]


def test_dispatch_collection_preserves_rank_lines_and_explicit_absence(tmp_path):
    source = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    block = source[source.index("for worker in 0 1 2; do", source.index("# Keep every process dispatch line")):].split("done", 1)[0] + "done\n"
    (tmp_path / "results").mkdir()
    for worker in range(3):
        (tmp_path / f"prefill-{worker}.log").write_text("unrelated line\n")
    lines = "".join(f"rank-{rank}: Prefill r29 qkv-tail using split 7936+52 rows projection=QKV M=7988 N=9216 K=4096 pid={rank}\n" for rank in range(4))
    (tmp_path / "prefill-0.log").write_text(lines)
    subprocess.run(["bash", "-c", 'set -euo pipefail\nROOT=$1\nART=$1\n' + block, "r29", str(tmp_path)], check=True)
    assert (tmp_path / "results/prefill-r29-dispatch-0.txt").read_text() == lines
    for worker in (1, 2):
        assert (tmp_path / f"results/prefill-r29-dispatch-{worker}.txt").read_text() == f"no Prefill r29 dispatch line in prefill-{worker}.log\n"


def test_gate_order_budget_node_local_and_dispatch():
    source = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    block = source.split("# Round 29 gate begin:")[1].split("# Round 29 gate end.")[0]
    assert source.index("# Decode greedy gate end.") < source.index("# Round 29 gate begin:") < source.index('"$(mounts prefill-0)"')
    assert "SECONDS + 100 + 660 <= 1020" in block
    assert "prefill_r29_started + 90" in block and "prefill_r29_started + 100" in block
    assert 'TRITON_CACHE_DIR="/tmp/qwen-prefill-r29-${SLURM_JOB_ID}-${worker}/triton"' in block
    assert 'DG_JIT_CACHE_DIR="/tmp/qwen-prefill-r29-${SLURM_JOB_ID}-${worker}/deep-gemm"' in block
    assert 'for worker in 0 1 2' in block and 'prefill_r29_pids+=("$!")' in block
    assert source.index("test -s") < source.index("prefill-r29-dispatch-${worker}.txt")
    assert 'grep -a \'Prefill r29 qkv-tail using \'' in source
    assert "unset SGLANG_PREFILL_QKV_TAIL_SPLIT" in source
    subprocess.run(["bash", "-n", str(ROOT / "platforms/firmus/framework-comparison/sglang.sbatch")], check=True)


def test_config_staging_and_image_pin():
    config = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    # Pin the round 25 values directly; CI does not need any git history.
    image = {"path": "/scratch/home/ACCOUNT/qwen/containers/41573354672.sqsh", "sha256": "a001f90142b571380894e9532e467c57b831397c9e2e9d613d9e04004c87def9", "size_bytes": 12298305536, "provenance": "/scratch/home/ACCOUNT/qwen/containers/41573354672.container-sidecar.json"}
    assert config["images"]["sglang"] == image
    assert config["container_build_run_id"] == 41573354672
    assert config["container_sha256"] == image["sha256"]
    assert config["container_size_bytes"] == image["size_bytes"]
    assert config["container_source_commit"] == "14c3e861d52185f4a2753620213caf629c5d0400"
    assert set(G.SPECS) == {"qkv-tail"} and G.SPECS["qkv-tail"]["shapes"] == ("QKV",)
    assert config["prefill"]["qkv_tail_split_gate"] is True
    assert config["sglang_overlay_patches"][-2:] == ["patches/sglang/decode-greedy-verify.patch", "patches/sglang/prefill-qkv-tail-r29.patch"]
    assert not any("r29" in p for p in config["deep_gemm_overlay_patches"])
    settings = (ROOT / "scripts/prepare-firmus-pp4-settings.py").read_text()
    for s in G.SPECS.values(): assert '"' + s["key"] + '": "0"' in settings
    provision = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    assert 'scripts/select-firmus-prefill-r29.py' in provision and 'scripts/firmus_prefill_r29.py' in provision


@pytest.mark.parametrize("elapsed,starts", ((260, True), (261, False)))
def test_actual_shell_budget_and_worker_timeout_keep_control(tmp_path, elapsed, starts):
    source = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    block = "# Round 29 gate begin:" + source.split("# Round 29 gate begin:")[1].split("# Round 29 gate end.")[0]
    for directory in ("source", "results", "logs"):
        (tmp_path / directory).mkdir()
    (tmp_path / "source/firmus-framework.json").write_text(json.dumps({"prefill": {"qkv_tail_split_gate": True}, "images": {"sglang": {"sha256": "a" * 64}}}))
    env = tmp_path / "source/prefill-kernels.env"
    pin = "SGLANG_PREFILL_FA3_VERIFY=control\nSGLANG_PREFILL_QKV_TAIL_SPLIT=0\n"
    env.write_text(pin)
    program = '''set -euo pipefail
ROOT=$1; ART=$1/logs; HEAD=head; TAIL=tail; SGLANG=unused; SLURM_JOB_ID=123; prefill_pp=4
mounts() { echo unused; }
date() { echo 1000; }
timeout() { printf 'timeout %s\n' "$*" >>"$ROOT/timeout-calls"; return 124; }
SECONDS=$2
''' + block
    subprocess.run(["bash", "-c", program, "r29", str(tmp_path), str(elapsed)], check=True)
    assert env.read_text() == pin
    assert (tmp_path / "timeout-calls").exists() is starts
    if starts:
        calls = (tmp_path / "timeout-calls").read_text().splitlines()
        assert len(calls) == 4 and sum("--nproc-per-node=4" in c for c in calls) == 3
        assert all('TRITON_CACHE_DIR=/tmp/qwen-prefill-r29-123-' in c for c in calls[:3])
    else:
        assert json.loads((tmp_path / "results/prefill-r29/selection.json").read_text())["adopted"] is False
