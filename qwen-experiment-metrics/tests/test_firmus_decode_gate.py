"""CPU checks for the Decode-only BF16 router gate overlay and its bench."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_function(path, name, namespace):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102 - actual implementation, no CUDA imports
    return namespace[name]


@pytest.fixture
def source(tmp_path):
    value = os.environ.get("QWEN_SGLANG_SOURCE")
    if value:
        return Path(value)
    subprocess.run(["git", "apply", "--include=python/sglang/kernels/ops/moe/decode_gate.py", str(ROOT / "patches/sglang/decode-gate-gemm.patch")], cwd=tmp_path, check=True)
    return tmp_path / "python/sglang"


@pytest.mark.parametrize("setting", [None, "", "triton", "cublas", "TRITON"])
@pytest.mark.parametrize("mode", ["decode", "target_verify", "extend", "draft_extend_v2", "idle"])
@pytest.mark.parametrize("rows", [1, 63, 64, 65, 128, 129, 144, 145, 4096])
def test_gate_selection_leaves_prefill_and_small_batches_unchanged(source, monkeypatch, setting, mode, rows):
    if setting is None:
        monkeypatch.delenv("SGLANG_DECODE_GATE_GEMM", raising=False)
    else:
        monkeypatch.setenv("SGLANG_DECODE_GATE_GEMM", setting)
    ns = {"os": os, "torch": types.SimpleNamespace(bfloat16="bf16")}
    call = load_function(source / "kernels/ops/moe/decode_gate.py", "can_use_decode_gate", ns)
    hidden = types.SimpleNamespace(ndim=2, shape=(rows, 4096), dtype="bf16", is_contiguous=lambda: True)
    weight = types.SimpleNamespace(shape=(128, 4096), dtype="bf16", is_contiguous=lambda: True)
    forward_mode = types.SimpleNamespace(is_decode=lambda: mode == "decode", is_target_verify=lambda: mode == "target_verify")
    assert call(hidden, weight, forward_mode) == (setting == "triton" and mode in ("decode", "target_verify") and 64 <= rows <= 144)
    weight.dtype = "fp32"
    assert not call(hidden, weight, forward_mode)
    weight.dtype = "bf16"
    hidden.is_contiguous = lambda: False
    assert not call(hidden, weight, forward_mode)


def test_existing_five_bench_overlays_remain_unchanged():
    paths = json.loads((ROOT / "configs/firmus-framework.json").read_text())["sglang_overlay_patches"]
    bench = sorted((ROOT / "patches/sglang-bench").glob("*.patch"))
    assert len(bench) == 5
    assert [p.read_bytes() for p in bench] == [(ROOT / p).read_bytes() for p in paths[:5]]


def test_gate_launch_covers_every_output_without_split_k(source):
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kw):
                launches.append((grid, args, kw))
            return launch

    ns = {"torch": types.SimpleNamespace(bfloat16="bf16", empty=lambda shape, **kw: (shape, kw)), "triton": types.SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b), "_decode_gate_kernel": Kernel()}
    call = load_function(source / "kernels/ops/moe/decode_gate.py", "decode_gate", ns)
    for rows in (64, 65, 128, 129, 144):
        hidden = types.SimpleNamespace(ndim=2, shape=(rows, 4096), dtype="bf16", is_cuda=True, device="cuda", is_contiguous=lambda: True)
        weight = types.SimpleNamespace(shape=(128, 4096), dtype="bf16", device="cuda", is_contiguous=lambda: True)
        result = call(hidden, weight)
        grid, args, kw = launches[-1]
        assert grid == ((rows + 15) // 16, 8)
        assert args[:2] == (hidden, weight) and args[3] == rows
        assert result[0] == (rows, 128)
        assert (kw["BM"], kw["BN"], kw["BK"]) == (16, 16, 256)


def test_graph_timer_runs_whole_weight_working_set():
    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait_stream(self, *args):
            pass

    calls = []
    event = types.SimpleNamespace(record=lambda: None, synchronize=lambda: None, elapsed_time=lambda end: 0.094)
    cuda = types.SimpleNamespace(Stream=Context, current_stream=Context, stream=lambda stream: Context(), graph=lambda graph: Context(), CUDAGraph=lambda: types.SimpleNamespace(replay=lambda: None), Event=lambda **kw: event)
    import statistics

    ns = {"torch": types.SimpleNamespace(cuda=cuda), "REPS": 20, "REPLAYS": 25, "statistics": statistics}
    timer = load_function(ROOT / "scripts/bench-firmus-sglang-kernels.py", "time_graph", ns)
    result = timer([lambda index=i: calls.append(index) for i in range(94)])
    # One warm-up call plus one captured call for every distinct layer weight.
    assert calls == list(range(94)) * 2
    assert result["median_us"] == pytest.approx(1.0)


@pytest.mark.parametrize("candidate", [False, True])
def test_gate_bench_pairs_all_layer_inputs_and_keeps_same_cases(monkeypatch, candidate):
    import importlib.util
    import inspect
    import sys

    made, operations, output = [], [], []

    class Tensor:
        def __truediv__(self, value):
            assert value == 64
            return self

    def randn(*args, **kw):
        value = Tensor()
        made.append(value)
        return value

    def linear(hidden, weight):
        operations.append(("cublas", hidden, weight))
        return "logits"

    def kernel(hidden, weight):
        operations.append(("candidate", hidden, weight))
        return "logits"

    def gate(logits, bias, topk, **kw):
        return "weights", "ids"

    def fused_gate(logits, bias, topk, num_token_non_padded=None, **kw):
        return "weights", "ids"

    module_name = "sglang.kernels.ops.moe.decode_gate"
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == module_name and candidate else None)
    monkeypatch.setitem(sys.modules, module_name, types.SimpleNamespace(decode_gate=kernel))
    monkeypatch.setitem(sys.modules, "sglang.kernels.ops.moe.moe_fused_gate", types.SimpleNamespace(moe_fused_gate=fused_gate if candidate else gate))

    def timing(calls):
        begin = len(operations)
        for call in calls:
            call()
        batch = operations[begin:]
        assert len(batch) in (1, 94)
        assert len({id(weight) for _, _, weight in batch}) == len(batch)
        # The exact inputs made for this rows/working-set pair are shared by
        # cuBLAS and production, both for GEMM-only and GEMM+router timings.
        expected = made[-2 * len(batch):]
        assert [(h, w) for _, h, w in batch] == list(zip(expected[::2], expected[1::2]))
        return {"median_us": 1.0, "p10_us": 1.0, "p90_us": 1.0}

    torch = types.SimpleNamespace(bfloat16="bf16", int32="i32", float32="fp32", nn=types.SimpleNamespace(functional=types.SimpleNamespace(linear=linear)), Generator=lambda **kw: types.SimpleNamespace(manual_seed=lambda seed: None), randn=randn, tensor=lambda *args, **kw: object(), zeros=lambda *args, **kw: object())
    ns = {"torch": torch, "inspect": inspect, "_mask_topk_ids_padded_region": lambda ids, count: None, "sha256": lambda *args: "hash", "time_graph": timing}
    bench = load_function(ROOT / "scripts/bench-firmus-sglang-kernels.py", "bench_decode_gate", ns)
    bench(lambda case, digest, timing, **kw: output.append((case, kw)))
    assert len(output) == 40 and len({case for case, _ in output}) == 40
    assert all(kw["calls_per_decode_step"] == 94 for _, kw in output)
    assert sum(kw["layers_in_graph"] == 94 for _, kw in output) == 20
    assert {kind for kind, _, _ in operations} == ({"cublas", "candidate"} if candidate else {"cublas"})


def selector_functions():
    import io
    import math
    import statistics
    import traceback
    from contextlib import redirect_stdout

    ns = {"math": math, "statistics": statistics, "GATE_SELECTION_PAIRS": 20, "io": io, "traceback": traceback, "redirect_stdout": redirect_stdout, "Path": Path, "__file__": str(ROOT / "scripts/bench-firmus-sglang-kernels.py")}
    for name in ("gate_timing_summary", "select_decode_gate", "check_gate_for_selection", "measure_gate_selection_blocks", "run_decode_gate_selection"):
        load_function(ROOT / "scripts/bench-firmus-sglang-kernels.py", name, ns)
    return ns


def rank_timings(rank, cublas=(1000.0, 1020.0), triton=(700.0, 710.0)):
    return {"rank": rank, "numeric": {"passed": True}, "blocks": [{name: {"samples_us": [value] * 20} for name, value in (("cublas", a), ("triton", b))} for a, b in zip(cublas, triton)]}


def test_selection_requires_200us_on_all_four_gpus_and_both_blocks_faster():
    select = selector_functions()["select_decode_gate"]
    ranks = [rank_timings(rank) for rank in range(4)]
    result = select(ranks)
    assert result["selected"] == "triton" and result["env"] == {"SGLANG_DECODE_GATE_GEMM": "triton"}
    assert result["rank_savings_us"][0]["pooled_median_savings_us"] == 305.0
    ranks[3] = rank_timings(3, (1000, 1000), (800, 800))
    assert select(ranks)["selected"] == "triton"  # inclusive threshold
    ranks[3] = rank_timings(3, (1000, 1000), (800.01, 800.01))
    assert select(ranks)["reason"] == "less_than_200us_on_at_least_one_gpu"
    ranks[3] = rank_timings(3, (1000, 1000), (200, 1000))
    result = select(ranks)
    assert result["env"] == {} and result["reason"] == "not_faster_in_both_blocks_on_every_gpu"


@pytest.mark.parametrize("failure", ["numeric", "skip", "error", "missing_block", "missing_sample", "nan", "inf", "negative", "missing_rank", "duplicate_rank"])
def test_selection_does_not_enable_failed_or_incomplete_comparisons(failure):
    ranks = [rank_timings(rank) for rank in range(4)]
    target = ranks[2]
    if failure == "numeric":
        target["numeric"] = {"passed": False, "error": "AssertionError"}
    elif failure == "skip":
        target["numeric"] = {}
    elif failure == "error":
        target["error"] = "CUDA graph failure"
    elif failure == "missing_block":
        target["blocks"].pop()
    elif failure == "missing_sample":
        target["blocks"][0]["triton"]["samples_us"].pop()
    elif failure in ("nan", "inf", "negative"):
        target["blocks"][0]["triton"]["samples_us"][0] = {"nan": float("nan"), "inf": float("inf"), "negative": -1}[failure]
    elif failure == "missing_rank":
        ranks.pop()
    else:
        target["rank"] = 0
    result = selector_functions()["select_decode_gate"](ranks)
    assert result["selected"] == "cublas" and result["env"] == {}


def test_block_timer_alternates_ab_and_preserves_94_layer_units():
    ns = selector_functions()
    order = []
    durations_ms = iter([value for a, b in ((0.8, 0.5), (0.82, 0.51)) for _ in range(20) for value in (a, b)])

    class Event:
        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return next(durations_ms)

    ns["torch"] = types.SimpleNamespace(cuda=types.SimpleNamespace(Event=lambda **kw: Event(), synchronize=lambda: None))
    graphs = {name: types.SimpleNamespace(replay=lambda name=name: order.append(name)) for name in ("cublas", "triton")}
    blocks = ns["measure_gate_selection_blocks"](graphs)
    assert order == ["cublas", "triton"] * (3 + 40)
    assert blocks[0]["order"] == "AB" * 20 and blocks[1]["order"] == "AB" * 20
    assert blocks[0]["cublas"]["median_us"] == pytest.approx(800)
    assert blocks[1]["triton"]["median_us"] == pytest.approx(510)
    assert len(blocks[0]["triton"]["samples_us"]) == 20
    summary = ns["gate_timing_summary"](list(range(1, 21)))
    assert summary == {"median_us": 10.5, "p10_us": 3, "p90_us": 19}


@pytest.mark.parametrize("status", [True, False, None, "exception"])
def test_numeric_selection_check_requires_explicit_pass_and_saves_log(status):
    ns = selector_functions()

    def check():
        print("numeric diagnostic")
        if status == "exception":
            raise AssertionError("wrong gate output")
        return status

    ns["run_path"] = lambda path: {"check_decode_gate": check}
    result = ns["check_gate_for_selection"]()
    assert result["passed"] is (status is True)
    assert "numeric diagnostic" in result["log"]
    if status == "exception":
        assert "wrong gate output" in result["error"]


@pytest.mark.parametrize("setting,rows,mode", [(None, 128, "target_verify"), ("cublas", 128, "target_verify"), ("triton", 128, "target_verify"), ("triton", 63, "target_verify"), ("triton", 128, "extend")])
def test_model_dispatch_preserves_original_gate_call_when_not_selected(source, monkeypatch, setting, rows, mode):
    import sys
    import textwrap

    if setting is None:
        monkeypatch.delenv("SGLANG_DECODE_GATE_GEMM", raising=False)
    else:
        monkeypatch.setenv("SGLANG_DECODE_GATE_GEMM", setting)
    ns = {"os": os, "torch": types.SimpleNamespace(bfloat16="bf16")}
    eligible = load_function(source / "kernels/ops/moe/decode_gate.py", "can_use_decode_gate", ns)
    calls = []
    logits = object()
    hidden = types.SimpleNamespace(ndim=2, shape=(rows, 4096), dtype="bf16", is_contiguous=lambda: True)
    weight = types.SimpleNamespace(shape=(128, 4096), dtype="bf16", is_contiguous=lambda: True)

    class Gate:
        def __init__(self):
            self.weight = weight

        def __call__(self, value):
            assert value is hidden
            calls.append("cublas")
            return logits, None

    def triton(value, matrix):
        assert value is hidden and matrix is weight
        calls.append("triton")
        return logits

    name = "sglang.kernels.ops.moe.decode_gate"
    module = types.ModuleType(name)
    module.can_use_decode_gate, module.decode_gate = eligible, triton
    monkeypatch.setitem(sys.modules, name, module if setting == "triton" else None)
    patch = (ROOT / "patches/sglang/decode-gate-gemm.patch").read_text()
    hunk = patch.split("+++ b/python/sglang/srt/models/qwen3_moe.py\n", 1)[1].split("\n", 1)[1]
    code = "".join(line[1:] for line in hunk.splitlines(True) if line.startswith((" ", "+"))).split("            # A static", 1)[0]
    environment = {"num_tokens": rows, "cap": 4096, "self": types.SimpleNamespace(gate=Gate()), "hidden_states": hidden, "forward_batch": types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda: mode == "decode", is_target_verify=lambda: mode == "target_verify"))}
    exec(compile(ast.parse(textwrap.dedent(code)), "actual_model_gate_dispatch", "exec"), environment)  # noqa: S102 - actual added model hunk, with the gate and CUDA imports mocked
    assert calls == (["triton"] if setting == "triton" and rows >= 64 and mode == "target_verify" else ["cublas"])
    assert environment["router_logits"] is logits


@pytest.mark.parametrize("failure", [None, "numeric_rank2", "timing"])
def test_four_gpu_selection_writes_artifact_and_never_times_after_numeric_failure(tmp_path, monkeypatch, failure):
    import copy
    import sys

    ns = selector_functions()
    operations = []
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(set_device=lambda device: operations.append(("device", device)), get_device_name=lambda: "H200")
    dist = types.ModuleType("torch.distributed")
    torch.distributed = dist
    dist.init_process_group = lambda **kw: operations.append(("init", kw))
    dist.get_rank = lambda: 0
    dist.barrier = lambda: operations.append("barrier")
    dist.destroy_process_group = lambda: operations.append("destroy")

    def gather(ranks, local):
        operations.append("gather")
        for rank in range(4):
            ranks[rank] = dict(copy.deepcopy(local), rank=rank)
        if failure == "numeric_rank2":
            ranks[2]["numeric"] = {"passed": False, "error": "numeric mismatch"}

    def measure():
        operations.append("measure")
        if failure == "timing":
            raise RuntimeError("timing graph failed")
        return {"blocks": rank_timings(0)["blocks"]}

    dist.all_gather_object = gather
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "0")
    ns.update(torch=torch, os=os, json=json, time=types.SimpleNamespace(monotonic=lambda: 1.0), check_gate_for_selection=lambda: {"passed": True, "log": "gate numerics"}, measure_decode_gate_selection=measure)
    output = tmp_path / "artifacts/selection.json"
    ns["run_decode_gate_selection"](output)
    result = json.loads(output.read_text())
    assert len(result["ranks"]) == 4 and operations[-1] == "destroy"
    assert result["shape"]["layers"] == 94
    if failure is None:
        assert result["env"] == {"SGLANG_DECODE_GATE_GEMM": "triton"}
        assert operations.count("gather") == 2
    else:
        assert result["selected"] == "cublas" and result["env"] == {}
        if failure == "numeric_rank2":
            assert "measure" not in operations and "barrier" not in operations
            assert operations.count("gather") == 1
        else:
            assert all("timing graph failed" in rank["error"] for rank in result["ranks"])


def test_selection_cli_bypasses_standalone_bench(monkeypatch):
    import argparse
    import sys

    calls = []
    ns = {"argparse": argparse, "run_decode_gate_selection": lambda output: calls.append(output)}
    main = load_function(ROOT / "scripts/bench-firmus-sglang-kernels.py", "main", ns)
    monkeypatch.setattr(sys, "argv", ["bench", "--select-decode-gate", "--output", "selection.json"])
    main()
    assert calls == ["selection.json"]


def test_job_comparison_captures_same_94_layers_and_reports_block_fluctuation(monkeypatch):
    import sys

    ns = selector_functions()
    inputs, calls, routes = [], [], []

    class Tensor:
        def __truediv__(self, divisor):
            assert divisor == 64
            return self

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait_stream(self, stream):
            pass

    def randn(*shape, **kw):
        assert shape in ((128, 4096),)
        tensor = Tensor()
        inputs.append(tensor)
        return tensor

    def gate(name):
        def call(hidden, weight):
            calls.append((name, hidden, weight))
            return "logits"
        return call

    def route(logits, bias, topk, **kw):
        assert logits == "logits" and bias is None and topk == 8
        routes.append(kw)
        return "weights", "ids"

    def blocks(graphs):
        assert len(graphs) == 2 and graphs["cublas"] is not graphs["triton"]
        result = rank_timings(0)["blocks"]
        for block in result:
            for name in ("cublas", "triton"):
                block[name].update(ns["gate_timing_summary"](block[name]["samples_us"]))
        return result

    count = object()
    cuda = types.SimpleNamespace(Stream=Context, current_stream=Context, stream=lambda stream: Context(), CUDAGraph=object, graph=lambda graph: Context())
    ns.update(torch=types.SimpleNamespace(Generator=lambda **kw: types.SimpleNamespace(manual_seed=lambda seed: None), randn=randn, tensor=lambda *args, **kw: count, bfloat16="bf16", int32="i32", cuda=cuda, nn=types.SimpleNamespace(functional=types.SimpleNamespace(linear=gate("cublas")))), measure_gate_selection_blocks=blocks)
    monkeypatch.setitem(sys.modules, "sglang.kernels.ops.moe.decode_gate", types.SimpleNamespace(decode_gate=gate("triton")))
    monkeypatch.setitem(sys.modules, "sglang.kernels.ops.moe.moe_fused_gate", types.SimpleNamespace(moe_fused_gate=route))
    call = load_function(ROOT / "scripts/bench-firmus-sglang-kernels.py", "measure_decode_gate_selection", ns)
    result = call()
    assert len(inputs) == 188 and len(calls) == len(routes) == 376
    paired = list(zip(inputs[::2], inputs[1::2]))
    assert [(h, w) for name, h, w in calls if name == "cublas"] == paired * 2
    assert [(h, w) for name, h, w in calls if name == "triton"] == paired * 2
    assert all(kw["num_token_non_padded"] is count for kw in routes)
    assert result["pooled"]["cublas"]["median_us"] == 1010.0
    assert result["block_fluctuation"]["triton"]["block1_minus_block0_median_us"] == 10.0


@pytest.mark.parametrize("setting", [None, "cublas", "triton"])
def test_regular_serving_gate_skips_disabled_candidate_but_keeps_other_checks(monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv("SGLANG_DECODE_GATE_GEMM", raising=False)
    else:
        monkeypatch.setenv("SGLANG_DECODE_GATE_GEMM", setting)
    source = ast.parse((ROOT / "scripts/check-firmus-decode-kernels.py").read_text())
    main = next(node for node in source.body if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")
    calls = []
    functions = {node.name for node in source.body if isinstance(node, ast.FunctionDef)}
    ns = {name: lambda *args, name=name, **kw: calls.append(name) for name in functions}
    ns.update(__name__="__main__", os=os, sys=sys, torch=types.SimpleNamespace(cuda=types.SimpleNamespace(set_device=lambda device: None)))
    exec(compile(ast.Module(body=[main], type_ignores=[]), "serving_gate_entrypoint", "exec"), ns)  # noqa: S102 - actual entrypoint, all numerical functions mocked
    assert ("check_decode_gate" in calls) is (setting == "triton")
    assert "check_decode_router_padding" in calls and "check_dense_gemm" in calls and "check_verify_scheduler_sharing" in calls
