"""GPU-free coverage of Decode config enumeration, capture modes and numeric gates."""

import ast
import json
import os
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENV = "DG_SM90_1D2D_DECODE_CONFIG"


def load_dense():
    source = ast.parse("\n".join((ROOT / "scripts" / name).read_text() for name in ("bench-firmus-sglang-kernels.py", "check-firmus-decode-kernels.py")))
    names = {"dense_gemm_mode", "dense_candidates", "require_dense_host", "check_dense_errors", "dense_call", "selected_dense_candidates"}
    selected = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    selected += [node for node in source.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith("DENSE_") for t in node.targets)]
    namespace = {"os": os, "json": json, "Path": Path, "contextmanager": contextmanager, "__file__": str(ROOT / "scripts/bench-firmus-sglang-kernels.py")}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "dense-bench", "exec"), namespace)  # noqa: S102 - selected tracked definitions only
    return namespace


@pytest.mark.parametrize("previous", [None, "", "QKV:128,80,7,1,0;O:128,32,10,1,0"])
def test_capture_switch_restored_after_nested_modes_and_failure(monkeypatch, previous):
    if previous is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, previous)
    ns = load_dense()
    stock, candidate = ns["dense_candidates"]("qkv")[:2]
    with pytest.raises(RuntimeError), ns["dense_gemm_mode"]("qkv", candidate):
        assert os.environ[ENV] == "QKV:64,144,5,1,0"
        with ns["dense_gemm_mode"]("qkv", stock):
            assert ENV not in os.environ
        assert os.environ[ENV] == "QKV:64,144,5,1,0"
        raise RuntimeError("capture failed")
    assert os.environ.get(ENV) == previous


def test_old_image_and_patch_drift_cannot_silently_benchmark_stock_twice(tmp_path):
    ns = load_dense()
    record = {"source_commit": "1c8138a59b3df5dac50b76b730834ed57982fdfb", "patches": {"sm90-1d2d-small-m.patch": "b" * 64}}
    path = tmp_path / "host.json"
    for digest in (None, "b" * 64, ns["DENSE_HOST_SHA256"]):
        if digest:
            record["patches"]["sm90-1d2d-decode-config.patch"] = digest
        path.write_text(json.dumps(record))
        if digest == ns["DENSE_HOST_SHA256"]:
            assert ns["require_dense_host"](path) == digest
        else:
            with pytest.raises(AssertionError, match="rebuilt Decode-config"):
                ns["require_dense_host"](path)
    import hashlib

    assert ns["DENSE_HOST_SHA256"] == hashlib.sha256((ROOT / "patches/deep-gemm-host/sm90-1d2d-decode-config.patch").read_bytes()).hexdigest()


@pytest.mark.parametrize("bad", [{"finite": False}, {"relative_l2": 0.006}, {"max_scaled_error": 0.02}, {"relative_l2": float("nan")}, {"max_scaled_error": float("inf")}])
def test_numeric_gate_rejects_missing_work_scale_errors_and_nonfinite(bad):
    metrics = {"finite": True, "relative_l2": 0.002, "max_scaled_error": 0.004}
    check = load_dense()["check_dense_errors"]
    check(metrics)
    with pytest.raises(AssertionError):
        check({**metrics, **bad})


@pytest.mark.parametrize("op,n,k", [("qkv", 9216, 4096), ("o", 4096, 8192)])
def test_bounded_candidate_table_is_legal_and_separates_stage_and_multicast(op, n, k):
    from math import gcd

    ns = load_dense()
    candidates = ns["dense_candidates"](op)
    assert len(candidates) == len({c["name"] for c in candidates}) == 11
    assert candidates[0]["name"] == "stock"
    for bm, bn, stages, mc, on_a in ns["DENSE_CONFIGS"][op]:
        assert bm in (64, 128) and bn % 16 == 0
        assert bn <= 128 or gcd(bn, 128) == bn - 128
        assert stages >= 4 and (mc, on_a) in ((1, 0), (2, 0), (2, 1))
        align = lambda x, a: ((x + a - 1) // a) * a
        smem = align(bm * bn * 2, 1024) + 16 * 8 * 2 + align((k // 128) * 4 * (1 if 128 % bn == 0 else 2), 8) + stages * ((bm + bn) * 128 + align(bm * 4, 128))
        assert smem <= 232448
    # Same tile/stages isolate A and B multicast; same tile/mc isolate stages.
    cfgs = set(ns["DENSE_CONFIGS"][op])
    base = (64, 144, 7) if op == "qkv" else (64, 64, 13)
    assert (*base, 2, 0) in cfgs and (*base, 2, 1) in cfgs
    assert any(c[:2] == base[:2] and c[2] < base[2] and c[3:] == (1, 0) for c in cfgs)


class Output:
    def fill_(self, _):
        return self

    def clone(self):
        return self

    def float(self):
        return self


def test_image_triton_call_keeps_group128_fp32_scale_strides_and_bf16_output(monkeypatch):
    import sys

    class Tensor:
        def __init__(self, shape, strides):
            self.shape, self.strides = shape, strides

        def stride(self, axis):
            return self.strides[axis]

    launches = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, args, kwargs))

            return launch

    monkeypatch.setitem(sys.modules, "sglang.kernels.ops.quantization.fp8_kernel", types.SimpleNamespace(_w8a8_block_fp8_matmul=Kernel()))
    ns = load_dense()
    a, sa = Tensor((132, 4096), (4096, 1)), Tensor((132, 32), (1, 132))
    b, sb = Tensor((9216, 4096), (4096, 1)), Tensor((72, 32), (32, 1))
    out = Tensor((132, 9216), (9216, 1))
    candidate = next(c for c in ns["dense_candidates"]("qkv") if c["backend"] == "triton")
    ns["dense_call"](candidate, ((a, sa), (b, sb), out))()
    grid, args, kwargs = launches[0]
    assert grid == (3 * 72,)
    assert args[:5] == (a, b, out, sa, sb)
    assert args[5:10] == (132, 9216, 4096, 128, 128)
    assert args[10:] == (4096, 1, 1, 4096, 9216, 1, 1, 132, 1, 32)
    assert kwargs == {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32, "needs_masking": False, "num_stages": 3, "num_warps": 4}




def selector():
    import runpy

    return runpy.run_path(str(ROOT / "scripts/select-firmus-decode-dense.py"))


def selection_records(blocks):
    """Identical four-rank fixtures; values are the two block medians."""
    candidates = [{"name": name} for name in blocks]
    return [{"rank": rank, "candidates": candidates, "results": {name: {"numeric_passed": values is not None, "blocks": [[value] * 5 for value in values] if values else [[], []]} for name, values in blocks.items()}} for rank in range(4)]


def test_selection_requires_half_us_and_each_block_and_chooses_fastest():
    records = selection_records({"stock": (10, 10), "threshold": (9.5, 9.5), "tiny": (9.501, 9.501), "unstable": (7, 10.1), "fast": (8, 8.5), "rejected": None})
    result = selector()["choose_candidate"](records)
    assert result["winner"] == "fast"
    assert result["summary"]["threshold"]["eligible"]
    for name in ("tiny", "unstable", "rejected"):
        assert not result["summary"][name]["eligible"]
    assert result["summary"]["fast"]["block_median_range_us"] == 0.5
    assert result["summary"]["fast"]["median_us"] == 8.25
    assert result["summary"]["fast"]["p10_us"] == 8
    assert result["summary"]["fast"]["p90_us"] == 8.5


@pytest.mark.parametrize("blocks", [{"stock": (10, 10)}, {"stock": (10, 10), "tiny": (9.6, 9.6)}, {"stock": (10, 10), "bad": None}, {"stock": (10, 10), "unstable": (8, 10)}])
def test_selection_defaults_to_stock_without_a_qualified_winner(blocks):
    assert selector()["choose_candidate"](selection_records(blocks))["winner"] == "stock"


def test_one_gpu_numeric_failure_excludes_candidate_on_all_gpus():
    records = selection_records({"stock": (10, 10), "fast": (8, 8)})
    records[3]["results"]["fast"] = {"numeric_passed": False, "blocks": [[], []]}
    assert selector()["choose_candidate"](records)["winner"] == "stock"
    records[3]["results"]["stock"] = {"numeric_passed": False, "blocks": [[], []]}
    with pytest.raises(AssertionError, match="stock numerical"):
        selector()["choose_candidate"](records)


def test_selection_applies_pooled_rule_and_reports_rank_local_noise():
    records = selection_records({"stock": (10, 10), "fast": (8, 8)})
    records[3]["results"]["fast"]["blocks"] = [[11] * 5, [11] * 5]
    result = selector()["choose_candidate"](records)
    assert result["winner"] == "fast"
    assert not result["summary"]["fast"]["all_ranks_win"]


@pytest.mark.parametrize("op1,op2", [(0, 0), (1, 0), (0, 1), (1, 2), (2, 1)])
def test_selection_writes_mixed_worker_environment_and_clears_stock(op1, op2):
    ns = load_dense()
    choices = {op: [ns["dense_candidates"](op)[i] for i in (0, 1, 9)] for op in ("qkv", "o")}
    selected = {op: {"winner": choices[op][i]["name"]} for op, i in (("qkv", op1), ("o", op2))}
    env = selector()["worker_environment"](selected, choices)
    assert set(env) == {ENV, "SGLANG_DECODE_DENSE_GEMM_QKV", "SGLANG_DECODE_DENSE_GEMM_O"}
    direct = []
    for op, i in (("qkv", op1), ("o", op2)):
        c = choices[op][i]
        assert env["SGLANG_DECODE_DENSE_GEMM_" + op.upper()] == (c["name"] if i == 2 else "")
        if i == 1:
            direct.append(op.upper() + ":" + ",".join(map(str, c["config"])))
    assert env[ENV] == ";".join(direct)


def test_round_robin_alternates_stock_and_reverses_the_second_block():
    schedule = selector()["round_robin"]
    assert schedule(["stock", "a", "b"], 0) == ["stock", "a", "stock", "b"]
    assert schedule(["stock", "a", "b"], 1) == ["stock", "b", "stock", "a"]
    assert schedule(["stock"], 0) == ["stock"]


def test_selector_candidate_mode_restores_all_worker_envs_on_failure(monkeypatch):
    ns = load_dense()
    previous = {ENV: "O:128,32,10,1,0", "SGLANG_DECODE_DENSE_GEMM_QKV": "old-q", "SGLANG_DECODE_DENSE_GEMM_O": "old-o"}
    for key, value in previous.items():
        monkeypatch.setenv(key, value)
    mode = selector()["candidate_mode"]
    with pytest.raises(ValueError), mode(ns, "qkv", ns["dense_candidates"]("qkv")[9]):
        assert ENV not in os.environ
        assert os.environ["SGLANG_DECODE_DENSE_GEMM_QKV"].startswith("triton-")
        assert "SGLANG_DECODE_DENSE_GEMM_O" not in os.environ
        raise ValueError("rejected")
    assert {key: os.environ[key] for key in previous} == previous


def overlay_source():
    patch = (ROOT / "patches/sglang/decode-dense-gemm.patch").read_text()
    added = patch.split("+++ b/python/sglang/srt/layers/quantization/decode_dense_gemm.py\n", 1)[1]
    return "".join(line[1:] for line in added.splitlines(True) if line.startswith("+"))


def load_overlay(monkeypatch):
    import sys

    torch = types.SimpleNamespace(float8_e4m3fn="fp8", float32="fp32", bfloat16="bf16", cuda=types.SimpleNamespace(get_device_capability=lambda _: (9, 0)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    ns = {}
    exec(compile(overlay_source(), "decode_dense_gemm.py", "exec"), ns)  # noqa: S102 - tracked overlay module
    return ns


def test_overlay_unset_and_prefill_return_original_linear_without_inspection(monkeypatch):
    ns = load_overlay(monkeypatch)
    monkeypatch.delenv("SGLANG_DECODE_DENSE_GEMM_QKV", raising=False)
    x = object()
    calls = []
    layer = lambda actual: calls.append(actual) or (actual, None)
    assert ns["decode_dense_linear"](None, "QKV", layer, x, None, decode_worker=True) == (x, None)
    monkeypatch.setenv("SGLANG_DECODE_DENSE_GEMM_QKV", "triton-64-128-128-s3-w4")
    assert ns["decode_dense_linear"](None, "QKV", layer, x, None, decode_worker=False) == (x, None)
    assert calls == [x, x]


def test_overlay_decode_extend_keeps_original_path(monkeypatch):
    ns = load_overlay(monkeypatch)
    monkeypatch.setenv("SGLANG_DECODE_DENSE_GEMM_O", "triton-64-128-128-s3-w4")
    batch = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda: False, is_target_verify=lambda: False))
    x = object()
    assert ns["decode_dense_linear"](None, "O", lambda actual: actual, x, batch, decode_worker=True) is x


def test_overlay_applies_to_pinned_image_qwen_source_and_has_only_two_call_sites(tmp_path):
    import subprocess
    import tarfile

    archive = ROOT.parent / "codex-g2-work/image39711569005-source.tar.gz"
    if not archive.exists():
        pytest.skip("pinned image source is a local investigation artifact")
    relative = "python/sglang/srt/models/qwen3_moe.py"
    with tarfile.open(archive) as tar:
        source = tar.extractfile("opt/sglang/source/" + relative).read()
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(source)
    patch = str(ROOT / "patches/sglang/decode-dense-gemm.patch")
    for args in (["git", "apply", "--check", patch], ["git", "apply", patch]):
        subprocess.run(args, cwd=tmp_path, check=True, capture_output=True)
    tree = ast.parse(target.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "decode_dense_linear"]
    assert len(calls) == 2
    assert {ast.literal_eval(c.args[1]) for c in calls} == {"QKV", "O"}
    ast.parse((tmp_path / "python/sglang/srt/layers/quantization/decode_dense_gemm.py").read_text())


def test_serving_preflight_checks_only_selected_and_rejects_ambiguous_env(monkeypatch):
    ns = load_dense()
    monkeypatch.setenv(ENV, "QKV:128,80,7,1,0")
    monkeypatch.setenv("SGLANG_DECODE_DENSE_GEMM_O", "triton-64-128-128-s3-w4")
    monkeypatch.delenv("SGLANG_DECODE_DENSE_GEMM_QKV", raising=False)
    assert ns["selected_dense_candidates"]("qkv")[0]["config"] == (128, 80, 7, 1, 0)
    assert ns["selected_dense_candidates"]("o")[0]["name"] == "triton-64-128-128-s3-w4"
    monkeypatch.setenv("SGLANG_DECODE_DENSE_GEMM_QKV", "triton-64-128-128-s3-w4")
    with pytest.raises(AssertionError, match="only one"):
        ns["selected_dense_candidates"]("qkv")
    monkeypatch.delenv(ENV)
    assert ns["selected_dense_candidates"]("qkv")[0]["backend"] == "triton"
    monkeypatch.setenv("SGLANG_DECODE_DENSE_GEMM_QKV", "")
    assert ns["selected_dense_candidates"]("qkv")[0]["name"] == "stock"


class OverlayTensor:
    def __init__(self, shape, dtype):
        self.shape, self.dtype, self.device = shape, dtype, "cuda:0"
        self.ndim = len(shape)

    def is_contiguous(self):
        return True

    def stride(self, dim):
        return self.shape[1] if dim == 0 else 1


def overlay_fixture(op="QKV", rows=128):
    attention = types.SimpleNamespace(num_heads=64, total_num_heads=64, num_kv_heads=4, total_num_kv_heads=4, hidden_size=4096, head_dim=128)
    n, k = (9216, 4096) if op == "QKV" else (4096, 8192)
    calls = []

    class Layer:
        weight = OverlayTensor((n, k), "fp8")
        weight_scale_inv = OverlayTensor((n // 128, k // 128), "fp32")
        quant_method = types.SimpleNamespace(block_quant=True, use_mxfp8=False, use_marlin=False, weight_block_size=[128, 128])
        bias = None

        def __call__(self, x):
            calls.append(x)
            return "stock-result"

    batch = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda: False, is_target_verify=lambda: True))
    x = OverlayTensor((rows, k), "bf16")
    return attention, Layer(), batch, x, calls


@pytest.mark.parametrize("op", ["QKV", "O"])
def test_overlay_selected_target_path_quantizes_once_or_preserves_tuple(monkeypatch, op):
    import sys

    ns = load_overlay(monkeypatch)
    monkeypatch.setenv("SGLANG_DECODE_DENSE_GEMM_" + op, "triton-64-128-128-s3-w4")
    attention, layer, batch, x, fallback = overlay_fixture(op)
    a = OverlayTensor(x.shape, "fp8")
    sa = OverlayTensor((128, x.shape[1] // 128), "fp32")
    quant, runs = [], []

    def quantize(value, group, **kwargs):
        quant.append((value, group, kwargs))
        return a, sa

    monkeypatch.setitem(sys.modules, "sglang.kernels.ops.quantization.fp8_kernel", types.SimpleNamespace(sglang_per_token_group_quant_fp8=quantize))
    ns["torch"].empty = lambda shape, **kwargs: OverlayTensor(shape, kwargs["dtype"])
    ns["run_decode_dense_gemm"] = lambda *args: runs.append(args)
    result = ns["decode_dense_linear"](attention, op, layer, x, batch, decode_worker=True)
    assert result[1] is None and result[0].dtype == "bf16"
    assert quant == [(x, 128, {"column_major_scales": True, "scale_tma_aligned": True})]
    assert runs[0][:4] == (op, "triton-64-128-128-s3-w4", (a, sa), (layer.weight, layer.weight_scale_inv))
    ns["decode_dense_linear"](attention, op, layer, (a, sa), batch, decode_worker=True)
    assert len(quant) == 1 and len(runs) == 2 and not fallback


@pytest.mark.parametrize("case", ["bf16-draft", "large-m", "wrong-heads", "bias", "other-scale"])
def test_overlay_target_guards_keep_other_linear_paths(monkeypatch, case):
    ns = load_overlay(monkeypatch)
    monkeypatch.setenv("SGLANG_DECODE_DENSE_GEMM_QKV", "triton-64-128-128-s3-w4")
    attention, layer, batch, x, fallback = overlay_fixture()
    if case == "bf16-draft":
        layer.weight = OverlayTensor(layer.weight.shape, "bf16")
    elif case == "large-m":
        x = OverlayTensor((132, 4096), "bf16")
    elif case == "wrong-heads":
        attention.num_heads = 32
    elif case == "bias":
        layer.bias = object()
    else:
        layer.weight_scale_inv = OverlayTensor(layer.weight_scale_inv.shape, "ue8m0")
    assert ns["decode_dense_linear"](attention, "QKV", layer, x, batch, decode_worker=True) == "stock-result"
    assert fallback == [x]


@pytest.mark.parametrize("error_type", [AssertionError, RuntimeError, ValueError])
@pytest.mark.parametrize("boundaries", [(), (4, 64, 65)])
def test_selector_prepares_same_four_weights_rejects_bad_candidate_and_continues(monkeypatch, boundaries, error_type):
    ns, checks = selector(), load_dense()
    made, validated, captured, comparisons = [], [], [], []
    candidates = [{"name": "stock", "backend": "deepgemm", "config": None}, {"name": "bad", "backend": "triton", "config": None}, {"name": "fast", "backend": "triton", "config": None}]

    def inputs(*_):
        result = (object(), object(), Output())
        made.append(result)
        return result

    def call(candidate, values):
        return lambda: comparisons.append((candidate["name"], values))

    def validate(op, candidate, values, *_):
        assert values in made
        validated.append(candidate["name"])
        if candidate["name"] == "bad":
            raise error_type("candidate mismatch")
        return {"changed_input_graph_passed": True}

    def capture(_, calls):
        assert validated[-1] != "bad" and len(calls) == 4
        for c in calls:
            c()
        assert len({id(s[0]) for _, s in comparisons[-4:]}) == 4
        captured.append(validated[-1])
        return object()

    checks.update(dense_inputs=inputs, dense_reference=lambda *_: object(), dense_call=call, dense_validate_candidate=validate, dense_errors=lambda *_: {})
    monkeypatch.setitem(ns["prepare_projection"].__globals__, "capture_graph", capture)
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None))
    result = ns["prepare_projection"](torch, checks, "qkv", 9216, 4096, candidates, boundary_rows=boundaries)
    assert len(made) == 4 + len(boundaries)
    assert validated == ["stock"] * (1 + len(boundaries)) + ["bad"] + ["fast"] * (1 + len(boundaries))
    assert captured == ["stock", "fast"] and set(result["graphs"]) == {"stock", "fast"}
    assert not result["results"]["bad"]["numeric_passed"]
    assert "candidate mismatch" in result["results"]["bad"]["error"]
    assert result["results"]["fast"]["numeric_passed"]


def test_legacy_kernel_bench_has_no_dense_search_left():
    source = ast.parse((ROOT / "scripts/bench-firmus-sglang-kernels.py").read_text())
    assert not any(isinstance(n, ast.FunctionDef) and n.name == "bench_dense_gemm" for n in source.body)
    assert not list((ROOT / "patches/sglang-bench").glob("*decode-dense*"))


def test_formal_overlay_registered_once_in_serving_config():
    config = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    assert config["sglang_overlay_patches"].count("patches/sglang/decode-dense-gemm.patch") == 1


@pytest.mark.parametrize("argv,selected", [(["check.py", "--dense-selected"], True), (["check-firmus-megamoe.py", "--dense-selected"], True), (["-c", "/opt/sglang/source/python", "/run/check-firmus-decode-kernels.py"], False)])
def test_numeric_gate_retains_run_path_caller_arguments(monkeypatch, argv, selected):
    source = ast.parse((ROOT / "scripts/check-firmus-decode-kernels.py").read_text())
    entry = next(n for n in source.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name) and n.test.left.id == "__name__")
    devices, calls = [], []
    monkeypatch.setenv("LOCAL_RANK", "2")
    ns = {"__name__": "__main__", "os": os, "sys": types.SimpleNamespace(argv=argv), "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(set_device=devices.append))}
    for node in ast.walk(entry):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.startswith("check_"):
            ns[node.func.id] = lambda *args, **kwargs: None
    ns["check_dense_gemm"] = lambda **kwargs: calls.append(kwargs)
    exec(compile(ast.Module(body=[entry], type_ignores=[]), "numeric-entry", "exec"), ns)  # noqa: S102 - tracked entrypoint with CPU mocks
    assert calls == [{"selected_only": selected}] and devices == [2]
