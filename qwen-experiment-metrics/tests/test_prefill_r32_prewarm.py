"""Pinned image source and actual process-local preparation, without a GPU."""
import ast
import gzip
import hashlib
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def overlaid(tmp_path):
    fixture = json.loads(gzip.decompress((ROOT / "qwen-experiment-metrics/tests/fixtures/prefill-r32-image-source.json.gz").read_bytes()))
    config = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    assert fixture["image_sha256"] == config["images"]["sglang"]["sha256"]
    for name, source in fixture["files"].items():
        assert hashlib.sha256(source.encode()).hexdigest() == fixture["sha256"][name]
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(source)
    for patch in config["sglang_overlay_patches"]:
        subprocess.run(["git", "apply", "--check", str(ROOT / patch)], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "apply", str(ROOT / patch)], cwd=tmp_path, check=True, capture_output=True)
    for p in tmp_path.rglob("*.py"):
        ast.parse(p.read_text())
    return tmp_path


@pytest.fixture
def prepared(overlaid, monkeypatch):
    np = pytest.importorskip("numpy")
    source = (overlaid / "python/sglang/srt/layers/moe/mega_moe_sm90.py").read_text()
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("prewarm_fc1_first_k", "_r32_dispatch")]
    calls = []
    args = SimpleNamespace(disaggregation_mode="prefill")

    class Array(np.ndarray):
        def to(self, dtype):
            return self.astype(dtype)

    class Tensor:
        def __init__(self, shape, dtype):
            self.shape, self.dtype, self.device = shape, dtype, "cuda:0"

    class MoE:
        config = SimpleNamespace(hidden_size=4096, moe_intermediate_size=1536, num_experts=128, num_experts_per_tok=8)
        experts = SimpleNamespace(mega_l1_weights=(Tensor((128, 3072, 4096), "fp8"), object()), mega_l2_weights=(Tensor((128, 4096, 1536), "fp8"), object()))

    moe = MoE()
    buf = SimpleNamespace(x=object(), x_sf=object(), topk_idx=object(), topk_weights=object())

    def buffer(*pos, **kw):
        calls.append(("buffer", pos, kw))
        return buf

    def pred(*pos, **kw):
        calls.append(("predispatch", pos, kw))

    def mega(*pos, **kw):
        calls.append(("megamoe", pos, kw))

    def sync(device):
        calls.append(("sync", device))

    torch = SimpleNamespace(inference_mode=nullcontext, bfloat16="bf16", float32="fp32", int32=np.int32, zeros=lambda shape, **kw: Tensor(shape, kw["dtype"]), full=lambda shape, value, **kw: Tensor(shape, kw["dtype"]), empty_like=lambda x: Tensor(x.shape, x.dtype), arange=lambda n, **kw: np.arange(n).view(Array), cuda=SimpleNamespace(synchronize=sync))
    modules = {
        "deep_gemm": SimpleNamespace(mega_moe_pre_dispatch_sm90=pred, fp8_mega_moe=mega),
        "sglang.srt.runtime_context": SimpleNamespace(get_server_args=lambda: args),
        "sglang.srt.distributed": SimpleNamespace(get_moe_ep_group=lambda: SimpleNamespace(world_size=1, device_group="EP1")),
        "sglang.srt.layers.moe.mega_moe": SimpleNamespace(_get_mega_moe_symm_buffer=buffer),
        "sglang.srt.models.qwen3_moe": SimpleNamespace(Qwen3MoeSparseMoeBlock=MoE),
    }
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    for key in ("DG_SM90_PREFILL_FC1_FIRST_K", "DG_SM90_PREFILL_2WG", "DG_SM90_PREFILL_4STAGE", "DG_SM90_PREFILL_LEAN", "DG_SM90_PREFILL_SPLIT_ACCUM"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("DG_SM90_PREFILL_TAIL_PAIR", "0")
    ns = {"os": os, "torch": torch, "envs": SimpleNamespace(SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=SimpleNamespace(get=lambda: 8192)), "is_sm90_fp8_mega_moe_available": lambda experts: True, "_r32_prewarmed": False, "_r32_dispatch_logged": False, "__name__": "r32-test"}
    exec(compile(tree, "r32-actual-helper", "exec"), ns)  # noqa: S102 - execute only the vendored, hash-checked source
    return SimpleNamespace(ns=ns, calls=calls, args=args, moe=moe, model=SimpleNamespace(modules=lambda: [moe]), torch=torch, modules=modules)


def test_real_weights_shared_buffer_once_and_separate_dispatch(prepared, caplog):
    p = prepared
    p.ns["prewarm_fc1_first_k"](p.model)
    p.ns["prewarm_fc1_first_k"](p.model)
    assert [call[0] for call in p.calls] == ["buffer", "predispatch", "megamoe", "sync"]
    _, _, kwargs = p.calls[0]
    assert kwargs == {"num_experts": 128, "num_max_tokens_per_rank": 8192, "num_topk": 8, "hidden": 4096, "intermediate_hidden": 1536}
    _, pos, kwargs = p.calls[1]
    assert pos[0].shape == (7988, 4096) and pos[0].dtype == "bf16"
    assert pos[1].shape == (7988, 8) and pos[1].dtype.name == "int32"
    assert kwargs == {"num_tokens": 7988, "group_size": 128, "routed_scaling_factor": 1.0}
    _, pos, kwargs = p.calls[2]
    assert pos[1] is p.moe.experts.mega_l1_weights and pos[2] is p.moe.experts.mega_l2_weights
    assert kwargs == {"recipe": (128, 128, 128), "activation": "swiglu", "activation_clamp": None, "fast_math": True}
    assert caplog.text.count("prewarm done") == 1 and "using candidate" not in caplog.text
    p.ns["_r32_dispatch"](p.moe, 7988)
    p.ns["_r32_dispatch"](p.moe, 7988)
    assert caplog.text.count("using candidate") == 1


@pytest.mark.parametrize("mode", ("flag-off", "decode", "sync-failure", "missing-layer"))
def test_no_false_preparation_completion(prepared, monkeypatch, caplog, mode):
    p = prepared
    if mode == "flag-off":
        monkeypatch.setenv("DG_SM90_PREFILL_FC1_FIRST_K", "0")
    elif mode == "decode":
        p.args.disaggregation_mode = "decode"
    elif mode == "sync-failure":
        p.torch.cuda.synchronize = lambda _: (_ for _ in ()).throw(RuntimeError("sync failed"))
    else:
        p.model.modules = list
    if mode in ("sync-failure", "missing-layer"):
        with pytest.raises(RuntimeError):
            p.ns["prewarm_fc1_first_k"](p.model)
    else:
        p.ns["prewarm_fc1_first_k"](p.model)
        assert not p.calls
    assert not p.ns["_r32_prewarmed"] and "prewarm done" not in caplog.text


def test_actual_initialization_order_and_launch_log(overlaid):
    runner = ast.parse((overlaid / "python/sglang/srt/model_executor/model_runner.py").read_text())
    initialize = next(n for n in ast.walk(runner) if isinstance(n, ast.FunctionDef) and n.name == "initialize")
    calls = [(ast.unparse(n.func), n.lineno) for n in ast.walk(initialize) if isinstance(n, ast.Call)]
    line = dict(calls)
    assert line["self.maybe_apply_post_load_model_transforms"] < line["self.configure_kv_cache_dtype"] < line["prewarm_fc1_first_k"]
    check = next(n for n in ast.walk(initialize) if isinstance(n, ast.If) and "prewarm_fc1_first_k" in ast.unparse(n))
    assert "not self.is_draft_worker" in ast.unparse(check.test) and "prefill" in ast.unparse(check.test)
    source = (overlaid / "python/sglang/srt/layers/moe/mega_moe_sm90.py").read_text()
    forward = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "run_sm90_mega_routed")
    calls = {ast.unparse(n.func): n.lineno for n in ast.walk(forward) if isinstance(n, ast.Call)}
    assert calls["deep_gemm.fp8_mega_moe"] < calls["_r32_dispatch"]
