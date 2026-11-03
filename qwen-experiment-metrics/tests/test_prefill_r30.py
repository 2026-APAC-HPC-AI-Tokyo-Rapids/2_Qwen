"""Pinned-image overlay order and process-local eager preparation contracts."""
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
PREFIX = "python/sglang/"


@pytest.fixture
def overlaid(tmp_path):
    # Vendored image source, not git show: CI uses a shallow checkout.
    with gzip.open(ROOT / "qwen-experiment-metrics/tests/fixtures/prefill-r30-image-source.json.gz", "rt") as f:
        fixture = json.load(f)
    config = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    assert fixture["image_sha256"] == config["images"]["sglang"]["sha256"]
    assert fixture["sglang_commit"] == "fc275674a97d78553b22597ee72c8fd48f5420f4"
    for name, source in fixture["files"].items():
        assert hashlib.sha256(source.encode()).hexdigest() == fixture["sha256"][name]
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    for patch in config["sglang_overlay_patches"]:
        subprocess.run(["git", "apply", "--check", str(ROOT / patch)], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "apply", str(ROOT / patch)], cwd=tmp_path, check=True, capture_output=True)
    for path in tmp_path.rglob("*.py"):
        ast.parse(path.read_text())
    return tmp_path


def function(root, relative, name):
    tree = ast.parse((root / PREFIX / relative).read_text())
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


@pytest.fixture
def prewarm(overlaid, monkeypatch):
    tree = ast.parse((overlaid / PREFIX / "srt/layers/attention/prefill_fp8.py").read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("prewarm_qkv_tail", "_qkv_tail_gemm", "dense_gemm", "_log_qkv_tail_dispatch")]
    calls, events, allocations = [], [], []

    class Tensor:
        def __init__(self, shape, dtype, stride=None, offset=0):
            self.shape, self.dtype, self.device = shape, dtype, "cuda:0"
            self.stride = stride or (shape[1], 1)
            self.offset = offset

        @property
        def T(self):
            return Tensor(self.shape[::-1], self.dtype, self.stride[::-1])

        def __getitem__(self, key):
            start, stop, _ = key.indices(self.shape[0])
            return Tensor((stop - start, self.shape[1]), self.dtype, self.stride, self.offset + start * self.stride[0])

    def allocate(kind, shape, *, dtype, device):
        assert device == "cuda:0"
        tensor = Tensor(shape, dtype)
        allocations.append((kind, tensor))
        return tensor

    def gemm(*args, **kwargs):
        events.append("gemm")
        calls.append((args, kwargs))

    class Attention:
        def __init__(self, supported=True):
            self.supported = supported
            self.qkv_proj = SimpleNamespace(weight=Tensor((9216, 4096), "fp8"), weight_scale_inv=Tensor((72, 32), "fp32"))

    args = SimpleNamespace(disaggregation_mode="prefill")
    monkeypatch.setenv("SGLANG_PREFILL_QKV_TAIL_SPLIT", "1")
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", SimpleNamespace(get_server_args=lambda: args))
    monkeypatch.setitem(sys.modules, "sglang.srt.models.qwen3_moe", SimpleNamespace(Qwen3MoeAttention=Attention))
    monkeypatch.setitem(sys.modules, "deep_gemm", SimpleNamespace(fp8_gemm_nt=gemm))
    ns = {"os": os, "_qkv_tail_prewarmed": False, "_qkv_tail_dispatch_logged": False,
          "linear_supported": lambda a, p, layer: a.supported and p == "QKV",
          "torch": SimpleNamespace(inference_mode=nullcontext, float8_e4m3fn="fp8", float32="fp32", bfloat16="bf16",
                                   zeros=lambda shape, **kw: allocate("zeros", shape, **kw), ones=lambda shape, **kw: allocate("ones", shape, **kw), empty=lambda shape, **kw: allocate("empty", shape, **kw),
                                   cuda=SimpleNamespace(synchronize=lambda device: events.append("sync:" + device)))}
    exec(compile(tree, "<actual r30 prewarm>", "exec"), ns)  # noqa: S102
    attention = Attention()
    model = SimpleNamespace(modules=lambda: iter((object(), Attention(False), attention, Attention())))
    return SimpleNamespace(ns=ns, model=model, attention=attention, args=args, calls=calls, events=events, allocations=allocations)


def test_prewarm_and_serving_share_exact_descriptors_once(prewarm, caplog):
    p = prewarm
    for _ in range(2):
        p.ns["prewarm_qkv_tail"](p.model)
    assert p.events == ["gemm", "gemm", "sync:cuda:0"]
    assert [kind for kind, _ in p.allocations] == ["zeros", "ones", "empty"]
    assert not p.ns["_qkv_tail_dispatch_logged"]
    logs = [r.getMessage() for r in caplog.records]
    assert len(logs) == 1 and logs[0].startswith("Prefill r30 qkv-tail prewarm done ms=") and f"pid={os.getpid()}" in logs[0]
    a, scales, out = (tensor for _, tensor in p.allocations)
    sa = scales.T
    p.ns["dense_gemm"](a, sa, p.attention.qkv_proj.weight, p.attention.qkv_proj.weight_scale_inv, out, tail_split=True)
    p.ns["dense_gemm"](a, sa, p.attention.qkv_proj.weight, p.attention.qkv_proj.weight_scale_inv, out, tail_split=True)
    assert len(p.calls) == 6
    for index, (args, kwargs) in enumerate(p.calls):
        (av, sv), (w, ws), ov = args
        body = index % 2 == 0
        assert kwargs == {"compiled_dims": "nk"}
        assert av.shape == ((7936 if body else 52), 4096) and av.dtype == "fp8" and av.stride == (4096, 1)
        assert sv.shape == ((7936 if body else 52), 32) and sv.dtype == "fp32" and sv.stride == (1, 7988)
        assert av.offset == (0 if body else 7936 * 4096) and sv.offset == (0 if body else 7936)
        assert ov.shape == ((7936 if body else 52), 9216) and ov.dtype == "bf16" and ov.stride == (9216, 1)
        assert ov.offset == (0 if body else 7936 * 9216)
        assert w is p.attention.qkv_proj.weight and ws is p.attention.qkv_proj.weight_scale_inv
    logs = [r.getMessage() for r in caplog.records]
    assert len(logs) == 2 and "Prefill r30 qkv-tail using split" in logs[1]


@pytest.mark.parametrize("mode,flag", (("decode", "1"), ("null", "1"), ("prefill", "0"), ("prefill", "")))
def test_prewarm_excludes_decode_and_disabled_flag(prewarm, monkeypatch, caplog, mode, flag):
    prewarm.args.disaggregation_mode = mode
    monkeypatch.setenv("SGLANG_PREFILL_QKV_TAIL_SPLIT", flag)
    prewarm.ns["prewarm_qkv_tail"](prewarm.model)
    assert not prewarm.calls and not prewarm.allocations and not caplog.records


@pytest.mark.parametrize("failure", ("no-layer", "launch", "sync"))
def test_failed_preparation_cannot_claim_completion(prewarm, monkeypatch, caplog, failure):
    def fail(*args, **kwargs):
        raise RuntimeError("test failure")
    if failure == "no-layer":
        prewarm.model.modules = lambda: iter(())
    elif failure == "launch":
        monkeypatch.setitem(sys.modules, "deep_gemm", SimpleNamespace(fp8_gemm_nt=fail))
    else:
        prewarm.ns["torch"].cuda.synchronize = fail
    with pytest.raises(RuntimeError):
        prewarm.ns["prewarm_qkv_tail"](prewarm.model)
    assert not prewarm.ns["_qkv_tail_prewarmed"] and not prewarm.ns["_qkv_tail_dispatch_logged"]
    assert not caplog.records


@pytest.mark.parametrize("mode,draft,flag,prepared", (("prefill", False, "1", True), ("decode", False, "1", False), ("prefill", True, "1", False), ("prefill", False, "0", False)))
def test_real_startup_hook_follows_transforms_and_precedes_pools_graphs(overlaid, monkeypatch, mode, draft, flag, prepared):
    events = []
    model = object()
    monkeypatch.setenv("SGLANG_PREFILL_QKV_TAIL_SPLIT", flag)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.attention.prefill_fp8", SimpleNamespace(prewarm_qkv_tail=lambda m: events.append("prewarm") if m is model else pytest.fail("wrong model")))
    node = function(overlaid, "srt/model_executor/model_runner.py", "initialize")
    ns = {"os": os, "create_sampler": lambda: None, "ExpertLocationUpdater": lambda: None,
          "prepare_moe_topk": lambda **kw: None, "resolve_layer_indices": lambda **kw: SimpleNamespace(start_layer=0, end_layer=24),
          "adjust_hybrid_swa_layer_ids": lambda **kw: None, "disable_routed_experts_capture_for_draft": lambda m: None}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<actual initialize>", "exec"), ns)  # noqa: S102

    class Runner:
        is_draft_worker, is_hybrid_swa = draft, False
        server_args = SimpleNamespace(disaggregation_mode=mode)
        model_config = spec_algorithm = None
        ps = SimpleNamespace(moe_ep_size=1, moe_ep_rank=0)
        remote_instance_weight_transporter = SimpleNamespace(maybe_register_and_publish_weight_info=lambda: None)

        def __getattr__(self, name):
            return lambda: events.append(name)

    runner = Runner()
    runner.model = model
    scheduler_init = function(overlaid, "srt/managers/scheduler.py", "init_model_worker")
    scheduler = SimpleNamespace(init_tp_model_worker=lambda: ns["initialize"](runner), maybe_init_draft_worker=lambda: None,
                                init_memory_pools=lambda: events.append("pools"), init_all_attention_backends=lambda: events.append("backends"), init_all_cuda_graphs=lambda: events.append("graphs"))
    # Limit to the weight/pool/backend/graph startup segment; later scheduler wiring
    # is unrelated to the hook and may require distributed runtime objects.
    scheduler_init.body = scheduler_init.body[:5]
    exec(compile(ast.Module(body=[scheduler_init], type_ignores=[]), "<actual scheduler startup segment>", "exec"), ns)  # noqa: S102
    ns["init_model_worker"](scheduler)
    assert events.index("load_model") < events.index("maybe_apply_post_load_model_transforms") < events.index("configure_kv_cache_dtype") < events.index("pools") < events.index("graphs")
    assert ("prewarm" in events) is prepared
    if prepared:
        assert events.index("configure_kv_cache_dtype") < events.index("prewarm") < events.index("pools")
    # The hook's os name must be a module import, not the constructor's local
    # exception-path import (which is absent in a successful startup).
    tree = ast.parse((overlaid / PREFIX / "srt/model_executor/model_runner.py").read_text())
    assert any(isinstance(n, ast.Import) and any(a.name == "os" for a in n.names) for n in tree.body)


def test_quantizers_match_prewarm_scale_layout(overlaid):
    source = (overlaid / PREFIX / "kernels/ops/layernorm/add_rmsnorm_group_quant.py").read_text()
    assert '(rows + 3) // 4 * 4' in source and 'transpose(0, 1)[:rows]' in source
    # 7988 is already TMA-aligned; slicing at 7936 preserves column stride.
    assert (7988 + 3) // 4 * 4 == 7988
    quant = (overlaid / PREFIX / "srt/layers/attention/prefill_fp8.py").read_text()
    assert 'column_major_scales=True, scale_tma_aligned=True' in quant
