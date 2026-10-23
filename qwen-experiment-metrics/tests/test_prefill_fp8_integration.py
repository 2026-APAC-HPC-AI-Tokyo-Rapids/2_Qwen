"""CPU contract checks for the installed Prefill helper; GPU gates run in Slurm."""
import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def helper(tmp_path):
    rel = "python/sglang/srt/layers/attention/prefill_fp8.py"
    subprocess.run(["git", "apply", "--include=" + rel, str(ROOT / "patches/sglang/prefill-fp8-integration.patch")], cwd=tmp_path, check=True)
    tree = ast.parse((tmp_path / rel).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("linear_supported", "linear")]
    for node in functions:
        node.body = [statement for statement in node.body if not isinstance(statement, (ast.Import, ast.ImportFrom))]
    class Method:
        block_quant = True
        use_marlin = use_mxfp8 = False
    backend = object()
    method = Method()
    method.weight_block_size = [128, 128]
    method.w8a8_block_fp8_linear = backend
    weight = SimpleNamespace(shape=(4096, 8192), dtype="fp8", device="cuda", is_contiguous=lambda: True)
    layer = SimpleNamespace(quant_method=method, weight=weight, weight_scale_inv=SimpleNamespace(dtype="fp32"), bias=None)
    attention = SimpleNamespace(num_heads=64, total_num_heads=64, num_kv_heads=4, total_num_kv_heads=4, hidden_size=4096, head_dim=128)
    calls = []
    ns = {
        "torch": SimpleNamespace(float8_e4m3fn="fp8", float32="fp32", bfloat16="bf16", cuda=SimpleNamespace(get_device_capability=lambda _: (9, 0)), empty=lambda shape, **kw: (shape, kw)),
        "Fp8LinearMethod": Method,
        "deep_gemm_wrapper": SimpleNamespace(DEEPGEMM_SCALE_UE8M0=False),
        "deepgemm_w8a8_block_fp8_linear_with_fallback": backend,
        "deep_gemm": SimpleNamespace(fp8_gemm_nt=lambda *args, **kw: calls.append((args, kw))),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(tmp_path / rel), "exec"), ns)  # noqa: S102
    return ns, attention, layer, calls


@pytest.mark.parametrize("unsupported", [None, "tp", "bias", "backend", "scales", "weight"])
def test_dense_guard_and_prequantized_consumer(helper, unsupported):
    ns, attention, layer, calls = helper
    if unsupported == "tp":
        attention.num_heads = 16
    elif unsupported == "bias":
        layer.bias = object()
    elif unsupported == "backend":
        layer.quant_method.w8a8_block_fp8_linear = object()
    elif unsupported == "scales":
        ns["deep_gemm_wrapper"].DEEPGEMM_SCALE_UE8M0 = True
    elif unsupported == "weight":
        layer.weight.shape = (1024, 8192)
    assert ns["linear_supported"](attention, "O", layer) == (unsupported is None)
    if unsupported is None:
        activation = SimpleNamespace(ndim=2, shape=(4096, 8192), device="cuda")
        scale = object()
        result, bias = ns["linear"](attention, "O", layer, (activation, scale))
        assert result[0] == (4096, 4096) and bias is None
        assert len(calls) == 1 and calls[0][0][0] == (activation, scale)
        assert calls[0][1] == {"compiled_dims": "nk"}


def test_small_batches_retain_the_original_linear(helper):
    ns, attention, _, calls = helper
    activation = SimpleNamespace(ndim=2, shape=(128, 8192))
    original = []
    pair = (activation, object())
    assert ns["linear"](attention, "O", lambda x: original.append(x) or "original", pair) == "original"
    assert original == [pair] and not calls


def test_dispatch_uses_pinned_runtime_accessors(monkeypatch):
    """Keep real accessor imports; mock only the GPU operations for this CPU check."""
    source = os.environ.get("QWEN_SGLANG_SOURCE")
    if not source:
        pytest.skip("requires the patched, pinned SGLang source")
    root = Path(source)

    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, root / relative)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    runtime = load("sglang.srt.runtime_context", "srt/runtime_context.py")
    context = load("sglang.srt.model_executor.forward_context", "srt/model_executor/forward_context.py")
    args = SimpleNamespace(enable_lora=True, disaggregation_mode="prefill", cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(backend="disabled")))
    # Publish only the slot consumed here; full config setup runs in the GPU probe.
    runtime.get_context()._server_args = args
    dense_tree = ast.parse((root / "srt/layers/quantization/decode_dense_gemm.py").read_text())
    dense = next(n for n in dense_tree.body if isinstance(n, ast.FunctionDef) and n.name == "decode_dense_linear")
    ns = {"os": os}
    exec(compile(ast.Module(body=[dense], type_ignores=[]), "installed-dense-dispatch", "exec"), ns)  # noqa: S102
    for name in ("QKV", "O"):
        monkeypatch.delenv("SGLANG_DECODE_DENSE_GEMM_" + name, raising=False)
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True))
    token = object()
    for decode in (True, False):
        assert ns["decode_dense_linear"](None, "QKV", lambda x: x, token, batch, decode_worker=decode) is token

    tree = ast.parse((root / "srt/models/qwen3_moe.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen3MoeAttention")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_core")
    calls = []
    args.enable_lora = False
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.attention.prefill_fp8", SimpleNamespace(linear_supported=lambda *_: True))
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.quantization.decode_dense_gemm", SimpleNamespace(decode_dense_linear=lambda *a, **kw: (a[3], None)))
    ns = {"get_server_args": runtime.get_server_args, "get_attn_backend": context.get_attn_backend}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), "installed-qwen-forward-core", "exec"), ns)  # noqa: S102
    model = SimpleNamespace(_used_fused_qk_norm_rope_last_call=False, _stored_fp8_kv_last_call=True, o_proj=None, attn=lambda *a, **kw: calls.append(kw) or token)
    with context.forward_context(context.ForwardContext(SimpleNamespace(fa_impl_ver=3))):
        assert ns["forward_core"](model, (None, batch, (token, token, token, batch))) is token
        assert calls[-1]["prefill_fp8_output"] is True
        args.cuda_graph_config.prefill.backend = "breakable"
        ns["forward_core"](model, (None, batch, (token, token, token, batch)))
        assert "prefill_fp8_output" not in calls[-1]
