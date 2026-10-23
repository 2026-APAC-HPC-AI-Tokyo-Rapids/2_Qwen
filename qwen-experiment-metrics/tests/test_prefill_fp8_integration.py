"""CPU contract checks for the installed Prefill helper; GPU gates run in Slurm."""
import ast
import subprocess
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
