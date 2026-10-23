"""CPU integration checks against an explicitly extracted, patched image tree.

Set QWEN_SGLANG_SOURCE to its python/sglang directory. GPU numerics/replay live
in check-firmus-decode-kernels.py and are deliberately not claimed here.
"""
from __future__ import annotations

import ast
import os
import random
import struct
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def source():
    path = os.environ.get("QWEN_SGLANG_SOURCE")
    if not path:
        pytest.skip("requires extracted Firmus image + configured overlays + Decode patch")
    return Path(path)


def function(path, name, namespace):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)  # noqa: S102 - execute the actual patched function without CUDA imports
    return namespace[name]


def flag(value):
    return types.SimpleNamespace(get=lambda: value)


@pytest.mark.parametrize("rows", [1, 128, 129, 132, 144, 145, 2048])
@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("remap", [False, True])
def test_select_experts_padding_fusion_eligibility(source, rows, jit, remap):
    tree = ast.parse((source / "srt/layers/moe/topk.py").read_text())
    select = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select_experts")
    # Exercise the actual selection condition and mutation together: a rejected
    # case must retain its device count for the ordinary post-process mask.
    branch = next(n for n in ast.walk(select) if isinstance(n, ast.If) and any(isinstance(a, ast.Assign) and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == "_fused_topk_kwargs" for t in a.targets) for a in n.body))
    count = types.SimpleNamespace(dtype="int32", numel=lambda: 1)
    ns = {"_is_cuda": True, "_use_aiter": False, "get_moe_a2a_backend": lambda: types.SimpleNamespace(is_megamoe=lambda: True), "router_logits": types.SimpleNamespace(shape=(rows, 128)), "top_k": 8, "scoring_func": "softmax", "correction_bias": None, "expert_location_dispatch_info": object() if remap else None, "num_fused_shared_experts": 0, "num_token_non_padded": count, "torch": types.SimpleNamespace(int32="int32"), "envs": types.SimpleNamespace(SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=flag(jit), SGLANG_SIMULATE_UNIFORM_EXPERTS=flag(False), SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS=flag(False)), "_fused_topk_kwargs": {}}
    branch.orelse = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[])), "eligibility", "exec"), ns)  # noqa: S102
    enabled = not remap and rows <= (144 if jit else 128)
    assert ns["_fused_topk_kwargs"] == ({"num_token_non_padded": count} if enabled else {})
    assert ns["num_token_non_padded"] is (None if enabled else count)


@pytest.mark.parametrize("fused", [False, True])
def test_router_wrapper_preserves_launch_geometry_and_runtime_count(source, fused):
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, args, kwargs))
            return launch

    class Tensor:
        def __init__(self, shape, dtype="bf16", device="cuda"):
            self.shape, self.dtype, self.device = shape, dtype, device
            self.ndim = len(shape)

        def size(self, index):
            return self.shape[index]

        def stride(self, index):
            return self.shape[1] if index == 0 else 1

        def numel(self):
            return 1

    scores = Tensor((132, 128))
    count = Tensor((1,), "int32") if fused else None
    bias = None if fused else Tensor((128,), "fp32")
    torch = types.SimpleNamespace(float32="fp32", float16="fp16", bfloat16="bf16", int32="int32", empty=lambda shape, **kw: Tensor(shape, **kw))
    ns = {"torch": torch, "triton": types.SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length(), cdiv=lambda a, b: (a + b - 1) // b), "_SCORING_FUNC_MAP": {"softmax": 2}, "_router_triton_kernel": Kernel(), "is_arch_support_pdl": lambda: True}
    call = function(source / "kernels/ops/moe/moe_fused_gate.py", "moe_fused_gate", ns)
    weights, ids = call(scores, bias, 8, scoring_func="softmax", num_token_non_padded=count)
    assert weights.shape == ids.shape == (132, 8)
    grid, args, options = launches[0]
    assert grid == (66,) and args[0] is scores and args[1] is bias
    assert options["num_token_non_padded"] is count
    assert options["FUSE_DECODE_PADDING"] is fused
    assert options["SCORING_FUNC"] == 2 and options["RENORMALIZE"] is True
    assert options["BLOCK_M"] == 2 and options["num_warps"] == 1
    assert options["USE_PDL"] and options["launch_pdl"]
    if fused:
        with pytest.raises(AssertionError):
            call(scores, Tensor((128,), "fp32"), 8, scoring_func="softmax", num_token_non_padded=count)
        with pytest.raises(AssertionError):
            call(Tensor((145, 128)), None, 8, scoring_func="softmax", num_token_non_padded=count)
        with pytest.raises(AssertionError):
            call(scores, None, 8, scoring_func="sigmoid", num_token_non_padded=count)


def test_eight_lane_head_reduction_retains_original_fp32_tree():
    def fp32(value):
        return struct.unpack("f", struct.pack("f", value))[0]

    gen = random.Random(61022)
    for _ in range(1000):
        # These are the original 32 lanes' positive four-element partial sums.
        partial = [fp32(10 ** gen.uniform(-8, 8)) for _ in range(32)]
        old = partial[:]
        for mask in (16, 8, 4, 2, 1):
            old = [fp32(old[lane] + old[lane ^ mask]) for lane in range(32)]
        new = [[partial[4 * lane + j] for j in range(4)] for lane in range(8)]
        for mask in (4, 2, 1):
            new = [[fp32(new[lane][j] + new[lane ^ mask][j]) for j in range(4)] for lane in range(8)]
        for lane in range(8):
            result = fp32(fp32(new[lane][0] + new[lane][2]) + fp32(new[lane][1] + new[lane][3]))
            assert all(result == old[4 * lane + j] for j in range(4))


@pytest.mark.parametrize("rows", [1, 128, 144])
@pytest.mark.parametrize("padding", [False, True])
def test_fused_topk_passes_runtime_count_without_zero_bias_launch(source, monkeypatch, rows, padding):
    import sys

    calls = []
    zeros = []
    module = types.ModuleType("sglang.kernels.ops.moe.moe_fused_gate")

    def gate(*args, **kw):
        calls.append((args, kw))
        return "weights", "ids"

    module.moe_fused_gate = gate
    monkeypatch.setitem(sys.modules, module.__name__, module)
    torch = types.SimpleNamespace(float32="fp32", int32="int32", empty=lambda *a, **kw: None, zeros=lambda *a, **kw: zeros.append(a) or "bias")
    envs = types.SimpleNamespace(SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=flag(True))
    ns = {"torch": torch, "envs": envs, "_is_cuda": True, "_use_aiter": False}
    call = function(source / "srt/layers/moe/topk.py", "fused_topk", ns)
    hidden = types.SimpleNamespace(shape=(rows, 4096), device="cuda")
    logits = types.SimpleNamespace(shape=(rows, 128), device="cuda")
    count = object() if padding else None
    assert call(hidden, logits, 8, True, num_token_non_padded=count) == ("weights", "ids")
    args, options = calls[0]
    assert args[0] is logits and args[1] == (None if padding else "bias")
    assert options["num_token_non_padded"] is count
    assert options["scoring_func"] == "softmax" and options["renormalize"]
    assert len(zeros) == (0 if padding else 1)
