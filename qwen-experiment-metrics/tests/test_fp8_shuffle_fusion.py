"""CPU contract checks; no torch/CUDA/source-mirror mutation."""
import ast
import importlib.util
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).parents[2]


def module():
    spec = importlib.util.spec_from_file_location("shuffle_probe", REPOSITORY / "scripts/benchmark-prefill-fp8-row-shuffle.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_pair_index_contract_preserves_bits_and_duplicates():
    q = [bytes(range(256)) * 16, bytes(reversed(range(256))) * 16]
    # Include signed-zero and NaN payload bytes without FP arithmetic.
    s = [bytes.fromhex("00000000000000800100c07f0100c0ff") * 8, bytes(range(128))]
    mapping = [1, 0, 1, 1, 0]
    oq, os = module().cpu_contract(q, s, mapping)
    assert oq == [q[1], q[0], q[1], q[1], q[0]]
    assert os == [s[1], s[0], s[1], s[1], s[0]]


@pytest.mark.parametrize("mapping", [[-1], [2]])
def test_invalid_native_map_is_not_a_padding_semantics(mapping):
    with pytest.raises(AssertionError):
        module().cpu_contract([b"a", b"b"], [b"c", b"d"], mapping)


def test_empty_contract():
    assert module().cpu_contract([], [], []) == ([], [])


def test_patch_is_fixed_bitcopy_with_prefill_hopper_ep1_guards():
    patch = (REPOSITORY / "patches/sglang/prefill-fp8-scale-row-shuffle.patch").read_text()
    for contract in ['_prefill_fused_shuffle', 'disaggregation_mode == "prefill"', 'moe_ep_size == 1', 'k == 4096', 'a1_scale.shape == (m, 32)', 'and is_sm90_supported()', 'q.view(torch.uint8)', 'scales.view(torch.int32)', 'num_warps=8']:
        assert contract in patch
    assert 'other=' not in patch and 'tl.maximum' not in patch
    assert 'SGLANG_PREFILL_FUSED_FP8_SHUFFLE", "0"' in patch
    ast.parse((REPOSITORY / "scripts/benchmark-prefill-fp8-row-shuffle.py").read_text())


@pytest.mark.parametrize("change", ["default_off", "decode", "other_gpu", "ep2", "K", "scale_shape", "map_shape", "q_strided", "scale_strided", "map_strided", "q_dtype", "scale_dtype", "map_dtype"])
def test_actual_patched_guard_excludes_other_contracts(change):
    from types import SimpleNamespace

    patch = (REPOSITORY / "patches/sglang/prefill-fp8-scale-row-shuffle.patch").read_text()
    added = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    expression = "(" + added.split("if (", 1)[1].split("):", 1)[0] + ")"
    q = SimpleNamespace(dtype="fp8", is_contiguous=lambda: True)
    scales = SimpleNamespace(dtype="float32", shape=(32, 32), is_contiguous=lambda: True)
    mapping = SimpleNamespace(dtype="int32", shape=(256,), is_contiguous=lambda: True)
    server = SimpleNamespace(disaggregation_mode="prefill")
    parallel = SimpleNamespace(moe_ep_size=1)
    scope = {"_prefill_fused_shuffle": True, "is_sm90_supported": lambda: True, "get_server_args": lambda: server, "get_parallel": lambda: parallel, "k": 4096, "m": 32, "topk": 8, "a_q": q, "a1_scale": scales, "a_map": mapping, "torch": SimpleNamespace(float8_e4m3fn="fp8", float32="float32", int32="int32")}
    code = compile(ast.parse(expression, mode="eval"), "actual-patched-guard", "eval")
    assert eval(code, {"__builtins__": {}}, scope)
    if change == "default_off":
        scope["_prefill_fused_shuffle"] = False
    elif change == "decode":
        server.disaggregation_mode = "decode"
    elif change == "other_gpu":
        scope["is_sm90_supported"] = lambda: False
    elif change == "ep2":
        parallel.moe_ep_size = 2
    elif change == "K":
        scope["k"] = 2048
    elif change == "scale_shape":
        scales.shape = (32, 16)
    elif change == "map_shape":
        mapping.shape = (32, 8)
    elif change.endswith("_strided"):
        {"q": q, "scale": scales, "map": mapping}[change.split("_")[0]].is_contiguous = lambda: False
    else:
        {"q": q, "scale": scales, "map": mapping}[change.split("_")[0]].dtype = "other"
    assert not eval(code, {"__builtins__": {}}, scope)
