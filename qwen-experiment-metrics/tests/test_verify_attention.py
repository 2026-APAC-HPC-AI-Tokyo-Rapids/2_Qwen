"""CPU contract checks for verify experiments; CUDA numerics run in the gate."""
import ast
import math
import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


class Tensor:
    def __init__(self, shape, values=None, start=0):
        self.shape, self.values, self.start = shape, values, start
        self.dtype = "fp8"

    def numel(self):
        return math.prod(self.shape)

    def __getitem__(self, part):
        lo, hi, _ = part.indices(self.shape[0]) if isinstance(part, slice) else (part, part + 1, 1)
        if isinstance(part, int):
            return self.values[part]
        return Tensor((hi - lo, *self.shape[1:]), None if self.values is None else self.values[part], self.start + lo)

    def __sub__(self, value):
        return Tensor(self.shape, [n - value for n in self.values], self.start)


def load_candidates(tmp_path, monkeypatch):
    subprocess.run(["git", "apply", str(ROOT / "patches/sglang-bench/04-verify-attention-candidates.patch")], cwd=tmp_path, check=True)
    calls, prepares = [], []

    def flash(**kwargs):
        calls.append(kwargs)
        return kwargs["out"]

    def prepare(**kwargs):
        prepares.append(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bf16", empty_like=lambda q, **kw: Tensor(q.shape)))
    monkeypatch.setitem(sys.modules, "sgl_kernel", types.ModuleType("sgl_kernel"))
    monkeypatch.setitem(sys.modules, "sgl_kernel.flash_attn", types.SimpleNamespace(flash_attn_with_kvcache=flash, get_scheduler_metadata=prepare))
    module = runpy.run_path(str(tmp_path / "python/sglang/kernels/ops/attention/verify_attention_candidates.py"))
    return module, calls, prepares


@pytest.mark.parametrize("batch", range(32, 37))
def test_fa3_partition_keeps_all_tokens_heads_causal_offsets_and_output_slices(tmp_path, monkeypatch, batch):
    module, calls, prepares = load_candidates(tmp_path, monkeypatch)
    common = {"q": Tensor((batch * 4, 64, 128)), "cache_seqlens": Tensor((batch,)), "page_table": Tensor((batch, 66)), "cu_seqlens_q": Tensor((batch + 1,), list(range(0, batch * 4 + 1, 4))), "cu_seqlens_k_new": Tensor((batch + 1,), [i * 4117 for i in range(batch + 1)]), "max_seqlen_q": 4, "causal": True}
    cases = module["fa3_cases"](common)
    assert len(cases) == 16
    assert prepares[0]["max_seqlen_k"] == 4224
    assert (prepares[0]["max_seqlen_q"], prepares[0]["num_heads"], prepares[0]["num_heads_k"], prepares[0]["num_splits"]) == (4, 64, 4, 1)
    for name, call in cases.items():
        calls.clear()
        call()
        if "batch-cut" in name:
            assert sum(c["q"].shape[0] for c in calls) == batch * 4
            assert [c["q"].start for c in calls] == [c["out"].start for c in calls]
            assert calls[0]["q"].start == 0
            assert calls[-1]["q"].start + calls[-1]["q"].shape[0] == batch * 4
            for c in calls:
                n = c["cache_seqlens"].numel()
                assert c["cu_seqlens_q"].values == list(range(0, 4 * n + 1, 4))
                assert c["cu_seqlens_k_new"].values == [i * 4117 for i in range(n + 1)]
                assert c["q"].shape[1:] == (64, 128) and c["causal"] is True
        else:
            assert len(calls) == 1 and calls[0]["max_seqlen_q"] >= 4


def test_serving_gate_keeps_shared_metadata_check_and_exploration_stays_in_bench():
    for filename in ("bench-firmus-sglang-kernels.py", "check-firmus-decode-kernels.py"):
        source = (ROOT / "scripts" / filename).read_text()
        ast.parse(source)
        assert 'overlay/sglang/python/sglang/kernels/ops/attention/verify_attention_candidates.py' in source
    gate = (ROOT / "scripts/check-firmus-decode-kernels.py").read_text()
    assert 'check_verify_scheduler_sharing()' in gate
    assert 'check_verify_attention_candidates' not in gate
    assert 'alternative_cases' not in gate and 'fa3_cases' not in gate
    assert 'variant=variant' not in gate
    bench = (ROOT / "scripts/bench-firmus-sglang-kernels.py").read_text()
    assert 'ops["alternative_cases"](common)' in bench
    assert 'ops["fa3_cases"](common)' in bench


@pytest.fixture
def backend_methods(tmp_path):
    # Execute the complete added production method and capture reset hook,
    # without importing SGLang/CUDA or depending on an external source checkout.
    patch = ROOT / "patches/sglang/decode-verify-scheduler-metadata.patch"
    text = patch.read_text()
    method = text.split('+    def _get_verify_scheduler_metadata(', 1)[1].split('     def _mxfp8_sf_kwargs', 1)[0]
    method = '    def _get_verify_scheduler_metadata(' + '\n'.join(line[1:] for line in method.splitlines() if line.startswith('+'))
    hook = text.split('     def init_forward_metadata_in_graph', 1)[1].split('         # Single-CG', 1)[0]
    hook = '    def init_forward_metadata_in_graph' + '\n'.join(line.removeprefix('+') for line in hook.splitlines())
    source = 'class FlashAttentionBackend:\n' + method + '\n' + hook
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FlashAttentionBackend")
    functions = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"_get_verify_scheduler_metadata", "init_forward_metadata_in_graph"}]
    for func in functions:
        func.returns = None
        for arg in func.args.args:
            arg.annotation = None
    namespace = {"torch": types.SimpleNamespace(float8_e4m3fn="fp8")}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(patch), "exec"), namespace)  # noqa: S102 - execute only repository-owned production AST
    return namespace


def verify_fixture(methods, batch=32):
    calls = []

    def prepare(**kwargs):
        calls.append(kwargs)
        return object()

    mode = types.SimpleNamespace(is_target_verify=lambda: True)
    fb = types.SimpleNamespace(forward_mode=mode, spec_info=None)
    layer = types.SimpleNamespace(tp_q_head_num=64, tp_k_head_num=4, tp_v_head_num=4, head_dim=128, v_head_dim=128, logit_cap=0, is_cross_attention=False)
    metadata = types.SimpleNamespace(cache_seqlens_int32=Tensor((batch,)), cu_seqlens_q=Tensor((batch + 1,)), max_seq_len_q=4, page_table=Tensor((batch, 6144)), verify_scheduler_metadata=None)
    backend = types.SimpleNamespace(forward_metadata=metadata, fa_impl_ver=3, _get_scheduler_metadata=prepare, use_mla=False, attn_cp_size=1, topk=1, page_size=64)
    q, kc = Tensor((batch * 4, 64, 128)), Tensor((1, 64, 4, 128))
    args = [backend, q, kc, layer, fb, True, (-1, -1), False, False]
    return args, calls


@pytest.mark.parametrize("batch", range(32, 37))
def test_prepare_once_across_94_layers_and_recapture_after_warmup(backend_methods, batch):
    args, calls = verify_fixture(backend_methods, batch)
    prepare = backend_methods["_get_verify_scheduler_metadata"]
    hook = backend_methods["init_forward_metadata_in_graph"]
    for _ in range(3):  # warmup, capture, recapture
        hook(args[0], args[4])
        pointers = [prepare(*args) for _ in range(94)]
        assert len({id(p) for p in pointers}) == 1
    assert len(calls) == 3
    for call in calls:
        assert call["max_seqlen_k"] == 393216  # native page-table bound
        assert (call["num_heads"], call["num_heads_k"], call["max_seqlen_q"], call["num_splits"]) == (64, 4, 4, 1)
    # A new eager forward owns new metadata; no cross-step reuse.
    args[0].forward_metadata = verify_fixture(backend_methods, batch)[0][0].forward_metadata
    prepare(*args)
    assert len(calls) == 4


@pytest.mark.parametrize("field,value", [("fa_impl_ver", 4), ("use_mla", True), ("attn_cp_size", 2), ("topk", 2), ("page_size", 1)])
def test_other_backends_shapes_and_cp_keep_original_path(backend_methods, field, value):
    args, calls = verify_fixture(backend_methods)
    setattr(args[0], field, value)
    assert backend_methods["_get_verify_scheduler_metadata"](*args) is None
    assert calls == []


@pytest.mark.parametrize("batch", [0, 1, 31, 37])
def test_unmeasured_batches_keep_original_split_heuristic(backend_methods, batch):
    args, calls = verify_fixture(backend_methods, batch)
    assert backend_methods["_get_verify_scheduler_metadata"](*args) is None
    assert calls == []


@pytest.mark.parametrize("field,value", [("tp_q_head_num", 16), ("tp_k_head_num", 1), ("tp_v_head_num", 1), ("head_dim", 64), ("v_head_dim", 64), ("logit_cap", 30), ("is_cross_attention", True)])
def test_incompatible_layer_does_not_reuse_earlier_metadata(backend_methods, field, value):
    args, calls = verify_fixture(backend_methods)
    method = backend_methods["_get_verify_scheduler_metadata"]
    assert method(*args) is not None
    setattr(args[3], field, value)
    assert method(*args) is None
    assert len(calls) == 1


@pytest.mark.parametrize("slot,value", [(5, False), (6, (127, 0)), (7, True), (8, True)])
def test_mask_and_cascade_paths_keep_original_metadata(backend_methods, slot, value):
    args, calls = verify_fixture(backend_methods)
    args[slot] = value
    assert backend_methods["_get_verify_scheduler_metadata"](*args) is None
    assert calls == []


def test_step_timer_includes_one_prepare_and_all_94_native_calls(tmp_path, monkeypatch):
    module, calls, prepares = load_candidates(tmp_path, monkeypatch)
    common = {"q": Tensor((128, 64, 128)), "cache_seqlens": Tensor((32,)), "page_table": Tensor((32, 66)), "cu_seqlens_q": Tensor((33,))}
    cases = module["step_cases"](common)
    cases["fa3-verify-step94-default"]()
    assert len(calls) == 94 and not prepares
    assert all("scheduler_metadata" not in call for call in calls)
    calls.clear()
    cases["fa3-verify-step94-shared"]()
    assert len(calls) == 94 and len(prepares) == 1
    assert len({id(call["scheduler_metadata"]) for call in calls}) == 1
    assert all(call["num_splits"] == 1 for call in calls)


def test_page_planning_preserves_partial_pages_and_four_causal_decode_lengths(tmp_path, monkeypatch):
    module, _, _ = load_candidates(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules["torch"].__dict__, "int32", "int32")
    monkeypatch.setitem(sys.modules["torch"].__dict__, "tensor", lambda data, **kwargs: data)
    monkeypatch.setitem(sys.modules["torch"].__dict__, "cat", lambda arrays: [item for array in arrays for item in array])

    class Table:
        def __getitem__(self, index):
            row, part = index
            return list(range(100 * row, 100 * row + 3))[part]

    lengths = [4, 63, 64, 65, 95, 96, 97, 128]
    ptr, ids, last = module["paged_indices"](Table(), lengths)
    assert ptr == [0, 1, 2, 3, 5, 7, 9, 11, 13]
    assert last == [4, 63, 64, 1, 31, 32, 33, 64]
    assert len(ids) == ptr[-1]
    # Every query i sees exactly L-3+i cache entries, including the query's
    # own new token. No zero-length or masked cache entry is substituted.
    for token in range(4):
        ptr, _, last = module["paged_indices"](Table(), [n - 3 + token for n in lengths])
        assert [(ptr[i + 1] - ptr[i] - 1) * 64 + last[i] for i in range(8)] == [n - 3 + token for n in lengths]
