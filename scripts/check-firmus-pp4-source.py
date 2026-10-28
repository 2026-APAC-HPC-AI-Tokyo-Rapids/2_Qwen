"""CPU-only PP4 source checks; pass a fully patched SGLang source tree. No GPU accuracy claim."""

import ast
import copy
import os
import runpy
import sys
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

repo = Path(__file__).resolve().parents[1]
source = Path(sys.argv[1]).resolve() / "python/sglang/srt"
candidate = source


def method(relative, cls, name, env):
    path = candidate / relative
    if not path.exists():
        path = source / relative
    tree = ast.parse(path.read_text())
    parent = (
        next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        if cls
        else tree
    )
    fn = copy.deepcopy(
        next(
            n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
    )
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
        arg.annotation = None
    code = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(code, str(path), "exec"), env)  # noqa: S102 - execute pinned source methods with CPU fixtures
    return env[name]


os.environ["SGLANG_QWEN_PP4_EP1"] = "1"
helper = runpy.run_path(str(candidate / "speculative/qwen_pp4.py"))
args = NS(
    disaggregation_mode="prefill",
    pp_size=4,
    tp_size=1,
    dp_size=1,
    ep_size=1,
    speculative_algorithm="EAGLE3",
    speculative_eagle_topk=1,
    speculative_num_draft_tokens=4,
    speculative_num_steps=3,
    disable_overlap_schedule=True,
    moe_a2a_backend="megamoe",
    ep_num_redundant_experts=0,
    enable_eplb=False,
    enable_two_batch_overlap=False,
    enable_lora=False,
    speculative_use_rejection_sampling=False,
    speculative_adaptive=False,
    disaggregation_transfer_backend="nixl",
)
assert helper["supported"](args)
for key, value in {
    "disaggregation_mode": "decode",
    "pp_size": 2,
    "ep_size": 4,
    "tp_size": 4,
    "dp_size": 4,
    "disable_overlap_schedule": False,
    "enable_eplb": True,
    "speculative_algorithm": "EAGLE",
    "enable_two_batch_overlap": True,
    "enable_lora": True,
    "speculative_use_rejection_sampling": True,
    "speculative_adaptive": True,
    "disaggregation_transfer_backend": "mooncake",
}.items():
    other = copy.copy(args)
    setattr(other, key, value)
    assert not helper["supported"](other), key
os.environ["SGLANG_QWEN_PP4_EP1"] = "0"
assert not helper["supported"](args)
os.environ["SGLANG_QWEN_PP4_EP1"] = "1"
print("PASS: opt-in PP4 prefill guard and 14 unsupported variants")

pairs = method(
    "disaggregation/utils.py", None, "build_transfer_entry_pairs", {"deque": deque}
)


def pool(start, end):
    return NS(
        start_layer=start,
        end_layer=end,
        layer_num=end - start,
        get_contiguous_buf_infos=lambda: (list(range(2 * (end - start))), [], []),
    )


ids = helper["kv_region_ids"]
dest = ids(pool(0, 94), 94, pool(0, 1))
written = []
for start, end in [(0, 24), (24, 48), (48, 71), (71, 94)]:
    src = ids(pool(start, end), 94, pool(0, 1) if end == 94 else None)
    result = pairs(src, dest, len(src), len(dest))
    for i, j in result:
        assert src[i] == dest[j]
        group = 0 if i < end - start else 1
        if i < 2 * (end - start):
            assert j == src[i] + 94 * group, (start, end, i, j)
    written.extend(j for _, j in result)
assert sorted(written) == list(range(190))
for dst in ([], dest[:-1]):
    try:
        pairs(src, dst, len(src), 190)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Missing layer metadata accepted")
print("PASS: four PP shards cover all 94 target K/V and one draft K/V exactly once")


class Proxy:
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, key):
        return self.tensors[key]


class Layer:
    def __init__(self, index):
        self.index = index
        self._is_layer_to_capture = index in (2, 47, 91)

    def __call__(self, positions, hidden, batch, residual, captured_last_layer_outputs):
        if captured_last_layer_outputs is not None:
            captured_last_layer_outputs.append(
                hidden.copy() if residual is None else hidden + residual
            )
        return hidden + (self.index + 1), np.full_like(hidden, self.index / 8)


env = {
    "PPProxyTensors": Proxy,
    "is_prefill_context_parallel_enabled": lambda: False,
    "is_cp_v2_active": lambda batch: False,
    "nullcontext": nullcontext,
    "check_cuda_graph_backend": lambda *a: True,
    "Phase": NS(PREFILL=0),
    "Backend": NS(TC_PIECEWISE=0),
}
forward = method("models/qwen2_moe.py", "Qwen2MoeModel", "forward", env)
layers = [Layer(i) for i in range(94)]


def model(start, end):
    return NS(
        pp_group=NS(is_first_rank=start == 0, is_last_rank=end == 94),
        layers=layers,
        start_layer=start,
        end_layer=end,
        embed_tokens=lambda x: x.copy(),
        norm=lambda hidden, residual: (hidden + residual, None),
    )


batch = NS(can_run_tbo=False)
x = np.arange(12, dtype=np.float32).reshape(3, 4)
reference, aux = forward(model(0, 94), x, None, batch)
proxy = None
for start, end in [(0, 24), (24, 48), (48, 71), (71, 94)]:
    result = forward(model(start, end), x, None, batch, pp_proxy_tensors=proxy)
    if end < 94:
        proxy = result
actual, actual_aux = result
np.testing.assert_array_equal(actual, reference)
assert len(actual_aux) == len(aux) == 3
for a, b in zip(actual_aux, aux):
    np.testing.assert_array_equal(a, b)
print(
    "PASS: patched model forward preserves output and all EAGLE captures across four stages"
)

env = {
    "CaptureHiddenMode": NS(NULL="null", FULL="full"),
    "speculative_moe_backend_context": nullcontext,
    "speculative_moe_a2a_backend_context": nullcontext,
    "spec_stage_span": lambda *a: nullcontext(),
}
worker_forward = method(
    "speculative/eagle_worker_v2.py", "EAGLEWorkerV2", "forward_batch_generation", env
)
for last in (False, True):
    calls = []
    payload = NS(
        logits_output=NS(hidden_states=actual_aux, mm_input_embeds=None),
        next_token_ids=[7],
    )

    def target(batch, calls=calls, payload=payload, **kw):
        assert kw == {"capture_hidden_mode": "full", "pp_proxy_tensors": proxy}
        calls.append("target")
        return payload

    def draft(*args, calls=calls):
        calls.append("draft")
        return "draft-result"

    worker = NS(
        speculative_algorithm=NS(is_standalone=lambda: False),
        target_worker=NS(
            pp_group=NS(is_last_rank=last), forward_batch_generation=target
        ),
        draft_worker=NS(
            draft_tp_context=lambda *a: nullcontext(),
            draft_runner=NS(tp_group=None),
            _draft_extend_for_prefill=draft,
        ),
    )
    batch = NS(forward_mode=NS(is_extend=lambda: True), seq_lens=[4001])
    out = worker_forward(worker, batch, pp_proxy_tensors=proxy)
    assert out is payload and calls == (["target", "draft"] if last else ["target"])
print("PASS: only final PP target stage executes EAGLE draft prefill")

config = __import__("json").loads(
    (repo / "configs/firmus-prefill-pp4.json").read_text()
)
import tempfile

_runtime = tempfile.TemporaryDirectory(prefix="pp4-settings-check-")
runtime = Path(_runtime.name)
for sub in ("source", "results"):
    (runtime / sub).mkdir(parents=True, exist_ok=True)
runpy.run_path(str(repo / "scripts/prepare-firmus-pp4-settings.py"))["prepare"](
    config, repo, runtime
)
print("PASS: prior kernel settings and original source hashes remain reproducible")
_runtime.cleanup()
