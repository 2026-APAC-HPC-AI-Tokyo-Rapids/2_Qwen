"""CPU regressions for the Firmus Tier 2 SGLang patches in a patched source tree.

decode-async-admission: every device-tensor construction on the Decode admission
path has a non-blocking branch under SGLANG_DECODE_ASYNC_ADMISSION, and deferred
page-index sends keep allocation order and skip aborted requests.
dp-total-requests-rotate-ties: equal loads rotate, the minimum load still wins.
megamoe-idle-prefill-geometry: an idle Prefill rank gets enough masked rows for
DeepGEMM's Prefill geometry; Decode and busy ranks are unchanged.
Run each check only on a tree that carries the corresponding patch.
"""
from __future__ import annotations

import argparse
import ast
import sys
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace

ADMISSION_SITES = {
    "python/sglang/srt/disaggregation/decode.py": ["alloc_for_decode_prealloc"],
    "python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py": ["prepare_for_prebuilt", "process_prebuilt"],
    "python/sglang/srt/sampling/penaltylib/min_new_tokens.py": ["_prepare"],
    "python/sglang/srt/speculative/eagle_disaggregation.py": ["build_eagle_disagg_draft_input"],
}


def function(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def gated_off(node, parents):
    """True when node sits in the disabled branch of an async_admission test."""
    child = node
    for parent in parents:
        if isinstance(parent, (ast.If, ast.IfExp)) and "ENABLED" in ast.unparse(parent.test):
            body = parent.orelse if isinstance(parent, ast.If) else [parent.orelse]
            if any(child is b or any(child is n for n in ast.walk(b)) for b in body):
                return True
        child = parent
    return False


def check_admission_sites(source: Path):
    for rel, names in ADMISSION_SITES.items():
        tree = ast.parse((source / rel).read_text())
        for name in names:
            fn = function(tree, name)
            parents = {}
            for parent in ast.walk(fn):
                for child in ast.iter_child_nodes(parent):
                    parents[child] = parent
            for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
                text = ast.unparse(call.func)
                blocking = text in ("torch.tensor", "torch.as_tensor") and any(k.arg == "device" for k in call.keywords)
                blocking |= text.endswith(".to") and len(call.args) == 1 and not call.keywords and "device" in ast.unparse(call.args[0])
                if not blocking:
                    continue
                chain, node = [], call
                while node in parents:
                    node = parents[node]
                    chain.append(node)
                # Radix-prefix last_loc is a device-to-device slice; DSA top-k indices
                # exist only for DSA models (None for Qwen3) and also need an .item().
                if "prefix_indices" in ast.unparse(call) or "dsa" in ast.unparse(call):
                    continue
                assert gated_off(call, chain), f"{rel}:{call.lineno} {name}: blocking device copy without a non-blocking branch"


def check_deferred_metadata(source: Path):
    path = source / "python/sglang/srt/disaggregation/decode.py"
    fn = function(ast.parse(path.read_text()), "flush_deferred_metadata")
    tree = ast.Module(body=[fn], type_ignores=[])
    sent = []

    class KVPoll(Enum):
        Failed = auto()
        WaitingForInput = auto()

    class Receiver:
        def __init__(self, name, state=KVPoll.WaitingForInput):
            self.name, self.state = name, state

        def poll(self):
            return self.state

        def send_metadata(self, pages, aux, state, decode_prefix_len):
            sent.append((self.name, list(pages), aux, decode_prefix_len))

    class Host:
        def __init__(self, values):
            self.values = values

        def numpy(self):
            return SimpleNamespace(astype=lambda _: self.values)

    namespace = {"KVPoll": KVPoll, "np": SimpleNamespace(int32="int32")}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)  # noqa: S102
    flush = namespace["flush_deferred_metadata"]
    done = SimpleNamespace(query=lambda: True)
    pending = SimpleNamespace(query=lambda: False)

    def entry(name, event, receiver="ok"):
        rx = None if receiver is None else Receiver(name, KVPoll.Failed if receiver == "failed" else KVPoll.WaitingForInput)
        return (SimpleNamespace(kv_receiver=rx, metadata_buffer_index=len(name)), Host([len(name)]), event, [], 0)

    queue = SimpleNamespace(_deferred_metadata=[entry("a", done), entry("bb", done, "failed"), entry("ccc", done, None), entry("dddd", pending), entry("eeeee", done)])
    flush(queue)
    assert [s[0] for s in sent] == ["a"], sent
    assert [e[0].metadata_buffer_index for e in queue._deferred_metadata] == [4, 5], "an unfinished readback must hold back later sends"
    queue._deferred_metadata[0] = (queue._deferred_metadata[0][0], queue._deferred_metadata[0][1], done, [], 0)
    flush(queue)
    assert [s[0] for s in sent] == ["a", "dddd", "eeeee"] and not queue._deferred_metadata

    defer = function(ast.parse(path.read_text()), "_defer_send_metadata")
    test = ast.unparse(defer.body[1].test)
    for guard in ("async_admission.ENABLED", "kv_indices.is_cuda", "is_rebootstrap", "enable_hisparse", "enable_staging"):
        assert guard in test, f"deferred send must fall back for {guard}"
    loop = function(ast.parse(path.read_text()), "process_decode_queue")
    assert "flush_deferred_metadata" in ast.unparse(loop.body[0]), "deferred sends must be flushed every iteration"


def check_dp_ties(source: Path):
    path = source / "python/sglang/srt/managers/data_parallel_controller.py"
    module = ast.parse(path.read_text())
    body = [n for n in module.body if isinstance(n, ast.ClassDef) and n.name in ("LoadBalanceMethod", "DPBudget")]
    tree = ast.Module(body=body, type_ignores=[])
    for rotate in (True, False):
        namespace = {"Enum": Enum, "auto": auto, "_ROTATE_TOTAL_REQUESTS_TIES": rotate}
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)  # noqa: S102
        budget = namespace["DPBudget"](4)
        method = namespace["LoadBalanceMethod"].TOTAL_REQUESTS
        picks = []
        for _ in range(8):
            budget.total_requests = [0, 0, 0, 0]
            picks.append(budget.dispatch(method))
        assert picks == ([0, 1, 2, 3] * 2 if rotate else [0] * 8), picks
        budget.total_requests = [2, 1, 3, 1]
        assert budget.dispatch(method) in (1, 3) and budget.total_requests.count(2) == 2, "the least-loaded rank must still win"


def check_idle_geometry(source: Path):
    path = source / "python/sglang/srt/layers/moe/mega_moe_sm90.py"
    module = ast.parse(path.read_text())
    wanted = {"idle_output_rows", "_use_idle_prefill_geometry", "run_sm90_mega_routed"}
    body = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *body], type_ignores=[])

    def heuristic(num_tokens, num_ranks=4, num_topk=8, num_experts=128):
        # DeepGEMM 0.1.5 get_block_config_for_mega_moe_sm90 / should_use_swap_ab.
        if num_tokens * num_ranks * num_topk / num_experts >= 64.0:
            return "prefill-128x256"
        return "swap-ab-64x128" if num_tokens <= 128 else "decode-64"

    shapes, calls = [], []

    class Rows:
        def __init__(self, shape):
            self.shape = shape

        def size(self, i):
            return self.shape[i]

        def __getitem__(self, item):
            return self

    def empty(shape, dtype, device):
        shapes.append(shape)
        return Rows(shape)

    server = SimpleNamespace(disaggregation_mode="prefill")
    fakes = {
        "deep_gemm": SimpleNamespace(mega_moe_pre_dispatch_sm90=lambda *a, **k: calls.append(("pre", k["num_tokens"])), fp8_mega_moe=lambda y, *a, **k: calls.append(("moe", heuristic(y.size(0))))),
        "sglang.srt.runtime_context": SimpleNamespace(get_server_args=lambda: server),
        "sglang.srt.distributed": SimpleNamespace(get_moe_ep_group=lambda: SimpleNamespace(world_size=4)),
    }
    saved = {name: sys.modules.get(name) for name in fakes}
    sys.modules.update(fakes)
    try:
        for mode, enabled, tokens, expect in (("prefill", True, 0, 256), ("prefill", False, 0, 1), ("decode", True, 0, 1), ("prefill", True, 4058, 4058), ("prefill", True, 5, 5)):
            shapes.clear()
            calls.clear()
            server.disaggregation_mode = mode
            namespace = {"torch": SimpleNamespace(bfloat16="bf16", empty=empty), "_IDLE_PREFILL_GEOMETRY": enabled}
            exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)  # noqa: S102
            moe = SimpleNamespace(experts=SimpleNamespace(should_fuse_routed_scaling_factor_in_topk=True, mega_l1_weights=None, mega_l2_weights=None), config=SimpleNamespace(hidden_size=4096, num_experts=128, num_experts_per_tok=8))
            buf = SimpleNamespace(x=None, x_sf=None, topk_idx=SimpleNamespace(shape=(4224, 8)), topk_weights=None)
            namespace["run_sm90_mega_routed"](moe, SimpleNamespace(device="cuda"), None, None, buf, tokens)
            assert shapes == [(expect, 4096)], (mode, enabled, tokens, shapes)
            assert calls[0] == ("pre", tokens), "pre-dispatch must still see the real token count (it pads every slot with -1)"
            assert calls[1][1] == heuristic(expect), calls
        assert calls and heuristic(256) == "prefill-128x256" and heuristic(1) == "swap-ab-64x128"
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    assert namespace["idle_output_rows"](128, 8, 4) == 256 and heuristic(255) != "prefill-128x256" and heuristic(256) == "prefill-128x256"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("checks", nargs="+", choices=["admission", "dp-ties", "idle-geometry"])
    args = parser.parse_args()
    if "admission" in args.checks:
        check_admission_sites(args.source)
        check_deferred_metadata(args.source)
        print("PASS: Decode admission device copies all have a non-blocking branch; deferred page-index sends keep order, wait for their readback and skip aborted/failed receivers")
    if "dp-ties" in args.checks:
        check_dp_ties(args.source)
        print("PASS: equal total_requests loads rotate across DP ranks; the least-loaded rank still wins; default keeps the lowest index")
    if "idle-geometry" in args.checks:
        check_idle_geometry(args.source)
        print("PASS: an idle Prefill rank passes 256 masked rows (DeepGEMM Prefill geometry); Decode, disabled and busy ranks keep their row counts (DeepGEMM mocked)")
