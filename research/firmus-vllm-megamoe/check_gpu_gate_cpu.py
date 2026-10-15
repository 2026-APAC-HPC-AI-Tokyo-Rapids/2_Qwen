"""Finite GPU-free row-ownership and exact native pipeline wiring checks."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import gpu_gate as gate

# Exact function source from bb77e51 scripts/check-firmus-megamoe.py.
# Compare both ASTs using the CURRENT Python runtime: ast.dump fields differ
# between Python versions even when equations are unchanged.
UPSTREAM_MATH_SOURCE = 'def quantized(x, group):\n    blocks = x.float().reshape(x.shape[0], -1, group)\n    scale = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0\n    return ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).reshape(x.shape)\n\ndef dequantized(w, scale):\n    return w.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)\n'
UPSTREAM_MATH_SHA256 = "e360d6603938bd46b4ba9bcef59607c3106df8bfb4972e593af3588d509ae27d"


def math_ast(source):
    tree = ast.parse(source).body[0]
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    return ast.dump(tree, include_attributes=False)


def math_digest(function):
    return hashlib.sha256(math_ast(inspect.getsource(function)).encode()).hexdigest()


def check_math_identity():
    assert hashlib.sha256(UPSTREAM_MATH_SOURCE.encode()).hexdigest() == UPSTREAM_MATH_SHA256
    expected = {node.name: ast.get_source_segment(UPSTREAM_MATH_SOURCE, node) for node in ast.parse(UPSTREAM_MATH_SOURCE).body}
    for name in ("quantized", "dequantized"):
        assert math_ast(inspect.getsource(getattr(gate, name))) == math_ast(expected[name]), name
    # A real equation change is still rejected on every Python version.
    changed = expected["quantized"].replace("/ 448.0", "/ 449.0")
    assert changed != expected["quantized"]
    assert math_ast(changed) != math_ast(expected["quantized"])


class Tensor:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def __getitem__(self, _):
        return Tensor(self.name + "_view", self.calls)

    def copy_(self, other):
        self.calls.append(("split_copy", self, other))


def pipeline_wiring():
    calls = []
    case = gate.Pipeline.__new__(gate.Pipeline)
    case.lib = SimpleNamespace(activation=object())
    case.full, case.local, case.masks = [
        Tensor(name, calls) for name in ("full", "local", "masks")
    ]
    case.native_output, case.local_output = (
        Tensor("native_output", calls),
        Tensor("local_output", calls),
    )
    case.workspace13, case.workspace2 = object(), object()
    case.start, case.stop, case.shard = 8, 16, 8
    weights, ids, quant, scale = object(), object(), object(), object()
    case.route = lambda hidden, masks, **kwargs: (
        calls.append(("routing", hidden, masks, kwargs)) or (weights, ids)
    )
    reduce_impl = object()
    case.experts = SimpleNamespace(
        quant_config=object(),
        apply=lambda *args: calls.append(("native_experts", args)),
        finalize_weight_and_reduce_impl=lambda: reduce_impl,
    )
    case.prepare = SimpleNamespace(
        prepare=lambda *args, **kwargs: (
            calls.append(("prepare", args, kwargs)) or (quant, scale, None, None, None)
        ),
        finalize=lambda *args: calls.append(("finalize", args)),
    )
    buffer = SimpleNamespace(
        x=object(), x_sf=object(), topk_idx=object(), topk_weights=object()
    )
    case.weights = SimpleNamespace(
        w13=object(),
        w2=object(),
        expert_map=object(),
        buffer=buffer,
        l1=object(),
        l2=object(),
    )
    native_result, mega_result = object(), object()
    case.group = SimpleNamespace(
        all_reduce=lambda value: calls.append(("all_reduce", value)) or native_result,
        all_gather=lambda value, **kwargs: (
            calls.append(("all_gather", value, kwargs)) or mega_result
        ),
    )
    case.mega = SimpleNamespace(
        mega_moe_pre_dispatch_sm90=lambda *args, **kwargs: calls.append(
            ("predispatch", args, kwargs)
        ),
        fp8_mega_moe=lambda *args, **kwargs: calls.append(("mega", args, kwargs)),
    )
    assert case.native_forward() is native_result
    assert [call[0] for call in calls] == [
        "routing",
        "prepare",
        "native_experts",
        "finalize",
        "all_reduce",
    ]
    assert calls[0][3] == {"native": True}
    apply = calls[2][1]
    assert apply[0] is case.native_output and apply[1] is quant
    assert (
        apply[7] == 128 and apply[8] is case.weights.expert_map and apply[-1] is False
    )
    assert calls[3][1] == (
        case.native_output,
        case.native_output,
        weights,
        ids,
        False,
        reduce_impl,
    )
    assert calls[4][1] is case.native_output
    calls.clear()
    assert case.mega_forward() is mega_result
    assert [call[0] for call in calls] == [
        "split_copy",
        "routing",
        "predispatch",
        "mega",
        "all_gather",
    ]
    assert calls[1][3] == {}
    assert calls[1][1] is case.local
    assert calls[2][1][0] is case.local
    assert calls[2][2] == {
        "num_tokens": 8,
        "group_size": 128,
        "routed_scaling_factor": 1.0,
    }
    assert calls[3][1] == (case.local_output, case.weights.l1, case.weights.l2, buffer)
    assert calls[3][2] == {
        "recipe": (128, 128, 128),
        "activation": "swiglu",
        "fast_math": True,
    }
    assert calls[4][1] is case.local_output and calls[4][2] == {"dim": 0}
    return {
        "native": "native route/prepare/TritonExperts/finalize/ordinary allreduce",
        "mega": "split/native route/predispatch/MegaMoE/allgather",
        "double_dispatch_or_allreduce": False,
    }


class MaskMatrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    def masked_fill_(self, mask, value):
        for row, flags in zip(self.rows, mask):
            for slot, disabled in enumerate(flags):
                if disabled:
                    row[slot] = value


def mask_contract():
    count = 0
    for rows in gate.ROWS:
        for pattern in gate.PATTERNS:
            for replay in range(gate.REPLAYS):
                plan = gate.layout(rows, pattern, replay)
                masks = [[not live or (slot == 7 and index == 0) for slot in range(8)] for live, index in zip(plan["live"], plan["indices"])]
                original_ids = [[(index * 13 + slot) % 128 for slot in range(8)] for index in plan["indices"]]
                original_weights = [[(slot + 1) / 36 for slot in range(8)] for _ in original_ids]
                nw, ni = MaskMatrix(original_weights), MaskMatrix(original_ids)
                mw, mi = MaskMatrix(original_weights), MaskMatrix(original_ids)
                gate.mask_routes(nw, ni, masks, native=True)
                gate.mask_routes(mw, mi, masks, native=False)
                assert ni.rows == original_ids
                for row in range(plan["padded"]):
                    expected = sum(w * (expert + 1) for expert, w, disabled in zip(original_ids[row], original_weights[row], masks[row]) if not disabled)
                    native = sum(w * (expert + 1) for expert, w in zip(ni.rows[row], nw.rows[row]))
                    mega = sum(w * (expert + 1) for expert, w in zip(mi.rows[row], mw.rows[row]) if expert >= 0)
                    assert abs(native - expected) < 1e-12 and abs(mega - expected) < 1e-12
                    assert all(0 <= expert < 128 for expert in ni.rows[row])
                    if not plan["live"][row]:
                        assert native == mega == 0
                # The previous shared -1 representation is deliberately rejected
                # by the native input contract, including an all-empty replay.
                if any(any(row) for row in masks):
                    assert any(expert < 0 for row in mi.rows for expert in row)
                count += 1
    return count


def check():
    check_math_identity()
    count = 0
    for rows in gate.ROWS:
        previous = None
        for pattern in gate.PATTERNS:
            for replay in range(gate.REPLAYS):
                info = gate.layout(rows, pattern, replay)
                assert info["padded"] == info["shard"] * 4
                assert len(info["indices"]) == len(info["live"]) == info["padded"]
                assert len(info["live_per_rank"]) == 4
                assert sum(info["live_per_rank"]) == sum(info["live"])
                assert not any(info["live"][rows:])
                assert all(0 <= index < 16 for index in info["indices"])
                if previous is not None and replay:
                    assert info["indices"] != previous
                previous = info["indices"]
                # Contiguous shards cover every global row exactly once.
                owned = [
                    row
                    for rank in range(4)
                    for row in range(rank * info["shard"], (rank + 1) * info["shard"])
                ]
                assert owned == list(range(info["padded"]))
                if pattern == "empty_tail" and replay % 2 == 0:
                    assert info["live_per_rank"][2] == 0
                count += 1
    for rows, pattern, replay in [
        (0, "random", 0),
        (8192, "random", 0),
        (128, "unknown", 0),
        (128, "random", 4),
    ]:
        try:
            gate.layout(rows, pattern, replay)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported GPU sweep accepted")
    assert gate.sample_summary([1, 2, 3, 4, 5]) == {
        "median_ms": 3,
        "p95_ms": 5,
        "min_ms": 1,
        "max_ms": 5,
        "samples": 5,
    }
    return {
        "schema": "vllm-sm90-megamoe-gpu-gate-cpu-check/v1",
        "passed": True,
        "finite_layout_cases": count,
        "rows": gate.ROWS,
        "patterns": gate.PATTERNS,
        "changed_input_replays": gate.REPLAYS,
        "pipeline_wiring": pipeline_wiring(),
        "native_mega_effective_mask_cases": mask_contract(),
        "reference_identity_method": "same-runtime AST equality against SHA256-pinned upstream function source",
        "reference_function_source_sha256": UPSTREAM_MATH_SHA256,
        "changed_equation_negative_control_rejected": True,
        "reference_quantized_ast_sha256": math_digest(gate.quantized),
        "reference_dequantized_ast_sha256": math_digest(gate.dequantized),
        "cuda_execution": False,
        "limitations": "Host row ownership, call-order and ABI argument wiring with explicit sentinels only; no GPU math, capture, distributed communication or performance pass is claimed.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        "PASS: finite row ownership, changed-input plans and native/Mega pipeline wiring; GPU validation remains open"
    )
