"""CPU checks of the production capture control and received-row analysis."""
import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]


def capture_namespace(monkeypatch, tmp_path, enabled=True):
    patch = (ROOT / "patches/sglang/decode-routing-capture.patch").read_text()
    tail = patch.split("+++ b/python/sglang/srt/layers/moe/decode_routing_capture.py\n", 1)[1]
    source = "\n".join(line[1:] for line in tail.splitlines() if line.startswith("+"))
    tree = ast.parse(source)
    # Execute actual Python control functions; only torch/Triton boundaries are
    # substituted. The real Triton kernel has its own CPU-interpreter check.
    tree.body = [node for node in tree.body if not (isinstance(node, ast.Import) and any(n.name in {"torch", "triton", "triton.language"} for n in node.names)) and not (isinstance(node, ast.FunctionDef) and node.name == "_histogram")]
    monkeypatch.setenv("SGLANG_DECODE_ROUTING_DIR", str(tmp_path) if enabled else "")
    monkeypatch.setenv("SGLANG_DECODE_ROUTING_INTERVAL", "2")
    monkeypatch.setenv("SGLANG_DECODE_ROUTING_SAMPLES", "2")
    allocations = []

    class Tensor:
        def __init__(self, value):
            self.value = value

        def fill_(self, value):
            self.value = value

        def zero_(self):
            self.value = 0

        def cpu(self):
            return self

        def tolist(self):
            return self.value

    def zeros(size, **kwargs):
        allocations.append(size)
        return Tensor(0 if size == () else [0] * size)

    torch = NS(zeros=zeros, int32="int32", Tensor=Tensor, stack=lambda values: Tensor([v.value for v in values]))
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", NS(get_moe_ep_group=lambda: NS(rank_in_group=2)))
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", NS(get_server_args=lambda: NS(disaggregation_mode="decode")))
    namespace = {"torch": torch}
    exec(compile(tree, "decode_routing_capture.py", "exec"), namespace)  # noqa: S102 - execute the actual patch control with GPU boundaries substituted
    return namespace, allocations


def test_disabled_allocates_nothing(monkeypatch, tmp_path):
    ns, allocations = capture_namespace(monkeypatch, tmp_path, enabled=False)
    assert ns["make_capture"](0) is None
    ns["begin_forward"]()
    ns["end_forward"]()
    assert not allocations and not list(tmp_path.iterdir())


def test_capture_window_interval_limit_and_layer_counts(monkeypatch, tmp_path):
    ns, allocations = capture_namespace(monkeypatch, tmp_path)
    first, last = ns["make_capture"](0), ns["make_capture"](93)
    assert allocations == [(), 130, 130]
    for _ in range(2):
        ns["begin_forward"]()
        ns["end_forward"]()
    assert not list(tmp_path.glob("*.jsonl"))
    (tmp_path / "active").touch()
    for step in range(2, 9):
        ns["begin_forward"]()
        if ns["_state"].selected:
            assert step in (2, 4)
            first.output.value = [8] + [0] * 127 + [1, 8]
            last.output.value = [0] * 127 + [16] + [2, 16]
        ns["end_forward"]()
    rows = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    assert [row["target_forward_step"] for row in rows] == [2, 4]
    assert all(row["source_ep_rank"] == 2 for row in rows)
    assert rows[0]["counts_by_layer"]["93"][-1] == 16
    assert rows[0]["live_rows_by_layer"] == {"0": 1, "93": 2}
    assert ns["_state"].gate.value == 0
    with pytest.raises(ValueError, match="Duplicate"):
        ns["make_capture"](0)


def test_prefill_is_excluded(monkeypatch, tmp_path):
    ns, allocations = capture_namespace(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", NS(get_server_args=lambda: NS(disaggregation_mode="prefill")))
    assert ns["make_capture"](0) is None
    assert not allocations


def analyzer():
    spec = importlib.util.spec_from_file_location("routing_summary", ROOT / "scripts/summarize-decode-routing.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_receive_counts_and_partial_samples(tmp_path):
    for rank in range(4):
        counts = [0] * 128
        counts[33] = 20
        row = {"schema": "decode-routing/v1", "physical_experts": 128, "target_forward_step": 100, "source_ep_rank": rank, "counts_by_layer": {"7": counts}, "valid_topk_slots_by_layer": {"7": 20}}
        lines = [row]
        if rank == 0:
            lines.append({**row, "target_forward_step": 110})
        (tmp_path / f"rank-{rank}-pid-1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    result = analyzer().summarize(tmp_path)
    layer = result["samples"][0]["layers"]["7"]
    assert layer["rows_by_physical_expert"][33] == 80
    assert layer["rows_by_destination_rank"] == [0, 80, 0, 0]
    assert layer["experts_over_64_rows"] == [33]
    assert layer["m64_blocks_by_destination_rank"] == [0, 2, 0, 0]
    assert result["incomplete_samples"] == [{"target_forward_step": 110, "source_ranks": [0]}]
