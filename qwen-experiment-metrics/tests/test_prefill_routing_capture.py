"""CPU tests for physical routing payload counts and bounded capture control."""
import ast
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[2]
summarize = runpy.run_path(str(ROOT / "scripts/summarize-prefill-routing.py"))["summarize"]


def fixture(tmp_path):
    placement = tmp_path / "placement.json"
    # Four hot experts local everywhere, then 31 distinct cold experts/rank.
    row = [e for r in range(4) for e in [0, 1, 2, 3, *range(4 + 31 * r, 35 + 31 * r)]]
    placement.write_text(json.dumps({"physical_to_logical_map": [row] * 94}))
    directory = tmp_path / "prefill-0"
    directory.mkdir()
    for rank in range(4):
        # Four local hot outputs; four remote cold outputs on the next rank.
        ids = np.tile([35 * rank + x for x in range(4)] + [35 * ((rank + 1) % 4) + x for x in range(4, 8)], (94, 2, 1)).astype(np.int32)
        metadata = {"schema": "prefill-routing/v1", "phase": "diagnostic", "worker": "prefill-0", "source_ep_rank": rank, "sample": 0, "target_forward_step": 101, "physical_experts": 140, "experts_per_destination_rank": 35, "hidden_size": 4096, "topk": 8, "layers": list(range(94)), "live_rows": 2}
        np.savez(directory / f"rank-{rank}-sample-0.npz", ids=ids, metadata=json.dumps(metadata))
    return placement


def alter(path, change):
    with np.load(path, allow_pickle=False) as data:
        ids, meta = data["ids"].copy(), json.loads(data["metadata"].item())
    change(ids, meta)
    np.savez(path, ids=ids, metadata=json.dumps(meta))


def test_remote_direction_bytes_and_cold_placement_bound(tmp_path):
    placement = fixture(tmp_path)
    result = summarize(tmp_path, placement, 1, 4, 1)
    layer = result["workers"][0]["layers"][0]
    assert result["valid"]
    assert layer["slots_source_to_expert_rank"] == [[8, 8, 0, 0], [0, 8, 8, 0], [0, 0, 8, 8], [8, 0, 0, 8]]
    assert layer["remote_slots"] == 32
    assert layer["remote_slot_fraction"] == 0.5
    assert layer["fc2_remote_output_bytes_by_expert_rank"] == [65536] * 4
    assert layer["combine_read_payload_bytes_by_source_rank"] == [131072] * 4
    assert layer["remote_token_destination_pairs"] == 8
    assert layer["cold_relocation_saved_remote_slots_upper_bound"] == 32


@pytest.mark.parametrize("failure", ["missing", "duplicate", "step", "negative", "logical_duplicate", "shape", "empty", "phase"])
def test_rejects_incomplete_or_invalid_evidence(tmp_path, failure):
    placement = fixture(tmp_path)
    path = tmp_path / "prefill-0/rank-0-sample-0.npz"
    if failure == "missing":
        path.unlink()
    elif failure == "duplicate":
        path.with_name("duplicate.npz").write_bytes(path.read_bytes())
    else:
        def change(ids, meta):
            if failure == "step": meta["target_forward_step"] = 102
            if failure == "phase": meta["phase"] = "normal"
            if failure == "negative": ids[0, 0, 0] = -1
            if failure == "logical_duplicate": ids[0, 0, 4] = 35  # another physical copy of logical 0
            if failure == "shape": meta["live_rows"] = 3
            if failure == "empty": meta["live_rows"] = 0
        alter(path, change)
    with pytest.raises(ValueError):
        summarize(tmp_path, placement, 1, 4, 1)


def capture_namespace(monkeypatch, tmp_path):
    patch = (ROOT / "patches/sglang/prefill-routing-capture.patch").read_text()
    tail = patch.split("+++ b/python/sglang/srt/layers/moe/prefill_routing_capture.py\n", 1)[1]
    source = "\n".join(line[1:] for line in tail.splitlines() if line.startswith("+"))
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Import) and any(x.name == "torch" for x in n.names))]
    operations = []

    class Tensor:
        def __init__(self, a): self.a = np.asarray(a)
        @property
        def shape(self): return self.a.shape
        @property
        def ndim(self): return self.a.ndim
        def __getitem__(self, key): return Tensor(self.a[key])
        def detach(self): return self
        def to(self, **kwargs):
            operations.append("copy")
            return Tensor(self.a.astype(np.int32, copy=True))
        def cpu(self):
            operations.append("cpu")
            return self
        def numpy(self): return self.a
        def item(self): return self.a.item()

    torch = NS(Tensor=Tensor, int32=np.int32, stack=lambda rows: Tensor(np.stack([r.a for r in rows])))
    monkeypatch.setenv("SGLANG_PREFILL_ROUTING_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_PREFILL_ROUTING_SAMPLES", "2")
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", NS(get_moe_ep_group=lambda: NS(rank_in_group=0)))
    ns = {"torch": torch}
    exec(compile(tree, "prefill_routing_capture.py", "exec"), ns)  # noqa: S102 - production capture with tensor boundaries substituted
    return ns, Tensor, operations


def test_no_tensor_work_outside_window_and_two_sample_limit(monkeypatch, tmp_path):
    ns, Tensor, operations = capture_namespace(monkeypatch, tmp_path)
    layers = [ns["LayerCapture"](i) for i in range(94)]
    ids = Tensor(np.tile(np.arange(8), (2, 1)))
    for step in range(5):
        if step == 1: (tmp_path / "active").touch()
        ns["begin_forward"](step)
        for layer in layers: layer.record(ids, 2)
        ns["end_forward"]()
        if step == 0: assert not operations and not list(tmp_path.glob("*.npz"))
    assert operations.count("copy") == 188
    assert operations.count("cpu") == 2
    assert len(list(tmp_path.glob("*.npz"))) == 2
    with np.load(tmp_path / "rank-0-sample-1.npz") as data:
        assert json.loads(data["metadata"].item())["target_forward_step"] == 2


def test_idle_rank_and_partial_layers(monkeypatch, tmp_path):
    ns, Tensor, _ = capture_namespace(monkeypatch, tmp_path)
    layers = [ns["LayerCapture"](i) for i in range(94)]
    (tmp_path / "active").touch()
    ns["begin_forward"](10)
    for layer in layers: layer.record(Tensor(np.empty((0, 8), dtype=np.int32)), 0)
    ns["end_forward"]()
    with np.load(tmp_path / "rank-0-sample-0.npz") as data: assert data["ids"].shape == (94, 0, 8)
    ns["begin_forward"](11)
    layers[0].record(Tensor(np.zeros((1, 8), dtype=np.int32)), 1)
    with pytest.raises(ValueError, match="every target MoE layer"):
        ns["end_forward"]()
