"""CPU checks of paired measured-frequency routing, without loading DeepGEMM."""
import ast
import hashlib
import json
import runpy
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def bench_namespace(rank=0):
    torch = pytest.importorskip("torch")
    helper = ast.parse((ROOT / "scripts/firmus_megamoe_replicas.py").read_text())
    data = next(ast.literal_eval(node.value) for node in helper.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "REAL_ROUTING_LAYERS")
    source = ast.parse((ROOT / "scripts/bench-firmus-megamoe.py").read_text())
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in ("case_name", "case_seed", "real_routing") or isinstance(node, (ast.Assign, ast.AugAssign)) and isinstance(node.targets[0] if isinstance(node, ast.Assign) else node.target, ast.Name) and (node.targets[0] if isinstance(node, ast.Assign) else node.target).id == "CASES"]
    context = {"torch": torch, "NUM_RANKS": 4, "NUM_EXPERTS": 128, "TOPK": 8, "REAL_ROUTING_LAYERS": data, "dist": types.SimpleNamespace(get_rank=lambda: rank)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "bench_cpu", "exec"), context)  # noqa: S102
    return context


def test_real_layer_data_and_case_selection():
    ns = bench_namespace()
    data = ns["REAL_ROUTING_LAYERS"]
    baseline = json.loads((ROOT / "configs/firmus-prefill-expert-location.json").read_text())["physical_to_logical_map"]
    assert list(data) == [43, 62, 7]
    assert [row["ascending_position"] for row in data.values()] == [47, 85, 10]
    assert [row["selection"] for row in data.values()] == ["median", "upper10", "lower10"]
    # Decode's 176-token case was added with the FP8 EAGLE3/IWG experiment.
    assert len(ns["CASES"]) == len(set(ns["CASES"])) == 20
    assert ("decode", "balanced", 176) in ns["CASES"]
    assert not any(case[1] in ("mixed50", "mixed64", "replica8") or case[1].endswith("-replica8") for case in ns["CASES"])
    for layer, row in data.items():
        assert len(row["counts"]) == 128 and min(row["counts"]) >= 0
        assert sum(row["counts"]) == 205343824
        hot = sorted(range(128), key=lambda e: (-row["counts"][e], e))[:8]
        assert row["hot8"] == hot
        assert row["hot8_share"] == sum(row["counts"][e] for e in hot) / sum(row["counts"])
        assert row["baseline_map"] == baseline[layer]
        for replicas, key in ((4, "replica4_map"),):
            mapping = row[key]
            width = (128 + 3 * replicas) // 4
            assert len(mapping) == 128 + 3 * replicas
            assert all(mapping.count(e) == (4 if e in hot[:replicas] else 1) for e in range(128))
            assert all(len(set(mapping[r * width:(r + 1) * width])) == width and mapping[r * width:r * width + replicas] == hot[:replicas] for r in range(4))
        for suffix in ("", "-replica4"):
            assert ("prefill", f"real{layer}{suffix}", 4096) in ns["CASES"]
    for index, case in enumerate(ns["CASES"][:12]):
        assert ns["case_seed"](case, 3) == index * 8 + 3


@pytest.mark.parametrize("layer", [43, 62, 7])
def test_paired_inputs_routing_and_local_preference(layer):
    ns = bench_namespace()
    torch = ns["torch"]
    row = ns["REAL_ROUTING_LAYERS"][layer]
    rank_hashes = []
    for rank in range(4):
        ns["dist"].get_rank = lambda rank=rank: rank
        pair = []
        for replicas in (0, 4):
            case = ("prefill", f"real{layer}" + (f"-replica{replicas}" if replicas else ""), 4096)
            generator = torch.Generator().manual_seed(ns["case_seed"](case, rank))
            hidden = torch.randn(128, 16, generator=generator)
            ids, weights, logical = ns["real_routing"](layer, replicas, 128, generator, "cpu")
            pair.append((hidden, weights, logical))
            mapping = row[{0: "baseline_map", 4: "replica4_map"}[replicas]]
            width = len(mapping) // 4
            assert ids.dtype == torch.int32 and ids.is_contiguous()
            assert torch.equal(torch.tensor(mapping)[ids.long()], logical.long())
            assert (logical.sort(1).values[:, 1:] != logical.sort(1).values[:, :-1]).all()
            assert torch.allclose(weights.sum(1), torch.ones(128))
            local_available = torch.isin(logical, torch.tensor(mapping[rank * width:(rank + 1) * width]))
            assert torch.equal(ids // width == rank, local_available)
            if replicas:
                assert ((ids // width == rank) | ~torch.isin(logical, torch.tensor(row["hot8"][:replicas]))).all()
        assert all(torch.equal(a, b) for candidate in pair[1:] for a, b in zip(pair[0], candidate))
        rank_hashes.append(hashlib.sha256(pair[0][2].numpy().tobytes()).hexdigest())
    assert len(set(rank_hashes)) == 4


def test_gumbel_zero_counts_and_unique_top8():
    ns = bench_namespace()
    ns["REAL_ROUTING_LAYERS"][43]["counts"] = [0] * 120 + list(range(1, 9))
    torch = ns["torch"]
    _, _, logical = ns["real_routing"](43, 0, 64, torch.Generator().manual_seed(17), "cpu")
    assert torch.equal(logical.sort(1).values, torch.arange(120, 128).expand(64, 8))


def test_logical_weight_and_scale_identity_across_placements():
    ns = bench_namespace()
    torch = ns["torch"]
    source = ast.parse((ROOT / "scripts/firmus_megamoe_replicas.py").read_text())
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in ("physical_to_logical", "replica_weights")]
    # Shrink only matrix dimensions; exercise actual logical-ID seeds and slot
    # initialization for every expert and all 4 ranks without allocating GiB.
    proxy = types.SimpleNamespace(**{name: getattr(torch, name) for name in ("float8_e4m3fn", "float32", "bfloat16", "Generator", "randn", "rand")})
    proxy.empty = lambda shape, **kwargs: torch.empty((shape[0], 2, 3), **kwargs)
    context = {"torch": proxy}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "weights_cpu", "exec"), context)  # noqa: S102
    reference = {}
    maps = [list(range(128)), context["physical_to_logical"]()]
    maps += [row[key] for row in ns["REAL_ROUTING_LAYERS"].values() for key in ("baseline_map", "replica4_map")]
    for mapping in maps:
        width = len(mapping) // 4
        for rank in range(4):
            weights = context["replica_weights"](rank, "cpu", mapping)
            for slot, expert in enumerate(mapping[rank * width:(rank + 1) * width]):
                payload = b"".join(tensor[slot].contiguous().view(torch.uint8).numpy().tobytes() for tensor in weights)
                assert payload == reference.setdefault(expert, payload)
    for rank in range(4):
        assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(context["replica_weights"](rank, "cpu"), context["replica_weights"](rank, "cpu", context["physical_to_logical"]())))


def test_summary_displays_all_real_routing_pairs(tmp_path, monkeypatch):
    existing = runpy.run_path(str(ROOT / "qwen-experiment-metrics/tests/test_firmus_kernel_bench.py"))
    summarize = existing["load_summary"](monkeypatch)
    (tmp_path / "variants.json").write_text(json.dumps({"deep_gemm": [{"name": "image", "patches": [], "timeline": "not-requested"}, {"name": "control-sf128-swizzle", "patches": [], "timeline": "not-requested"}], "sglang": None}))
    (tmp_path / "steps.tsv").write_text("image\tbench\t0\t50\ncontrol-sf128-swizzle\tgate\t0\t87\ncontrol-sf128-swizzle\tbench\t0\t85\n")
    rows = [existing["bench_row"](f"prefill-real{layer}{suffix}-4096", 1.5, f"{layer}{suffix}") for layer in (43, 62, 7) for suffix in ("", "-replica4")]
    for variant in ("image", "control-sf128-swizzle"):
        existing["write_rows"](tmp_path / variant / "bench.jsonl", rows)
    text, failed = summarize(tmp_path)
    assert failed == []
    for row in rows:
        assert f"| {row['case']} | 1.5000 | 1.5000 (+0.0%) |" in text


def test_replica4_gate_keeps_idle_mixed_and_replay_coverage():
    source = ast.parse((ROOT / "scripts/firmus_megamoe_replicas.py").read_text())
    gate = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "check_replicas")
    nodes = [node for node in gate.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "cases" for target in node.targets) or isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "cases"]
    context = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "gate_cases", "exec"), context)  # noqa: S102
    cases = context["cases"]
    assert len(cases) == 22
    by_name = {name: (counts, kind, padded) for name, counts, kind, padded in cases}
    assert by_name["mixed-tiny"] == ([7, 0, 65, 129], "cold", False)
    assert by_name["mixed-prefill-geometry"] == ([4096, 1, 0, 303], "mixed", True)
    assert by_name["idle-prefill-geometry"] == ([4096, 4096, 4096, 0], "mixed", True)
    assert cases[-2:] == [("all-idle", [0] * 4, "hot", True), ("live-after-idle", [4096] * 4, "mixed", True)]
    for rows in (279, 280, 281):
        assert by_name[f"boundary-{rows}"] == ([rows] * 4, "mixed", False)
    replay = next(node for node in ast.walk(gate) if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "replay")
    assert ast.unparse(replay.iter) == "range(2)"
