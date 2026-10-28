"""CPU-only raw capture, placement search and publication threshold tests."""
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/optimize-prefill-expert-placement.py"
optimizer = runpy.run_path(str(SCRIPT))


def placement(layers=2):
    row = [e for rank in range(4) for e in [0, 1, 2, 3, *range(4 + 31 * rank, 35 + 31 * rank)]]
    return np.tile(row, (layers, 1))


def write_capture(tmp_path, layers=2, tokens=256, uniform=False):
    mapping = placement(layers)
    rng = np.random.default_rng(42)
    directory = tmp_path / "routing"
    for worker in range(3):
        folder = directory / f"prefill-{worker}"
        folder.mkdir(parents=True)
        for rank in range(4):
            for sample in range(4):
                if uniform:
                    logical = np.stack([np.stack([rng.choice(128, 8, replace=False) for _ in range(tokens)]) for _ in range(layers)])
                else:
                    # Ranks 0/1 exchange equally busy experts; ranks 2/3
                    # have less work, avoiding a four-rank max-time plateau.
                    rows = tokens if rank < 2 else max(1, tokens // 8)
                    destination = rank ^ 1
                    logical = np.tile(np.arange(4 + 31 * destination, 12 + 31 * destination), (layers, rows, 1))
                ids = np.empty_like(logical, dtype=np.int32)
                for layer in range(layers):
                    for e in np.unique(logical[layer]):
                        slots = np.flatnonzero(mapping[layer] == e)
                        local = slots[slots // 35 == rank]
                        ids[layer][logical[layer] == e] = int(local[0] if len(local) else slots[0])
                # Sample indices differ across ranks; align using forward IDs.
                step = 100 + (sample if rank % 2 == 0 else 3 - sample)
                meta = {"schema": "prefill-routing/v1", "phase": "diagnostic", "worker": folder.name, "source_ep_rank": rank, "sample": sample, "target_forward_step": step, "time_ns": 123, "physical_experts": 140, "experts_per_destination_rank": 35, "hidden_size": 4096, "topk": 8, "layers": list(range(layers)), "live_rows": logical.shape[1]}
                np.savez_compressed(folder / f"rank-{rank}-sample-{sample}.npz", ids=ids, metadata=json.dumps(meta))
    path = tmp_path / "placement.json"
    path.write_text(json.dumps({"physical_to_logical_map": mapping.tolist()}))
    return directory, path, mapping


def test_affinity_preserves_constraints_is_deterministic_and_balanced(tmp_path):
    directory, _, mapping = write_capture(tmp_path)
    training, held_out, split, provenance = optimizer["load_capture"](directory, mapping)
    assert training.shape == held_out.shape == (6, 2, 4, 128)
    assert all(s["training"] == [100, 101] and s["held_out"] == [102, 103] for s in split)
    assert len(provenance) == 48
    candidate, report = optimizer["plan"](training, held_out, mapping)
    again, repeat = optimizer["plan"](training, held_out, mapping)
    assert np.array_equal(candidate, again)
    assert report == repeat
    optimizer["validate_placement"](candidate, 12, layers=2)
    for half in ("training", "held_out"):
        for before, after in zip(report[half]["current"]["layers"], report[half]["candidate"]["layers"], strict=True):
            assert sum(after["mean_fc2_sent_bytes_by_rank"]) < sum(before["mean_fc2_sent_bytes_by_rank"])
            assert max(after["mean_useful_rows_by_rank"]) <= max(before["mean_useful_rows_by_rank"])
            assert max(after["mean_padded_rows_by_rank"]) <= max(before["mean_padded_rows_by_rank"])
            assert after["predicted_time_us"]["0.5"] < before["predicted_time_us"]["0.5"]
    assert not report["accepted"]  # Two layers are not extrapolated to 94.


def test_padding_is_per_forward_per_expert_and_fc2_direction():
    counts = np.zeros((2, 4, 128), dtype=np.int64)
    counts[0, 0, 4] = 128  # Sole owner rank 0; second forward originates on rank 1.
    counts[1, 1, 4] = 64
    metrics = optimizer["contributions"](counts)[4, 0]
    assert metrics[:, 0, 1].tolist() == [128, 128]
    assert metrics[:, :, 2].tolist() == [[0] * 4, [64 * 8192, 0, 0, 0]]
    assert metrics[:, :, 3].tolist() == [[0] * 4, [0, 64 * 8192, 0, 0]]
    hot = optimizer["contributions"](counts)[4, 4]
    assert hot[..., 2:].sum() == 0
    assert hot[:, :, 0].tolist() == counts[:, :, 4].tolist()
    # Average of max per forward differs from max of average rank load.
    metrics = np.zeros((2, 4, 4), dtype=np.int64)
    metrics[0, 0, 1] = metrics[1, 1, 1] = 128
    assert optimizer["times"](metrics, 0).mean() == pytest.approx(128 * (optimizer["FC1_US_PER_ROW"] + optimizer["FC2_US_PER_ROW"]))


def test_replica_set_exchange_can_promote_a_cold_expert():
    row = placement(1)[0]
    counts = np.zeros((2, 4, 128), dtype=np.int64)
    counts[:, :, 4] = 1024
    candidate, moves = optimizer["optimize_layer"](counts, row.copy())
    optimizer["validate_placement"](candidate[None], 12, layers=1)
    assert moves > 0
    assert (candidate == 4).sum() == 4
    assert sum((candidate == e).sum() == 4 for e in range(128)) == 4


def cli(tmp_path, directory, current, output="optimized.json", report="report.json"):
    return subprocess.run([sys.executable, str(SCRIPT), "--directory", str(directory), "--placement", str(current), "--redundant-experts", "12", "--output", str(tmp_path / output), "--report", str(tmp_path / report)], text=True, capture_output=True, check=False)


def test_cli_publishes_94_layer_affinity_candidate(tmp_path):
    directory, path, mapping = write_capture(tmp_path, layers=94, tokens=512)
    result = cli(tmp_path, directory, path)
    assert result.returncode == 0, result.stderr
    candidate = np.asarray(json.loads((tmp_path / "optimized.json").read_text())["physical_to_logical_map"])
    optimizer["validate_placement"](candidate, 12)
    assert candidate.shape == mapping.shape
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["accepted"]
    assert report["training"]["predicted_saving_us_per_iteration"]["0.5"] >= 2000
    assert report["held_out"]["predicted_saving_us_per_iteration"] == report["training"]["predicted_saving_us_per_iteration"]
    assert set(report["training"]["predicted_saving_us_per_iteration"]) == {"0", "0.5", "1"}
    assert len(report["training"]["candidate"]["layers"]) == 94


def test_uniform_random_refuses_without_placement_output(tmp_path):
    directory, path, _ = write_capture(tmp_path, layers=94, tokens=16, uniform=True)
    result = cli(tmp_path, directory, path)
    assert result.returncode != 0
    assert "Refused placement" in result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert not report["accepted"]
    assert report["training"]["predicted_saving_us_per_iteration"]["0.5"] < 2000
    assert not (tmp_path / "optimized.json").exists()


def test_held_out_reversal_refuses():
    mapping = placement(94)
    train = np.zeros((1, 94, 4, 128), dtype=np.int64)
    held = np.zeros_like(train)
    for rank in range(4):
        rows = 512 if rank < 2 else 64
        destination = rank ^ 1
        train[:, :, rank, 4 + 31 * destination:12 + 31 * destination] = rows
        held[:, :, rank, 4 + 31 * rank:12 + 31 * rank] = rows
    _, report = optimizer["plan"](train, held, mapping)
    assert report["training"]["predicted_saving_us_per_iteration"]["0.5"] >= 2000
    assert report["held_out"]["predicted_saving_us_per_iteration"]["0.5"] < 0
    assert not report["accepted"]
    assert any("below half" in r for r in report["refusal_reasons"])


def test_symmetric_max_time_plateau_does_not_accept_equal_cost_moves():
    mapping = placement()
    counts = np.zeros((1, 2, 4, 128), dtype=np.int64)
    for rank in range(4):
        destination = (rank + 1) % 4
        counts[:, :, rank, 4 + 31 * destination:12 + 31 * destination] = 256
    candidate, report = optimizer["plan"](counts, counts, mapping)
    assert np.array_equal(candidate, mapping)
    assert report["moves_by_layer"] == [0, 0]
    assert not report["accepted"]


@pytest.mark.parametrize("failure", ["missing_rank", "step", "duplicate_step", "duplicate_logical", "wrong_dispatch", "shape", "phase", "negative"])
def test_reject_invalid_capture(tmp_path, failure):
    directory, _, mapping = write_capture(tmp_path, tokens=8)
    path = directory / "prefill-0/rank-0-sample-0.npz"
    if failure == "missing_rank":
        path.unlink()
    else:
        with np.load(path, allow_pickle=False) as data:
            ids, meta = data["ids"].copy(), json.loads(data["metadata"].item())
        if failure == "step": meta["target_forward_step"] = 999
        if failure == "duplicate_step": meta["target_forward_step"] = 101
        if failure == "duplicate_logical": ids[0, 0, 1] = ids[0, 0, 0]
        if failure == "wrong_dispatch": ids[0, 0, 0] = 35  # Nonlocal copy of logical hot expert 0.
        if failure == "shape": meta["live_rows"] = 9
        if failure == "phase": meta["phase"] = "normal"
        if failure == "negative": ids[0, 0, 0] = -1
        np.savez_compressed(path, ids=ids, metadata=json.dumps(meta))
    with pytest.raises(ValueError):
        optimizer["load_capture"](directory, mapping)


def test_summary_is_not_supported(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text('{"schema":"prefill-routing-summary/v1"}')
    with pytest.raises(ValueError, match="summary JSON is unsupported"):
        optimizer["load_capture"](summary, placement())


@pytest.mark.parametrize("failure", ["redundancy", "missing", "rank_duplicate", "partial_replica", "layers"])
def test_reject_checker_incompatible_placement(failure):
    mapping = placement()
    redundancy = 12
    if failure == "redundancy": redundancy = 8
    if failure == "missing": mapping[0, 4] = 5
    if failure == "rank_duplicate": mapping[0, 0], mapping[0, 35] = mapping[0, 35], mapping[0, 4]
    if failure == "partial_replica": mapping[0, 0] = 4
    if failure == "layers": mapping = mapping[:1]
    with pytest.raises(ValueError):
        optimizer["validate_placement"](mapping, redundancy, layers=2)
