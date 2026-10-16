from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ANALYZE = runpy.run_path(str(ROOT / "research/firmus-vllm-structural/analyze_stage_budget.py"))["analyze"]


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_stage_targets_use_measured_goodput_not_local_gate(tmp_path: Path, fraction: float) -> None:
    goodput = 4 * fraction / 120
    score = {
        "request_count": 4,
        "concurrency": 192,
        "good_request_fraction": fraction,
        "eligible": fraction >= 0.9,
        "score_output_tokens_per_second": goodput * 200 if fraction >= 0.9 else 0,
        "observed_input_sequence_length_sum": 4 * 3986,
    }
    export = {"goodput": {"avg": goodput}, "benchmark_duration": {"avg": 120}, "output_token_throughput": {"avg": 4 * 200 / 120}}
    rows = [{"metadata": {"benchmark_phase": "profiling", "request_start_ns": start * 10**9, "request_end_ns": (start + 4) * 10**9}, "metrics": {name: {"value": value} for name, value in (("time_to_first_token", 1000), ("decode_duration", 3000), ("request_latency", 4000), ("inter_token_latency", 15))}} for start in (1, 31, 61, 91)]
    (tmp_path / "score.json").write_text(json.dumps(score))
    (tmp_path / "profile_export_aiperf.json").write_text(json.dumps(export))
    (tmp_path / "profile_export.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    original = {path: path.read_bytes() for path in tmp_path.iterdir()}

    result = ANALYZE(tmp_path)

    assert result["good_output_tokens_per_second"] == pytest.approx(goodput * 200)
    assert result["local_gated_value"] == score["score_output_tokens_per_second"]
    assert "official_stored_score" not in result
    targets = result["target_sensitivity"]
    assert len(targets) == (3 if fraction else 2)
    if fraction:
        assert targets[1]["target_good_output_tokens_per_second"] == pytest.approx(2 * goodput * 200)
    else:
        assert all(target["input_tokens_per_second_at_current_good_fraction"] is None for target in targets)
    assert original == {path: path.read_bytes() for path in tmp_path.iterdir()}
