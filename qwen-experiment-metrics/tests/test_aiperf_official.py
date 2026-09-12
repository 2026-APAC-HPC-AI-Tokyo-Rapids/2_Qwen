from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).parents[2]
PARSER = REPOSITORY / "scripts" / "parse-aiperf-result.py"


def metrics(fraction: float = 0.9) -> dict[str, object]:
    return {
        "goodput": {"unit": "requests/sec", "avg": 4.25},
        "good_request_fraction": {"unit": "fraction", "avg": fraction},
        "request_count": {"unit": "requests", "avg": 512},
        "error_request_count": None,
        "input_sequence_length": {"unit": "tokens", "avg": 4000},
        "output_sequence_length": {"unit": "tokens", "avg": 200},
    }


def test_parser_emits_official_score(tmp_path: Path) -> None:
    source = tmp_path / "profile_export_aiperf.json"
    output = tmp_path / "score.json"
    source.write_text(json.dumps(metrics()))
    subprocess.run([sys.executable, PARSER, source, output, "--concurrency", "16", "--require-eligible"], check=True)
    result = json.loads(output.read_text())
    assert result["eligible"] is True
    assert result["score_output_tokens_per_second"] == 850
    assert result["error_request_count"] == 0


def test_parser_rejects_ineligible_official_run(tmp_path: Path) -> None:
    source = tmp_path / "profile_export_aiperf.json"
    output = tmp_path / "score.json"
    source.write_text(json.dumps(metrics(0.899)))
    result = subprocess.run([sys.executable, PARSER, source, output, "--concurrency", "16", "--require-eligible"], check=False)
    assert result.returncode != 0
    assert json.loads(output.read_text())["score_output_tokens_per_second"] == 0


def test_official_command_contains_locked_workload() -> None:
    script = (REPOSITORY / "scripts" / "run-aiperf.sh").read_text()
    required = (
        "--endpoint /v1/chat/completions",
        "--tokenizer Qwen/Qwen3-235B-A22B-Instruct-2507",
        "--warmup-duration 60",
        "--benchmark-duration 120",
        "--benchmark-grace-period 30",
        "--synthetic-input-tokens-mean 4000",
        "--output-tokens-mean 200",
        "--num-dataset-entries 12800",
        "--random-seed 2026",
        "--cache-bust first_turn_prefix",
        '--goodput "time_to_first_token:3000 inter_token_latency:20"',
    )
    assert all(argument in script for argument in required)
    assert "--request-rate" not in script
