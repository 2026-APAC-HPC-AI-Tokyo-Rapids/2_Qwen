from __future__ import annotations

import copy
import json
import runpy
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PARSE = runpy.run_path(str(ROOT / "scripts/parse-firmus-reference-result.py"))["parse_reference"]


def fixture() -> dict:
    return json.loads((ROOT / "qwen-experiment-metrics/tests/fixtures/firmus-reference.json").read_text())


def test_real_reference_run_excludes_warmup_and_checks_joint_sla() -> None:
    raw = fixture()
    result = PARSE(raw["export"], raw["records"], 8)
    assert result["request_count"] == 85
    assert result["good_request_count"] == 78
    assert result["good_request_fraction"] == pytest.approx(78 / 85)
    assert result["goodput_requests_per_second"] == pytest.approx(78 / 62.862709504)
    assert result["official_submission"] is False
    assert result["official_validation"]["valid"] is False


@pytest.mark.parametrize("corruption", ["duration", "phase", "records", "tokenizer", "errors", "tokens", "latency"])
def test_rejects_incomplete_or_changed_reference_workload(corruption: str) -> None:
    raw = copy.deepcopy(fixture())
    export, records = raw["export"], raw["records"]
    if corruption == "duration":
        export["benchmark_duration"]["avg"] = 20
    elif corruption == "phase":
        export["input_config"]["phases"][0]["duration"] = 60
    elif corruption == "records":
        records.pop()
    elif corruption == "tokenizer":
        export["input_config"]["tokenizer"]["name"] = "unverified"
    elif corruption == "errors":
        export["error_summary"] = ["request failed"]
    else:
        measured = next(row for row in records if row["metadata"]["benchmark_phase"] == "profiling")
        if corruption == "tokens":
            measured["metrics"]["output_sequence_length"]["value"] = 100
        else:
            measured["metrics"]["time_to_first_token"]["value"] = float("nan")
    with pytest.raises(ValueError):
        PARSE(export, records, 8)


def test_benchmark_completion_does_not_claim_sla_success() -> None:
    raw = fixture()
    for row in raw["records"]:
        row["metrics"]["time_to_first_token"]["value"] = 4000
    result = PARSE(raw["export"], raw["records"], 8)
    assert result["reference_validation"]["valid"] is True
    assert result["eligible"] is False
    assert result["score_output_tokens_per_second"] == 0


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_provenance_distinguishes_server_client_and_prepared_cache(tmp_path: Path, framework: str) -> None:
    runtime = tmp_path / "runtime"
    (runtime / "source").mkdir(parents=True)
    provision = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    commands = "jq -n --arg model_path" + provision.split("jq -n --arg model_path", 1)[1]
    config = ROOT / "configs/firmus-framework.json"
    environment = {**os.environ, "runtime_dir": str(runtime), "model": "/scratch/model", "framework": framework, "config": str(config), "GITHUB_SHA": "a" * 40, "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}
    subprocess.run(["bash", "-euc", commands], env=environment, check=True)
    identity = json.loads((runtime / "source/provenance.json").read_text())
    images = json.loads(config.read_text())["images"]
    assert identity["container"]["sha256"] == images[framework]["sha256"]
    assert identity["client_container"]["sha256"] == images["sglang"]["sha256"]
    assert identity["cache_source"] == ("matched-settings-comparison/home" if framework == "vllm" else None)
    model = json.loads((runtime / "source/model.json").read_text())
    assert model["schema_version"] == "qwen3-pd-model/v1"
    assert model["model"]["path"] == "/scratch/model"
    assert model["model"]["shards"] == 24
