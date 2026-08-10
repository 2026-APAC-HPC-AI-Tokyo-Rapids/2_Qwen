from __future__ import annotations

from pathlib import Path

from qwen_experiment_metrics.canonical import jcs_bytes, load_json_strict
from qwen_experiment_metrics.domain import Fact
from qwen_experiment_metrics.manifest import load_manifest, manifest_facts
from qwen_experiment_metrics.merge import merge_run
from qwen_experiment_metrics.recording import Recorder


def _record_runner(manifest_path: Path, runtime: Path) -> None:
    with Recorder.open(manifest_path, None, "runner", runtime) as recorder:
        recorder.record_many(manifest_facts(load_manifest(manifest_path)))


def test_merge_dedupes_identical_rows_but_requires_execution_graph(
    manifest_path: Path, runtime: Path
) -> None:
    manifest = load_manifest(manifest_path)
    with Recorder.open(manifest_path, None, "runner", runtime) as recorder:
        for fact in manifest_facts(manifest):
            recorder.record(fact)
            recorder.record(fact)
    outcome = merge_run(manifest_path, runtime)
    assert outcome.collection_complete is False
    rows = [
        load_json_strict(line) for line in outcome.output_path.read_bytes().splitlines()
    ]
    assert all(list(row) == ["run", "key", "value"] for row in rows)
    assert len(rows) == 5
    assert {row["key"]: row["value"] for row in rows}["collection_complete"] == "false"


def test_merge_preserves_conflicting_distinct_rows_without_choosing_one(
    manifest_path: Path, runtime: Path
) -> None:
    _record_runner(manifest_path, runtime)
    with Recorder.open(manifest_path, None, "stage:benchmark", runtime) as recorder:
        recorder.record(
            Fact(
                "run.stage.benchmark.launch",
                {"argv": ["true"], "cwd": "/tmp", "stderr_mode": "inherit"},
            )
        )
        recorder.record(
            Fact("run.stage.benchmark.termination", {"kind": "exit", "code": 0})
        )
        recorder.record(Fact("perf.stage.benchmark.wall_time_sec", 1.0))
        recorder.record(Fact("perf.stage.benchmark.wall_time_sec", 2.0))
    outcome = merge_run(manifest_path, runtime)
    assert outcome.collection_complete is False
    assert outcome.conflicting_keys == ("perf.stage.benchmark.wall_time_sec",)
    rows = [
        load_json_strict(line) for line in outcome.output_path.read_bytes().splitlines()
    ]
    conflict_rows = [
        row for row in rows if row["key"] == "perf.stage.benchmark.wall_time_sec"
    ]
    assert {row["value"] for row in conflict_rows} == {"1", "2"}
    assert list((runtime / "domain/fragments").glob("*.jsonl"))


def test_merge_marks_missing_expected_producer_incomplete(
    manifest_path: Path, runtime: Path
) -> None:
    _record_runner(manifest_path, runtime)
    outcome = merge_run(manifest_path, runtime)
    assert outcome.collection_complete is False
    assert outcome.missing_producers == (
        "configuration:node-0",
        "configuration:node-1",
        "result",
        "stage:benchmark",
    )


def test_merge_marks_missing_required_fact_in_completed_fragment(
    manifest_path: Path, runtime: Path
) -> None:
    _record_runner(manifest_path, runtime)
    with Recorder.open(manifest_path, None, "result", runtime) as recorder:
        recorder.record(Fact("result.request.returned_count", 3))
    outcome = merge_run(manifest_path, runtime)
    assert outcome.collection_complete is False
    assert set(outcome.missing_required_facts) == {
        "result:error_count",
        "result:generation_wall",
    }


def test_merge_revalidates_untrusted_encoded_fragment_values(
    manifest_path: Path, runtime: Path
) -> None:
    _record_runner(manifest_path, runtime)
    with Recorder.open(manifest_path, None, "stage:benchmark", runtime) as recorder:
        recorder.record(
            Fact(
                "run.stage.benchmark.launch",
                {"argv": ["true"], "cwd": "/tmp", "stderr_mode": "inherit"},
            )
        )
        recorder.record(
            Fact("run.stage.benchmark.termination", {"kind": "exit", "code": 0})
        )
        recorder.record(Fact("perf.stage.benchmark.wall_time_sec", 1))
        fragment = recorder.completed_path
    rows = [load_json_strict(line) for line in fragment.read_bytes().splitlines()]
    for row in rows:
        if row["key"] == "perf.stage.benchmark.wall_time_sec":
            row["value"] = "01"
    fragment.write_bytes(b"".join(jcs_bytes(row) + b"\n" for row in rows))

    outcome = merge_run(manifest_path, runtime)
    assert outcome.collection_complete is False
    assert len(outcome.invalid_fragments) == 1
    assert fragment.exists(), "invalid evidence must remain available to the run bundle"
