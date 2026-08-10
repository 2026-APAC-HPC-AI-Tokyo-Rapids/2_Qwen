from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from qwen_experiment_metrics.manifest import load_manifest
from qwen_experiment_metrics.orchestration import record_network_sources
from qwen_experiment_metrics.recording import producer_from_fragment

FIXTURES = Path(__file__).parent / "fixtures"
NCCL_COMMIT = "717b68318278e93f371d8ffb46b076069d7c7851"


def _closed_network_capture(runtime: Path) -> tuple[Path, dict[str, Path]]:
    imb_dir = runtime / "source/network/imb-mpi1"
    imb_dir.mkdir(parents=True)
    imb_text = (FIXTURES / "imb-mpi1-complete.txt").read_text()
    imb_text = imb_text[: imb_text.rfind("# Benchmarking PingPong")]
    imb_stdout = imb_dir / "stdout.txt"
    imb_stdout.write_text(imb_text)
    (imb_dir / "stderr.txt").write_text("")
    (imb_dir / "argv.json").write_text(
        json.dumps(["IMB-MPI1", "PingPong", "Sendrecv", "Allreduce"])
    )

    nccl_root = runtime / "source/network/nccl-tests"
    nccl_root.mkdir()
    (nccl_root / "sidecar.json").write_text(
        json.dumps(
            {"schema_version": "nccl-tests-sidecar/v1", "commit": NCCL_COMMIT}
        )
    )
    results = {}
    fixtures = {
        "all_reduce": "nccl-all-reduce.json",
        "all_gather": "nccl-all-gather.json",
        "reduce_scatter": "nccl-reduce-scatter.json",
    }
    for collective, fixture in fixtures.items():
        directory = nccl_root / collective
        directory.mkdir()
        result = directory / "result.json"
        shutil.copyfile(FIXTURES / fixture, result)
        (directory / "stdout.txt").write_text("complete\n")
        (directory / "stderr.txt").write_text("")
        (directory / "argv.json").write_text(
            json.dumps(
                [
                    f"/opt/nccl-tests/bin/{collective}_perf",
                    "-J",
                    "/network/result.json",
                ]
            )
        )
        results[collective] = result
    return imb_stdout, results


def test_network_orchestration_validates_raw_and_records_only_fact_index(
    runtime: Path, manifest_path: Path
) -> None:
    imb_stdout, results = _closed_network_capture(runtime)
    record_network_sources(
        manifest_path,
        runtime,
        imb_stdout=imb_stdout,
        imb_version="2021.3",
        nccl_results=results,
        nccl_tests_commit=NCCL_COMMIT,
        expected_mpi_processes=2,
        expected_gpus_per_process=1,
    )

    fragments = list((runtime / "domain/fragments").glob("*.jsonl"))
    assert [producer_from_fragment(path) for path in fragments] == ["network"]
    assert not (runtime / "source/network/projections").exists()
    assert load_manifest(manifest_path).run_id.encode() in fragments[0].read_bytes()


def test_network_orchestration_rejects_open_or_tampered_capture(
    runtime: Path, manifest_path: Path
) -> None:
    imb_stdout, results = _closed_network_capture(runtime)
    (imb_stdout.parent / "unexpected.txt").write_text("foreign")
    with pytest.raises(ValueError, match="membership"):
        record_network_sources(
            manifest_path,
            runtime,
            imb_stdout=imb_stdout,
            imb_version="2021.3",
            nccl_results=results,
            nccl_tests_commit=NCCL_COMMIT,
            expected_mpi_processes=2,
            expected_gpus_per_process=1,
        )
    assert not list((runtime / "domain/fragments").glob("*.jsonl"))


def test_workflow_and_pjm_enforce_runner_compute_boundary() -> None:
    repository = Path(__file__).parents[2]
    workflow = (repository / ".github/workflows/genkai.yml").read_text()
    pjm = (repository / "scripts/run-qwen35.pjm").read_text()
    node = (repository / "scripts/run-qwen35-node.sh").read_text()

    assert workflow.index("init-run") < workflow.index("pjsub")
    assert workflow.index('"${metrics_cli}" merge') < workflow.index(
        '"${metrics_cli}" package'
    )
    assert workflow.index("Upload fixed v2 artifact set") < workflow.index(
        'qwen-experiment-metrics" publish'
    )
    assert "EXPERIMENT_METRICS_GOOGLE_CREDENTIALS" not in pjm
    assert " publish " not in pjm and " merge " not in pjm
    assert "collect " not in pjm and "monitor-node-gpu" not in node
    assert "serve-events" in node and "sample-gpu" in node
