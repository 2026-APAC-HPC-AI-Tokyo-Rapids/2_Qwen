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
        json.dumps({"schema_version": "nccl-tests-sidecar/v1", "commit": NCCL_COMMIT})
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
                    "mpirun",
                    "-n",
                    "2",
                    "bash",
                    "json_args=(-J /network/result.json)",
                    f"/opt/nccl-tests/bin/{collective}_perf",
                    "-g",
                    "1",
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
    aspire = (repository / ".github/workflows/aspire2ap.yml").read_text()
    pjm = (repository / "platforms/genkai/check-qwen-container.pjm").read_text()
    pbs = (repository / "platforms/aspire2ap/check-qwen-container.pbs").read_text()
    check = (repository / "scripts/run-qwen-check-env.sh").read_text()

    assert "enroot version" in check
    assert "enroot start" in check
    assert "-m sglang.check_env" in check
    assert "provision-qwen-check-env.sh" in workflow
    assert "provision-qwen-check-env.sh" in aspire
    assert "configure-aws-credentials" not in workflow
    assert "configure-aws-credentials" not in aspire
    assert "send-qwen-fragments.sh" not in workflow
    assert "send-qwen-fragments.sh" not in aspire
    assert "GOOGLE_CREDENTIAL" not in pjm
    assert "GOOGLE_CREDENTIAL" not in pbs
    assert "AWS_ACCESS_KEY" not in pjm and "AWS_ACCESS_KEY" not in pbs
