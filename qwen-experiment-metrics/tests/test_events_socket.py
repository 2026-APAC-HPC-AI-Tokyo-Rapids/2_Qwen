from __future__ import annotations

import socket
import struct
import threading
from pathlib import Path

import pytest
from conftest import RUN_ID

from qwen_experiment_metrics.canonical import jcs_bytes, load_json_strict
from qwen_experiment_metrics.events import (
    DomainEventError,
    facts_from_event,
    parse_primary_result,
)
from qwen_experiment_metrics.local_socket import (
    LocalEventFragmentRecorder,
    UnixEventServer,
)


def _configuration_event(sequence: int = 0) -> dict[str, object]:
    return {
        "schema_version": "qwen-domain-event/v1",
        "run_id": RUN_ID,
        "producer_id": "sglang-node-0",
        "node_rank": 0,
        "sequence": sequence,
        "event_type": "bench_args_effective",
        "payload": {
            "schema_version": "config-snapshot/v1",
            "config_type": "BenchArgs",
            "definition": {
                "source_path": "python/sglang/benchmark/offline_throughput.py",
                "class_name": "BenchArgs",
                "source_blob_oid": "a" * 40,
            },
            "fields": [
                {"name": "dataset_name", "declared_type": "str", "value": "sharegpt"},
                {"name": "num_prompts", "declared_type": "int", "value": 32},
            ],
        },
    }


def test_configuration_event_parser_is_pure_and_closed() -> None:
    parsed = facts_from_event(_configuration_event(), expected_run_id=RUN_ID)
    assert parsed.producer_instance == "configuration:node-0"
    assert [fact.key for fact in parsed.facts] == [
        "sglang.node.0.bench_args_effective.dataset_name",
        "sglang.node.0.bench_args_effective.num_prompts",
    ]
    broken = _configuration_event()
    broken["payload"]["fields"].reverse()  # type: ignore[index]
    with pytest.raises(DomainEventError, match="sorted"):
        facts_from_event(broken, expected_run_id=RUN_ID)


def test_primary_result_parser_preserves_null_as_absent() -> None:
    facts = parse_primary_result(
        {
            "schema_version": "primary-result/v1",
            "run_id": RUN_ID,
            "invocation_id": "generation-0",
            "generation_wall_sec": 12.5,
            "returned_count": 32,
            "error_count": 0,
            "observed_input_tokens": None,
            "observed_output_tokens": 8192,
            "ordered_output_token_ids_sha256": "f" * 64,
        },
        expected_run_id=RUN_ID,
    )
    keys = {fact.key for fact in facts}
    assert "perf.benchmark.generation.wall_time_sec" in keys
    assert "result.token.input.total" not in keys
    assert "result.token.output.total" in keys
    assert "result.token.output.ordered_token_ids_sha256" in keys


def test_worker_ready_is_closed_and_bound_to_global_rank() -> None:
    event = {
        "schema_version": "qwen-domain-event/v1",
        "run_id": RUN_ID,
        "producer_id": "sglang-worker-3",
        "node_rank": 0,
        "global_rank": 3,
        "sequence": 0,
        "event_type": "worker_ready",
        "payload": {
            "tp_rank": 3,
            "gpu_id": 3,
            "gpu_uuid": "GPU-3",
            "cpu_affinity": [0, 2],
            "numa_mems": "0",
            "prefill_attention_backend": "flashinfer",
            "decode_attention_backend": "flashinfer",
            "torch_distributed_backend": "nccl",
            "transport_backend": "nccl",
        },
    }
    parsed = facts_from_event(event, expected_run_id=RUN_ID)
    assert parsed.producer_instance == "worker:global-rank-3"
    assert {fact.key for fact in parsed.facts} == {
        "parallel.worker.3.node_rank",
        "parallel.worker.3.tp_rank",
        "parallel.worker.3.gpu_id",
        "parallel.worker.3.gpu_uuid",
        "parallel.worker.3.cpu_affinity",
        "parallel.worker.3.numa_mems",
        "parallel.worker.3.attention.prefill_backend",
        "parallel.worker.3.attention.decode_backend",
        "parallel.worker.3.torch_distributed_backend",
        "parallel.worker.3.transport_backend",
    }
    event["payload"]["tp_rank"] = 2  # type: ignore[index]
    with pytest.raises(DomainEventError, match="global_rank"):
        facts_from_event(event, expected_run_id=RUN_ID)


def test_local_socket_routes_to_atomic_local_fragment(
    manifest_path: Path, runtime: Path, tmp_path: Path
) -> None:
    adapter = LocalEventFragmentRecorder(manifest_path, runtime, node_rank=0)
    socket_path = tmp_path / "events.sock"
    server = UnixEventServer(socket_path, adapter)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    assert server.ready.wait(2)
    payload = jcs_bytes(_configuration_event())
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        client.sendall(struct.pack(">I", len(payload)) + payload)
    server.stop_requested.set()
    thread.join(2)
    paths = adapter.close()
    assert not thread.is_alive()
    assert server.errors == []
    assert len(paths) == 1 and paths[0].exists()
    rows = [load_json_strict(line) for line in paths[0].read_bytes().splitlines()]
    assert len(rows) == 2
    archive = runtime / "source/events/node-0.jsonl"
    assert archive.read_bytes() == payload + b"\n"


def test_local_event_archive_rejects_foreign_node(
    manifest_path: Path, runtime: Path
) -> None:
    adapter = LocalEventFragmentRecorder(manifest_path, runtime, node_rank=1)
    with pytest.raises(ValueError, match="another node"):
        adapter.record_bytes(jcs_bytes(_configuration_event()))
    adapter.abort()
    assert not (runtime / "source/events/node-1.jsonl").exists()


def test_local_recorder_rejects_sequence_gaps(
    manifest_path: Path, runtime: Path
) -> None:
    adapter = LocalEventFragmentRecorder(manifest_path, runtime)
    with pytest.raises(ValueError, match="sequence"):
        adapter.record_bytes(jcs_bytes(_configuration_event(sequence=1)))
    adapter.abort()
