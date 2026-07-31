import json
import socket
import struct

from qwen_experiment_metrics.stage import EventReceiver, event_metrics, fallback_result


def test_configuration_event() -> None:
    metrics = event_metrics(
        {
            "type": "sglang.configuration",
            "payload": {
                "bench_args": {"num_prompts": 2},
                "server_args": {"attention_backend": "flashinfer"},
            },
        }
    )
    values = {metric.key: metric.value for metric in metrics}
    assert values["sglang.benchmark_arg.num_prompts"] == "2"
    assert values["sglang.server_arg.attention_backend"] == "flashinfer"


def test_result_event_mapping() -> None:
    metrics = event_metrics(
        {
            "type": "sglang.benchmark_result",
            "payload": {
                "backend": "engine",
                "successful_requests": 2,
                "output_hashes": {
                    "request_count": 1,
                    "request": [{"text_sha256": "def", "token_count": 4}],
                    "manifest_sha256": "abc",
                },
            },
        }
    )
    values = {metric.key: metric.value for metric in metrics}
    assert values["result.benchmark.backend"] == "engine"
    assert values["result.output.manifest_sha256"] == "abc"
    assert values["result.output.0.text_sha256"] == "def"
    assert values["result.output.0.token_count"] == "4"


def test_event_receiver_sends_batch(tmp_path, monkeypatch) -> None:
    sent = []
    monkeypatch.setattr(
        "qwen_experiment_metrics.stage.send",
        lambda run, batch: sent.append((run, batch)) or True,
    )
    path = tmp_path / "events.sock"
    receiver = EventReceiver(path, "qwen-test")
    receiver.start()
    assert receiver.ready.wait(2)
    event = {
        "type": "sglang.benchmark_result",
        "payload": {"backend": "engine", "successful_requests": 1},
    }
    payload = json.dumps(event).encode()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
        stream.connect(str(path))
        stream.sendall(struct.pack(">I", len(payload)) + payload)
    for _ in range(100):
        if sent:
            break
        receiver.stop_requested.wait(0.01)
    receiver.stop()
    receiver.join(2)
    assert sent[0][0] == "qwen-test"
    assert "sglang.benchmark_result" in receiver.received


def test_result_file_fallback(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text('{"backend":"engine","successful_requests":3}\n')
    values = {metric.key: metric.value for metric in fallback_result(path)}
    assert values["result.benchmark.backend"] == "engine"
    assert values["result.benchmark.successful_requests"] == "3"
    assert values["perf.benchmark.generation.wall_time_sec"] == ""
