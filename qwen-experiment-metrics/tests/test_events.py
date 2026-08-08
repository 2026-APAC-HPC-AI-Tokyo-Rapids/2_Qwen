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
                "server_info": {
                    "attention_backend": "flashinfer",
                    "internal_states": {"memory_usage": 0.5},
                },
                "model_config": {
                    "model_type": "qwen3_5_moe",
                    "quantization_config": {
                        "quant_method": "fp8",
                        "modules_to_not_convert": ["model.visual"],
                    },
                },
                "workload": {
                    "request_count": 1,
                    "input_tokens": [100],
                    "requested_output_tokens": [40],
                },
            },
        }
    )
    values = {metric.key: metric.value for metric in metrics}
    assert values["sglang.benchmark_arg.num_prompts"] == "2"
    assert values["sglang.server_arg.attention_backend"] == "flashinfer"
    assert not any(key.startswith("sglang.runtime.") for key in values)
    assert not any(key.startswith("model.config.") for key in values)
    assert not any(key.startswith("sglang.workload.") for key in values)


def test_result_event_mapping() -> None:
    metrics = event_metrics(
        {
            "type": "sglang.benchmark_result",
            "payload": {
                "backend": "engine",
                "successful_requests": 2,
                "total_input_tokens": 100,
                "total_output_tokens": 40,
            },
        }
    )
    values = {metric.key: metric.value for metric in metrics}
    assert values["result.benchmark.backend"] == "engine"
    assert values["result.token.input.total"] == "100"
    assert values["result.token.output.total"] == "40"
    assert not any(key.startswith("result.output") for key in values)


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
