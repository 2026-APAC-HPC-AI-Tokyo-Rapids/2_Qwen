from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/collect-eagle-acceptance.py"


@pytest.mark.parametrize(("requested_quantization", "reported_quantization"), [(None, None), ("fp8", "fp8"), ("fp8", None)])
def test_collects_dp_state_and_log_statistics(tmp_path: Path, requested_quantization, reported_quantization) -> None:
    config = tmp_path / "genkai-serving.json"
    config.write_text(
        json.dumps(
            {
                "speculation": {
                    "algorithm": "EAGLE3",
                    "steps": 3,
                    "draft_tokens": 4,
                    "draft_quantization": requested_quantization,
                    "draft_repository": "fixture/draft",
                    "draft_revision": "fixture-revision",
                }
            }
        )
    )
    endpoints = tmp_path / "worker-endpoints.tsv"
    endpoints.write_text(
        "prefill\t192.0.2.1\t30100\t9000\ndecode\t192.0.2.2\t30200\tunused\n"
    )
    server_info = tmp_path / "server-info-decode-30200.json"
    server_info.write_text(
        json.dumps(
            {
                "speculative_algorithm": "EAGLE3",
                "speculative_draft_model_quantization": reported_quantization,
                "internal_states": [
                    {"avg_spec_accept_length": 2.2},
                    {"avg_spec_accept_length": 2.4},
                ],
            }
        )
    )
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "decode-0.log").write_text(
        "[fixture] Decode batch, accept len: 2.20, accept rate: 0.40\n"
        "[fixture] Decode batch, accept len: 2.40, accept rate: 0.50\n"
    )
    output = tmp_path / "eagle-acceptance.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config),
            "--endpoints",
            str(endpoints),
            "--server-info-dir",
            str(tmp_path),
            "--log-dir",
            str(logs),
            "--output",
            str(output),
            "--cluster",
            "genkai",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if requested_quantization != reported_quantization:
        assert completed.returncode == 1 and "requested draft quantization was not applied" in completed.stderr
        assert not output.exists()
        return
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["speculation"]["draft_quantization"] == requested_quantization
    assert result["route"]["stages"][-1] == "EAGLE3 verification"
    assert result["speculation"]["tokenizer_backend"] == "huggingface"
    assert result["statistics"]["server_info_state_count"] == 2
    assert result["statistics"]["server_info_accepted_draft_length"]["mean"] == 1.3
    assert result["statistics"]["decode_log_samples"]["accept_rate"]["count"] == 2


def test_requires_gigatoken_on_every_decode_state(tmp_path: Path) -> None:
    config = tmp_path / "genkai-serving.json"
    config.write_text(json.dumps({"tokenizer_backend": "gigatoken", "speculation": {"algorithm": "EAGLE3"}}))
    endpoints = tmp_path / "worker-endpoints.tsv"
    endpoints.write_text("decode\t192.0.2.2\t30200\tunused\n")
    (tmp_path / "server-info-decode-30200.json").write_text(json.dumps({"speculative_algorithm": "EAGLE3", "tokenizer_backend": "gigatoken", "internal_states": [{"tokenizer_backend": "gigatoken", "avg_spec_accept_length": 2.0}]}))
    logs = tmp_path / "logs"
    logs.mkdir()
    output = tmp_path / "eagle-acceptance.json"
    subprocess.run([sys.executable, str(SCRIPT), "--config", str(config), "--endpoints", str(endpoints), "--server-info-dir", str(tmp_path), "--log-dir", str(logs), "--output", str(output), "--cluster", "genkai"], check=True)
    assert json.loads(output.read_text())["decode_workers"][0]["tokenizer_backend"] == "gigatoken"
    (tmp_path / "server-info-decode-30200.json").write_text(json.dumps({"speculative_algorithm": "EAGLE3", "tokenizer_backend": "huggingface", "internal_states": [{"tokenizer_backend": "huggingface", "avg_spec_accept_length": 2.0}]}))
    assert subprocess.run([sys.executable, str(SCRIPT), "--config", str(config), "--endpoints", str(endpoints), "--server-info-dir", str(tmp_path), "--log-dir", str(logs), "--output", str(output), "--cluster", "genkai"], check=False).returncode == 1


@pytest.mark.parametrize("invalid_state", [None, 0, 1, 2, 3, 4])
def test_firmus_pp4_gigatoken_acceptance(tmp_path: Path, invalid_state: int | None) -> None:
    config = ROOT / "configs/firmus-prefill-pp4.json"
    assert json.loads(config.read_text())["tokenizer_backend"] == "gigatoken"
    endpoints = tmp_path / "worker-endpoints.tsv"
    endpoints.write_text("prefill-0\t192.0.2.1\t30100\t9000\nprefill-1\t192.0.2.1\t30101\t9001\nprefill-2\t192.0.2.2\t30102\t9002\ndecode-0\t192.0.2.2\t30200\t4\n")
    info = {"speculative_algorithm": "EAGLE3", "speculative_draft_model_quantization": "fp8", "tokenizer_backend": "gigatoken", "internal_states": [{"tokenizer_backend": "gigatoken", "avg_spec_accept_length": 2.0} for _ in range(4)]}
    if invalid_state is not None:
        [info, *info["internal_states"]][invalid_state].pop("tokenizer_backend")
    # Firmus names server-info files by worker, without the endpoint port.
    (tmp_path / "server-info-decode-0.json").write_text(json.dumps(info))
    output = tmp_path / "eagle-acceptance.json"
    result = subprocess.run([sys.executable, str(SCRIPT), "--cluster", "firmus", "--config", str(config), "--endpoints", str(endpoints), "--server-info-dir", str(tmp_path), "--log-dir", str(tmp_path), "--output", str(output)], capture_output=True, text=True, check=False)
    if invalid_state is not None:
        assert result.returncode == 1
        assert "server tokenizer_backend is not gigatoken" in result.stderr
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        report = json.loads(output.read_text())
        assert report["speculation"]["tokenizer_backend"] == "gigatoken"
        assert report["decode_workers"][0]["role"] == "decode-0"
        assert report["decode_workers"][0]["tokenizer_backend"] == "gigatoken"
        assert report["statistics"]["server_info_state_count"] == 4
