import base64
import json
import os

from qwen_experiment_metrics import core


def test_credentials_are_removed_from_environment(monkeypatch) -> None:
    encoded = base64.b64encode(json.dumps({"project_id": "test"}).encode()).decode()
    monkeypatch.setenv(core.BASE64_ENV, encoded)
    monkeypatch.setattr(core, "_captured", None)
    core.capture_credentials()
    assert core.BASE64_ENV not in os.environ
    assert core._captured is not None
    assert core._captured.info == {"project_id": "test"}
