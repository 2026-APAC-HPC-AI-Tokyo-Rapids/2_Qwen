from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_runtime_has_no_merge_projection_or_metrics_sidecar() -> None:
    package = ROOT / "qwen-experiment-metrics/src/qwen_experiment_metrics"
    for name in (
        "merge.py",
        "projections.py",
        "projection_router.py",
        "physical_publish.py",
    ):
        assert not (package / name).exists()
    workflow = (ROOT / ".github/workflows/genkai.yml").read_text(encoding="utf-8")
    assert "metrics.jsonl" not in workflow
    assert '"${metrics_cli}" merge' not in workflow
    assert 'qwen-experiment-metrics" publish' not in workflow
