from pathlib import Path


def test_workflow_and_scheduler_adapters_enforce_runner_compute_boundary() -> None:
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
