"""Producer WRITE completion must release its lease without D receive state."""

import hashlib
import json
import runpy
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATCH = runpy.run_path(str(ROOT / "containers/fix-vllm-prefill-handoff.py"))
FIXTURE = Path(__file__).parent / "fixtures/vllm-push-completion.py.txt"


def test_original_native_guard_reproduces_missing_producer_completion():
    with pytest.raises(AssertionError, match="producer_done"):
        PATCH["validate_push_release"](FIXTURE.read_bytes())


def test_repaired_method_covers_success_pending_failure_expiry_and_receiver(monkeypatch):
    source = FIXTURE.read_bytes()
    patch = PATCH["patch_push_worker"]
    monkeypatch.setitem(patch.__globals__, "PUSH_SOURCE_SHA256", hashlib.sha256(source).hexdigest())
    repaired = patch(source)
    result = PATCH["validate_push_release"](repaired)
    assert result["passed"] and result["cases"] == 7
    assert "if req_id in self._reqs_to_send" in repaired.decode()
    assert "done_pushing - failed_pushing" in repaired.decode()


def test_push_installer_rejects_unreviewed_source():
    with pytest.raises(ValueError, match="Unsupported pinned.*push worker"):
        PATCH["patch_push_worker"](b"unreviewed source")


def test_push_installer_rejects_changed_anchor(monkeypatch):
    source = FIXTURE.read_bytes().replace(b"if req_id in self._recving_metadata", b"if True")
    patch = PATCH["patch_push_worker"]
    monkeypatch.setitem(patch.__globals__, "PUSH_SOURCE_SHA256", hashlib.sha256(source).hexdigest())
    with pytest.raises(ValueError, match="Nonunique producer WRITE"):
        patch(source)


def test_combined_installer_validates_push_before_writing_engine(tmp_path, monkeypatch):
    engine = tmp_path / "core.py"
    push = tmp_path / "push_worker.py"
    original = b"engine must remain unchanged"
    engine.write_bytes(original)
    push.write_bytes(b"unreviewed push source")
    monkeypatch.setitem(PATCH["install"].__globals__, "patch", lambda source: source + b"modified")
    with pytest.raises(ValueError, match="Unsupported pinned.*push worker"):
        PATCH["install"](engine, push)
    assert engine.read_bytes() == original


@pytest.mark.parametrize("metadata", [None, {}, {"push_worker_source_sha256": "wrong", "push_release_cpu_contract": {"passed": True}}, {"push_worker_source_sha256": PATCH["PUSH_SOURCE_SHA256"], "push_release_cpu_contract": {"passed": False}}])
def test_worker_rejects_images_without_verified_push_release(tmp_path, metadata):
    source = (ROOT / "platforms/firmus/framework-comparison/vllm.sbatch").read_text()
    worker = source.split("<<'EOF'\n", 1)[1].split("\nEOF", 1)[0]
    init = worker.split("role=${name%%-*}", 1)[0]
    report = tmp_path / "repair.json"
    if metadata is not None:
        report.write_text(json.dumps(metadata))
    init = init.replace("/opt/apac/prefill-handoff-repair.json", str(report))
    result = subprocess.run(["bash", "-c", init, "worker.sh", "prefill-0", "30100", "/model", "1", "192.0.2.1"], check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "FATAL: serving image lacks" in result.stdout + result.stderr
