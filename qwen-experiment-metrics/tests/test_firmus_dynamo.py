from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("role,tp,port", [("prefill", 2, 30100), ("decode", 4, 30200)])
def test_dynamo_worker_launch_uses_native_pd_backend(tmp_path: Path, backend: str, role: str, tp: int, port: int) -> None:
    (tmp_path / "source").mkdir()
    stub = tmp_path / "python3"
    stub.write_text('#!/bin/bash\nprintf "%s\\0" "$@" >"$ARGV_RECORD"\n')
    stub.chmod(0o755)
    worker = (ROOT / "platforms/firmus/dynamo/worker.sh").read_text()
    launch = worker[worker.index('if [[ "${framework}" == vllm') :]
    script = tmp_path / "worker.sh"
    script.write_text(f'set -euo pipefail\nframework={backend}\nrole={role}\nname={role}-0\nport={port}\nmodel=/model\ntp={tp}\nruntime_dir="$1"\n' + launch)
    subprocess.run(["bash", str(script), str(tmp_path)], env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "ARGV_RECORD": str(tmp_path / "argv")}, check=True)
    argv = (tmp_path / "argv").read_bytes().decode().rstrip("\0").split("\0")
    assert argv[:2] == ["-m", f"dynamo.{backend}"]
    assert argv[argv.index("--disaggregation-mode") + 1] == role
    assert argv[argv.index("--served-model-name") + 1] == "qwen"
    if backend == "vllm":
        assert argv[argv.index("--tensor-parallel-size") + 1] == str(tp)
        assert json.loads(argv[argv.index("--kv-transfer-config") + 1]) == {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
        assert ("--enable-expert-parallel" in argv) is (role == "decode")
    else:
        assert argv[argv.index("--tp-size") + 1] == str(tp)
        assert argv[argv.index("--disaggregation-transfer-backend") + 1] == "nixl"
        assert argv[argv.index("--disaggregation-bootstrap-port") + 1] == str(port - 21100)
        assert ("--ep-size" in argv) is (role == "decode")
    assert shlex.split((tmp_path / f"source/{role}-0.argv").read_text()) == ["python3", *argv]
    assert "sglang.launch_server" not in argv and "serve" not in argv


def test_dynamo_run_tag_does_not_also_launch_reference() -> None:
    workflow = (ROOT / ".github/workflows/qwen.yml").read_text()
    native = workflow.split("  firmus:\n", 1)[1].split("  firmus-dynamo:\n", 1)[0]
    assert "!contains(github.event.head_commit.message, '[dynamo]')" in native
    assert "needs: [genkai, aspire2ap, firmus, firmus-dynamo, firmus-dynamo-recover, cutlass-grid]" in workflow
    assert "uses: ./.github/workflows/firmus-dynamo.yml" in workflow
