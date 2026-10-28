"""Exercise bootstrap refresh failures without installing host packages."""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("failures,updates,success", [(0, 1, True), (1, 2, True), (2, 2, False)])
def test_bootstrap_refresh_is_bounded_and_gates_install(tmp_path, failures, updates, success):
    source = (ROOT / "aws/ec2-user-data.sh").read_text()
    block = source[source.index("export DEBIAN_FRONTEND="):source.index("curl --fail")]
    apt = tmp_path / "apt-get"
    apt.write_text('''#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >>"$RECORD"
if [[ "$*" == *" update" ]]; then
  count=0
  if [[ -f "$COUNT" ]]; then read -r count <"$COUNT"; fi
  count=$((count + 1))
  printf '%s\n' "$count" >"$COUNT"
  if ((count <= FAILURES)); then exit 100; fi
fi
''')
    sleep = tmp_path / "sleep"
    sleep.write_text('#!/bin/bash\nprintf "sleep %s\\n" "$*" >>"$RECORD"\n')
    for command in (apt, sleep):
        command.chmod(0o755)
    record = tmp_path / "record"
    result = subprocess.run(["bash", "-euc", block], env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "COUNT": str(tmp_path / "count"), "RECORD": str(record), "FAILURES": str(failures)}, check=False)
    assert (result.returncode == 0) is success
    calls = record.read_text().splitlines()
    assert calls.count("-o Acquire::By-Hash=force -o APT::Update::Error-Mode=any update") == updates
    assert calls.count("sleep 30") == (updates - 1)
    assert sum(call.startswith("install -y ") for call in calls) == int(success)
