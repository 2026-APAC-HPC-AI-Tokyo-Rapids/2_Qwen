"""CPU checks for the bounded diagnostic's timing protocol and launch scope."""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]


def diagnostic():
    spec = importlib.util.spec_from_file_location('silu_diagnostic', ROOT / 'scripts/diagnose-silu-quant.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_graph_timing_excludes_setup_and_keeps_replay_count():
    events, operations, replays = [], [], []

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing
            events.append(self)

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 0.25

    class Graph:
        def replay(self):
            replays.append(1)

    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None, CUDAGraph=Graph, graph=lambda _: contextlib.nullcontext(), Event=Event))
    timing, output = diagnostic().graph_timing(torch, lambda: operations.append(1) or 'result')
    assert operations == [1] * 4  # Three JIT/setup calls and one capture.
    assert len(replays) == 40  # Eight warmups plus32 timed graph replays.
    assert len(events) == 64 and timing['median_us'] == 250 and output == 'result'
    assert [s['rows'] for s in diagnostic().shape_specs()] == [8192, 32768, 65536]


@pytest.mark.parametrize('platform,rank,status', [('aspire2ap', '0', 0), ('aspire2ap', '0', 124), ('aspire2ap', '1', 0), ('firmus', '0', 0), ('genkai', '0', 0)])
def test_real_diagnostic_hook_is_bounded_and_failure_is_nonfatal(tmp_path, platform, rank, status):
    source = (ROOT / 'submission/run/submit.sbatch').read_text()
    hook = source.split('  # Isolated synthetic kernel timing before node0 starts any model workers.\n', 1)[1].split('  if [[ "${QWEN_PLATFORM}" == aspire2ap && "$(jq -r', 1)[0]
    for name in ('logs', 'results'):
        (tmp_path / name).mkdir()
    record = tmp_path / 'command.txt'
    harness = 'set -euo pipefail\ntimeout() { printf "%s\\n" "$@" >"${RECORD}"; return "$DIAGNOSTIC_STATUS"; }\n' + hook + '\nprintf "continued\\n"\n'
    env = {'PATH': os.environ['PATH'], 'QWEN_PLATFORM': platform, 'node_rank': rank, 'node_tmp': str(tmp_path / 'node'), 'OUTPUT_DIR': str(tmp_path), 'CONTAINER_IMAGE': '/image.sqsh', 'RECORD': str(record), 'DIAGNOSTIC_STATUS': str(status)}
    result = subprocess.run(['bash', '-c', harness], env=env, capture_output=True, text=True, check=True)
    assert result.stdout == 'continued\n'
    active = platform == 'aspire2ap' and rank == '0'
    assert record.exists() == active
    if active:
        args = record.read_text().splitlines()
        assert args[:3] == ['--signal=TERM', '--kill-after=5s', '120s']
        assert 'NVIDIA_VISIBLE_DEVICES=0' in args and 'CUDA_VISIBLE_DEVICES=0' in args
        assert args[-1] == '/diagnose-silu-quant.py'
    if active and status:
        output = json.loads((tmp_path / 'results/silu-quant-diagnostic.json').read_text())
        assert output['exit_status'] == 124 and output['status'] == 'error_or_timeout'


def test_diagnostic_failure_preserves_json_and_reports_failure(tmp_path, monkeypatch):
    module = diagnostic()
    monkeypatch.setattr(module, 'OUTPUT', tmp_path / 'diagnostic.json')

    def fail(_):
        raise RuntimeError('fixture JIT failure')

    monkeypatch.setattr(module, 'run', fail)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 1
    output = json.loads(module.OUTPUT.read_text())
    assert output['status'] == 'error'
    assert output['error'] == 'RuntimeError: fixture JIT failure'
    assert output['shapes'] == [] and output['elapsed_wall_seconds'] >= 0
