"""CPU checks of profile-only Prefill capture launch and terminal validation."""
import json
import os
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SBATCH = ROOT / 'platforms/firmus/framework-comparison/sglang.sbatch'


def test_worker_capture_is_prefill_only_and_inherited_values_are_cleared(tmp_path):
    source = SBATCH.read_text().split("<<'EOF'\n", 1)[1].split('\nEOF', 1)[0]
    # Exercise the actual worker environment block without GPU-dependent startup.
    source = source[:source.index('# Prefill: a 3994-token prompt')]
    source = source.replace('/opt/sglang/venv/bin/python', shutil.which('python3'))
    (tmp_path / 'source').mkdir()
    worker = tmp_path / 'source/worker.sh'
    worker.write_text(source + '\nprintf "%s|%s" "${SGLANG_PREFILL_ROUTING_DIR:-unset}" "${SGLANG_PREFILL_ROUTING_SAMPLES:-unset}"\n')
    for samples, role, expected in [(0, 'prefill', 'unset|unset'), (8, 'decode', 'unset|unset'), (8, 'prefill', f'{tmp_path}/results/prefill-routing/prefill-0|8')]:
        (tmp_path / 'source/firmus-framework.json').write_text(json.dumps({'prefill': {'routing_capture_samples': samples}}))
        result = subprocess.run(['bash', str(worker), f'{role}-0', '30100', 'model', '4', '127.0.0.1'], env={**os.environ, 'SGLANG_PREFILL_ROUTING_DIR': 'inherited', 'SGLANG_PREFILL_ROUTING_SAMPLES': '99'}, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert result.stdout == expected
    assert not list(tmp_path.rglob('active'))


@pytest.mark.skipif(shutil.which('jq') is None, reason='jq required by deployment scripts')
@pytest.mark.parametrize('summary,expected', [(None, 1), ({'schema': 'prefill-routing-summary/v1', 'valid': False, 'expected_workers': 3, 'expected_ranks': 4, 'samples_per_rank': 8}, 1), ({'schema': 'prefill-routing-summary/v1', 'valid': True, 'expected_workers': 3, 'expected_ranks': 4, 'samples_per_rank': 7}, 1), ({'schema': 'prefill-routing-summary/v1', 'valid': True, 'expected_workers': 3, 'expected_ranks': 4, 'samples_per_rank': 8}, 0)])
def test_terminal_check_requires_configured_capture_summary(tmp_path, summary, expected):
    for name in ('logs', 'results', 'source', 'bin'):
        (tmp_path / name).mkdir()
    (tmp_path / 'source/firmus-framework.json').write_text(json.dumps({'framework': 'sglang', 'prefill': {'routing_capture_samples': 8}}))
    for name in ('profile_export_aiperf.json', 'score.json'):
        (tmp_path / 'results' / name).write_text('{}')
    (tmp_path / 'run-success').touch()
    (tmp_path / 'logs/slurm-123.log').touch()
    for name, body in [('squeue', 'exit 0'), ('sacct', "echo '123|COMPLETED|0:0|00:01|2'")]:
        command = tmp_path / 'bin' / name
        command.write_text('#!/bin/bash\n' + body + '\n')
        command.chmod(0o755)
    if summary is not None:
        (tmp_path / 'results/prefill-routing-summary.json').write_text(json.dumps(summary))
    result = subprocess.run(['bash', str(ROOT / 'platforms/firmus/check-qwen-job.sh')], env={**os.environ, 'PATH': f"{tmp_path}/bin:{os.environ['PATH']}", 'RUNTIME_DIR': str(tmp_path), 'SLURM_JOB_ID': '123'}, capture_output=True, text=True, check=False)
    assert (result.returncode != 0) == bool(expected), result.stderr


def test_marker_lifetime_and_summary_use_separate_profile_pass():
    source = SBATCH.read_text()
    benchmark = source.index('benchmark_status=0')
    activate = source.index('do touch "${ROOT}/results/prefill-routing/${worker}/active"')
    profile = source.index('"${ROOT}/scripts/capture-firmus-profile.sh"')
    aggregate = source.index('"${ROOT}/scripts/summarize-prefill-routing.py"')
    assert benchmark < activate < profile < aggregate
    assert '(( benchmark_status == 0 )) || exit "${benchmark_status}"' in source[benchmark:activate]
    assert 'do rm -f "${ROOT}/results/prefill-routing/${worker}/active"' in source[source.index('cleanup_all()'):benchmark]
    assert '--samples-per-rank "${routing_capture_samples}"' in source


def test_actions_summary_identifies_diagnostic_and_normal_score(tmp_path):
    (tmp_path / 'firmus-framework.json').write_text(json.dumps({'prefill': {'routing_capture_samples': 8, 'workers': 3, 'tp': 4, 'dp': 4, 'ep': 4}}))
    (tmp_path / 'prefill-routing-summary.json').write_text(json.dumps({'valid': True, 'expected_workers': 3, 'expected_ranks': 4, 'samples_per_rank': 8}))
    summarize = runpy.run_path(str(ROOT / 'scripts/summarize-firmus-run.py'))['summarize']
    report = summarize(tmp_path, '123', 'completed')
    assert 'routing diagnostic: **valid**; 3 workers × 4 ranks × 8 samples' in report
    assert 'do not use those timings for performance adoption' in report
    assert 'score above remains from the normal benchmark pass' in report


@pytest.mark.skipif(shutil.which('jq') is None, reason='jq required by deployment scripts')
@pytest.mark.parametrize('samples', [0, 8])
def test_capture_overlay_is_appended_only_to_enabled_runtime_config(tmp_path, samples):
    patch = 'patches/sglang/prefill-routing-capture.patch'
    config = tmp_path / 'firmus-framework.json'
    config.write_text(json.dumps({'sglang_overlay_patches': ['base.patch', patch, 'last.patch', patch]}))
    source = (ROOT / 'scripts/provision-firmus-framework.sh').read_text()
    block = source.split('config="${runtime_dir}/source/firmus-framework.json"\n', 1)[1].split('if [[ "${framework}"', 1)[0]
    subprocess.run(['bash', '-euc', block], env={**os.environ, 'config': str(config), 'routing_capture_samples': str(samples)}, check=True)
    patches = json.loads(config.read_text())['sglang_overlay_patches']
    assert patches == (['base.patch', 'last.patch', patch] if samples else ['base.patch', patch, 'last.patch', patch])
