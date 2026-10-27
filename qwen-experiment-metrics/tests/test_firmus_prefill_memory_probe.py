"""Exercise the one-GPU non-counter protocol without CUDA or a scheduler."""
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = 'kernels/probes/prepare-prefill-memory.sh'


@pytest.mark.parametrize('benchmark', [BENCHMARK, 'kernels/probes/prepare-fa3-exp.sh'])
@pytest.mark.parametrize('gate', ['pass', 'fail', 'missing-pass'])
def test_memory_probe_skips_counter_preflight_and_requires_gate(tmp_path, gate, benchmark):
    for name in ('source', 'results', 'bin'):
        (tmp_path / name).mkdir()
    (tmp_path / 'source/provenance.json').write_text(json.dumps({'probe_gpus': 1, 'benchmark': benchmark}))
    probe = tmp_path / 'source/probe'
    probe.write_text('#!/bin/bash\nset -eu\n[[ "$CUDA_VISIBLE_DEVICES" == 0 ]]\necho invoked >> calls\n' + ('echo "PASS exact gate; eligible=False"\n' if gate == 'pass' else 'exit 7\n' if gate == 'fail' else 'echo "eligible=False"\n'))
    probe.chmod(0o755)
    # Counter invocation would fail deterministically if this branch regresses.
    ncu = tmp_path / 'bin/ncu'
    ncu.write_text('#!/bin/bash\necho unexpected-counter >&2\nexit 99\n')
    ncu.chmod(0o755)
    result = subprocess.run(['bash', str(ROOT / 'platforms/firmus/kernel-probe/run-trials.sh')], env={**os.environ, 'RUNTIME_DIR': str(tmp_path)}, capture_output=True, text=True, check=False)
    assert (result.returncode == 0) == (gate == 'pass'), result.stderr
    assert (tmp_path / 'calls').read_text() == 'invoked\n'
    assert (tmp_path / 'run-success').exists() == (gate == 'pass')
    assert 'unexpected-counter' not in result.stderr
    assert not (tmp_path / 'results/ncu-permission.csv').exists()


def test_memory_summary_uses_candidate_name_and_reports_ineligible(tmp_path):
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'results/prefill-memory').mkdir(parents=True)
    (tmp_path / 'logs/provenance.json').write_text(json.dumps({'probe_gpus': 1, 'benchmark': BENCHMARK}))
    (tmp_path / 'results/prefill-memory/results.json').write_text(json.dumps({'candidate_name': 'paired-round', 'candidate_eligible': False, 'timing': [{'rows': 4096, 'median_us': {'control': 10, 'paired-round': 10.1}, 'ratio': 1.01}]}))
    result = subprocess.run(['bash', str(ROOT / 'platforms/firmus/kernel-probe/summary.sh')], env={**os.environ, 'RUNTIME_DIR': str(tmp_path)}, capture_output=True, text=True, check=True)
    assert 'paired-round' in result.stdout
    assert '"candidate_eligible": false' in result.stdout
    assert '"control": 10' in result.stdout
    assert 'NCU permission check 30 seconds' not in result.stdout


def test_existing_probe_still_requires_hardware_counter_permission(tmp_path):
    for name in ('source', 'results', 'bin'):
        (tmp_path / name).mkdir()
    (tmp_path / 'source/provenance.json').write_text(json.dumps({'probe_gpus': 1, 'benchmark': 'kernels/probes/prepare-fa3-prefill.sh', 'calibration': True}))
    probe = tmp_path / 'source/probe'
    probe.write_text('#!/bin/bash\ntouch invoked\necho PASS\n')
    probe.chmod(0o755)
    ncu = tmp_path / 'bin/ncu'
    ncu.write_text('#!/bin/bash\necho ERR_NVGPUCTRPERM\nexit 1\n')
    ncu.chmod(0o755)
    result = subprocess.run(['bash', str(ROOT / 'platforms/firmus/kernel-probe/run-trials.sh')], env={**os.environ, 'RUNTIME_DIR': str(tmp_path)}, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert 'ERR_NVGPUCTRPERM' in result.stderr
    assert not (tmp_path / 'invoked').exists()
    assert not (tmp_path / 'run-success').exists()


def test_memory_prepare_archives_overlay_control_and_generator(tmp_path):
    workspace = tmp_path / 'workspace'
    runtime = tmp_path / 'runtime'
    (workspace / 'kernels/probes').mkdir(parents=True)
    (workspace / 'configs').mkdir()
    header = runtime / 'source/python/sglang/kernels/jit/csrc/elementwise/rope.cuh'
    header.parent.mkdir(parents=True)
    header.write_text('configured overlay control')
    (runtime / 'results').mkdir()
    (workspace / 'kernels/probes/prepare-sglang.sh').write_text('#!/bin/bash\n: "${RUNTIME_DIR:?}"\n')
    for name in ('prefill-memory.py', 'prefill-paired-round.py', 'prepare-prefill-memory.sh'):
        (workspace / 'kernels/probes' / name).write_text(name)
    overlay = workspace / 'overlay.patch'
    overlay.write_text('configured patch')
    (workspace / 'configs/firmus-framework.json').write_text(json.dumps({'sglang_overlay_patches': ['overlay.patch']}))
    subprocess.run(['bash', str(ROOT / BENCHMARK)], env={**os.environ, 'RUNTIME_DIR': str(runtime), 'GITHUB_WORKSPACE': str(workspace), 'GITHUB_SHA': 'fixture-commit'}, capture_output=True, text=True, check=True)
    saved = runtime / 'results/source'
    assert (saved / 'rope-control.cuh').read_text() == header.read_text()
    assert (saved / 'prefill-paired-round.py').read_text() == 'prefill-paired-round.py'
    assert (saved / 'prefill-memory.py').read_text() == 'prefill-memory.py'
    assert not (runtime / 'source/bench.py').exists()
    provenance = json.loads((runtime / 'source/provenance.json').read_text())
    assert provenance['commit'] == 'fixture-commit'
    assert set(provenance['sglang_overlays']) == {'overlay.patch'}
    assert 'rope-control.cuh' in provenance['sources']
    wrapper = (runtime / 'source/probe').read_text()
    assert 'python "${RUNTIME_DIR}/source/prefill-paired-round.py"' in wrapper
    assert '--header "${RUNTIME_DIR}/source/python/sglang/kernels/jit/csrc/elementwise/rope.cuh"' in wrapper
    assert '--output "${RUNTIME_DIR}/results/prefill-memory" --run' in wrapper


def test_fa3_exp_summary_distinguishes_pass_from_eligibility(tmp_path):
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'results/fa3-exp').mkdir(parents=True)
    (tmp_path / 'logs/provenance.json').write_text(json.dumps({'probe_gpus': 1, 'benchmark': 'kernels/probes/prepare-fa3-exp.sh'}))
    (tmp_path / 'results/fa3-exp/results.json').write_text(json.dumps({'candidate_name': 'exp-candidate', 'candidate_eligible': False, 'timing': []}))
    result = subprocess.run(['bash', str(ROOT / 'platforms/firmus/kernel-probe/summary.sh')], env={**os.environ, 'RUNTIME_DIR': str(tmp_path)}, capture_output=True, text=True, check=True)
    assert '8 GPUs allocated' in result.stdout
    assert 'Numerical PASS and performance eligibility are separate' in result.stdout
    assert 'exp-candidate' in result.stdout
    assert '"candidate_eligible": false' in result.stdout


def test_fa3_exp_provision_extracts_compiler_without_counter_toolchain(tmp_path):
    import hashlib

    workspace, binary = tmp_path / 'workspace', tmp_path / 'bin'
    for name in ('configs', 'platforms/firmus/kernel-probe', 'kernels/probes'):
        (workspace / name).mkdir(parents=True)
    binary.mkdir()
    image = tmp_path / 'image.sqsh'
    image.write_bytes(b'x')
    digest = hashlib.sha256(b'x').hexdigest()
    image.with_suffix('.container-sidecar.json').write_text(json.dumps({'container_artifact': {'sha256': digest}}))
    (workspace / 'configs/firmus-framework.json').write_text(json.dumps({'images': {'sglang': {'path': str(image), 'size_bytes': 1, 'sha256': digest}}}))
    for name in ('run-container.sh', 'run-trials.sh', 'run.sbatch'):
        (workspace / 'platforms/firmus/kernel-probe' / name).write_text('#!/bin/bash\n')
    (workspace / 'platforms/firmus/kernel-probe/build-env.sh').write_text('touch "${RUNTIME_DIR}/build-env-called"\nnvcc() { echo mock-nvcc; }\nexport CXX=/usr/bin/true\n')
    (workspace / 'kernels/probes/prepare-fa3-exp.sh').write_text('#!/bin/bash\nset -eu\ntest -f "${RUNTIME_DIR}/build-env-called"\nmkdir -p "${RUNTIME_DIR}/results/source"\necho "{}" > "${RUNTIME_DIR}/source/provenance.json"\nprintf "#!/bin/bash\\necho PASS\\n" > "${RUNTIME_DIR}/source/probe"\nchmod +x "${RUNTIME_DIR}/source/probe"\n')
    for name, body in {'stat': 'echo 1', 'unsquashfs': '[[ "$*" != *nsight-compute* ]] || exit 88\nwhile [[ "$1" != -d ]]; do shift; done\nmkdir -p "$2"'}.items():
        executable = binary / name
        executable.write_text('#!/bin/bash\nset -eu\n'+body+'\n')
        executable.chmod(0o755)
    output = tmp_path / 'github-output'
    env = {**os.environ, 'GITHUB_WORKSPACE': str(workspace), 'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_SHA': 'fixture', 'GITHUB_OUTPUT': str(output), 'RUNTIME_ROOT': str(tmp_path / 'runtime'), 'BENCHMARK': 'kernels/probes/prepare-fa3-exp.sh', 'CANDIDATE_PATCH': '', 'PATH': str(binary)+':'+os.environ['PATH']}
    subprocess.run(['bash', str(ROOT / 'scripts/provision-firmus-kernel-probe.sh')], env=env, capture_output=True, text=True, check=True)
    runtime = Path(output.read_text().strip().split('=', 1)[1])
    assert (runtime / 'logs/extract-cuda.log').exists()
    assert not (runtime / 'logs/extract-ncu.log').exists()
    assert not (runtime / 'source/ncu-permission.cu').exists()
    provenance = json.loads((runtime / 'results/source/provenance.json').read_text())
    assert provenance['allocated_gpus'] == 8
    assert provenance['probe_gpus'] == 1
