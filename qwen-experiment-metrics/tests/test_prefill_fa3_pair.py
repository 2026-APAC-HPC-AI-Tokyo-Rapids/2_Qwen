"""CPU contracts for pair dispatch, all-twelve adoption and timeout wiring."""

import ast
import json
import os
import runpy
import shlex
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/select-firmus-prefill-fa3-pair.py'
GATE = runpy.run_path(str(SCRIPT))
PIN = GATE['KEY'] + '=' + GATE['CONTROL'] + '\nDG_SM90_PREFILL_4STAGE=1\n'


@pytest.fixture
def dispatch(tmp_path):
    # Apply the real module overlays, then execute its two routing functions with
    # the actual candidate JSON. No Torch/device API is available to them.
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    for name in ('prefill-fp8-integration.patch', 'prefill-fa3-pair.patch'):
        subprocess.run(['git', 'apply', '--include=python/sglang/srt/layers/attention/prefill_fp8.py', str(ROOT / 'patches/sglang' / name)], cwd=tmp_path, check=True)
    text = (tmp_path / 'python/sglang/srt/layers/attention/prefill_fp8.py').read_text()
    program = ast.parse(text)
    program.body = [n for n in program.body if isinstance(n, ast.FunctionDef) and n.name in ('prefill_candidate', 'pair_candidate', 'attention')]
    class Table:
        def __init__(self, path):
            pass
        def read_text(self):
            return (ROOT / 'kernels/fa3-verify/prefill-candidates.json').read_text()
    scope = {'json': json, 'os': os, 'lru_cache': lru_cache, 'Path': Table}
    exec(compile(program, '<real dispatch>', 'exec'), scope)  # noqa: S102 - real routing functions, no CUDA imports
    return scope


@pytest.mark.parametrize('lengths,expected', [
    ((3994, 3994), True), ((3900, 4096), True), ((4096, 3900), True),
    ((3994,), False), ((3994, 3994, 3994), False), ((3899, 4096), False),
    ((3900, 4097), False), ((), False), (None, False), ((7800,), False),
])
def test_host_pair_dispatch_bounds(dispatch, monkeypatch, lengths, expected):
    monkeypatch.setenv(GATE['KEY'], GATE['DISPATCH'])
    selected = dispatch['pair_candidate'](lengths)
    assert selected['name'] == (GATE['CANDIDATE'] if expected else GATE['CONTROL'])
    assert dispatch['prefill_candidate']()['name'] == GATE['CONTROL']
    # Alternation cannot evict the control table and turn singles into pairs.
    assert dispatch['pair_candidate']((3994, 3994))['name'] == GATE['CANDIDATE']
    assert dispatch['pair_candidate']((3994,))['name'] == GATE['CONTROL']


def test_disabled_pair_value_keeps_control(dispatch, monkeypatch):
    monkeypatch.setenv(GATE['KEY'], GATE['CONTROL'])
    assert dispatch['pair_candidate']((3994, 3994))['name'] == GATE['CONTROL']


def numeric():
    return {'numeric_passed': True, 'changed_input_replays': 2, 'precision_reference': GATE['CONTROL'], 'metrics': [{'relative_l2': .01, 'max_scaled_error': .02}]*3, 'precision_reference_metrics': [{'relative_l2': .01, 'max_scaled_error': .02}]*3}


@pytest.fixture
def screen(tmp_path):
    env = tmp_path / 'prefill.env'
    env.write_text(PIN)
    now = time.time()
    for worker in range(3):
        directory = tmp_path / f'prefill-{worker}'
        directory.mkdir()
        for rank in range(4):
            record = {'schema': GATE['SCHEMA'], 'run_id': 'current', 'worker': f'prefill-{worker}', 'rank': rank, 'device': 'NVIDIA H200', 'uuid': f'{worker}-{rank}', 'finished_at': now, 'elapsed_seconds': 60, 'passed': True,
                          'shapes': {'+'.join(map(str, shape)): {'numeric': numeric(), 'samples_us': {order: {'control': [800.]*GATE['ROUNDS'], 'candidate': [795.]*GATE['ROUNDS']} for order in ('forward', 'reverse')}} for shape in GATE['SHAPES']},
                          'endurance': {'calls': 1200, 'bytes_equal': True, 'direct_equal': True, 'initial_fp32': {'relative_l2': .01, 'max_scaled_error': .02}, 'final_fp32': {'relative_l2': .01, 'max_scaled_error': .02}}}
            (directory / f'rank-{rank}.json').write_text(json.dumps(record))
    return tmp_path, env, now + 100


def aggregate(screen, **kwargs):
    directory, env, deadline = screen
    return GATE['aggregate'](directory, env, kwargs.get('run_id', 'current'), kwargs.get('deadline', deadline), kwargs.get('workers_ok', True))


def edit(screen, change):
    path = screen[0] / 'prefill-2/rank-3.json'
    record = json.loads(path.read_text())
    change(record)
    path.write_text(json.dumps(record))


def test_exact_five_us_adopts_only_after_twelve_pass(screen):
    result = aggregate(screen)
    assert result['adopted'] and len(result['gpus']) == 12
    assert screen[1].read_text() == PIN.replace(GATE['CONTROL'], GATE['DISPATCH'])


@pytest.mark.parametrize('case', ['missing', 'malformed', 'stale', 'rank', 'uuid', 'late', 'elapsed', 'failed', 'shape', 'numeric', 'replay', 'metrics', 'timing', 'reverse', 'nan', 'inf', 'calls', 'bytes', 'direct', 'endurance_numeric'])
def test_one_bad_cell_keeps_exact_env(screen, case):
    path = screen[0] / 'prefill-2/rank-3.json'
    if case == 'missing':
        path.unlink()
    elif case == 'malformed':
        path.write_text('{')
    else:
        def mutate(r):
            shape = r['shapes']['3994+3994']
            if case == 'stale': r['run_id'] = 'old'
            elif case == 'rank': r['rank'] = 1
            elif case == 'uuid': r['uuid'] = '0-0'
            elif case == 'late': r['finished_at'] = screen[2] + 1
            elif case == 'elapsed': r['elapsed_seconds'] = 150
            elif case == 'failed': r['passed'] = False
            elif case == 'shape': r['shapes'].pop('3900+4096')
            elif case == 'numeric': shape['numeric']['metrics'][0]['relative_l2'] = .05001
            elif case == 'replay': shape['numeric']['changed_input_replays'] = 1
            elif case == 'metrics': shape['numeric']['precision_reference_metrics'].pop()
            elif case == 'timing': shape['samples_us']['forward']['control'].pop()
            elif case == 'reverse': shape['samples_us']['reverse']['candidate'] = [795.001]*GATE['ROUNDS']
            elif case in ('nan', 'inf'): shape['samples_us']['forward']['candidate'][0] = float(case)
            elif case == 'calls': r['endurance']['calls'] = 1199
            elif case == 'bytes': r['endurance']['bytes_equal'] = False
            elif case == 'direct': r['endurance']['direct_equal'] = False
            elif case == 'endurance_numeric': r['endurance']['final_fp32']['max_scaled_error'] = .251
        edit(screen, mutate)
    result = aggregate(screen)
    assert not result['adopted'] and screen[1].read_text() == PIN
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('kwargs', [{'workers_ok': False}, {'deadline': 0}, {'run_id': 'other'}])
def test_failed_timeout_or_stale_process_keeps_pin(screen, kwargs):
    assert not aggregate(screen, **kwargs)['adopted']
    assert screen[1].read_text() == PIN


def test_publication_failure_is_atomic(screen, monkeypatch):
    def fail(*args):
        raise OSError('rename failure')
    monkeypatch.setattr(Path, 'replace', fail)
    assert not aggregate(screen)['adopted'] and screen[1].read_text() == PIN
    assert not list(screen[0].glob('*.tmp'))


def block():
    shell = (ROOT / 'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    return '# Default off. The pair-only FA3' + shell.split('# Default off. The pair-only FA3', 1)[1].split('\n# Default off. Native V', 1)[0].split('# P is revalidated; optional P3', 1)[0]


def test_wiring_and_native_build_inputs():
    config = json.loads((ROOT / 'configs/firmus-prefill-pp4.json').read_text())
    assert config['prefill']['fa3_pair_gate'] is True  # round 12 serves with the gate on; the sbatch default stays off
    shell = block()
    assert '.prefill.fa3_pair_gate // false' in shell
    assert 'for worker in 0 1 2; do' in shell and '--nproc-per-node=4' in shell
    assert 'timeout --kill-after=2 150 srun' in shell
    assert 'wait "${pid}" || prefill_fa3_pair_ok=0' in shell
    assert '--env-output "${ROOT}/source/prefill-kernels.env"' in shell
    provision = (ROOT / 'scripts/provision-firmus-framework.sh').read_text()
    assert '"${GITHUB_WORKSPACE}/scripts/select-firmus-prefill-fa3-pair.py"' in provision
    assert 'cp "${runtime_dir}/scripts/select-firmus-prefill-fa3-pair.py" "${runtime_dir}/source/"' in provision
    build = (ROOT / 'kernels/fa3-verify/build.py').read_text()
    assert 'str(root / "prefill-geometry.patch")' in build
    assert 'generate_prefill_dispatch(root, work)' in build
    assert 'cp -r "${GITHUB_WORKSPACE}/kernels/fa3-verify"' in (ROOT / 'scripts/build-sglang-container.sh').read_text()
    assert 'COPY containers/fa3-verify /opt/apac/fa3-verify' in (ROOT / 'containers/Dockerfile').read_text()
    assert 'cp -a "${GITHUB_WORKSPACE}" "${input_root}/qwen"' in (ROOT / '.github/workflows/aws-container.yml').read_text()
    assert config['images']['sglang']['sha256'] == config['container_sha256']
    assert config['container_sha256'].startswith('5018d322')


def test_timed_out_workers_continue_to_control_serving(tmp_path):
    (tmp_path / 'source').mkdir()
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'source/pp4-kernel-settings.json').write_text('{}')
    (tmp_path / 'source/prefill-kernels.env').write_text(PIN)
    (tmp_path / 'scripts' / SCRIPT.name).write_text(SCRIPT.read_text())
    preamble = f'''set -euo pipefail
ROOT={shlex.quote(str(tmp_path))}
ART="$ROOT/logs"
prefill_pp=4
HEAD=head
TAIL=tail
SLURM_JOB_ID=fixture
SGLANG=unused
jq() {{ printf 'true\\n'; }}
mounts() {{ printf 'unused\\n'; }}
timeout() {{ return 124; }}
uv() {{ shift 5; {shlex.quote(sys.executable)} "$@"; }}
'''
    subprocess.run(['bash', '-c', preamble + block() + '\nprintf continued >"$ROOT/continued"\n'], check=True, capture_output=True, text=True)
    assert (tmp_path / 'continued').read_text() == 'continued'
    assert (tmp_path / 'source/prefill-kernels.env').read_text() == PIN
    assert not json.loads((tmp_path / 'results/prefill-fa3-pair/selection.json').read_text())['adopted']


@pytest.mark.parametrize('fixed,lengths,expected', [(True, (3994, 3994), 'candidate'), (True, (3994,), 'control'), (True, (3899, 4096), 'control'), (True, (3994, 3994, 3994), 'control'), (False, (3994, 3994), 'prepared')])
def test_attention_calls_the_selected_native_row_without_reading_cu(dispatch, monkeypatch, fixed, lengths, expected):
    monkeypatch.setenv(GATE['KEY'], GATE['DISPATCH'])
    class Tensor:
        shape = (sum(lengths), 64, 128)
        device = 'fixture'
        @property
        def T(self):
            return self
        def __getitem__(self, key):
            return self
        def view(self, *args):
            return self
    calls = []
    ops = SimpleNamespace(prefill_variant_fwd_out=lambda *args: calls.append(args[-1]), prefill_fwd_out=lambda *args: calls.append('prepared'))
    dispatch['attention'].__globals__['torch'] = SimpleNamespace(empty=lambda *args, **kwargs: Tensor(), float8_e4m3fn='fp8', float32='fp32', int32='i32', uint8='u8', ops=SimpleNamespace(apac_fa3_verify=ops))
    monkeypatch.setitem(sys.modules, 'sglang.srt.layers.attention.decode_fa3_verify', SimpleNamespace(load=lambda: None))
    # A bare object cannot expose a CUDA tensor's contents or synchronize.
    dispatch['attention'](Tensor(), Tensor(), Tensor(), object(), max(lengths), 128**-.5, sum(lengths), fixed=fixed, lengths=lengths)
    rows = json.loads((ROOT / 'kernels/fa3-verify/prefill-candidates.json').read_text())
    wanted = 'prepared' if expected == 'prepared' else next(i for i, row in enumerate(rows) if row['name'] == GATE[expected.upper()])
    assert calls == [wanted]


@pytest.mark.parametrize('enabled', [False, True])
def test_prepare_keeps_control_and_records_optional_gate(tmp_path, enabled):
    config = json.loads((ROOT / 'configs/firmus-prefill-pp4.json').read_text())
    config['prefill']['fa3_pair_gate'] = enabled
    config['prefill']['fa3_prepack_gate'] = enabled
    config['prefill']['fa3_v3_gate'] = enabled
    config['prefill']['small_bundle_r37_gate'] = enabled
    config['decode_greedy_verify_gate'] = enabled
    for name in ('source', 'results'):
        (tmp_path / name).mkdir()
    runpy.run_path(str(ROOT / 'scripts/prepare-firmus-pp4-settings.py'))['prepare'](config, ROOT, tmp_path)
    env = (tmp_path / 'source/prefill-kernels.env').read_text()
    assert GATE['KEY'] + '=' + GATE['CONTROL'] + '\n' in env
    record = json.loads((tmp_path / 'source/pp4-kernel-settings.json').read_text())
    assert record['no_gpu_selection_rerun'] is not enabled
    assert record['fa3_prepack_gate'] is enabled and record['fa3_v3_gate'] is enabled
    assert record['decode_greedy_verify_gate'] is enabled
    assert 'megamoe_fc1_first_k_gate' not in record
    assert record['pp4_selected']['DG_SM90_PREFILL_FC1_FIRST_K'] == '1'
    assert record['pp4_selected_runs']['DG_SM90_PREFILL_FC1_FIRST_K'] == [41767886175]
    assert 'SGLANG_DECODE_GREEDY_VERIFY=0\n' in (tmp_path/'source/decode-kernels.env').read_text()



def test_timing_retains_precapture_buffers_for_both_graphs():
    import weakref

    class Step:
        pass
    class Graph:
        def __init__(self, step):
            self.step = weakref.ref(step)
        def replay(self):
            assert self.step() is not None, 'captured output/metadata storage was released'
    class Event:
        def __init__(self, **kwargs):
            pass
        def record(self):
            pass
        def synchronize(self):
            pass
        def elapsed_time(self, other):
            return 1.0
    checks = {'prefill_selection_step': lambda *args: Step(), 'prefill_selection_capture': Graph}
    samples = GATE['measure'](SimpleNamespace(cuda=SimpleNamespace(Event=Event)), checks, {})
    assert all(len(values) == GATE['ROUNDS'] for modes in samples.values() for values in modes.values())



def test_endurance_checks_column_major_scale_bytes_and_1200_calls(monkeypatch):
    torch = pytest.importorskip('torch')
    output = torch.zeros((4, 64, 128), dtype=torch.float8_e4m3fn)
    scales = torch.ones((64, 4)).T  # The serving scale contract has stride(-1) != 1.
    assert scales.stride(-1) != 1
    calls = []
    def attention(*args, **kwargs):
        assert kwargs['fixed'] is True and kwargs['lengths'] == (3994, 3994)
        calls.append(None)
        return output.view(4, 8192).clone(), scales.clone()
    def prefill_candidate():
        pass
    prefill_candidate.cache_clear = lambda: None
    module = SimpleNamespace(attention=attention, pair_candidate=lambda lengths: {'name': GATE['CANDIDATE']}, prefill_candidate=prefill_candidate)
    monkeypatch.setitem(sys.modules, 'sglang.srt.layers.attention.prefill_fp8', module)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setenv(GATE['KEY'], GATE['CONTROL'])
    def direct():
        pass
    direct.output = output, scales
    def numeric(actual, reference, label):
        torch.testing.assert_close(actual, reference)
        return {'relative_l2': 0., 'max_scaled_error': 0.}
    checks = {'prefill_selection_inputs': lambda *args, **kwargs: {'q': None, 'k': None, 'v': None, 'cu': None, 'rows': 4, 'lengths': (3994, 3994)}, 'prefill_selection_step': lambda *args: direct, 'prefill_fp32_reference': lambda common: ([0], torch.zeros((1, 64, 128))), 'prefill_assert_errors': numeric}
    result = GATE['endurance'](torch, checks)
    assert result['calls'] == 1200 and len(calls) == 1201
    assert result['direct_equal'] and result['bytes_equal']
