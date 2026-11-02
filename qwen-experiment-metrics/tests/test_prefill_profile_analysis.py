"""Real-export schema fixtures: concurrency, causal ambiguity, PP order and coverage."""
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import runpy
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('analyze_prefill_profile', ROOT / 'scripts/analyze_prefill_profile.py')
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)
EPOCH = 1793592000000000000


def fixture(path, layers=(24, 24, 24, 22), count=75):
    """120 ms pair rotations with all-stream overlap, memcpy, late API launch."""
    with sqlite3.connect(path) as db:
        db.executescript((ROOT / 'qwen-experiment-metrics/tests/fixtures/pp4-nsys-schema.sql').read_text())
        db.execute('INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?)', (EPOCH,))
        names = ('sm90_fp8_mega_moe_impl', 'sm90_fp8_gemm_1d2d_impl 9216 4096', 'sm90_fp8_gemm_1d2d_impl 4096 8192', 'qknorm_rope_prefill_fp32_kernel', 'cudaLaunchKernel', 'cudaStreamSynchronize', 'FlashAttnFwdSm90', '_post_norm_moe_input_onepass')
        db.executemany('INSERT INTO StringIds VALUES (?,?)', enumerate(names, 1))
        for stage, layer_count in enumerate(layers):
            db.execute('INSERT INTO TARGET_INFO_GPU VALUES (?,?,?)', (stage, f'0000:{stage+1:02x}:00.0', stage))
            process = (stage+1) << 24
            corr = 0
            for cycle in range(count):
                for layer in range(layer_count):
                    base = cycle*120_000_000 + layer*4_000_000
                    for offset, end, name, stream in ((0, 2_000_000, 1, 0), (2_000_000, 2_500_000, 2, 0), (2_500_000, 3_000_000, 3, 0), (2_100_000, 2_700_000, 4, 1), (3_000_000, 3_500_000, 7, 0), (3_500_000, 3_900_000, 8, 0)):
                        corr += 1
                        a, b = base+offset, base+end
                        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?)', (a, b, stage, 1, stream, corr, process, name, 250, 72, None))
                        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)', (a-100, a-50, process+7, corr, 5))
                a = cycle*120_000_000+layer_count*4_000_000
                corr += 1
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?,?,?,?)', (a, a+1_000_000, stage, 1, 7, corr, process, 64, 8))
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)', (a-100, a-50, process+7, corr, 5))


def runtime_fixture(tmp_path):
    for directory in ('source', 'profile', 'logs'):
        (tmp_path / directory).mkdir(exist_ok=True)
    database = tmp_path / 'fixture.sqlite'
    fixture(database)
    # Row order is PP order; physical IDs can be reordered.
    (tmp_path / 'profile/gpu-map-prefill-0.csv').write_text(''.join(f'GPU-uuid{s}, 00000000:{s+1:02x}:00.0\n' for s in range(4)))
    active = dt.datetime.fromtimestamp(EPOCH/1e9, dt.UTC)
    stop = active+dt.timedelta(seconds=8)
    (tmp_path / 'logs/profile-capture.log').write_text(f'capture_active utc={active.isoformat()}\nstop_requested utc={stop.isoformat()}\n')
    with (tmp_path / 'logs/gpu-telemetry-fixture.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(('timestamp', 'index', 'uuid', 'power', 'sm', 'mem', 'throttle', 'temp'))
        for tick in range(90):
            for stage in range(4):
                stamp = active+dt.timedelta(seconds=tick/10)
                writer.writerow((stamp.strftime('%Y/%m/%d %H:%M:%S.%f'), stage, f'GPU-uuid{stage}', '700 W', '1500 MHz', '3200 MHz', '0x4', '70'))
    return database


def test_pair_analysis_runtime_covers_all_stages_and_preserves_accounting(tmp_path):
    database = runtime_fixture(tmp_path)
    original = database.read_bytes()
    summary = ANALYSIS.validate_runtime(tmp_path, 'prefill-0', database)
    assert summary['devices_in_pp_order'] == [0, 1, 2, 3]
    window = summary['window_ns']
    samples = ANALYSIS.telemetry_samples([tmp_path / 'logs/gpu-telemetry-fixture.csv'], tmp_path / 'profile/gpu-map-prefill-0.csv', EPOCH, 0)
    result = ANALYSIS.analyze(database, window, samples=samples)
    for stage in result['stages'].values():
        assert stage['period_us']['median'] == 120000
        for cycle in stage['cycles']:
            assert cycle['kernel_union_us']+cycle['copy_only_us']+cycle['gpu_idle_us'] == pytest.approx(cycle['period_us'])
            assert sum(cycle['exclusive_busy_us'].values()) == pytest.approx(cycle['kernel_union_us']+cycle['copy_only_us'])
            assert cycle['summed_kernel_us']['writer/pack'] > 0
            assert cycle['exclusive_busy_us']['overlap'] > 0
            assert cycle['power_w_median'] == 700 and cycle['sm_clock_mhz_median'] == 1500
            assert cycle['launch_api_matched_kernels'] == cycle['kernel_count']
            assert cycle['gate_ratios']['MegaMoE']['ratio_range'] == pytest.approx([2000/2500, 2000/2334])
    assert database.read_bytes() == original
    markdown = ANALYSIS.markdown(result)
    assert '120.000' in markdown and '1500.00' in markdown
    assert summary['coverage']['0']['nvml_sample_interval_ms']['median'] > 0


def test_missing_samples_are_null_and_incomplete_cycles_fail(tmp_path):
    database = tmp_path / 'data.sqlite'
    fixture(database, count=10)
    result = ANALYSIS.analyze(database)
    assert result['stages']['0']['cycles'][0]['power_w_median'] is None
    with sqlite3.connect(database) as db:
        db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=0 AND demangledName=2 AND start BETWEEN 300000000 AND 700000000')
    with pytest.raises(ValueError, match='dense calls'):
        ANALYSIS.analyze(database)


def test_lightweight_validation_rejects_missing_copy_telemetry_or_launch(tmp_path):
    database = runtime_fixture(tmp_path)
    with sqlite3.connect(database) as db:
        db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE deviceId=2')
    with pytest.raises(ValueError, match='missing memcpy'):
        ANALYSIS.validate_runtime(tmp_path, 'prefill-0', database)
    fixture2 = tmp_path / 'fixture2.sqlite'
    fixture(fixture2)
    (tmp_path / 'logs/gpu-telemetry-fixture.csv').unlink()
    with pytest.raises(ValueError, match='NVML'):
        ANALYSIS.validate_runtime(tmp_path, 'prefill-0', fixture2)
    runtime_fixture_directory = tmp_path / 'second'
    runtime_fixture_directory.mkdir()
    database = runtime_fixture(runtime_fixture_directory)
    with sqlite3.connect(database) as db:
        db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE globalTid=?', ((3 << 24)+7,))
    with pytest.raises(ValueError, match='launch API'):
        ANALYSIS.validate_runtime(runtime_fixture_directory, 'prefill-0', database)


def test_process_scoped_correlations_and_pp_mapping(tmp_path):
    database = tmp_path / 'data.sqlite'
    fixture(database, count=10)
    with sqlite3.connect(database) as db:
        db.row_factory = sqlite3.Row
        kernels, _, _ = ANALYSIS.load(db)
        assert all(k['api']['globalPid'] == k['globalPid'] for k in kernels)
        mapping = tmp_path / 'gpu-map.csv'
        mapping.write_text(''.join(f'GPU-uuid{i}, 00000000:{i+1:02x}:00.0\n' for i in (2, 0, 3, 1)))
        assert ANALYSIS.mapped_devices(db, mapping) == [2, 0, 3, 1]


def test_gap_classification_is_evidence_not_causal_assertion():
    def kernel(a, b, name, api=None):
        return {'start': a, 'end': b, 'name': name, 'kind': 'kernel', 'globalPid': 1, 'category': 'other', 'api': api}
    api = {'name': 'cudaLaunchKernel', 'start': 15, 'end': 16, 'globalPid': 1, 'correlationId': 7}
    ks = [kernel(0, 10, 'a'), kernel(20, 30, 'b', api)]
    out = ANALYSIS.account(0, 30, ks, [], [], [])
    gap = out['gaps'][0]
    assert gap['before_kernel'] == 'a' and gap['after_kernel'] == 'b'
    assert gap['cause'] == 'late launch candidate' and gap['pre_api_idle_us'] == .005
    sync = dict(api, name='cudaStreamSynchronize', start=8, end=19)
    assert ANALYSIS.account(0, 30, ks, [], [sync], [])['gaps'][0]['cause'] == 'synchronization wait candidate'
    copy = dict(ks[1], kind='copy', start=18, end=20, category='PP send/recv or copy')
    assert ANALYSIS.account(0, 30, ks, [copy], [], [])['gaps'][0]['cause'] == 'transfer wait candidate'


def test_tail_categories_preserve_transfer_and_split_gemm():
    rows = [{'name': n, 'category': ANALYSIS.category(n)} for n in ('nvjet_sm90_tst_512x8', 'FlashAttnFwdSm90', 'ncclDevKernel_SendRecv', 'sm90_fp8_gemm_1d2d_impl 9216 4096', '_post_norm_moe_input_onepass')]
    ANALYSIS.tail_categories(rows, 3)
    assert [r['category'] for r in rows] == ['lm_head/draft', 'lm_head/draft', 'PP send/recv or copy', 'QKV/O GEMM', 'MoE input']


def test_lightweight_validator_exports_analyzes_and_removes_only_sqlite(tmp_path, monkeypatch):
    database = runtime_fixture(tmp_path)
    for worker in ('prefill-1', 'prefill-2'):
        shutil.copyfile(tmp_path / 'profile/gpu-map-prefill-0.csv', tmp_path / f'profile/gpu-map-{worker}.csv')
    workers = ('prefill-0', 'prefill-1', 'prefill-2', 'decode-0')
    (tmp_path / 'source/worker-endpoints.tsv').write_text(''.join(f'{worker}\thost\t30000\t4\n' for worker in workers))
    (tmp_path / 'source/firmus-framework.json').write_text(json.dumps({'framework': 'sglang', 'workload': 'official', 'profile_only': True, 'prefill': {'pp': 4}}))
    for worker in workers:
        (tmp_path / f'profile/nsys-{worker}.nsys-rep').write_text('native retained')
    def export(command, check):
        assert check
        shutil.copyfile(database, command[3].removeprefix('--output='))
    monkeypatch.setattr(subprocess, 'run', export)
    monkeypatch.setitem(sys.modules, 'analyze_prefill_profile', ANALYSIS)
    monkeypatch.setattr(sys, 'argv', ['validate', str(tmp_path)])
    runpy.run_path(str(ROOT / 'scripts/validate-firmus-profile.py'))['main']()
    validation = json.loads((tmp_path / 'profile/validation.json').read_text())
    assert sum(w['gpu_count'] for w in validation['workers'].values()) == 16
    assert all('steady_capture' in validation['workers'][worker] for worker in workers[:3])
    assert not list((tmp_path / 'profile').glob('pair-analysis-*.json'))
    assert len(list((tmp_path / 'profile').glob('*.nsys-rep'))) == 4
    assert not list((tmp_path / 'profile').glob('*.sqlite'))
    assert not (tmp_path / 'profile/torch').exists()


@pytest.mark.parametrize('profile,interval', [('true', '--loop-ms=100'), ('false', '--loop=1')])
def test_telemetry_shell_keeps_utc_and_sampling_frequency(tmp_path, profile, interval):
    source = (ROOT / 'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    block = 'telemetry_interval=' + source.split('telemetry_interval=', 1)[1].split('\ndone', 1)[0] + '\ndone\nwait\n'
    prelude = f'''set -euo pipefail
PROFILE={profile}
HEAD=head
TAIL=tail
ART={tmp_path}
telemetry_pids=()
srun() {{ printf '%s\\n' "$*"; }}
'''
    subprocess.run(['bash', '-c', prelude + block], check=True)
    for host in ('head', 'tail'):
        command = (tmp_path / f'gpu-telemetry-{host}.csv').read_text()
        assert 'env TZ=UTC nvidia-smi' in command
        assert interval in command and 'power.draw,clocks.sm' in command


def test_profile_configuration_keeps_image_gates_and_serving_except_profile_only():
    # Baseline 599c093 fingerprints, independent of CI shallow checkout history.
    current = json.loads((ROOT / 'configs/firmus-prefill-pp4.json').read_text())
    assert current.pop('profile_only') is True
    assert hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest() == '2f32a350466695f927337d4950bb46e35357048d3628e0c21ffec63852898d9f'
    source = (ROOT / 'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    start, stop = '# Default off. The pair-only FA3 gate', '# Decode greedy gate end.'
    gate = source.split(start, 1)[1].split(stop, 1)[0]
    assert hashlib.sha256(gate.encode()).hexdigest() == '3231015be921fe9f116d27f3cf4711673222c7e9ecead7920bc68f8d0f8d3579'
