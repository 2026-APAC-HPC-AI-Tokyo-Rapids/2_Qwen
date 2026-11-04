"""CPU coverage for the fail-closed all-rank Prefill performance rule."""
import copy
import json
import runpy
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def selector():
    return runpy.run_path(str(ROOT / 'scripts/select-firmus-prefill-fa3.py'))


def records(selector, names=None):
    names = names or selector['NAMES'][1:]
    keys = dict.fromkeys(key for name in names for key in selector['shapes'](name))
    return [{'rank': rank, 'lengths': {key: {name: {'numeric_passed': True, 'control': [350.0]*20, 'candidate': [344.0]*20} for name in names if key in selector['shapes'](name)} for key in keys}} for rank in range(4)]


def test_no_winner_when_one_rank_or_tail_misses_threshold(selector):
    ranks = records(selector)
    for name in selector['NAMES'][1:]:
        ranks[3]['lengths']['4027'][name]['candidate'] = [345.01]*20
    assert selector['choose_candidate'](ranks)['winner'] == 'control'
    assert selector['choose_candidate'](ranks)['env'] == {'SGLANG_PREFILL_FA3_VERIFY': 'control'}


def test_exact_boundary_and_numeric_failure(selector):
    ranks = records(selector)
    for rank in ranks:
        for row in rank['lengths'].values():
            for entry in row.values():
                entry['candidate'] = [345.0]*20
    assert selector['choose_candidate'](ranks)['winner'] == 'regs'
    last = selector['NAMES'][-1]
    for name in selector['NAMES'][1:-1]:
        ranks[1]['lengths']['3986'][name]['numeric_passed'] = False
    assert selector['choose_candidate'](ranks)['winner'] == last
    ranks[0]['lengths']['4096'][last]['candidate'].pop()
    assert selector['choose_candidate'](ranks)['winner'] == 'control'


def test_rejected_candidate_does_not_hide_another_winner(selector):
    ranks = records(selector)
    ranks[0]['lengths']['4096']['regs']['error'] = 'byte mismatch'
    original = copy.deepcopy(ranks)
    assert selector['choose_candidate'](ranks)['winner'] == 'c2'
    assert ranks == original


def test_prefill_dispatch_is_independent_of_decode(tmp_path):
    ns = runpy.run_path(str(ROOT / 'kernels/fa3-verify/build.py'))
    tasks = ns['generate_prefill_dispatch'](ROOT / 'kernels/fa3-verify', tmp_path)
    rows = json.loads((ROOT / 'kernels/fa3-verify/prefill-candidates.json').read_text())
    assert len(tasks) == sum(not r.get('retired') and r['name'] != 'control' for r in rows)
    assert {r["name"] for r in rows if r["variant"] & 8} == set()
    assert {r['name'] for r in rows if r['byte_image'] and not r.get('native_v_n')} == {'c2', 'regs-c2'}
    assert (tmp_path / 'prefill-candidates.h').read_text().count('prefill_fixed') == 2
    assert all('prefill_variant.cu' in p.read_text() for p, _, _ in tasks)


def test_generated_worker_checks_only_prefill_and_uses_assigned_gpus(tmp_path):
    shell = (ROOT / 'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    worker = shell.split('cat >"${ROOT}/source/worker.sh" <<\'EOF\'\n', 1)[1].split('\nEOF', 1)[0]
    path = tmp_path / 'worker.sh'
    path.write_text(worker)
    subprocess.run(['bash', '-n', str(path)], check=True)
    assert 'if [[ "$role" == prefill ]]; then set -a; source "$(dirname "$0")/prefill-kernels.env"; set +a; fi' in worker
    # Only the PP4 Prefill stage order permutes the GPUs Slurm assigned (test_pp4_gpu_order.py).
    assert [line.strip() for line in worker.splitlines() if 'CUDA_VISIBLE_DEVICES=' in line] == ['export CUDA_VISIBLE_DEVICES="${gpu_order}"']
    assert shell.index('select-firmus-prefill-kernels.sh') < shell.index('bash "${ROOT}/source/worker.sh" prefill-0')
    assert 'for worker in 0 1 2; do' in shell and '--aggregate' in shell


@pytest.mark.parametrize("workers", [2, 3])
def test_aggregation_requires_every_worker_and_every_rank(selector, tmp_path, workers):
    out, env = tmp_path / 'selection.json', tmp_path / 'prefill.env'
    for worker in range(workers):
        path = tmp_path / f'prefill-{worker}'
        path.mkdir()
        ranks = records(selector)
        (path / 'status.txt').write_text('fa3=0\n')
        if worker == workers - 1:
            for name in selector['NAMES'][1:]:
                ranks[3]['lengths']['4096'][name]['candidate'] = [345.1]*20
        (path / 'fa3.json').write_text(json.dumps({'ranks': ranks}))
    selector['aggregate_selection'](tmp_path, out, env, workers)
    assert json.loads(out.read_text())['winner'] == 'control'
    assert json.loads(out.read_text())['rule']['gpus'] == workers * 4
    (tmp_path / f'prefill-{workers - 1}/status.txt').write_text('fa3=124\n')
    selector['aggregate_selection'](tmp_path, out, env, workers)
    assert json.loads(out.read_text())['failed_workers'] == [workers - 1]
    assert env.read_text() == 'SGLANG_PREFILL_FA3_VERIFY=control\n'


def test_pp4_screen_times_varlen_rows_on_the_request_pair(selector):
    names = ['n192-varlen', 'n192-direct-o-varlen']
    assert selector['shapes']('n192') == ['3986', '4027', '4096']
    assert selector['shapes']('n192-varlen') == ['3986', '4027', '4096', '3986+4027']
    ranks = records(selector, names)
    assert set(ranks[0]['lengths']['4096']) == set(names)
    result = selector['choose_candidate'](ranks)
    assert result['rule']['screened'] == names and result['winner'] == 'n192-varlen'
    for rank in ranks:
        rank['lengths']['3986+4027']['n192-direct-o-varlen']['candidate'] = [345.5]*20
        rank['lengths']['3986']['n192-direct-o-varlen']['candidate'] = [300.0]*20
    assert selector['choose_candidate'](ranks)['winner'] == 'n192-varlen'  # a slow pair alone disqualifies
    del ranks[2]['lengths']['3986+4027']['n192-varlen']
    assert selector['choose_candidate'](ranks)['winner'] == 'control'


def test_pp4_aggregate_keeps_the_megamoe_switch_and_the_pinned_row(selector, tmp_path):
    # PP4: prefill-kernels.env holds the pinned row and the MegaMoE winner before the FA3 aggregate.
    names = ['n192-varlen', 'n192-direct-o-varlen']
    out, env = tmp_path / 'selection.json', tmp_path / 'prefill.env'
    for worker in range(3):
        path = tmp_path / f'prefill-{worker}'
        path.mkdir()
        (path / 'status.txt').write_text('fa3=0\n')
        (path / 'fa3.json').write_text(json.dumps({'ranks': records(selector, names)}))
    env.write_text('SGLANG_PREFILL_FA3_VERIFY=n192\nDG_SM90_PREFILL_4STAGE=1\n')
    selector['aggregate_selection'](tmp_path, out, env)
    assert env.read_text() == 'DG_SM90_PREFILL_4STAGE=1\nSGLANG_PREFILL_FA3_VERIFY=n192-varlen\n'
    env.write_text('SGLANG_PREFILL_FA3_VERIFY=n192\nDG_SM90_PREFILL_4STAGE=1\n')
    (tmp_path / 'prefill-1/status.txt').write_text('fa3=124\n')
    selector['aggregate_selection'](tmp_path, out, env)
    assert json.loads(out.read_text())['winner'] == 'control'
    assert env.read_text() == 'DG_SM90_PREFILL_4STAGE=1\nSGLANG_PREFILL_FA3_VERIFY=n192\n'


def test_pp4_pins_the_repeated_selections_instead_of_screening(tmp_path):
    shell = (ROOT / 'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    config = json.loads((ROOT / 'configs/firmus-prefill-pp4.json').read_text())
    rows = {r['name']: r for r in json.loads((ROOT / 'kernels/fa3-verify/prefill-candidates.json').read_text())}
    prepare = runpy.run_path(str(ROOT / 'scripts/prepare-firmus-pp4-settings.py'))
    # The fixed pipe row passed its gate and served in 41040348302; no V byte image, as fp32 requires.
    row = rows[prepare['PP4_SELECTED']['SGLANG_PREFILL_FA3_VERIFY']]
    assert row['name'] == 'n192-direct-o-varlen-pipe' and row.get('varlen') and not row['byte_image']
    assert 'prefill_fa3_screen' not in config
    for sub in ('source', 'results'):
        (tmp_path / sub).mkdir()
    prepare['prepare'](config, ROOT, tmp_path)
    # Settled pins, including round 34 FC1 and round 38 dense first-K; shell quoting decoded, last assignment wins.
    expected = {
        'prefill': {
            'SGLANG_PREFILL_MOE_INPUT': 'onepass',
            'SGLANG_PREFILL_QKNORM': 'fp32',
            'DG_SM90_PREFILL_4STAGE': '1',
            'DG_SM90_PREFILL_LEAN': '1',
            'DG_SM90_PREFILL_SPLIT_ACCUM': '1',
            'DG_SM90_PREFILL_FC1_FIRST_K': '1',
            'DG_SM90_PREFILL_DENSE_FIRST_K': '1',
            'SGLANG_PREFILL_QKNORM_FIXED': '0',
            'SGLANG_PREFILL_ONEPASS_DIV448': '0',
            'SGLANG_PREFILL_FA3_VERIFY': 'n192-direct-o-varlen-pipe',
            'SGLANG_PP4_PROXY_TRANSPORT': 'copy',
        },
        'decode': {
            'SGLANG_DECODE_FA3_VERIFY': 'p128split2-s3-fp8',
            'DG_SM90_1D2D_DECODE_CONFIG': 'QKV:128,144,5,1,0;QKV>128:192,80,5,1,0',
            'SGLANG_DECODE_DENSE_GEMM_QKV': '',
            'SGLANG_DECODE_DENSE_GEMM_O': '',
            'SGLANG_DECODE_NORM_QUANT': 'wide',
            'SGLANG_DECODE_MOE_INPUT': 'fused',
            'SGLANG_DECODE_GREEDY_VERIFY': '0',
        },
    }
    for role, values in expected.items():
        lines = (tmp_path / f'source/{role}-kernels.env').read_text().splitlines()
        assert dict(shlex.split(line)[0].split('=', 1) for line in lines) == values
    settings = json.loads((tmp_path / 'source/pp4-kernel-settings.json').read_text())
    assert settings['pp4_selected_runs']['SGLANG_PREFILL_FA3_VERIFY'] == [41040348302] and settings['environment']['SGLANG_PREFILL_MOE_INPUT'] == 'onepass'
    # The one-pass MoE input won in 40964714831 alone (13.6% on every GPU, about ten times the noise).
    assert settings['pp4_selected_runs']['SGLANG_PREFILL_MOE_INPUT'] == [40964714831]
    # The FP32 Q/K norm writer won in 40911121619 and 40925558339.
    assert settings['pp4_selected_runs']['SGLANG_PREFILL_QKNORM'] == [40911121619, 40925558339] and set(settings['pp4_selected_runs']) == set(settings['pp4_selected'])
    # After every PP4 overlay; only prefill-moe-input (disjoint files), fp32 Q/K norm, onepass MoE input and Decode MoE input follow it.
    overlays=config['sglang_overlay_patches']
    start=overlays.index('patches/sglang/prefill-fa3-varlen.patch')
    assert overlays[start:start+6] == ['patches/sglang/prefill-fa3-varlen.patch', 'patches/sglang/prefill-fa3-pair.patch', 'patches/sglang/prefill-moe-input.patch', 'patches/sglang/prefill-qknorm-fp32.patch', 'patches/sglang/prefill-moe-input-onepass.patch', 'patches/sglang/decode-moe-input.patch']
    # The twelve-GPU screen runs only outside PP4 and without a reused selection.
    assert 'prefill_screen' not in shell
    assert 'if [[ ! -s "${ROOT}/source/reused-kernel-selection.json" && ! -s "${ROOT}/source/pp4-kernel-settings.json" ]]; then' in shell
    pp4 = shell[shell.index('if [[ -s "${ROOT}/source/pp4-kernel-settings.json" ]]; then'):shell.index('elif [[ -s "${ROOT}/source/reused-kernel-selection.json" ]]; then')]
    assert 'select-firmus-prefill-kernels.sh' not in pp4 and 'srun ' not in pp4
    script = (ROOT / 'scripts/select-firmus-prefill-kernels.sh').read_text()
    assert 'shift 2' in script and '--output "${out}/fa3.json" "$@"' in script
