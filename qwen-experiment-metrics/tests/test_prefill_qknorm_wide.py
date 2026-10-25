"""CPU contract coverage for exact Prefill QK-norm dispatch and selection."""
import copy
import json
import re
import runpy
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / 'patches/sglang/prefill-qknorm-wide.patch'


@pytest.fixture
def selector():
    return runpy.run_path(str(ROOT / 'scripts/select-firmus-prefill-fa3.py'))


def ranks(selector, count=4):
    return [{'rank': rank, 'qknorm': {'lengths': {str(length): {dtype: {'bitwise_passed': True, 'control': [65.5] * selector['QKNORM_PAIRS'], 'wide': [57.5] * selector['QKNORM_PAIRS']} for dtype in selector['QKNORM_ID_TYPES']} for length in selector['LENGTHS']}}} for rank in range(count)]


def test_exact_eight_us_boundary_and_no_mutation(selector):
    records = ranks(selector)
    original = copy.deepcopy(records)
    assert selector['choose_qknorm'](records)['winner'] == 'wide'
    assert records == original
    records[3]['qknorm']['lengths']['4027']['int64']['wide'] = [57.501] * 50
    assert selector['choose_qknorm'](records)['winner'] == 'control'


@pytest.mark.parametrize('defect', ['mismatch', 'exception', 'short', 'missing_shape', 'missing_dtype', 'nan', 'infinity', 'zero', 'bool', 'missing_qknorm', 'duplicate_rank', 'missing_rank'])
def test_every_rank_shape_dtype_and_output_must_pass(selector, defect):
    records = ranks(selector)
    entry = records[2]['qknorm']['lengths']['4096']['int32']
    if defect == 'mismatch':
        entry['bitwise_passed'] = False
    elif defect == 'exception':
        records[2]['qknorm']['error'] = 'TimeoutError: 14 seconds'
    elif defect == 'short':
        entry['control'].pop()
    elif defect == 'missing_shape':
        del records[2]['qknorm']['lengths']['3986']
    elif defect == 'missing_dtype':
        del records[2]['qknorm']['lengths']['4027']['int64']
    elif defect == 'missing_qknorm':
        del records[2]['qknorm']
    elif defect == 'duplicate_rank':
        records[2]['rank'] = 1
    elif defect == 'missing_rank':
        records.pop()
    else:
        entry['wide'][0] = {'nan': float('nan'), 'infinity': float('inf'), 'zero': 0, 'bool': True}[defect]
    assert selector['choose_qknorm'](records)['winner'] == 'control'


def write_fragments(selector, directory):
    for worker in range(3):
        path = directory / f'prefill-{worker}'
        path.mkdir()
        (path / 'status.txt').write_text('fa3=0\n')
        records = ranks(selector)
        for record in records:
            record['lengths'] = {str(length): {name: {'numeric_passed': True, 'control': [350.] * 20, 'candidate': [350.] * 20} for name in selector['NAMES'][1:]} for length in selector['LENGTHS']}
        (path / 'fa3.json').write_text(json.dumps({'ranks': records}))


def test_aggregate_requires_all_twelve_gpus_and_exports_both_choices(selector, tmp_path):
    write_fragments(selector, tmp_path)
    output, env = tmp_path / 'selection.json', tmp_path / 'prefill.env'
    selector['aggregate_selection'](tmp_path, output, env)
    selected = json.loads(output.read_text())
    assert selected['winner'] == 'control'
    assert selected['qknorm']['winner'] == 'wide'
    assert selected['qknorm']['rule']['gpus'] == 12
    assert env.read_text() == 'SGLANG_PREFILL_FA3_VERIFY=control\nSGLANG_PREFILL_QKNORM=wide\n'
    path = tmp_path / 'prefill-2/fa3.json'
    records = json.loads(path.read_text())
    records['ranks'][3]['qknorm']['lengths']['3986']['int64']['bitwise_passed'] = False
    path.write_text(json.dumps(records))
    selector['aggregate_selection'](tmp_path, output, env)
    assert json.loads(output.read_text())['qknorm']['winner'] == 'control'
    (tmp_path / 'prefill-2/status.txt').write_text('fa3=124\n')
    selector['aggregate_selection'](tmp_path, output, env)
    assert json.loads(output.read_text())['qknorm']['winner'] == 'control'


def test_overlay_applies_last_and_control_body_is_untouched(tmp_path):
    overlays = json.loads((ROOT / 'configs/firmus-framework.json').read_text())['sglang_overlay_patches']
    assert len(overlays) == len(set(overlays))
    assert overlays[-1] == str(PATCH.relative_to(ROOT))
    assert overlays.index('patches/sglang/prefill-qkv-fast-fp8.patch') < overlays.index('patches/sglang/decode-qknorm-fast-fp8.patch') < len(overlays) - 1
    # Rebuild the three exact preimage contexts at their old line offsets.
    lines = []
    offset = None
    for line in PATCH.read_text().splitlines(keepends=True):
        match = re.match(r'@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@', line)
        if match:
            offset = int(match[1]) - 1
            while len(lines) < offset:
                lines.append('// fixture padding\n')
        elif offset is not None and (line.startswith((' ', '-')) or line == '\n'):
            lines.append(line[1:] if line != '\n' else line)
    header = tmp_path / 'python/sglang/kernels/jit/csrc/elementwise/rope.cuh'
    header.parent.mkdir(parents=True)
    original = ''.join(lines)
    header.write_text(original)
    subprocess.run(['git', 'apply', '--check', str(PATCH)], cwd=tmp_path, check=True)
    subprocess.run(['git', 'apply', str(PATCH)], cwd=tmp_path, check=True)
    candidate = header.read_text()
    assert 'qknorm_rope_prefill_wide_kernel' in candidate
    assert '!v_image && num_qo_heads==64 && num_kv_heads==4' in candidate
    assert 'std::strcmp(qknorm,"wide")==0' in candidate
    # No deleted line: the previous kernels and their launches remain verbatim.
    assert not any(line.startswith('-') and not line.startswith('---') for line in PATCH.read_text().splitlines())
    subprocess.run(['git', 'apply', '--reverse', str(PATCH)], cwd=tmp_path, check=True)
    assert header.read_text() == original


@pytest.mark.parametrize('defect', ['missing_fragment', 'bad_json', 'duplicate_rank', 'missing_gate'])
def test_aggregate_bad_fragments_retain_control(selector, tmp_path, defect):
    write_fragments(selector, tmp_path)
    fragment = tmp_path / 'prefill-1/fa3.json'
    if defect == 'missing_fragment':
        fragment.unlink()
    elif defect == 'bad_json':
        fragment.write_text('{')
    else:
        records = json.loads(fragment.read_text())
        if defect == 'duplicate_rank':
            records['ranks'][3]['rank'] = 0
        else:
            del records['ranks'][3]['qknorm']
        fragment.write_text(json.dumps(records))
    output, env = tmp_path / 'selection.json', tmp_path / 'prefill.env'
    selector['aggregate_selection'](tmp_path, output, env)
    assert json.loads(output.read_text())['qknorm']['winner'] == 'control'
    assert 'SGLANG_PREFILL_QKNORM=control\n' in env.read_text()


def test_reduction_and_rounding_expressions_are_identical():
    control = (ROOT / 'patches/sglang/prefill-qkv-fast-fp8.patch').read_text()
    wide = PATCH.read_text()
    for start, stop in [('            #pragma unroll\n            for (int j=0;j<4;++j)', '            const int64_t pos='), ('            #pragma unroll\n            for(int j=0;j<4;++j)', '        if constexpr (kVImage)')]:
        # Compare arithmetic text independent of unified diff prefixes.
        old = '\n'.join(line[1:] for line in control.splitlines() if line.startswith('+') and not line.startswith('+++'))
        new = '\n'.join(line[1:] for line in wide.splitlines() if line.startswith('+') and not line.startswith('+++'))
        expected = old.split(start, 1)[1].split(stop, 1)[0]
        if stop.startswith('        if constexpr'):
            expected = expected.rstrip()
        assert expected in new
