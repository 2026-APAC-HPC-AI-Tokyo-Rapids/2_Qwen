"""CPU checks for the diagonal-trim Prefill FA3 row, its sub-block bound and its PP4 in-job screen and gate."""
import json
import math
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FA3 = ROOT / 'kernels/fa3-verify'
NAME = 'n192-direct-o-varlen-pipe-diag'


def added(name):
    return '\n'.join(line[1:] for line in (FA3 / name).read_text().splitlines() if line.startswith('+') and not line.startswith('+++'))


def test_diag_row_is_the_pinned_pipe_row_plus_bit_4096():
    rows = {r['name']: r for r in json.loads((FA3 / 'prefill-candidates.json').read_text())}
    row, pin = rows[NAME], rows['n192-direct-o-varlen-pipe']
    assert row['variant'] == pin['variant'] | 4096 and not pin['variant'] & 4096
    assert {k: v for k, v in row.items() if k not in ('name', 'variant')} == {k: v for k, v in pin.items() if k not in ('name', 'variant')}
    assert [name for name, r in rows.items() if r['variant'] & 4096] == [NAME]


def test_diag_patch_is_last_in_the_build_chain_and_keeps_the_pinned_path():
    source = (FA3 / 'build.py').read_text()
    assert source.index('"decode-stream-h4.patch"') < source.index('"prefill-diag.patch"') < source.index('cuda = Path("/usr/local/cuda")')
    text = added('prefill-diag.patch')
    assert 'static constexpr bool PrefillDiag = (PrefillVariant & 4096) != 0;' in text and 'static constexpr int kDiagN = 32;' in text
    # The pinned rows keep their full-block mask and softmax; only the diagonal-trim row dispatches on sub-blocks.
    assert 'if constexpr (!PrefillDiag) {\n                    make_mask(nxt).template apply<true /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, nxt.m_block, nxt.n_block_max - 1);' in text
    # QK and PV GEMMs are untouched (a wgmma or its wait in a branch serializes every wgmma of the kernel).
    assert 'flash::gemm' not in text and 'warpgroup_wait' not in text
    # The sub-block count is warp-uniform for ptxas, and the skipped P bytes are written as zeros.
    assert 'return __shfl_sync(0xffffffff, std::min(std::max(cute::ceil_div(limit, int(kDiagN)), 1), int(kDiagSubs)), 0);' in text
    # Every byte of the skipped k-blocks: the 32-bit view must cover the whole tail (a quarter of it did in 7f4a64e).
    assert 'static_assert(size(decltype(tOrP_tail.layout()){}) * 4 == cosize(decltype(tOrP.layout()){}) - kHead);' in text and 'cute::fill(tOrP_tail, 0u);' in text


def diag_subs(m_block, wg, seqlen_q, seqlen_k, group=16, block_m=128, block_n=192, diag_n=32):
    """prefill-diag.patch diag_subs and the diagonal block n_block_max - 1 of get_n_block_min_max."""
    m_idx_max = ((m_block + 1) * block_m - 1) // group + 1
    n_block_max = min(math.ceil(seqlen_k / block_n), math.ceil((m_idx_max + seqlen_k - seqlen_q) / block_n))
    token_last = (m_block * block_m + (wg + 1) * 64 - 1) // group
    limit = min(token_last + 1 + seqlen_k - seqlen_q, seqlen_k) - (n_block_max - 1) * block_n
    return n_block_max - 1, min(max(math.ceil(limit / diag_n), 1), block_n // diag_n)


def test_skipped_sub_blocks_are_masked_for_every_row_of_the_warpgroup():
    # Mask (causal + seqlen_k) keeps column c of row r in block n only if c < min(t + 1 + sk - sq, sk) - n * 192,
    # with t the row's token (PackGQA: packed row // 16, padding rows beyond seqlen_q included).
    cases = [(n, n) for n in (1, 7, 8, 9, 191, 192, 193, 3585, 3986, 4001, 4027, 4032, 4096)] + [(100, 4000), (3986, 4100), (7, 200)]
    trimmed = total = 0
    for seqlen_q, seqlen_k in cases:
        for m_block in range(math.ceil(seqlen_q * 16 / 128)):
            for wg in (0, 1):
                n, subs = diag_subs(m_block, wg, seqlen_q, seqlen_k)
                assert 1 <= subs <= 6
                tokens = [(m_block * 128 + wg * 64 + r) // 16 for r in range(64)]
                limits = [min(t + 1 + seqlen_k - seqlen_q, seqlen_k) - n * 192 for t in tokens]
                assert max(limits) <= subs * 32, (seqlen_q, seqlen_k, m_block, wg)
                assert subs == 1 or max(limits) > (subs - 1) * 32
                trimmed += 6 - subs
                total += 6
    assert 0.35 < trimmed / total < 0.5


def test_trim_removes_about_two_fifths_of_the_diagonal_softmax_at_serving_lengths():
    for length in (3986, 4001, 4027, 4096):
        subs = [diag_subs(m, wg, length, length)[1] for m in range(math.ceil(length * 16 / 128)) for wg in (0, 1)]
        assert 3.45 < sum(subs) / len(subs) < 3.52


def pp4_branch():
    shell = (ROOT / 'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    return shell[shell.index('if [[ -s "${ROOT}/source/pp4-kernel-settings.json" ]]; then'):shell.index('elif [[ -s "${ROOT}/source/reused-kernel-selection.json" ]]; then')]


def test_pp4_screens_and_gates_only_the_m192_row_against_the_pin():
    pp4 = pp4_branch()
    bundle = (ROOT / 'scripts/select-firmus-pp4-bundle.sh').read_text()
    assert '"prefill-${worker}" screen "${pinned_fa3}"' in pp4
    m192 = 'm192n96-direct-o-varlen-pipe'
    assert bundle.count('bash "${root}/scripts/select-firmus-prefill-kernels.sh"') == 1 and f'--names {m192} --control "${{control}}"' in bundle
    assert bundle.count('select-firmus-prefill-fa3.py" --endurance "${root}/results/prefill-selection"') == 1 and 'timeout --kill-after=5 90' in bundle
    assert '--aggregate "${ROOT}/results/prefill-selection" --endurance "${ROOT}/results/prefill-fa3-endurance"' in pp4
    # The gate runs after all three screens, and the aggregate after all three gates.
    assert pp4.index('for pid in "${prefill_gpu_selections[@]}"') < pp4.index('"prefill-${worker}" endurance') < pp4.index('for pid in "${prefill_gpu_gates[@]}"') < pp4.index('--aggregate')
    for script in ('select-firmus-megamoe', 'select-firmus-prefill-elementwise', 'select-firmus-decode', 'check-firmus-pp4-copy-engine'):
        assert script not in pp4
    prepare = runpy.run_path(str(ROOT / 'scripts/prepare-firmus-pp4-settings.py'))
    assert prepare['PP4_SELECTED']['SGLANG_PREFILL_FA3_VERIFY'] == 'n192-direct-o-varlen-pipe'


def screen(tmp_path, saving):
    selector = runpy.run_path(str(ROOT / 'scripts/select-firmus-prefill-fa3.py'))
    directory = tmp_path / 'prefill-selection'
    for worker in range(3):
        path = directory / f'prefill-{worker}'
        path.mkdir(parents=True, exist_ok=True)
        (path / 'status.txt').write_text('fa3=0\n')
        ranks = [{'rank': rank, 'control_row': 'n192-direct-o-varlen-pipe', 'lengths': {key: {NAME: {'numeric_passed': True, 'control': [400.0] * 20, 'candidate': [400.0 - saving] * 20}} for key in selector['shapes'](NAME)}} for rank in range(4)]
        (path / 'fa3.json').write_text(json.dumps({'ranks': ranks}))
    return selector, directory


def gate(tmp_path, row=NAME, calls=None, passed=True, skip=None):
    selector = runpy.run_path(str(ROOT / 'scripts/select-firmus-prefill-fa3.py'))
    directory = tmp_path / 'prefill-fa3-endurance'
    shapes = {'+'.join(map(str, shape)) for shape in selector['ENDURANCE_SHAPES']}
    for worker in range(3):
        (directory / f'prefill-{worker}').mkdir(parents=True, exist_ok=True)
        for rank in range(4):
            if (worker, rank) == skip:
                continue
            entry = {'rank': rank, 'row': row, 'needed': True, 'calls': selector['ENDURANCE_CALLS'] if calls is None else calls, 'reference_mode': 'pinned-output', 'bytes_equal': {key: passed for key in shapes}, 'passed': passed}
            (directory / f'prefill-{worker}' / f'rank-{rank}.json').write_text(json.dumps(entry))
    return directory


def test_aggregate_adopts_the_diag_row_only_after_every_gpu_passed_the_gate(tmp_path):
    out, env = tmp_path / 'selection.json', tmp_path / 'prefill.env'
    pinned = 'SGLANG_PREFILL_MOE_INPUT=onepass\nSGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen-pipe\nSGLANG_PP4_PROXY_TRANSPORT=copy\n'
    selector, directory = screen(tmp_path, 12.0)
    cases = [
        ({}, NAME, 'passed'),
        ({'skip': (1, 2)}, 'n192-direct-o-varlen-pipe', 'no endurance result from prefill-1 rank 2'),
        ({'calls': 600}, 'n192-direct-o-varlen-pipe', 'failed the endurance gate: 600 calls'),
        ({'passed': False}, 'n192-direct-o-varlen-pipe', 'failed output stability for every endurance shape'),
    ]
    for index, (kwargs, expected, reason) in enumerate(cases):
        endurance = gate(tmp_path / f'case-{index}', **kwargs)
        env.write_text(pinned)
        selector['aggregate_selection'](directory, out, env, 3, endurance)
        assert env.read_text() == f'SGLANG_PREFILL_MOE_INPUT=onepass\nSGLANG_PP4_PROXY_TRANSPORT=copy\nSGLANG_PREFILL_FA3_VERIFY={expected}\n', kwargs
        assert reason in json.loads(out.read_text())['endurance']
    # A screen the pin wins (saving below 5 us on one GPU and shape) keeps the pin without gate results.
    selector, directory = screen(tmp_path / 'lost', 4.0)
    env.write_text(pinned)
    selector['aggregate_selection'](directory, out, env, 3, tmp_path / 'missing')
    assert env.read_text().endswith('SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen-pipe\n') and 'endurance' not in json.loads(out.read_text())


def test_gate_runs_the_diag_row_and_skips_served_rows(tmp_path, monkeypatch):
    selector = runpy.run_path(str(ROOT / 'scripts/select-firmus-prefill-fa3.py'))
    assert set(selector['UNGATED']) == {'control', 'n192', 'n192-direct-o-varlen', 'n192-direct-o-varlen-pipe'}
    assert NAME not in selector['UNGATED']
    selector, directory = screen(tmp_path, 3.0)
    monkeypatch.setenv('LOCAL_RANK', '1')
    selector['run_endurance'](directory, tmp_path / 'gate', 3)
    entry = json.loads((tmp_path / 'gate/rank-1.json').read_text())
    assert entry['row'] == 'control' and not entry['needed'] and entry['passed']


def test_genkai_probe_times_the_diag_row_against_the_pin():
    script = (ROOT / 'kernels/probes/prepare-fa3-prefill-diag.sh').read_text()
    rows = {r['name']: r for r in json.loads((FA3 / 'prefill-candidates.json').read_text())}
    assert f"variants={rows['n192-direct-o-varlen-pipe']['variant']},{rows[NAME]['variant']}" in script and 'names=pin,diag' in script
    # The chain comes from build.py, so the probe builds what the image builds; a serialized wgmma fails the build.
    assert 'kernels/fa3-verify/build.py' in script and "grep -q 'Performance Loss'" in script
