"""CPU checks for the pipelined-tile Prefill FA3 row, its PP4 in-job screen against the pin and its endurance gate."""
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FA3 = ROOT / 'kernels/fa3-verify'


def added(name):
    return '\n'.join(line[1:] for line in (FA3 / name).read_text().splitlines() if line.startswith('+') and not line.startswith('+++'))


def test_pipe_row_is_the_pinned_row_plus_bit_128():
    rows = {r['name']: r for r in json.loads((FA3 / 'prefill-candidates.json').read_text())}
    row, pin = rows['n192-direct-o-varlen-pipe'], rows['n192-direct-o-varlen']
    assert row['variant'] == pin['variant'] | 128 and not pin['variant'] & 128
    assert {k: v for k, v in row.items() if k not in ('name', 'variant')} == {k: v for k, v in pin.items() if k not in ('name', 'variant')}
    assert [name for name, r in rows.items() if r['variant'] & 128] == ['n192-direct-o-varlen-pipe']


def test_pipe_patch_is_last_in_the_build_chain():
    source = (FA3 / 'build.py').read_text()
    assert source.index('"decode-wide-h2.patch"') < source.index('"prefill-pipe.patch"') < source.index('cuda = Path("/usr/local/cuda")')
    tasks = runpy.run_path(str(FA3 / 'build.py'))
    assert 'generate_prefill_dispatch' in tasks


def test_pipe_patch_changes_other_rows_only_behind_the_flag():
    text = added('prefill-pipe.patch')
    assert 'static constexpr bool PrefillPipe = (PrefillVariant & 128) != 0;' in text
    # The producer hands tiles over with Q, so only this row runs the scheduler among producer warps only.
    assert 'CollectiveMainloop::PrefillPipe ? 0 : CollectiveMainloop::NumMmaThreads * VerifyHeads, NumProducerThreads,' in text
    # Neither the per-tile Q load nor the producer tail waits on barrier_O, whose phase work_idx no longer tracks.
    assert text.count('if constexpr (!PrefillPipe) { shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2); }') == 2
    assert 'if constexpr (CollectiveMainloop::PrefillPipe) { mainloop.pipe_end(shared_storage); }' in text
    # The tile hand-off lives in barrier_Qv, never an mbarrier without HasQv; SharedStorage keeps its layout.
    # The AppendKV pipeline storage that held it before has mbarriers initialized at kernel start.
    assert 'reinterpret_cast<int4*>(&shared_storage.pipelines.barrier_Qv)' in text and 'static_assert(!HasQv' in text
    assert 'pipeline_k_new' not in text
    # The consumers walk the producer's KV block count instead of recomputing it from cu_seqlens.
    assert 'make_int4(m_block, bidh, bidb, n_block_max)' in text and 'int const n_block_max = reinterpret_cast<int const volatile&>(tile->w);' in text
    assert 'BlockMN_t::get_n_block_min_max' not in text
    # Consumers end on bidb < 0, and every tile releases its Q exactly once after its last QK.
    assert text.count('query_empty();') == 3 and 'if (bidb1 < 0) { break; }' in text and 'make_int4(0, 0, -1, 0)' in text
    # Only this row launches without PDL, so the tile counter memset is ordered by the stream alone.
    assert '((FixedPrefill && !CollectiveMainloop::PrefillPipe) || (!params.skip_scheduler_metadata_computation && params.prepare_varlen_pdl)) /*launch_with_pdl*/' in text


def pp4_branch():
    shell = (ROOT / 'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    return shell[shell.index('if [[ -s "${ROOT}/source/pp4-kernel-settings.json" ]]; then'):shell.index('elif [[ -s "${ROOT}/source/reused-kernel-selection.json" ]]; then')]


def test_pp4_screens_the_pipe_row_against_the_pin_and_gates_it_before_the_workers():
    pp4 = pp4_branch()
    megamoe = pp4.index('select-firmus-prefill-megamoe.py" --ranks 1')
    screen = pp4.index('select-firmus-prefill-kernels.sh" "${ROOT}" "prefill-${worker}" --names n192-direct-o-varlen-pipe --control "${pinned_fa3}"')
    # Same worker, same four GPUs, after the MegaMoE selection; both run while the Decode-GPU selections run.
    assert megamoe < screen < pp4.index('prefill_gpu_selections+=("$!")') < pp4.index('for pid in "${prefill_gpu_selections[@]}"; do wait "${pid}" || true; done')
    assert 'pinned_fa3=$(sed -n \'s/^SGLANG_PREFILL_FA3_VERIFY=//p\' "${ROOT}/source/prefill-kernels.env")' in pp4
    # The FA3 gate follows the MegaMoE gate on the same GPUs, after every screen; then both aggregates.
    megamoe_gate = pp4.index('select-firmus-prefill-megamoe.py" --endurance')
    fa3_gate = pp4.index('timeout --kill-after=5 60 /opt/sglang/venv/bin/torchrun --standalone --nproc-per-node=4 "${ROOT}/scripts/select-firmus-prefill-fa3.py" --endurance "${ROOT}/results/prefill-selection" --output "${ROOT}/results/prefill-fa3-endurance/prefill-${worker}"')
    assert pp4.index('for pid in "${prefill_gpu_selections[@]}"') < megamoe_gate < fa3_gate < pp4.index('prefill_gpu_gates+=("$!")') < pp4.index('for pid in "${prefill_gpu_gates[@]}"')
    megamoe_aggregate = pp4.index('select-firmus-prefill-megamoe.py" --aggregate')
    fa3_aggregate = pp4.index('select-firmus-prefill-fa3.py" --aggregate "${ROOT}/results/prefill-selection" --endurance "${ROOT}/results/prefill-fa3-endurance"')
    assert pp4.index('for pid in "${prefill_gpu_gates[@]}"') < megamoe_aggregate < fa3_aggregate


def test_selector_times_the_screened_row_against_the_given_control(tmp_path):
    selector = runpy.run_path(str(ROOT / 'scripts/select-firmus-prefill-fa3.py'))
    source = (ROOT / 'scripts/select-firmus-prefill-fa3.py').read_text()
    assert 'control = checks["prefill_selection_step"](common, control_row, LAYERS)' in source
    assert 'choices=NAMES' in source and 'args.control not in names' in source
    # A losing or failed screen keeps the pinned row; a winning one replaces it.
    name = 'n192-direct-o-varlen-pipe'
    out, env = tmp_path / 'selection.json', tmp_path / 'prefill.env'
    for saving, expected in ((4.0, 'n192-direct-o-varlen'), (12.0, name)):
        for worker in range(3):
            path = tmp_path / f'prefill-{worker}'
            path.mkdir(exist_ok=True)
            (path / 'status.txt').write_text('fa3=0\n')
            ranks = [{'rank': rank, 'control_row': 'n192-direct-o-varlen', 'lengths': {key: {name: {'numeric_passed': True, 'control': [400.0] * 20, 'candidate': [400.0 - saving] * 20}} for key in selector['shapes'](name)}} for rank in range(4)]
            (path / 'fa3.json').write_text(json.dumps({'ranks': ranks}))
        env.write_text('SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen\nDG_SM90_PREFILL_4STAGE=1\n')
        selector['aggregate_selection'](tmp_path, out, env)
        assert env.read_text() == f'DG_SM90_PREFILL_4STAGE=1\nSGLANG_PREFILL_FA3_VERIFY={expected}\n'


def screen(tmp_path, saving, name='n192-direct-o-varlen-pipe'):
    selector = runpy.run_path(str(ROOT / 'scripts/select-firmus-prefill-fa3.py'))
    directory = tmp_path / 'prefill-selection'
    for worker in range(3):
        path = directory / f'prefill-{worker}'
        path.mkdir(parents=True, exist_ok=True)
        (path / 'status.txt').write_text('fa3=0\n')
        ranks = [{'rank': rank, 'control_row': 'n192-direct-o-varlen', 'lengths': {key: {name: {'numeric_passed': True, 'control': [400.0] * 20, 'candidate': [400.0 - saving] * 20}} for key in selector['shapes'](name)}} for rank in range(4)]
        (path / 'fa3.json').write_text(json.dumps({'ranks': ranks}))
    return selector, directory


def gate(tmp_path, row='n192-direct-o-varlen-pipe', calls=None, passed=True, skip=None):
    selector = runpy.run_path(str(ROOT / 'scripts/select-firmus-prefill-fa3.py'))
    directory = tmp_path / 'prefill-fa3-endurance'
    for worker in range(3):
        (directory / f'prefill-{worker}').mkdir(parents=True, exist_ok=True)
        for rank in range(4):
            if (worker, rank) == skip:
                continue
            entry = {'rank': rank, 'row': row, 'needed': True, 'calls': selector['ENDURANCE_CALLS'] if calls is None else calls, 'bytes_equal': {'4001+4001': passed}, 'passed': passed}
            (directory / f'prefill-{worker}' / f'rank-{rank}.json').write_text(json.dumps(entry))
    return directory


def test_gate_runs_the_serving_shapes_through_the_serving_call_path():
    selector = runpy.run_path(str(ROOT / 'scripts/select-firmus-prefill-fa3.py'))
    source = (ROOT / 'scripts/select-firmus-prefill-fa3.py').read_text()
    # Every 40991744289 microbatch was two 4001-token requests (#new-token 8064 is the 64-token page round-up).
    assert (4001, 4001) in selector['ENDURANCE_SHAPES'] and (4032, 4032) in selector['ENDURANCE_SHAPES'] and (3986, 4027) in selector['ENDURANCE_SHAPES']
    assert {len(shape) for shape in selector['ENDURANCE_SHAPES']} == {1, 2} and selector['ENDURANCE_CALLS'] >= 600
    assert all(3584 <= n <= 4096 for shape in selector['ENDURANCE_SHAPES'] for n in shape)
    assert 'from sglang.srt.layers.attention.prefill_fp8 import attention, prefill_candidate' in source
    assert 'last[key] = attention(x["q"], x["k"], x["v"], x["cu"], x["maxlen"], 128**-.5, x["rows"], fixed=True)' in source
    assert 'os.environ["SGLANG_PREFILL_FA3_VERIFY"] = row' in source and 'prefill_candidate()["name"] == row' in source
    # Served rows need no gate.
    assert set(selector['UNGATED']) == {'control', 'n192', 'n192-direct-o-varlen'}


def test_gate_skips_the_gpu_when_the_screen_keeps_a_served_row(tmp_path, monkeypatch):
    selector, directory = screen(tmp_path, 4.0)
    monkeypatch.setenv('LOCAL_RANK', '2')
    out = tmp_path / 'gate'
    selector['run_endurance'](directory, out, 3)
    entry = json.loads((out / 'rank-2.json').read_text())
    assert entry['row'] == 'control' and not entry['needed'] and entry['passed'] and entry['calls'] == 0


def test_aggregate_adopts_the_pipe_row_only_after_every_gpu_passed_the_gate(tmp_path):
    out, env = tmp_path / 'selection.json', tmp_path / 'prefill.env'
    selector, directory = screen(tmp_path, 12.0)
    cases = [
        ({}, 'n192-direct-o-varlen-pipe', 'passed'),
        ({'skip': (2, 3)}, 'n192-direct-o-varlen', 'no endurance result from prefill-2 rank 3'),
        ({'calls': 250}, 'n192-direct-o-varlen', 'failed the endurance gate: 250 calls'),
        ({'passed': False}, 'n192-direct-o-varlen', 'failed the endurance gate'),
        ({'row': 'control'}, 'n192-direct-o-varlen', 'gated control, not n192-direct-o-varlen-pipe'),
    ]
    for index, (kwargs, expected, reason) in enumerate(cases):
        endurance = gate(tmp_path / f'case-{index}', **kwargs)
        env.write_text('SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen\nDG_SM90_PREFILL_4STAGE=1\n')
        selector['aggregate_selection'](directory, out, env, 3, endurance)
        assert env.read_text() == f'DG_SM90_PREFILL_4STAGE=1\nSGLANG_PREFILL_FA3_VERIFY={expected}\n', kwargs
        assert reason in json.loads(out.read_text())['endurance']
    # A screen the pin wins needs no gate results.
    selector, directory = screen(tmp_path / 'lost', 4.0)
    env.write_text('SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen\n')
    selector['aggregate_selection'](directory, out, env, 3, tmp_path / 'missing')
    assert env.read_text() == 'SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen\n' and 'endurance' not in json.loads(out.read_text())
