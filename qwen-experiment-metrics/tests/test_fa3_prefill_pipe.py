"""CPU checks for the pipelined-tile Prefill FA3 row, its PP4 in-job screen against the pin and its endurance gate."""
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FA3 = ROOT / 'kernels/fa3-verify'
M192_NAME = 'm192n96-direct-o-varlen-pipe'


def added(name):
    return '\n'.join(line[1:] for line in (FA3 / name).read_text().splitlines() if line.startswith('+') and not line.startswith('+++'))


def test_pipe_row_is_the_pinned_row_plus_bit_128():
    rows = {r['name']: r for r in json.loads((FA3 / 'prefill-candidates.json').read_text())}
    row, pin = rows['n192-direct-o-varlen-pipe'], rows['n192-direct-o-varlen']
    assert row['variant'] == pin['variant'] | 128 and not pin['variant'] & 128
    assert {k: v for k, v in row.items() if k not in ('name', 'variant')} == {k: v for k, v in pin.items() if k not in ('name', 'variant')}
    assert [name for name, r in rows.items() if r['variant'] & 128] == ['n192-direct-o-varlen-pipe', 'n192-direct-o-varlen-pipe-diag', M192_NAME]


def test_m192n96_row_preserves_the_served_pipe_features_and_declares_precision_reference():
    rows = {r['name']: r for r in json.loads((FA3 / 'prefill-candidates.json').read_text())}
    row, pipe = rows[M192_NAME], rows['n192-direct-o-varlen-pipe']
    assert row['variant'] == pipe['variant'] | 8192 == 8416
    assert row['block_n'] == 96 and row['varlen'] and not row['bitwise'] and not row['byte_image']
    assert row['precision_reference'] == pipe['name'] and 'bitwise_reference' not in row
    assert row['variant'] & pipe['variant'] == pipe['variant']
    assert [name for name, spec in rows.items() if spec['variant'] & 8192] == [M192_NAME]


def test_m192_geometry_is_scoped_to_fixed_varlen_and_compiled_after_other_prefill_patches():
    build = (FA3 / 'build.py').read_text()
    geometry = (FA3 / 'prefill-geometry.patch').read_text()
    assert build.index('"prefill-diag.patch"') < build.index('"prefill-geometry.patch"') < build.index('cuda = Path("/usr/local/cuda")')
    assert 'FixedPrefill && (PrefillVariant & 8192) ? 192 :' in geometry
    assert 'kBlockM == 192 && kBlockN == 96 && FixedVarlen' in geometry
    assert '(kBlockM == 128 || (kBlockM == 192 && get<2>(TileShape_MNK_PV{}) == 96))' in geometry
    # The fixed scheduler uses packed cu_seqlens and kBlockM; it does not depend on the wrapper's 128-rounded max length.
    scheduler = (FA3 / 'prefill-varlen.patch').read_text()
    assert 'params.cu_seqlens[b + 1] - params.cu_seqlens[b]' in scheduler and ', kBlockM);' in scheduler
    assert 'seqlen_q_rounded' not in scheduler and 'seqlen_k_rounded' not in scheduler
    assert 'p.seqlen_q_rounded=p.seqlen_k_rounded=(maxlen+127)/128*128;' in (FA3 / 'wrapper.cpp').read_text()


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


def test_pp4_pins_the_served_pipe_row_and_keeps_the_gate_scripts_staged():
    # The PP4 job screens only the new M192/N96 varlen row against this pin.
    pp4 = pp4_branch()
    assert '"prefill-${worker}" screen "${pinned_fa3}"' in pp4
    assert f'--names {M192_NAME}' in (ROOT / 'scripts/select-firmus-pp4-bundle.sh').read_text()
    prepare = runpy.run_path(str(ROOT / 'scripts/prepare-firmus-pp4-settings.py'))
    assert prepare['PP4_SELECTED']['SGLANG_PREFILL_FA3_VERIFY'] == 'n192-direct-o-varlen-pipe'
    assert prepare['PP4_SELECTED_RUNS']['SGLANG_PREFILL_FA3_VERIFY'] == [41040348302]
    provision = (ROOT / 'scripts/provision-firmus-framework.sh').read_text()
    for script in ('select-firmus-prefill-fa3.py', 'select-firmus-prefill-kernels.sh'):
        assert f'"${{GITHUB_WORKSPACE}}/scripts/{script}"' in provision


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
    assert all(key in source for key in ('numeric_validation', 'precision_reference_metrics', 'serving_path_equal', 'candidate-first-output'))
    assert 'torch.equal(first[0].view(x["rows"], 8192).view(torch.uint8), candidate_out.view(torch.uint8))' in source
    assert 'torch.equal(first[1], candidate_scales)' in source
    assert "checks[\"prefill_validate_candidate\"](common, row)" in source
    # Served rows need no gate.
    assert set(selector['UNGATED']) == {'control', 'n192', 'n192-direct-o-varlen', 'n192-direct-o-varlen-pipe'}


def test_precision_candidate_endurance_requires_all_shapes_and_both_numeric_and_serving_path_gates(tmp_path):
    selector = runpy.run_path(str(ROOT / 'scripts/select-firmus-prefill-fa3.py'))
    row = M192_NAME
    shapes = {'+'.join(map(str, shape)) for shape in selector['ENDURANCE_SHAPES']}
    assert selector['endurance_reference_mode'](next(spec for spec in selector['CANDIDATES'] if spec['name'] == row)) == 'candidate-first-output'
    directory = tmp_path / 'endurance'
    for worker in range(3):
        path = directory / f'prefill-{worker}'
        path.mkdir(parents=True)
        for rank in range(4):
            numeric = {
                key: {
                    'numeric_passed': True,
                    'precision_reference': 'n192-direct-o-varlen-pipe',
                    'changed_input_replays': 2,
                    'metrics': [{}] * 3,
                    'precision_reference_metrics': [{}] * 3,
                }
                for key in shapes
            }
            entry = {
                'rank': rank, 'row': row, 'needed': True, 'calls': selector['ENDURANCE_CALLS'], 'passed': True,
                'reference_mode': 'candidate-first-output', 'numeric_passed': True, 'numeric_validation': numeric,
                'serving_path_equal': {key: True for key in shapes}, 'bytes_equal': {key: True for key in shapes},
            }
            (path / f'rank-{rank}.json').write_text(json.dumps(entry))
    assert selector['endurance_failure'](directory, row, 3) is None
    path = directory / 'prefill-1/rank-2.json'
    entry = json.loads(path.read_text())
    entry['numeric_validation'].pop('4001+4001')
    path.write_text(json.dumps(entry))
    assert 'all-shape stock/FP32 gate' in selector['endurance_failure'](directory, row, 3)
    entry['numeric_validation']['4001+4001'] = {
        'numeric_passed': True, 'precision_reference': 'n192-direct-o-varlen-pipe',
        'changed_input_replays': 2, 'metrics': [{}] * 3, 'precision_reference_metrics': [{}] * 3,
    }
    entry['serving_path_equal']['3585'] = False
    path.write_text(json.dumps(entry))
    assert 'serving wrapper differs' in selector['endurance_failure'](directory, row, 3)
    entry['serving_path_equal']['3585'] = True
    entry['bytes_equal'].pop('4096')
    path.write_text(json.dumps(entry))
    assert 'output stability for every endurance shape' in selector['endurance_failure'](directory, row, 3)


def test_gate_skips_the_gpu_when_the_screen_keeps_a_served_row(tmp_path, monkeypatch):
    selector, directory = screen(tmp_path, 4.0)
    monkeypatch.setenv('LOCAL_RANK', '2')
    out = tmp_path / 'gate'
    selector['run_endurance'](directory, out, 3)
    entry = json.loads((out / 'rank-2.json').read_text())
    assert entry['row'] == 'control' and not entry['needed'] and entry['passed'] and entry['calls'] == 0
