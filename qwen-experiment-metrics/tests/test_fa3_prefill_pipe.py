"""CPU checks for the pipelined-tile Prefill FA3 row and its PP4 in-job screen against the pin."""
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
    assert 'if constexpr (!PrefillPipe) { shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2); }' in text
    assert 'if constexpr (CollectiveMainloop::PrefillPipe) { mainloop.pipe_end(shared_storage); }' in text
    # The tile hand-off uses the unused AppendKV storage; SharedStorage keeps its layout.
    assert 'reinterpret_cast<int4*>(&shared_storage.pipelines.pipeline_k_new)' in text and 'static_assert(!AppendKV' in text
    # Consumers end on bidb < 0, and every tile releases its Q exactly once after its last QK.
    assert text.count('query_empty();') == 3 and 'if (bidb1 < 0) { break; }' in text and 'make_int4(0, 0, -1, 0)' in text


def test_pp4_no_longer_screens_the_pipe_row_and_keeps_the_pin():
    # 40991744289 adopted the pipe row with the split MegaMoE and stopped serving; the row stays a candidate.
    shell = (ROOT / 'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    pp4 = shell[shell.index('if [[ -s "${ROOT}/source/pp4-kernel-settings.json" ]]; then'):shell.index('elif [[ -s "${ROOT}/source/reused-kernel-selection.json" ]]; then')]
    assert 'select-firmus-prefill-kernels.sh' not in pp4 and 'select-firmus-prefill-fa3.py' not in pp4 and 'pinned_fa3' not in pp4
    assert 'n192-direct-o-varlen-pipe' in (ROOT / 'kernels/fa3-verify/prefill-candidates.json').read_text()


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
