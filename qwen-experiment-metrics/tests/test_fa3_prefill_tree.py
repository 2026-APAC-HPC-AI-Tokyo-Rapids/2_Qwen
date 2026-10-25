"""Exercise the added row-max source and probe dispatch without a GPU."""
import ast
import json
import runpy
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("columns", (48, 56))
def test_tree_max_actual_source_matches_serial_max(tmp_path, columns):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('g++ required')
    patch = (ROOT / 'kernels/fa3-verify/prefill-overlap.patch').read_text()
    added = '\n'.join(line[1:] for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++'))
    function = added.split('    template<bool First, class Scores>', 1)[1].split('\n    }', 1)[0] + '\n    }'
    source = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <random>
#include <type_traits>
#define __forceinline__ inline
#define __device__
#define CUTE_STATIC_V(x) decltype(x)::value
struct Scores { float x[2][COLS]; float operator()(int i, int j) const { return x[i][j]; } };
template<int N> auto size(Scores const&) { return std::integral_constant<int,N == 0 ? 2 : COLS>{}; }
struct Rows { float x[2]; float& operator()(int i) { return x[i]; } };
template<class T> struct MaxOp { T operator()(T a,T b) { return std::fmax(a,b); } };
template<class... T> void quad_allreduce_(T&&...) {}
namespace flash { template<bool First, class... T> void reduce_max(T&&...) {} }
struct Softmax { static constexpr bool TreeMax = true; static constexpr int kNRows = 2; Rows row_max;
    template<bool First, class Scores>
''' + function + r'''
};
int main() {
    std::mt19937 rng(20261025); std::uniform_real_distribution<float> random(-40.f,40.f);
    for (int trial=0; trial<1000; ++trial) {
        Scores scores;
        for (auto& row : scores.x) for (float& x : row) x = trial == 0 ? -INFINITY : trial == 1 ? 0.f : random(rng);
        Softmax first{}, update{}; update.row_max = {{random(rng), random(rng)}};
        float previous[2] = {update.row_max(0), update.row_max(1)};
        first.prefill_reduce_max<true>(scores); update.prefill_reduce_max<false>(scores);
        for (int row=0; row<2; ++row) {
            float expected = -INFINITY;
            for (float x : scores.x[row]) expected = std::fmax(expected,x);
            assert(first.row_max(row) == expected);
            assert(update.row_max(row) == std::fmax(previous[row],expected));
        }
    }
}
'''
    path = tmp_path / 'max.cpp'
    path.write_text(source.replace('COLS', str(columns)))
    binary = tmp_path / 'max'
    subprocess.run([compiler, '-O3', '-std=c++17', str(path), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_probe_new_modes_use_dedicated_library_and_fused_output():
    tree = ast.parse((ROOT / 'kernels/probes/fa3-prefill.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('PV_MODES', 'E4_MODES', 'VARIANT_MODES') for t in n.targets)]
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'call']
    calls = []
    namespace = {'torch': SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda: SimpleNamespace(cuda_stream=99)))}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'probe-dispatch', 'exec'), namespace)  # noqa: S102 (extract actual pure routing without CUDA imports)
    class Tensor:
        shape = (17, 64, 128)
        def data_ptr(self):
            return 123
        def stride(self, axis):
            return 20
    data = ([17], *(Tensor() for _ in range(9)))
    namespace['libraries'] = {mode: SimpleNamespace(prefill_run=lambda *args, mode=mode: calls.append((mode, args)) or 0) for mode in (*namespace['E4_MODES'], *namespace['PV_MODES'])}
    for mode in (*namespace['E4_MODES'], *namespace['PV_MODES']):
        assert namespace['call'](data, mode) == data[8:10]
        assert calls[-1][0] == mode
        assert calls[-1][1][11:13] == (123, 123)
    rows = json.loads((ROOT / 'kernels/fa3-verify/prefill-candidates.json').read_text())
    assert not any(r['variant'] & (2 | 8) for r in rows)  # Rejected serving candidates; retain probe modes.


def test_sass_max_depth_tracks_independent_chains_and_wait_position():
    schedule = runpy.run_path(str(ROOT / 'kernels/probes/fa3-prefill-sass.py'))['softmax_schedule']
    instructions = [
        (0, 'FMNMX', 'R0, R10, R11, !PT'),
        (16, 'FMNMX', 'R1, R12, R13, !PT'),
        (32, 'FMNMX', 'R0, R0, R14, !PT'),
        (48, 'FMNMX', 'R1, R1, R15, !PT'),
        (64, 'FMNMX', 'R0, R0, R1, !PT'),
        (80, 'MUFU.EX2', 'R2, R3'),
        (96, 'WARPGROUP.DEPBAR.LE', 'SB0, 0x0'),
    ]
    result = schedule(instructions)
    assert result['fmnmx_dependency_depth'] == 3
    assert result['max_pcs'] == ['0x0', '0x40']
    assert result['exp_pcs'] == ['0x50']
    assert result['wait0_pcs'] == ['0x60']


def test_sass_pv_reads_and_full_softmax_wait_gate(capsys):
    module = runpy.run_path(str(ROOT / 'kernels/probes/fa3-prefill-sass.py'))
    instructions = [
        (0, 'QGMMA.64x128x32.F32.E4M3.E4M3', 'R136, R220, gdesc[UR12], R136'),
        (16, 'MUFU.EX2', 'R24, R24'),
        (32, 'WARPGROUP.DEPBAR.LE', 'gsb0, 0x0'),
        *[(48 + i*16, 'PRMT', f'R{220+i}, R24, UR11, R{220+i}') for i in range(4)],
    ]
    result = module['softmax_schedule'](instructions)
    assert result['pv_a_register_words'] == result['pv_a_words_read_by_prmt_after_last_exp'] == 4
    assert result['wait0_after_last_exp']
    instructions[1], instructions[2] = instructions[2], instructions[1]
    instructions = [(i*16, op, operands) for i, (_, op, operands) in enumerate(instructions)]
    assert not module['softmax_schedule'](instructions)['wait0_after_last_exp']
    module['report']('Function : test\n /*0000*/ SETMAXNREG.INC 0xf0;\n /*0010*/ WARPGROUP.ARRIVE;\n /*0020*/ QGMMA.64x128x32.F32.E4M3.E4M3 R136, R220, gdesc[UR12], R136;\n /*0030*/ MUFU.EX2 R24, R24;\n /*0040*/ F2FP.F8.PACK_AB R25, R24, R24;\n /*0050*/ BRA 0x10;\n')
    assert "'pv_a_register_words': 4" in capsys.readouterr().out
