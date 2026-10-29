"""CPU execution of the actual uneven metadata preparation and dispatch ABI."""
import json
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_uneven_prepare_actual_cpp_on_cpu(tmp_path):
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('g++ is required for the GPU-independent scheduler fixture')
    source = (ROOT / 'kernels/fa3-verify/decode_prepare.cu').read_text()
    kernel = source.split('__global__ void prepare_uneven', 1)[1].split('void decode_prepare', 1)[0]
    fixture = '''#include <algorithm>
#include <cassert>
#include <numeric>
using std::max; using std::min;
struct { int x = 0; } threadIdx;
void prepare_uneven''' + kernel + '''
int main() {
    for (int b = 32; b <= 48; ++b) {
        int rounded = (b + 3) / 4 * 4;
        for (int mode = 0; mode < 3; ++mode) {
            int lengths[48], metadata[194], seen[48] = {};
            std::fill(metadata, metadata + 194, -123);
            for (int i = 0; i < b; ++i) { lengths[i] = mode == 0 ? 4096 : mode == 1 ? 4 + (i * 37) % 129 : 3900 + (37 * i + 13) % 301; }
            prepare_uneven(lengths, metadata, b);
            int fours = 0, previous = 1000000, total = 0;
            for (int i = 0; i < b; ++i) {
                int r = metadata[2 * rounded + i], s = metadata[i];
                assert(r >= 0 && r < b && !seen[r]++);
                assert(s == 3 || s == 4);
                fours += s == 4; total += 4 * s;
                int cost = ((lengths[r] + 127) / 128 + s - 1) / s;
                assert(cost <= previous); previous = cost;
                assert(metadata[rounded + i] == 1);
                int heads = metadata[3 * rounded + i];
                assert(heads == 1 || heads == 2 || heads == 4);
                // Split partitions cover each N tile exactly once.
                int covered = 0;
                for (int split = 0; split < s; ++split) { covered += max(0, min(cost, (lengths[r] + 127) / 128 - split * cost)); }
                assert(covered == (lengths[r] + 127) / 128);
                if (s == 4) {
                    for (int j = 0; j < b; ++j) { if (metadata[j] == 3) { assert(lengths[r] >= lengths[metadata[2 * rounded + j]]); } }
                }
            }
            assert(fours == max(0, min(b, 132 - 3 * b)));
            if (b >= 34 && b <= 40) { assert(total == 528); }
            assert(metadata[4 * rounded] == 0);
            assert(metadata[4 * rounded + 1] == -123);
        }
    }
}
'''
    path = tmp_path / 'prepare.cpp'
    path.write_text(fixture)
    binary = tmp_path / 'prepare'
    subprocess.run([compiler, '-std=c++17', '-O2', str(path), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_main_identity_distinguishes_merge_but_shares_output_and_split_capacity(tmp_path):
    ns = runpy.run_path(str(ROOT / 'kernels/fa3-verify/build.py'))
    tasks = ns['generate_dispatch'](ROOT / 'kernels/fa3-verify', tmp_path)
    rows = json.loads((ROOT / 'kernels/fa3-verify/candidates.json').read_text())
    identities = {(r['block_n'], r['stages'], r['splits'] > 1, r.get('merge', False), r.get('ctas', 1), r.get('heads', 1)) for r in rows}
    assert len(tasks) == len(identities) == 10
    assert sum(r.get('merge', False) for r in rows) == 6
    assert sum(r.get('uneven', False) for r in rows) == 7
    for path, _, _ in tasks:
        assert '#define VERIFY_MERGE ' in path.read_text()
    assert {r['stages'] for r in rows if r.get('merge')} == {2, 3, 4}
    assert {r['stages'] for r in rows if r.get('uneven')} == {2, 3, 4}
