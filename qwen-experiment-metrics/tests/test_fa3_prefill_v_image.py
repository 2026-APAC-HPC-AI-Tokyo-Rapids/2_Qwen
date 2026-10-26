"""The image producer must include the matrix instructions' cross-lane routes."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_c2_map_matches_deployed_qkv_and_covers_tile(tmp_path):
    header = (ROOT / 'kernels/fa3-verify/prefill_v_layout.cuh').read_text()
    patch = (ROOT / 'patches/sglang/prefill-qkv-fast-fp8.patch').read_text()
    assert '\n'.join('+' + line for line in header.splitlines()[1:]) in patch
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('g++ required')
    source = tmp_path / 'map.cpp'
    source.write_text('#define __host__\n#define __device__\n' + header + r'''
#include <array>
#include <cassert>
int main() {
    // A token byte crosses LDSM lanes before the STSM row is selected.
    // These addresses distinguish the matrix routing from a thread-local copy.
    assert(prefill_v_byte_offset(0,1)==256);
    assert(prefill_v_byte_offset(1,0)==1);
    assert(prefill_v_byte_offset(0,2)==32);
    std::array<bool,224*128> seen{};
    for(int t=0;t<224;++t) for(int d=0;d<128;++d) {
        int offset=prefill_v_byte_offset(t,d);
        assert(offset>=0 && offset<int(seen.size()) && !seen[offset]);
        seen[offset]=true;
    }
    for(bool bit:seen) assert(bit);
}
''')
    binary = tmp_path / 'map'
    subprocess.run([compiler, '-std=c++17', str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_c2_dump_gate_precedes_output_gate():
    source = (ROOT / 'scripts/check-firmus-decode-kernels.py').read_text().split('def prefill_validate_candidate(', 1)[1].split('\ndef ', 1)[0]
    assert source.index('prefill_dump_v(') < source.index('"FP8 bytes"')
    # Keep all three poisoned-image / changed-input replays and exact payload gates.
    assert 'range(3)' in source and 'fill_(0x7f)' in source
    assert 'common["vc"][1:rows+1]' in source
    assert re.search(r'assert torch.equal\(dump, common\["image"\]\)', source)
