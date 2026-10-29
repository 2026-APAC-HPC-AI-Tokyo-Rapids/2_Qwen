"""Compile real call sites against the combined Prefill/Decode template ABI."""
import re
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / 'kernels/fa3-verify'


def test_combined_template_keeps_prefill_variant_and_decode_merge_distinct(tmp_path):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('g++ required for CPU template ABI check')
    # The last patch that changes the template signature (build.py applies it last).
    patch = (ROOT / 'decode-wide-h2.patch').read_text()
    new_lines = '\n'.join(line[1:] for line in patch.splitlines() if line.startswith((' ', '+')) and not line.startswith('+++'))
    signature = re.search(r'template <int Arch,.*?\nvoid run_flash_fwd', new_lines, re.DOTALL)[0].removesuffix('\nvoid run_flash_fwd')
    builder = runpy.run_path(str(ROOT / 'build.py'))
    tasks = builder['generate_dispatch'](ROOT, tmp_path) + builder['generate_prefill_dispatch'](ROOT, tmp_path)
    text = '''#include <cassert>
namespace cutlass { struct float_e4m3_t {}; struct bfloat16_t {}; }
struct Flash_fwd_params { int variant = -1; bool merge = false; int ctas = 0; int heads = 0; };
using cudaStream_t = void*;
''' + signature + '''
void run_flash_fwd(Flash_fwd_params& p, cudaStream_t, int*) {
    static_assert(FixedPrefill || PrefillVariant == 0);
    static_assert(!FixedPrefill || !VerifyMerge);
    p.variant = PrefillVariant; p.merge = VerifyMerge; p.ctas = VerifyCtas; p.heads = VerifyHeads;
}
'''
    calls = []
    for path, name, _ in tasks:
        generated = path.read_text()
        definitions = re.findall(r'#define (\w+) (\S+)', generated)
        prefill = name.startswith('prefill_')
        source = (ROOT / ('prefill_variant.cu' if prefill else 'p128split.cu')).read_text()
        source = '\n'.join(line for line in source.splitlines() if not line.startswith('#include'))
        for macro, value in definitions:
            source = re.sub(r'\b' + macro + r'\b', value, source)
        text += source + '\n'
        values = dict(definitions)
        calls.append(f'apac_fa3_verify::{name}(p, nullptr, nullptr); assert(p.variant == {values.get("PREFILL_VARIANT", "0")} && p.merge == {values.get("VERIFY_MERGE", "false")} && p.ctas == {values.get("VERIFY_CTAS", "1")} && p.heads == {values.get("VERIFY_HEADS", "1")});')
    path = tmp_path / 'calls.cpp'
    path.write_text(text + '\nint main() { Flash_fwd_params p;\n' + '\n'.join(calls) + '\n}\n')
    binary = tmp_path / 'calls'
    subprocess.run([compiler, '-std=c++17', str(path), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
