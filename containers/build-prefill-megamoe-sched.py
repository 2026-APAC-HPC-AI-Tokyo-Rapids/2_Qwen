"""Build L from the final PP4 control, preserving ordinary source and SASS."""
import argparse
import hashlib
import json
import runpy
import shutil
import subprocess
import tempfile
from pathlib import Path


def build(package, patch_dir, config, nvcc, output, dependencies=None):
    helper = runpy.run_path(str(Path(__file__).with_name('build-prefill-megamoe-diag.py')))
    original = package/'include/deep_gemm/impls/sm90_fp8_mega_moe.cuh'
    before = original.read_bytes()
    with tempfile.TemporaryDirectory(prefix='r20-', dir=output.parent) as tmp:
        overlay = Path(tmp)/'overlay'
        shutil.copytree(package/'include', overlay/'deep_gemm/include')
        for p in json.loads(config.read_text())['deep_gemm_overlay_patches']:
            subprocess.run(['git', 'apply', str(patch_dir.parent/Path(p).relative_to('patches'))], cwd=overlay, check=True)
        include = overlay/'deep_gemm/include'
        control = overlay/helper['HEADER']
        control_bytes = control.read_bytes()
        baseline = helper['compile_one'](nvcc, include, dependencies or package/'include', output.parent, 'r20-before', False)
        subprocess.run(['git', 'apply', str(patch_dir/'expert-greedy.patch')], cwd=overlay, check=True)
        target = control.with_name('sm90_fp8_mega_moe_sched.cuh')
        control.rename(target)
        control.write_bytes(control_bytes)
        ordinary = helper['compile_one'](nvcc, include, dependencies or package/'include', output.parent, 'r20-after', False)
        candidate = helper['compile_one'](nvcc, include, dependencies or package/'include', output.parent, 'r20-L', False, suffix='_sched')
        assert baseline['sass_encoding_sha256'] == ordinary['sass_encoding_sha256'], 'control SASS changed'
        assert not candidate['c75xx'], 'C75xx'
        assert len(candidate['spill_store_load_bytes']) == len(baseline['spill_store_load_bytes']) == 1
        assert all(int(a) <= int(b) for a,b in zip(candidate['spill_store_load_bytes'][0],baseline['spill_store_load_bytes'][0],strict=True)), 'L spills exceed control'
        assert original.read_bytes() == before
        shutil.copy2(target, original.with_name(target.name))
        result = {'schema':'prefill-megamoe-sched-build/v1', 'toolchain':subprocess.check_output([str(nvcc), '--version'],text=True), 'template':helper['TEMPLATE'], 'control_header_sha256':hashlib.sha256(control_bytes).hexdigest(), 'candidate_header_sha256':hashlib.sha256(target.read_bytes()).hexdigest(), 'control_sass_equal':True, 'before':baseline, 'after':ordinary, 'candidate':candidate, 'gpu_validated':False}
        output.write_text(json.dumps(result, indent=2)+'\n')
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--patch-dir', type=Path, default=Path('/opt/apac/deep-gemm-sched'))
    p.add_argument('--config', type=Path, default=Path('/opt/apac/firmus-prefill-pp4.json'))
    p.add_argument('--nvcc', type=Path, default=Path('/usr/local/cuda/bin/nvcc'))
    p.add_argument('--output', type=Path, default=Path('/opt/apac/prefill-megamoe-sched-build.json'))
    p.add_argument('--dependencies', type=Path)
    a = p.parse_args()
    build(a.package, a.patch_dir, a.config, a.nvcc, a.output, a.dependencies)
