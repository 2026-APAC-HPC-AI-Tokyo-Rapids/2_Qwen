"""Build an isolated diagnostic header and prove the ordinary header/SASS unchanged."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HEADER = Path('deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh')
# Same host-selected capacity and geometry as PP4/EP1, not the token count.
TEMPLATE = '8448,4096,1536,128,8,128,128,256,128,92160,1474560,4,64,64,256,132,1,cute::numeric_limits<float>::infinity(),true,0,true,true,false,false,false,true,false,true'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compile_one(nvcc, include, dependencies, directory, name, diagnostic):
    source = directory / (name + '.cu')
    suffix = '_diag' if diagnostic else ''
    source.write_text(f'#include <deep_gemm/impls/sm90_fp8_mega_moe{suffix}.cuh>\nextern "C" void* instantiate() {{ return reinterpret_cast<void*>(&deep_gemm::sm90_fp8_mega_moe{suffix}_impl<{TEMPLATE}>); }}\n')
    command = [str(nvcc), '--cubin', '--std=c++20', '--expt-relaxed-constexpr', '--expt-extended-lambda', '-O3', '--ptxas-options=-v,-warn-spills', '-gencode', 'arch=compute_90a,code=sm_90a', '-I'+str(include), '-I'+str(dependencies), str(source), '-o', str(source.with_suffix('.cubin'))]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    log = result.stdout + result.stderr
    source.with_suffix('.log').write_text(log)
    assert result.returncode == 0, log
    sass = subprocess.check_output([str(nvcc.parent/'cuobjdump'), '-sass', str(source.with_suffix('.cubin'))], text=True)
    source.with_suffix('.sass').write_text(sass)
    encoded = b''.join(int(w, 16).to_bytes(8, 'little') for w in re.findall(r'/\* (0x[0-9a-f]{16}) \*/', sass))
    assert encoded
    source.with_suffix('.sass.bin').write_bytes(encoded)
    return {'sass_encoding_sha256': sha(encoded), 'encoding_bytes': len(encoded), 'registers': re.findall(r'Used (\d+) registers', log), 'max_referenced_register': max(map(int, re.findall(r'\bR(\d+)\b', sass))), 'spill_store_load_bytes': re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads', log), 'c75xx': [line for line in log.splitlines() if re.search(r'C75\d+', line)], 'qgmma': sass.count('QGMMA.'), 'depbar': sass.count('WARPGROUP.DEPBAR'), 'command': command}


def build(package, patch_dir, config, nvcc, output, dependencies=None):
    dependencies = dependencies or package/'include'
    original = package/'include/deep_gemm/impls/sm90_fp8_mega_moe.cuh'
    before = original.read_bytes()
    with tempfile.TemporaryDirectory(prefix='r18-', dir=output.parent) as tmp:
        directory = Path(tmp)
        overlay = directory/'overlay'
        shutil.copytree(package/'include', overlay/'deep_gemm/include')
        for p in json.loads(config.read_text())['deep_gemm_overlay_patches']:
            subprocess.run(['git', 'apply', str(patch_dir.parent/Path(p).relative_to('patches'))], cwd=overlay, check=True)
        control_header = (overlay/HEADER).read_bytes()
        serving_include = overlay/'deep_gemm/include'
        baseline = compile_one(nvcc, serving_include, dependencies, output.parent, 'r18-before', False)
        subprocess.run(['git', 'apply', str(patch_dir/'split-timeline.patch')], cwd=overlay, check=True)
        generated = overlay/HEADER
        target = serving_include/'deep_gemm/impls/sm90_fp8_mega_moe_diag.cuh'
        generated.rename(target)
        generated.write_bytes(control_header)
        shutil.copy2(patch_dir/'trace.cuh', target.with_name('r18_trace.cuh'))
        ordinary = compile_one(nvcc, serving_include, dependencies, output.parent, 'r18-after', False)
        diagnostic = compile_one(nvcc, serving_include, dependencies, output.parent, 'r18-diagnostic', True)
        assert baseline['sass_encoding_sha256'] == ordinary['sass_encoding_sha256'], 'Serving SASS changed'
        assert original.read_bytes() == before
        # Install ONLY new diagnostic headers. The package's ordinary header stays baked;
        # provisioning applies its ordinary overlays as before.
        for name in ('sm90_fp8_mega_moe_diag.cuh', 'r18_trace.cuh'):
            shutil.copy2(target.with_name(name), original.with_name(name))
        result = {'schema':'prefill-megamoe-diag-build/v1', 'toolchain':subprocess.check_output([str(nvcc), '--version'], text=True), 'template':TEMPLATE, 'serving_header_sha256':sha(control_header), 'diagnostic_header_sha256':sha(target.read_bytes()), 'trace_sha256':sha((patch_dir/'trace.cuh').read_bytes()), 'serving_sass_equal':True, 'before':baseline, 'after':ordinary, 'diagnostic':diagnostic, 'gpu_validated':False}
        output.write_text(json.dumps(result, indent=2)+'\n')
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--patch-dir', type=Path, default=Path('/opt/apac/deep-gemm-diag'))
    p.add_argument('--config', type=Path, default=Path('/opt/apac/firmus-prefill-pp4.json'))
    p.add_argument('--nvcc', type=Path, default=Path('/usr/local/cuda/bin/nvcc'))
    p.add_argument('--output', type=Path, default=Path('/opt/apac/prefill-megamoe-diag-build.json'))
    p.add_argument('--dependencies', type=Path)
    a = p.parse_args()
    build(a.package, a.patch_dir, a.config, a.nvcc, a.output, a.dependencies)


if __name__ == '__main__':
    main()
