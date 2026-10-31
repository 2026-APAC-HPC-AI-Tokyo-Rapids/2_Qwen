"""Build the exact checked-out FA3 patch chain and the split-PV probe, without CUDA initialization."""
import ctypes
import hashlib
import json
import os
import re
import runpy
import shutil
import subprocess
import tarfile
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(os.environ['GITHUB_WORKSPACE'])
    runtime = Path(os.environ['RUNTIME_DIR'])
    source = runtime / 'source'
    logs = runtime / 'logs'
    record = runtime / 'results/source'
    for path in (source, logs, record):
        path.mkdir(parents=True, exist_ok=True)
    pins = runpy.run_path(str(root / 'kernels/fa3-verify/build.py'))['PINS']
    trees = {}
    for name, (repo, commit, digest) in pins.items():
        archive = source / (name + '.tar.gz')
        subprocess.run(['curl', '-fLsS', f'https://github.com/{repo}/archive/{commit}.tar.gz', '-o', str(archive)], check=True)
        assert sha(archive) == digest, (name, 'archive digest')
        target = source / name
        target.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(target)  # Exact SHA256 verified above.
        trees[name] = next(target.iterdir())
    chain = re.findall(r'str\(root / "([a-z0-9-]+\.patch)"\)', (root / 'kernels/fa3-verify/build.py').read_text())
    assert chain[-1] == 'prefill-geometry.patch' and len(chain) == 14, chain
    patches = []
    for name in chain:
        patch = root / ('kernels/probes/fa3-fp8-output.patch' if name == 'fp8-output.patch' else 'kernels/fa3-verify/' + name)
        subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(patch)], cwd=trees['fa3'], check=True)
        shutil.copy2(patch, record / patch.name)
        patches.append({'path': str(patch.relative_to(root)), 'sha256': sha(patch)})
    filenames = ('fa3-split-pv.cu', 'fa3-split-pv-layout.cu', 'fa3-split-pv.py', 'fa3-prefill-pipe.cu', 'fa3-prefill-pipe.py')
    for name in filenames:
        shutil.copy2(root / 'kernels/probes' / name, source / name)
        shutil.copy2(source / name, record / name)
    (source / 'pf-variants.h').write_text('#define PF_VARIANTS 224\n')
    flags = ['-O3', '-std=c++17', '--use_fast_math', '--expt-relaxed-constexpr', '--expt-extended-lambda', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a', '-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED', '-DCUTLASS_DEBUG_TRACE_LEVEL=0', '-DNDEBUG', '-Xptxas=-v,--warn-on-spills', '-Xcompiler=-fPIC', '-Dflash=apac_fa3_private', '-DFlash_fwd_params=ApacFA3VerifyParams']
    if os.environ.get('CXX'):
        flags += ['-ccbin', os.environ['CXX']]
    includes = [source, root / 'kernels/fa3-verify', trees['fa3'] / 'hopper', trees['cutlass'] / 'include', trees['cutlass'] / 'tools/util/include']
    flags += ['-I' + str(p) for p in includes]
    commands = []

    def run(name, command):
        start = time.monotonic()
        with (logs / (name + '.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        commands.append({'name': name, 'argv': command, 'exit_code': result.returncode, 'seconds': time.monotonic() - start})
        (record / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        if result.returncode:
            print((logs / (name + '.log')).read_text()[-12000:], flush=True)
            raise SystemExit(result.returncode)

    run('layout-build', ['nvcc', *flags, str(source / 'fa3-split-pv-layout.cu'), '-o', str(source / 'layout-host')])
    run('layout-check', [str(source / 'layout-host')])
    layout = json.loads((logs / 'layout-check.log').read_text())
    assert layout['shared_bytes'] == 217088 and not any(v for k, v in layout.items() if k.endswith('_errors') or k == 'noncontiguous_words'), layout
    (record / 'layout.json').write_text(json.dumps(layout, indent=2) + '\n')
    units = {'split-pv': source / 'fa3-split-pv.cu', 'control': source / 'fa3-prefill-pipe.cu', 'scheduler': trees['fa3'] / 'hopper/flash_prepare_scheduler.cu'}
    for name, path in units.items():
        run('build-' + name, ['nvcc', *flags, '-c', str(path), '-o', str(source / (name + '.o'))])
    for name, objects in [('split-pv', ['split-pv']), ('control', ['control', 'scheduler'])]:
        run('link-' + name, ['nvcc', *[str(source / (obj + '.o')) for obj in objects], '--shared', '--cudart', 'static', '-Xlinker=-Bsymbolic', '-o', str(source / (name + '.so'))])
        run('sass-' + name, ['cuobjdump', '--dump-sass', str(source / (name + '.o'))])
        run('resources-' + name, ['cuobjdump', '--dump-resource-usage', str(source / (name + '.o'))])
        assert not re.search(r'C751[139]', (logs / ('build-' + name + '.log')).read_text()), 'WGMMA serialization warning'
    abi = {}
    for name in ('control', 'split-pv'):
        library = ctypes.CDLL(str(source / (name + '.so')))
        abi[name] = {'count': library.pf_count(), 'shared_bytes': library.pf_resources(0)}
        assert abi[name]['count'] == 1 and 0 < abi[name]['shared_bytes'] <= 227*1024
    (record / 'abi.json').write_text(json.dumps(abi, indent=2) + '\n')
    sass = (logs / 'sass-split-pv.log').read_text()
    qk_start = re.search(r'USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0x98', sass)
    producer_start = re.search(r'USETMAXREG.DEALLOC.CTAPOOL 0x18', sass)
    assert qk_start and producer_start
    qk_code = sass[qk_start.start():producer_start.start()]
    assert not re.search(r'\b(?:LDL|STL)\b', qk_code), 'QK/softmax role spilled registers'
    assert 'QGMMA.64x192x32' in qk_code and 'QGMMA.64x128x32' not in qk_code, 'PV still in QK role'
    (record / 'sass-gate.json').write_text(json.dumps({'qk_role_has_no_local_load_or_store': True, 'qk_role_has_no_pv_mma': True, 'gpu_executed': False}, indent=2) + '\n')
    data = {'schema': 'fa3-split-pv-source/v1', 'commit': os.environ.get('GITHUB_SHA', 'local'), 'pins': pins, 'patches': patches, 'gpu_executed': False, 'compiler': subprocess.check_output(['nvcc', '--version'], text=True), 'sources': {name: sha(source / name) for name in filenames}, 'binaries': {name: sha(source / (name + '.so')) for name in ('control', 'split-pv')}, 'patched_headers': {name: sha(trees['fa3'] / 'hopper' / name) for name in ('utils.h', 'softmax.h', 'mainloop_fwd_sm90_tma_gmma_ws.hpp', 'flash_fwd_launch_template.h')}}
    (record / 'build.json').write_text(json.dumps(data, indent=2) + '\n')
    provenance = {'commit': data['commit'], 'kernel': 'FA3 M128/N192 QK-softmax/PV split warpgroups', 'scope': 'one GPU; numerical gate before timing; no serving', 'calibration': True}
    (source / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'cpu_build_passed': True, 'layout': layout}), flush=True)


if __name__ == '__main__':
    main()
