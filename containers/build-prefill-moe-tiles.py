"""Compile the finite SM90 candidates without GPUs; install a prebuilt module."""
import hashlib
import json
import os
import shutil
import sysconfig
from pathlib import Path

from torch.utils.cpp_extension import load

root = Path('/opt/apac/prefill-moe-tiles')
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
os.environ['MAX_JOBS'] = '1'
module = load(name='apac_prefill_moe_tiles', sources=[str(root / 'moe_tiles.cu'), *map(str, sorted(root.glob('moe_grid_cluster_*.cu')))], extra_include_paths=['/opt/apac/cutlass/include', '/opt/apac/cutlass/tools/util/include'], extra_cuda_cflags=['-O3', '-DNDEBUG', '-DCUTE_USE_PACKED_TUPLE=1', '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1', '-DCUTLASS_DEBUG_TRACE_LEVEL=0', '--expt-relaxed-constexpr', '--expt-extended-lambda', '-std=c++17', '-Xptxas=-v'], build_directory=str(root), verbose=True)
source = Path(module.__file__)
destination = Path(sysconfig.get_paths()['platlib']) / source.name
shutil.copyfile(source, destination)
configurations = json.loads((root / 'grid-configurations.json').read_text())
resolved = {entry['mode']: entry for entry in module.configurations()}
for entry in configurations:
    entry.update(resolved[entry['mode']])
    entry['compile_status'] = 'compiled'
source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.iterdir()) if path.suffix in ('.cu', '.cuh') or path.name == 'grid-configurations.json'}
manifest = {'schema': 'prefill-moe-grid/v2', 'cutlass_revision': '57e3cfb47a2d9e0d46eb6335c3dc411498efa198', 'architecture': 'sm_90a', 'source_sha256': source_hashes['moe_tiles.cu'], 'source_hashes': source_hashes, 'module_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(), 'module_path': str(destination), 'configurations': configurations, 'compile_max_jobs': 1, 'translation_units': 5}
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest))
