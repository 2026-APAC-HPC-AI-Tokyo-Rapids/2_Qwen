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
os.environ['MAX_JOBS'] = '2'
module = load(name='apac_prefill_moe_tiles', sources=[str(root / 'moe_tiles.cu')], extra_include_paths=['/opt/apac/cutlass/include', '/opt/apac/cutlass/tools/util/include'], extra_cuda_cflags=['-O3', '-DNDEBUG', '-DCUTE_USE_PACKED_TUPLE=1', '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1', '-DCUTLASS_DEBUG_TRACE_LEVEL=0', '--expt-relaxed-constexpr', '--expt-extended-lambda', '-std=c++17'], build_directory=str(root), verbose=True)
source = Path(module.__file__)
destination = Path(sysconfig.get_paths()['platlib']) / source.name
shutil.copyfile(source, destination)
manifest = {'schema': 'prefill-moe-tiles/v1', 'cutlass_revision': '57e3cfb47a2d9e0d46eb6335c3dc411498efa198', 'architecture': 'sm_90a', 'source_sha256': hashlib.sha256((root / 'moe_tiles.cu').read_bytes()).hexdigest(), 'module_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(), 'module_path': str(destination), 'modes': {'0': 'stock128x128cluster1x2', '1': 'down128x128cluster1x1', '3': 'down64x128pingpong2x1', '4': 'up64x128pingpong2x1'}}
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest))
