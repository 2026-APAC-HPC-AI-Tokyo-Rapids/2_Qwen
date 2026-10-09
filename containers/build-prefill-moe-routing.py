"""Build one routing helper TU on a CPU build host; no GEMM template dependency."""
import hashlib
import json
import os
import shutil
import sysconfig
from pathlib import Path

from torch.utils.cpp_extension import load

root = Path('/opt/apac/prefill-moe-routing')
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
os.environ['MAX_JOBS'] = '1'
module = load(name='apac_prefill_moe_routing', sources=[str(root / 'moe_routing.cu')], extra_cuda_cflags=['-O3', '-std=c++17', '-Xptxas=-v'], build_directory=str(root), verbose=True)
source = Path(module.__file__)
destination = Path(sysconfig.get_paths()['platlib']) / source.name
shutil.copyfile(source, destination)
manifest = {'schema': 'prefill-moe-routing/v1', 'architecture': 'sm_90', 'source_sha256': hashlib.sha256((root / 'moe_routing.cu').read_bytes()).hexdigest(), 'build_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'module_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(), 'module_path': str(destination), 'experts': 128, 'topk': 8, 'tile_rows': 512, 'max_tokens': 8192, 'intermediate_sizes': [384, 768], 'production_invalid_ids': 'unsupported', 'floating_point_arithmetic': False, 'compile_max_jobs': 1, 'translation_units': 1, 'kernels': 3, 'compiled': True}
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest))
