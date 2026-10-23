#!/usr/bin/env bash
set -euo pipefail
source "${GITHUB_WORKSPACE}/kernels/probes/prepare-sglang.sh"
cp "${GITHUB_WORKSPACE}/kernels/probes/fa3-prefill.cu" "${GITHUB_WORKSPACE}/scripts/check-firmus-decode-kernels.py" .
cp "${GITHUB_WORKSPACE}/kernels/probes/fa3-prefill.py" bench.py
python3 - <<'PY'
from pathlib import Path
import hashlib, os, runpy, subprocess, tarfile
root=Path(os.environ['GITHUB_WORKSPACE'])
pins=runpy.run_path(str(root/'kernels/fa3-verify/build.py'))['PINS']
sources={}
for name,(repo,commit,digest) in pins.items():
    archive=Path(name+'.tar.gz')
    subprocess.run(['curl','-fLsS',f'https://github.com/{repo}/archive/{commit}.tar.gz','-o',str(archive)],check=True)
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==digest
    dest=Path(name); dest.mkdir()
    with tarfile.open(archive) as tar: tar.extractall(dest,filter='data')
    sources[name]=next(dest.iterdir())
# Only reuse the launch-template override; Decode's B32..36 scheduler is unrelated.
patch=(root/'kernels/fa3-verify/fixed-specializations.patch').read_text().split('--- a/hopper/flash_prepare_scheduler.cu')[0]
subprocess.run(['patch','--batch','--fuzz=0','-p1'],input=patch,universal_newlines=True,cwd=sources['fa3'],check=True)
subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(root/'kernels/probes/fa3-fp8-output.patch')],cwd=sources['fa3'],check=True)
scheduler=sources['fa3']/'hopper/flash_prepare_scheduler.cu'
s=scheduler.read_text().replace('int num_warps = cutlass::ceil_div(params.b, 31);','assert(params.b >= 1 && params.b <= 4); int num_warps = 1;')
s=s.replace('BOOL_SWITCH(params.varlen_sort_batches, Sort, [&] {','{ constexpr bool Sort = true; [&] {').replace('NUM_WARP_SWITCH(num_warps, NumWarps, [&] {','{ constexpr int NumWarps = 1; [&] {')
s=s.replace('        });','        }(); }').replace('    });','    }(); }')
scheduler.write_text(s)
Path('fa3-includes').write_text(str(sources['fa3']/'hopper')+'\n'+str(sources['cutlass']/'include')+'\n')
PY
mapfile -t includes <fa3-includes
module load cuda/12.9.1
nvcc --std=c++17 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda -O3 --ptxas-options=-v,-warn-spills -ccbin /opt/rh/gcc-toolset-13/root/usr/bin/g++ -gencode arch=compute_90a,code=sm_90a -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DNDEBUG -I"${includes[0]}" -I"${includes[1]}" --shared --cudart static -Xcompiler -fPIC fa3-prefill.cu "${includes[0]}/flash_prepare_scheduler.cu" -o fa3-prefill.so >"${RUNTIME_DIR}/logs/build.log" 2>&1 || { tail -100 "${RUNTIME_DIR}/logs/build.log"; exit 1; }
cat "${RUNTIME_DIR}/logs/build.log"
jq -n --arg commit "${GITHUB_SHA}" '{commit:$commit,kernel:"FA3 Prefill N192 fused group-128 FP8 output",scope:"one GPU, no serving"}' >provenance.json
