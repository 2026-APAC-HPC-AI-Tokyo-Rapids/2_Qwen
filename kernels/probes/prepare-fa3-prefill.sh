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
# Compile the same N224 kernel twice; only the FP8 epilogue's runtime branch differs.
epilogue=sources['fa3']/'hopper/epilogue_fwd.hpp'
s=epilogue.read_text()
assert s.count('if (params.ptr_O_fp8) {') == 2
s=s.replace('if (params.ptr_O_fp8) {','if (APAC_PREFILL_FP8_ONLY || params.ptr_O_fp8) {')
epilogue.write_text(s)
scheduler=sources['fa3']/'hopper/flash_prepare_scheduler.cu'
s=scheduler.read_text().replace('int num_warps = cutlass::ceil_div(params.b, 31);','assert(params.b >= 1 && params.b <= 4); int num_warps = 1;')
s=s.replace('BOOL_SWITCH(params.varlen_sort_batches, Sort, [&] {','{ constexpr bool Sort = true; [&] {').replace('NUM_WARP_SWITCH(num_warps, NumWarps, [&] {','{ constexpr int NumWarps = 1; [&] {')
s=s.replace('        });','        }(); }').replace('    });','    }(); }')
scheduler.write_text(s)
Path('fa3-includes').write_text(str(sources['fa3']/'hopper')+'\n'+str(sources['cutlass']/'include')+'\n')
PY
mapfile -t includes <fa3-includes
module load cuda/12.9.1
for variant in 0 1; do
  binary=fa3-prefill.so
  if [[ "${variant}" == 1 ]]; then binary=fa3-prefill-candidate.so; fi
  nvcc --std=c++17 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda -O3 --ptxas-options=-v,-warn-spills -ccbin /opt/rh/gcc-toolset-13/root/usr/bin/g++ -gencode arch=compute_90a,code=sm_90a -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DNDEBUG -I"${includes[0]}" -I"${includes[1]}" --shared --cudart static -Xcompiler -fPIC -Xlinker -Bsymbolic -Dflash="apac_fa3_probe_${variant}" fa3-prefill.cu "${includes[0]}/flash_prepare_scheduler.cu" -DAPAC_PREFILL_FP8_ONLY="${variant}" -o "${binary}" >"${RUNTIME_DIR}/logs/build-${variant}.log" 2>&1 || { tail -100 "${RUNTIME_DIR}/logs/build-${variant}.log"; exit 1; }
  cat "${RUNTIME_DIR}/logs/build-${variant}.log"
done
python3 - <<'PY_CHECK'
import ctypes, os, re
from pathlib import Path
spills=[]
for i,name in enumerate(("fa3-prefill.so","fa3-prefill-candidate.so")):
    log=(Path(os.environ["RUNTIME_DIR"])/f"logs/build-{i}.log").read_text()
    values=[tuple(map(int,m)) for m in re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads",log)]
    spills.append(max((sum(v) for v in values),default=0))
    assert "C7512" not in log, "WGMMA serialization"
    lib=ctypes.CDLL(str(Path(name).resolve()))
    lib.prefill_shared_bytes.argtypes=[ctypes.c_int]
    lib.prefill_shared_bytes.restype=ctypes.c_int
    size=lib.prefill_shared_bytes(0)
    print(f"host_resources fp8_only={i} shared_bytes={size} spill_bytes={spills[-1]}",flush=True)
    assert size==191488 and size<=227*1024, (size,"Unexpected N224 shared-memory footprint")
assert spills[1] < spills[0], (spills,"FP8-only specialization did not reduce spills")
PY_CHECK
jq -n --arg commit "${GITHUB_SHA}" '{commit:$commit,kernel:"FA3 Prefill fused group-128 FP8 output, N224 runtime epilogue versus FP8-only specialization",scope:"one GPU, no serving"}' >provenance.json
