#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --compile-only ]]; then
    install -d "${RUNTIME_DIR}/source" "${RUNTIME_DIR}/logs" "${RUNTIME_DIR}/results"
    cd "${RUNTIME_DIR}/source"
else
    source "${GITHUB_WORKSPACE}/kernels/probes/prepare-sglang.sh"
fi
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
    with tarfile.open(archive) as tar: tar.extractall(dest)  # Archives are SHA256-pinned above.
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
python3 "${GITHUB_WORKSPACE}/kernels/probes/fa3-prefill-instrument.py" "${includes[0]}"
if [[ -f "${RUNTIME_DIR}/scripts/build-env.sh" ]]; then source "${RUNTIME_DIR}/scripts/build-env.sh"; else source "${GITHUB_WORKSPACE}/platforms/genkai/kernel-probe/build-env.sh"; fi
variants=(control nobarrier regs232 lazy lazy-counter)
if [[ "${1:-}" == --profile ]]; then variants=(control); fi
for variant in "${variants[@]}"; do
    flags=()
    n=224
    case "${variant}" in
        nobarrier) flags+=(-DFA3_PROBE_NOBARRIER=1) ;;
        regs232) flags+=(-DFA3_PROBE_REGS232=1) ;;
        lazy) flags+=(-DFA3_PROBE_LAZY=1) ;;
        lazy-counter) flags+=(-DFA3_PROBE_LAZY=1 -DFA3_PROBE_COUNTER=1) ;;
    esac
    if [[ "${variant}" != control ]]; then flags+=(-DFA3_PROBE_SINGLE_N="${n}"); fi
    # Resource-main queries only constexpr host resources, never CUDA attributes.
    nvcc --std=c++17 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda -O3 -lineinfo --ptxas-options=-v,-warn-spills -ccbin "${CXX}" -gencode arch=compute_90a,code=sm_90a -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DNDEBUG -I"${includes[0]}" -I"${includes[1]}" "${flags[@]}" -DFA3_PROBE_RESOURCE_MAIN="${n}" -Xcompiler -fPIC -c fa3-prefill.cu -o "${variant}.o" >"${RUNTIME_DIR}/logs/build-${variant}.log" 2>&1 || { tail -100 "${RUNTIME_DIR}/logs/build-${variant}.log"; exit 1; }
    nvcc "${variant}.o" "${includes[0]}/flash_prepare_scheduler.cu" --std=c++17 -O3 --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_90a,code=sm_90a -ccbin "${CXX}" -I"${includes[0]}" -I"${includes[1]}" -Xcompiler -fPIC --shared --cudart static -o "fa3-prefill-${variant}.so" >>"${RUNTIME_DIR}/logs/build-${variant}.log" 2>&1
    nvcc "${variant}.o" -L. -l:fa3-prefill-${variant}.so --cudart static -Xlinker -rpath -Xlinker '$ORIGIN' -o "resource-${variant}"
    ./resource-"${variant}" | tee "${RUNTIME_DIR}/logs/resource-${variant}.log"
    cuobjdump -sass "${variant}.o" >"${RUNTIME_DIR}/logs/sass-${variant}.txt"
    python3 "${GITHUB_WORKSPACE}/kernels/probes/fa3-prefill-sass.py" "${RUNTIME_DIR}/logs/sass-${variant}.txt" >"${RUNTIME_DIR}/results/sass-${variant}.txt"
    cat "${RUNTIME_DIR}/logs/build-${variant}.log" "${RUNTIME_DIR}/results/sass-${variant}.txt"
done
cp fa3-prefill-control.so fa3-prefill.so
python3 - <<'PYMETA'
import json, os
from pathlib import Path
runtime = Path(os.environ['RUNTIME_DIR'])
info = {name: {'ptxas': (runtime/'logs'/('build-'+name+'.log')).read_text(), 'resource': (runtime/'logs'/('resource-'+name+'.log')).read_text()} for name in ('control','nobarrier','regs232','lazy','lazy-counter') if (runtime/'logs'/('build-'+name+'.log')).exists()}
(runtime/'results'/'ptxas.json').write_text(json.dumps(info, indent=2))
PYMETA
jq -n --arg commit "${GITHUB_SHA}" '{commit:$commit,kernel:"FA3 Prefill M0/M1",scope:"one GPU, no serving",calibration:true}' >provenance.json
