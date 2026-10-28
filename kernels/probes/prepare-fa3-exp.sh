#!/usr/bin/env bash
# Two finite fixed-Prefill builds; all work here is CPU-only.
set -euo pipefail
if [[ "${1:-}" == --compile-only ]]; then
    install -d "${RUNTIME_DIR}/source" "${RUNTIME_DIR}/logs" "${RUNTIME_DIR}/results"
    cd "${RUNTIME_DIR}/source"
else
    source "${GITHUB_WORKSPACE}/kernels/probes/prepare-sglang.sh"
fi
install -d "${RUNTIME_DIR}/results/source" "${RUNTIME_DIR}/metadata"
install -m 0644 "${GITHUB_WORKSPACE}/kernels/probes/fa3-exp.cu" .
install -m 0644 "${GITHUB_WORKSPACE}/kernels/probes/fa3-exp.py" bench.py
cat >"${RUNTIME_DIR}/source/probe" <<'PROBE'
#!/usr/bin/env bash
set -euo pipefail
: "${RUNTIME_DIR:?}"
exec /opt/sglang/venv/bin/python "${RUNTIME_DIR}/source/bench.py" --output "${RUNTIME_DIR}/results/fa3-exp"
PROBE
chmod 0755 "${RUNTIME_DIR}/source/probe"
# The probe mounts metadata over /opt/apac/fa3-verify, so preserve the pinned
# native control and its candidate table before entering that container.
unsquashfs -no-progress -no-xattrs -processors 1 -d "${RUNTIME_DIR}/native-control" "${RUNTIME_DIR}/sglang.sqsh" opt/apac/fa3-verify/libapac_fa3_verify.so opt/apac/fa3-verify/manifest.json opt/apac/fa3-verify/candidates.json opt/apac/fa3-verify/prefill-candidates.json >"${RUNTIME_DIR}/logs/extract-native-control.log" 2>&1
chmod -R u+w "${RUNTIME_DIR}/native-control"
mv "${RUNTIME_DIR}/native-control/opt/apac/fa3-verify/"* "${RUNTIME_DIR}/metadata/"
rm -rf -- "${RUNTIME_DIR}/native-control"
[[ -s "${RUNTIME_DIR}/metadata/libapac_fa3_verify.so" && -s "${RUNTIME_DIR}/metadata/manifest.json" ]]
cp "${RUNTIME_DIR}/metadata/manifest.json" "${RUNTIME_DIR}/results/source/image-fa3-manifest.json"
python3 - <<'PY'
from pathlib import Path
import hashlib, json, os, runpy, shutil, subprocess, tarfile
root=Path(os.environ['GITHUB_WORKSPACE'])
runtime=Path(os.environ['RUNTIME_DIR'])
record=runtime/'results/source'
pins=runpy.run_path(str(root/'kernels/fa3-verify/build.py'))['PINS']
manifest=json.loads((runtime/'metadata/manifest.json').read_text())
for table in ('candidates.json','prefill-candidates.json'):
    path=runtime/'metadata'/table
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['source_sha256'][table], (table,'image table mismatch')
    shutil.copy2(path,record/table)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(runtime/'metadata/libapac_fa3_verify.so')==manifest['binary_sha256'], 'Pinned FA3 binary does not match image manifest'
sources={}
for name,(repo,commit,digest) in pins.items():
    pin=manifest['pins'][name]
    assert (pin['repository'],pin['commit'],pin['archive_sha256'])==(repo,commit,digest), (name,'image/source pin mismatch')
    archive=record/(name+'.tar.gz')
    subprocess.run(['curl','-fLsS',f'https://github.com/{repo}/archive/{commit}.tar.gz','-o',str(archive)],check=True)
    assert sha(archive)==digest
    target=Path(name); target.mkdir()
    with tarfile.open(archive) as tar: tar.extractall(target)
    sources[name]=next(target.iterdir())
patches=[root/'kernels/fa3-verify/fixed-specializations.patch',root/'kernels/probes/fa3-fp8-output.patch',*[root/'kernels/fa3-verify'/name for name in ('fixed-prefill.patch','prefill-variants.patch','prefill-overlap.patch','prefill-pv.patch','decode-merge.patch')]]
patch_records=[]
for patch in patches:
    image_name='fp8-output.patch' if patch.name=='fa3-fp8-output.patch' else patch.name
    assert sha(patch)==manifest['source_sha256'][image_name], (image_name,'image/control patch mismatch')
    subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(patch)],cwd=sources['fa3'],check=True)
    shutil.copy2(patch,record/patch.name)
    patch_records.append({'path':str(patch.relative_to(root)),'sha256':sha(patch)})
softmax=sources['fa3']/'hopper/softmax.h'
control_sha=sha(softmax)
assert control_sha=='24edd56147c62c779b6dfea518d2abae99b2da8a53406259ee3d96b26b486095', 'Unexpected full-chain control softmax'
shutil.copy2(softmax,record/'softmax-control.h')
patch=root/'kernels/probes/fa3-prefill-exp.patch'
subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(patch)],cwd=sources['fa3'],check=True)
shutil.copy2(patch,record/patch.name)
shutil.copy2(softmax,record/'softmax-patched.h')
# The full serving launch header retains an external scheduler symbol even
# with metadata skipped at runtime. Its common TU is linked into both DSOs.
for name in ('decode_merge.cuh','merge_fp8.cuh'):
    path=root/'kernels/fa3-verify'/name
    assert sha(path)==manifest['source_sha256'][name], (name,'image/control header mismatch')
    shutil.copy2(path,Path(name))
Path('fa3-exp-includes').write_text(str(sources['fa3']/'hopper')+'\n'+str(sources['cutlass']/'include')+'\n'+str(sources['cutlass']/'tools/util/include')+'\n')
metadata={'schema':'fa3-exp-source/v1','pins':manifest['pins'],'patches':patch_records,'experiment_patch':{'path':str(patch.relative_to(root)),'sha256':sha(patch)},'softmax_control_sha256':control_sha,'softmax_patched_sha256':sha(softmax),'pinned_native_sha256':sha(runtime/'metadata/libapac_fa3_verify.so'),'manifest_sha256':sha(runtime/'metadata/manifest.json'),'image_source_sha_verified':True}
(record/'source-provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
for name in ('fa3-exp.cu','bench.py','probe'):
    shutil.copy2(Path(name),record/name)
shutil.copy2(root/'kernels/probes/prepare-fa3-exp.sh',record/'prepare-fa3-exp.sh')
PY
mapfile -t includes <fa3-exp-includes
if [[ -f "${RUNTIME_DIR}/scripts/build-env.sh" ]]; then source "${RUNTIME_DIR}/scripts/build-env.sh"; else source "${GITHUB_WORKSPACE}/platforms/genkai/kernel-probe/build-env.sh"; fi
flags=(--std=c++17 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda -O3 -lineinfo --ptxas-options=-v,-warn-spills -ccbin "${CXX}" -gencode arch=compute_90a,code=sm_90a -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DCUTLASS_DEBUG_TRACE_LEVEL=0 -DNDEBUG -I. -I"${includes[0]}" -I"${includes[1]}" -I"${includes[2]}" -Xcompiler -fPIC)
nvcc --version >"${RUNTIME_DIR}/results/source/compiler.txt"
ptxas --version >>"${RUNTIME_DIR}/results/source/compiler.txt"
"${CXX}" --version >>"${RUNTIME_DIR}/results/source/compiler.txt"
sha256sum "$(command -v nvcc)" "$(command -v ptxas)" >"${RUNTIME_DIR}/results/source/compiler.sha256"
printf '%q ' nvcc "${flags[@]}" >"${RUNTIME_DIR}/results/source/compile-flags.txt"
printf '\n' >>"${RUNTIME_DIR}/results/source/compile-flags.txt"
nvcc "${flags[@]}" -c "${includes[0]}/flash_prepare_scheduler.cu" -o scheduler.o >"${RUNTIME_DIR}/logs/build-scheduler.log" 2>&1 || { tail -100 "${RUNTIME_DIR}/logs/build-scheduler.log"; exit 1; }
for variant in control poly; do
    candidate_flags=()
    if [[ "${variant}" == poly ]]; then candidate_flags=(-DAPAC_FA3_PREFILL_EXP_POLY=1); fi
    nvcc "${flags[@]}" "${candidate_flags[@]}" -DFA3_EXP_RESOURCE_MAIN=1 -c fa3-exp.cu -o "${variant}.o" >"${RUNTIME_DIR}/logs/build-${variant}.log" 2>&1 || { tail -100 "${RUNTIME_DIR}/logs/build-${variant}.log"; exit 1; }
    nvcc "${variant}.o" scheduler.o --shared --cudart static -o "fa3-exp-${variant}.so" >>"${RUNTIME_DIR}/logs/build-${variant}.log" 2>&1
    nvcc "${variant}.o" scheduler.o --cudart static -o "resource-${variant}" >>"${RUNTIME_DIR}/logs/build-${variant}.log" 2>&1
    ./resource-"${variant}" >"${RUNTIME_DIR}/logs/resource-${variant}.log"
    cuobjdump --dump-sass "${variant}.o" >"${RUNTIME_DIR}/logs/sass-${variant}.txt"
    cat "${RUNTIME_DIR}/logs/build-${variant}.log" "${RUNTIME_DIR}/logs/resource-${variant}.log"
done
python3 - <<'PY'
from pathlib import Path
import hashlib,json,os
runtime=Path(os.environ['RUNTIME_DIR'])
record=runtime/'results/source'
path=record/'source-provenance.json'
data=json.loads(path.read_text())
data['variants']={name:{'macro':'APAC_FA3_PREFILL_EXP_POLY=1' if name=='poly' else 'default','binary_sha256':hashlib.sha256(Path('fa3-exp-'+name+'.so').read_bytes()).hexdigest(),'resource':(runtime/'logs'/('resource-'+name+'.log')).read_text(),'ptxas':(runtime/'logs'/('build-'+name+'.log')).read_text()} for name in ('control','poly')}
data['gpu_executed']=False
data.update(commit=os.environ['GITHUB_SHA'],kernel='FA3 fixed Prefill exp polynomial',scope='one GPU numerical-quality gate; approximate candidate',calibration=False)
(runtime/'source/provenance.json').write_text(json.dumps(data,indent=2)+'\n')
path.write_text(json.dumps(data,indent=2)+'\n')
(record/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in sorted(record.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
PY
