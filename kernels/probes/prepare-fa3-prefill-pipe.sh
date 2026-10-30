#!/usr/bin/env bash
# One-GPU paired timing of the serving Prefill FA3 N192 varlen rows, built on the login node
# from the pinned sources and the exact patch chain of kernels/fa3-verify/build.py.
set -euo pipefail
: "${RUNTIME_DIR:?}" "${GITHUB_WORKSPACE:?}"
[[ -z "${CANDIDATE_PATCH:-}" ]]
root="${GITHUB_WORKSPACE}"
# PrefillVariant values in timing order (index 0 is the pinned n192-direct-o-varlen) and their names.
variants=96,96,224
names=pin,pin-again,pipe
source "${root}/platforms/genkai/kernel-probe/prepare-runtime.sh"
install -m 0644 "${root}/kernels/fa3-verify/candidates.json" "${RUNTIME_DIR}/source/fa3-verify-candidates.json"
install -m 0755 "${root}/kernels/probes/run-container.sh" "${RUNTIME_DIR}/scripts/run-container.sh"
cd "${RUNTIME_DIR}/source"
for name in fa3 cutlass; do
    read -r repository commit digest < <(sed -n "s/^    \"${name}\": (\"\([^\"]*\)\", \"\([0-9a-f]*\)\", \"\([0-9a-f]*\)\"),$/\1 \2 \3/p" "${root}/kernels/fa3-verify/build.py")
    curl -fLsS "https://github.com/${repository}/archive/${commit}.tar.gz" -o "${name}.tar.gz"
    printf '%s  %s\n' "${digest}" "${name}.tar.gz" | sha256sum -c -
    mkdir "${name}"
    tar -xzf "${name}.tar.gz" -C "${name}"
done
fa3=$(echo "${PWD}"/fa3/*)
cutlass=$(echo "${PWD}"/cutlass/*)
mapfile -t chain < <(sed -n 's/.*str(root \/ "\([a-z0-9-]*\.patch\)").*/\1/p' "${root}/kernels/fa3-verify/build.py")
for patch in "${chain[@]}"; do
    source_patch="${root}/kernels/fa3-verify/${patch}"
    if [[ "${patch}" == fp8-output.patch ]]; then source_patch="${root}/kernels/probes/fa3-fp8-output.patch"; fi
    patch --batch --fuzz=0 -p1 -d "${fa3}" -i "${source_patch}"
done
# The pipelined-tile candidate until build.py carries it.
if [[ " ${chain[*]} " != *" prefill-pipe.patch "* ]]; then
    patch --batch --fuzz=0 -p1 -d "${fa3}" -i "${root}/kernels/fa3-verify/prefill-pipe.patch"
    chain+=(prefill-pipe.patch)
fi
printf '%s\n' "${chain[@]}" >"${RUNTIME_DIR}/results/patch-chain.txt"
cp -a "${fa3}" stamp-tree
patch --batch --fuzz=0 -p1 -d stamp-tree -i "${root}/kernels/probes/fa3-prefill-stamp.patch"
source "${RUNTIME_DIR}/scripts/build-env.sh"
flags=(-O3 -std=c++17 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda -lineinfo -gencode=arch=compute_90a,code=sm_90a -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DCUTLASS_DEBUG_TRACE_LEVEL=0 -DNDEBUG -Xptxas=-v,--warn-on-spills -ccbin "${CXX}" -Xcompiler=-fPIC -Dflash=apac_fa3_private -DFlash_fwd_params=ApacFA3VerifyParams -I"${root}/kernels/fa3-verify" -I"${cutlass}/include" -I"${cutlass}/tools/util/include")
nvcc --version >"${RUNTIME_DIR}/results/compiler.txt"
mkdir pipe-variants stamp-variants
printf '#define PF_VARIANTS %s\n' "${variants}" >pipe-variants/pf-variants.h
printf '#define PF_VARIANTS 96\n' >stamp-variants/pf-variants.h
nvcc "${flags[@]}" -I"${fa3}/hopper" -c "${fa3}/hopper/flash_prepare_scheduler.cu" -o scheduler.o >"${RUNTIME_DIR}/logs/build-scheduler.log" 2>&1
nvcc "${flags[@]}" -I"${fa3}/hopper" -I"${PWD}/pipe-variants" -c "${root}/kernels/probes/fa3-prefill-pipe.cu" -o pipe.o >"${RUNTIME_DIR}/logs/build-pipe.log" 2>&1 &
pipe_build=$!
nvcc "${flags[@]}" -I"${PWD}/stamp-tree/hopper" -I"${PWD}/stamp-variants" -DPF_STAMP -c "${root}/kernels/probes/fa3-prefill-pipe.cu" -o stamp.o >"${RUNTIME_DIR}/logs/build-stamp.log" 2>&1 &
stamp_build=$!
wait "${pipe_build}" || { tail -100 "${RUNTIME_DIR}/logs/build-pipe.log"; exit 1; }
wait "${stamp_build}" || { tail -100 "${RUNTIME_DIR}/logs/build-stamp.log"; exit 1; }
for library in pipe stamp; do
    nvcc "${library}.o" scheduler.o --shared --cudart static -o "fa3-prefill-${library}.so"
    cuobjdump -sass "${library}.o" >"${RUNTIME_DIR}/results/sass-${library}.txt"
    grep -E 'registers|spill' "${RUNTIME_DIR}/logs/build-${library}.log" | tee "${RUNTIME_DIR}/results/ptxas-${library}.txt"
done
install -m 0644 "${root}/kernels/probes/fa3-prefill-pipe.py" bench.py
cat >probe <<PROBE
#!/usr/bin/env bash
set -euo pipefail
# Phase clocks of the pin once per job, in their own process: a fault there must not hide the timing.
if [[ ! -e "\${RUNTIME_DIR}/results/stamps-attempted" ]]; then
    touch "\${RUNTIME_DIR}/results/stamps-attempted"
    bash "\${RUNTIME_DIR}/scripts/run-container.sh" /opt/sglang/venv/bin/python "\${RUNTIME_DIR}/source/bench.py" --stamp-library "\${RUNTIME_DIR}/source/fa3-prefill-stamp.so" --output "\${RUNTIME_DIR}/results/fa3-stamps.json" || true
fi
bash "\${RUNTIME_DIR}/scripts/run-container.sh" /opt/sglang/venv/bin/python "\${RUNTIME_DIR}/source/bench.py" --library "\${RUNTIME_DIR}/source/fa3-prefill-pipe.so" --names ${names} --output "\${RUNTIME_DIR}/results/fa3-pipe.json"
PROBE
chmod 0755 probe
jq -n --arg commit "${GITHUB_SHA:-local}" --arg variants "${variants}" --arg names "${names}" '{commit:$commit,kernel:"FA3 fixed Prefill N192 varlen rows",variants:$variants,names:$names,scope:"one GPU, two packed requests 3986+4027, no serving",calibration:false}' >provenance.json
