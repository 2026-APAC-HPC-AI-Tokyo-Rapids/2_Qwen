#!/usr/bin/env bash
set -euo pipefail
config=${GITHUB_WORKSPACE}/${1:-configs/firmus-basic.json}
archive=/scratch/home/ACCOUNT/qwen/basic-containers
build_run=$(jq -er .container_build_run_id "${config}")
root=${RUNTIME_ROOT}/basic-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}
install -d -m 2775 "${RUNTIME_ROOT}"
bash "${GITHUB_WORKSPACE}/submission/basic/prepare.sh" "${root}" "${archive}/${build_run}.sqsh" /scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8 /scratch/home/ACCOUNT/qwen/framework-comparison/sources/tokenizer/ac9c66cc9b46af7306746a9250f23d47083d689e "${config}"
jq -n --arg commit "${GITHUB_SHA}" --arg run "${GITHUB_RUN_ID}" --arg attempt "${GITHUB_RUN_ATTEMPT}" '{tier:"basic",framework:"sglang",workload:"official",profile:false,commit:$commit,run_id:$run,run_attempt:$attempt}' >"${root}/source/provenance.json"
printf 'runtime_dir=%s\n' "${root}" >>"${GITHUB_OUTPUT}"
