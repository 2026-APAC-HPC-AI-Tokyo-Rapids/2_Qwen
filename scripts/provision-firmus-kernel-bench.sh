#!/usr/bin/env bash
# Stage a 1-node SM90 MegaMoE kernel bench on the Firmus login node: the pinned
# serving image (no rebuild), its DeepGEMM package and SGLang Python tree, one
# header overlay per DeepGEMM candidate (patches/deep-gemm-bench/<name>/*.patch),
# timeline overlays where a candidate directory holds a `timeline` marker (an
# empty marker uses the stock timeline patch; a non-empty one is that
# candidate's own timeline patch, for candidates the stock patch cannot hit), and
# an SGLang tree with patches/sglang-bench/*.patch. A candidate patch that does
# not apply stops here, before any Slurm submission; a timeline patch that does
# not apply on top of a candidate is recorded and that timeline is skipped.
set -euo pipefail
shopt -s inherit_errexit

: "${GITHUB_WORKSPACE:?}"
: "${GITHUB_RUN_ID:?}"
: "${GITHUB_RUN_ATTEMPT:?}"
: "${GITHUB_SHA:?}"
: "${GITHUB_OUTPUT:?}"
: "${RUNTIME_ROOT:?}"

# The hpcai partition caps jobs at 17 minutes. Every Slurm step (container
# start, imports, JIT, work) took 71-77 s in run 39085655145, so a job holds at
# most 12 steps: the image bench, the image timeline if requested, 2 SGLang
# checks (+2 benches with bench-firmus-sglang-kernels.py), and per DeepGEMM
# candidate a gate, a bench and the timeline if requested. The Prefill probes
# (an empty patches/sglang-bench/probes marker) add 3 steps on the image tree.
max_steps=12
bench_dir="${GITHUB_WORKSPACE}/patches/deep-gemm-bench"
shopt -s nullglob
flat_patches=("${bench_dir}"/*.patch)
candidate_dirs=("${bench_dir}"/*/)
sglang_patches=("${GITHUB_WORKSPACE}"/patches/sglang-bench/*.patch)
shopt -u nullglob
if (( ${#flat_patches[@]} )); then printf 'Move %s into patches/deep-gemm-bench/<candidate>/ (one directory per candidate).\n' "${flat_patches[@]#"${GITHUB_WORKSPACE}/"}" >&2; exit 1; fi
steps=1
if [[ -e "${bench_dir}/timeline" ]]; then steps=$((steps + 1)); fi
if (( ${#sglang_patches[@]} )); then steps=$((steps + 2)); fi
if (( ${#sglang_patches[@]} )) && [[ -e "${GITHUB_WORKSPACE}/scripts/bench-firmus-sglang-kernels.py" ]]; then steps=$((steps + 2)); fi
probes=false
if [[ -e "${GITHUB_WORKSPACE}/patches/sglang-bench/probes" ]]; then probes=true; steps=$((steps + 3)); fi
for dir in "${candidate_dirs[@]}"; do
  name="$(basename "${dir}")"
  [[ "${name}" =~ ^[a-z0-9][a-z0-9._-]*$ && "${name}" != image ]] || { printf 'Candidate directory name %s must match [a-z0-9][a-z0-9._-]* and not be "image".\n' "${name}" >&2; exit 1; }
  compgen -G "${dir}*.patch" >/dev/null || { printf 'Candidate directory %s has no .patch file.\n' "${name}" >&2; exit 1; }
  steps=$((steps + 2))
  if [[ -e "${dir}timeline" ]]; then steps=$((steps + 1)); fi
done
if (( steps > max_steps )); then printf '%s Slurm steps (about 75 s each) exceed the %s that fit in one 17-minute job; drop candidates or timeline markers.\n' "${steps}" "${max_steps}" >&2; exit 1; fi

config="${GITHUB_WORKSPACE}/configs/firmus-framework.json"
image="$(jq -er .images.sglang.path "${config}")"
expected_sha="$(jq -er .images.sglang.sha256 "${config}")"
[[ "$(stat -c '%s' "${image}")" == "$(jq -er .images.sglang.size_bytes "${config}")" ]]
install -d -m 2775 "${RUNTIME_ROOT}"
runtime_dir="$(mktemp -d "${RUNTIME_ROOT}/kernel-bench-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.XXXXXXXXXX")"
printf 'runtime_dir=%s\n' "${runtime_dir}" >>"${GITHUB_OUTPUT}"
install -d -m 0755 "${runtime_dir}/logs" "${runtime_dir}/results" "${runtime_dir}/scripts" "${runtime_dir}/source/patches" "${runtime_dir}/scratch" "${runtime_dir}/overlay/candidates" "${runtime_dir}/overlay/timeline" "${runtime_dir}/home"
echo "Checking sglang image SHA-256"
while sleep 15; do echo "container_sha256_check_in_progress=sglang"; done &
progress_pid=$!
trap 'kill "${progress_pid}" 2>/dev/null || true' EXIT
sha256sum "${image}" >"${runtime_dir}/source/sglang-image.sha256"
kill "${progress_pid}" 2>/dev/null || true
wait "${progress_pid}" 2>/dev/null || true
trap - EXIT
[[ "$(awk '{print $1}' "${runtime_dir}/source/sglang-image.sha256")" == "${expected_sha}" ]]
ln -s "${image}" "${runtime_dir}/sglang.sqsh"

# The image's DeepGEMM already carries patches/deep-gemm/*.patch (header SHA in
# its build record), and its SGLang source already carries patches/sglang/*.patch.
package=opt/sglang/venv/lib/python3.12/site-packages/deep_gemm
sglang_package=opt/sglang/source/python/sglang
extract=("${package}" opt/apac/sm90-megamoe-build.json)
if (( ${#sglang_patches[@]} )); then extract+=("${sglang_package}"); fi
unsquashfs -no-progress -no-xattrs -processors 1 -d "${runtime_dir}/image" "${image}" "${extract[@]}"
chmod -R u+w "${runtime_dir}/image"
cp "${runtime_dir}/image/opt/apac/sm90-megamoe-build.json" "${runtime_dir}/source/"
[[ "$(sha256sum "${runtime_dir}/image/${package}/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh" | awk '{print $1}')" == "$(jq -er .header_sha256 "${runtime_dir}/source/sm90-megamoe-build.json")" ]]

timeline_patch="${GITHUB_WORKSPACE}/patches/deep-gemm-timeline/sm90-mega-moe-timeline.patch"
apply_patches() {
  local root=$1
  shift
  for patch in "$@"; do
    (cd "${root}" && git apply --check "${patch}" && git apply "${patch}")
  done
}
patch_records() {
  for patch in "$@"; do jq -n --arg path "${patch#"${GITHUB_WORKSPACE}/"}" --arg sha "$(sha256sum "${patch}" | awk '{print $1}')" '{path:$path,sha256:$sha}'; done | jq -s .
}
# Timeline overlay roots mirror site-packages like the candidate overlays; the
# timeline patch's context lines can collide with a candidate's edits.
add_timeline() {
  local name=$1 base=$2 dir=$3
  if [[ ! -e "${dir}/timeline" ]]; then echo not-requested; return; fi
  install -d -m 0755 "${runtime_dir}/overlay/timeline/${name}"
  cp -a "${base}" "${runtime_dir}/overlay/timeline/${name}/deep_gemm"
  local patch=${timeline_patch}
  if [[ -s "${dir}/timeline" ]]; then patch="${dir}/timeline"; fi
  if (cd "${runtime_dir}/overlay/timeline/${name}" && git apply --check "${patch}" 2>"${runtime_dir}/logs/timeline-apply-${name}.log" && git apply "${patch}"); then rm -f -- "${runtime_dir}/logs/timeline-apply-${name}.log"; echo ready; return; fi
  cat "${runtime_dir}/logs/timeline-apply-${name}.log" >&2
  rm -rf -- "${runtime_dir}/overlay/timeline/${name}"
  echo apply-failed
}
timeline="$(add_timeline image "${runtime_dir}/image/${package}" "${bench_dir}")"
variants="$(jq -n --arg timeline "${timeline}" '[{name:"image",patches:[],timeline:$timeline}]')"
for dir in "${candidate_dirs[@]}"; do
  name="$(basename "${dir}")"
  patches=("${dir}"*.patch)
  install -d -m 0755 "${runtime_dir}/overlay/candidates/${name}"
  cp -a "${runtime_dir}/image/${package}" "${runtime_dir}/overlay/candidates/${name}/deep_gemm"
  apply_patches "${runtime_dir}/overlay/candidates/${name}" "${patches[@]}"
  timeline="$(add_timeline "${name}" "${runtime_dir}/overlay/candidates/${name}/deep_gemm" "${dir}")"
  variants="$(jq --arg name "${name}" --argjson patches "$(patch_records "${patches[@]}")" --arg timeline "${timeline}" '. + [{name:$name,patches:$patches,timeline:$timeline}]' <<<"${variants}")"
done
sglang=null
if (( ${#sglang_patches[@]} )); then
  # git apply paths are relative to /opt/sglang/source (python/sglang/...).
  install -d -m 0755 "${runtime_dir}/overlay/sglang/python"
  cp -a "${runtime_dir}/image/${sglang_package}" "${runtime_dir}/overlay/sglang/python/sglang"
  apply_patches "${runtime_dir}/overlay/sglang" "${sglang_patches[@]}"
  sglang="$(jq -n --argjson patches "$(patch_records "${sglang_patches[@]}")" --argjson bench "$(if [[ -e "${GITHUB_WORKSPACE}/scripts/bench-firmus-sglang-kernels.py" ]]; then echo true; else echo false; fi)" '{patches:$patches,bench_script:$bench}')"
fi
jq -n --argjson deep_gemm "${variants}" --argjson sglang "${sglang}" '{deep_gemm:$deep_gemm,sglang:$sglang}' >"${runtime_dir}/results/variants.json"
for part in deep-gemm-bench sglang-bench; do
  if [[ -d "${GITHUB_WORKSPACE}/patches/${part}" ]]; then cp -a "${GITHUB_WORKSPACE}/patches/${part}" "${runtime_dir}/source/patches/"; fi
done
cp "${timeline_patch}" "${runtime_dir}/source/patches/"

install -m 0755 "${GITHUB_WORKSPACE}/platforms/firmus/kernel-bench/run.sbatch" "${runtime_dir}/scripts/run.sbatch"
scripts=(bench-firmus-megamoe.py check-firmus-megamoe.py check-firmus-decode-kernels.py)
if [[ -e "${GITHUB_WORKSPACE}/scripts/bench-firmus-sglang-kernels.py" ]]; then scripts+=(bench-firmus-sglang-kernels.py); fi
if [[ "${probes}" == true ]]; then scripts+=(bench-firmus-deepgemm-gemm.py); fi
for script in "${scripts[@]}"; do install -m 0644 "${GITHUB_WORKSPACE}/scripts/${script}" "${runtime_dir}/scripts/"; cp "${runtime_dir}/scripts/${script}" "${runtime_dir}/source/"; done
cp "${runtime_dir}/scripts/run.sbatch" "${runtime_dir}/source/"
jq -n --arg commit "${GITHUB_SHA}" --arg run_id "${GITHUB_RUN_ID}" --arg attempt "${GITHUB_RUN_ATTEMPT}" --argjson image "$(jq .images.sglang "${config}")" --arg timeline_sha "$(sha256sum "${timeline_patch}" | awk '{print $1}')" '{schema_version:"qwen3-megamoe-kernel-bench/v2",cluster:"firmus",commit:$commit,workflow_run_id:$run_id,workflow_run_attempt:$attempt,image:($image | {path,sha256,size_bytes}),timeline_patch_sha256:$timeline_sha,variants:"results/variants.json"}' >"${runtime_dir}/source/provenance.json"
