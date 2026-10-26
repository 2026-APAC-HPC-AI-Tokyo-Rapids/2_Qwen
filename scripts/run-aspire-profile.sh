#!/usr/bin/env bash
# Called after the normal official AIPerf pass, within the same PBS allocation.
set -euo pipefail
runtime_dir=${1:?runtime directory required}
: "${CONTAINER_IMAGE:?container image required}"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/qwen-aspire-profile.XXXXXXXXXX")"
capture_pid=
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "${capture_pid}" ]]; then kill "${capture_pid}" 2>/dev/null || true; wait "${capture_pid}" 2>/dev/null || true; fi
  rm -rf -- "${scratch:?}"
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export ENROOT_CACHE_PATH="${scratch}/cache" ENROOT_DATA_PATH="${scratch}/data" ENROOT_RUNTIME_PATH="${scratch}/runtime" ENROOT_TEMP_PATH="${scratch}/temp"
install -d -m 0700 "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}" "${ENROOT_TEMP_PATH}"
container=(enroot start -e NVIDIA_VISIBLE_DEVICES=none -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -m "${runtime_dir}:${runtime_dir}:none:bind,rw,x-create=dir" -m "${runtime_dir}/nsys:/opt/nsys:none:bind,ro,x-create=dir" "${CONTAINER_IMAGE}")
"${container[@]}" /opt/nsys/bin/nsys --version >"${runtime_dir}/profile/nsys-version.txt"
printf 'Normal official benchmark (60/120/30) completes before a separate profile-only client (20/30/90). Nsight capture is disabled during the normal pass, but its launcher is present. Capture begins five seconds into the profile phase and lasts five seconds after every start acknowledgement. Exclude tracing/export tails; profile/client is diagnostic only. Four worker reports and all sixteen GPU ranks are required.\n' >"${runtime_dir}/profile/interpretation.txt"
# Serving endpoints store bootstrap ports; profiling needs unique worker names and TP counts.
endpoints="${runtime_dir}/profile/worker-endpoints.tsv"
: >"${endpoints}"
while IFS=$'\t' read -r role worker node port tp; do
  host=$(awk -v rank="${node}" '$1 == rank {print $3}' "${runtime_dir}/source/node-addresses.tsv")
  [[ "${host}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]
  printf '%s-%s\t%s\t%s\t%s\n' "${role}" "${worker}" "${host}" "${port}" "${tp}" >>"${endpoints}"
done < <(jq -r '.workers[] | [.role,.index,.node,.port,.tp] | @tsv' "${runtime_dir}/source/aspire-serving.json")
expected_reports=$(wc -l <"${endpoints}")
[[ "${expected_reports}" -eq 4 ]]
[[ "$(find "${runtime_dir}/profile" -maxdepth 1 -type f -name 'nsys-*.argv' | wc -l)" -eq "${expected_reports}" ]]
bash "${runtime_dir}/scripts/capture-firmus-profile.sh" "${runtime_dir}" "${endpoints}" >"${runtime_dir}/logs/profile-capture.log" 2>&1 &
capture_pid=$!
client_status=0
"${container[@]}" bash "${runtime_dir}/scripts/official-aiperf.sh" "${runtime_dir}" "$(jq -er .concurrency "${runtime_dir}/source/aspire-serving.json")" profile >"${runtime_dir}/logs/profile-aiperf.log" 2>&1 || client_status=$?
capture_status=0
wait "${capture_pid}" || capture_status=$?
capture_pid=
printf 'profile_client_status=%s\nprofile_capture_status=%s\n' "${client_status}" "${capture_status}" >"${runtime_dir}/logs/profile-status.txt"
(( capture_status == 0 )) || exit "${capture_status}"
deadline=$((SECONDS + 180))
while [[ "$(find "${runtime_dir}/profile" -maxdepth 1 -name '*.nsys-rep' -size +0c | wc -l)" -ne "${expected_reports}" ]]; do (( SECONDS < deadline )); sleep 2; done
"${container[@]}" /opt/sglang/venv/bin/python "${runtime_dir}/scripts/validate-firmus-profile.py" "${runtime_dir}" --endpoints "${endpoints}" >"${runtime_dir}/logs/profile-validation.log" 2>&1
(( client_status == 0 )) || exit "${client_status}"
