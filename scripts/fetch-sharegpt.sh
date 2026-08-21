#!/usr/bin/env bash
set -euo pipefail

[[ "$#" -eq 1 ]]

dataset_root="$1"
repository=anon8231489123/ShareGPT_Vicuna_unfiltered
revision=192ab2185289094fc556ec8ce5ce1e8e587154ca
filename=ShareGPT_V3_unfiltered_cleaned_split.json
expected_sha256=35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4
target_dir="${dataset_root}/${revision}"
target_file="${target_dir}/${filename}"

install -d -m 2775 "${dataset_root}"
if [[ -e "${target_dir}" ]]; then test -f "${target_file}"; echo "${expected_sha256}  ${target_file}" | sha256sum --check --status; printf '%s\n' "${target_file}"; exit 0; fi

staging_dir="$(mktemp -d "${dataset_root}/.${revision}.XXXXXXXXXX")"
cleanup() {
  rm -rf -- "${staging_dir:?}"
}
trap cleanup EXIT

curl --fail --location --silent --show-error "https://huggingface.co/datasets/${repository}/resolve/${revision}/${filename}" --output "${staging_dir}/${filename}"
echo "${expected_sha256}  ${staging_dir}/${filename}" | sha256sum --check --status
chmod 0664 "${staging_dir}/${filename}"
chmod 2775 "${staging_dir}"
mv -T "${staging_dir}" "${target_dir}"
trap - EXIT
printf '%s\n' "${target_file}"
