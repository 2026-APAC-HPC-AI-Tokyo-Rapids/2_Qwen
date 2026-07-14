#!/usr/bin/env bash
# Materialize the exact ShareGPT input used by the benchmark on Genkai.
set -euo pipefail

repo=anon8231489123/ShareGPT_Vicuna_unfiltered
revision=192ab2185289094fc556ec8ce5ce1e8e587154ca
filename=ShareGPT_V3_unfiltered_cleaned_split.json
sha256=35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4
dataset_root=/fast/pj26000080/datasets/ShareGPT_Vicuna_unfiltered
target_dir="${dataset_root}/${revision}"
target_file="${target_dir}/${filename}"

install -d -m 2775 "${dataset_root}"

if [[ -f "${target_file}" ]]; then
  echo "${sha256}  ${target_file}" | sha256sum --check --status
  echo "ShareGPT is already present: ${target_file}"
  exit 0
fi

staging_dir="$(mktemp -d "${dataset_root}/.${revision}.XXXXXXXXXX")"
cleanup() {
  rm -rf -- "${staging_dir:?}"
}
trap cleanup EXIT

uvx --from huggingface_hub hf download "${repo}" \
  --repo-type dataset \
  --revision "${revision}" \
  --include "${filename}" \
  --local-dir "${staging_dir}"

downloaded_file="${staging_dir}/${filename}"
test -f "${downloaded_file}"
echo "${sha256}  ${downloaded_file}" | sha256sum --check
rm -rf -- "${staging_dir}/.cache"
chmod 0664 "${downloaded_file}"
chmod 2775 "${staging_dir}"

# The rename is within /fast, so consumers see either no revision or a
# completely verified revision; never a partially downloaded dataset.
mv "${staging_dir}" "${target_dir}"
trap - EXIT
echo "ShareGPT installed: ${target_file}"
