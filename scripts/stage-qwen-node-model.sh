#!/usr/bin/env bash
set -euo pipefail
source_path="$1"
destination="$2"
log="$3"
[[ -d "${source_path}" && ! -e "${destination}" ]]
exec >"${log}" 2>&1
printf 'source=%s\ndestination=%s\nhost=%s\n' "${source_path}" "${destination}" "$(hostname -s)"
df -PT "${source_path}" "$(dirname "${destination}")"
printf 'copy_start=%s\n' "$(date --iso-8601=ns)"
SECONDS=0
cp -rL -- "${source_path}" "${destination}"
printf 'copy_seconds=%s\ncopy_end=%s\n' "${SECONDS}" "$(date --iso-8601=ns)"
cmp "${source_path}/model.safetensors.index.json" "${destination}/model.safetensors.index.json"
while IFS= read -r shard; do [[ "$(stat -c %s "${source_path}/${shard}")" == "$(stat -c %s "${destination}/${shard}")" ]]; done < <(jq -r '.weight_map[]' "${source_path}/model.safetensors.index.json" | sort -u)
du -sb "${destination}"
printf 'copy_verified=true\n'
