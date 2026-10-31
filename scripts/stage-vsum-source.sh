#!/usr/bin/env bash
# Temporary input transport for this experiment while the Genkai login fails.
set -euo pipefail
destination=${1:?build input directory required}
install -d -m 0755 "${destination}"
archive="${destination}/sglang-fc275674.tar.zst"
trap 'rm -f -- "${archive}"' EXIT
gh api -H 'Accept: application/octet-stream' repos/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/releases/assets/696424908 >"${archive}"
printf 'c79a052e9e12d6a919081d0dfe029b37b88fe3737eaf3f960f4540d095833712  %s\n' "${archive}" | sha256sum --check
tar -I zstd -xf "${archive}" -C "${destination}" --no-same-owner --no-same-permissions
source_mirror="${destination}/git-mirror/5_SGLang_Src.git"
git -C "${source_mirror}" fsck --no-dangling
[[ "$(git -C "${source_mirror}" rev-parse HEAD)" == fc275674a97d78553b22597ee72c8fd48f5420f4 ]]
[[ ! -e "${source_mirror}/objects/info/alternates" ]]
