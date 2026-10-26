#!/usr/bin/env bash
set -euo pipefail

# The team mirror's history before 6b350686 was rewritten and is not public.
# Upstream 0d6bef6b has the identical tree (9ea4c730), so fetch it, recreate the
# 6b350686 commit object byte-for-byte as a shallow boundary, and add the three
# team commits from the bundle. HEAD must then be exactly fc275674.
: "${GIT_MIRROR_DIR:?directory for the public-source mirror is required}"
package="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${GIT_MIRROR_DIR}"
mirror=${GIT_MIRROR_DIR}/5_SGLang_Src.git
test ! -e "${mirror}"
git init --bare "${mirror}"
git -C "${mirror}" fetch --depth 1 https://github.com/sgl-project/sglang.git 0d6bef6b6d2f4fa03187fb4d02773ad6e1256adb
[[ "$(git -C "${mirror}" rev-parse 0d6bef6b6d2f4fa03187fb4d02773ad6e1256adb^{tree})" == 9ea4c7300d38bd9e024330b45bf5fc9cbf70f97d ]]
[[ "$(git -C "${mirror}" hash-object -t commit -w --stdin <"${package}/build/sglang-6b350686.commit")" == 6b3506869d4840b18678de19d18d751a34f9b737 ]]
printf '%s\n' 6b3506869d4840b18678de19d18d751a34f9b737 >>"${mirror}/shallow"
git -C "${mirror}" bundle verify "${package}/build/sglang-fc275674.bundle"
git -C "${mirror}" fetch "${package}/build/sglang-fc275674.bundle" refs/heads/publication-source:refs/heads/main
git -C "${mirror}" symbolic-ref HEAD refs/heads/main
[[ "$(git -C "${mirror}" rev-parse HEAD)" == fc275674a97d78553b22597ee72c8fd48f5420f4 ]]
printf 'Verified %s at %s\n' "${mirror}" "$(git -C "${mirror}" rev-parse HEAD)"
