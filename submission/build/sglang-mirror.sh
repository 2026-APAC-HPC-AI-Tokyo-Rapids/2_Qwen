#!/usr/bin/env bash
# Rebuild the public SGLang source snapshot from public sources into GIT_MIRROR_DIR/5_SGLang_Src.git.
#
# The team mirror's history before 6b350686 was rewritten and is not public. Upstream 0d6bef6b has the
# identical tree (9ea4c730), so fetch it, recreate the 6b350686 commit object byte-for-byte as a shallow
# boundary, and import the sanitized team commits. Runtime Python and uv.lock match the recorded source.
#
# usage: GIT_MIRROR_DIR=/new/directory sglang-mirror.sh
set -euo pipefail

package=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mirror=${GIT_MIRROR_DIR}/5_SGLang_Src.git

mkdir -p "${GIT_MIRROR_DIR}"
git init --bare "${mirror}"
git -C "${mirror}" fetch --depth 1 https://github.com/sgl-project/sglang.git 0d6bef6b6d2f4fa03187fb4d02773ad6e1256adb
git -C "${mirror}" hash-object -t commit -w --stdin <"${package}/build/sglang-6b350686.commit"
printf '%s\n' 6b3506869d4840b18678de19d18d751a34f9b737 >>"${mirror}/shallow"
git -C "${mirror}" fetch "${package}/build/sglang-fc275674.bundle" refs/heads/publication-source:refs/heads/main
git -C "${mirror}" symbolic-ref HEAD refs/heads/main
printf 'Built %s at %s\n' "${mirror}" "$(git -C "${mirror}" rev-parse HEAD)"
