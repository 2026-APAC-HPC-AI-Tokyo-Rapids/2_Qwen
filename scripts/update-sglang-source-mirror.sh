#!/usr/bin/env bash
set -euo pipefail

: "${GIT_MIRROR_DIR:?}"

readonly SGLANG_MIRROR="${GIT_MIRROR_DIR}/5_SGLang_Src.git"
readonly SGLANG_MIRROR_LOCK="${GIT_MIRROR_DIR}/5_SGLang_Src.lock"
readonly FETCH_RETRIES="${SGLANG_MIRROR_FETCH_RETRIES:-3}"

if [[ ! -d "${SGLANG_MIRROR}" ]] ||
   [[ "$(git -C "${SGLANG_MIRROR}" rev-parse --is-bare-repository 2>/dev/null)" != true ]]; then
  echo "The SGLang source mirror is not initialized: ${SGLANG_MIRROR}" >&2
  echo "Create it once from a Genkai login session before running this workflow." >&2
  exit 1
fi

exec {sglang_mirror_lock_fd}>>"${SGLANG_MIRROR_LOCK}"
flock -x "${sglang_mirror_lock_fd}"

for ((attempt = 1; attempt <= FETCH_RETRIES; attempt++))
do
  if git -C "${SGLANG_MIRROR}" fetch --prune --prune-tags origin; then
    break
  fi

  if ((attempt == FETCH_RETRIES)); then
    echo "Could not update the SGLang source mirror after ${FETCH_RETRIES} attempts." >&2
    exit 1
  fi

  echo "Fetch attempt ${attempt}/${FETCH_RETRIES} failed; retrying." >&2
  sleep "$((attempt * 2))"
done

if ! git -C "${SGLANG_MIRROR}" show-ref --verify --quiet refs/heads/main; then
  echo "The required SGLang source ref is unavailable: refs/heads/main" >&2
  exit 1
fi

git -C "${SGLANG_MIRROR}" rev-parse refs/heads/main
