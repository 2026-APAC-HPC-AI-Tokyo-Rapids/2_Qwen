#!/usr/bin/env bash
# One architecture; exact N192 oracle and served M192/N96 control. CPU gates precede Slurm.
set -euo pipefail
: "${RUNTIME_DIR:?}" "${GITHUB_WORKSPACE:?}"
[[ -z "${CANDIDATE_PATCH:-}" ]]
install -d "${RUNTIME_DIR}/source" "${RUNTIME_DIR}/logs" "${RUNTIME_DIR}/results/source" "${RUNTIME_DIR}/scripts"
install -m 0644 "${GITHUB_WORKSPACE}/kernels/fa3-verify/candidates.json" "${RUNTIME_DIR}/source/fa3-verify-candidates.json"
if [[ ! -f "${RUNTIME_DIR}/scripts/run-container.sh" ]]; then install -m 0755 "${GITHUB_WORKSPACE}/kernels/probes/run-container.sh" "${RUNTIME_DIR}/scripts/run-container.sh"; fi
if [[ -f "${RUNTIME_DIR}/scripts/build-env.sh" ]]; then source "${RUNTIME_DIR}/scripts/build-env.sh"; fi
python3 "${GITHUB_WORKSPACE}/kernels/probes/build-fa3-split-pv.py"
cat >"${RUNTIME_DIR}/source/probe" <<'PROBE'
#!/usr/bin/env bash
set -euo pipefail
: "${RUNTIME_DIR:?}"
exec bash "${RUNTIME_DIR}/scripts/run-container.sh" /opt/sglang/venv/bin/python "${RUNTIME_DIR}/source/fa3-split-pv.py" --directory "${RUNTIME_DIR}/source" --output "${RUNTIME_DIR}/results/fa3-split-pv.json"
PROBE
chmod 0755 "${RUNTIME_DIR}/source/probe"
