#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --compile-only ]]; then
    install -d "${RUNTIME_DIR}/source" "${RUNTIME_DIR}/logs" "${RUNTIME_DIR}/results"
    cd "${RUNTIME_DIR}/source"
else
    source "${GITHUB_WORKSPACE}/kernels/probes/prepare-sglang.sh"
fi
cp -r "${GITHUB_WORKSPACE}/kernels/dense-dg1" .
cp "${GITHUB_WORKSPACE}/scripts/check-firmus-decode-kernels.py" "${GITHUB_WORKSPACE}/scripts/select-firmus-decode-dense.py" .
cp "${GITHUB_WORKSPACE}/kernels/probes/dense-decode.py" bench.py
unsquashfs -no-progress -no-xattrs -processors 1 -d "${RUNTIME_DIR}/dense-headers" "${RUNTIME_DIR}/sglang.sqsh" opt/sglang/venv/lib/python3.12/site-packages/deep_gemm/include >"${RUNTIME_DIR}/logs/extract-dense-headers.log" 2>&1
source "${RUNTIME_DIR}/scripts/build-env.sh"
DEEP_GEMM_INCLUDE="${RUNTIME_DIR}/dense-headers/opt/sglang/venv/lib/python3.12/site-packages/deep_gemm/include" bash dense-dg1/build.sh >"${RUNTIME_DIR}/logs/build-dense-dg1.log" 2>&1
cat >probe <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${RUNTIME_DIR}/source/dense-dg1:${PYTHONPATH:-}"
exec /opt/sglang/venv/bin/python "${RUNTIME_DIR}/source/bench.py"
RUN
chmod +x probe
python3 - <<'META'
import ast, json
from pathlib import Path
source = ast.parse(Path("check-firmus-decode-kernels.py").read_text())
nodes = [node for node in source.body if (isinstance(node, ast.FunctionDef) and node.name == "dense_candidates") or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DENSE_CONFIGS" for t in node.targets))]
ns = {}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "dense-candidates", "exec"), ns)
Path("dense-candidates.json").write_text(json.dumps({op: ns["dense_candidates"](op) for op in ("qkv", "o")}, indent=2) + "\n")
META
jq -n --arg commit "${GITHUB_SHA}" '{commit:$commit,kernel:"Decode dense DG1",scope:"one H200, QKV/O, M128/160/176, four rotating weights",calibration:true}' >provenance.json
