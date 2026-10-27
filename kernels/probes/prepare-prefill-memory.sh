#!/usr/bin/env bash
# Use the configured serving overlays; compile/JIT only inside the allocated GPU step.
set -euo pipefail
source "${GITHUB_WORKSPACE}/kernels/probes/prepare-sglang.sh"
install -m 0644 "${GITHUB_WORKSPACE}/kernels/probes/prefill-memory.py" "${RUNTIME_DIR}/source/prefill-memory.py"
install -m 0644 "${GITHUB_WORKSPACE}/kernels/probes/prefill-paired-round.py" "${RUNTIME_DIR}/source/prefill-paired-round.py"
cat >"${RUNTIME_DIR}/source/probe" <<'PROBE'
#!/usr/bin/env bash
set -euo pipefail
: "${RUNTIME_DIR:?}"
exec /opt/sglang/venv/bin/python "${RUNTIME_DIR}/source/prefill-paired-round.py" --header "${RUNTIME_DIR}/source/python/sglang/kernels/jit/csrc/elementwise/rope.cuh" --output "${RUNTIME_DIR}/results/prefill-memory" --run
PROBE
chmod 0755 "${RUNTIME_DIR}/source/probe"
python3 - <<'PY'
import hashlib, json, os, shutil
from pathlib import Path
root = Path(os.environ['GITHUB_WORKSPACE'])
runtime = Path(os.environ['RUNTIME_DIR'])
config = json.loads((root/'configs/firmus-framework.json').read_text())
source = runtime/'source'
saved = runtime/'results/source'
saved.mkdir()
for original, name in ((source/'prefill-memory.py', 'prefill-memory.py'), (source/'prefill-paired-round.py', 'prefill-paired-round.py'), (source/'probe', 'probe'), (source/'python/sglang/kernels/jit/csrc/elementwise/rope.cuh', 'rope-control.cuh'), (root/'kernels/probes/prepare-prefill-memory.sh', 'prepare-prefill-memory.sh'), (root/'configs/firmus-framework.json', 'firmus-framework.json')):
    shutil.copyfile(original, saved/name)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
provenance = {'commit': os.environ['GITHUB_SHA'], 'kernel': 'Prefill QKV exact paired rounding', 'scope': 'one GPU, no serving', 'sglang_overlays': {name: sha(root/name) for name in config['sglang_overlay_patches']}, 'sources': {path.name: sha(path) for path in saved.iterdir()}}
(source/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
PY
