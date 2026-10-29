import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SBATCH = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
FREEZE = next(line for line in SBATCH.splitlines() if "/freeze_gc\"" in line)


def test_gc_is_frozen_on_every_worker_after_the_warm_up_request_and_before_the_official_client():
    sanity = SBATCH.index('>"${ROOT}/results/sanity-chat.json"')
    freeze = SBATCH.index(FREEZE)
    client = SBATCH.index('bash "${ROOT}/scripts/official-aiperf.sh" "${ROOT}" "${N}" >"${ART}/aiperf.log"')
    assert sanity < freeze < client
    assert FREEZE.startswith("while IFS=$'\\t' read -r worker host port _; do ")
    assert FREEZE.endswith('done <"${ROOT}/source/worker-endpoints.tsv"')


def test_a_failed_freeze_is_logged_and_the_job_continues(tmp_path):
    (tmp_path / "source").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "source/worker-endpoints.tsv").write_text("prefill-0\t10.0.0.1\t30100\t4\nprefill-1\t10.0.0.1\t30101\t4\ndecode-0\t10.0.0.2\t30200\t4\n")
    fake = tmp_path / "bin/curl"
    fake.parent.mkdir()
    # The second Prefill worker refuses; the others answer like SGLang's /freeze_gc.
    fake.write_text('#!/bin/sh\nfor a in "$@"; do last=$a; done\necho "$last" >>"$CALLS"\ncase "$last" in *30101*) echo "curl: (22) 500" >&2; exit 22;; esac\necho "Garbage collection frozen."\n')
    fake.chmod(0o755)
    script = f'set -euo pipefail\nROOT={tmp_path}\nART={tmp_path}/logs\n{FREEZE}\necho done\n'
    env = dict(os.environ, PATH=f"{fake.parent}:{os.environ['PATH']}", CALLS=str(tmp_path / "calls"))
    out = subprocess.run(["bash", "-c", script], env=env, check=True, capture_output=True, text=True).stdout
    assert out == "done\n"
    assert (tmp_path / "calls").read_text().split() == ["http://10.0.0.1:30100/freeze_gc", "http://10.0.0.1:30101/freeze_gc", "http://10.0.0.2:30200/freeze_gc"]
    log = (tmp_path / "logs/freeze-gc.log").read_text()
    assert log.count("Garbage collection frozen.") == 2
    assert "freeze_gc failed on prefill-1" in log
