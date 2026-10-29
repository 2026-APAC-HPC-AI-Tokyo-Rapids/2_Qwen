import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TABLE = json.loads((ROOT / "configs/firmus-gpu-sm-clock.json").read_text())
ORDER = runpy.run_path(str(ROOT / "scripts/order-firmus-pp4-gpus.py"))["order"]
UUID = {(g["host"], g["index"]): u for u, g in TABLE["gpus"].items()}


def smi(host, indices, local=range(4)):
    return "\n".join(f"{i}, {UUID[(host, j)]}" for i, j in zip(local, indices)) + "\n"


def test_table_covers_the_twelve_prefill_gpus_of_ten_ep4_runs():
    assert len(TABLE["runs"]) == 10 and len(TABLE["gpus"]) == 12
    assert set(UUID) == {("g1", i) for i in range(8)} | {("g4", i) for i in range(4)}
    clock = {k: TABLE["gpus"][u]["sm_mhz"] for k, u in UUID.items()}
    assert min(range(4), key=lambda i: clock[("g1", i)]) == 3
    assert min(range(4, 8), key=lambda i: clock[("g1", i)]) == 6
    assert min(range(4), key=lambda i: clock[("g4", i)]) == 2


@pytest.mark.parametrize(("host", "indices", "expected"), [("g1", range(4), "0,1,2,3"), ("g1", range(4, 8), "0,1,3,2"), ("g4", range(4), "0,1,3,2")])
def test_the_last_stage_moves_to_the_slowest_gpu_of_each_worker(host, indices, expected):
    assert ORDER(TABLE, smi(host, indices), "0,1,2,3") == expected


def test_entries_of_an_unconstrained_visible_list_are_kept():
    text = smi("g1", range(4, 8), local=range(4, 8))
    assert ORDER(TABLE, text, "4,5,6,7") == "4,5,7,6"


def test_unknown_gpu_or_small_difference_keeps_the_order():
    text = smi("g4", range(4)).replace(UUID[("g4", 0)], "GPU-unknown")
    assert ORDER(TABLE, text, "0,1,2,3") == "0,1,2,3"
    close = json.loads(json.dumps(TABLE))
    for index in (1, 2):
        close["gpus"][UUID[("g4", index)]]["sm_mhz"] = close["gpus"][UUID[("g4", 3)]]["sm_mhz"] - 4.0
    assert ORDER(close, smi("g4", range(4)), "0,1,2,3") == "0,1,2,3"
    assert ORDER(TABLE, smi("g4", range(3)), "0,1,2") == "0,1,2"


def test_worker_sets_the_order_only_for_pp4_prefill_and_logs_it(tmp_path):
    shell = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    worker = shell[shell.index("cat >\"${ROOT}/source/worker.sh\" <<'EOF'"):]
    worker = worker[:worker.index("\nEOF\n")]
    pp4 = worker[worker.index('  if [[ "$role" == prefill ]]; then\n    # The last stage'):]
    pp4 = pp4[:pp4.index("\nfi\n")]
    block = pp4[pp4.index('    if [[ -s "${runtime_dir}/source/firmus-gpu-sm-clock.json" ]]; then'):]
    block = block[:block.index("\n    fi\n") + 8]
    assert worker.index(block) < worker.index('echo "$name host=$(hostname)') < worker.index('exec "${command[@]}"')
    # Run the block with a fake nvidia-smi on g4 GPUs 0-3 (tail Prefill worker).
    (tmp_path / "source").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "source/firmus-gpu-sm-clock.json").write_text(json.dumps(TABLE))
    (tmp_path / "scripts/order-firmus-pp4-gpus.py").write_text((ROOT / "scripts/order-firmus-pp4-gpus.py").read_text())
    fake = tmp_path / "bin/nvidia-smi"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\ncat <<'X'\n" + smi("g4", range(4)) + "X\n")
    fake.chmod(0o755)
    script = "set -euo pipefail\nruntime_dir=$1\n" + block.replace("/opt/sglang/venv/bin/python", sys.executable) + '\necho "gpu=$CUDA_VISIBLE_DEVICES"\n'
    env = dict(os.environ, PATH=f"{fake.parent}:{os.environ['PATH']}", CUDA_VISIBLE_DEVICES="0,1,2,3")
    out = subprocess.run(["bash", "-c", script, "worker", str(tmp_path)], env=env, check=True, capture_output=True, text=True).stdout
    assert out == "gpu=0,1,3,2\n"
    (tmp_path / "source/firmus-gpu-sm-clock.json").unlink()
    out = subprocess.run(["bash", "-c", script, "worker", str(tmp_path)], env=env, check=True, capture_output=True, text=True).stdout
    assert out == "gpu=0,1,2,3\n"


def test_pp4_config_names_the_table_and_provision_stages_it():
    config = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    assert (ROOT / config["prefill"]["gpu_sm_clock"]).is_file()
    provision = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    assert 'install -m 0644 "${GITHUB_WORKSPACE}/${gpu_clock}" "${runtime_dir}/source/firmus-gpu-sm-clock.json"' in provision
    assert 'install -m 0644 "${GITHUB_WORKSPACE}/scripts/order-firmus-pp4-gpus.py" "${runtime_dir}/scripts/"' in provision
    assert "gpu_sm_clock" not in (ROOT / "configs/firmus-framework.json").read_text()
