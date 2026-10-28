"""CPU fixtures of the ASPIRE EP4 launcher; no container or GPU work is executed."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_firmus_kernel_bench import ROOT, add_candidate, workspace

ENROOT = '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
with open(os.environ["ENROOT_LOG"], "a") as out:
    out.write(json.dumps(sys.argv[1:]) + "\\n")
args = sys.argv[1:]
if "--output" in args:
    path = Path(args[args.index("--output") + 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"case":"fixture","median_ms":1}\\n')
'''


def aspire_fixture(tmp_path):
    ws, patches = workspace(tmp_path)
    add_candidate(ws, "fc2-address", patches["comment"])
    campaign = ws / "patches/deep-gemm-bench"
    (campaign / "candidates").write_text("fc2-address\n")
    (campaign / "prefill-only").touch()
    (campaign / "timeline").touch()
    # The real script maps the existing image basename into this fixed ASPIRE directory.
    remote_path = "/scratch/users/industry/ai-hpc/ACCOUNT/sglang-containers/"
    mapped = tmp_path / "aspire-containers"
    mapped.mkdir()
    shutil.copy(tmp_path / "containers/image.sqsh", mapped / "40197369083.sqsh")
    config_path = ws / "configs/firmus-framework.json"
    config = json.loads(config_path.read_text())
    config["images"]["sglang"]["path"] = str(tmp_path / "containers/40197369083.sqsh")
    config["deep_gemm_overlay_patches"] = []
    config_path.write_text(json.dumps(config))
    image = config["images"]["sglang"]
    (ws / "configs/aspire-serving.json").write_text(json.dumps({"container_sha256": image["sha256"], "container_size_bytes": image["size_bytes"], "container_build_run_id": 40197369083, "deep_gemm_overlay_patches": []}))
    (mapped / "rootfs").symlink_to(tmp_path / "containers/rootfs", target_is_directory=True)
    script = ws / "scripts/provision-firmus-kernel-bench.sh"
    text = script.read_text()
    assert remote_path in text
    script.write_text(text.replace(remote_path, str(mapped) + "/"))
    aspire = ws / "platforms/aspire2ap/kernel-bench"
    aspire.mkdir(parents=True)
    for name in ("run.pbs", "preflight.sh"):
        shutil.copy(ROOT / "platforms/aspire2ap/kernel-bench" / name, aspire / name)
    home = tmp_path / "home"
    (home / ".local").mkdir(parents=True)
    library, system = tmp_path / "library", tmp_path / "system"
    library.mkdir()
    (system / "hooks.d").mkdir(parents=True)
    (library / "runtime.sh").write_text("local timeout=100 pid=-1 id=0\n")
    (system / "hooks.d/98-nvidia.sh").write_text('--ldconfig=@$(command -v ldconfig.real || command -v ldconfig)\n')
    (home / ".local/.setenv").write_text(f'export ENROOT_LIBRARY_PATH="{library}" ENROOT_SYSCONF_PATH="{system}"\n')
    if sys.platform == "darwin":
        # The production target uses GNU sed; adapt BSD's in-place CLI in this fixture.
        sed = tmp_path / "bin/sed"
        sed.write_text("#!/bin/bash\nif [[ $1 == -i ]]; then shift; exec /usr/bin/sed -i '' \"$@\"; else exec /usr/bin/sed \"$@\"; fi\n")
        sed.chmod(0o755)
    enroot = tmp_path / "bin/enroot"
    enroot.write_text(ENROOT)
    enroot.chmod(0o755)
    env = {**os.environ, "HOME": str(home), "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "GITHUB_WORKSPACE": str(ws), "GITHUB_RUN_ID": "1", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": "abc", "GITHUB_OUTPUT": str(tmp_path / "output"), "RUNTIME_ROOT": str(tmp_path / "runs"), "ENROOT_LOG": str(tmp_path / "enroot.log")}
    return ws, env


def prepare(tmp_path):
    ws, env = aspire_fixture(tmp_path)
    result = subprocess.run(["bash", str(ws / "scripts/provision-firmus-kernel-bench.sh"), "aspire2ap"], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    runtime = Path((tmp_path / "output").read_text().strip().split("=", 1)[1])
    return runtime, env


@pytest.mark.skipif(shutil.which("jq") is None, reason="provisioning uses jq")
def test_aspire_provision_reuses_pinned_image_and_cpu_only_preflight(tmp_path):
    runtime, env = prepare(tmp_path)
    provenance = json.loads((runtime / "source/provenance.json").read_text())
    assert provenance["cluster"] == "aspire2ap"
    assert provenance["image"]["path"] == str(tmp_path / "aspire-containers/40197369083.sqsh")
    config = json.loads((runtime / "results/variants.json").read_text())
    assert [variant["name"] for variant in config["deep_gemm"]] == ["image", "fc2-address"]
    assert config["prefill_only"] and config["prefill_memory"] is None
    assert (runtime / "source/aspire-serving.json").read_bytes() == (Path(env["GITHUB_WORKSPACE"]) / "configs/aspire-serving.json").read_bytes()
    assert "timeout=5000" in (runtime / "enroot-library/runtime.sh").read_text()
    assert (runtime / "enroot-system/hooks.d/98-nvidia.sh").read_text() == "--ldconfig=/sbin/ldconfig\n"
    calls = [json.loads(line) for line in Path(env["ENROOT_LOG"]).read_text().splitlines()]
    assert len(calls) == 1 and "NVIDIA_VISIBLE_DEVICES=void" in calls[0]
    assert str(runtime) in calls[0] and "-" in calls[0]
    assert (runtime / "scripts/run-megamoe-kernel-bench.sh").is_file()
    assert (runtime / "scripts/run.pbs").is_file()


@pytest.mark.parametrize("mask,gpu_file,gpu_count,passes", [(None, False, 4, True), ("3,1,0,2", False, 4, True), (None, True, 4, True), ("1,3,5,7", False, 8, True), (None, True, 8, True), ("0,1,2", False, 4, False), ("0,0,1,2", False, 4, False), (None, False, 8, False)])
def test_pbs_ep4_scope_and_shared_experiment_order(tmp_path, mask, gpu_file, gpu_count, passes):
    runtime, env = prepare(tmp_path)
    Path(env["ENROOT_LOG"]).write_text("")
    smi = tmp_path / "bin/nvidia-smi"
    smi.write_text('#!/bin/bash\nif [[ "$*" == "topo -m" ]]; then echo "fixture NVLink topology"; else for ((i=0;i<${FIXTURE_GPUS};i++)); do echo "$i, GPU-fixture-$i, NVIDIA H100, 81559 MiB"; done; fi\n')
    smi.chmod(0o755)
    env.update({"RUNTIME_DIR": str(runtime), "PBS_JOBID": "123.server", "FIXTURE_GPUS": str(gpu_count), "TMPDIR": str(tmp_path)})
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("PBS_GPUFILE", None)
    if mask is not None:
        env["CUDA_VISIBLE_DEVICES"] = mask
    if gpu_file:
        gpu_path = tmp_path / "pbs-gpus"
        gpu_path.write_text("".join(f"node-gpu{i}\n" for i in range(4)))
        env["PBS_GPUFILE"] = str(gpu_path)
    result = subprocess.run(["bash", str(runtime / "scripts/run.pbs")], env=env, capture_output=True, text=True, check=False)
    driver = (runtime / "logs/driver.log").read_text()
    assert (result.returncode == 0) is passes, driver
    assert ((runtime / "run-success").exists()) is passes
    assert ((runtime / "results/exit-code").read_text().strip() == "0") is passes
    assert not list(tmp_path.glob("megamoe-ep4-123.server.*"))
    calls = [json.loads(line) for line in Path(env["ENROOT_LOG"]).read_text().splitlines()]
    if not passes:
        assert calls == []  # Refuse the mask before any container/GPU operation.
        return
    selected = [f"GPU-fixture-{i}" for i in (map(int, mask.split(",")) if mask else range(4))]
    visible = ",".join(selected)
    assert len(calls) == 6  # Device preflight, image, gate, candidate, repeat, control timeline.
    assert all(f"NVIDIA_VISIBLE_DEVICES={visible}" in call and f"CUDA_VISIBLE_DEVICES={visible}" in call for call in calls)
    assert "torch.cuda.device_count() == 4" in " ".join(calls[0])
    steps = [tuple(line.split("\t")[:2]) for line in (runtime / "results/steps.tsv").read_text().splitlines()]
    assert steps == [("image", "bench"), ("fc2-address", "gate"), ("fc2-address", "bench"), ("control-repeat", "bench"), ("image", "timeline")]
    assert all("--prefill-only" in call for call in calls if any("bench-firmus-megamoe.py" in item for item in call))
    assert all(not any("prefill-memory.py" in item for item in call) for call in calls)
    assert all(any(str(runtime) + ":" + str(runtime) in item for item in call) for call in calls)


@pytest.mark.parametrize("state,status,success,steps,expected", [("Q", "", False, False, 75), ("R", "", False, False, 75), ("F", "0", True, True, 0), ("F", "1", True, True, 1), ("F", "0", False, True, 1), ("F", "0", True, False, 1)])
def test_pbs_checker_requires_terminal_success_and_protocol(tmp_path, state, status, success, steps, expected):
    runtime = tmp_path / "runtime"
    (runtime / "logs").mkdir(parents=True)
    (runtime / "results/image").mkdir(parents=True)
    if success:
        (runtime / "run-success").touch()
    if steps:
        (runtime / "results/steps.tsv").write_text("image\tbench\t0\t1\n")
    (runtime / "results/exit-code").write_text("0\n")
    (runtime / "results/image/bench.jsonl").write_text("{}\n")
    qstat = tmp_path / "qstat"
    qstat.write_text(f'#!/bin/bash\nprintf "    job_state = {state}\\n    Exit_status = {status}\\n"\n')
    qstat.chmod(0o755)
    checker = tmp_path / "check-job.sh"
    checker.write_text((ROOT / "platforms/aspire2ap/kernel-bench/check-job.sh").read_text().replace("/opt/pbs/bin/qstat", str(qstat)))
    result = subprocess.run(["bash", str(checker)], env={**os.environ, "RUNTIME_DIR": str(runtime), "PBS_JOB_ID": "123.server"}, capture_output=True, text=True, check=False)
    assert result.returncode == expected, result.stdout + result.stderr


def test_aspire_serving_baseline_mismatch_rejects_before_staging(tmp_path):
    ws, env = aspire_fixture(tmp_path)
    path = ws / "configs/aspire-serving.json"
    config = json.loads(path.read_text())
    config["deep_gemm_overlay_patches"] = ["different.patch"]
    path.write_text(json.dumps(config))
    result = subprocess.run(["bash", str(ws / "scripts/provision-firmus-kernel-bench.sh"), "aspire2ap"], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "ASPIRE serving baseline differs" in result.stderr
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "enroot.log").exists()
