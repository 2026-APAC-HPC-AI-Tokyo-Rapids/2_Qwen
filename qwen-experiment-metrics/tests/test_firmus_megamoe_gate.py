"""Exercise finite-gate staging and terminal checks without GPU access."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
spec = importlib.util.spec_from_file_location("kernel_tags", ROOT / "scripts/check-kernel-experiment-tags.py")
TAGS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TAGS)


@pytest.mark.parametrize("extra", ["[profile]", "[nodes:2]", "[cluster:aspire2ap]", "[require-build]", "[dynamo]", "[cutlass-grid-probe]"])
def test_megamoe_rejects_mixed_or_rebuilt_stage(extra):
    with pytest.raises(ValueError):
        TAGS.validate("[cluster:firmus] [nodes:1] [megamoe-gate] " + extra)


def test_megamoe_explicit_finite_stage():
    TAGS.validate("[cluster:firmus] [nodes:1] [megamoe-gate] finite numerical and graph gate")
    with pytest.raises(ValueError):
        TAGS.validate("[cluster:firmus] [megamoe-gate]")


@pytest.mark.parametrize("corrupt", [False, True, "image"])
def test_staging_checks_image_before_creating_runtime(tmp_path, corrupt):
    workspace = tmp_path / "workspace"
    for relative in ["configs", "platforms/firmus", "research/firmus-vllm-megamoe"]:
        (workspace / relative).mkdir(parents=True)
    image = tmp_path / "image.sqsh"
    image.write_bytes(b"fixture image")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(json.dumps({"source_commits": {"qwen": "build"}, "container_artifact": {"sha256": "wrong" if corrupt is True else digest, "size_bytes": image.stat().st_size}}))
    config = {"diagnostic": "vllm-sm90-megamoe", "world_size": 4, "image": {"path": str(image), "sha256": digest, "size_bytes": image.stat().st_size, "provenance": str(sidecar), "build_commit": "build"}}
    (workspace / "configs/firmus-vllm-megamoe-gate.json").write_text(json.dumps(config))
    if corrupt == "image":
        image.write_bytes(b"Fixture image")
    (workspace / "platforms/firmus/megamoe-gate.sbatch").write_text("#!/bin/bash\nexit 0\n")
    (workspace / "research/firmus-vllm-megamoe/gpu_gate.py").write_text("# fixture\n")
    output = tmp_path / "outputs"
    result = subprocess.run(["bash", str(ROOT / "scripts/provision-firmus-megamoe-gate.sh")], check=False, capture_output=True, text=True, env={**os.environ, "GITHUB_WORKSPACE": str(workspace), "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": "compute", "GITHUB_OUTPUT": str(output), "RUNTIME_ROOT": str(tmp_path / "runs"), "PROFILE": "false", "REQUIRE_BUILD": "false"})
    if corrupt:
        assert result.returncode != 0
        assert not output.exists() and not (tmp_path / "runs").exists()
    else:
        assert result.returncode == 0, result.stderr
        runtime = Path(output.read_text().strip().split("=", 1)[1])
        identity = json.loads((runtime / "source/provenance.json").read_text())
        assert identity["measurement_mode"] == "kernel-gate" and identity["workload"] == "finite-kernel"
        assert identity["commit"] == "compute" and identity["world_size"] == 4 and not identity["profile"]
        assert (runtime / "vllm.sqsh").resolve() == image
        assert (runtime / "source/diagnostic-sources.sha256").is_file()
        assert not (runtime / "model").exists() and not (runtime / "tokenizer").exists()


@pytest.mark.parametrize("fault,expected", [(None, 0), ("math", 1), ("graph", 1), ("coverage", 1), ("score", 1), ("scheduler", 1)])
def test_terminal_gate_requires_math_graph_four_ranks_without_score(tmp_path, fault, expected):
    for relative in ["bin", "source", "logs", "results"]:
        (tmp_path / relative).mkdir()
    (tmp_path / "run-success").touch()
    (tmp_path / "logs/slurm-7.log").write_text("gate finished\n")
    (tmp_path / "source/provenance.json").write_text(json.dumps({"diagnostic": "vllm-sm90-megamoe", "measurement_mode": "kernel-gate", "profile": False, "world_size": 4}))
    result = {"schema": "vllm-sm90-megamoe-gpu-gate/v1", "completed": True, "correctness_passed": fault != "math", "graph_passed": fault != "graph", "world_size": 4, "gpus": [{"rank": rank} for rank in range(3 if fault == "coverage" else 4)], "useful_for_serving": False}
    (tmp_path / "results/gate.json").write_text(json.dumps(result))
    if fault == "score":
        (tmp_path / "results/score.json").write_text("{}")
    state = "FAILED|1:0" if fault == "scheduler" else "COMPLETED|0:0"
    for name, body in {"squeue": "#!/bin/sh\nexit 0\n", "sacct": f"#!/bin/sh\necho '7|{state}|00:10:00|1'\n"}.items():
        path = tmp_path / "bin" / name
        path.write_text(body)
        path.chmod(0o755)
    checked = subprocess.run(["bash", str(ROOT / "platforms/firmus/check-megamoe-gate.sh")], check=False, capture_output=True, text=True, env={**os.environ, "PATH": f"{tmp_path}/bin:{os.environ['PATH']}", "SLURM_JOB_ID": "7", "RUNTIME_DIR": str(tmp_path)})
    assert checked.returncode == expected, checked.stdout + checked.stderr


def test_finite_harness_cpu_ownership_and_call_contract(tmp_path):
    report = tmp_path / "cpu-check.json"
    subprocess.run([sys.executable, str(ROOT / "research/firmus-vllm-megamoe/check_gpu_gate_cpu.py"), "--output", str(report)], check=True)
    result = json.loads(report.read_text())
    assert result["passed"] is True and result["cuda_execution"] is False
    assert result["finite_layout_cases"] == 160
    assert result["native_mega_effective_mask_cases"] == 160
    assert result["pipeline_wiring"]["double_dispatch_or_allreduce"] is False


@pytest.mark.parametrize("job_ids", ["0,1,2,3,4,5,6,7", "4,5,6,7"])
def test_batch_passes_four_local_devices_instead_of_global_job_ids(tmp_path, job_ids):
    for relative in ["bin", "source", "logs", "results"]:
        (tmp_path / relative).mkdir()
    launcher = tmp_path / "bin/srun"
    launcher.write_text("#!/bin/sh\nprintf '%s' \"$NVIDIA_VISIBLE_DEVICES\" >\"$SLURM_SUBMIT_DIR/visible\"\nprintf '%s\\n' \"$*\" >\"$SLURM_SUBMIT_DIR/command\"\nprintf '{}' >\"$SLURM_SUBMIT_DIR/results/gate.json\"\n")
    launcher.chmod(0o755)
    result = subprocess.run(["bash", str(ROOT / "platforms/firmus/megamoe-gate.sbatch")], check=False, capture_output=True, text=True, env={**os.environ, "PATH": f"{tmp_path}/bin:{os.environ['PATH']}", "SLURM_SUBMIT_DIR": str(tmp_path), "SLURM_NNODES": "1", "SLURM_JOB_GPUS": job_ids, "CUDA_VISIBLE_DEVICES": job_ids})
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "visible").read_text() == "0,1,2,3"
    assert "--gres=gpu:4" in (tmp_path / "command").read_text()
    assert f"job_global_gpu_ids={job_ids}" in (tmp_path / "source/gpu-visibility.txt").read_text()
