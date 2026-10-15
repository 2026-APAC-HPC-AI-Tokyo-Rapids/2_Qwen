"""Exercise finite-gate staging and terminal checks without GPU access."""
import ast
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
    (workspace / "research/firmus-vllm-megamoe/adapter_gpu_gate.py").write_text("# actual adapter fixture\n")
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
        assert (runtime / "source/adapter_gpu_gate.py").read_text() == "# actual adapter fixture\n"
        assert (runtime / "source/gpu_gate.py").read_text() == "# fixture\n"
        assert not (runtime / "model").exists() and not (runtime / "tokenizer").exists()


@pytest.mark.parametrize("fault,expected", [(None, 0), *[(fault, 1) for fault in ("math", "graph", "coverage", "score", "scheduler", "rows", "replays", "uuid", "replay_domain", "record_rows", "nan", "l2", "maximum", "empty")]])
def test_terminal_gate_requires_math_graph_four_ranks_without_score(tmp_path, fault, expected):
    for relative in ["bin", "source", "logs", "results", "scripts/megamoe"]:
        (tmp_path / relative).mkdir(parents=True)
    (tmp_path / "scripts/megamoe/validate_adapter_gate.py").write_bytes((ROOT / "research/firmus-vllm-megamoe/validate_adapter_gate.py").read_bytes())
    (tmp_path / "run-success").touch()
    (tmp_path / "logs/slurm-7.log").write_text("gate finished\n")
    (tmp_path / "source/provenance.json").write_text(json.dumps({"diagnostic": "vllm-sm90-megamoe", "measurement_mode": "kernel-gate", "profile": False, "world_size": 4}))
    result = {"schema": "vllm-sm90-actual-adapter-gate/v1", "completed": True, "correctness_passed": fault != "math", "graph_passed": fault != "graph", "world_size": 4, "gpus": [{"rank": rank, "uuid": f"GPU-{rank}"} for rank in range(3 if fault == "coverage" else 4)], "official_benchmark_score": None}
    result.update(adapter_manifest={"backend": "apac_sm90_megamoe"}, capacity_requested=4096, cases=[{"counts": counts, "records": [{"rank": rank, "replay": replay, "rows": counts[rank], "errors": {"relative_l2": 0.01, "max_scaled_error": 0.02, "masked_rows_zero": True} if counts[rank] else {"empty_rank_participated": True}} for rank in range(4) for replay in range(4)]} for counts in [[1,1,1,1],[128,0,17,64],[128,128,128,128],[4096,4096,4096,4096]]])
    if fault == "rows":
        result["cases"][-1]["counts"] = [384] * 4
    elif fault == "replays":
        result["cases"][0]["records"][-1] = result["cases"][0]["records"][0]
    elif fault == "uuid":
        result["gpus"][0]["uuid"] = "GPU-1"
    elif fault == "replay_domain":
        result["cases"][0]["records"][0]["replay"] = 4
    elif fault == "record_rows":
        result["cases"][0]["records"][0]["rows"] = 999
    elif fault in ("nan", "l2", "maximum"):
        key, value = {"nan": ("relative_l2", float("nan")), "l2": ("relative_l2", 0.025), "maximum": ("max_scaled_error", 0.08)}[fault]
        result["cases"][0]["records"][0]["errors"][key] = value
    elif fault == "empty":
        result["cases"][1]["records"][4]["errors"]["empty_rank_participated"] = False
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
    if fault is None:
        config_path = tmp_path / "serving.json"
        config_path.write_text(json.dumps({"images": {"vllm": {"sha256": "a" * 64, "size_bytes": 123}}}))
        provenance_path = tmp_path / "source/provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["container"] = {"sha256": "a" * 64, "size_bytes": 123}
        provenance_path.write_text(json.dumps(provenance))
        for name in ("adapter_gpu_gate.py", "gpu_gate.py"):
            (tmp_path / "source" / name).write_bytes((ROOT / "research/firmus-vllm-megamoe" / name).read_bytes())
        command = [sys.executable, str(ROOT / "research/firmus-vllm-megamoe/validate_adapter_gate.py"), str(tmp_path), "--serving-config", str(config_path)]
        assert subprocess.run(command, capture_output=True, check=False).returncode == 0
        for key, bad in (("sha256", "b" * 64), ("size_bytes", 124)):
            original = provenance["container"][key]
            provenance["container"][key] = bad
            provenance_path.write_text(json.dumps(provenance))
            assert subprocess.run(command, capture_output=True, check=False).returncode != 0
            provenance["container"][key] = original
        provenance_path.write_text(json.dumps(provenance))
        (tmp_path / "source/adapter_gpu_gate.py").write_text("# stale diagnostic\n")
        assert subprocess.run(command, capture_output=True, check=False).returncode != 0


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


def test_actual_adapter_initializes_native_world_before_device_communicator():
    # Reproduce the failed dependency: Torch PG alone does not set vLLM node state.
    from datetime import timedelta
    from types import SimpleNamespace

    state = {}

    def torch_init(backend, *, timeout, device_id):
        assert backend == "nccl" and device_id == "cuda:2"
        state["torch"] = True

    def native_init(*, world_size, rank, local_rank, backend, timeout):
        assert state["torch"]
        assert (world_size, rank, local_rank, backend) == (4, 2, 2, "nccl")
        assert timeout == timedelta(seconds=300)
        state["node_count"] = 1

    def coordinator(ranks, local_rank, backend, **kwargs):
        assert state.get("node_count") == 1, "distributed environment is not initialized"
        assert ranks == [[0, 1, 2, 3]] and local_rank == 2
        assert kwargs["use_device_communicator"] is True
        state["coordinator"] = True

    path = ROOT / "research/firmus-vllm-megamoe/adapter_gpu_gate.py"
    run = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "run")
    start = next(i for i, node in enumerate(run.body) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "dist.init_process_group")
    end = next(i for i, node in enumerate(run.body) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "group" for target in node.targets))
    setup = ast.Module(body=run.body[start:end + 1], type_ignores=[])
    namespace = {"dist": SimpleNamespace(init_process_group=torch_init), "device": "cuda:2", "rank": 2, "timedelta": timedelta, "init_distributed_environment": native_init, "GroupCoordinator": coordinator}
    exec(compile(setup, str(path), "exec"), namespace)  # noqa: S102 -- Selected local probe initialization, with stubbed collectives.
    assert state == {"torch": True, "node_count": 1, "coordinator": True}


@pytest.mark.parametrize("rank", [0, 2])
def test_adapter_initialization_failure_is_archived_without_collectives(tmp_path, monkeypatch, rank):
    monkeypatch.syspath_prepend(str(ROOT / "research/firmus-vllm-megamoe"))
    import adapter_gpu_gate

    monkeypatch.setenv("RANK", str(rank))

    def fail(_output):
        raise RuntimeError("distributed environment is not initialized")

    monkeypatch.setattr(adapter_gpu_gate, "run", fail)
    output = tmp_path / "gate.json"
    with pytest.raises(RuntimeError, match="distributed environment"):
        adapter_gpu_gate.main(output)
    error = json.loads((tmp_path / f"gate.rank{rank}.error.json").read_text())
    assert error["rank"] == rank and "distributed environment is not initialized" in error["traceback"]
    if rank == 0:
        result = json.loads(output.read_text())
        assert result["completed"] is False and result["phase"] == "initialization"
    else:
        assert not output.exists()  # Other ranks must not overwrite rank 0's evidence.


def test_full_serving_is_blocked_before_staging_without_adapter_gate(tmp_path):
    environment = {**os.environ, "GITHUB_WORKSPACE": str(ROOT), "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": "a" * 40, "GITHUB_OUTPUT": str(tmp_path / "outputs"), "RUNTIME_ROOT": str(tmp_path / "runs"), "PROFILE": "false", "REQUIRE_BUILD": "false"}
    result = subprocess.run(["bash", str(ROOT / "scripts/provision-firmus-framework.sh")], env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 64 and "no successful actual-adapter GPU gate" in result.stderr
    assert not (tmp_path / "runs").exists() and not (tmp_path / "outputs").exists()
