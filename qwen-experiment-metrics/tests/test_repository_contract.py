from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_container_build_contract() -> None:
    build = (ROOT / "scripts/build-sglang-container.sh").read_text()
    recipe = (ROOT / "containers/Dockerfile").read_text()
    dependencies = (ROOT / "containers/install-python-dependencies.sh").read_text()
    source = (ROOT / "containers/install-sglang-source.sh").read_text()
    toolchain = (ROOT / "containers/install-system-toolchain.sh").read_text()

    for command in ("buildah --root", "skopeo copy", "umoci stat", "mksquashfs"):
        assert command in build
    assert "--isolation chroot" in build
    assert "BUILDAH_ISOLATION=chroot" in build
    assert "STORAGE_DRIVER=overlay" in build
    assert "--layers" in build
    assert "--squash" not in build
    assert "-all-root -no-xattrs" in build
    assert "-processors 8 -mem 8G" in build
    assert "COPY sglang-project-metadata /opt/sglang-project-metadata" in recipe
    assert "COPY sglang-source /opt/sglang/source" in recipe
    assert recipe.index("RUN /opt/apac/install-python-dependencies.sh") < recipe.index(
        "COPY sglang-source /opt/sglang/source"
    )
    assert '"aiperf==0.12.0" "sglang-router==0.3.2"' in dependencies
    assert "SGLANG_BUILD_RUST_EXTS=none" in source
    assert "rustup" not in toolchain


def test_workflow_keeps_build_and_cluster_controls_explicit() -> None:
    entry = (ROOT / ".github/workflows/qwen.yml").read_text()
    container = (ROOT / ".github/workflows/aws-container.yml").read_text()
    validation = (ROOT / ".github/workflows/validate.yml").read_text()

    for cluster in ("genkai", "aspire2ap", "firmus"):
        assert f"[cluster:{cluster}]" in entry
        assert f"run_{cluster}:" in entry
    for tag in ("[nodes:1]", "[nodes:4]", "[require-build]", "[profile]"):
        assert tag in entry
    assert "startsWith(github.event.head_commit.message, '[require-build]')" in entry
    assert "uses: ./.github/workflows/validate.yml" in entry
    assert "fail-fast: false" in container
    assert "m7i.2xlarge" in container
    assert "actionlint-py==1.7.12.24" in validation
    assert "-name '*.slurm'" in validation


def test_firmus_pd_contract_uses_all_sixteen_gpus() -> None:
    workflow = (ROOT / ".github/workflows/firmus.yml").read_text()
    batch = (ROOT / "platforms/firmus/run-qwen-pd.slurm").read_text()
    job = (ROOT / "scripts/run-qwen-pd-job.sh").read_text()
    node = (ROOT / "scripts/run-qwen-pd-node.sh").read_text()
    router = (ROOT / "scripts/run-qwen-pd-router.sh").read_text()

    assert "test '${{ inputs.node_count }}' = 2" in workflow
    assert "#SBATCH --nodes=2" in batch
    assert "#SBATCH --gpus-per-node=8" in batch
    assert "#SBATCH --time=00:17:00" in batch
    assert job.count("/health") >= 6
    assert 'models.pending.json' in job
    assert 'index("qwen") != null' in job
    assert node.count("start_worker prefill") == 4
    assert node.count("start_worker decode") == 2
    assert "--tp-size \"${tp}\"" in node
    assert "--ep-size 4 --max-running-requests 256" in node
    assert "--disaggregation-transfer-backend nixl" in node
    assert 'SGLANG_HOST_IP=${node_ip}' in node
    assert 'HEAD_IP="${head_ip}" TAIL_IP="${tail_ip}"' in job
    assert 'http://${HEAD_IP}:30100' in router
    assert router.count("--prefill ") == 4
    assert router.count("--decode ") == 2
    assert "--pd-disaggregation" in router
    assert '-m "${MODEL_PATH}:/model"' in router
    assert "HF_HUB_OFFLINE=1" in router
    assert "sample-system.sh" in node
    assert "sample-system.sh" in (ROOT / "scripts/provision-qwen-pd.sh").read_text()


def test_submission_script_keeps_profile_data_separate() -> None:
    submission = (ROOT / "submission/run/submit.sbatch").read_text()

    assert 'gpu-node-${node_rank}.csv' in submission
    for prefix in ("cpu", "network", "infiniband"):
        assert f"{prefix}-node-${{rank}}.csv" in submission
    assert '${OUTPUT_DIR}/profile' in submission
    assert 'SGLANG_HOST_IP=${node_ip}' in submission


def test_compute_scripts_have_no_cloud_credentials() -> None:
    compute_files = [
        ROOT / "scripts/run-qwen-pd-job.sh",
        ROOT / "scripts/run-qwen-pd-node.sh",
        ROOT / "scripts/run-qwen-pd-router.sh",
        ROOT / "scripts/run-aiperf-container.sh",
        ROOT / "scripts/run-aiperf.sh",
        ROOT / "platforms/firmus/run-qwen-pd.slurm",
    ]
    for path in compute_files:
        content = path.read_text()
        assert "GOOGLE_CREDENTIAL" not in content
        assert "BIGQUERY" not in content
        assert "AWS_ACCESS_KEY_ID" not in content
        assert "AWS_SECRET_ACCESS_KEY" not in content


def test_active_shell_commands_have_no_line_continuations() -> None:
    paths = list((ROOT / "scripts").glob("*.sh"))
    paths += list((ROOT / "platforms").rglob("*.pjm"))
    paths += list((ROOT / "platforms").rglob("*.pbs"))
    paths += list((ROOT / "platforms").rglob("*.slurm"))
    paths += list((ROOT / "submission").rglob("*.sh"))
    paths += list((ROOT / "submission").rglob("*.sbatch"))
    for path in paths:
        assert not any(
            line.endswith("\\") for line in path.read_text().splitlines()
        ), path


def test_submission_contains_reproducible_official_entrypoint() -> None:
    setup = (ROOT / "submission" / "build" / "setup.sh").read_text()
    batch = (ROOT / "submission" / "run" / "submit.sbatch").read_text()

    assert 'exec "${repository}/scripts/build-sglang-container.sh"' in setup
    assert "#SBATCH --nodes=2" in batch
    assert "#SBATCH --gpus-per-node=8" in batch
    assert batch.count("start_worker prefill") == 4
    assert batch.count("start_worker decode") == 2
    assert "--pd-disaggregation" in batch
    assert "--warmup-duration 60 --benchmark-duration 120 --benchmark-grace-period 30" in batch
    assert "--synthetic-input-tokens-mean 4000" in batch
    assert "--output-tokens-mean 200" in batch
    assert "--random-seed 2026" in batch
    assert "--cache-bust first_turn_prefix" in batch
    assert "goodput * 200 if eligible else 0.0" in batch
