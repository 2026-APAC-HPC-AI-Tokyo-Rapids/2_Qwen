from __future__ import annotations

import ast
import json
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_container_build_uses_root_overlay_and_validates_oci() -> None:
    script = (ROOT / "scripts/build-sglang-container.sh").read_text(encoding="utf-8")
    for command in ("buildah", "buildah --root", "skopeo copy", "umoci stat", "mksquashfs"):
        assert command in script
    assert "--isolation chroot" in script
    assert "BUILDAH_ISOLATION=chroot" in script
    assert "STORAGE_DRIVER=overlay" in script
    assert "STORAGE_OPTS" not in script
    assert '[[ "$(id -u)" -eq 0 ]]' in script
    assert 'storage_root="${BUILD_STORAGE_ROOT:-/var/lib/containers/storage}"' in script
    assert "safe.directory" not in script
    assert 'tar --no-same-owner -I zstd -xf "${input_archive}"' in (ROOT / "aws/run-build-instance.sh").read_text(encoding="utf-8")
    assert "--layers" in script
    assert "--squash" not in script
    assert "--layers=false" not in script
    assert "rmi --all" not in script
    assert 'test_container="$(buildah ' in script
    assert 'run --isolation chroot "${test_container}"' in script
    assert 'mount "${test_container}"' in script
    assert "unshare --mount" not in script
    assert "umoci unpack" not in script
    assert "test -x /sbin/ldconfig" in script
    assert "--no-preserve=ownership,xattr" in script
    assert '"container-sidecar/v1"' in script
    assert "-all-root -no-xattrs" in script
    assert "-no-duplicates" in script
    assert "-no-fragments" in script
    assert "-processors 8 -mem 8G" in script
    assert 'squashfs_work="${BUILD_DIR}/sglang.sqsh"' in script
    assert "singularity" not in script.lower()


def test_dockerfile_resolves_sglang_dependencies_before_copying_source() -> None:
    recipe = (ROOT / "containers/Dockerfile").read_text(encoding="utf-8")
    dependencies = (ROOT / "containers/install-python-dependencies.sh").read_text(
        encoding="utf-8"
    )
    source = (ROOT / "containers/install-sglang-source.sh").read_text(encoding="utf-8")
    assert "COPY sglang-project-metadata /opt/sglang-project-metadata" in recipe
    assert "COPY sglang-source /opt/sglang/source" in recipe
    assert recipe.index(
        "COPY sglang-project-metadata /opt/sglang-project-metadata"
    ) < recipe.index("COPY sglang-source /opt/sglang/source")
    assert recipe.index("RUN /opt/apac/install-python-dependencies.sh") < recipe.index(
        "COPY sglang-source /opt/sglang/source"
    )
    assert recipe.index("RUN /opt/apac/install-sglang-source.sh") < recipe.index(
        "COPY apac-manifest.json /opt/apac/manifest.json"
    )
    assert "build-sglang-image.sh" not in recipe
    assert "UV_PROJECT_ENVIRONMENT=/opt/sglang/venv" in dependencies
    assert (
        "uv sync --project /opt/sglang-project-metadata/python --no-install-project"
        in dependencies
    )
    assert "--locked" not in dependencies
    assert "python/uv.lock" not in (
        ROOT / "scripts/build-sglang-container.sh"
    ).read_text(encoding="utf-8")
    assert (
        "uv pip install --python /opt/sglang/venv/bin/python --no-deps -e ." in source
    )
    assert "from sglang.srt.qwen_experiment_metrics import send_event" in (
        ROOT / "scripts/build-sglang-container.sh"
    ).read_text(encoding="utf-8")


def test_rustup_download_retries_observed_dns_failure() -> None:
    toolchain = (ROOT / "containers/install-system-toolchain.sh").read_text(encoding="utf-8")
    rustup = next(line for line in toolchain.splitlines() if "https://sh.rustup.rs" in line)
    assert "--retry 3 --retry-all-errors --retry-delay 2" in rustup


def test_first_parent_reuse_has_only_size_and_sha_gates() -> None:
    selector = (ROOT / "scripts/select-qwen-container.sh").read_text(encoding="utf-8")
    aws_workflow = (ROOT / ".github/workflows/aws-container.yml").read_text(encoding="utf-8")
    assert "scripts/select-qwen-container.sh" in aws_workflow
    assert aws_workflow.count("scripts/select-qwen-container.sh") == 1
    assert "select-aspire2ap:" not in aws_workflow and "select-firmus:" not in aws_workflow
    assert "decide:" not in aws_workflow and "finalize:" not in aws_workflow
    for name in ("genkai.yml", "aspire2ap.yml", "firmus.yml"):
        workflow = (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
        assert "select-qwen-container.sh" not in workflow
        assert "build-sglang-container.sh" not in workflow
        assert "provision-qwen-check-env.sh" in workflow
    assert 'git rev-parse --verify "${commit}^"' in selector
    assert "size_bytes" in selector and "sha256sum" in selector
    assert "CONTAINER_ARCHIVE_DIR" in selector
    assert "gh run view" not in selector
    assert "inventory" not in selector
    assert "enroot start" not in selector


def test_aws_builder_distributes_only_new_builds_and_reuse_stays_local() -> None:
    infrastructure = (ROOT / "aws/qwen-build-infrastructure.yml").read_text(encoding="utf-8")
    aws_workflow = (ROOT / ".github/workflows/aws-container.yml").read_text(encoding="utf-8")
    materialize = (ROOT / "scripts/materialize-qwen-container.sh").read_text(encoding="utf-8")
    provision = (ROOT / "scripts/provision-qwen-check-env.sh").read_text(encoding="utf-8")
    check = (ROOT / "scripts/run-qwen-check-env.sh").read_text(encoding="utf-8")
    assert "m7i.2xlarge" in aws_workflow
    assert "publish-reuse:" not in aws_workflow
    assert "materialize:" in aws_workflow
    assert "fail-fast: false" in aws_workflow
    assert "build時はS3から取得し、再利用時はarchive内でハードリンク" in aws_workflow
    assert '"VolumeSize":300' in aws_workflow
    assert '"Iops":6000' in aws_workflow and '"Throughput":500' in aws_workflow
    assert "--instance-initiated-shutdown-behavior terminate" in aws_workflow
    assert 'build_command="set -eu;' in aws_workflow
    assert "ec2 terminate-instances" in aws_workflow
    assert "systemd-run --unit=qwen-build-expiration --on-active=6h" in (ROOT / "aws/ec2-user-data.sh").read_text(encoding="utf-8")
    assert "awscli-exe-linux-x86_64.zip" in (ROOT / "aws/ec2-user-data.sh").read_text(encoding="utf-8")
    assert "BlockPublicPolicy: true" in infrastructure
    assert 's3-object-transfer.py" download' in materialize
    assert 'ln -f -- "${source_image}" "${current_image}"' in materialize
    assert '"${CONTAINER_ARCHIVE_DIR}/${GITHUB_RUN_ID}.sqsh"' in materialize
    assert 'cp --reflink=auto --sparse=always "${archive_image}" "${runtime_dir}/sglang.sqsh"' in provision
    assert "s3-object-transfer.py" not in provision and "configure-aws-credentials" not in provision
    assert 'local_archive:true,enroot_direct_squashfs:true' in provision
    assert 'enroot start -e NVIDIA_VISIBLE_DEVICES=all' in check
    assert "CUDA available: True" in check
    assert "^GPU [0-9,]*0[0-9,]*:" in check
    assert "enroot create" not in check and "unsquashfs" not in check


def test_entry_workflow_keeps_push_and_dispatch_controls_closed() -> None:
    workflow = (ROOT / ".github/workflows/qwen.yml").read_text(encoding="utf-8")
    validation = (ROOT / ".github/workflows/validate.yml").read_text(encoding="utf-8")
    assert "push:" in workflow and "workflow_dispatch:" in workflow
    assert "uses: ./.github/workflows/validate.yml" in workflow
    for cluster in ("genkai", "aspire2ap", "firmus"):
        assert f"[cluster:{cluster}]" in workflow
    for tag in ("[nodes:1]", "[nodes:4]", "[require-build]"):
        assert tag in workflow
    assert "startsWith(github.event.head_commit.message, '[require-build]')" in workflow
    assert workflow.count("needs: [validate, container]") == 3
    assert all(f"run_{cluster}:" in workflow for cluster in ("genkai", "aspire2ap", "firmus"))
    assert "needs: [genkai, aspire2ap, firmus]" in workflow
    assert "QWEN_ARTIFACT_ARCHIVE_ROOT: /hdd_raid/samba/net_share/qwen-artifacts" in workflow
    assert "shell: home-lab-shell {0}" in workflow
    assert "actions/download-artifact@v8" in workflow
    assert "SHA256SUMS" in workflow
    assert "actions/artifacts/${artifact_id}" in workflow
    assert "[skip ci]" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "actionlint-py==1.7.12.24" in validation


def test_enroot_runtime_is_dummy_and_compute_local() -> None:
    node = (ROOT / "scripts/run-qwen35-node.sh").read_text(encoding="utf-8")
    provision = (ROOT / "scripts/provision-qwen-runtime.sh").read_text(encoding="utf-8")
    fetch_dataset = (ROOT / "scripts/fetch-sharegpt.sh").read_text(encoding="utf-8")
    job = (ROOT / "scripts/run-qwen35-job.sh").read_text(encoding="utf-8")
    assert "enroot start" in node
    assert "--load-format dummy" in node
    assert "--dataset-name random" in node
    assert "--dataset-path /dataset/ShareGPT_V3_unfiltered_cleaned_split.json" in node
    assert 'dataset_name:"random"' in provision
    assert 'kind:"hub_file"' in provision
    assert '"${dataset_path}:/dataset/ShareGPT_V3_unfiltered_cleaned_split.json:none:x-create=file,bind,ro,nosuid,nodev"' in node
    assert 'scripts/fetch-sharegpt.sh" "${DATASET_CACHE_ROOT}"' in provision
    assert "revision=192ab2185289094fc556ec8ce5ce1e8e587154ca" in fetch_dataset
    assert (
        'if [[ -s "${target_file}" ]]; then printf \'%s\\n\' "${target_file}"; exit 0; fi'
        in fetch_dataset
    )
    assert 'test -s "${staging_dir}/${filename}"' in fetch_dataset
    assert 'mv -T "${staging_dir}" "${target_dir}"' in fetch_dataset
    assert "sha256sum" not in fetch_dataset
    assert "singularity exec" not in node
    assert "NVIDIA_VISIBLE_DEVICES=all" in node
    assert "NVIDIA_DISABLE_REQUIRE=1" in node
    assert "MELLANOX_VISIBLE_DEVICES=all" in node
    assert 'unsquashfs -cat "${CONTAINER_ARCHIVE}" opt/sglang/offline-throughput-source-blob-oid.txt' in provision
    assert 'unsquashfs -cat "${CONTAINER_ARCHIVE}" opt/sglang/server-args-source-blob-oid.txt' in provision
    assert "enroot start -e NVIDIA_VISIBLE_DEVICES=void" not in provision
    assert "HOME=/runtime-home" in node
    assert "XDG_CACHE_HOME=/runtime-home/.cache" in node
    assert '"${container_home}:/runtime-home"' in node
    assert "FLASHINFER_WORKSPACE_BASE=/jit-cache" in node
    assert 'jit_temporary_root="${TMPDIR:-${TEMPDIR:-${TEMP:-/tmp}}}"' in node
    assert 'jit_temporary="$(mktemp -d "${jit_temporary_root%/}/qwen-jit.XXXXXXXXXX")"' in node
    assert 'jit_cache="${jit_temporary}/cache"' in node
    assert 'enroot_data="${jit_temporary}/enroot-data"' in node
    assert 'export ENROOT_DATA_PATH="${enroot_data}"' in node
    assert 'export ENROOT_RUNTIME_PATH="${enroot_runtime}"' in node
    assert 'export ENROOT_CACHE_PATH="${enroot_cache}"' in node
    assert 'export ENROOT_TEMP_PATH="${enroot_temp}"' in node
    assert 'container_created=0' in node
    assert 'trap cleanup EXIT' in node
    assert 'enroot create --name "${container_name}" "${CONTAINER_IMAGE}"' in node
    assert 'container_created=1' in node
    assert node.index("trap cleanup EXIT") < node.index("enroot create --name ")
    assert (
        'if [[ "${container_created}" -eq 1 ]]; then enroot remove -f "${container_name}" 2>/dev/null || true; fi'
        in node
    )
    assert 'rm -rf -- "${jit_temporary:?}"' in node
    assert "enroot start -e NVIDIA_VISIBLE_DEVICES=void" not in node
    assert '"qwen-runtime-source-oids/v1"' in provision
    assert "runtime-source-oids.json" in node
    assert "capture-topology" in node
    assert "--container-argv" in node
    assert "--container-launcher" in node
    assert "--container-target" in node
    assert '--container-target "${container_name}"' in node
    assert '--container-image "${CONTAINER_IMAGE}"' not in node
    assert "QWEN_EXPERIMENT_METRICS_DATASET_MANIFEST_SHA256" not in node
    assert (
        'if [[ "${node_rank}" -eq 0 && -s "${RUNTIME_DIR}/source/workload-warmup.json" ]]; then "${metrics_cli}" record-workload --manifest "${manifest}" --runtime-dir "${RUNTIME_DIR}" --kind warmup --source "${RUNTIME_DIR}/source/workload-warmup.json"; fi'
        in node
    )
    assert (
        'if [[ "${node_rank}" -eq 0 && -s "${RUNTIME_DIR}/source/workload-measured.json" ]]; then "${metrics_cli}" record-workload --manifest "${manifest}" --runtime-dir "${RUNTIME_DIR}" --kind measured --source "${RUNTIME_DIR}/source/workload-measured.json"; fi'
        in node
    )
    assert '"dummy-random-dataset/v1"' in provision
    assert 'export UV_CACHE_DIR="${RUNNER_TEMP}/uv-cache"' in provision
    assert 'export UV_PYTHON_INSTALL_DIR="${RUNNER_TEMP}/uv-python"' in provision
    assert 'export UV_PYTHON_INSTALL_DIR="${runtime_dir}/qwen-experiment-metrics/.uv-python"' in provision
    assert "--managed-python --frozen" in provision
    for pattern in ("*.json", "*.txt", "*.model", "*.tiktoken"):
        assert f"--include '{pattern}'" in provision
    for pattern in ("*.safetensors", "*.bin"):
        assert f"--exclude '{pattern}'" in provision
    assert "record-provenance" in provision
    assert "capture-runtime-launchers" in job
    assert "start-run-lifecycle" in job and "finish-run-lifecycle" in job
    assert "(.generation_wall_sec | tonumber) > 0" in job
    assert ".generation_wall_sec | numbers" not in job
    assert 'ENROOT_SYSCONF_PATH="${RUNTIME_DIR}/enroot-system"' in job
    assert "--ldconfig=/sbin/ldconfig" in provision
    assert "sed -i 's%--ldconfig=" in provision
    assert "sed -i 's|--ldconfig=" not in provision
    assert "10-cgroups.sh" in provision
    assert "--no-preserve=ownership,xattr" in provision


def test_topology_capture_uses_enroot_vocabulary_and_driver_injection() -> None:
    cli = (
        ROOT / "qwen-experiment-metrics/src/qwen_experiment_metrics/cli.py"
    ).read_text(encoding="utf-8")
    sources = (
        ROOT / "qwen-experiment-metrics/src/qwen_experiment_metrics/system_sources.py"
    ).read_text(encoding="utf-8")
    contract = (
        ROOT
        / "qwen-experiment-metrics/src/qwen_experiment_metrics/contracts/qwen-v2.toml"
    ).read_text(encoding="utf-8")
    for option in ("--container-argv", "--container-launcher", "--container-target"):
        assert option in cli
    assert '"start"' in sources
    assert '"NVIDIA_VISIBLE_DEVICES=all"' in sources
    assert '"NVIDIA_DISABLE_REQUIRE=1"' in sources
    assert "container_argv" in contract
    assert "singularity_argv" not in contract
    assert "sif-sidecar" not in sources
    assert "sif-sidecar" not in contract


def test_job_success_gate_accepts_primary_result_decimal_encodings() -> None:
    job = (ROOT / "scripts/run-qwen35-job.sh").read_text(encoding="utf-8")
    line = next(row for row in job.splitlines() if "jq -e" in row and "generation_wall_sec" in row)
    jq_filter = line.split("jq -e '", 1)[1].split("' source/primary-result.json", 1)[0]
    for wall in ("2.342779523", 2.342779523):
        payload = json.dumps({"error_count": 0, "generation_wall_sec": wall})
        completed = subprocess.run(["jq", "-e", jq_filter], input=payload, text=True, capture_output=True, check=False)
        assert completed.returncode == 0


def test_aspire_uses_authorized_budget_project_and_default_routing() -> None:
    workflow = (ROOT / ".github/workflows/aspire2ap.yml").read_text(encoding="utf-8")
    submit = workflow[
        workflow.index("PBSジョブを投入") : workflow.index("PBSジョブの終了を確認")
    ]
    assert "/opt/pbs/bin/qsub -P 59010014" in submit
    assert submit.index('cd "${RUNTIME_DIR}"') < submit.index("/opt/pbs/bin/qsub ")
    assert "-q aiq2" not in submit
    assert "ncpus=112:ngpus=8:mem=1500gb:mpiprocs=1" in submit
    assert "-l place=scatter:excl " in submit
    assert '<"${RUNTIME_DIR}/scripts/check-qwen-container.pbs"' in submit
    assert "CONTAINER_IMAGE=" not in submit
    assert 'cat "${RUNTIME_DIR}/logs/pbs-accounting.txt"' in workflow
    assert 'cat "${RUNTIME_DIR}/logs/pbs.log"' in workflow


def test_scheduler_submission_uses_persistent_runtime_working_directory() -> None:
    genkai = (ROOT / ".github/workflows/genkai.yml").read_text(encoding="utf-8")
    assert "ENROOT_SYSCONF_PATH: /home/pj26000080/ku60000563/.local/etc/enroot" in genkai
    assert "ENROOT_LIBRARY_PATH: /home/pj26000080/ku60000563/.local/lib/enroot" in genkai
    submit = genkai[genkai.index("PJMジョブを投入") : genkai.index("PJMジョブの開始を待機")]
    assert submit.index('cd "${RUNTIME_DIR}"') < submit.index("pjsub ")


def test_firmus_uses_shared_beegfs_and_compute_local_enroot() -> None:
    workflow = (ROOT / ".github/workflows/firmus.yml").read_text(encoding="utf-8")
    script = (ROOT / "platforms/firmus/check-qwen-container.slurm").read_text(encoding="utf-8")
    assert "runs-on: [self-hosted, home-lab]" in workflow
    assert "shell: firmus-shell --github-shell {0}" in workflow
    assert "[[ '${{ inputs.node_count }}' =~ ^[12]$ ]]" in workflow
    assert "RUNTIME_ROOT: /scratch/home/apacsc14/qwen/runs" in workflow
    assert '"${GITHUB_WORKSPACE}/scripts/provision-qwen-check-env.sh"' in workflow
    assert "CLUSTER_NAME: firmus" in workflow
    assert "sbatch --parsable" in workflow
    assert "--gpus-per-node=1" in workflow
    assert "sacct --allocations" in workflow
    assert "#SBATCH --partition=hpcai" in script
    assert "#SBATCH --time=00:17:00" in script
    assert 'exec "${RUNTIME_DIR}/scripts/run-qwen-check-env.sh"' in script
    assert "GOOGLE_CREDENTIAL" not in script
    assert "BIGQUERY" not in script


def test_cluster_steps_use_home_lab_remote_shells() -> None:
    workflows = {
        "genkai.yml": "genkai-shell",
        "aspire2ap.yml": "aspire2p-shell",
        "firmus.yml": "firmus-shell",
    }
    for filename, remote_shell in workflows.items():
        workflow = (ROOT / ".github/workflows" / filename).read_text(encoding="utf-8")
        assert f"shell: {remote_shell} --github-shell {{0}}" in workflow
        assert "runs-on: [self-hosted, home-lab]" in workflow
    aws = (ROOT / ".github/workflows/aws-container.yml").read_text(encoding="utf-8")
    assert "runner_labels" not in aws
    assert "remote_shell" in aws


def test_validation_includes_slurm_adapters() -> None:
    workflow = (ROOT / ".github/workflows/validate.yml").read_text(encoding="utf-8")
    assert "-name '*.slurm'" in workflow


def test_credentials_stop_at_login_runner() -> None:
    compute_files = [
        ROOT / "scripts/run-qwen35-job.sh",
        ROOT / "scripts/run-qwen35-node.sh",
        ROOT / "platforms/genkai/run-qwen35.pjm",
        ROOT / "platforms/aspire2ap/run-qwen35.pbs",
    ]
    for path in compute_files:
        text = path.read_text(encoding="utf-8")
        assert "GOOGLE_CREDENTIAL" not in text
        assert "BIGQUERY" not in text
        assert "smoke-send" not in text
    for name in ("genkai.yml", "aspire2ap.yml", "firmus.yml"):
        workflow = (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
        assert "configure-aws-credentials" not in workflow
        assert "send-qwen-fragments.sh" not in workflow
        assert "AWS_ACCESS_KEY_ID" not in workflow
        assert "AWS_SECRET_ACCESS_KEY" not in workflow
    assert "configure-aws-credentials" in (ROOT / ".github/workflows/aws-container.yml").read_text(encoding="utf-8")


def test_sglang_patches_close_primary_config_worker_and_profile_sources() -> None:
    patch = (
        ROOT / "patches/sglang/0002-close-qwen-v2-observation-sources.patch"
    ).read_text(encoding="utf-8")
    ordering = (
        ROOT / "patches/sglang/0003-record-primary-before-profiler-finalization.patch"
    ).read_text(encoding="utf-8")
    observed_counts = (
        ROOT / "patches/sglang/0004-observe-primary-token-counts-from-responses.patch"
    ).read_text(encoding="utf-8")
    assert '+            "error_count": 0' not in patch
    assert "measured DatasetRow has no source array index" not in patch
    assert "+        if array_index is not None:" in patch
    assert "QWEN_EXPERIMENT_METRICS_DATASET_MANIFEST_SHA256" not in patch
    assert "--- a/python/sglang/srt/entrypoints/engine.py" in patch
    assert '+            "python/sglang/srt/server_args.py",' in patch
    assert 'producer_id=f"sglang-worker-{global_rank}"' in patch
    assert 'expected = f"qwen-profile-TP-{global_rank}.trace.json.gz"' in patch
    assert "capture_profiler_trace(trace_path, self.ps.tp_rank)" in patch
    assert ordering.index("observed_output_tokens,\n         )") < ordering.index(
        "+    if profile:"
    )
    assert '+            "observed_input_tokens": observed_input_tokens,' in (
        observed_counts
    )
    assert '+            meta_info.get("prompt_tokens")' in observed_counts


def test_sglang_patches_are_syntactically_valid_unified_diffs() -> None:
    patch_dir = ROOT / "patches/sglang"
    patches = sorted(patch_dir.glob("*.patch"))
    assert len(patches) >= 6
    for patch in patches:
        result = subprocess.run(
            ["git", "apply", "--numstat", str(patch)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert (
            result.returncode == 0
        ), f"Patch {patch.name} is malformed:\n{result.stderr}"
        assert result.stdout.strip(), f"Patch {patch.name} produced no numstat output"


def test_compute_core_has_no_bigquery_or_network_client_dependency() -> None:
    root = Path(__file__).parents[1]
    with (root / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)
    default_dependencies = pyproject["project"]["dependencies"]
    assert not any("google-auth" in item for item in default_dependencies)
    assert not any("requests" in item for item in default_dependencies)

    forbidden = {"google", "requests", "urllib", "httpx", "aiohttp"}
    compute_modules = {
        "__init__.py",
        "canonical.py",
        "domain.py",
        "events.py",
        "local_socket.py",
        "manifest.py",
        "recording.py",
        "registry.py",
        "stage_runner.py",
    }
    for path in (
        root / "src/qwen_experiment_metrics" / name for name in compute_modules
    ):
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in (
                node.names
                if isinstance(node, ast.Import)
                else [ast.alias(name=node.module or "")]
            )
        }
        assert imports.isdisjoint(forbidden), (
            f"forbidden dependency in {path}: {imports}"
        )


def test_active_shell_commands_have_no_line_continuations() -> None:
    paths = (
        list((ROOT / "scripts").glob("*.sh"))
        + list((ROOT / "platforms").rglob("*.pjm"))
        + list((ROOT / "platforms").rglob("*.pbs"))
    )
    for path in paths:
        assert not any(
            line.endswith("\\")
            for line in path.read_text(encoding="utf-8").splitlines()
        ), path
