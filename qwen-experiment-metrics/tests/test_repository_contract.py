from __future__ import annotations

import os
import shlex
import subprocess
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
    assert "sglang_revision=fc275674a97d78553b22597ee72c8fd48f5420f4" in build
    assert 'checkout --detach "${sglang_revision}"' in build
    assert '[[ "${sglang_commit}" = "${sglang_revision}" ]]' in build
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
        assert f"needs.container.outputs.{cluster}_container_sha256 != ''" in entry
        assert f"needs.container.outputs.{cluster}_container_size != ''" in entry
    for tag in ("[nodes:1]", "[nodes:4]", "[require-build]", "[profile]"):
        assert tag in entry
    assert "startsWith(github.event.head_commit.message, '[require-build]')" in entry
    assert "uses: ./.github/workflows/validate.yml" in entry
    assert "QWEN_ARTIFACT_ARCHIVE_ROOT: /hdd_raid/samba/net_share/qwen-artifacts" in entry
    assert "pattern: qwen-*-${{ github.run_id }}-${{ github.run_attempt }}" in entry
    assert "id: artifacts" in entry
    assert entry.count("if: steps.artifacts.outputs.present == 'true'") == 2
    assert "needs.genkai.result != 'skipped'" in entry
    assert "fail-fast: false" in container
    assert "m7i.2xlarge" in container
    assert "actionlint-py==1.7.12.24" in validation
    assert "-name '*.slurm'" in validation


def test_container_reuse_hash_check_keeps_remote_shell_alive() -> None:
    selection = (ROOT / "scripts/select-qwen-container.sh").read_text()

    assert "verify_sha256" in selection
    assert "while sleep 15" in selection
    assert "container_sha256_check_in_progress=" in selection
    assert selection.index("hash_progress_pid=$!") < selection.index(
        'actual="$(sha256sum'
    )
    hash_index = selection.index('actual="$(sha256sum')
    assert hash_index < selection.index("stop_hash_progress", hash_index)


def test_scheduler_monitor_tolerates_temporary_connection_failures() -> None:
    monitor = (ROOT / "scripts/monitor-qwen-job.sh").read_text()

    assert '[[ "${status}" -eq 75 ]]' in monitor
    assert '[[ "${status}" -eq 124 || "${status}" -eq 255 ]]' in monitor
    assert '[[ "${connection_failure_seconds}" -lt 300 ]]' in monitor
    assert 'timeout --foreground 60 "${remote_shell}"' in monitor
    assert "sleep 15" in monitor
    assert "sleep 30" in monitor


def test_genkai_uses_bounded_concurrency_for_its_eight_gpu_allocation() -> None:
    workflow = (ROOT / ".github/workflows/genkai.yml").read_text()

    assert 'job_environment="CONCURRENCY=8,' in workflow


def test_submission_selects_the_final_export_and_uses_container_python() -> None:
    submission = (ROOT / "submission/run/submit.sbatch").read_text()

    assert '-mindepth 2 -maxdepth 2 -type f -name profile_export_aiperf.json' in submission
    assert '"${CONTAINER_IMAGE}" /opt/sglang/venv/bin/python - /results/profile_export_aiperf.json' in submission
    assert '  python3 - "${OUTPUT_DIR}/results/profile_export_aiperf.json"' not in submission


def test_cluster_branches_do_not_cancel_each_other() -> None:
    container = (ROOT / ".github/workflows/aws-container.yml").read_text()
    entry = (ROOT / ".github/workflows/qwen.yml").read_text()

    assert container.count("continue-on-error: true") >= 4
    assert "fail-fast: false" in container
    assert entry.count("needs: [validate, container]") == 3


def test_firmus_pd_contract_uses_all_sixteen_gpus() -> None:
    workflow = (ROOT / ".github/workflows/firmus.yml").read_text()
    submission = (ROOT / "submission/run/submit.sbatch").read_text()

    assert "test '${{ inputs.node_count }}' = 2" in workflow
    assert '"${RUNTIME_DIR}/scripts/submit.sbatch"' in workflow
    assert "#SBATCH --nodes=2" in submission
    assert "#SBATCH --gpus-per-node=8" in submission
    assert "#SBATCH --time=00:17:00" in submission
    assert 'worker_urls+=("http://${host}:${port}/health")' in submission
    assert 'models.pending.json' in submission
    assert 'index("qwen") != null' in submission
    assert "nixl-prewarm" not in submission
    assert submission.index('index("qwen") != null') < submission.rindex('"${script_path}" aiperf')
    firmus_block = submission.split("firmus)", 1)[1].split(";;", 1)[0]
    assert firmus_block.count("start_worker prefill") == 4
    assert firmus_block.count("start_worker decode") == 2
    assert "--tp-size \"${tp}\"" in submission
    assert '--ep-size "${ep}" --max-running-requests 256' in submission
    assert "--disaggregation-transfer-backend nixl" in submission
    assert 'grep -q ACTIVE "${state_file}"' in submission
    assert "/link_layer" not in submission
    assert 'SGLANG_HOST_IP=${node_ip}' in submission
    assert 'router_args+=(--prefill "http://${host}:${port}" "${bootstrap}")' in submission
    assert 'router_args+=(--decode "http://${host}:${port}")' in submission
    assert "--pd-disaggregation" in submission
    assert '-m "${MODEL_PATH}:/model"' in submission
    assert "HF_HUB_OFFLINE=1" in submission
    assert "sample_system()" in submission


def test_firmus_pd_bounds_each_workers_kv_capacity() -> None:
    submission = (ROOT / "submission/run/submit.sbatch").read_text()

    expected_arg = "--max-total-tokens 131072"
    assert expected_arg in submission


def test_submission_script_keeps_profile_data_separate() -> None:
    submission = (ROOT / "submission/run/submit.sbatch").read_text()
    provision = (ROOT / "scripts/provision-qwen-pd.sh").read_text()

    assert "nixl-prewarm" not in submission
    assert submission.index('index("qwen") != null') < submission.rindex('"${script_path}" aiperf')
    assert 'gpu-node-${node_rank}.csv' in submission
    for prefix in ("cpu", "network", "infiniband"):
        assert f"{prefix}-node-${{rank}}.csv" in submission
    assert '${OUTPUT_DIR}/profile' in submission
    assert 'SGLANG_HOST_IP=${node_ip}' in submission
    assert 'grep -q ACTIVE "${state_file}"' in submission
    assert "/link_layer" not in submission
    assert 'if [[ "${PROFILE}" == true ]]; then worker_command=(env "QWEN_PROFILER_INVOCATION_PATH=' in submission
    assert '"${GITHUB_WORKSPACE}/scripts/exec-profiler.sh" "${runtime_dir}/scripts/"' in provision
    assert "profiler-invocation-*.argv" in submission


def test_exec_profiler_preserves_argv_and_exit_status(tmp_path: Path) -> None:
    profiler = ROOT / "scripts/exec-profiler.sh"
    invocation = tmp_path / "profile/profiler-invocation.argv"
    command = ["/bin/sh", "-c", 'test "$1" = "argument with spaces"; exit 23', "_", "argument with spaces"]

    process = subprocess.run([profiler, *command], check=False, env=os.environ | {"QWEN_PROFILER_INVOCATION_PATH": str(invocation)})

    assert process.returncode == 23
    assert shlex.split(invocation.read_text()) == command
    assert profiler.stat().st_mode & 0o111
    assert 'exec "$@"' in profiler.read_text()


def test_submission_is_the_only_firmus_pd_runtime() -> None:
    obsolete_paths = (
        "configs/firmus-pd.json",
        "platforms/firmus/run-qwen-pd.slurm",
        "scripts/run-aiperf-container.sh",
        "scripts/run-aiperf.sh",
        "scripts/run-qwen-pd-job.sh",
        "scripts/run-qwen-pd-node.sh",
        "scripts/run-qwen-pd-router.sh",
    )
    assert all(not (ROOT / path).exists() for path in obsolete_paths)
    provision = (ROOT / "scripts/provision-qwen-pd.sh").read_text()
    assert "run-config.json" not in provision


def test_compute_scripts_have_no_cloud_credentials() -> None:
    compute_files = [ROOT / "submission/run/submit.sbatch"]
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
    firmus_block = batch.split("firmus)", 1)[1].split(";;", 1)[0]
    assert firmus_block.count("start_worker prefill") == 4
    assert firmus_block.count("start_worker decode") == 2
    assert "--pd-disaggregation" in batch
    assert "--warmup-duration 60 --benchmark-duration 120 --benchmark-grace-period 30" in batch
    assert "--synthetic-input-tokens-mean 4000" in batch
    assert "--output-tokens-mean 200" in batch
    assert "--random-seed 2026" in batch
    assert "--cache-bust first_turn_prefix" in batch
    assert "goodput * 200 if eligible else 0.0" in batch
    assert "e156cb4efae43fbee1a1ab073f946a1377e6b969" in batch
    assert "model-%05d-of-00024.safetensors" in batch
    assert '.quantization_config.quant_method == "fp8"' in batch
    for digest in (
        "ad59a339665ee1c68cbff06ac72c9e59f5183ebfaf3698c98737e003a870e9e2",
        "835fffe355c9438e7a25be099b3fccaa98350b83451f9fd2d99512e74f1ade48",
        "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3",
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ):
        assert digest in batch
    for logfile in ("server-step.log", "router.log", "aiperf.log"):
        assert f'${{OUTPUT_DIR}}/logs/{logfile}' in batch
    assert 'caller_enroot_library_path="${ENROOT_LIBRARY_PATH:-}"' in batch
    assert 'cp -L "${TOKENIZER_PATH}/${tokenizer_file}"' in batch
    assert 'shared_script_path="${OUTPUT_DIR}/submit.sbatch"' in batch
    assert 'install -m 0755 "${script_path}" "${shared_script_path}"' in batch
    assert batch.index('install -m 0755 "${script_path}" "${shared_script_path}"') < batch.index('"${script_path}" node')
