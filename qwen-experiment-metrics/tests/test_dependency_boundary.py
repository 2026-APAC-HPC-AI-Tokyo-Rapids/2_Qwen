from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_sif_build_uses_uv_python_for_oci_resolver() -> None:
    script = (ROOT / "scripts/build-sglang-sif.sh").read_text(encoding="utf-8")
    resolver_line = next(
        line
        for line in script.splitlines()
        if '"${GITHUB_WORKSPACE}/scripts/resolve-oci.py"' in line
    )
    assert resolver_line.lstrip().startswith("python ")
    assert 'uv run --project "${GITHUB_WORKSPACE}/qwen-experiment-metrics"' in script
    assert '\npython3 "${GITHUB_WORKSPACE}/scripts/resolve-oci.py"' not in script


def test_sif_build_consumes_the_tracked_sglang_lock() -> None:
    script = (ROOT / "scripts/build-sglang-sif.sh").read_text(encoding="utf-8")
    recipe = (ROOT / "containers/sglang.def").read_text(encoding="utf-8")

    assert '${SGLANG_SOURCE_DIR}/python/uv.lock' in script
    assert 'lock_manifest:{path:"python/uv.lock"' in script
    assert "UV_PROJECT_ENVIRONMENT=/opt/sglang/venv" in recipe
    assert (
        "uv sync --project /opt/sglang/source/python --locked "
        "--no-install-project"
    ) in recipe
    assert "uv pip install --python /opt/sglang/venv/bin/python --no-deps -e ." in recipe


def test_sif_recipe_tests_the_shared_v2_event_emitter() -> None:
    recipe = (ROOT / "containers/sglang.def").read_text(encoding="utf-8")

    assert "from sglang.srt.qwen_experiment_metrics import send_event" in recipe
    assert "from sglang.benchmark.offline_throughput import _send_metrics_event" not in recipe


def test_genkai_workflow_captures_closed_accounting_source() -> None:
    workflow = (ROOT / ".github/workflows/genkai.yml").read_text(encoding="utf-8")
    assert "Checkout runner publication files" in workflow
    assert "pjstat --history -s" in workflow
    assert (
        "--choose jid,st,sdt,elp,ec,sn,pc,nnumu,mmszu,uctmut,sctmut"
        in workflow
    )
    assert "--data --delimiter $'\\t'" in workflow
    assert 'source/scheduler/accounting.stderr"' in workflow
    assert '"${metrics_cli}" record-accounting' in workflow


def test_node_launcher_connects_closed_sources_and_fixed_profile_overlay() -> None:
    script = (ROOT / "scripts/run-qwen35-node.sh").read_text(encoding="utf-8")
    for command in (
        "record-environment",
        "capture-hardware",
        "capture-topology",
        "qwen_experiment_metrics.node_agent",
        "record-workload",
    ):
        assert command in script
    assert "singularity_namespace_args=()" in script
    assert "singularity_namespace_args=(--userns)" in script
    assert '"${singularity_namespace_args[@]}" "${singularity_runtime_args[@]}"' in script
    assert script.index("singularity_namespace_args=()") < script.index(
        'exec "${singularity_runtime_args[@]}" sglang.sif true'
    ) < script.index("singularity_namespace_args=(--userns)")
    assert "SGLANG_PROFILE_ID=qwen-profile" in script
    assert "qwen-profile-wrapper.py:/qwen-profile-wrapper.py:ro" in script
    assert "QWEN_EXPERIMENT_METRICS_SERVER_ARGS_BLOB_OID" in script


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
        "merge.py",
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
