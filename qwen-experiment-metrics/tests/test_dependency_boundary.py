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
