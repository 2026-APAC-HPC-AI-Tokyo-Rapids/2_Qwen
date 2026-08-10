from __future__ import annotations

import ast
import tomllib
from pathlib import Path


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
