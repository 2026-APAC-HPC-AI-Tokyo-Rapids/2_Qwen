"""Exercise the actual cache transfer functions without a GPU or container."""

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "platforms/shared/qwen-pd.sbatch"
FUNCTIONS = SCRIPT.read_text().split("restore_worker_cache() {", 1)[1].split("run_node() {", 1)[0]
FUNCTIONS = "restore_worker_cache() {" + FUNCTIONS


def run_cache(command: str, *args: Path, env=None):
    return subprocess.run(
        ["bash", "-euc", FUNCTIONS + command, "cache-test", *map(str, args)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_cache_roundtrip_keeps_nested_files_and_updates(tmp_path):
    local = tmp_path / "local"
    archive = tmp_path / "shared" / "worker.tar"
    restored = tmp_path / "restored"
    assert run_cache('restore_worker_cache "$1" "$2"', archive, local).returncode == 0
    (local / "triton").mkdir()
    (local / "triton" / "kernel").write_bytes(b"first kernel")
    assert run_cache('save_worker_cache "$1" "$2"', local, archive).returncode == 0
    assert run_cache('restore_worker_cache "$1" "$2"', archive, restored).returncode == 0
    assert (restored / "triton" / "kernel").read_bytes() == b"first kernel"
    (local / "triton" / "kernel").write_bytes(b"replacement kernel")
    assert run_cache('save_worker_cache "$1" "$2"', local, archive).returncode == 0
    assert run_cache('restore_worker_cache "$1" "$2"', archive, restored).returncode == 0
    assert (restored / "triton" / "kernel").read_bytes() == b"replacement kernel"


def test_failed_collection_keeps_previous_archive(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    archive = tmp_path / "worker.tar"
    assert run_cache('save_worker_cache "$1" "$2"', local, archive).returncode == 0
    original = archive.read_bytes()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    tar = fake_bin / "tar"
    tar.write_text('#!/bin/bash\nprintf partial > "$2"\nexit 1\n')
    tar.chmod(0o755)
    result = run_cache(
        'save_worker_cache "$1" "$2"', local, archive,
        env={**os.environ, "PATH": str(fake_bin) + ":" + os.environ["PATH"]},
    )
    assert result.returncode == 1
    assert archive.read_bytes() == original
    assert list(tmp_path.glob("worker.tar.*")) == []


def test_failed_job_cleanup_still_collects_and_preserves_exit_status(tmp_path):
    script = SCRIPT.read_text()
    cleanup = script.split("  cleanup_node() {", 1)[1].split("  trap cleanup_node EXIT", 1)[0]
    node = tmp_path / "node"
    cache = node / "prefill-0" / "home" / ".cache"
    cache.mkdir(parents=True)
    (cache / "kernel").write_text("compiled before benchmark failure")
    shared = tmp_path / "shared"
    command = (
        'node_tmp="$1"; cache_root="$2"; pids=(); cache_workers=(prefill-0); '
        'sampler_pid=; system_sampler_pid=; cleanup_node() {' + cleanup
        + '\ntrap cleanup_node EXIT\nexit 7\n'
    )
    result = run_cache(command, node, shared)
    assert result.returncode == 7
    assert not node.exists()
    restored = tmp_path / "restored"
    assert run_cache('restore_worker_cache "$1" "$2"', shared / "prefill-0.tar", restored).returncode == 0
    assert (restored / "kernel").read_text() == "compiled before benchmark failure"
