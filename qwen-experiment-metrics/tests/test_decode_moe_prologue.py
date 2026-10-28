"""CPU checks for the opt-in Decode prologue; GPU numerics live in the gate."""
import ast
import copy
import os
import random
import re
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]
SELECTOR = runpy.run_path(str(ROOT / "scripts/select-firmus-decode-prologue.py"))


def added_file(name):
    patch = (ROOT / "patches/sglang/decode-router-cluster.patch").read_text()
    section = patch.split("+++ b/python/sglang/" + name + "\n", 1)[1].split("--- ", 1)[0]
    return "\n".join(line[1:] for line in section.splitlines() if line.startswith("+")) + "\n"


def records(saving=6):
    return [{"rows": rows, "block": block, "numeric_passed": True, "control": [230.0] * 15, "candidate": [230.0 - saving] * 15} for rows in SELECTOR["ROWS"] for block in range(2)]


@pytest.mark.parametrize("saving,expected", [(4.999, False), (5, True), (6, True), (0, False)])
def test_joint_threshold(saving, expected):
    assert SELECTOR["choose_bundle"](records(saving)) is expected


@pytest.mark.parametrize("failure", ["missing", "duplicate", "numeric", "nan", "inf", "empty", "zero", "regression"])
def test_incomplete_or_failed_candidates_keep_control(failure):
    data = records()
    if failure == "missing":
        data.pop()
    elif failure == "duplicate":
        data[-1] = copy.deepcopy(data[0])
    elif failure == "numeric":
        data[-1]["numeric_passed"] = False
    elif failure == "empty":
        data[-1]["candidate"] = []
    else:
        value = {"nan": float("nan"), "inf": float("inf"), "zero": 0, "regression": 231}[failure]
        data[-1]["candidate"] = [value] * 15
    assert not SELECTOR["choose_bundle"](data)


def test_capture_mode_restores_inherited_values_on_failure(monkeypatch):
    monkeypatch.setenv("DG_SM90_DECODE_PREPARE", "old")
    monkeypatch.delenv("SGLANG_DECODE_ROUTER", raising=False)
    with pytest.raises(RuntimeError), SELECTOR["bundle_mode"](True):
        assert os.environ["DG_SM90_DECODE_PREPARE"] == "1"
        assert os.environ["SGLANG_DECODE_ROUTER"] == "cluster"
        with SELECTOR["bundle_mode"](False):
            assert os.environ["DG_SM90_DECODE_PREPARE"] == "0"
            assert os.environ["SGLANG_DECODE_ROUTER"] == ""
        raise RuntimeError("numeric gate rejected candidate")
    assert os.environ["DG_SM90_DECODE_PREPARE"] == "old"
    assert "SGLANG_DECODE_ROUTER" not in os.environ


@pytest.mark.parametrize("case,status,selected", [("router-gate", 1, False), ("prepare-gate", 1, False), ("timeout", 124, False), ("not-faster", 0, False), ("selected", 0, True), ("stale-failure", 1, False)])
def test_shell_rejection_preserves_control_and_continues(tmp_path, case, status, selected):
    shell = (ROOT / "scripts/select-firmus-decode-kernels.sh").read_text()
    start = '  DG_SM90_DECODE_PREPARE=1 SGLANG_DECODE_ROUTER=cluster timeout '
    block = start.lstrip() + shell.split(start, 1)[1].split('\nelif ', 1)[0]
    control = "SGLANG_DECODE_DENSE=control\n"
    candidate = "DG_SM90_DECODE_PREPARE=1\nSGLANG_DECODE_ROUTER=cluster\n"
    (tmp_path / "dense.env").write_text(control)
    (tmp_path / "prologue-selected.env").write_text(candidate if selected or case == "stale-failure" else "")
    # Mock only the GPU command; execute the production exit-status/env handling.
    script = f'set -uo pipefail\nroot=$1\nout=$1\ntorchrun=unused-gpu-command\ntimeout() {{ return {status}; }}\n'
    result = subprocess.run(["bash", "-c", script + block + '\nprintf serving-continues\n', "selector-test", str(tmp_path)], capture_output=True, text=True, check=True)
    assert result.stdout == "serving-continues"
    assert (tmp_path / "dense.env").read_text() == control + (candidate if selected else "")
    assert (tmp_path / "prologue-status.txt").read_text() == f"prologue={status}\n"


@pytest.mark.parametrize("rows", [0, 1, 128, 143, 144, 145, 152, 160, 168, 175, 176, 177, 192, 193, 4096])
@pytest.mark.parametrize("mode", ["decode", "verify", "prefill"])
def test_router_selection_shape_and_mode(monkeypatch, rows, mode):
    tree = ast.parse(added_file("kernels/ops/moe/decode_router.py"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "can_use_decode_router")
    namespace = {"os": os, "torch": NS(bfloat16="bf16")}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "router-selection", "exec"), namespace)  # noqa: S102 - execute the tracked host guard
    x = NS(ndim=2, shape=(rows, 4096), dtype="bf16", is_cuda=True, device="cuda:0", is_contiguous=lambda: True)
    w = NS(shape=(128, 4096), dtype="bf16", device="cuda:0", is_contiguous=lambda: True)
    forward_mode = NS(is_decode=lambda: mode == "decode", is_target_verify=lambda: mode == "verify")
    guard = namespace["can_use_decode_router"]
    monkeypatch.delenv("SGLANG_DECODE_ROUTER", raising=False)
    assert not guard(x, w, forward_mode)
    monkeypatch.setenv("SGLANG_DECODE_ROUTER", "cluster")
    assert guard(x, w, forward_mode) == (144 <= rows <= 176 and mode != "prefill")
    w.dtype = "fp16"
    assert not guard(x, w, forward_mode)
    w.dtype, w.device = "bf16", "cuda:1"
    assert not guard(x, w, forward_mode)
    w.device, w.is_contiguous = "cuda:0", lambda: False
    assert not guard(x, w, forward_mode)


@pytest.mark.parametrize("rows", [0, 1, 7, 31, 32, 33, 128, 144, 145, 152, 160, 168, 175, 176, 192])
@pytest.mark.parametrize("pattern", ["padded", "hot", "balanced", "random"])
def test_warp_prefix_and_publication(rows, pattern):
    patch = (ROOT / "patches/deep-gemm-overlay/sm90-decode-prepare.patch").read_text()
    assert "const uint32_t offset = count + __popc(mask & ((1u << lane_idx) - 1u));" in patch
    assert "(static_cast<uint64_t>(kNumSMs) << 32) | count" in patch
    rng = random.Random(20261028 + rows)
    routes = [{"padded": -1, "hot": 3, "balanced": i % 128, "random": rng.randrange(-1, 128)}[pattern] for i in range(rows * 8)]
    owners = {(sm * 2 + warp): (sm, warp) for sm in range(132) for warp in range(2) if sm * 2 + warp < 128}
    assert set(owners) == set(range(128))
    for expert in owners:
        count = 0
        entries = {}
        for base in range(0, len(routes), 32):
            mask = sum(1 << lane for lane, route in enumerate(routes[base:base + 32]) if route == expert)
            for lane, route in enumerate(routes[base:base + 32]):
                if route == expert:
                    offset = count + (mask & ((1 << lane) - 1)).bit_count()
                    assert offset not in entries
                    entries[offset] = base + lane
            count += mask.bit_count()
        expected = [i for i, route in enumerate(routes) if route == expert]
        assert [entries[i] for i in range(count)] == expected
        status = (132 << 32) | count
        assert status >> 32 == 132 and status & 0xffffffff == len(expected)
        # Four source ranks publish the unchanged scheduler readiness high word.
        assert sum([status] * 4) >> 32 == 132 * 4


@pytest.mark.parametrize("rows", [144, 145, 152, 160, 168, 175, 176])
def test_cluster_tile_and_split_ownership(rows):
    source = added_file("kernels/jit/csrc/gemm/decode_router_kernel.cuh")
    assert "__shared__ __align__(32) float partial[32 * 32];" in source
    assert source.count("cluster.sync();") == 2
    assert "value += cluster.map_shared_rank(partial, peer)[i];" in source
    assert "wmma::accumulator, 16, 16, 16, float" in source
    assert [k for rank in range(4) for k in range(rank * 1024, (rank + 1) * 1024)] == list(range(4096))
    seen = set()
    for tile in range(((rows + 31) // 32) * 4):
        for i in range(32 * 32):
            row, expert = (tile // 4) * 32 + i // 32, (tile % 4) * 32 + i % 32
            if row < rows:
                assert (row, expert) not in seen
                seen.add((row, expert))
    assert len(seen) == rows * 128


def test_prepare_host_guard_and_old_instantiation():
    patch = (ROOT / "patches/deep-gemm-host/sm90-decode-prepare.patch").read_text()
    expression = re.search(r"\+        \.decode_prepare = (.*),", patch)[1]
    expression = expression.replace('get_env<int>("DG_SM90_DECODE_PREPARE", 0)', "enabled")
    for enabled in (0, 1):
        for use_swap_ab in (False, True):
            for rows in (1, 144, 176, 192, 193, 256, 4096):
                for experts in (128, 140):
                    ns = {"use_swap_ab": use_swap_ab, "hidden": 4096, "intermediate_hidden": 1536, "num_experts": experts, "num_topk": 8, "num_ranks": 4, "num_tokens": rows, "enabled": enabled}
                    assert bool(eval(expression, {}, ns)) == (enabled == 1 and use_swap_ab and rows <= 192 and experts == 128)
    assert 'args.use_swap_ab ? (args.decode_prepare ? "true, true" : "true") : "false"' in patch
