"""CPU contract checks for the actual patched SGLang staging functions."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace


def load_function(path: Path, name: str, namespace: dict):
    tree = ast.parse(path.read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)  # noqa: S102 - named function from pinned build input
    return namespace[name]


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid transfer geometry was accepted")


def check(source: Path):
    root = source / "python/sglang/srt/disaggregation"
    namespace = {"logger": SimpleNamespace(info=lambda *args: None)}
    buffers = load_function(root / "common/staging_buffer.py", "get_staging_kv_buffers", namespace)

    def tensor(name, dtype="fp8"):
        return SimpleNamespace(name=name, shape=(4096, 4, 128), stride=lambda: (512, 128, 1), dtype=dtype, device="cuda:0", is_contiguous=lambda: True)

    target = SimpleNamespace(page_size=64, k_buffer=[tensor(f"target-k-{i}") for i in range(94)], v_buffer=[tensor(f"target-v-{i}") for i in range(94)])
    draft = SimpleNamespace(page_size=64, k_buffer=[tensor("draft-k")], v_buffer=[tensor("draft-v")])
    k, v, page = buffers(target, draft)
    assert (len(k), len(v), page) == (95, 95, 64)
    assert k[-1] is draft.k_buffer[0] and v[-1] is draft.v_buffer[0]
    assert k[:94] == target.k_buffer and v[:94] == target.v_buffer
    draft.k_buffer[0].dtype = "bf16"
    rejected(lambda: buffers(target, draft))
    draft.k_buffer[0].dtype = "fp8"
    draft.page_size = 1
    rejected(lambda: buffers(target, draft))

    conn = root / "nixl/conn.py"
    load_function(conn, "_homogeneous_kv_mem_kind", namespace)
    prepare = load_function(conn, "_prepare_payload_xfer", namespace)
    set_buffers = load_function(conn, "set_kv_buffer_tensors", namespace)
    calls = []
    manager = SimpleNamespace(kv_args=SimpleNamespace(kv_item_lens=[32768] * 190, kv_data_ptrs=list(range(190))), src_mem_kind="VRAM", is_mla_backend=False, is_hybrid_mla_backend=False, attn_tp_size=1, enable_staging=True, kv_buffer_tensors=None, _init_hetero_tp_prep_handle=lambda *args, **kwargs: calls.append(args))
    rejected(lambda: set_buffers(manager, k[:94], v[:94], 64))
    set_buffers(manager, k, v, 64)
    peer = SimpleNamespace(dst_kv_item_lens=[8192] * 190, decode_tp_size=4, dst_kv_mem_kinds=["VRAM"] * 190, agent_name="decode-rank0", staging=object())
    prepare(manager, peer)
    assert not calls, "staged transfer constructed a full-pool descriptor list"
    peer.staging = None
    rejected(lambda: prepare(manager, peer))
    peer.staging = object()
    peer.dst_kv_item_lens.pop()
    rejected(lambda: prepare(manager, peer))
    peer.dst_kv_item_lens.append(8192)
    manager.enable_staging = False
    prepare(manager, peer)
    assert len(calls) == 1, "legacy non-staged path was changed"
    print("PASS: target+draft KV ordering/geometry; negotiated staging skips full-pool descriptors; invalid peers rejected; legacy direct path retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    check(parser.parse_args().source)
