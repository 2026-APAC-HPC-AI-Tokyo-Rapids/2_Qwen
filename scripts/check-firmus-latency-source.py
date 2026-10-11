"""CPU regressions against the source installed in the Firmus image."""
from __future__ import annotations

import argparse
import ast
import threading
from pathlib import Path
from types import SimpleNamespace


def method(source, name, namespace):
    node = next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n, ast.FunctionDef) and n.name == name)
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    # Exercise the actual function from the pinned, locally patched source.
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), namespace)  # noqa: S102
    return namespace[name]


def check_staging(source):
    path = source / "python/sglang/srt/disaggregation/common/staging_handler.py"
    calls = []
    ready = False
    event = SimpleNamespace(record=lambda _: calls.append("record"), query=lambda: ready)
    ns = {"torch": SimpleNamespace(cuda=SimpleNamespace(Event=lambda: event)), "logger": SimpleNamespace(warning=lambda *_: calls.append("warning"))}
    methods = {name: method(path, name, ns) for name in ("submit_last_scatter_async", "advance_scatter", "is_done", "release_room", "unregister_decode_req")}

    def room():
        receiver = SimpleNamespace(chunk_staging_infos=[(7, 0, 0, 0, 63)])
        req = SimpleNamespace(req=SimpleNamespace(bootstrap_room=1), kv_receiver=receiver, _chunk_events=[], _staging_scatter_done=False)
        handler = SimpleNamespace(_room_to_decode_req={1: req}, _submit_last_scatter=lambda _: 7, _free_and_send_watermark=lambda i, _: calls.append(("free", i)), staging_allocator=SimpleNamespace(_scatter_stream=SimpleNamespace(synchronize=lambda: calls.append("sync"))))
        handler._scatter_lock = threading.Lock()
        handler.is_done = lambda r: methods["is_done"](handler, r)
        return handler, req, receiver

    h, r, rx = room()
    methods["submit_last_scatter_async"](h, 1)
    assert rx.chunk_staging_infos[-1][0] == -1, "last allocation retained two owners"
    methods["advance_scatter"](h, r)
    assert not h.is_done(r) and ("free", 7) not in calls
    ready = True
    methods["advance_scatter"](h, r)
    methods["release_room"](h, 1, r, rx)
    methods["release_room"](h, 1, r, rx)
    assert calls == ["record", ("free", 7)], calls

    calls.clear()
    h, r, rx = room()
    methods["submit_last_scatter_async"](h, 1)
    methods["release_room"](h, 1, r, rx)
    assert calls == ["record", "sync", ("free", 7)], "abort must drain and free an outstanding final scatter exactly once"
    assert r._scatter_event is None and r._scatter_alloc_id == -1

    calls.clear()
    h, r, rx = room()
    r._chunk_events = [(event, 6)]
    methods["release_room"](h, 1, r, rx)
    assert calls == ["sync", "warning", ("free", 7), ("free", 6)], calls
    # Force an abort while the background thread is between submitting its
    # scatter and publishing its event. Cleanup must wait for publication.
    calls.clear()
    h, r, rx = room()
    submitted, allow_publish, abort_entered = (threading.Event() for _ in range(3))
    raw_lock = threading.Lock()

    class ObservedLock:
        def __enter__(self):
            if threading.current_thread().name == "abort":
                abort_entered.set()
            raw_lock.acquire()

        def __exit__(self, *_):
            raw_lock.release()

    h._scatter_lock = ObservedLock()
    h._room_to_receiver = {1: rx}
    h.kv_manager = SimpleNamespace(_staging_ctx=SimpleNamespace(room_receivers={1: rx}, room_bootstrap={1: "peer"}))
    h.release_room = lambda *args: methods["release_room"](h, *args)
    h._submit_last_scatter = lambda _: (submitted.set(), allow_publish.wait(2), 7)[-1]
    publisher = threading.Thread(target=lambda: methods["submit_last_scatter_async"](h, 1), daemon=True)
    abort = threading.Thread(name="abort", target=lambda: methods["unregister_decode_req"](h, 1), daemon=True)
    publisher.start()
    try:
        assert submitted.wait(2)
        abort.start()
        assert abort_entered.wait(2)
        assert 1 in h._room_to_decode_req and calls == []
    finally:
        allow_publish.set()
        publisher.join(2)
        if abort.ident is not None:
            abort.join(2)
    assert not publisher.is_alive() and not abort.is_alive()
    assert calls == ["record", "sync", ("free", 7)] and not h._room_to_decode_req, calls
    print("PASS: staging completes without duplicate frees/watermarks or unrelated stream waits; abort drains pending work")


def check_chunk_release(source):
    path = source / "python/sglang/srt/mem_cache/chunk_cache.py"
    release = method(path, "cache_finished_req", {})
    calls = []
    # Preserve the real request-row slice, including a protected partial page.
    class Rows:
        def __getitem__(self, key):
            row, span = key
            assert row == 3
            return list(range(span.start, span.stop))

    allocator = SimpleNamespace(free_segment=lambda indices, *, start_pos: calls.append((indices, start_pos)))
    cache = SimpleNamespace(req_to_token_pool=SimpleNamespace(req_to_token=Rows()), token_to_kv_pool_allocator=allocator)
    for start, end in ((0, 0), (0, 4224), (64, 4187), (67, 4187)):
        release(cache, SimpleNamespace(req_pool_idx=3, cache_protected_len=start), kv_len_to_handle=end)
        assert calls[-1] == (list(range(start, end)), start)
    print("PASS: ChunkCache uses fixed-shape page release with the correct protected-prefix offset")


def check_moe_fusion_gate(source):
    path = source / "python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py"
    tree = ast.parse(path.read_text())
    gate = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "fuse_decode_silu_quant" for t in n.targets))
    expression = compile(ast.Expression(gate), str(path), "eval")
    base = {"_is_cuda": True, "is_sm90_supported": lambda: True, "torch": SimpleNamespace(bfloat16="bf16"), "hidden_states": SimpleNamespace(dtype="bf16"), "E": 128, "N": 768, "w1": SimpleNamespace(shape=(128, 768, 4096)), "topk": 8, "use_fp8_w8a8": True, "block_shape": [128, 128], "activation": "silu", "is_gated": True, "gemm1_alpha": None, "gemm1_limit": None, "swiglu_limit": None, "filter_expert": False, "down_moe_use_tma": False, "hooks": None}
    assert eval(expression, base)
    for key, value in {"_is_cuda": False, "is_sm90_supported": lambda: False, "hidden_states": SimpleNamespace(dtype="fp16"), "E": 64, "N": 1536, "w1": SimpleNamespace(shape=(128, 768, 2048)), "topk": 4, "use_fp8_w8a8": False, "block_shape": None, "activation": "gelu", "is_gated": False, "gemm1_alpha": 1.0, "gemm1_limit": 7, "swiglu_limit": 10, "filter_expert": True, "down_moe_use_tma": True, "hooks": object()}.items():
        assert not eval(expression, {**base, key: value}), key
    print("PASS: fused FC2 boundary selects Qwen SM90 TP4 only; other precision, clamp, filter, TMA and LoRA paths retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    source = parser.parse_args().source
    check_staging(source)
    check_chunk_release(source)
    check_moe_fusion_gate(source)
