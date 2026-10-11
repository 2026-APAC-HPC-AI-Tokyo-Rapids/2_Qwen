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
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), namespace)
    return namespace[name]


def check_staging(source):
    path = source / "python/sglang/srt/disaggregation/common/staging_handler.py"
    calls = []
    ready = False
    event = SimpleNamespace(record=lambda _: calls.append("record"), query=lambda: ready)
    ns = {"torch": SimpleNamespace(cuda=SimpleNamespace(Event=lambda: event)), "logger": SimpleNamespace(warning=lambda *_: calls.append("warning"))}
    methods = {name: method(path, name, ns) for name in ("submit_last_scatter_async", "advance_scatter", "is_done", "release_room")}

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    source = parser.parse_args().source
    check_staging(source)
    check_chunk_release(source)
