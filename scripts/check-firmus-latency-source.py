"""CPU regressions against the source installed in the Firmus image."""
from __future__ import annotations

import argparse
import ast
import asyncio
import threading
from concurrent.futures import Future
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace


def method(source, name, namespace):
    node = next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
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


def check_dp_routing(source):
    ns = {"Enum": Enum, "auto": auto}
    controller = source / "python/sglang/srt/managers/data_parallel_controller.py"
    classes = [node for node in ast.parse(controller.read_text()).body if isinstance(node, ast.ClassDef) and node.name in {"LoadBalanceMethod", "DPBudget"}]
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(controller), "exec"), ns)  # noqa: S102
    budget = ns["DPBudget"](4)
    loads = [SimpleNamespace(dp_rank=rank, timestamp=1.0, num_running_reqs=0, num_waiting_reqs=waiting, num_total_tokens=waiting * 4000) for rank, waiting in enumerate([0, 10, 0, 0])]
    budget.update_budget(loads)
    calls = []
    ns["sock_send"] = lambda worker, req: calls.append(worker)
    dispatch = method(controller, "total_requests_scheduler", ns)
    target = SimpleNamespace(dp_budget=budget, workers=list(range(4)), maybe_external_dp_rank_routing=lambda _: False)
    for _ in range(12):
        dispatch(target, None)
        budget.update_budget(loads)  # Re-reading a stale snapshot must not erase assignments.
    assert calls == [0, 2, 3] * 4 and budget.total_requests == [4, 10, 4, 4]
    # A completed request changes the next assignment.
    loads[1].timestamp, loads[1].num_waiting_reqs = 2.0, 0
    budget.update_budget(loads)
    dispatch(target, None)
    assert calls[-1] == 1

    # Exercise the real snapshot producer too: count bootstrap on P and
    # prealloc, transfer and retracted requests on D, not only running batches.
    inquirer_path = source / "python/sglang/srt/managers/scheduler_components/load_inquirer.py"
    modes = SimpleNamespace(PREFILL="prefill", DECODE="decode")
    load_ns = {name: SimpleNamespace for name in ("LoadSnapshot", "MemoryMetrics", "SpeculativeMetrics", "LoRAMetrics", "DisaggregationMetrics", "QueueMetrics")}
    load_ns.update(DisaggregationMode=modes, time=SimpleNamespace(time=lambda: 3.0), logger=SimpleNamespace(debug=lambda *_: None))
    get_loads = method(inquirer_path, "get_loads", load_ns)
    stats = SimpleNamespace(**dict.fromkeys(("kv_transfer_speed_gb_s", "kv_transfer_latency_ms", "num_grammar_queue_reqs", "num_paused_reqs", "num_retracted_reqs", "gen_throughput", "cache_hit_rate", "utilization"), 0))
    item = SimpleNamespace(seqlen=4000)
    inquirer = SimpleNamespace(get_stats=lambda: stats, get_running_batch=lambda: SimpleNamespace(reqs=[item] * 32), get_waiting_queue=lambda: [item] * 2, get_disagg_prefill_bootstrap_queue=lambda: SimpleNamespace(queue=[item] * 3), get_disagg_prefill_inflight_queue=list, get_disagg_decode_prealloc_queue=lambda: SimpleNamespace(queue=[item] * 5, retracted_queue=[item] * 6), get_disagg_decode_transfer_queue=lambda: SimpleNamespace(queue=[item] * 4), pool_stats_observer=SimpleNamespace(get_pool_stats=lambda: SimpleNamespace(get_kv_token_stats=lambda: (128000, .5))), spec_algorithm=SimpleNamespace(is_none=lambda: True), server_args=SimpleNamespace(enable_lora=False), ps=SimpleNamespace(dp_rank=0), max_total_num_tokens=393216, max_running_requests=32, get_decode_moment_totals=lambda: [0] * 6, get_num_waiting_uncached_tokens=lambda: 0, get_total_prefill_uncached_tokens=lambda: 0, get_total_prefill_busy_us=lambda: 0)
    for mode, waiting in ((modes.PREFILL, 5), (modes.DECODE, 17)):
        inquirer.disaggregation_mode = mode
        load = get_loads(inquirer)
        assert (load.num_running_reqs, load.num_waiting_reqs) == (32, waiting)

    publish = method(source / "python/sglang/srt/managers/scheduler.py", "publish_load_snapshot", {})
    snapshots = []
    scheduler = SimpleNamespace(load_snapshot_writer=SimpleNamespace(publish_interval=1, publish_counter=0, write=snapshots.append), load_inquirer=SimpleNamespace(get_loads=lambda: load))
    publish(scheduler)
    publish(scheduler)
    assert snapshots == [load, load], "routing load must update every iteration"

    conn = source / "python/sglang/srt/disaggregation/common/conn.py"
    rooms = {}
    server = SimpleNamespace(room_to_dp_rank=rooms, lock=asyncio.Lock())
    io = {"logger": SimpleNamespace(debug=lambda *_: None, error=lambda *args: (_ for _ in ()).throw(AssertionError(args))), "time": SimpleNamespace(time=lambda: 1.0), "web": SimpleNamespace(Response=lambda **kw: SimpleNamespace(status_code=kw["status"]), json_response=lambda data, **kw: SimpleNamespace(status_code=kw["status"], json=lambda: data))}
    register = method(conn, "_handle_register_dp_rank", io)
    query = method(conn, "_handle_query_dp_ranks", io)

    def post(url, *, json, timeout):
        assert timeout == 5

        async def payload():
            return json

        handler = register if url.endswith("/register_dp_rank") else query
        return asyncio.run(handler(server, SimpleNamespace(json=payload)))

    io.update(requests=SimpleNamespace(post=post), _get_bootstrap_session=lambda _: SimpleNamespace(post=post))
    sender_register = method(conn, "_register_prefill_dp_rank", io)
    receiver_query = method(conn, "query_prefill_dp_ranks", io)
    decode = source / "python/sglang/srt/disaggregation/decode.py"
    submitted = []

    class NonblockingFuture(Future):
        def result(self, *args, **kwargs):
            assert self.done(), "scheduler blocked waiting for HTTP"
            return super().result(*args, **kwargs)

    def submit(function, *args):
        future = NonblockingFuture()
        submitted.append((function, args, future))
        return future

    def complete(index):
        function, args, future = submitted[index]
        future.set_result(function(*args))

    pending = SimpleNamespace(pending_reqs=[], _prefill_dp_rank_queries={}, _prefill_dp_rank_executor=SimpleNamespace(submit=submit), kv_manager=SimpleNamespace(prefill_info_table={"peer:9000": SimpleNamespace(dp_size=4, follow_bootstrap_room=False)}))
    decode_ns = {"_bootstrap_addr": lambda _: "peer:9000", "CommonKVReceiver": SimpleNamespace(query_prefill_dp_ranks=receiver_query), "FINISH_ABORT": type("Aborted", (), {}), "envs": SimpleNamespace(SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK=SimpleNamespace(get=lambda: False))}
    resolve_rank = method(decode, "_resolve_prefill_dp_rank", decode_ns)
    query_ranks = method(decode, "_query_prefill_dp_ranks", decode_ns)
    resolve_pending = method(decode, "_resolve_pending_reqs", decode_ns)
    pending._resolve_prefill_dp_rank = lambda req: resolve_rank(pending, req)
    pending._query_prefill_dp_ranks = lambda addr, rooms: query_ranks(pending, addr, rooms)
    pending._ensure_prefill_info = lambda grouped: (grouped, [])
    initialized = []
    for room in (1, 5):  # Both hash to rank 1, but queue-aware routing chooses 0 and 3.
        req = SimpleNamespace(bootstrap_room=room, disagg_prefill_dp_rank=None, finished_reason=None)
        assert resolve_rank(pending, req) is None
        pending.pending_reqs.append(SimpleNamespace(req=req, kv_receiver=SimpleNamespace(init=lambda rank, room=room: initialized.append((room, rank)))))
    resolve_pending(pending)
    assert len(pending.pending_reqs) == 2 and initialized == [], "unknown room must not guess a source rank"
    for _ in range(3):
        resolve_pending(pending)
    assert len(submitted) == 1, "polling must not queue duplicate HTTP calls"
    complete(0)
    resolve_pending(pending)
    assert len(pending.pending_reqs) == 2 and initialized == []
    for room, rank in ((1, 0), (5, 3)):
        sender_register(SimpleNamespace(bootstrap_server_url="peer:9000", bootstrap_room=room, kv_mgr=SimpleNamespace(attn_dp_rank=rank)))
    resolve_pending(pending)
    # A request arriving after submission was not in that HTTP batch. It must
    # stay pending, without borrowing another request's resolved source rank.
    late = SimpleNamespace(bootstrap_room=9, disagg_prefill_dp_rank=None, finished_reason=None)
    pending.pending_reqs.append(SimpleNamespace(req=late, kv_receiver=SimpleNamespace(init=lambda _: initialized.append("aborted"))))
    sender_register(SimpleNamespace(bootstrap_server_url="peer:9000", bootstrap_room=9, kv_mgr=SimpleNamespace(attn_dp_rank=2)))
    complete(1)
    resolve_pending(pending)
    assert [item.req.bootstrap_room for item in pending.pending_reqs] == [9] and initialized == [(1, 0), (5, 3)]
    resolve_pending(pending)
    late.finished_reason = decode_ns["FINISH_ABORT"]()  # Receiver has not yet been cleared.
    resolve_pending(pending)
    assert len(pending._prefill_dp_rank_queries) == 1
    complete(2)
    resolve_pending(pending)
    assert not pending.pending_reqs and not pending._prefill_dp_rank_queries and initialized == [(1, 0), (5, 3)]
    # Pin the sender constructor's registration guard and bootstrap advertisement
    # as well as the executed register/query path above.
    tree = ast.parse(conn.read_text())
    sender = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CommonKVSender")
    registration = next(node.test for node in ast.walk(sender) if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and "load_balance_method" in ast.unparse(node.test))
    assert eval(compile(ast.Expression(registration), str(conn), "eval"), {"self": SimpleNamespace(kv_mgr=SimpleNamespace(server_args=SimpleNamespace(load_balance_method="total_requests")))})
    advertisement = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute) and target.attr == "follow_bootstrap_room" for target in node.targets) and isinstance(node.value, ast.Compare))
    assert eval(compile(ast.Expression(advertisement), str(conn), "eval"), {"load_balance_method": "total_requests"}) is False
    print("PASS: DP load includes PD queues, refreshes every iteration and preserves stale-snapshot increments; KV lookup never waits on unfinished HTTP or duplicates calls; late arrivals and aborted requests retain correct source ranks (HTTP mocked)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    source = parser.parse_args().source
    check_staging(source)
    check_chunk_release(source)
    check_moe_fusion_gate(source)
    check_dp_routing(source)
