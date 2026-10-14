"""CPU regressions against the source installed in the Firmus image."""
from __future__ import annotations

import argparse
import ast
import asyncio
import builtins
import threading
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
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


def check_deepgemm_activation(source):
    path = source / "python/sglang/srt/layers/moe/moe_runner/deep_gemm.py"
    wrapper_path = source / "python/sglang/kernels/ops/attention/dsv4/moe.py"
    calls, allocations = [], []

    class Tensor:
        def __init__(self, shape, *, device="cuda", dtype="fp8"):
            self.shape, self.device, self.dtype = shape, device, dtype

        def size(self, dim):
            return self.shape[dim]

    def empty(shape, **kwargs):
        allocations.append((shape, kwargs["dtype"]))
        return Tensor(shape, **kwargs)

    def fused_module(group, ue8m0, swizzle, clamp):
        assert (group, ue8m0, clamp) == (128, False, False)

        def run(x, out, scales, transposed, limit):
            assert x.shape[0] > 0, "zero expert rows must not launch a zero-block CUDA kernel"
            assert (x.shape[1], out.shape[1], scales.shape[1]) == (3072, 1536, 12)
            assert not transposed and limit == 0
            calls.append(("activation", swizzle))

        return SimpleNamespace(run=run)

    fused = method(wrapper_path, "silu_and_mul_contig_post_quant", {"_jit_silu_mul_quant_contig_module": fused_module})
    modules = {
        "sglang.kernels.ops.attention.dsv4": SimpleNamespace(silu_and_mul_contig_post_quant=fused),
        "sglang.kernels.ops.moe.ep_moe_kernels": SimpleNamespace(tma_align_input_scale=lambda x: x),
        "sglang.kernels.ops.quantization.fp8_kernel": SimpleNamespace(create_per_token_group_quant_fp8_output_scale=lambda x_shape, device, group_size, **_: Tensor((x_shape[0], x_shape[1] // group_size), device=device, dtype="fp32")),
    }

    def import_dependency(name, *args, **kwargs):
        return modules[name] if name in modules else builtins.__import__(name, *args, **kwargs)

    class Base:
        def __init__(self, config):
            self.config = config

    backend, flags = SimpleNamespace(mega=False), SimpleNamespace(fix=True)
    ns = {"__builtins__": {**vars(builtins), "__import__": import_dependency}, "MoeRunnerCore": Base, "get_moe_a2a_backend": lambda: SimpleNamespace(is_megamoe=lambda: backend.mega), "envs": SimpleNamespace(SGLANG_OPT_FIX_MEGA_MOE_MEMORY=SimpleNamespace(get=lambda: flags.fix), SGLANG_OPT_SWIGLU_CLAMP_FUSION=SimpleNamespace(get=lambda: True), SGLANG_OPT_USE_JIT_EP_ACTIVATION=SimpleNamespace(get=lambda: True)), "torch": SimpleNamespace(empty=empty, bfloat16="bf16", float8_e4m3fn="fp8"), "deep_gemm_wrapper": SimpleNamespace(DEEPGEMM_NEED_TMA_ALIGNED_SCALES=True, DEEPGEMM_SCALE_UE8M0=False, grouped_gemm_nt_f8f8bf16_contig=lambda lhs, rhs, out, indices, **_: calls.append(("gemm", out.shape))), "dispose_tensor": lambda _: None, "use_symmetric_memory": lambda *_, **__: nullcontext(), "get_tp_group": lambda: None, "is_allocation_symmetric": lambda: False}
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "DeepGemmRunnerCore")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"__init__", "_run_contiguous_gemm"}]
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)  # noqa: S102
    config = SimpleNamespace(activation="silu", is_gated=True, swiglu_limit=None)
    for mega, fix in ((False, False), (False, True), (True, False), (True, True)):
        backend.mega, flags.fix = mega, fix
        runner = ns["DeepGemmRunnerCore"](config)
        assert runner.use_swizzle is (mega and fix), "natural DeepEP weights were interpreted as interleaved MegaMoE weights"
        if not fix:
            continue
        for rows in (128, 32768):
            calls.clear()
            allocations.clear()
            inputs = SimpleNamespace(hidden_states=Tensor((rows, 4096)), hidden_states_scale=Tensor((rows, 32)), m_indices=object())
            quant = SimpleNamespace(w13_weight=Tensor((32, 3072, 4096)), w2_weight=Tensor((32, 4096, 1536)), w13_scale=object(), w2_scale=object(), is_fp4_experts=False)
            state = {"all_tokens": rows, "hidden_states_device": "cuda", "hidden_states_dtype": "fp8", "hidden_states_shape": (rows, 4096)}
            out = runner._run_contiguous_gemm(inputs, quant, state)
            assert out.shape == (rows, 4096)
            assert calls == [("gemm", (rows, 3072)), ("activation", mega), ("gemm", (rows, 4096))], calls
            assert ((rows, 1536), "bf16") not in allocations, "fusion regressed to a materialized BF16 intermediate"
    calls.clear()
    fused(Tensor((0, 3072)), Tensor((0, 1536)), Tensor((0, 12)), 128)
    assert not calls, "empty expert rank must skip local activation"
    forward = method(source / "python/sglang/srt/models/qwen3_moe.py", "forward_deepep", {})
    block = SimpleNamespace(topk=SimpleNamespace(empty_topk_output=lambda _: "empty"), experts=lambda **kwargs: calls.append(kwargs))
    forward(block, Tensor((0, 4096)), None)
    assert len(calls) == 1 and calls[0]["topk_output"] == "empty", "an empty local DP rank must still enter distributed experts"
    print("PASS: real DeepGEMM runner preserves natural/interleaved weight layouts, uses one fused activation without a BF16 intermediate, and skips empty activation while empty DP ranks still enter experts (CUDA arithmetic mocked)")


def check_prefill_ready_handoff(source):
    path = source / "python/sglang/srt/disaggregation/prefill.py"
    loop = method(path, "event_loop_overlap_disagg_prefill", {"deque": deque, "torch": SimpleNamespace(no_grad=lambda: lambda f: f)})
    # Exercise the real loop with ready/not-ready CUDA events and independent
    # attention ranks. Transport, batches and events are CPU stand-ins.
    def scenario(ready, tp, cp, has_event):
        calls, processed = [], []
        iterations = iter(range(5))
        state = SimpleNamespace(iteration=-1)

        def receive():
            state.iteration = next(iterations)
            return []

        def prepare(**_):
            calls.append(("prepare", state.iteration))
            batch = SimpleNamespace(id=state.iteration) if state.iteration < 3 else None
            if batch is not None:
                batch.copy = lambda: SimpleNamespace(id=batch.id)
            return SimpleNamespace(running_batch=None, batch_to_run=batch)

        def query():
            calls.append(("query", state.iteration))
            return ready

        def run(batch):
            calls.append(("run", batch.id))
            return SimpleNamespace(id=batch.id, copy_done=SimpleNamespace(query=query) if has_event else None)

        def process(batch, result):
            assert batch.id == result.id and batch.id not in processed
            assert ("prepare", state.iteration) in calls, "chunk bounds/counters must be prepared before processing the previous result"
            processed.append(batch.id)
            calls.append(("result", batch.id))

        def sample(result, batch):
            if batch is not None:
                assert result.id == batch.id
                assert batch.id == 0 or batch.id - 1 in processed, "grammar/result processing must precede the next delayed sample"
                calls.append(("sample", batch.id))

        scheduler = SimpleNamespace(request_receiver=SimpleNamespace(recv_requests=receive), process_input_requests=lambda _: None, _engine_paused=False, waiting_queue=[], disagg_prefill_bootstrap_queue=SimpleNamespace(pop_bootstrapped=list), get_next_disagg_prefill_batch_to_run=prepare, running_batch=None, last_batch=None, chunked_req=None, ngram_embedding_manager=SimpleNamespace(prepare_for_forward=lambda batch, **_: batch), enable_staging=True, maybe_prefetch_staging_for_batch=lambda batch: calls.append(("staging", batch.id)), run_batch=run, _apply_war_barrier=lambda: calls.append(("war", state.iteration)), process_batch_result=process, process_disagg_prefill_inflight_queue=lambda: calls.append(("poll", state.iteration)), launch_batch_sample_if_needed=sample, on_idle=lambda: calls.append(("idle", state.iteration)), ps=SimpleNamespace(attn_tp_size=tp, attn_cp_size=cp))
        try:
            loop(scheduler)
        except StopIteration:
            pass
        assert processed == [0, 1, 2] and not scheduler.result_queue, "each result must drain exactly once, including the final idle iteration"
        early = ready and tp == cp == 1 and has_event
        for i in (1, 2):
            assert (calls.index(("result", i - 1)) < calls.index(("run", i))) is early
            assert calls.index(("run", i)) < calls.index(("war", i)) < calls.index(("poll", i)), "KV release must stay after the forward WAR barrier"
        assert ("idle", 3) not in calls and ("idle", 4) in calls
        if tp > 1 or cp > 1 or not has_event:
            assert not any(kind == "query" for kind, _ in calls), "rank-local readiness must not reorder multi-rank attention collectives"
    for case in ((True, 1, 1, True), (False, 1, 1, True), (True, 2, 1, True), (True, 1, 2, True), (True, 1, 1, False)):
        scenario(*case)
    print("PASS: ready Prefill results start handoff before the next forward; unfinished/TP/CP results retain ordering, final results drain once, chunk preparation/WAR/delayed sampling remain ordered (CUDA and transport mocked)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    source = parser.parse_args().source
    check_staging(source)
    check_chunk_release(source)
    check_moe_fusion_gate(source)
    check_dp_routing(source)
    check_deepgemm_activation(source)
    check_prefill_ready_handoff(source)
