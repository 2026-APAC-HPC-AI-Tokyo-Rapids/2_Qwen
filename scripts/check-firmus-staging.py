"""CPU contract checks for the actual patched SGLang staging functions."""
from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict, deque
from contextlib import contextmanager
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


# These checks execute the patched methods, replacing only arrays/GPU/transport
# with CPU objects. They run before the image's Python dependencies are installed.
NP = SimpleNamespace(int32=int, int64=int, empty=lambda *_a, **_k: [], array=lambda x, **_k: x, concatenate=lambda xs: [v for x in xs for v in x])
POLL = SimpleNamespace(Failed=0, WaitingForInput=2, Transferring=3, Success=4)


@contextmanager
def staging_module(root: Path, gather):
    name = "sglang.srt.disaggregation.common.staging_buffer"
    namespace = {}
    for function in ("compute_head_slice_params", "compute_staging_layout", "resolve_total_kv_heads"):
        load_function(root / "common/staging_buffer.py", function, namespace)
    module = SimpleNamespace(**namespace, gather_all_layers_to_staging=gather, StagingAllocator=SimpleNamespace(ALLOC_OVERSIZED=-2))
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if previous is None:
            del sys.modules[name]
        else:
            sys.modules[name] = previous


def check_chunk_grid(root: Path):
    namespace = {"np": NP, "time": SimpleNamespace(perf_counter=lambda: 1)}
    send = load_function(root / "nixl/conn.py", "send", namespace)
    prepare = load_function(root / "common/conn.py", "_prepare_send_indices", namespace)
    for sizes, staged in (([1, 62], True), ([1, 65, 64], True), ([3, 5, 120], True), ([0], True), ([1, 62], False)):
        calls = []
        mgr = SimpleNamespace(enable_staging=staged, enable_all_cp_ranks_for_transfer=False, is_dummy_cp_rank=False, server_args=SimpleNamespace(chunked_prefill_size=4096), kv_args=SimpleNamespace(page_size=64), add_transfer_request=lambda *args, calls=calls: calls.append(args))
        sender = SimpleNamespace(kv_mgr=mgr, bootstrap_room=7, aux_index=9, curr_idx=0, num_kv_indices=sum(sizes), chunk_id=0, _staging_pending_indices=[], _send_failed=False, _transfer_start_time=None, _record_transfer_indices=lambda *_: None)
        sender._prepare_send_indices = lambda indices, state, sender=sender: prepare(sender, indices, state)
        offset = 0
        for size in sizes:
            send(sender, list(range(offset, offset + size)))
            offset += size
        assert [v for args in calls for v in args[1]] == list(range(sum(sizes)))
        assert [args[4] for args in calls] == list(range(len(calls)))
        assert [args[3] for args in calls] == [False] * (len(calls) - 1) + [True]
        expected_sizes = ([64] * (sum(sizes) // 64) + ([sum(sizes) % 64] if sum(sizes) % 64 else [])) or [0]
        assert [len(args[1]) for args in calls] == (expected_sizes if staged else sizes), "partial scheduler batch escaped the fixed staging grid"
        assert [(args[2].start, args[2].stop) for args in calls] == [(sum(len(c[1]) for c in calls[:i]), sum(len(c[1]) for c in calls[:i + 1])) for i in range(len(calls))]
        assert calls[-1][5] == 9 and not sender._staging_pending_indices
    print("PASS: partial prefills 1+62, 1+65+64, 3+5+120 pages; cached aux-only request; non-staged path")


def check_worker_resume(root: Path):
    failures, sent, aux_pending, statuses = [], [], set(), {}
    logger = SimpleNamespace(exception=lambda *a: failures.append(a), warning=lambda *a: failures.append(a))
    namespace = {"logger": logger, "KVPoll": POLL, "time": SimpleNamespace(sleep=lambda *_: None), "STAGING_WATERMARK_WAIT_S": 0, "_NIXL_TRANSPORT_ERRORS": ()}
    worker = load_function(root / "nixl/conn.py", "transfer_worker", namespace)
    chunks = [SimpleNamespace(room=room, chunk_id=chunk, prefill_kv_indices=[chunk], index_slice=slice(chunk, chunk + 1), is_last_chunk=last, prefill_aux_index=9 if last else None, state_indices=None) for room, chunk, last in ((10, 0, False), (10, 1, True), (11, 0, True))]
    items = deque(chunks)
    pops = []

    def get():
        assert not aux_pending, "deferred transfer dropped live aux handles"
        if not items:
            raise StopIteration
        assert len(pops) < 30, "transfer scheduling stopped making progress"
        chunk = items.popleft()
        pops.append((chunk.room, chunk.chunk_id))
        return chunk

    queue = SimpleNamespace(get=get, put=items.append)
    deferred = set()

    def transfer(_strategy, chunk, _src, req, _dst, q):
        key = (chunk.room, chunk.chunk_id, req.agent_name)
        # Stall both an intermediate chunk and a final chunk after peer 0/1.
        if chunk.room == 10 and req.agent_name == "p2" and key not in deferred:
            deferred.add(key)
            q.put(chunk)
            return None, True
        sent.append(key)
        return key, False

    def aux(peer, _src, _dst, _index, notification):
        handle = (peer, notification)
        aux_pending.add(handle)
        return handle

    def poll(handle):
        aux_pending.discard(handle)
        return "DONE"

    peers = {f"p{i}": SimpleNamespace(decode_tp_size=4, staging=object(), dst_aux_ptrs=[0]) for i in range(4)}
    rooms = {room: {p: SimpleNamespace(room=room, agent_name=p, is_dummy=lambda: False, dst_kv_indices=[0, 1], dst_aux_index=0) for p in peers} for room in (10, 11)}
    mgr = SimpleNamespace(enable_staging=True, is_mla_backend=False, is_hybrid_mla_backend=False, attn_tp_size=1, kv_args=SimpleNamespace(kv_data_ptrs=[1]), transfer_source_rank=0, transfer_infos=rooms, decode_kv_args_table=peers, req_to_decode_prefix_len={}, _staging_ctx=SimpleNamespace(prefetched_rooms={10, 11}, prefetch_requested=set()), exceptions={}, check_status=lambda room: statuses.get(room, POLL.WaitingForInput), update_status=lambda room, state: statuses.__setitem__(room, state), _try_create_staging_strategy=lambda _: object(), _do_staging_transfer=transfer, send_aux=aux, agent=SimpleNamespace(check_xfer_state=poll), record_failure=lambda *a: failures.append(a))
    try:
        worker(mgr, queue, object())
    except StopIteration:
        pass
    expected = {(room, chunk, peer) for room, chunk in ((10, 0), (10, 1), (11, 0)) for peer in peers}
    assert set(sent) == expected and len(sent) == len(expected), "a completed peer was replayed or lost"
    assert max(i for i, (r, c, _) in enumerate(sent) if (r, c) == (10, 0)) < min(i for i, (r, c, _) in enumerate(sent) if (r, c) == (10, 1)), "final chunk overtook deferred prefix"
    assert sent.index((11, 0, "p0")) < sent.index((10, 0, "p2")), "one stalled room blocked unrelated requests"
    assert not failures and statuses == {10: POLL.Success, 11: POLL.Success}, failures
    assert not rooms and not aux_pending
    print("PASS: deferred prefix/final chunks retain order, resume at unsent peers, and drain aux; other rooms progress")


def check_source_lifetime(root: Path):
    namespace = {"np": NP, "time": SimpleNamespace(sleep=lambda *_: None)}
    send = load_function(root / "nixl/conn.py", "send_kvcache_staged", namespace)
    received, handles = {}, []
    staging = SimpleNamespace(value=b"", fits=lambda _: True, get_ptr=lambda: 100)
    buffers = [SimpleNamespace(shape=(2, 4, 1), element_size=lambda: 1) for _ in range(95)]

    def payload(head):
        # Different bytes for every K/V layer, including the EAGLE layer.
        return bytes((head * 61 + layer * 7 + token) % 256 for layer in range(190) for token in range(2))

    def gather(_k, _v, _pages, buf, head, _heads, _page, _gpu):
        buf.value = payload(head)

    def init(_op, _src, _dst, peer, _notif):
        handle = SimpleNamespace(peer=peer, polls=0, fail=False)
        handles.append(handle)
        return handle

    def poll(handle):
        if handle.fail:
            return "ERR"
        handle.polls += 1
        if handle.polls == 1:
            return "PROC"
        if handle.peer not in received:
            received[handle.peer] = staging.value
        return "DONE"

    agent = SimpleNamespace(get_xfer_descs=lambda x, _: x, initialize_xfer=init, transfer=lambda _: "PROC", check_xfer_state=poll)
    mgr = SimpleNamespace(kv_buffer_tensors={"k_buffers": buffers, "v_buffers": buffers, "page_size": 1}, kv_args=SimpleNamespace(total_kv_head_num=4, engine_rank=0, gpu_id=0), attn_tp_size=1, agent=agent)
    with staging_module(root, gather):
        for rank in range(4):
            send(mgr, str(rank), [0, 1], 1000, 10000, rank, rank, 4, 1, "notif", staging_buffer=staging)
        # Complete any writes still pending after the four peers reused the buffer.
        for handle in handles:
            while poll(handle) != "DONE":
                pass
        assert received == {str(rank): payload(rank) for rank in range(4)}, "RDMA read a later peer's target/draft head data"
        agent.transfer = lambda handle: setattr(handle, "fail", True) or "PROC"
        try:
            send(mgr, "failure", [0, 1], 1000, 10000, 0, 0, 4, 1, "notif", staging_buffer=staging)
        except RuntimeError as exc:
            assert "bulk transfer failed" in str(exc)
        else:
            raise AssertionError("asynchronous transport error was ignored")
    print("PASS: asynchronous TP1 -> TP4 writes preserve all 94 target + 1 draft K/V layers; terminal transport errors fail")


def check_last_scatter(root: Path):
    namespace = {}
    names = ("_handle_stg_notification", "_handle_aux_notification", "_track_kv_arrival", "_maybe_submit_last_scatter")
    functions = {name: load_function(root / "nixl/conn.py", name, namespace) for name in names}
    for order in (("aux", "first", "last"), ("first", "last", "aux"), ("last", "aux", "first")):
        calls = []
        status = SimpleNamespace(received_kvs_per_pp=defaultdict(set), expected_kvs_per_pp={}, num_pp_ranks_expected=None, received_aux=False, staging_scatter_submitted=False)
        handler = SimpleNamespace(is_staging_room=lambda _: True, submit_last_scatter_async=lambda room, calls=calls: calls.append(("last", room)) or True)
        mgr = SimpleNamespace(transfer_statuses={7: status}, enable_staging=True, _staging_handler=handler, _chunk_writer_counts={}, required_prefill_response_num_table={7: 1}, _handle_staging_chunk_arrived=lambda *args, calls=calls: calls.append(("first", args[0])))
        for name, fn in functions.items():
            setattr(mgr, name, lambda *args, fn=fn, mgr=mgr: fn(mgr, *args))
        for event in (*order, "aux"):
            if event == "aux":
                mgr._handle_aux_notification(7, ["7", "aux"])
            else:
                index = int(event == "last")
                mgr._handle_stg_notification(["7", "stg", str(index), str(index), "0", str(index), str(64 * index), "64", "peer"], 7)
        assert calls == [("first", 7), ("last", 7)], (order, calls)
    print("PASS: final scatter exactly once, after every KV chunk and aux, regardless of notification order")


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
    check_chunk_grid(root)
    check_worker_resume(root)
    check_source_lifetime(root)
    check_last_scatter(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    check(parser.parse_args().source)
