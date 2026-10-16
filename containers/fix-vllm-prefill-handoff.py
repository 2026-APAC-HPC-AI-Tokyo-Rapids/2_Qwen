"""Repair prefill result publication and producer WRITE completion ownership.

Keep the regular engine step in the same DP iteration: returning early would
make DPEngineCoreProc insert a dummy forward and alter collective ordering.
Completed WRITEs release live producer leases; receive metadata belongs to D.
Submit completed GPU Push blocks / consumer registrations before unrelated
forward work, fencing prior producer stream writes before the RDMA handoff.
"""

import ast
import copy
import hashlib
import json
import queue
import sys
import threading
from collections import deque
from concurrent.futures import Future, InvalidStateError
from contextlib import nullcontext, suppress
from pathlib import Path
from types import SimpleNamespace

SOURCE_SHA256 = "268bf40534a853812867ae8e80040f190983cb364b41182b55969aace20c93f5"
PUSH_SOURCE_SHA256 = "87de3ea383dd3aa95fb2c287ea26ded8a330859cc1295f9faa69047e80865c2c"
MP_SOURCE_SHA256 = "3ed0149a8a8e1550c4f062af40e645d1b734e70f99981bc5bf15f0360507648a"
GPU_KV_SOURCE_SHA256 = "b89a4333f1e552d86e1815674b59e9f1335f0f49c0a7b46082439076f62c6a81"
MQ_SOURCE_SHA256 = "f3e7a3f3ff713fccbd535e0d41650ee05bdf8677e83d4b90519910645022c30b"
MP_POLL = '''    def poll(self) -> bool:
        """Progress available responses in FIFO order without waiting."""
        while not self.done():
            pending = self.futures_queue[-1]
            if not pending.response_ready():
                return False
            self.futures_queue.pop()._wait_for_response()
        return True

'''
MQ_READY = '''    def is_ready(self) -> bool:
        """Inspect the next response without consuming its SHM slot or payload."""
        if self._is_local_reader:
            with self.buffer.get_metadata(self.current_idx) as metadata:
                memory_fence()
                if not metadata[0] or metadata[self.local_reader_rank + 1]:
                    return False
                with self.buffer.get_data(self.current_idx) as buf:
                    # The writer publishes an overflow flag before sending
                    # its socket payload. Both must exist before dequeue.
                    return buf[0] != 1 or bool(self.local_socket.poll(timeout=0))
        if self._is_remote_reader:
            return bool(self.remote_socket.poll(timeout=0))
        raise RuntimeError("Only readers can inspect response readiness")

'''
PUSH_GUARD = '''        done_pushing = {
            req_id
            for req_id in done_pushing - failed_pushing
            if req_id in self._recving_metadata
        }
'''
CONSUME = '''        with (
            self.capture_iteration_details(scheduler_output) as iteration_details,
            self.log_error_detail(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                # None from sample_tokens() implies that the original execute_model()
                # call failed - raise that exception.
                exec_model_fut.result()
                raise RuntimeError("unexpected error")

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        self._attach_iteration_details(engine_core_outputs, iteration_details)
'''
HELPER = '''    def _consume_batch_queue_output(self, future, scheduler_output, exec_model_fut):
        """Consume one scheduled result using the normal abort/metrics path."""
'''+CONSUME+'''        return engine_core_outputs

'''
EARLY = '''        # Publish a ready P result before launching another long forward.
        # Still execute step_fn below: the DP busy loop counts one forward per
        # iteration and inserts a dummy if this method returns False.
        kv_config = self.vllm_config.kv_transfer_config
        parallel = self.vllm_config.parallel_config
        if (
            self.batch_queue
            and kv_config is not None
            and kv_config.kv_role == "kv_producer"
            and parallel.tensor_parallel_size == 1
            and parallel.pipeline_parallel_size == 1
            and parallel.distributed_executor_backend == "mp"
            and self.batch_queue[-1][0].poll()
        ):
            future, scheduled, executed = self.batch_queue.pop()
            ready_outputs = self._consume_batch_queue_output(future, scheduled, executed)
            for output in ready_outputs.items() if ready_outputs else ():
                self.output_queue.put_nowait(output)

'''


GPU_KV_ANCHOR = '''        if scheduler_output.has_sync_kv_loads:
            # Sync loads need to run before this step's forward.
            self._start_load_kv()
        # Otherwise start the async load after forward to keep submission
        # off the critical path.
'''
GPU_KV_EARLY = '''        # Completed producer blocks and consumer registrations concern prior
        # work, not this forward. Do not defer them behind unrelated compute.
        kv_config = self.vllm_config.kv_transfer_config
        parallel = self.vllm_config.parallel_config
        early_push = False
        producer_ready = False
        if (
            kv_config is not None
            and kv_config.kv_connector == "NixlPushConnector"
            and kv_config.kv_buffer_device == "cuda"
            and parallel.tensor_parallel_size == 1
            and parallel.pipeline_parallel_size == 1
            and parallel.prefill_context_parallel_size == 1
            and parallel.decode_context_parallel_size == 1
            and self.vllm_config.kernel_config.moe_backend == "apac_sm90_megamoe"
        ):
            finished = getattr(kv_connector_metadata, "push_finished_blocks", None)
            producer_ready = (
                kv_config.kv_role == "kv_producer"
                and bool(finished)
                and not set(finished).intersection(kwargs.get("request_ids") or ())
            )
            early_push = producer_ready or (
                kv_config.kv_role == "kv_consumer"
                and bool(getattr(kv_connector_metadata, "push_registrations", None))
            )
        if producer_ready:
            # Sampling's output-copy event precedes draft work. Complete prior
            # target/draft KV stores on this stream before handing blocks to
            # the background RDMA writer; do not synchronize a new forward.
            torch.cuda.current_stream().synchronize()
        if scheduler_output.has_sync_kv_loads or early_push:
            self._start_load_kv()
        # Other async loads retain their original post-forward submission.
'''


def patch_gpu_kv(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != GPU_KV_SOURCE_SHA256:
        raise ValueError("Unsupported pinned vLLM GPU KV wrapper source")
    text = source.decode()
    if text.count(GPU_KV_ANCHOR) != 1:
        raise ValueError("Nonunique GPU KV pre-forward anchor")
    result = text.replace(GPU_KV_ANCHOR, GPU_KV_EARLY).encode()
    ast.parse(result)
    return result


def patch(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Unsupported pinned vLLM engine core source")
    text = source.decode()
    helper_anchor = "    def step_with_batch_queue(\n"
    step_anchor = "        # Step the engine core.\n        outputs, model_executed = self.step_fn()\n"
    for anchor in (CONSUME, helper_anchor, step_anchor):
        if text.count(anchor) != 1:
            raise ValueError("Nonunique prefill handoff patch anchor")
    text = text.replace(CONSUME, "        engine_core_outputs = self._consume_batch_queue_output(\n            future, scheduler_output, exec_model_fut\n        )\n")
    text = text.replace(helper_anchor, HELPER + helper_anchor)
    text = text.replace(step_anchor, EARLY + step_anchor)
    result = text.encode()
    ast.parse(result)
    return result


def patch_push_worker(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != PUSH_SOURCE_SHA256:
        raise ValueError("Unsupported pinned vLLM NIXL push worker source")
    text = source.decode()
    if text.count(PUSH_GUARD) != 1:
        raise ValueError("Nonunique producer WRITE completion guard")
    replacement = "        # Completed WRITEs own producer leases, not D receive metadata.\n" + PUSH_GUARD.replace("self._recving_metadata", "self._reqs_to_send")
    result = text.replace(PUSH_GUARD, replacement).encode()
    ast.parse(result)
    return result


def patch_multiproc(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != MP_SOURCE_SHA256:
        raise ValueError("Unsupported pinned vLLM multiprocess executor source")
    replacements = [
        ("        aggregate: Callable = lambda x: x,\n", "        aggregate: Callable = lambda x: x,\n        *,\n        response_ready: Callable[[], bool],\n"),
        ("        self.get_response = get_response\n", "        self.get_response = get_response\n        self.response_ready = response_ready\n"),
        ("    def result(self, timeout=None):\n", MP_POLL + "    def result(self, timeout=None):\n"),
        ("            self.futures_queue, get_response=get_response, aggregate=aggregate\n", "            self.futures_queue, get_response=get_response, aggregate=aggregate,\n            response_ready=lambda: all(mq.is_ready() for mq in response_mqs),\n"),
    ]
    text = source.decode()
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError("Nonunique MP response readiness patch anchor")
        text = text.replace(old, new)
    result = text.encode()
    ast.parse(result)
    return result


def patch_message_queue(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != MQ_SOURCE_SHA256:
        raise ValueError("Unsupported pinned vLLM shared-memory queue source")
    text = source.decode()
    anchor = "    def dequeue(\n"
    if text.count(anchor) != 1:
        raise ValueError("Nonunique message queue readiness patch anchor")
    result = text.replace(anchor, MQ_READY + anchor).encode()
    ast.parse(result)
    return result


def selected_future_class(source: bytes):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FutureWrapper")
    selected = ast.parse("from __future__ import annotations\n")
    selected.body.append(copy.deepcopy(node))
    namespace = {"Future": Future, "suppress": suppress, "InvalidStateError": InvalidStateError}
    exec(compile(ast.fix_missing_locations(selected), "installed-mp-future-contract", "exec"), namespace)  # noqa: S102
    return namespace["FutureWrapper"]


def validate_executor_readiness(mp_source: bytes, mq_source: bytes) -> dict:
    """Check the actual Future/probe methods with injected CPU response state."""
    wrapper = selected_future_class(mp_source)
    events = []
    responses = deque()
    futures = deque()

    def get_response():
        events.append("consume")
        value = responses.popleft()
        if isinstance(value, Exception):
            raise value
        return value

    executed = wrapper(futures, get_response, response_ready=lambda: bool(responses))
    sampled = wrapper(futures, get_response, response_ready=lambda: bool(responses))
    assert not sampled.poll() and not executed.done() and not events
    responses.append(None)
    assert not sampled.poll() and executed.done() and not sampled.done()
    responses.append("ready output")
    assert sampled.poll() and sampled.result() == "ready output" and not futures
    assert sampled.poll() and events == ["consume", "consume"]
    failed = wrapper(futures, get_response, response_ready=lambda: bool(responses))
    responses.append(ValueError("worker failed"))
    assert failed.poll()
    try:
        failed.result()
    except ValueError as error:
        assert str(error) == "worker failed"
    else:
        raise AssertionError("Worker failure was not propagated")

    tree = ast.parse(mq_source)
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MessageQueue")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "is_ready")
    selected = ast.parse("class SelectedQueue:\n    pass\n")
    selected.body[0].body = [copy.deepcopy(method)]
    namespace = {"memory_fence": lambda: None}
    exec(compile(ast.fix_missing_locations(selected), "installed-mq-ready-contract", "exec"), namespace)  # noqa: S102
    probe = namespace["SelectedQueue"]()
    flags, data = bytearray([0, 0]), bytearray([0])
    available = []
    socket = SimpleNamespace(poll=lambda **kw: bool(available))
    probe._is_local_reader, probe._is_remote_reader = True, False
    probe.current_idx, probe.local_reader_rank = 0, 0
    probe.buffer = SimpleNamespace(get_metadata=lambda _: nullcontext(flags), get_data=lambda _: nullcontext(data))
    probe.local_socket = socket
    assert not probe.is_ready()  # unwritten
    flags[0] = 1
    assert probe.is_ready() and flags == bytearray([1, 0]) and probe.current_idx == 0
    flags[1] = 1
    assert not probe.is_ready()  # already consumed
    flags[1], data[0] = 0, 1
    assert not probe.is_ready() and flags == bytearray([1, 0])  # delayed overflow
    available.append(True)
    assert probe.is_ready() and flags == bytearray([1, 0]) and probe.current_idx == 0
    probe._is_local_reader, probe._is_remote_reader = False, True
    probe.remote_socket = socket
    assert probe.is_ready()
    available.clear()
    assert not probe.is_ready()
    return {"passed": True, "scope": "AST-selected installed MP Future and MQ readiness methods; injected CPU response state; no GPU/IPC transport"}


def validate_native_response_transport() -> dict:
    """Exercise installed collective_rpc and actual local SHM/socket responses."""
    import pickle

    import torch
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc

    assert not torch.cuda.is_initialized()
    writer = MessageQueue(n_reader=1, n_local_reader=1, max_chunk_bytes=256, max_chunks=4)
    reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
    commands = []
    executor = SimpleNamespace(rpc_broadcast_mq=SimpleNamespace(enqueue=commands.append), is_failed=False, response_mqs=[reader], futures_queue=deque())

    def rpc():
        return MultiprocExecutor.collective_rpc(executor, "readiness_contract", non_block=True, unique_reply_rank=0)

    try:
        writer.wait_until_ready()
        reader.wait_until_ready()
        executed, sampled = rpc(), rpc()
        assert not sampled.poll() and not executed.done()
        writer.enqueue((WorkerProc.ResponseStatus.SUCCESS, None))
        assert not sampled.poll() and executed.done() and not sampled.done()
        writer.enqueue((WorkerProc.ResponseStatus.SUCCESS, "sampled output"))
        assert sampled.poll() and sampled.result() == "sampled output" and not executor.futures_queue

        # Control the native overflow publication order: SHM flag first, socket
        # payload later. Readiness must not consume the flag while waiting.
        overflow = rpc()
        index = reader.current_idx
        with writer.acquire_write(timeout=1) as buf:
            buf[0] = 1
        assert not overflow.poll() and reader.current_idx == index
        with reader.buffer.get_metadata(index) as flags:
            assert flags[0] == 1 and flags[reader.local_reader_rank + 1] == 0
        payload = "overflow" * 128
        writer.local_socket.send_multipart([pickle.dumps((WorkerProc.ResponseStatus.SUCCESS, payload))])
        assert reader.local_socket.poll(timeout=1000)
        assert overflow.poll() and overflow.result() == payload
        assert reader.current_idx == (index + 1) % reader.buffer.max_chunks

        failed = rpc()
        writer.enqueue((WorkerProc.ResponseStatus.FAILURE, "contract worker failure"))
        assert failed.poll()
        try:
            failed.result()
        except RuntimeError as error:
            assert "contract worker failure" in str(error)
        else:
            raise AssertionError("Native worker failure was not propagated")
        assert not executor.futures_queue and len(commands) == 4
        assert not torch.cuda.is_initialized()
        return {"passed": True, "cases": 4, "cuda_initialized": False, "scope": "installed MP collective_rpc/Future + real local SHM/socket responses; injected command publisher; CPU only"}
    finally:
        writer.shutdown()
        reader.shutdown()
        for socket in (writer.local_socket, writer._spin_condition.local_notify_socket, reader.local_socket, reader._spin_condition.local_notify_socket, reader._spin_condition.read_cancel_socket, reader._spin_condition.write_cancel_socket):
            socket.close(linger=0)


def validate_push_release(source: bytes) -> dict:
    """Execute the installed completion method with CPU transfer outcomes.

    This checks the producer lease/notification flow; it does not simulate DMA,
    UCX, a complete handshake or GPU execution.
    """
    tree = ast.parse(source)
    worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NixlPushConnectorWorker")
    method = next(n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name == "get_transfer_results")

    class Base:
        def get_transfer_results(self):
            return SimpleNamespace(finished_sending=set(self.super_finished))

        def _pop_done_transfers(self, _transfers):
            return set(self.completed), set(self.failed)

    selected = ast.parse("from __future__ import annotations\nclass SelectedPush(Base):\n    pass\n")
    selected.body[1].body = [copy.deepcopy(method)]
    namespace = {"Base": Base}
    exec(compile(ast.fix_missing_locations(selected), "installed-push-release-contract", "exec"), namespace)  # noqa: S102
    # Name, completed writes, failed writes, live P leases, D-only metadata,
    # completions already produced by the base path, expected final notifications.
    cases = [
        ("producer_done", {"p"}, set(), {"p"}, set(), set(), {"p"}),
        ("pending", set(), set(), {"p"}, set(), set(), set()),
        ("failed", {"p"}, {"p"}, {"p"}, set(), set(), set()),
        ("late_after_expiry", {"p"}, set(), set(), set(), set(), set()),
        ("receiver_is_not_producer", {"p"}, set(), set(), {"p"}, set(), set()),
        ("mixed", {"p", "q"}, {"q"}, {"p", "q"}, set(), set(), {"p"}),
        ("base_expiry_preserved", {"p"}, set(), set(), set(), {"p"}, {"p"}),
    ]
    for name, completed, failed, leases, receiving, super_finished, expected in cases:
        state = namespace["SelectedPush"]()
        state.completed, state.failed = completed, failed
        state.super_finished = super_finished
        state._push_writer_wake = threading.Event()
        state._sending_transfers_lock = threading.Lock()
        state._sending_transfers = {}
        state._recving_metadata = dict.fromkeys(receiving)
        state._reqs_to_send = dict.fromkeys(leases, 30.0)
        state._reqs_to_process = set(leases)
        state.consumer_notification_counts_by_req = dict.fromkeys(leases, 0)
        state._evict_finished_inbox = queue.SimpleQueue()
        result = state.get_transfer_results()
        assert result.finished_sending == expected, name
        remaining = leases - expected
        assert set(state._reqs_to_send) == remaining, name
        assert state._reqs_to_process == remaining, name
        assert set(state.consumer_notification_counts_by_req) == remaining, name
        evicted = []
        while not state._evict_finished_inbox.empty():
            evicted.append(state._evict_finished_inbox.get_nowait())
        assert len(evicted) == len(expected) and set(evicted) == expected, name
    return {"passed": True, "cases": len(cases), "scope": "AST-selected installed completion method; injected transfer outcomes; CPU only"}


def validate_push_start(source: bytes) -> dict:
    """Run actual wrapper methods with CPU ordering/data-readiness fixtures."""
    tree = ast.parse(source)
    wrapper = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ActiveKVConnector")
    methods = [copy.deepcopy(n) for n in wrapper.body if isinstance(n, ast.FunctionDef) and n.name in ("pre_forward", "post_forward")]
    assert len(methods) == 2
    selected = ast.parse("from __future__ import annotations\n")
    selected.body.extend(methods)
    namespace = {"KVConnectorOutput": SimpleNamespace}
    exec(compile(ast.fix_missing_locations(selected), "installed-push-start-contract", "exec"), namespace)  # noqa: S102

    def exercise(**changes):
        settings = {"role": "kv_producer", "connector": "NixlPushConnector", "buffer": "cuda", "backend": "apac_sm90_megamoe", "tp": 1, "pp": 1, "pcp": 1, "dcp": 1, "finished": True, "registrations": False, "overlap": False, "sync": False, "disabled": False, "fence_failure": False}
        settings.update(changes)
        events = []

        def fence():
            events.append("prior_stream_ready")
            if settings["fence_failure"]:
                raise RuntimeError("prior stream failed")

        namespace["torch"] = SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=fence)))
        connector = SimpleNamespace(handle_preemptions=lambda _: None, bind_connector_metadata=lambda _: None, wait_for_save=lambda: None, get_transfer_results=lambda _: SimpleNamespace(finished_sending=set(), finished_recving=set(), failed_recving=set()), get_block_ids_with_load_errors=list, get_kv_connector_stats=lambda: None, get_kv_connector_kv_cache_events=lambda: None, build_connector_worker_meta=lambda: None, clear_connector_metadata=lambda: None)
        config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_role=settings["role"], kv_connector=settings["connector"], kv_buffer_device=settings["buffer"]), parallel_config=SimpleNamespace(tensor_parallel_size=settings["tp"], pipeline_parallel_size=settings["pp"], prefill_context_parallel_size=settings["pcp"], decode_context_parallel_size=settings["dcp"]), kernel_config=SimpleNamespace(moe_backend=settings["backend"]))
        state = SimpleNamespace(_disabled=settings["disabled"], vllm_config=config, kv_connector=connector, _pending_load_kwargs=None)

        def start():
            events.append("start_load_kv")
            state._pending_load_kwargs = None

        state._start_load_kv = start
        metadata = SimpleNamespace(push_finished_blocks={"finished": [[1]]} if settings["finished"] else {}, push_registrations={"registration": {}} if settings["registrations"] else {})
        schedule = SimpleNamespace(kv_connector_metadata=metadata, has_sync_kv_loads=settings["sync"])
        try:
            namespace["pre_forward"](state, schedule, request_ids=["finished"] if settings["overlap"] else ["current"])
        except RuntimeError:
            assert settings["fence_failure"] and events == ["prior_stream_ready"]
            return events
        assert not settings["fence_failure"]
        events.append("new_forward")
        namespace["post_forward"](state, set())
        return events

    early_p = ["prior_stream_ready", "start_load_kv", "new_forward"]
    early_d = ["start_load_kv", "new_forward"]
    deferred = ["new_forward", "start_load_kv"]
    cases = [
        ("producer_finished", {}, early_p),
        ("consumer_registration", {"role": "kv_consumer", "registrations": True, "finished": False}, early_d),
        ("consumer_without_registration", {"role": "kv_consumer", "finished": False}, deferred),
        ("producer_without_finished_blocks", {"finished": False}, deferred),
        ("current_request_overlap", {"overlap": True}, deferred),
        ("host_buffer", {"buffer": "cpu"}, deferred),
        ("other_connector", {"connector": "NixlConnector"}, deferred),
        ("other_backend", {"backend": "triton"}, deferred),
        ("tp2", {"tp": 2}, deferred),
        ("pp2", {"pp": 2}, deferred),
        ("pcp2", {"pcp": 2}, deferred),
        ("dcp2", {"dcp": 2}, deferred),
        ("disabled", {"disabled": True}, ["new_forward"]),
        ("original_sync_load", {"finished": False, "sync": True}, early_d),
        ("prior_stream_failure", {"fence_failure": True}, ["prior_stream_ready"]),
    ]
    for name, changes, expected in cases:
        assert exercise(**changes) == expected, name
    return {"passed": True, "cases": len(cases), "scope": "Actual GPU KV wrapper pre/post methods; CPU stubs establish submission/fence order, not DMA correctness or speed", "producer_prior_stream_fence": True, "current_request_overlap_retains_original_order": True}


def install(engine: Path, push_worker: Path, multiproc: Path, message_queue: Path, gpu_kv: Path) -> dict:
    # Validate every source and CPU contract before modifying any installed file.
    output = patch(engine.read_bytes())
    push_output = patch_push_worker(push_worker.read_bytes())
    mp_output = patch_multiproc(multiproc.read_bytes())
    mq_output = patch_message_queue(message_queue.read_bytes())
    gpu_output = patch_gpu_kv(gpu_kv.read_bytes())
    push_start = validate_push_start(gpu_output)
    contract = validate_push_release(push_output)
    readiness = validate_executor_readiness(mp_output, mq_output)
    gpu_kv.write_bytes(gpu_output)
    engine.write_bytes(output)
    push_worker.write_bytes(push_output)
    multiproc.write_bytes(mp_output)
    message_queue.write_bytes(mq_output)
    return {"source_sha256": SOURCE_SHA256, "patched_sha256": hashlib.sha256(output).hexdigest(), "scope": "P output readiness plus direct-GPU Push metadata ordering; TP1/PP1/PCP1/DCP1; preserve DP forward cadence", "push_worker_source_sha256": PUSH_SOURCE_SHA256, "push_worker_patched_sha256": hashlib.sha256(push_output).hexdigest(), "push_release_cpu_contract": contract, "multiproc_source_sha256": MP_SOURCE_SHA256, "multiproc_patched_sha256": hashlib.sha256(mp_output).hexdigest(), "message_queue_source_sha256": MQ_SOURCE_SHA256, "message_queue_patched_sha256": hashlib.sha256(mq_output).hexdigest(), "executor_readiness_cpu_contract": readiness, "gpu_kv_source_sha256": GPU_KV_SOURCE_SHA256, "gpu_kv_patched_sha256": hashlib.sha256(gpu_output).hexdigest(), "push_start_cpu_contract": push_start}


if __name__ == "__main__":
    print(json.dumps(install(*(Path(arg) for arg in sys.argv[1:6])), indent=2))
