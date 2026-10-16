"""Repair prefill result publication and producer WRITE completion ownership.

Keep the regular engine step in the same DP iteration: returning early would
make DPEngineCoreProc insert a dummy forward and alter collective ordering.
Completed WRITEs release live producer leases; receive metadata belongs to D.
"""

import ast
import copy
import hashlib
import json
import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

SOURCE_SHA256 = "268bf40534a853812867ae8e80040f190983cb364b41182b55969aace20c93f5"
PUSH_SOURCE_SHA256 = "87de3ea383dd3aa95fb2c287ea26ded8a330859cc1295f9faa69047e80865c2c"
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
            and self.batch_queue[-1][0].done()
        ):
            future, scheduled, executed = self.batch_queue.pop()
            ready_outputs = self._consume_batch_queue_output(future, scheduled, executed)
            for output in ready_outputs.items() if ready_outputs else ():
                self.output_queue.put_nowait(output)

'''


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


def install(engine: Path, push_worker: Path) -> dict:
    # Validate both original payloads and the CPU contract before either write.
    output = patch(engine.read_bytes())
    push_output = patch_push_worker(push_worker.read_bytes())
    contract = validate_push_release(push_output)
    engine.write_bytes(output)
    push_worker.write_bytes(push_output)
    return {"source_sha256": SOURCE_SHA256, "patched_sha256": hashlib.sha256(output).hexdigest(), "scope": "kv_producer/TP1/PP1; preserve DP forward cadence", "push_worker_source_sha256": PUSH_SOURCE_SHA256, "push_worker_patched_sha256": hashlib.sha256(push_output).hexdigest(), "push_release_cpu_contract": contract}


if __name__ == "__main__":
    print(json.dumps(install(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
