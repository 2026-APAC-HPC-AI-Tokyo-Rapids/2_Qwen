"""Ready results must precede submissions without adding a DP dummy step."""

import ast
import hashlib
import runpy
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATCH = runpy.run_path(str(ROOT / "containers/fix-vllm-prefill-handoff.py"))


class ReadyFuture(Future):
    """Simple ordering fixture; real native Future coverage is below."""

    def poll(self):
        return self.done()


def harness():
    # The fixture contains the original consumption and engine-step anchors.
    source = (
        "class Core:\n"
        "    def step_with_batch_queue(\n        self, future, scheduler_output, exec_model_fut\n    ):\n"
        + PATCH["CONSUME"]
        + "        return engine_core_outputs\n\n"
        "    def _process_engine_step(self):\n"
        "        # Step the engine core.\n"
        "        outputs, model_executed = self.step_fn()\n"
        "        for output in outputs.items() if outputs else ():\n"
        "            self.output_queue.put_nowait(output)\n"
        "        self.post_step(model_executed)\n"
        "        return model_executed\n"
    ).encode()
    patch = PATCH["patch"]
    original = patch.__globals__["SOURCE_SHA256"]
    try:
        patch.__globals__["SOURCE_SHA256"] = hashlib.sha256(source).hexdigest()
        patched = patch(source)
    finally:
        patch.__globals__["SOURCE_SHA256"] = original
    namespace = {}
    exec(compile(ast.parse(patched), "patched-engine-fixture", "exec"), namespace)  # noqa: S102
    return namespace["Core"]()


@pytest.mark.parametrize("role,tp,pp,ready,backend,early", [("kv_producer", 1, 1, True, "mp", True), ("kv_producer", 1, 1, False, "mp", False), ("kv_consumer", 1, 1, True, "mp", False), ("kv_producer", 2, 1, True, "mp", False), ("kv_producer", 1, 2, True, "mp", False), ("kv_producer", 1, 1, True, "uni", False)])
def test_ready_publish_preserves_forward_cadence(role, tp, pp, ready, backend, early):
    core = harness()
    events = []
    future = ReadyFuture()
    if ready:
        future.set_result("old_result")
    core.batch_queue = deque([(future, "old_schedule", Future())])
    core.vllm_config = NS(kv_transfer_config=NS(kv_role=role), parallel_config=NS(tensor_parallel_size=tp, pipeline_parallel_size=pp, distributed_executor_backend=backend))
    core.capture_iteration_details = lambda scheduled: nullcontext("old_metrics")
    core.log_error_detail = lambda scheduled: nullcontext()
    core._process_aborts_queue = lambda: events.append("aborts")

    def update(scheduled, result):
        assert (scheduled, result) == ("old_schedule", "old_result")
        events.append("update")
        return {0: "old_output"}

    core.scheduler = NS(update_from_output=update)
    core._attach_iteration_details = lambda outputs, details: events.append(("metrics", details))
    core.output_queue = NS(put_nowait=lambda output: events.append(("publish", output)))

    def step():
        events.append("submit")
        return {0: "new_output"}, True

    core.step_fn = step
    core.post_step = lambda executed: events.append(("post", executed))
    assert core._process_engine_step() is True  # DP caller must not insert a dummy.
    assert events.count("submit") == 1
    assert len(core.batch_queue) == (0 if early else 1)
    if early:
        assert events[:4] == ["aborts", "update", ("metrics", "old_metrics"), ("publish", (0, "old_output"))]
        assert events.index(("publish", (0, "old_output"))) < events.index("submit")
    else:
        assert events[0] == "submit"
    assert events[-1] == ("post", True)


def test_ready_failure_raises_before_new_forward():
    core = harness()
    failed = ReadyFuture()
    failed.set_exception(ValueError("worker failed"))
    core.batch_queue = deque([(failed, None, Future())])
    core.vllm_config = NS(kv_transfer_config=NS(kv_role="kv_producer"), parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, distributed_executor_backend="mp"))
    core.capture_iteration_details = lambda _: nullcontext()
    core.log_error_detail = lambda _: nullcontext()
    core.step_fn = lambda: pytest.fail("must not submit after an observed worker failure")
    with pytest.raises(ValueError, match="worker failed"):
        core._process_engine_step()


def test_installer_rejects_different_source():
    with pytest.raises(ValueError, match="Unsupported pinned"):
        PATCH["patch"](b"unreviewed source")


def test_real_mp_future_progresses_before_new_dp_step(monkeypatch):
    source = (Path(__file__).parent / "fixtures/vllm-mp-completion.py.txt").read_bytes()
    patch = PATCH["patch_multiproc"]
    monkeypatch.setitem(patch.__globals__, "MP_SOURCE_SHA256", hashlib.sha256(source).hexdigest())
    wrapper = PATCH["selected_future_class"](patch(source))
    responses, futures = deque([None, "old_result"]), deque()
    executed = wrapper(futures, responses.popleft, response_ready=lambda: bool(responses))
    sampled = wrapper(futures, responses.popleft, response_ready=lambda: bool(responses))
    assert not sampled.done()  # Native done() does not inspect available IPC.
    core, events = harness(), []
    core.batch_queue = deque([(sampled, "old_schedule", executed)])
    core.vllm_config = NS(kv_transfer_config=NS(kv_role="kv_producer"), parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, distributed_executor_backend="mp"))
    core.capture_iteration_details = lambda _: nullcontext(None)
    core.log_error_detail = lambda _: nullcontext()
    core._process_aborts_queue = lambda: events.append("aborts")
    core.scheduler = NS(update_from_output=lambda scheduled, result: {0: (scheduled, result)})
    core._attach_iteration_details = lambda *_: None
    core.output_queue = NS(put_nowait=lambda output: events.append(("publish", output)))
    core.step_fn = lambda: (events.append("submit") or ({}, True))
    core.post_step = lambda executed: events.append(("post", executed))
    assert core._process_engine_step() is True
    assert events == ["aborts", ("publish", (0, ("old_schedule", "old_result"))), "submit", ("post", True)]
    assert not futures and not responses and not core.batch_queue
