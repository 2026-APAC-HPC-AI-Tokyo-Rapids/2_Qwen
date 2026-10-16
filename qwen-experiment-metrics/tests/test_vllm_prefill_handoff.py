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


@pytest.mark.parametrize("role,tp,pp,ready,early", [("kv_producer", 1, 1, True, True), ("kv_producer", 1, 1, False, False), ("kv_consumer", 1, 1, True, False), ("kv_producer", 2, 1, True, False), ("kv_producer", 1, 2, True, False)])
def test_ready_publish_preserves_forward_cadence(role, tp, pp, ready, early):
    core = harness()
    events = []
    future = Future()
    if ready:
        future.set_result("old_result")
    core.batch_queue = deque([(future, "old_schedule", Future())])
    core.vllm_config = NS(kv_transfer_config=NS(kv_role=role), parallel_config=NS(tensor_parallel_size=tp, pipeline_parallel_size=pp))
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
    failed = Future()
    failed.set_exception(ValueError("worker failed"))
    core.batch_queue = deque([(failed, None, Future())])
    core.vllm_config = NS(kv_transfer_config=NS(kv_role="kv_producer"), parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1))
    core.capture_iteration_details = lambda _: nullcontext()
    core.log_error_detail = lambda _: nullcontext()
    core.step_fn = lambda: pytest.fail("must not submit after an observed worker failure")
    with pytest.raises(ValueError, match="worker failed"):
        core._process_engine_step()


def test_installer_rejects_different_source():
    with pytest.raises(ValueError, match="Unsupported pinned"):
        PATCH["patch"](b"unreviewed source")
