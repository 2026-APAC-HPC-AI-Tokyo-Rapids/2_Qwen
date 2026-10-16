"""Publish already-ready prefill results before submitting the next forward.

Keep the regular engine step in the same DP iteration: returning early would
make DPEngineCoreProc insert a dummy forward and alter collective ordering.
"""

import ast
import hashlib
import json
import sys
from pathlib import Path

SOURCE_SHA256 = "268bf40534a853812867ae8e80040f190983cb364b41182b55969aace20c93f5"
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


if __name__ == "__main__":
    target = Path(sys.argv[1])
    output = patch(target.read_bytes())
    target.write_bytes(output)
    print(json.dumps({"source_sha256": SOURCE_SHA256, "patched_sha256": hashlib.sha256(output).hexdigest(), "scope": "kv_producer/TP1/PP1; preserve DP forward cadence"}, indent=2))
