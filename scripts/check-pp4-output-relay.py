"""Check PP result serialization and FutureMap with real CPU tensors.

Usage: python scripts/check-pp4-output-relay.py /path/to/patched/sglang
This exercises source methods; it does not test NCCL, streams, or GPU accuracy.
"""

import ast
import copy
import dataclasses
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch


def source_class(source, relative, name, env):
    path = source / relative
    tree = ast.parse(path.read_text())
    cls = copy.deepcopy(
        next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    )
    program = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(program), str(path), "exec"), env)  # noqa: S102 - execute pinned source methods with CPU fixtures
    return env[name]


class SpecInput:
    def __init__(self, kind):
        self.kind = kind


class Result(NS):
    next_draft_input = None
    draft_output_cpu = None


class Proxy:
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, name):
        return self.tensors[name]


def main():
    source = Path(sys.argv[1]) / "python/sglang/srt"
    env = {
        "__name__": "__main__",
        "dataclass": dataclasses.dataclass,
        "torch": torch,
        "SpecInput": SpecInput,
        "SpecInputType": NS(EAGLE_DRAFT=1),
        "CaptureHiddenMode": NS(FULL=1),
        "DynamicGradMode": lambda: lambda fn: fn,
    }
    draft_cls = source_class(
        source, "speculative/eagle_info.py", "EagleDraftInput", env
    )
    source_class(source, "managers/overlap_utils.py", "RelayPayload", env)
    future_cls = source_class(source, "managers/overlap_utils.py", "FutureMap", env)
    scheduler_cls = source_class(
        source, "managers/scheduler_pp_mixin.py", "SchedulerPPMixin", env
    )
    modules = {}
    for name, fields in {
        "sglang.srt.speculative.spec_utils": {"spec_need_hidden_states": lambda: True},
        "sglang.srt.speculative.eagle_info": {"EagleDraftInput": draft_cls},
        "sglang.srt.managers.scheduler": {"GenerationBatchResult": Result},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(fields)
        modules[name] = module

    count = 0
    with patch.dict(sys.modules, modules):
        for speculative in (False, True):
            for size in (0, 1, 2):
                stages = []
                for _ in range(4):
                    future = future_cls.__new__(future_cls)
                    future.spec_algo = NS(
                        is_ngram=lambda: False,
                        is_some=lambda speculative=speculative: speculative,
                        need_topk=lambda: True,
                    )
                    future.device = "cpu"
                    future.req_pool_size = 8
                    future._forward_buf_initialized = False
                    future.dsa_topk_indices_buf = None
                    future.output_tokens_buf = torch.full((8,), -1, dtype=torch.int64)
                    stages.append(NS(future_map=future))
                for iteration in range(3):
                    slots = torch.tensor([1, 5][:size], dtype=torch.int64)
                    if iteration == 1:
                        slots = slots.flip(0)
                    tokens = torch.arange(size, dtype=torch.int64) + 7 + iteration
                    hidden = (
                        torch.arange(size * 4, dtype=torch.float32).reshape(size, 4)
                        + iteration
                    )
                    probs = torch.full((size, 1), 0.5 + iteration / 10)
                    indices = tokens[:, None] + 2
                    draft = (
                        draft_cls(
                            hidden_states=hidden,
                            topk_p=probs,
                            topk_index=indices,
                            bonus_tokens=tokens,
                        )
                        if speculative
                        else None
                    )
                    produced = Result(next_token_ids=tokens, next_draft_input=draft)
                    send_batch = NS(return_logprob=False)
                    wire = scheduler_cls._pp_prepare_tensor_dict(
                        NS(), produced, send_batch
                    )
                    for stage in stages:
                        # Each hop transfers the complete tensor dictionary independently.
                        received = Proxy(
                            {key: value.clone() for key, value in wire.items()}
                        )
                        batch = NS(
                            return_logprob=False,
                            req_pool_indices=slots,
                            input_ids=tokens,
                            spec_info=None,
                        )
                        result = scheduler_cls._pp_prep_batch_result(
                            stage, batch, NS(can_run_cuda_graph=False), received
                        )
                        torch.testing.assert_close(result.next_token_ids, tokens)
                        torch.testing.assert_close(
                            stage.future_map.output_tokens_buf[slots], tokens
                        )
                        assert batch.input_ids is None
                        if speculative:
                            assert batch.spec_info is result.next_draft_input
                            for actual, expected in zip(
                                result.draft_output_cpu,
                                (hidden, probs, indices),
                                strict=True,
                            ):
                                torch.testing.assert_close(actual, expected)
                            if size:
                                for buf, expected in (
                                    ("hidden_states_buf", hidden),
                                    ("topk_p_buf", probs),
                                    ("topk_index_buf", indices),
                                ):
                                    torch.testing.assert_close(
                                        getattr(stage.future_map, buf)[slots], expected
                                    )
                        else:
                            assert result.next_draft_input is None
                        count += 1
    print(
        f"PASS: {count} PP receive/stash cases: four stages, EAGLE/non-spec, empty/one/two requests, three reuse/reorder iterations"
    )


if __name__ == "__main__":
    main()
