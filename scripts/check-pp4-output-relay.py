"""Check PP relay through result processing, KV send and release on real CPU tensors.

Usage: python scripts/check-pp4-output-relay.py /path/to/patched/sglang
This exercises source methods; it does not test NCCL, streams, or GPU accuracy.
"""

import ast
import copy
import dataclasses
import sys
from pathlib import Path
from types import MethodType, ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch


def source_class(source, relative, name, env):
    path = source / relative
    tree = ast.parse(path.read_text())
    cls = copy.deepcopy(
        next(n for n in tree.body if isinstance(n, ast.ClassDef | ast.FunctionDef) and n.name == name)
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
    page_indices_cpu = None
    copy_done = None
    routed_experts_output = None
    indexer_topk_output = None


class Proxy:
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, name):
        return self.tensors[name]


class Abort:
    pass


class Sender:
    def __init__(self):
        self.state = "transferring"
        self.sent = []
        self.cleared = 0

    def should_send_kv_chunk(self, count, last):
        return count > 0 or last

    def send(self, pages, states):
        self.sent.append((pages.copy(), states))

    def clear(self):
        self.cleared += 1


def request(slot, iteration):
    return NS(
        rid=f"{slot}:{iteration}", req_pool_idx=slot,
        origin_input_ids=list(range(5 + iteration)), extend_range=NS(end=5 + iteration),
        start_send_idx=0, pending_bootstrap=False, inflight_middle_chunks=0,
        output_ids=[], return_logprob=False, return_sampling_mask=False,
        grammar=None, finished_reason=None, bootstrap_host="fake",
        time_stats=NS(**{name: lambda: None for name in (
            "set_prefill_finished_time", "set_prefill_transfer_queue_entry_time",
            "set_prefill_kv_transfer_finish_time", "set_completion_time",
        )}),
        disagg_kv_sender=Sender(), kv_releases=0, metadata_releases=0,
    )


def lifecycle(stage, batch, result, hidden, probs, indices, speculative):
    stage.process_batch_result_disagg_prefill(batch, result)
    for i, req in enumerate(batch.reqs):
        assert req.output_ids == [int(result.next_token_ids[i])]
        assert req.start_send_idx == len(req.origin_input_ids)
        assert len(req.disagg_kv_sender.sent) == 1
        pages, states = req.disagg_kv_sender.sent[0]
        expected = stage.req_to_token_pool.req_to_token[
            req.req_pool_idx, :len(req.origin_input_ids):2
        ] // 2
        torch.testing.assert_close(torch.from_numpy(pages), expected)
        assert states == []
        if speculative:
            assert req.page_indices_cpu is result.page_indices_cpu[i]
            for actual, reference in (
                (req.hidden_states_tensor, hidden[i]),
                (req.output_topk_p, probs[i]),
                (req.output_topk_index, indices[i]),
            ):
                torch.testing.assert_close(actual, reference)
    # A prior PP stage's consensus is insufficient while this stage still sends.
    rids = [req.rid for req in batch.reqs]
    assert stage.process_disagg_prefill_inflight_queue(rids) == []
    assert all(req.kv_releases == req.metadata_releases == 0 for req in batch.reqs)
    for req in batch.reqs:
        req.disagg_kv_sender.state = "success"
    # Local completion alone is insufficient without PP consensus.
    assert stage.process_disagg_prefill_inflight_queue([]) == []
    assert stage.process_disagg_prefill_inflight_queue(rids) == batch.reqs
    assert stage.process_disagg_prefill_inflight_queue(rids) == []
    for req in batch.reqs:
        assert req.kv_releases == req.metadata_releases == req.disagg_kv_sender.cleared == 1


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
        "scheduler_nvtx_method": lambda name: lambda fn: fn,
        "StateType": NS(**{name: name for name in ("MAMBA", "SWA", "DSA", "MINIMAX_INDEX_K", "SWA_RING", "C128_STATE")}),
        "_is_npu": False,
        "KVPoll": NS(Success="success", Failed="failed", WaitingForInput="waiting", Transferring="transferring"),
        "poll_and_all_reduce_attn_cp_tp_group": lambda senders, *args: [sender.state for sender in senders],
        "maybe_cache_unfinished_req": lambda *args: None,
        "release_kv_cache": lambda req, cache: setattr(req, "kv_releases", req.kv_releases + 1),
        "maybe_release_metadata_buffer": lambda req, allocator: setattr(req, "metadata_releases", req.metadata_releases + 1),
        "FINISH_ABORT": Abort,
        "FINISH_LENGTH": NS,
        "FAKE_BOOTSTRAP_HOST": "fake",
        "logger": NS(warning_once=lambda *args: None),
        "kv_to_page_indices": lambda indices, page_size: (indices[::page_size] // page_size).numpy(),
    }
    draft_cls = source_class(
        source, "speculative/eagle_info.py", "EagleDraftInput", env
    )
    source_class(source, "managers/overlap_utils.py", "RelayPayload", env)
    future_cls = source_class(source, "managers/overlap_utils.py", "FutureMap", env)
    scheduler_cls = source_class(
        source, "managers/scheduler_pp_mixin.py", "SchedulerPPMixin", env
    )
    prefill_cls = source_class(source, "disaggregation/prefill.py", "SchedulerDisaggregationPrefillMixin", env)
    async_d2h = source_class(source, "managers/utils.py", "_async_d2h", env)
    modules = {}
    for name, fields in {
        "sglang.srt.speculative.spec_utils": {"spec_need_hidden_states": lambda: True},
        "sglang.srt.speculative.eagle_info": {"EagleDraftInput": draft_cls},
        "sglang.srt.managers.scheduler": {"GenerationBatchResult": Result},
        "sglang.srt.managers.utils": {"_async_d2h": async_d2h},
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
                    stage = NS(
                        future_map=future, spec_algorithm=NS(is_eagle=lambda speculative=speculative: speculative),
                        req_to_token_pool=NS(req_to_token=torch.arange(128).reshape(8, 16)),
                        token_to_kv_pool_allocator=NS(page_size=2),
                        batch_result_processor=NS(move_logprobs_to_cpu=lambda **kwargs: None),
                        tree_cache=None, disagg_prefill_inflight_queue=[],
                        metrics_reporter=NS(report_prefill_stats=lambda **kwargs: None),
                        disagg_metadata_buffers=NS(set_buf=lambda req: None),
                        disagg_prefill_bootstrap_queue=NS(kv_manager=NS(kv_args=NS(state_types=[]))),
                        attn_cp_cpu_group=None, attn_tp_cpu_group=None, ps=NS(pp_rank=len(stages)),
                        output_streamer=NS(stream_output=lambda *args: None),
                        req_to_metadata_buffer_idx_allocator=None,
                    )
                    for method in ("prefetch_disagg_prefill_outputs", "process_batch_result_disagg_prefill", "send_kv_chunk", "process_disagg_prefill_inflight_queue"):
                        setattr(stage, method, MethodType(getattr(prefill_cls, method), stage))
                    stages.append(stage)
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
                            reqs=[request(int(slot), iteration) for slot in slots],
                            prefill_stats=None, dp_cooperation_info=None,
                        )
                        result = scheduler_cls._pp_prep_batch_result(
                            stage, batch, NS(can_run_cuda_graph=False), received
                        )
                        torch.testing.assert_close(result.next_token_ids, tokens)
                        assert batch.input_ids is None
                        if speculative:
                            assert not stage.future_map._forward_buf_initialized
                            assert (stage.future_map.output_tokens_buf == -1).all()
                            assert batch.spec_info is result.next_draft_input
                            for actual, expected in zip(
                                result.draft_output_cpu,
                                (hidden, probs, indices),
                                strict=True,
                            ):
                                torch.testing.assert_close(actual, expected)
                        else:
                            assert result.next_draft_input is None
                            torch.testing.assert_close(stage.future_map.output_tokens_buf[slots], tokens)
                        lifecycle(stage, batch, result, hidden, probs, indices, speculative)
                        count += 1
    print(
        f"PASS: {count} PP relay/result/KV-send/consensus-release cases: four stages, EAGLE/non-spec, empty/one/two requests, three reuse/reorder iterations"
    )


if __name__ == "__main__":
    main()
