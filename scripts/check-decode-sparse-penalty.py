"""CPU snapshot/lifecycle oracle and optional actual CUDA kernel parity/timing.

Read the staged source without importing SGLang or initializing CUDA on the CPU
path. --gpu additionally checks the Triton implementation against the retained
dense path, including changed-input CUDA graph replay and non-contiguous logits.
"""

import argparse
import ast
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def definitions(path, wanted, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in wanted]
    assert len(nodes) == len(wanted)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace


def load_min_penalty(source):
    class Base:
        def __init__(self, orchestrator):
            self.orchestrator = orchestrator
            self._is_prepared = True

    path = source / "python/sglang/srt/sampling/penaltylib/min_new_tokens.py"
    return definitions(path, {"BatchedMinNewTokensPenalizer"}, {"torch": torch, "_BatchedPenalizer": Base, "async_admission": SimpleNamespace(ENABLED=False)})["BatchedMinNewTokensPenalizer"]


def request(minimum, stops=(), extra=(), eos=None):
    return SimpleNamespace(sampling_params=SimpleNamespace(min_new_tokens=minimum, stop_token_ids=set(stops)), tokenizer=SimpleNamespace(additional_stop_token_ids=set(extra), eos_token_id=eos))


def prepare(cls, requests, vocab=521, device="cpu"):
    owner = SimpleNamespace(device=device, vocab_size=vocab, reqs=lambda: requests)
    penalty = cls(owner)
    penalty._prepare()
    return penalty


def sparse_oracle(logits, ids, active):
    repeat = logits.shape[0] // ids.shape[0]
    for req in range(ids.shape[0]):
        if active[req, 0]:
            valid = ids[req][ids[req] < logits.shape[1]]
            logits[req * repeat:(req + 1) * repeat, valid] += float("-inf")


def assert_same(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def cpu_checks(source):
    cls = load_min_penalty(source)
    a = prepare(cls, [request(2, [2, 8], [8], 2), request(0), request(4, [1], [4, 9], 7)])
    b = prepare(cls, [request(1, [1, 2, 3, 4, 5, 6], [], 0)])
    a._merge(b)
    assert a.stop_token_ids.shape == (4, 7)
    checked = 0
    for step in range(6):
        for repeat in (1, 4, 8):
            logits = torch.randn(4 * repeat, 521)
            logits[0, 2] = float("inf")
            logits[-1, -1] = float("nan")
            additive = torch.zeros(4, 521)
            a._apply(additive)
            expected = logits + additive.repeat_interleave(repeat, dim=0)
            actual = logits.clone()
            sparse_oracle(actual, a.stop_token_ids, a.len_output_tokens < a.min_new_tokens)
            assert_same(actual, expected)
            checked += 1
        a._cumulate_output_tokens(torch.zeros(4, 1))
    a._filter(torch.tensor([3, 0]))
    assert a.stop_token_ids.shape == (2, 7)
    actual = torch.randn(2, 521)
    expected = actual.clone()
    a._apply(expected)
    sparse_oracle(actual, a.stop_token_ids, a.len_output_tokens < a.min_new_tokens)
    assert_same(actual, expected)

    # Run the actual scheduler update method. Only device identity is mocked;
    # all mask, merge, filter and snapshot tensor operations use real CPU Torch.
    path = source / "python/sglang/srt/sampling/sampling_batch_info.py"
    sampling = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "SamplingBatchInfo")
    update = next(n for n in sampling.body if isinstance(n, ast.FunctionDef) and n.name == "update_penalties")
    namespace = {"torch": torch, "penaltylib": SimpleNamespace(BatchedMinNewTokensPenalizer=cls)}
    exec(compile(ast.Module(body=[update], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102

    class Temperatures:
        is_cuda = True
        device = "cpu"

        def __len__(self):
            return 2

    a = prepare(cls, [request(1, [3]), request(2, [7, 9])])
    owner = SimpleNamespace(is_required=True, penalizers={cls: a}, accumulate_additive_penalties=a._apply, accumulate_scaling_penalties=lambda: None)
    info = SimpleNamespace(temperatures=Temperatures(), vocab_size=521, penalizer_orchestrator=owner)
    namespace["update_penalties"](info)
    old_ids, old_active = info.min_new_tokens_penalty
    assert info.acc_additive_penalties is None and info.acc_scaling_penalties is None
    a._cumulate_output_tokens(torch.zeros(2, 1))
    namespace["update_penalties"](info)
    assert old_active.tolist() == [[True], [True]]
    assert info.min_new_tokens_penalty[1].tolist() == [[False], [True]]
    a._filter(torch.tensor([1]))
    assert old_ids.shape == (2, 2), "filter must not mutate the worker's snapshot"

    a = prepare(cls, [request(1, [3]), request(2, [7, 9])])
    owner.penalizers = {cls: a, "another": SimpleNamespace(_is_prepared=True)}
    owner.accumulate_additive_penalties = a._apply
    namespace["update_penalties"](info)
    assert info.min_new_tokens_penalty is None and info.acc_additive_penalties.shape == (2, 521)
    owner.penalizers = {cls: a}
    info.temperatures.is_cuda = False
    namespace["update_penalties"](info)
    assert info.min_new_tokens_penalty is None and info.acc_additive_penalties.shape == (2, 521)
    info.temperatures.is_cuda = True
    wide = prepare(cls, [request(1, range(257)), request(0)])
    owner.penalizers = {cls: wide}
    owner.accumulate_additive_penalties = wide._apply
    namespace["update_penalties"](info)
    assert info.min_new_tokens_penalty is None and info.acc_additive_penalties.shape == (2, 521)
    owner.is_required = False
    namespace["update_penalties"](info)
    assert info.min_new_tokens_penalty is None and info.acc_additive_penalties is None
    a._teardown()
    assert not hasattr(a, "stop_token_ids")
    print(f"PASS CPU: {checked} dense parity cases, ragged merge/filter, immutable overlap snapshots, mixed-penalty fallback and teardown")


def gpu_checks(source):
    path = source / "python/sglang/srt/sampling/penaltylib/sparse_min_new_tokens.py"
    spec = importlib.util.spec_from_file_location("sparse_min_new_tokens", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    apply = module.apply_min_new_tokens_penalty
    torch.manual_seed(34)
    count = 0
    for batch in (1, 2, 31, 44, 48):
        for repeat in (1, 4, 8):
            for stops in (0, 1, 3, 65, 256):
                ids = torch.arange(stops, device="cuda").expand(batch, -1).clone()
                if stops:
                    ids[0, -1] = 521  # Padded sentinel is never addressed.
                active = (torch.arange(batch, device="cuda") % 2 == 0).reshape(-1, 1)
                for stride in (1, 2):
                    actual = torch.randn(batch * repeat, 521 * stride, device="cuda")[:, ::stride]
                    if stops > 1:
                        actual[0, 0] = float("inf")
                    expected = actual.clone()
                    sparse_oracle(expected, ids, active)
                    apply(actual, ids, active)
                    assert_same(actual, expected)
                    count += 1
    batch, vocab, repeat = 48, 151936, 4
    ids = torch.tensor([[151643, 151645]] * batch, device="cuda")
    active = torch.ones(batch, 1, dtype=torch.bool, device="cuda")
    logits = torch.randn(batch * repeat, vocab, device="cuda")
    apply(logits, ids, active)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        apply(logits, ids, active)
    for enabled in (False, True, False, True):
        active.fill_(enabled)
        logits.normal_()
        expected = logits.clone()
        sparse_oracle(expected, ids, active)
        graph.replay()
        assert_same(logits, expected)

    stop_penalties = torch.zeros(batch, vocab, device="cuda")
    stop_penalties.scatter_(1, ids, float("-inf"))

    def dense():
        accumulated = torch.zeros_like(stop_penalties)
        accumulated.add_(torch.where(active, stop_penalties, 0.0))
        logits.unflatten(0, (batch, repeat)).add_(accumulated.unsqueeze(1))

    def timed(fn):
        for _ in range(10):
            fn()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 10

    times = {"dense_us": [], "sparse_us": []}
    for _ in range(5):
        times["dense_us"].append(timed(dense))
        times["sparse_us"].append(timed(lambda: apply(logits, ids, active)))
    result = {"gpu": torch.cuda.get_device_name(), "uuid": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid), "cases": count, "changed_input_graph_replays": 4, "batch": batch, "vocab": vocab, "verify_width": repeat, **{k: statistics.median(v) for k, v in times.items()}, "samples": times}
    assert result["sparse_us"] < result["dense_us"], result
    print(json.dumps(result, sort_keys=True))
    print("PASS GPU: dense parity, padding, strides, changed-input graph replay and measured speedup")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    cpu_checks(args.source)
    if args.gpu:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        gpu_checks(args.source)
