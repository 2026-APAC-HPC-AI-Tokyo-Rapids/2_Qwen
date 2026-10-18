"""CPU regression for request-aligned Prefill chunks in the patched SGLang source.

Runs the real PrefillAdder from the pinned, locally patched source with mocked
KV pools and requests: a disaggregated Prefill rank must not start a request
that it cannot finish in the current batch once the batch holds a request.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace

PAGE = 64
CHUNK = 4096  # chunked_prefill_size 16384 over four DP attention ranks
ISL = 3994  # official 4000-token prompt as tokenized by the server (A1 profile)


def load_adder(source: Path, aligned_env: bool):
    path = source / "python/sglang/srt/managers/schedule_policy.py"
    module = ast.parse(path.read_text())
    wanted = {"AddReqResult", "PrefillAdder"}
    body = [n for n in module.body if isinstance(n, ast.ClassDef) and n.name in wanted]
    assert len(body) == 2, "PrefillAdder or AddReqResult moved"
    assert any(isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_PREFILL_REQUEST_ALIGNED_CHUNKS" for t in n.targets) for n in module.body), "request-aligned chunk switch is missing"
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *body], type_ignores=[])

    class Unused:
        pass

    namespace = {
        "Enum": Enum,
        "auto": auto,
        "contextmanager": contextmanager,
        "CLIP_MAX_NEW_TOKENS": 4096,
        "IGNORE_EOS_RESERVE_TOKENS": 1,
        "_PREFILL_REQUEST_ALIGNED_CHUNKS": aligned_env,
        "is_dsa_prefill_cp_in_seq_split": lambda: False,
        "is_prefill_context_parallel_enabled": lambda: False,
        "SWATokenToKVPoolAllocator": Unused,
        "DeepSeekV4HiSparseTokenToKVPoolAllocator": Unused,
        "PureSWATokenToKVPoolAllocator": Unused,
        "UnifiedMambaTokenToKVPoolAllocator": Unused,
    }
    # Exercise the actual classes from the pinned, locally patched source.
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)  # noqa: S102
    return namespace["PrefillAdder"], namespace["AddReqResult"]


class Req:
    def __init__(self, rid: str, tokens: int, prefix: int = 0):
        self.rid = rid
        self.full_untruncated_fill_ids = list(range(tokens))
        self.origin_input_ids = self.full_untruncated_fill_ids
        self.prefix_indices = list(range(prefix))
        self.sampling_params = SimpleNamespace(max_new_tokens=1, ignore_eos=True)
        self.output_ids = []
        self.host_hit_length = 0
        self.last_node = object()
        self.retracted_stain = False
        self.mamba_pool_idx = None
        self.extend_range = None

    def needs_host_load_back(self):
        return False

    def set_extend_range(self, start: int, end: int):
        self.extend_range = SimpleNamespace(start=start, end=end, length=end - start)


def make_adder(PrefillAdder, aligned: bool, radix_disabled: bool = False):
    tree_cache = SimpleNamespace(disable=radix_disabled, supports_mamba=lambda: False, evictable_size=lambda: 0, is_tree_cache=lambda: False, inc_lock_ref=lambda _: SimpleNamespace(), dec_lock_ref=lambda *_: None)
    allocator = SimpleNamespace(available_size=lambda: 131072)
    running = SimpleNamespace(reqs=[], batch_size=lambda: 0)
    return PrefillAdder(PAGE, tree_cache, allocator, running, 1.0, 16384, CHUNK, request_aligned_chunks=aligned)


def check_policy(source: Path):
    for radix_disabled in (False, True):  # True takes the ignore_eos admission path
        PrefillAdder, Result = load_adder(source, aligned_env=True)

        # Upstream behaviour: the second request is cut to the 64 leftover tokens.
        adder = make_adder(PrefillAdder, aligned=False, radix_disabled=radix_disabled)
        a, b = Req("a", ISL), Req("b", ISL)
        assert adder.add_one_req(a, False, None) == Result.CONTINUE
        adder.add_one_req(b, False, None)
        assert adder.can_run_list == [a, b] and adder.new_chunked_req is b and b.extend_range.length == CHUNK - 4032

        # Aligned: the second request stays queued and nothing is chunked.
        adder = make_adder(PrefillAdder, aligned=True, radix_disabled=radix_disabled)
        a, b = Req("a", ISL), Req("b", ISL)
        assert adder.add_one_req(a, False, None) == Result.CONTINUE
        assert adder.add_one_req(b, False, None) == Result.OTHER
        assert adder.can_run_list == [a] and adder.new_chunked_req is None and b.extend_range is None

        # A short request that fits the leftover budget is still admitted.
        probe = Req("probe", 64)
        adder.add_one_req(probe, False, None)
        assert probe in adder.can_run_list and probe.extend_range.length == 64

        # A lone request longer than the budget still chunks normally.
        adder = make_adder(PrefillAdder, aligned=True, radix_disabled=radix_disabled)
        long = Req("long", 5000)
        adder.add_one_req(long, False, None)
        assert adder.can_run_list == [long] and adder.new_chunked_req is long and long.extend_range.length == CHUNK

        # The tail of an already chunked request continues; no new request joins it.
        adder = make_adder(PrefillAdder, aligned=True, radix_disabled=radix_disabled)
        tail = Req("tail", ISL, prefix=128)
        assert adder.add_chunked_req(tail) is None and tail.extend_range.length == ISL - 128
        b = Req("b", ISL)
        assert adder.add_one_req(b, False, None) == Result.OTHER and adder.can_run_list == [tail]

        # Without the environment switch the scheduler flag alone changes nothing.
        PrefillAdder, Result = load_adder(source, aligned_env=False)
        adder = make_adder(PrefillAdder, aligned=True, radix_disabled=radix_disabled)
        a, b = Req("a", ISL), Req("b", ISL)
        adder.add_one_req(a, False, None)
        adder.add_one_req(b, False, None)
        assert adder.new_chunked_req is b, "the switch must default to upstream behaviour"


def check_scheduler(source: Path):
    path = source / "python/sglang/srt/managers/scheduler.py"
    calls = [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "PrefillAdder"]
    assert len(calls) == 1, "expected one PrefillAdder construction"
    keyword = next((k for k in calls[0].keywords if k.arg == "request_aligned_chunks"), None)
    assert keyword is not None, "scheduler does not pass request_aligned_chunks"
    text = ast.unparse(keyword.value)
    assert text == "self.disaggregation_mode == DisaggregationMode.PREFILL", text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    source = parser.parse_args().source
    check_policy(source)
    check_scheduler(source)
    print("PASS: disaggregated Prefill does not start a request it cannot finish in the batch; lone over-budget requests and chunked tails still chunk; default off (KV pools mocked)")
