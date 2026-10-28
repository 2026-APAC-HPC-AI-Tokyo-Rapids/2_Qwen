"""Execute the overlay's host-only gate with four lockstep rank fixtures."""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "patches/sglang/prefill-start-align.patch"


class Tensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class Group:
    def __init__(self):
        self.barrier = Barrier(4, timeout=5)
        self.tensors = []
        self.calls = 0

    def all_reduce(self, tensor, group):
        assert group is self
        self.tensors.append(tensor)
        if self.barrier.wait() == 0:
            total = [sum(column) for column in zip(*(t.values for t in self.tensors), strict=True)]
            for t in self.tensors:
                t.values = total[:]
            self.tensors.clear()
            self.calls += 1
        self.barrier.wait()


@pytest.fixture
def worker():
    text = PATCH.read_text()
    methods = re.search(r"\+    def disagg_prefill_start_idle.*?(?=\n     def maybe_prefetch)", text, re.DOTALL)[0]
    methods = "\n".join(line[1:] for line in methods.splitlines())
    clock = SimpleNamespace(now=0.0)
    group = Group()

    def tensor(values, *, dtype, device):
        assert dtype == "int64" and device == "cpu"
        return Tensor(values)

    ns = {
        "time": SimpleNamespace(monotonic=lambda: clock.now),
        "torch": SimpleNamespace(tensor=tensor, int64="int64", distributed=group),
        "is_aborted": lambda req: req.aborted,
    }
    exec("from __future__ import annotations\nclass Gate:\n" + methods, ns)  # noqa: S102 - methods extracted from the tracked overlay
    ranks = []
    for _ in range(4):
        rank = ns["Gate"]()
        rank.server_args = SimpleNamespace(enable_dp_attention=True)
        rank.ps = SimpleNamespace(attn_dp_size=4, tp_size=4)
        rank.tp_cpu_group = group
        rank.running_batch = SimpleNamespace(is_empty=lambda: True)
        rank.last_batch = None
        rank.result_queue = []
        rank.chunked_req = None
        rank.waiting_queue = []
        rank.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
        rank.disagg_prefill_inflight_queue = []
        rank.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        ranks.append(rank)
    return ranks, clock, group


def step(worker, before=None):
    ranks, _, _ = worker
    idle = before if before is not None else [r.disagg_prefill_start_idle() for r in ranks]
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(lambda args: args[0].delay_disagg_prefill_start(args[1]), zip(ranks, idle, strict=True)))


def ready(rank):
    rank.waiting_queue.append(SimpleNamespace(aborted=False))


def test_idle_then_partial_arrival_and_early_release(worker):
    ranks, clock, group = worker
    assert step(worker) == [False] * 4
    before = [r.disagg_prefill_start_idle() for r in ranks]
    ready(ranks[2])
    assert step(worker, before) == [True] * 4
    clock.now = .040
    ready(ranks[0])
    assert step(worker) == [True] * 4
    clock.now = .060
    ready(ranks[1])
    ready(ranks[3])
    assert step(worker) == [False] * 4
    assert group.calls == 4
    assert all(r._prefill_start_deadline is None for r in ranks)


def test_timeout_is_fixed_and_can_rearm_after_drain(worker):
    ranks, clock, _ = worker
    ready(ranks[0])
    assert step(worker, [True] * 4) == [True] * 4
    clock.now = .099
    ready(ranks[1])
    assert step(worker) == [True] * 4
    clock.now = .100
    assert step(worker) == [False] * 4
    # Existing queued work must not start a second collection window.
    assert step(worker) == [False] * 4
    for rank in ranks:
        rank.waiting_queue.clear()
    assert step(worker) == [False] * 4
    ready(ranks[3])
    assert step(worker, [True] * 4) == [True] * 4


def test_any_rank_expiration_releases_every_rank(worker):
    ranks, clock, _ = worker
    ready(ranks[0])
    assert step(worker, [True] * 4) == [True] * 4
    ranks[3]._prefill_start_deadline = .080
    clock.now = .080
    assert step(worker) == [False] * 4


@pytest.mark.parametrize("busy", ["running", "last_batch", "result_queue", "chunked_req", "waiting_queue", "bootstrap", "inflight", "grammar"])
def test_existing_work_on_any_rank_never_delays(worker, busy):
    ranks, _, _ = worker
    rank = ranks[1]
    if busy == "running":
        rank.running_batch.is_empty = lambda: False
    elif busy == "last_batch":
        rank.last_batch = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False))
    elif busy in ("result_queue", "waiting_queue"):
        getattr(rank, busy).append(SimpleNamespace(aborted=False))
    elif busy == "chunked_req":
        rank.chunked_req = object()
    elif busy == "bootstrap":
        rank.disagg_prefill_bootstrap_queue.queue.append(object())
    elif busy == "inflight":
        rank.disagg_prefill_inflight_queue.append(object())
    else:
        rank.grammar_manager.has_waiting_grammars = lambda: True
    before = [r.disagg_prefill_start_idle() for r in ranks]
    ready(ranks[0])
    assert step(worker, before) == [False] * 4


def test_bootstrap_queue_starts_window_but_is_not_ready(worker):
    ranks, clock, _ = worker
    for rank in ranks:
        rank.disagg_prefill_bootstrap_queue.queue.append(object())
    assert step(worker, [True] * 4) == [True] * 4
    clock.now = .100
    assert step(worker) == [False] * 4


def test_all_ranks_ready_at_once_never_wait(worker):
    for rank in worker[0]:
        ready(rank)
    assert step(worker, [True] * 4) == [False] * 4


def test_cancelled_requests_do_not_release_early(worker):
    ranks, _clock, _ = worker
    for rank in ranks:
        rank.waiting_queue.append(SimpleNamespace(aborted=True))
    assert step(worker, [True] * 4) == [True] * 4
    for rank in ranks:
        rank.waiting_queue.clear()
    assert step(worker) == [False] * 4


@pytest.mark.parametrize("disabled", ["dp_attention", "single_rank"])
def test_non_dp_bypass_has_no_collective(worker, disabled):
    ranks, _, group = worker
    for rank in ranks:
        if disabled == "dp_attention":
            rank.server_args.enable_dp_attention = False
        else:
            rank.ps.attn_dp_size = 1
    assert step(worker, [True] * 4) == [False] * 4
    assert group.calls == 0


def test_both_loops_receive_and_poll_before_gate_and_admission():
    text = PATCH.read_text()
    assert text.count("+            was_idle = self.disagg_prefill_start_idle()") == 2
    assert text.count("+            if self.delay_disagg_prefill_start(was_idle):") == 2
    assert "device=\"cpu\"" in text
    assert "synchronize(" not in text and ".cuda(" not in text
