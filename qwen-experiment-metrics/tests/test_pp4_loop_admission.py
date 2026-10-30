"""Execute the PP4 overlay's admission/pending methods with CPU fixtures.

The full four-stage source harness is scripts/check-pp4-direct-loop.py; these
small tests also cover stage-0 metadata backpressure without importing SGLang.
"""

from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "patches/sglang/pp4-loop-latency.patch"


class Abort:
    pass


def request(rid, poll="bootstrapping", aborted=False):
    return NS(
        rid=rid,
        disagg_kv_sender=NS(poll=lambda: poll),
        pending_bootstrap=True,
        metadata_buffer_index=-1,
        finished_reason=Abort() if aborted else None,
        time_stats=NS(set_wait_queue_entry_time=lambda: None),
    )


@pytest.fixture
def stage_factory():
    added = "\n".join(line[1:] for line in PATCH.read_text().splitlines() if line.startswith("+") and not line.startswith("+++"))
    methods = added[added.index("    def _qpp_admit_bootstrapped"):]
    env = {
        "KVPoll": NS(Bootstrapping="bootstrapping", WaitingForInput="waiting", Failed="failed"),
        "FINISH_ABORT": Abort,
        "poll_and_all_reduce_attn_cp_tp_group": lambda senders, *groups: [sender.poll() for sender in senders],
    }
    exec("from __future__ import annotations\nclass Stage:\n" + methods, env)  # noqa: S102 - tracked overlay methods with CPU fixtures

    def make(rank=0, capacity=8):
        stage = env["Stage"]()
        stage.pp_group = NS(is_first_rank=rank == 0)
        stage.attn_cp_cpu_group = stage.attn_tp_cpu_group = None
        stage.waiting_queue = []
        stage.disagg_prefill_inflight_queue = []
        stage.failed = []
        stage.finalized = []
        stage.sent = []
        stage.free_metadata = list(range(capacity))

        def reserve(req):
            if req.metadata_buffer_index < 0:
                if not stage.free_metadata:
                    return False
                req.metadata_buffer_index = stage.free_metadata.pop(0)
            return True

        def finalize(req):
            assert req.pending_bootstrap
            if not reserve(req):
                return False
            req.pending_bootstrap = False
            stage.finalized.append(req.rid)
            return True

        def send(req, last_chunk):
            assert last_chunk and not req.pending_bootstrap
            stage.sent.append(req.rid)

        stage.disagg_prefill_bootstrap_queue = NS(queue=[], ensure_metadata_buffer=reserve, finalize_bootstrap=finalize)
        stage.handle_bootstrap_failure = lambda req: stage.failed.append(req.rid)
        stage.send_kv_chunk = send
        return stage, NS(pending=[])

    return make


def test_stage_zero_admits_arrivals_in_order_and_routes_failures(stage_factory):
    stage, loop = stage_factory()
    reqs = [request("early"), request("ready", "waiting"), request("aborted", aborted=True), request("failed", "failed"), request("other", "transferring")]
    stage.disagg_prefill_bootstrap_queue.queue = reqs[:]
    assert stage._qpp_admit_bootstrapped(loop) == [["early", "ready"], ["aborted", "failed"]]
    assert stage.waiting_queue == reqs[:2]
    assert loop.pending == [reqs[0]] and reqs[0].pending_bootstrap
    assert all(req.metadata_buffer_index >= 0 for req in reqs[:2])
    assert stage.finalized == ["ready"]
    assert stage.failed == ["aborted", "failed"]
    assert stage.disagg_prefill_bootstrap_queue.queue == [reqs[-1]]


@pytest.mark.parametrize("poll", ["bootstrapping", "waiting"])
def test_metadata_exhaustion_forwards_only_actual_admissions(stage_factory, poll):
    stage, loop = stage_factory(capacity=1)
    reqs = [request("first", poll), request("blocked", poll)]
    stage.disagg_prefill_bootstrap_queue.queue = reqs[:]
    assert stage._qpp_admit_bootstrapped(loop) == [["first"], []]
    assert stage.disagg_prefill_bootstrap_queue.queue == [reqs[1]]
    assert reqs[1].metadata_buffer_index == -1 and reqs[1] not in loop.pending
    assert stage._qpp_admit_bootstrapped(loop) == [[], []]
    # A metadata slot becomes available at a subsequent iteration.
    stage.free_metadata.append(1)
    assert stage._qpp_admit_bootstrapped(loop) == [["blocked"], []]
    assert stage.waiting_queue == reqs and not stage.disagg_prefill_bootstrap_queue.queue


def test_every_later_stage_follows_stage_zero_order_despite_local_polls(stage_factory):
    for rank in (1, 2, 3):
        stage, loop = stage_factory(rank)
        reqs = [request("a", "failed"), request("b", "waiting"), request("c"), request("bad")]
        stage.disagg_prefill_bootstrap_queue.queue = reqs[:]
        stage._pp_recv_pyobj_from_prev_stage = lambda: [["a", "b", "c"], ["bad"]]
        assert stage._qpp_admit_bootstrapped(loop) == [["a", "b", "c"], ["bad"]]
        assert stage.waiting_queue == reqs[:3]
        assert loop.pending == [reqs[0], reqs[2]]
        assert stage.finalized == ["b"] and stage.failed == ["bad"]


@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("prefill_finished", [False, True])
def test_later_poll_finalizes_reserved_sender_and_sends_parked_kv(stage_factory, rank, prefill_finished):
    stage, loop = stage_factory(rank)
    req = request("late")
    stage.disagg_prefill_bootstrap_queue.queue = [req]
    stage._pp_recv_pyobj_from_prev_stage = lambda: [["late"], []]
    assert stage._qpp_admit_bootstrapped(loop) == [["late"], []]
    reserved = req.metadata_buffer_index
    assert req.pending_bootstrap and loop.pending == [req]
    stage._qpp_finalize_pending(loop)
    assert req.pending_bootstrap and loop.pending == [req] and not stage.sent
    if prefill_finished:
        stage.disagg_prefill_inflight_queue.append(req)
    req.disagg_kv_sender.poll = lambda: "waiting"
    stage._qpp_finalize_pending(loop)
    assert not req.pending_bootstrap and not loop.pending
    assert req.metadata_buffer_index == reserved and stage.finalized == ["late"]
    assert stage.sent == (["late"] if prefill_finished else [])
    stage._qpp_finalize_pending(loop)
    assert stage.finalized == ["late"]  # finalize is not idempotent


def test_failed_pending_stage_zero_defers_abort_and_kv_release(stage_factory):
    stage, loop = stage_factory()
    req = request("fails_after_admission")
    stage.disagg_prefill_bootstrap_queue.queue = [req]
    stage._qpp_admit_bootstrapped(loop)
    stage.disagg_prefill_inflight_queue.append(req)
    req.disagg_kv_sender.poll = lambda: "failed"
    stage._qpp_finalize_pending(loop)
    assert req.pending_bootstrap and stage.disagg_prefill_inflight_queue == [req]
    assert not stage.failed and not stage.finalized and not stage.sent
    # Failed leaves the polling list but retains pending_bootstrap. The actual
    # release-consensus abort on every stage is checked by the source harness.
    assert not loop.pending and req.metadata_buffer_index >= 0


def test_later_stage_admission_mismatch_is_asserted(stage_factory):
    stage, loop = stage_factory(1)
    stage._pp_recv_pyobj_from_prev_stage = lambda: [["missing"], []]
    with pytest.raises(AssertionError, match="PP stages diverged in admission"):
        stage._qpp_admit_bootstrapped(loop)


def test_pending_chunk_and_common_result_step_guards_remain():
    text = PATCH.read_text()
    assert "+            assert self.chunked_req is None or not self.chunked_req.pending_bootstrap" in text
    assert "+        if loop.pending:\n+            self._qpp_finalize_pending(loop)" in text
    assert "+        if not is_last:\n+            self._qpp_result_step(loop, next_mb_id)" in text
    assert "+                assert bootstrap_queue.ensure_metadata_buffer(req)" in text
