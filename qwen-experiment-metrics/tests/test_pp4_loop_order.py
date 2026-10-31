"""Order of one PP4 loop iteration in the overlay (pp4-loop-latency.patch).

The four-stage execution of the patched loop is scripts/check-pp4-direct-loop.py
(run by the PP4 CPU gate on the staged source). This test pins the order of
the iteration's steps in the overlay text itself:
- every stage after the first takes its result (and sends its KV) before its
  launch, stage 0 after it
- every stage polls its KV transfers after its launch, and forwards the
  transfer list only then
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "patches/sglang/pp4-loop-latency.patch"


def step_post_image():
    """Context and added lines of the hunks inside _qpp_step, in order."""
    lines = []
    inside = False
    for line in PATCH.read_text().splitlines():
        if line.startswith(("@@", "-", "+++", "diff ", "index ")):
            continue
        text = line[1:]
        if text.startswith("    def _qpp_step("):
            inside = True
        elif text.startswith("    def ") and inside:
            break
        if inside:
            lines.append(text.strip())
    return lines


def position(lines, statement, after=0):
    hits = [i for i, line in enumerate(lines) if line == statement and i >= after]
    assert hits, statement
    return hits[0]


def guarded(lines, condition):
    """The statement each `condition` line guards, comments skipped."""
    return [
        next(lines[j] for j in range(i + 1, len(lines)) if lines[j] and not lines[j].startswith("#"))
        for i, line in enumerate(lines)
        if line == condition
    ]


def test_result_step_precedes_the_launch_after_stage_zero():
    lines = step_post_image()
    assert lines.count("self._qpp_result_step(loop, next_mb_id)") == 2
    assert "is_first = self.pp_group.is_first_rank" in lines
    result = "self._qpp_result_step(loop, next_mb_id)"
    assert guarded(lines, "if not is_first:") == [result]
    assert guarded(lines, "if is_first:") == [result]
    early = position(lines, "if not is_first:")
    launch = position(lines, "loop.launches[mb_id] = self._qpp_launch(loop, batch)")
    late = position(lines, "if is_first:")
    assert early < launch < late
    # The old placement (the last stage before its launch, the others after it) is gone.
    assert result not in guarded(lines, "if is_last:") + guarded(lines, "if not is_last:")


def test_transfer_poll_and_its_forward_follow_the_launch():
    lines = step_post_image()
    admit = position(lines, "admitted_rids = self._qpp_admit_bootstrapped(loop)")
    early_send = position(lines, "for data in (recv_reqs, admitted_rids):", admit)
    launch = position(lines, "loop.launches[mb_id] = self._qpp_launch(loop, batch)")
    poll = position(lines, "transferred_rids = self._pp_pd_get_prefill_transferred_ids()")
    assert lines.count("transferred_rids = self._pp_pd_get_prefill_transferred_ids()") == 1
    record = position(lines, "loop.tmbs[mb_id] = transferred_rids")
    forward = position(lines, "transferred_rids, async_send=True", poll)
    consensus = position(lines, "if is_last and loop.tmbs[consensus_mb_id] is not None:")
    release = position(lines, "self.process_disagg_prefill_inflight_queue(release_rids)")
    assert admit < early_send < launch < poll < record < forward < consensus < release
    # Before the launch only requests and admissions go to the next stage.
    assert "for data in (recv_reqs, admitted_rids, transferred_rids):" not in lines
