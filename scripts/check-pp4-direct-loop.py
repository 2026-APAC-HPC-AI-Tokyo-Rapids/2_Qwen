"""Run the patched Qwen PP4 prefill loop on four CPU stages with real torch tensors.

Usage: python scripts/check-pp4-direct-loop.py /path/to/patched/sglang

The actual source of SchedulerPPMixin (event_loop_pp_disagg_prefill_direct and
its _qpp_* helpers, init_pp_loop_state, the bootstrap/transfer consensus
helpers and the pyobj/proxy senders), process_batch_result_disagg_prefill,
send_kv_chunk, prefetch_disagg_prefill_outputs, process_prefill_chunk,
process_disagg_prefill_inflight_queue and GenerationBatchResult runs in four
threads, one per stage. Gloo, PyNccl and the tokenizer are replaced by
blocking in-process queues (a gloo send completes only through wait(), and
wait() blocks until the receiver took the message, as in ProcessGroupGloo);
batch formation, KV allocation, KV senders and the model forward are
deterministic fixtures. Requests arrive in bursts with gaps under a small KV
budget, every stage has its own bootstrap and transfer latency, one bootstrap
fails on a later stage only, and one request bootstraps on the last stage only
after its prefill finished.

Checked: identical microbatches and scheduler state (waiting, bootstrap,
inflight queues, KV allocator) on every stage at every iteration; every stage
admits a request at the first iteration at which stage 0's own bootstrap is
done, whatever its own bootstrap state; a stage sends a request's KV only
after its own bootstrap (at the result step, or later from the inflight queue
when the bootstrap completes after the prefill); a bootstrap that fails on a
later stage only is aborted there at the release every stage applies at the
same iteration; each result is applied pp_loop_size - 1 iterations after its
launch on every stage with the last stage's tokens and EAGLE draft outputs,
before the launch on the last stage and after it elsewhere; the last stage
publishes a result no later than the iteration after its launch; release
happens on every stage at the same iteration and only after all four stages
finished their transfer; gloo sends are waited on and never accumulate, and
the publisher never waits for an unconsumed result; proxy hidden states and
EAGLE aux tensors arrive bit-exact; proxies move on both paths for the
first validation launches (in-stream PyNccl receive -> forward -> send, the
copy-engine slot compared on the receiver), then every stage switches to the
same transport: the copy engine (no PyNccl op afterwards), or PyNccl when one
receiver saw a corrupted slot; a copy-engine slot is overwritten only after
the receiver issued its wait for the previous transfer in it and the sender
applied that microbatch's result (so on a GPU every stage finished its
forward of it), and every wait
refers to the latest copy into its slot; the pipeline reaches pp_loop_size
microbatches in flight, drains, and leaves no message unconsumed except the
last stage's final release consensus rounds.
No NCCL, CUDA stream timing, IPC mappings, NIXL or GPU numerics are exercised.
"""

import ast
import copy
import dataclasses
import itertools
import json
import pickle
import queue
import signal
import sys
import threading
import traceback
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

PP = 4
# The async batch depth the server runs with (configs/firmus-prefill-pp4.json).
DEPTH = json.loads((Path(__file__).resolve().parents[1] / "configs/firmus-prefill-pp4.json").read_text())["prefill"]["pp_async_batch_depth"]
LOOP = PP + DEPTH
STEPS = 96
PAGE = 2
CAPACITY = 120
POOL_ROWS = 32
HIDDEN = 4
TIMEOUT = 20
GPU_LAG = 1
# Copy-engine slots: rows per proxy tensor and launches validated on both paths.
CE_ROWS = 32
VALIDATE = 8
# Tokenizer arrivals at stage 0 by iteration: a burst, a trickle, a gap, a burst.
ARRIVALS = {
    0: [(f"r{i}", 5 + i % 4) for i in range(14)],
    **{step: [(f"r{step + 10}", 6 + step % 3)] for step in range(5, 14)},
    24: [(f"r{30 + i}", 8 + i % 2) for i in range(16)],
    28: [("r50", 9)],
}


def extract(source, relative, names, env):
    tree = ast.parse((source / relative).read_text())
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for name in names:
        node = next(
            n for n in tree.body
            if (isinstance(n, ast.ClassDef | ast.FunctionDef) and n.name == name)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets))
        )
        body.append(copy.deepcopy(node))
    program = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(program, str(source / relative), "exec"), env)  # noqa: S102 - execute pinned source with CPU fixtures
    return [env[name] for name in names]


class Stop(Exception):
    pass


class Work:
    """A gloo send: is_completed() is False until wait(), which blocks until the
    receiver took the message (ProcessGroupGloo SendWork)."""

    def __init__(self, transport, channel, consumed):
        self.transport = transport
        self.channel = channel
        self.consumed = consumed
        self.done = False
        with transport.lock:
            transport.outstanding += 1
            transport.max_outstanding = max(transport.max_outstanding, transport.outstanding)

    def wait(self):
        if not self.consumed.is_set():
            self.transport.blocked_waits[self.channel] += 1
        assert self.consumed.wait(TIMEOUT), "send never consumed"
        if not self.done:
            self.done = True
            with self.transport.lock:
                self.transport.outstanding -= 1
        return True

    def is_completed(self):
        return self.done


class Transport:
    """Blocking FIFO channels keyed by (channel, src, dst, tag)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.queues = defaultdict(queue.Queue)
        self.connections = defaultdict(threading.Event)
        self.published = []
        self.stages = []
        self.outstanding = 0
        self.max_outstanding = 0
        self.blocked_waits = defaultdict(int)

    def put(self, key, value):
        consumed = threading.Event()
        with self.lock:
            self.queues[key].put((value, consumed))
        if key[0] == "nccl":
            return []
        return [NS(work=Work(self, key[0], consumed), payload=None)]

    def get(self, key):
        with self.lock:
            channel = self.queues[key]
        value, consumed = channel.get(timeout=TIMEOUT)
        consumed.set()
        return value

    def connection(self, pair):
        with self.lock:
            return self.connections[pair]

    def leftovers(self):
        return {key: q.qsize() for key, q in self.queues.items() if q.qsize()}


class Event:
    """A device event completes GPU_LAG scheduler iterations after its record."""

    def __init__(self, stage):
        self.stage = stage
        self.stream = None

    def record(self, stream=None):
        self.stream = self.stage.current_stream
        self.step = self.stage.step

    def synchronize(self):
        assert self.stream is not None
        stage = self.stage
        if threading.current_thread() is stage.thread:
            # The scheduler thread may only wait where the forward is done.
            assert stage.step >= self.step + GPU_LAG, "scheduler thread waits on a running forward"
            return
        with stage.progress:
            assert stage.progress.wait_for(lambda: stage.step >= self.step + GPU_LAG, TIMEOUT), "event never completed"


class Comm:
    """PyNccl stand-in: records the stream every point-to-point op runs on."""

    def __init__(self, stage):
        self.stage = stage
        self.available = True
        self.device = torch.device("cpu")

    @contextmanager
    def change_state(self, enable=None):
        yield

    def send(self, tensor, dst):
        stage = self.stage
        stage.stream_log.append(("send", stage.current_stream))
        stage.transport.put(("nccl", stage.rank, dst, 0), tensor.clone())
        # NCCL connects a peer pair lazily: the first send returns only after
        # the peer's first receive, so the peer must be able to reach it.
        assert stage.transport.connection((stage.rank, dst)).wait(TIMEOUT), "PyNccl connection deadlock"

    def recv(self, tensor, src):
        stage = self.stage
        stage.stream_log.append(("recv", stage.current_stream))
        stage.transport.connection((src, stage.rank)).set()
        tensor.copy_(stage.transport.get(("nccl", src, stage.rank, 0)))


class Group:
    def __init__(self, stage):
        self.stage = stage
        self.rank_in_group = self.rank = stage.rank
        self.ranks = list(range(PP))
        self.is_first_rank = stage.rank == 0
        self.is_last_rank = stage.rank == PP - 1
        self.device = torch.device("cpu")
        self.pynccl_comm = Comm(stage)

    def send_object(self, obj, dst, async_send=False, tag=0):
        stage = self.stage
        return stage.transport.put(("pp", stage.rank, dst, tag), pickle.dumps(obj))

    def recv_object(self, src, tag=0):
        stage = self.stage
        return pickle.loads(stage.transport.get(("pp", src, stage.rank, tag)))

    # torch.distributed path used before the in-stream proxy patch.
    def send_tensor_dict(self, tensor_dict, all_gather_group=None, async_send=False):
        stage = self.stage
        stage.stream_log.append(("send", stage.current_stream))
        return stage.transport.put(("pp", stage.rank, stage.rank + 1, 0), {k: v.clone() if torch.is_tensor(v) else v for k, v in tensor_dict.items()})

    def recv_tensor_dict(self, all_gather_group=None):
        stage = self.stage
        stage.stream_log.append(("recv", stage.current_stream))
        return stage.transport.get(("pp", stage.rank - 1, stage.rank, 0))


def make_point_to_point(transport):
    def point_to_point_pyobj(data, rank, group=None, src=0, dst=1, async_send=False):
        channel = "world" if group is None else group
        if rank == src:
            if channel == "result":
                transport.published.append((transport.stages[src].step, data[0]))
            return transport.put((channel, src, dst, 0), pickle.dumps(data))
        assert rank == dst
        return pickle.loads(transport.get((channel, src, dst, 0)))

    return point_to_point_pyobj


# Bootstrap of r5 fails on stage 2 only; r7 bootstraps on the last stage only
# after its prefill result (its KV is sent later, from the inflight queue).
FAILED_BOOTSTRAP = (2, "r5")
LATE_BOOTSTRAP = (PP - 1, "r7", 12)


class Sender:
    """KV sender of one request on one stage with stage-specific latencies."""

    def __init__(self, stage, rid):
        self.stage = stage
        self.rid = rid
        self.created = stage.step
        self.sent = []
        self.cleared = 0
        self.inited = False
        self.fail = (stage.rank, rid) == FAILED_BOOTSTRAP
        self.latency = LATE_BOOTSTRAP[2] if (stage.rank, rid) == LATE_BOOTSTRAP[:2] else 1 + stage.rank

    def poll(self):
        stage = self.stage
        if self.fail:
            return "failed"
        if not self.sent:
            if stage.step - self.created >= self.latency:
                return "waiting"
            return "bootstrapping"
        if stage.step - self.sent_step >= 2 + (3 * stage.rank) % 4:
            return "success"
        return "transferring"

    def should_send_kv_chunk(self, count, last):
        return count > 0 or last

    def init(self, num_pages, metadata_index):
        assert not self.inited and self.poll() == "waiting"
        self.inited = True

    def pop_decode_prefix_len(self):
        return 0

    def failure_exception(self):
        return None

    def send(self, pages, states):
        assert self.inited and not self.sent
        self.sent.append((pages.copy(), states))
        self.sent_step = self.stage.step

    def clear(self):
        self.cleared += 1


class Batch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.return_logprob = False
        self.spec_info = None
        self.input_ids = torch.cat([torch.tensor(r.origin_input_ids) for r in reqs])
        self.req_pool_indices = torch.tensor([r.req_pool_idx for r in reqs])
        self.forward_mode = NS(is_extend=lambda: True)
        self.chunked_req = None
        self.prefill_stats = None
        self.dp_cooperation_info = None

    def __bool__(self):
        return bool(self.reqs)

    def batch_size(self):
        return len(self.reqs)

    def filter_batch(self, chunked_req_to_exclude=None):
        assert not chunked_req_to_exclude


def rows(rids, salt):
    values = [torch.arange(HIDDEN, dtype=torch.float32) + int(rid[1:]) * 10 + salt for rid in rids]
    return torch.stack(values)


def make_stage(rank, transport, classes, env):
    mixin, prefill, result_cls = classes

    class Stage(mixin, prefill):
        pass

    stage = Stage.__new__(Stage)
    stage.rank = rank
    stage.transport = transport
    stage.step = -1
    stage.progress = threading.Condition()
    stage.current_stream = "schedule"
    stage.stream_log = []
    stage.launch_steps = {}
    stage.result_steps = {}
    stage.release_steps = {}
    stage.forwards = []
    stage.states = []
    stage.set_bufs = {}
    stage.in_flight = 0
    stage.max_in_flight = 0
    stage.idle = 0
    stage.ps = NS(pp_rank=rank, pp_size=PP, tp_size=1, tp_rank=0, attn_tp_rank=0, attn_cp_rank=0, attn_dp_rank=0, attn_cp_size=1, attn_tp_size=1)
    stage.server_args = NS(pp_async_batch_depth=DEPTH, enable_dsa_prefill_context_parallel=False)
    stage.pp_group = Group(stage)
    stage.world_group = NS(cpu_group=None)
    stage.attn_cp_cpu_group = stage.attn_tp_cpu_group = stage.attn_tp_group = None
    stage.device_module = NS(Event=lambda: Event(stage))

    @contextmanager
    def stream(name):
        previous = stage.current_stream
        stage.current_stream = name
        try:
            yield
        finally:
            stage.current_stream = previous

    class Ctx:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            self.cm = stream(self.name)
            return self.cm.__enter__()

        def __exit__(self, *exc):
            return self.cm.__exit__(*exc)

    stage.forward_stream_ctx = Ctx("forward")
    stage.copy_stream_ctx = Ctx("copy")
    def wait_event(event):
        # Copy-engine receive: the latest record of the slot event must be the
        # copy of the transfer being received.
        ce = stage.ce
        index = ce.received - 1
        assert event.records and event.records[-1] == index, (stage.rank, index, event.records[-3:])
        stage.ce_waits.add(index)
        stage.stream_log.append(("ce_wait", stage.current_stream))

    stage.forward_stream = stage.copy_stream = stage.schedule_stream = NS(wait_stream=lambda other: None, wait_event=wait_event)
    stage.ce_waits = set()
    stage.spec_algorithm = NS(is_eagle=lambda: True)
    stage.req_to_token_pool = NS(req_to_token=torch.zeros(POOL_ROWS, 32, dtype=torch.int64))
    stage.free_rows = list(range(POOL_ROWS))
    stage.available = CAPACITY
    stage.token_to_kv_pool_allocator = NS(page_size=PAGE)
    stage.batch_result_processor = NS(move_logprobs_to_cpu=lambda **kwargs: None)
    stage.metrics_reporter = NS(report_prefill_stats=lambda **kwargs: None, enable_metrics=False)
    stage.enable_hicache_storage = False

    def set_buf(req):
        assert req.rid not in stage.set_bufs
        stage.set_bufs[req.rid] = (req.output_ids[0], req.hidden_states_tensor.clone(), req.output_topk_p.clone(), req.output_topk_index.clone())

    stage.disagg_metadata_buffers = NS(set_buf=set_buf)
    stage.seen_reqs = []
    stage.disagg_prefill_bootstrap_queue = NS(queue=[], kv_manager=NS(kv_args=NS(state_types=[])))
    stage.disagg_prefill_inflight_queue = []
    stage.waiting_queue = []
    stage.chunked_req = None
    stage.tree_cache = None
    stage.output_streamer = NS(stream_output=lambda *args: None)
    stage.req_to_metadata_buffer_idx_allocator = None
    stage.dp_attn_adapter = NS(maybe_prepare_mlp_sync_batch=lambda batch: batch)
    stage.enable_overlap = False

    def ensure_metadata_buffer(req):
        req.metadata_buffer_index = 0
        return True

    def finalize_bootstrap(req):
        # PrefillBootstrapQueue.finalize_bootstrap: not idempotent, initializes the sender.
        assert req.pending_bootstrap, "finalize_bootstrap is not idempotent"
        ensure_metadata_buffer(req)
        req.disagg_kv_sender.init(len(req.origin_input_ids), req.metadata_buffer_index)
        req.pending_bootstrap = False
        stage.finalized[req.rid] = stage.step
        return True

    def pop_bootstrapped(return_failed_reqs, pp_good_rids, pp_bad_rids):
        good = [r for r in stage.disagg_prefill_bootstrap_queue.queue if r.rid in pp_good_rids]
        bad = [r for r in stage.disagg_prefill_bootstrap_queue.queue if r.rid in pp_bad_rids]
        stage.disagg_prefill_bootstrap_queue.queue = [r for r in stage.disagg_prefill_bootstrap_queue.queue if r not in good and r not in bad]
        for req in good:
            finalize_bootstrap(req)
        for req in bad:
            stage.handle_bootstrap_failure(req)
        return good, bad

    stage.finalized = {}
    stage.disagg_prefill_bootstrap_queue.pop_bootstrapped = pop_bootstrapped
    stage.disagg_prefill_bootstrap_queue.finalize_bootstrap = finalize_bootstrap
    stage.disagg_prefill_bootstrap_queue.ensure_metadata_buffer = ensure_metadata_buffer

    def recv_requests():
        if rank == 0:
            return list(env["arrivals"].get(stage.step, []))
        return env["point_to_point_pyobj"]([], rank, None, rank - 1, rank)

    stage.request_receiver = NS(recv_requests=recv_requests)

    def process_input_requests(reqs):
        for rid, length in reqs:
            req = NS(
                rid=rid, origin_input_ids=list(range(length)), output_ids=[], extend_range=NS(end=length),
                start_send_idx=0, pending_bootstrap=True, inflight_middle_chunks=0, return_logprob=False,
                return_sampling_mask=False, grammar=None, finished_reason=None, bootstrap_host="host",
                req_pool_idx=None, kv=None, mamba_pool_idx=None, metadata_buffer_index=-1, bootstrap_room=0,
                time_stats=NS(**{name: (lambda: None) for name in (
                    "set_prefill_finished_time", "set_prefill_transfer_queue_entry_time",
                    "set_prefill_kv_transfer_finish_time", "set_completion_time", "compute_and_observe_kv_transfer_metrics",
                    "set_wait_queue_entry_time",
                )}, trace_ctx=NS(abort=lambda **kwargs: None)),
            )
            req.time_stats.compute_and_observe_kv_transfer_metrics = lambda metric: None
            req.disagg_kv_sender = Sender(stage, rid)
            req.disagg_kv_sender.get_transfer_metric = lambda: None
            stage.disagg_prefill_bootstrap_queue.queue.append(req)
            stage.seen_reqs.append(req)

    stage.process_input_requests = process_input_requests

    def get_new_batch_prefill(running_batch):
        chosen = []
        while stage.waiting_queue and len(chosen) < 2:
            req = stage.waiting_queue[0]
            need = len(req.origin_input_ids)
            if need > stage.available or not stage.free_rows:
                break
            stage.waiting_queue.pop(0)
            stage.available -= need
            req.req_pool_idx = stage.free_rows.pop(0)
            stage.req_to_token_pool.req_to_token[req.req_pool_idx, :need] = torch.arange(need) + 100 * req.req_pool_idx
            chosen.append(req)
        return NS(batch_to_run=Batch(chosen) if chosen else None, running_batch=running_batch)

    stage.get_new_batch_prefill = get_new_batch_prefill

    def run_batch(batch, pp_proxy_tensors=None):
        rids = [r.rid for r in batch.reqs]
        stage.order.append(("launch", stage.step))
        stage.stream_log.append(("forward", stage.current_stream))
        stage.forwards.append((stage.step, rids))
        if rank == 0:
            assert pp_proxy_tensors is None
            hidden = rows(rids, 0)
            aux = {}
        else:
            got = pp_proxy_tensors.tensors
            torch.testing.assert_close(got["hidden_states"], rows(rids, rank - 1), rtol=0, atol=0)
            torch.testing.assert_close(got["residual"], rows(rids, 100 + rank - 1), rtol=0, atol=0)
            aux = {k: v for k, v in got.items() if k.startswith("eagle_aux_")}
            assert sorted(aux) == [f"eagle_aux_{i}" for i in range(min(rank, 2))]
            for key, value in aux.items():
                torch.testing.assert_close(value, rows(rids, 200 + int(key[-1])), rtol=0, atol=0)
            hidden = got["hidden_states"] + 1
        if rank < 2:
            aux[f"eagle_aux_{rank}"] = rows(rids, 200 + rank)
        if rank < PP - 1:
            proxy = {"hidden_states": hidden, "residual": rows(rids, 100 + rank), **aux}
            return result_cls(pp_hidden_states_proxy_tensors=env["PPProxyTensors"](proxy), can_run_cuda_graph=False)
        tokens = (hidden.sum(dim=1) + sum(v.sum(dim=1) for v in aux.values())).to(torch.int64)
        draft = NS(hidden_states=(hidden * 2).to(torch.bfloat16), topk_p=hidden[:, :1] / 100, topk_index=tokens[:, None] + 1, dsa_topk_indices=None)
        return result_cls(next_token_ids=tokens, next_draft_input=draft, can_run_cuda_graph=False)

    stage.run_batch = run_batch

    def process_batch_result(batch, result):
        stage.order.append(("result", stage.step))
        stage.result_steps[tuple(r.rid for r in batch.reqs)] = stage.step
        stage.process_batch_result_disagg_prefill(batch, result)

    stage.process_batch_result = process_batch_result

    def on_idle():
        stage.idle += 1

    stage.on_idle = on_idle
    real_step = stage._qpp_step

    def step(loop, mb_id):
        with stage.progress:
            stage.step += 1
            stage.progress.notify_all()
        if stage.step == STEPS:
            raise Stop
        stage.states.append((
            [r.rid for r in stage.waiting_queue],
            [r.rid for r in stage.disagg_prefill_bootstrap_queue.queue],
            [r.rid for r in stage.disagg_prefill_inflight_queue],
            stage.available, list(stage.free_rows),
        ))
        return real_step(loop, mb_id)

    stage._qpp_step = step
    real_result = stage._qpp_batch_result

    def batch_result(loop, batch, launch):
        # Launched microbatches whose results are not yet applied, this one included.
        outstanding = sum(item is not None for item in loop.launches)
        stage.max_in_flight = max(stage.max_in_flight, outstanding)
        return real_result(loop, batch, launch)

    stage._qpp_batch_result = batch_result
    real_admit = stage._qpp_admit_bootstrapped

    def admit(loop):
        admitted = real_admit(loop)
        for rid in admitted[0]:
            stage.admit_steps[rid] = stage.step
        return admitted

    stage._qpp_admit_bootstrapped = admit
    stage.order = []
    stage.admit_steps = {}
    return stage


class SlotEvent:
    """Interprocess event of one copy-engine slot: the transfers recorded on it."""

    def __init__(self):
        self.records = []


def make_ce(stages, cls, corrupt):
    ces = []
    for stage in stages:
        ce = cls(slots=LOOP, width=4, rows=CE_ROWS, hidden=HIDDEN, dtype=torch.float32)
        if stage.rank:
            ce.buffers = torch.full((LOOP, 4, CE_ROWS, HIDDEN), float("nan"))
            ce.mismatches = torch.zeros((), dtype=torch.int64)
            ce.checked = NS(record=lambda stream=None: None, synchronize=lambda: None)
        if stage.rank < PP - 1:
            ce.stream = "ce"
            ce.events = [SlotEvent() for _ in range(LOOP)]
        ces.append(ce)
    for stage, ce in zip(stages, ces):
        if stage.rank < PP - 1:
            ce.peer_buffers = ces[stage.rank + 1].buffers
        if stage.rank:
            ce.peer_events = ces[stage.rank - 1].events
        stage.ce = ce
        stage._qpp_ce_setup = lambda ce=ce: ce

        def copy(ce, slot, tensors, stage=stage):
            index = ce.sent - 1
            receiver = stages[stage.rank + 1]
            if index >= ce.slots:
                # The previous transfer in this slot: the receiver issued its wait
                # for it, and this stage already applied that microbatch's result,
                # i.e. on a GPU every stage has finished its forward of it.
                assert index - ce.slots in receiver.ce_waits, (stage.rank, index)
                previous = tuple(stage.forwards[index - ce.slots][1])
                assert previous in stage.result_steps, (stage.rank, index, previous)
            for row, tensor in enumerate(tensors):
                value = tensor + 1 if corrupt and stage.rank == 1 else tensor
                ce.peer_buffers[slot, row, : tensor.shape[0]] = value
            ce.events[slot].records.append(index)
            stage.stream_log.append(("ce_copy", "ce"))

        stage._qpp_ce_copy = copy
    return ces


def run(source, corrupt=False):
    transport = Transport()
    env = {
        "__name__": "__main__",
        "torch": torch, "dataclass": dataclasses.dataclass, "deque": deque, "defaultdict": defaultdict,
        "queue": queue, "threading": threading, "signal": signal,
        "DynamicGradMode": lambda: (lambda fn: fn),
        "ScheduleBatch": lambda reqs, batch_is_full: NS(reqs=reqs, batch_is_full=batch_is_full),
        "scheduler_nvtx_method": lambda name: (lambda fn: fn),
        "KVPoll": NS(Success="success", Failed="failed", WaitingForInput="waiting", Transferring="transferring", Bootstrapping="bootstrapping"),
        "FINISH_ABORT": type("FINISH_ABORT", (), {}), "FINISH_LENGTH": NS, "FAKE_BOOTSTRAP_HOST": "fake",
        "poll_and_all_reduce_attn_cp_tp_group": lambda senders, *args: [s.poll() for s in senders],
        "maybe_cache_unfinished_req": lambda *args, **kwargs: None,
        "maybe_release_metadata_buffer": lambda req, allocator: None,
        "should_force_retry": lambda req: False,
        "is_aborted": lambda req: False,
        "prepare_abort": lambda req, message, status_code=None: setattr(req, "finished_reason", env["FINISH_ABORT"]()),
        "HTTPStatus": NS(INTERNAL_SERVER_ERROR=500, BAD_REQUEST=400),
        "StateType": NS(**{n: n for n in ("MAMBA", "SWA", "DSA", "MINIMAX_INDEX_K", "SWA_RING", "C128_STATE")}),
        "_is_npu": False,
        "logger": NS(warning_once=lambda *a: None, debug=lambda *a: None, info=lambda *a: None, warning=lambda *a: None, error=print),
        "kv_to_page_indices": lambda indices, page_size: (indices[::page_size] // page_size).numpy(),
        "point_to_point_pyobj": make_point_to_point(transport),
        "broadcast_pyobj": None,
        "arrivals": ARRIVALS,
    }
    (tensor_metadata,) = extract(source, "distributed/parallel_state.py", ["TensorMetadata"], {**env, "namedtuple": __import__("collections").namedtuple})
    # Proxy metadata is pickled on the CPU group; make the class importable.
    globals()["TensorMetadata"] = tensor_metadata
    split = extract(source, "distributed/parallel_state.py", ["_split_tensor_dict"], {**env, "TensorMetadata": tensor_metadata})[0]
    (proxy_cls,) = extract(source, "model_executor/forward_batch_info.py", ["PPProxyTensors"], dict(env))
    env["PPProxyTensors"] = proxy_cls
    async_d2h, result_cls = extract(source, "managers/utils.py", ["_async_d2h", "GenerationBatchResult"], {**env, "dataclasses": dataclasses})
    env["GenerationBatchResult"] = result_cls
    extract(source, "managers/scheduler_pp_mixin.py", ["PPBatchMetadata", "_QWEN_PP_CONSENSUS_LAG", "_QWEN_PP_CE_VALIDATE", "_QwenPPCopyEngine", "_QwenPPLaunch", "_QwenPPLoop", "_QwenPPResultPublisher"], env)
    # Fewer validation launches than the server uses, so both phases run here.
    assert env["_QWEN_PP_CE_VALIDATE"] >= VALIDATE
    env["_QWEN_PP_CE_VALIDATE"] = VALIDATE
    (mixin,) = extract(source, "managers/scheduler_pp_mixin.py", ["SchedulerPPMixin"], env)
    (prefill,) = extract(source, "disaggregation/prefill.py", ["SchedulerDisaggregationPrefillMixin"], env)
    modules = {}
    for name, fields in {
        "sglang.srt.managers.utils": {"_async_d2h": async_d2h},
        "sglang.srt.distributed.parallel_state": {"_split_tensor_dict": split, "TensorMetadata": tensor_metadata},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(fields)
        modules[name] = module
    stages = [make_stage(rank, transport, (mixin, prefill, result_cls), env) for rank in range(PP)]
    transport.stages = stages
    make_ce(stages, env["_QwenPPCopyEngine"], corrupt)
    errors = {}
    local = threading.local()

    def release_kv_cache(req, cache):
        # The extracted methods share one globals dict; resolve the calling stage.
        stage = local.stage
        stage.release_steps[req.rid] = stage.step
        stage.available += len(req.origin_input_ids)
        stage.free_rows.append(req.req_pool_idx)
        stage.free_rows.sort()

    env["release_kv_cache"] = release_kv_cache

    def worker(stage):
        local.stage = stage
        stage.thread = threading.current_thread()
        try:
            stage.event_loop_pp_disagg_prefill_direct()
        except Stop:
            pass
        except BaseException:
            errors[stage.rank] = traceback.format_exc()
            raise

    votes = {}
    barrier = threading.Barrier(PP, timeout=TIMEOUT)

    def all_reduce(tensor, op=None, group=None):
        # The copy-engine vote: one MIN over the four stages.
        stage = local.stage
        votes[stage.rank] = int(tensor)
        barrier.wait()
        tensor.fill_(min(votes.values()))
        stage.ce_vote = int(tensor)

    distributed = {"new_group": lambda ranks, backend: "result", "get_world_size": lambda: PP, "all_reduce": all_reduce}
    with patch.dict(sys.modules, modules), patch.multiple(torch.distributed, **distributed):
        threads = [threading.Thread(target=worker, args=(stage,), daemon=True) for stage in stages]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=TIMEOUT * 4)
            assert not thread.is_alive(), "stage did not finish: deadlock"
    if errors:
        raise AssertionError(errors[min(errors)])
    return stages, transport, env["_QWEN_PP_CONSENSUS_LAG"]


def check(stages, transport, lag, corrupt):
    first, last = stages[0], stages[-1]
    # Identical microbatches and scheduler state at every iteration.
    for stage in stages[1:]:
        assert stage.forwards == first.forwards, stage.rank
        assert stage.states == first.states, stage.rank
        assert stage.result_steps == first.result_steps, stage.rank
        assert stage.release_steps == first.release_steps, stage.rank
    launched = {tuple(rids): step for step, rids in first.forwards}
    assert any(len(rids) == 2 for rids in launched) and any(len(rids) == 1 for rids in launched)
    # Empty microbatches between non-empty ones (KV pressure and arrival gaps).
    steps_with_batch = sorted(launched.values())
    assert any(b - a > 1 for a, b in itertools.pairwise(steps_with_batch))
    # Results are applied pp_loop_size - 1 iterations after launch.
    for rids, step in launched.items():
        assert first.result_steps[rids] == step + LOOP - 1, (rids, step)
    # The last stage publishes from its thread once the forward is done,
    # while its scheduler iterations keep going (it never waits on a forward).
    published = {tuple(rids): step for step, rids in transport.published[:: PP - 1]}
    assert published.keys() == launched.keys()
    for rids, step in launched.items():
        assert step + GPU_LAG <= published[rids] <= step + LOOP, (rids, step, published[rids])
    # Pipeline reaches the full in-flight depth on stage 0.
    assert first.max_in_flight == LOOP, first.max_in_flight
    # Every stage admits every request at the first iteration at which stage 0's
    # own bootstrap is done (one iteration after its arrival), whatever its own
    # bootstrap state, so a failing later-stage bootstrap is admitted too.
    admitted = [rid for rids in launched for rid in rids]
    arrived = {rid: step for step, reqs in ARRIVALS.items() for rid, _ in reqs}
    assert sorted(admitted) == sorted(arrived)
    for stage in stages:
        assert stage.admit_steps == {rid: step + 1 for rid, step in arrived.items()}, stage.rank
    # A later stage sends KV only after its own bootstrap: r7 bootstraps on the
    # last stage after its result step and is sent from the inflight queue.
    late_rank, late_rid, _ = LATE_BOOTSTRAP
    late = next(req for req in stages[late_rank].seen_reqs if req.rid == late_rid)
    late_result = next(step for rids, step in stages[late_rank].result_steps.items() if late_rid in rids)
    assert stages[late_rank].finalized[late_rid] > late_result
    assert late.disagg_kv_sender.sent_step == stages[late_rank].finalized[late_rid]
    for stage in stages:
        for req in stage.seen_reqs:
            if req.disagg_kv_sender.sent:
                assert stage.finalized[req.rid] <= req.disagg_kv_sender.sent_step
    # The failed later-stage bootstrap never sends and is aborted on that stage
    # only, at the release iteration every stage applies.
    fail_rank, fail_rid = FAILED_BOOTSTRAP
    for stage in stages:
        req = next(req for req in stage.seen_reqs if req.rid == fail_rid)
        assert (stage.rank == fail_rank) == (not req.disagg_kv_sender.sent)
        assert (stage.rank == fail_rank) == (type(req.finished_reason).__name__ == "FINISH_ABORT"), stage.rank
    # Every admitted request got the last stage's token and EAGLE outputs on every stage.
    for rid in admitted:
        reference = last.set_bufs[rid]
        for stage in stages:
            if (stage.rank, rid) == FAILED_BOOTSTRAP:
                assert rid not in stage.set_bufs
                continue
            token, hidden, probs, index = stage.set_bufs[rid]
            assert token == reference[0]
            for actual, expected in zip((hidden, probs, index), reference[1:], strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert reference[1].dtype == torch.bfloat16
    # Release only after every stage finished its transfer, and all released.
    assert sorted(first.release_steps) == sorted(admitted)
    for stage in stages:
        assert not stage.disagg_prefill_inflight_queue and not stage.waiting_queue
        assert stage.available == CAPACITY and stage.free_rows == list(range(POOL_ROWS))
    for rid in admitted:
        latest = max(
            next(req for req in stage.seen_reqs if req.rid == rid).disagg_kv_sender.sent_step + 2 + (3 * stage.rank) % 4
            for stage in stages
            if (stage.rank, rid) != FAILED_BOOTSTRAP
        )
        assert first.release_steps[rid] >= latest, (rid, first.release_steps[rid], latest)
    # The last stage takes a result before its launch of the same iteration, the
    # other stages after it.
    for stage in stages:
        position = {item: index for index, item in enumerate(stage.order)}
        both = [step for kind, step in stage.order if kind == "result" and ("launch", step) in position]
        assert both
        for step in both:
            assert (position[("result", step)] < position[("launch", step)]) == (stage.rank == PP - 1), stage.rank
    # Gloo sends are waited on once old: the outstanding count stays bounded
    # (it would grow with every iteration otherwise), and the publisher never
    # waits for a result a stage has not taken yet.
    assert transport.max_outstanding <= 8 * (LOOP + 2) * PP, transport.max_outstanding
    assert transport.blocked_waits["result"] == 0, dict(transport.blocked_waits)
    # Proxy transfer runs on the forward stream: receive -> forward -> send.
    # Proxies: both paths for the first VALIDATE launches, then one transport
    # chosen identically on every stage (PyNccl if a slot was corrupted). The
    # PyNccl ops and copy-engine waits run on the forward stream, copies on
    # their side stream.
    mode = "nccl" if corrupt else "copy"
    for stage in stages:
        assert stage.ce.mode == mode and stage.ce_vote == (mode == "copy"), stage.rank
        expected = []
        for index, _ in enumerate(first.forwards):
            nccl = index < VALIDATE or mode == "nccl"
            copied = index < VALIDATE or mode == "copy"
            if stage.rank:
                expected += ["recv"] * nccl + ["ce_wait"] * copied
            expected.append("forward")
            if stage.rank < PP - 1:
                expected += ["ce_copy"] * copied + ["send"] * nccl
        ops = [op for op, _ in stage.stream_log]
        compact = [op for i, op in enumerate(ops) if i == 0 or op != ops[i - 1]]
        assert compact == expected, (stage.rank, compact[:12], expected[:12])
        assert all((stream == "ce") == (op == "ce_copy") for op, stream in stage.stream_log), stage.rank
        assert all(stream in ("forward", "ce") for _, stream in stage.stream_log), stage.rank
    # Only the last stage's final release consensus rounds (one for each of the
    # last `lag` iterations) are left unconsumed.
    left = transport.leftovers()
    assert left == {("world", PP - 1, 0, 0): lag}, left
    return len(admitted), len(launched)


def main():
    source = Path(sys.argv[1]).resolve() / "python/sglang/srt"
    for corrupt in (False, True):
        stages, transport, lag = run(source, corrupt)
        requests, batches = check(stages, transport, lag, corrupt)
    print(f"PASS: four-stage direct-result PP loop, {STEPS} iterations, {batches} microbatches, {requests} requests: identical batches/state per iteration, stage-0 admission followed by every stage, KV only after each stage's own bootstrap (late bootstrap sent from the inflight queue, later-stage failure aborted at the common release), results at +{LOOP - 1} (last stage before its launch), threaded publish after completion, release after all transfers, release lag {lag}, bounded gloo sends, {LOOP} in flight, drained; proxies validated on both paths for {VALIDATE} launches, then copy engine (PyNccl after a corrupted slot)")


if __name__ == "__main__":
    main()
