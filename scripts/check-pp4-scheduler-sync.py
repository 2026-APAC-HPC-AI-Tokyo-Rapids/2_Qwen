"""Exercise the pinned MLPSyncBatchInfo with real CPU tensors (no GPU required).

Usage: python scripts/check-pp4-scheduler-sync.py /path/to/dp_attn.py
Collectives are stubbed; this checks aliasing, fallback values and the CPU-only
singleton path, including a caller that requests CUDA. It does not test NCCL.
"""

import ast
import dataclasses
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


def check_case(info_class, namespace, ranks, active, reduce, tokens, device="cpu", world_size=None):
    world_size = ranks if world_size is None else world_size
    local_only = ranks == world_size == 1
    calls = []
    active_mask = torch.full((ranks,), int(active), dtype=torch.int64)
    namespace["get_tp_group"] = lambda: SimpleNamespace(
        active_ranks_cpu=active_mask, active_ranks=active_mask
    )
    info = info_class(
        dp_size=ranks,
        tp_size=1,
        cp_size=1,
        num_tokens=tokens,
        num_tokens_for_logprob=2 if tokens else 0,
        can_cuda_graph=False,
        is_extend_in_batch=bool(tokens),
        local_can_run_tbo=False,
        local_forward_mode=1 if tokens else 4,
        can_run_breakable_cuda_graph=False,
    )
    local = info._get_local_tensor("cpu").clone()
    fallback = info._get_fallback_tensor("cpu").clone()

    def gather(out, local, group):
        assert not local_only, "singleton must not launch a collective"
        calls.append("gather")
        out.copy_(local.repeat(ranks))

    def all_reduce(*args, **kwargs):
        assert not local_only, "singleton must not launch a collective"
        calls.append("reduce")

    with (
        patch.object(torch.distributed, "all_gather_into_tensor", gather),
        patch.object(torch.distributed, "all_reduce", all_reduce),
        patch.object(torch.distributed, "get_rank", lambda group: 0),
        patch.object(torch.distributed, "get_world_size", lambda group: world_size),
    ):
        info.all_gather(device, None, use_all_reduce=reduce)

    assert calls == ([] if local_only else ["reduce" if reduce else "gather"])

    expected = fallback.repeat(ranks, 1)
    if active:
        if reduce:
            expected[0] = local
        else:
            expected[:] = local
    torch.testing.assert_close(info.tp0_info, expected)
    assert info.global_num_tokens == expected[:, 0].tolist()
    assert info.global_num_tokens_for_logprob == expected[:, 1].tolist()
    assert info.can_cuda_graph == bool(expected[:, 2].min())
    assert info.is_extend_in_batch == bool(expected[:, 3].max())
    assert info.can_run_breakable_cuda_graph == bool(expected[:, 6].min())


def main():
    source = Path(sys.argv[1])
    tree = ast.parse(source.read_text())
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MLPSyncBatchInfo"
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
    namespace = {
        "__name__": "__main__",
        "dataclass": dataclasses.dataclass,
        "torch": torch,
        "ForwardMode": SimpleNamespace(IDLE=SimpleNamespace(value=4)),
        "_ENABLE_METRICS_DP_ATTENTION": False,
    }
    exec(compile(ast.fix_missing_locations(program), str(source), "exec"), namespace)  # noqa: S102 - execute pinned source with CPU fixtures
    count = 0
    for case in product((1, 4), (True, False), (False, True), (0, 8002)):
        check_case(namespace["MLPSyncBatchInfo"], namespace, *case)
        count += 1
    for active, reduce, tokens in product((True, False), (False, True), (0, 8002)):
        # A CUDA device requested by the real PP caller must never initialize CUDA.
        check_case(namespace["MLPSyncBatchInfo"], namespace, 1, active, reduce, tokens, device="cuda")
        # Logical size one alone must not bypass a larger actual process group.
        check_case(namespace["MLPSyncBatchInfo"], namespace, 1, active, reduce, tokens, world_size=4)
        count += 2
    print(f"PASS: {count} real CPU tensor cases: singleton CUDA bypass, ranks 1/4, active/idle, gather/reduce, empty/live")


if __name__ == "__main__":
    main()
