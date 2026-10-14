"""Preallocated device metadata prototype for linear native Qwen EP4 routes.

Research-only: not installed in vLLM and not a complete compact MoE kernel.
The serving call has no host count read, sorting, or tensor allocation.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

LOCAL_EXPERTS = 32
TOPK = 8
MAX_TOKENS = 768


@triton.jit
def _count_reset(
    Ids,
    Valid,
    Counts,
    Inverse,
    Routes,
    R: tl.constexpr,
    B: tl.constexpr,
    RESET: tl.constexpr,
    BASE: tl.constexpr,
):
    expert = tl.program_id(0)
    route = tl.arange(0, B)
    ids = tl.load(Ids + route, route < R, other=-1).to(tl.int32)
    valid = tl.load(Valid + route // 8, route < R, other=0)
    match = (route < R) & valid & (ids == BASE + expert)
    tl.store(Counts + expert, tl.sum(match.to(tl.int32), 0))
    # Every inverse and unused payload row is reset each invocation. These
    # disjoint stores finish before the later scatter launch reads its offsets.
    reset = expert * RESET + tl.arange(0, RESET)
    tl.store(Inverse + reset, -1, reset < R)
    tl.store(Routes + reset, -1, reset < R)


@triton.jit
def _offsets(Counts, Offsets):
    expert = tl.arange(0, 32)
    count = tl.load(Counts + expert)
    inclusive = tl.cumsum(count, 0)
    tl.store(Offsets + expert, inclusive - count)
    tl.store(Offsets + 32, tl.sum(count, 0))


@triton.jit
def _scatter(
    Ids,
    Valid,
    Offsets,
    Inverse,
    Routes,
    R: tl.constexpr,
    B: tl.constexpr,
    BASE: tl.constexpr,
):
    expert = tl.program_id(0)
    route = tl.arange(0, B)
    ids = tl.load(Ids + route, route < R, other=-1).to(tl.int32)
    valid = tl.load(Valid + route // 8, route < R, other=0)
    match = (route < R) & valid & (ids == BASE + expert)
    local_position = tl.cumsum(match.to(tl.int32), 0) - 1
    offset = tl.load(Offsets + expert)
    packed = offset + local_position
    tl.store(Routes + packed, route, match)
    tl.store(Inverse + route, packed, match)


@dataclass
class RouteWorkspace:
    """Fixed buffers; offsets[32] is a device-resident live-route count."""

    tokens: int
    rank: int
    counts: torch.Tensor
    offsets: torch.Tensor
    inverse: torch.Tensor
    routes: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, rank: int, device: torch.device):
        if not 1 <= tokens <= MAX_TOKENS or not 0 <= rank < 4:
            raise ValueError("prototype supports linear EP4, top8, M1..768")
        return cls(
            tokens,
            rank,
            torch.empty(32, dtype=torch.int32, device=device),
            torch.empty(33, dtype=torch.int32, device=device),
            torch.empty(tokens * TOPK, dtype=torch.int32, device=device),
            torch.empty(tokens * TOPK, dtype=torch.int32, device=device),
        )


def launch(workspace: RouteWorkspace, ids: torch.Tensor, valid: torch.Tensor):
    """Launch three fixed-grid metadata kernels into existing buffers.

    `valid` is caller-owned physical-row validity, not an assumed vLLM API.
    Production integration must audit that interface. Full capacity remains
    M*8, including all-routes-on-one-rank. No .item()/CPU transfer or allocation.
    CPU calls are useful only through Triton's explicit interpreter harness.
    """
    if ids.shape != (workspace.tokens, TOPK) or valid.shape != (workspace.tokens,):
        raise ValueError("fixed workspace/route shape mismatch")
    if ids.dtype not in (torch.int32, torch.int64) or valid.dtype != torch.bool:
        raise ValueError("integer expert IDs and boolean row validity required")
    if not ids.is_contiguous() or not valid.is_contiguous():
        raise ValueError("contiguous inputs required")
    if ids.device != valid.device or ids.device != workspace.counts.device:
        raise ValueError("input and workspace devices differ")
    routes = workspace.tokens * TOPK
    block = triton.next_power_of_2(routes)
    reset = triton.next_power_of_2(triton.cdiv(routes, LOCAL_EXPERTS))
    _count_reset[(LOCAL_EXPERTS,)](
        ids,
        valid,
        workspace.counts,
        workspace.inverse,
        workspace.routes,
        routes,
        block,
        reset,
        workspace.rank * LOCAL_EXPERTS,
        num_warps=8,
    )
    _offsets[(1,)](workspace.counts, workspace.offsets, num_warps=1)
    _scatter[(LOCAL_EXPERTS,)](
        ids,
        valid,
        workspace.offsets,
        workspace.inverse,
        workspace.routes,
        routes,
        block,
        workspace.rank * LOCAL_EXPERTS,
        num_warps=8,
    )
