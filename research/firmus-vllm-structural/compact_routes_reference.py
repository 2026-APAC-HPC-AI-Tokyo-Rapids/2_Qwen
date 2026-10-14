"""CPU numerical reference for compact local-expert routes, not a GPU benchmark.

Fixtures use unchanged finite FP8-coded weights. BF16/FP8 arithmetic here is a
reference contract; agreement does not certify CUDA kernels or model accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EXPERTS, RANKS, TOPK, HIDDEN, INTERMEDIATE, BLOCK = 128, 4, 8, 128, 128, 128


def bf16(value: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def fp8_values() -> np.ndarray:
    code = np.arange(127, dtype=np.int32)
    exponent, mantissa = code >> 3, code & 7
    return np.where(
        exponent == 0, mantissa * 2.0**-9, np.ldexp(1 + mantissa / 8, exponent - 7)
    ).astype(np.float32)


FP8 = fp8_values()


def encode_fp8(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("reference input must be finite")
    magnitude = np.minimum(np.abs(value), np.float32(448))
    upper = np.minimum(np.searchsorted(FP8, magnitude), 126)
    lower = np.maximum(upper - 1, 0)
    low_distance, high_distance = magnitude - FP8[lower], FP8[upper] - magnitude
    choose_upper = (high_distance < low_distance) | (
        (high_distance == low_distance) & ((upper & 1) == 0)
    )
    code = np.where(choose_upper, upper, lower).astype(np.uint8)
    return code | (np.signbit(value).astype(np.uint8) << 7)


def decode_fp8(code: np.ndarray) -> np.ndarray:
    magnitude = code & np.uint8(127)
    if (magnitude == 127).any():
        raise ValueError("NaN FP8 code")
    return FP8[magnitude] * np.where(
        (code & np.uint8(128)) != 0, np.float32(-1), np.float32(1)
    )


def quantize_rows(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if value.shape[-1] % BLOCK:
        raise ValueError("activation rows must have complete128-element groups")
    groups = value.reshape(-1, value.shape[-1] // BLOCK, BLOCK)
    scale = np.maximum(np.max(np.abs(groups), axis=-1), np.float32(1e-12)) / np.float32(
        448
    )
    code = encode_fp8(groups / scale[..., None])
    decoded = decode_fp8(code) * scale[..., None]
    return code.reshape(value.shape), scale, decoded.reshape(value.shape)


def activate(value: np.ndarray) -> np.ndarray:
    gate, up = np.split(value, 2, axis=-1)
    exp = np.exp(-np.abs(gate))
    sigmoid = np.where(gate >= 0, 1 / (1 + exp), exp / (1 + exp))
    return bf16(gate * sigmoid * up)


@dataclass
class RoutePlan:
    slots: np.ndarray
    experts: np.ndarray
    reverse: np.ndarray
    capacity: int


def plan_routes(ids: np.ndarray, valid: np.ndarray, rank: int) -> RoutePlan:
    if ids.ndim != 2 or ids.shape[1] != TOPK or valid.shape != (ids.shape[0],):
        raise ValueError("route shape")
    if ((ids < 0) | (ids >= EXPERTS)).any() or not 0 <= rank < RANKS:
        raise ValueError("invalid expert/rank")
    local = (ids // (EXPERTS // RANKS) == rank) & valid[:, None]
    slots = np.flatnonzero(local).astype(np.int32)
    experts = ids.reshape(-1)[slots]
    order = np.lexsort((slots, experts))
    slots, experts = slots[order], experts[order]
    reverse = np.full(ids.shape, -1, dtype=np.int32)
    reverse.reshape(-1)[slots] = np.arange(slots.size, dtype=np.int32)
    # All routes can legally belong to one rank. Capacity cannot assume1/4.
    return RoutePlan(slots, experts, reverse, ids.size)


def sum_slots(branches: np.ndarray) -> np.ndarray:
    out = np.zeros((branches.shape[0], HIDDEN), dtype=np.float32)
    for slot in range(TOPK):
        out += branches[:, slot]
    return bf16(out)


def full_rank(
    x: np.ndarray,
    ids: np.ndarray,
    weights: np.ndarray,
    valid: np.ndarray,
    rank: int,
    w1: np.ndarray,
    w2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gate_up = np.zeros((ids.shape[0], TOPK, INTERMEDIATE * 2), dtype=np.float32)
    for expert in range(rank * 32, (rank + 1) * 32):
        token, slot = np.nonzero((ids == expert) & valid[:, None])
        gate_up[token, slot] = bf16(x[token] @ w1[expert].T)
    codes, scales, hidden = quantize_rows(activate(gate_up).reshape(-1, INTERMEDIATE))
    hidden = hidden.reshape(ids.shape[0], TOPK, INTERMEDIATE)
    branches = np.zeros((ids.shape[0], TOPK, HIDDEN), dtype=np.float32)
    for expert in range(rank * 32, (rank + 1) * 32):
        token, slot = np.nonzero((ids == expert) & valid[:, None])
        branches[token, slot] = bf16(
            (hidden[token, slot] @ w2[expert].T) * weights[token, slot, None]
        )
    return sum_slots(branches), codes, scales


def compact_rank(
    x: np.ndarray,
    ids: np.ndarray,
    weights: np.ndarray,
    valid: np.ndarray,
    rank: int,
    w1: np.ndarray,
    w2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, RoutePlan]:
    plan = plan_routes(ids, valid, rank)
    count = plan.slots.size
    # Poison unused fixed-capacity rows; correctness must not read them.
    packed = np.full((plan.capacity, INTERMEDIATE * 2), np.nan, dtype=np.float32)
    for expert in range(rank * 32, (rank + 1) * 32):
        rows = np.flatnonzero(plan.experts == expert)
        token = plan.slots[rows] // TOPK
        packed[rows] = bf16(x[token] @ w1[expert].T)
    if count:
        codes, scales, hidden = quantize_rows(activate(packed[:count]))
    else:
        codes = np.empty((0, INTERMEDIATE), dtype=np.uint8)
        scales = np.empty((0, INTERMEDIATE // BLOCK), dtype=np.float32)
        hidden = np.empty((0, INTERMEDIATE), dtype=np.float32)
    down = np.full((plan.capacity, HIDDEN), np.nan, dtype=np.float32)
    for expert in range(rank * 32, (rank + 1) * 32):
        rows = np.flatnonzero(plan.experts == expert)
        slot_weight = weights.reshape(-1)[plan.slots[rows]]
        down[rows] = bf16((hidden[rows] @ w2[expert].T) * slot_weight[:, None])
    out = np.zeros((ids.shape[0], HIDDEN), dtype=np.float32)
    for slot in range(TOPK):
        rows = plan.reverse[:, slot]
        present = rows >= 0
        out[present] += down[rows[present]]
    return bf16(out), codes, scales, plan


def run_checks() -> dict:
    rng = np.random.default_rng(20261014)
    # Frozen fixture weights are already FP8-coded; no checkpoint loading or
    # runtime weight requantization is part of the proposed route change.
    codes1 = rng.choice(
        np.array([8, 16, 24, 32, 40, 136, 144, 152, 160, 168], dtype=np.uint8),
        size=(EXPERTS, INTERMEDIATE * 2, HIDDEN),
    )
    codes2 = rng.choice(
        np.array([8, 16, 24, 32, 40, 136, 144, 152, 160, 168], dtype=np.uint8),
        size=(EXPERTS, HIDDEN, INTERMEDIATE),
    )
    scale1 = rng.uniform(0.09, 0.21, (EXPERTS, 2, 1)).astype(np.float32)
    scale2 = rng.uniform(0.09, 0.21, (EXPERTS, 1, 1)).astype(np.float32)
    weight_digest = hashlib.sha256(
        codes1.tobytes() + codes2.tobytes() + scale1.tobytes() + scale2.tobytes()
    ).hexdigest()
    w1 = decode_fp8(codes1) * np.repeat(np.repeat(scale1, BLOCK, axis=1), BLOCK, axis=2)
    w2 = decode_fp8(codes2) * scale2
    all_codes = np.arange(256, dtype=np.uint8)
    finite = (all_codes & 127) != 127
    assert np.array_equal(encode_fp8(decode_fp8(all_codes[finite])), all_codes[finite])
    midpoints = (FP8[:-1] + FP8[1:]) / 2
    expected_ties = np.where(
        np.arange(126) % 2 == 0, np.arange(126), np.arange(1, 127)
    ).astype(np.uint8)
    assert np.array_equal(encode_fp8(midpoints), expected_ties)
    cases = []
    for tokens in [1, 7, 32, 97, 288]:
        _, _, x = quantize_rows(
            bf16(rng.normal(size=(tokens, HIDDEN)).astype(np.float32))
        )
        for mode in [
            "uniform",
            "all_rank0",
            "boundary",
            "masked",
            "empty",
            "changing_routes",
        ]:
            ids = np.vstack(
                [rng.choice(EXPERTS, TOPK, replace=False) for _ in range(tokens)]
            ).astype(np.int32)
            valid = np.ones(tokens, dtype=bool)
            if mode == "all_rank0":
                ids = np.vstack(
                    [rng.choice(32, TOPK, replace=False) for _ in range(tokens)]
                ).astype(np.int32)
            if mode == "boundary":
                ids[:] = [0, 31, 32, 63, 64, 95, 96, 127]
            if mode == "masked":
                valid[::3] = False
            if mode == "empty":
                valid[:] = False
            if mode == "changing_routes":
                previous = plan_routes(np.zeros_like(ids), valid, 0)
                ids = (ids + 32) % EXPERTS
                current = plan_routes(ids, valid, 0)
                assert not np.array_equal(previous.reverse, current.reverse)
            raw_weights = rng.uniform(0.01, 1, ids.shape).astype(np.float32)
            weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)
            counts, full_outputs, compact_outputs = [], [], []
            for rank in range(RANKS):
                reference, ref_codes, ref_scales = full_rank(
                    x, ids, weights, valid, rank, w1, w2
                )
                compact, qcodes, qscales, plan = compact_rank(
                    x, ids, weights, valid, rank, w1, w2
                )
                assert np.array_equal(qcodes, ref_codes[plan.slots])
                assert np.array_equal(qscales, ref_scales[plan.slots])
                assert np.array_equal(
                    reference.view(np.uint32), compact.view(np.uint32)
                )
                assert np.isfinite(compact).all() and not compact[~valid].any()
                assert plan.slots.size <= plan.capacity
                assert np.array_equal(
                    plan.reverse.reshape(-1)[plan.slots], np.arange(plan.slots.size)
                )
                counts.append(int(plan.slots.size))
                full_outputs.append(reference)
                compact_outputs.append(compact)
            full = bf16(np.sum(full_outputs, axis=0, dtype=np.float32))
            compact = bf16(np.sum(compact_outputs, axis=0, dtype=np.float32))
            assert np.array_equal(full.view(np.uint32), compact.view(np.uint32))
            assert sum(counts) == int(valid.sum()) * TOPK
            if mode == "all_rank0":
                assert counts == [tokens * TOPK, 0, 0, 0]
            cases.append(
                {
                    "physical_tokens": tokens,
                    "mode": mode,
                    "active_tokens": int(valid.sum()),
                    "local_routes_per_rank": counts,
                    "fixed_capacity_per_rank": tokens * TOPK,
                    "bit_exact_reference_output": True,
                    "activation_fp8_bytes_and_scales_exact": True,
                }
            )
    assert (
        weight_digest
        == hashlib.sha256(
            codes1.tobytes() + codes2.tobytes() + scale1.tobytes() + scale2.tobytes()
        ).hexdigest()
    )
    return {
        "schema": "compact-local-routes-cpu-reference/v1",
        "seed": 20261014,
        "numpy_version": np.__version__,
        "fixture": {
            "global_experts": EXPERTS,
            "ranks": RANKS,
            "topk": TOPK,
            "hidden": HIDDEN,
            "expert_intermediate": INTERMEDIATE,
            "activation_group": BLOCK,
            "weight_digest_unchanged": weight_digest,
        },
        "cases": cases,
        "checks": {
            "finite_fp8_codes_roundtrip": 254,
            "fp8_midpoint_rne_ties": 126,
            "full_vs_compact_rank_comparisons": len(cases) * RANKS,
            "rank_reduce_comparisons": len(cases),
        },
        "limitations": "CPU reference and fixture arithmetic only, not native CUDA/BF16/FP8 numerical certification, graph capture, benchmark timing, official-model accuracy or speedup. Physical worst-case capacity stays full top8; only active local-route work is compacted. The reference retains original route-slot combine order. No live serving source, checkpoint, image or GPU job changed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run_checks()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["checks"]))


if __name__ == "__main__":
    main()
