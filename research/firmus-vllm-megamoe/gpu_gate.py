"""Finite four-H200 SM90 MegaMoE correctness/graph/full-path diagnostic.

Run with torchrun --standalone --nproc_per_node=4 gpu_gate.py --output PATH.
No scheduler, model server, checkpoint, or official benchmark is started here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

SCHEMA = "vllm-sm90-megamoe-gpu-gate/v1"
ROWS = (1, 4, 7, 32, 97, 128, 288, 324, 384, 768)
PATTERNS = ("random", "hot", "empty_tail", "masked")
TIMING_ROWS = (128, 288, 324, 768)
REPLAYS = 4
WARMUPS = 20
ITERATIONS = 100
NATIVE_HASHES = {
    "model_executor/layers/fused_moe/config.py": "a3a87cebcc7ac85d73f471dcfafc2d899c6fc08f264777b74a0e919adac9e03f",
    "model_executor/layers/fused_moe/experts/triton_moe.py": "806fbccfe30f5fbd578acf773add99bdf022b12d617958bd6281aef6de369701",
    "model_executor/layers/fused_moe/fused_moe.py": "7072eb06237be9d33dcb0ef7101410f886a6363c98cbee70a014c68b70f639cb",
    "model_executor/layers/fused_moe/modular_kernel.py": "8dc4e0dbbf35b3aac8add63ca6f2a7bbfdadaabbd60826e24598b22c7d8e5c22",
    "model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py": "adbbf529cd22059c95b5380952943a70cf11e05e321dcd707f296bebde61a6e7",
    "model_executor/layers/fused_moe/router/fused_topk_router.py": "1c80f4af4d781fd66b12aa6a7a0fd30082bc521df40f1252987a8ac4c0bef51c",
    "distributed/parallel_state.py": "d9b10ad56833ebc08a753e1a050ef0989c30d4134021b7b7b00150b7bd01adc6",
    "distributed/device_communicators/cuda_communicator.py": "102d95c67ec798cf352b817ba92e8abc9da37f30d1bbdf3aab45d3927b627b93",
}


def layout(rows: int, pattern: str, replay: int = 0) -> dict:
    """Host-known fixed-shape row ownership; never reads GPU counts."""
    if rows not in ROWS or pattern not in PATTERNS or replay not in range(REPLAYS):
        raise ValueError("Unsupported finite diagnostic case")
    shard = (rows + 3) // 4
    padded = shard * 4
    indices = [(row + replay) % 16 for row in range(padded)]
    live = [row < rows for row in range(padded)]
    for row in range(padded):
        rank, local = divmod(row, shard)
        if pattern == "empty_tail":
            limit = (
                (0 if replay % 2 == 0 else shard)
                if rank == 2
                else max(0, shard - replay * 7)
            )
            live[row] = live[row] and local < limit
        elif pattern == "masked":
            live[row] = live[row] and (row + replay) % 3 != 0
    return {
        "rows": rows,
        "shard": shard,
        "padded": padded,
        "indices": indices,
        "live": live,
        "live_per_rank": [
            sum(live[rank * shard : (rank + 1) * shard]) for rank in range(4)
        ],
    }


# These two equations are unchanged from Komori bb77e51 scripts/check-firmus-megamoe.py.
def quantized(x, group):
    import torch

    blocks = x.float().reshape(x.shape[0], -1, group)
    scale = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0
    return ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).reshape(x.shape)


def dequantized(w, scale):
    return w.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)


def independent_reference(
    hidden, ids, weights, w13, s13, w2, s2, rank, *, mega_contract
):
    """Reuse Komori's equations/tolerances; native group128 contract is explicit."""
    import torch
    import torch.distributed as dist

    x = quantized(hidden, 128)
    reference = torch.zeros(hidden.shape[0], 4096, device=hidden.device)
    for local_expert in range(32):
        rows, slots = torch.where(ids == rank * 32 + local_expert)
        if rows.numel() == 0:
            continue
        fc1 = x[rows] @ dequantized(w13[local_expert], s13[local_expert]).T
        if not mega_contract:
            # Actual Triton FC1 writes its BF16 intermediate_cache1 before
            # fused float SiLU+Mul/group128 quantization. Mega differs here.
            fc1 = fc1.to(torch.bfloat16).float()
        gate, up = fc1.chunk(2, dim=-1)
        activation = torch.nn.functional.silu(gate) * up
        out = (
            quantized(activation, 64 if mega_contract else 128)
            @ dequantized(w2[local_expert], s2[local_expert]).T
        )
        if mega_contract:
            reference.index_add_(
                0, rows, out.to(torch.bfloat16).float() * weights[rows, slots, None]
            )
        else:
            reference.index_add_(
                0, rows, (out * weights[rows, slots, None]).to(torch.bfloat16).float()
            )
    if not mega_contract:
        reference = reference.to(torch.bfloat16)
    dist.all_reduce(reference)
    return reference.to(torch.bfloat16)


def errors(observed, expected, live) -> dict:
    import torch

    if not torch.isfinite(observed).all():
        raise AssertionError("Non-finite output")
    observed, expected = observed.float(), expected.float()
    relative_l2 = (
        (observed - expected).norm() / expected.norm().clamp_min(1e-10)
    ).item()
    maximum = (
        (observed - expected).abs().max() / expected.abs().max().clamp_min(1e-10)
    ).item()
    zero_ok = bool(torch.count_nonzero(observed[~live]).item() == 0)
    if relative_l2 >= 0.025 or maximum >= 0.08 or not zero_ok:
        raise AssertionError(
            {
                "relative_l2": relative_l2,
                "max_scaled_error": maximum,
                "masked_rows_zero": zero_ok,
            }
        )
    return {
        "relative_l2": relative_l2,
        "max_scaled_error": maximum,
        "masked_rows_zero": zero_ok,
    }


class Pipeline:
    def __init__(
        self,
        lib,
        group,
        mega,
        weights,
        gate,
        basis,
        rows,
        pattern,
        native_experts,
        prepare,
    ):
        self.lib, self.group, self.mega = lib, group, mega
        self.weights, self.gate, self.basis = weights, gate, basis
        self.rows, self.pattern = rows, pattern
        self.experts, self.prepare = native_experts, prepare
        torch = lib.torch
        self.rank = lib.dist.get_rank()
        self.shard = (rows + 3) // 4
        self.padded = self.shard * 4
        self.full = torch.empty(
            self.padded, 4096, dtype=torch.bfloat16, device=gate.device
        )
        self.local = torch.empty(
            self.shard, 4096, dtype=torch.bfloat16, device=gate.device
        )
        self.masks = torch.empty(self.padded, 8, dtype=torch.bool, device=gate.device)
        self.bias = torch.zeros(128, dtype=torch.bfloat16, device=gate.device)
        if pattern == "hot":
            self.bias.fill_(-100)
            self.bias[:8] = 100
        self.native_output = torch.empty_like(self.full)
        shapes = self.experts.workspace_shapes(
            self.padded, 3072, 4096, 8, 128, 32, None, lib.activation
        )
        self.workspace13 = torch.empty(
            shapes[0], dtype=torch.bfloat16, device=gate.device
        )
        self.workspace2 = torch.empty(
            shapes[1], dtype=torch.bfloat16, device=gate.device
        )
        self.local_output = torch.empty_like(self.local)
        self.native_result = self.mega_result = None
        self.start = self.rank * self.shard
        self.stop = self.start + self.shard
        if not group.device_communicator.ca_comm.should_custom_ar(self.native_output):
            raise RuntimeError(
                "Native comparator would not use ordinary custom all-reduce for this shape"
            )
        self.load(0)

    def load(self, replay):
        torch = self.lib.torch
        info = layout(self.rows, self.pattern, replay)
        indices = torch.tensor(
            info["indices"], dtype=torch.int64, device=self.full.device
        )
        self.live = torch.tensor(
            info["live"], dtype=torch.bool, device=self.full.device
        )
        self.full.copy_(self.basis[indices])
        self.masks.copy_((~self.live)[:, None].expand(-1, 8))
        self.masks[:, -1].logical_or_(indices == 0)
        self.indices = indices
        self.info = info

    def route(self, hidden, masks):
        logits = self.lib.torch.nn.functional.linear(hidden, self.gate) + self.bias
        weights, ids, _ = self.lib.fused_topk(
            hidden, logits, 8, True, indices_type=self.lib.torch.int32
        )
        ids.masked_fill_(masks, -1)
        return weights, ids

    def native_forward(self, marks=None):
        if marks:
            marks[0].record()
        weights, ids = self.route(self.full, self.masks)
        self.native_routes = (weights, ids)
        if marks:
            marks[1].record()
        a1q, scale, _, _, _ = self.prepare.prepare(
            self.full,
            weights,
            ids,
            128,
            self.weights.expert_map,
            False,
            self.experts.quant_config,
            defer_input_quant=False,
        )
        if marks:
            marks[2].record()
        self.experts.apply(
            self.native_output,
            a1q,
            self.weights.w13,
            self.weights.w2,
            weights,
            ids,
            self.lib.activation,
            128,
            self.weights.expert_map,
            scale,
            None,
            self.workspace13,
            self.workspace2,
            None,
            False,
        )
        self.prepare.finalize(
            self.native_output,
            self.native_output,
            weights,
            ids,
            False,
            self.experts.finalize_weight_and_reduce_impl(),
        )
        if marks:
            marks[3].record()
        self.native_result = self.group.all_reduce(self.native_output)
        if marks:
            marks[4].record()
        return self.native_result

    def mega_forward(self, marks=None):
        if marks:
            marks[0].record()
        self.local.copy_(self.full[self.start : self.stop])
        if marks:
            marks[1].record()
        weights, ids = self.route(self.local, self.masks[self.start : self.stop])
        self.mega_routes = (weights, ids)
        if marks:
            marks[2].record()
        buf = self.weights.buffer
        self.mega.mega_moe_pre_dispatch_sm90(
            self.local,
            ids,
            weights,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            num_tokens=self.shard,
            group_size=128,
            routed_scaling_factor=1.0,
        )
        if marks:
            marks[3].record()
        self.mega.fp8_mega_moe(
            self.local_output,
            self.weights.l1,
            self.weights.l2,
            buf,
            recipe=(128, 128, 128),
            activation="swiglu",
            fast_math=True,
        )
        if marks:
            marks[4].record()
        self.mega_result = self.group.all_gather(self.local_output, dim=0)
        if marks:
            marks[5].record()
        return self.mega_result

    def capture(self, name):
        torch = self.lib.torch
        call = self.native_forward if name == "native" else self.mega_forward
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        self.lib.dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with (
            self.group.graph_capture() as context,
            torch.cuda.graph(graph, stream=context.stream),
        ):
            call()
        torch.cuda.synchronize()
        return graph


def sample_summary(values):
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(values),
        "p95_ms": ordered[math.ceil(len(values) * 0.95) - 1],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples": len(values),
    }


def timing(case, graphs):
    torch, dist = case.lib.torch, case.lib.dist
    case.load(0)
    for _ in range(WARMUPS):
        graphs["native"].replay()
        graphs["mega"].replay()
    torch.cuda.synchronize()
    events = {
        name: [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(ITERATIONS)
        ]
        for name in graphs
    }
    wall_start = time.perf_counter()
    for iteration in range(ITERATIONS):
        for name in ("native", "mega") if iteration % 2 == 0 else ("mega", "native"):
            first, last = events[name][iteration]
            first.record()
            graphs[name].replay()
            last.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    local = {
        name: [first.elapsed_time(last) for first, last in pairs]
        for name, pairs in events.items()
    }
    components = {}
    for name, fn, labels in [
        (
            "native",
            case.native_forward,
            (
                "routing",
                "prepare_quant",
                "native_experts_finalize",
                "ordinary_custom_allreduce",
            ),
        ),
        (
            "mega",
            case.mega_forward,
            (
                "split_copy",
                "routing",
                "predispatch",
                "mega_dispatch_experts_combine",
                "output_allgather",
            ),
        ),
    ]:
        marks = [
            [torch.cuda.Event(enable_timing=True) for _ in range(len(labels) + 1)]
            for _ in range(ITERATIONS)
        ]
        for points in marks:
            fn(points)
        torch.cuda.synchronize()
        components[name] = {
            label: [points[index].elapsed_time(points[index + 1]) for points in marks]
            for index, label in enumerate(labels)
        }
    gathered = [None] * 4
    dist.all_gather_object(
        gathered,
        {
            "rank": case.rank,
            "graph_ms": local,
            "eager_component_ms": components,
            "paired_host_wall_seconds": wall_seconds,
        },
    )
    maximum = {
        name: [
            max(rank["graph_ms"][name][i] for rank in gathered)
            for i in range(ITERATIONS)
        ]
        for name in graphs
    }
    summary = {name: sample_summary(values) for name, values in maximum.items()}
    summary["native_over_mega_median_ratio"] = (
        summary["native"]["median_ms"] / summary["mega"]["median_ms"]
    )
    summary["rank_samples"] = gathered
    summary["eager_component_summary_by_rank"] = [
        {
            "rank": rank["rank"],
            "components": {
                name: {label: sample_summary(values) for label, values in parts.items()}
                for name, parts in rank["eager_component_ms"].items()
            },
        }
        for rank in gathered
    ]
    summary["measurement"] = (
        "CUDA events around uninstrumented graph replay; maximum rank per paired iteration. Components are separate eager measurements and are not an engine critical-path attribution."
    )
    return summary


def save(output: Path, result: dict):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def run(output: Path):
    for name in (
        "VLLM_USE_DEEP_GEMM_E8M0",
        "VLLM_ALLREDUCE_USE_SYMM_MEM",
        "VLLM_ALLREDUCE_USE_FLASHINFER",
        "NCCL_SYMMEM_ENABLE",
    ):
        if os.environ.get(name, "0") != "0":
            raise ValueError("Unsupported comparator environment: " + name)
        os.environ[name] = "0"
    import torch
    import torch.distributed as dist
    import vllm
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
        fp8_w8a8_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk

    package = Path(vllm.__file__).parent
    for relative, expected in NATIVE_HASHES.items():
        actual = hashlib.sha256((package / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Unsupported native source: " + relative)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4 or rank != local_rank:
        raise ValueError("Requires one node, exactly four local H200 ranks")
    if torch.cuda.device_count() != 4:
        raise ValueError("Exactly the four allocated GPUs must be visible")
    torch.cuda.set_device(local_rank)
    properties = torch.cuda.get_device_properties(local_rank)
    if (properties.major, properties.minor) != (9, 0) or "H200" not in properties.name:
        raise ValueError("Requires H200 SM90")
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=240),
        device_id=torch.device("cuda", local_rank),
    )
    group = GroupCoordinator(
        [list(range(4))],
        local_rank,
        "nccl",
        use_device_communicator=True,
        group_name="tp",
    )
    ca = group.device_communicator.ca_comm
    if ca is None or ca.disabled:
        raise RuntimeError("Ordinary vLLM custom all-reduce must be active")
    helper = Path("/opt/apac/firmus-sm90-megamoe")
    sys.path.insert(0, str(helper))
    provider = importlib.import_module("provider")
    if Path(provider.__file__).resolve() != helper / "provider.py":
        raise RuntimeError("Unexpected provider loader module")
    mega = provider.load_provider(Path("/opt/apac/sm90-provider"))
    torch.manual_seed(713 + rank)
    device = torch.device("cuda", local_rank)
    w13 = (torch.randn(32, 3072, 4096, dtype=torch.bfloat16, device=device) * 4).to(
        torch.float8_e4m3fn
    )
    w2 = (torch.randn(32, 4096, 1536, dtype=torch.bfloat16, device=device) * 4).to(
        torch.float8_e4m3fn
    )
    s13 = torch.rand(32, 24, 32, device=device) * 0.005 + 0.005
    s2 = torch.rand(32, 32, 12, device=device) * 0.005 + 0.005
    l1, l2 = mega.transform_weights_for_mega_moe_sm90((w13, s13), (w2, s2))
    before_workspace = torch.cuda.memory_allocated()
    buffer = mega.get_symm_buffer_for_mega_moe(
        group.device_group,
        128,
        192,
        8,
        4096,
        1536,
        use_fp8_dispatch=True,
        activation="swiglu",
    )
    torch.cuda.synchronize()
    after_workspace = torch.cuda.memory_allocated()
    expert_map = torch.full((128,), -1, dtype=torch.int32, device=device)
    expert_map[rank * 32 : (rank + 1) * 32] = torch.arange(
        32, dtype=torch.int32, device=device
    )
    weights = SimpleNamespace(
        w13=w13,
        w2=w2,
        s13=s13,
        s2=s2,
        l1=l1,
        l2=l2,
        buffer=buffer,
        expert_map=expert_map,
    )
    parallel = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=4,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=rank,
        sp_size=1,
        use_ep=True,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    config = FusedMoEConfig(
        num_experts=128,
        experts_per_token=8,
        hidden_dim=4096,
        intermediate_size=1536,
        num_local_experts=32,
        num_logical_experts=128,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=RoutingMethodType.RenormalizeNaive,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        moe_backend="triton",
        max_num_tokens=768,
    )
    quant = fp8_w8a8_moe_quant_config(s13, s2, block_shape=[128, 128])
    experts = TritonExperts(config, quant)
    prepare = MoEPrepareAndFinalizeNoDPEPModular()
    torch.manual_seed(713)
    basis = torch.randn(16, 4096, dtype=torch.bfloat16, device=device) * 0.25
    gate = torch.randn(128, 4096, dtype=torch.bfloat16, device=device) * 0.02
    lib = SimpleNamespace(
        torch=torch, dist=dist, activation=MoEActivation.SILU, fused_topk=fused_topk
    )
    gpus = [None] * 4
    dist.all_gather_object(
        gpus,
        {
            "rank": rank,
            "local_rank": local_rank,
            "name": properties.name,
            "uuid": str(getattr(properties, "uuid", "unavailable")),
            "total_memory_bytes": properties.total_memory,
        },
    )
    known_uuids = [gpu["uuid"] for gpu in gpus if gpu["uuid"] != "unavailable"]
    if len(known_uuids) != len(set(known_uuids)):
        raise ValueError("GPU UUIDs must be distinct where available")
    memory = [None] * 4
    dist.all_gather_object(
        memory,
        {
            "rank": rank,
            "symmetric_buffer_bytes": buffer.buffer.numel()
            * buffer.buffer.element_size(),
            "torch_workspace_allocation_delta": after_workspace - before_workspace,
            "aligned_tokens_per_rank": buffer.num_max_tokens_per_rank,
            "free_total_bytes": list(torch.cuda.mem_get_info()),
            "decoder_kv_coexistence": "unmeasured; no full target/draft model or KV allocation in this standalone gate",
        },
    )
    result = {
        "schema": SCHEMA,
        "completed": False,
        "correctness_passed": False,
        "graph_passed": False,
        "world_size": 4,
        "gpus": gpus,
        "gpu_visibility": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "SLURM_JOB_GPUS": os.environ.get("SLURM_JOB_GPUS"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        },
        "reference_source_compute": "bb77e51f79789c4861a4f0262d6a777443f98821",
        "reference_source_sha256": "9f7e3fe8c173f11f099eee7b0eb02eb038e90a729011c721d9b8a7bb8893eea3",
        "native_source_sha256": NATIVE_HASHES,
        "private_provider": mega.__name__,
        "memory": memory,
        "cases": [],
        "timings": [],
        "useful_for_serving": None,
        "official_benchmark_score": None,
        "limitations": "Synthetic fixed exact-shape weights; full MoE pipeline diagnostic, not a model quality/official throughput result. Native reference includes BF16 FC1 storage, group128 FC2 input quantization, pre-BF16 router weighting and BF16 per-rank reduction; Mega uses original group64/late-FP32 contract. References are tolerance-based, not a bitwise native-kernel emulation. KV coexistence and full server integration remain open.",
    }
    if rank == 0:
        save(output, result)
    for rows in ROWS:
        for pattern in PATTERNS:
            case = Pipeline(
                lib, group, mega, weights, gate, basis, rows, pattern, experts, prepare
            )
            graphs = {"native": case.capture("native"), "mega": case.capture("mega")}
            records = []
            for replay in range(REPLAYS):
                case.load(replay)
                graphs["native"].replay()
                graphs["mega"].replay()
                torch.cuda.synchronize()
                # Recompute each backend's routing with its exact GEMM shape.
                # Shape-dependent BF16 gate GEMM rounding may choose different
                # boundary routes; it must not cause a false kernel rejection.
                nw, ni = case.route(case.full, case.masks)
                mw, mi = case.route(
                    case.full[case.start : case.stop].contiguous(),
                    case.masks[case.start : case.stop],
                )
                for actual_routes, expected_routes in (
                    (case.native_routes, (nw, ni)),
                    (case.mega_routes, (mw, mi)),
                ):
                    if not torch.equal(
                        actual_routes[1], expected_routes[1]
                    ) or not torch.equal(actual_routes[0], expected_routes[0]):
                        raise AssertionError(
                            "Changed-input graph routing differs from exact-shape eager routing"
                        )
                mega_ids = torch.empty(case.padded, 8, dtype=torch.int32, device=device)
                mega_weights = torch.empty(
                    case.padded, 8, dtype=torch.float32, device=device
                )
                dist.all_gather_into_tensor(mega_ids, mi, group=group.device_group)
                dist.all_gather_into_tensor(mega_weights, mw, group=group.device_group)
                expected_mega = independent_reference(
                    case.full,
                    mega_ids,
                    mega_weights,
                    w13,
                    s13,
                    w2,
                    s2,
                    rank,
                    mega_contract=True,
                )
                expected_native = independent_reference(
                    case.full, ni, nw, w13, s13, w2, s2, rank, mega_contract=False
                )
                active = case.live[:, None].expand(-1, 8)
                active_slots = int(active.sum().item())
                route_id_differences = int(((mega_ids != ni) & active).sum().item())
                weight_delta = (mega_weights - nw).abs()
                routing_equivalence = {
                    "live_slots": active_slots,
                    "ordered_id_different_slots": route_id_differences,
                    "ordered_id_different_fraction": route_id_differences / active_slots
                    if active_slots
                    else 0.0,
                    "weight_different_slots": int(
                        ((weight_delta != 0) & active).sum().item()
                    ),
                    "max_abs_weight_difference": weight_delta[active].max().item()
                    if active_slots
                    else 0.0,
                    "mean_abs_weight_difference": weight_delta[active].mean().item()
                    if active_slots
                    else 0.0,
                    "interpretation": "Recorded without an acceptance threshold; route equivalence and model quality require subsequent official task validation.",
                }
                records.append(
                    {
                        "replay": replay,
                        "routing_equivalence": routing_equivalence,
                        "live_per_rank": case.info["live_per_rank"],
                        "mega": errors(case.mega_result, expected_mega, case.live),
                        "native": errors(
                            case.native_result, expected_native, case.live
                        ),
                    }
                )
            rank_records = [None] * 4
            dist.all_gather_object(rank_records, {"rank": rank, "checks": records})
            result["cases"].append(
                {
                    "rows": rows,
                    "pattern": pattern,
                    "shard_rows": case.shard,
                    "all_rank_checks": rank_records,
                }
            )
            if pattern == "random" and rows in TIMING_ROWS:
                result["timings"].append({"rows": rows, **timing(case, graphs)})
            if rank == 0:
                save(output, result)
                print(
                    json.dumps(
                        {"stage": "case_passed", "rows": rows, "pattern": pattern}
                    ),
                    flush=True,
                )
            del graphs, case
            gc.collect()
    peak = [None] * 4
    dist.all_gather_object(
        peak,
        {
            "rank": rank,
            "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
            "free_total_bytes": list(torch.cuda.mem_get_info()),
        },
    )
    result.update(
        completed=True, correctness_passed=True, graph_passed=True, peak_memory=peak
    )
    result["timing_advantage"] = all(
        item["native_over_mega_median_ratio"] > 1.0 for item in result["timings"]
    )
    result["next_gate"] = (
        "Review timing and actual decoder memory budget before any serving adapter; a valid negative timing result is a completed experiment."
    )
    if rank == 0:
        save(output, result)
        print(
            "PASS: finite native/MegaMoE correctness, changed-input graphs and complete pipeline timing",
            flush=True,
        )
    buffer.destroy()
    group.destroy()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.output.resolve())
    except Exception as error:
        if os.environ.get("RANK", "0") == "0":
            existing = (
                json.loads(args.output.read_text())
                if args.output.exists()
                else {
                    "schema": SCHEMA,
                    "completed": False,
                    "correctness_passed": False,
                    "graph_passed": False,
                    "world_size": 4,
                }
            )
            existing["failure"] = {"type": type(error).__name__, "message": str(error)}
            save(args.output.resolve(), existing)
        raise


if __name__ == "__main__":
    main()
