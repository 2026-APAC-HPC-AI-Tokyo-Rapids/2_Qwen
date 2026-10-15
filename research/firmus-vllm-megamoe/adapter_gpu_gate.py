"""Bounded actual vLLM adapter math/graph/workspace gate; no serving score."""

import argparse
import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS

from gpu_gate import errors, independent_reference, save

COUNTS = ((1, 1, 1, 1), (128, 0, 17, 64), (128, 128, 128, 128), (4096, 4096, 4096, 4096))
REPLAYS = 4


def run(output):
    os.environ["VLLM_USE_DEEP_GEMM_E8M0"] = "0"
    import torch
    import torch.distributed as dist
    import vllm
    from vllm.distributed.parallel_state import (
        GroupCoordinator,
        destroy_distributed_environment,
        init_distributed_environment,
    )
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
    from vllm.model_executor.layers.quantization import apac_sm90 as adapter
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 4 and int(os.environ["RANK"]) == rank
    assert torch.cuda.device_count() == 4
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    props = torch.cuda.get_device_properties(device)
    assert "H200" in props.name and (props.major, props.minor) == (9, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=300), device_id=device)
    # Native communicator setup reads vLLM world/node state, not only Torch PG.
    init_distributed_environment(world_size=4, rank=rank, local_rank=rank, backend="nccl", timeout=timedelta(seconds=300))
    group = GroupCoordinator([list(range(4))], rank, "nccl", use_device_communicator=True, group_name="tp")
    manifest = json.loads(Path("/opt/apac/vllm-sm90-adapter.json").read_text())
    assert manifest["adapter_sha256"] == hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest()
    for name, digest in manifest["patched_sha256"].items():
        assert hashlib.sha256((Path(vllm.__file__).parent / name).read_bytes()).hexdigest() == digest
    # The actual method and native weight registration run unchanged. Only the
    # outer serving-config/group lookup is supplied by this standalone process.
    moe_parallel = FusedMoEParallelConfig(tp_size=1, tp_rank=0, dp_size=4, dp_rank=rank, ep_size=4, ep_rank=rank, pcp_size=1, pcp_rank=0, sp_size=1, use_ep=True, all2all_backend="allgather_reducescatter", enable_eplb=False)
    moe = FusedMoEConfig(num_experts=128, experts_per_token=8, hidden_dim=4096, intermediate_size=1536, num_local_experts=32, num_logical_experts=128, activation=MoEActivation.SILU, device=device, routing_method=RoutingMethodType.RenormalizeNaive, moe_parallel_config=moe_parallel, in_dtype=torch.bfloat16, moe_backend="apac_sm90_megamoe", max_num_tokens=4096)
    parallel = NS(tensor_parallel_size=1, data_parallel_size=4, enable_expert_parallel=True, enable_eplb=False, enable_dbo=False, expert_placement_strategy="linear")
    config = NS(parallel_config=parallel, model_config=NS(hf_config=NS(model_type="qwen3_moe")), scheduler_config=NS(max_num_batched_tokens=4096))
    quant = Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", weight_block_size=[128, 128])
    layer = torch.nn.Module()
    layer.moe_config = moe
    layer.layer_name = "adapter-gate"
    layer.renormalize = True
    layer.use_grouped_topk = False
    layer.scoring_func = "softmax"
    layer.custom_routing_function = None
    layer.apply_router_weight_on_input = False
    layer.routed_scaling_factor = 1.0
    adapter.get_current_vllm_config = lambda: config
    adapter.get_ep_group = lambda: group
    method = adapter.Sm90MegaMoEMethod(quant, layer)
    from vllm import distributed as distributed_context

    original_tp_size = distributed_context.get_tensor_model_parallel_world_size
    try:
        distributed_context.get_tensor_model_parallel_world_size = lambda: 1
        with torch.device(device):
            method.create_weights(layer, 32, 4096, 1536, torch.bfloat16)
    finally:
        distributed_context.get_tensor_model_parallel_world_size = original_tp_size
    torch.manual_seed(713 + rank)
    with torch.no_grad():
        layer.w13_weight.copy_((torch.randn(layer.w13_weight.shape, device=device, dtype=torch.bfloat16) * 4).to(torch.float8_e4m3fn))
        layer.w2_weight.copy_((torch.randn(layer.w2_weight.shape, device=device, dtype=torch.bfloat16) * 4).to(torch.float8_e4m3fn))
        layer.w13_weight_scale_inv.copy_(torch.rand(layer.w13_weight_scale_inv.shape, device=device) * 0.005 + 0.005)
        layer.w2_weight_scale_inv.copy_(torch.rand(layer.w2_weight_scale_inv.shape, device=device) * 0.005 + 0.005)
    # Independent un-interleaved reference tensors; production method drops raw L1.
    raw = [getattr(layer, name).detach().clone() for name in ("w13_weight", "w13_weight_scale_inv", "w2_weight", "w2_weight_scale_inv")]
    method.process_weights_after_loading(layer)
    assert method.buffer is adapter.workspace(group.device_group)
    torch.manual_seed(713)
    basis = torch.randn(16, 4096, device=device, dtype=torch.bfloat16) * 0.25
    gate = torch.randn(128, 4096, device=device, dtype=torch.bfloat16) * 0.02
    gpus = [None] * 4
    dist.all_gather_object(gpus, {"rank": rank, "uuid": str(props.uuid), "name": props.name})
    assert len({x["uuid"] for x in gpus}) == 4
    result = {"schema": "vllm-sm90-actual-adapter-gate/v1", "completed": False, "correctness_passed": False, "graph_passed": False, "adapter_manifest": manifest, "gpus": gpus, "world_size": 4, "capacity_requested": adapter.CAPACITY, "capacity_aligned": method.buffer.num_max_tokens_per_rank, "workspace_bytes": method.buffer.buffer.numel() * method.buffer.buffer.element_size(), "cases": [], "official_benchmark_score": None, "limitations": "Actual method with standalone config/group; synthetic original-layout weights. Not trained-model accuracy, full-model KV coexistence or throughput proof."}
    if rank == 0:
        save(output, result)
    for counts in COUNTS:
        rows, padded = counts[rank], max(counts)
        x = torch.empty(rows, 4096, dtype=torch.bfloat16, device=device)
        routes = {}
        outputs = {}

        def load(replay, x=x, rows=rows):
            x.copy_(basis[(torch.arange(rows, device=device) + replay + rank * 3) % 16])

        def forward(x=x, routes=routes, outputs=outputs):
            weights, ids, _ = fused_topk(x, x @ gate.T, 8, True, indices_type=torch.int32)
            routes["weights"], routes["ids"] = weights, ids
            outputs["y"] = method.apply(layer, x, weights, ids, None, None)

        load(0)
        for _ in range(3):
            forward()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with group.graph_capture() as context, torch.cuda.graph(graph, stream=context.stream):
            forward()
        torch.cuda.synchronize()
        records = []
        for replay in range(REPLAYS):
            load(replay)
            graph.replay()
            torch.cuda.synchronize()
            weights, ids, _ = fused_topk(x, x @ gate.T, 8, True, indices_type=torch.int32)
            assert torch.equal(routes["ids"], ids) and torch.equal(routes["weights"], weights)
            full = {}
            for key, local in (("x", x), ("ids", ids), ("weights", weights)):
                source = torch.zeros((padded, *local.shape[1:]), dtype=local.dtype, device=device)
                source[:rows].copy_(local)
                target = torch.empty((4 * padded, *local.shape[1:]), dtype=local.dtype, device=device)
                dist.all_gather_into_tensor(target, source, group=group.device_group)
                full[key] = torch.cat([target[i * padded:i * padded + count] for i, count in enumerate(counts)])
            expected = independent_reference(full["x"], full["ids"], full["weights"], raw[0], raw[1], raw[2], raw[3], rank, mega_contract=True)
            start = sum(counts[:rank])
            if rows:
                error = errors(outputs["y"], expected[start:start + rows], torch.ones(rows, dtype=torch.bool, device=device), context={"backend": "actual-adapter", "counts": counts, "rank": rank, "replay": replay})
            else:
                assert outputs["y"].shape == (0, 4096)
                error = {"empty_rank_participated": True}
            gathered = [None] * 4
            dist.all_gather_object(gathered, {"rank": rank, "replay": replay, "rows": rows, "errors": error})
            records.extend(gathered)
        result["cases"].append({"counts": counts, "records": records})
        if rank == 0:
            save(output, result)
        del graph
        torch.cuda.synchronize()
        dist.barrier()
    result.update(completed=True, correctness_passed=True, graph_passed=True)
    if rank == 0:
        save(output, result)
    dist.barrier()
    group.destroy()
    destroy_distributed_environment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
