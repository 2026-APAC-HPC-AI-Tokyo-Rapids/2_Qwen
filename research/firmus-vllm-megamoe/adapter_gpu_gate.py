"""Bounded actual vLLM adapter math/graph/workspace gate; no serving score."""

import argparse
import hashlib
import json
import os
import traceback
from datetime import timedelta
from pathlib import Path

from gpu_gate import errors, independent_reference, save

COUNTS = ((1, 1, 1, 1), (128, 0, 17, 64), (128, 128, 128, 128), (4096, 4096, 4096, 4096))
REPLAYS = 4
PATTERNS = ("random", "hot_rank0", "boundary_masked", "all_masked")


def run(output):
    os.environ["VLLM_USE_DEEP_GEMM_E8M0"] = "0"
    import torch
    import torch.distributed as dist
    import vllm
    from vllm.config import DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig, set_current_vllm_config
    from vllm.config.kernel import KernelConfig
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_dp_group,
        get_ep_group,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
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
    config = VllmConfig(
        model_config=ModelConfig(model="/opt/apac/firmus-sm90-megamoe/qwen3-config", dtype="bfloat16", max_model_len=8192, enforce_eager=True, skip_tokenizer_init=True),
        parallel_config=ParallelConfig(tensor_parallel_size=1, data_parallel_size=4, data_parallel_size_local=4, data_parallel_rank=rank, enable_expert_parallel=True, distributed_executor_backend="external_launcher", all2all_backend="allgather_reducescatter", disable_custom_all_reduce=True),
        scheduler_config=SchedulerConfig(max_model_len=8192, is_encoder_decoder=False, max_num_batched_tokens=4096, max_num_seqs=48),
        device_config=DeviceConfig(device="cuda"),
        kernel_config=KernelConfig(moe_backend="apac_sm90_megamoe"),
    )
    with set_current_vllm_config(config):
        dist.init_process_group("nccl", timeout=timedelta(seconds=300), device_id=device)
        # Native communicator setup reads vLLM world/node state, not only Torch PG.
        init_distributed_environment(world_size=4, rank=rank, local_rank=rank, backend="nccl", timeout=timedelta(seconds=300))
        initialize_model_parallel(tensor_model_parallel_size=1, backend="nccl")
        group = get_ep_group()
        assert get_tp_group().ranks == [rank]
        assert get_dp_group().ranks == group.ranks == list(range(4))
        manifest = json.loads(Path("/opt/apac/vllm-sm90-adapter.json").read_text())
        assert manifest["adapter_sha256"] == hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest()
        for name, digest in manifest["patched_sha256"].items():
            assert hashlib.sha256((Path(vllm.__file__).parent / name).read_bytes()).hexdigest() == digest
        quant = Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", weight_block_size=[128, 128])
        with torch.device(device):
            runner = FusedMoEFactory(128, 8, 4096, 1536, params_dtype=torch.bfloat16, quant_config=quant, prefix="native_gpu_gate")
        layer = runner.routed_experts
        method = layer.quant_method
        assert isinstance(method, adapter.Sm90MegaMoEMethod)
        assert not runner.do_naive_dispatch_combine and runner._fused_output_is_reduced
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
        result = {"schema": "vllm-sm90-actual-adapter-gate/v1", "completed": False, "correctness_passed": False, "graph_passed": False, "native_factory_runner": True, "parallelism": {"tp": 1, "dp": 4, "ep": 4}, "adapter_manifest": manifest, "gpus": gpus, "world_size": 4, "capacity_requested": adapter.CAPACITY, "capacity_aligned": method.buffer.num_max_tokens_per_rank, "workspace_bytes": method.buffer.buffer.numel() * method.buffer.buffer.element_size(), "cases": [], "official_benchmark_score": None, "limitations": "Native factory/router/runner and TP1/DP4/EP4; synthetic original-layout weights. Not trained-model accuracy, full-model KV coexistence or throughput proof."}
        if rank == 0:
            save(output, result)
        for counts in COUNTS:
            live_rows, padded = counts[rank], max(counts)
            # Native topk launches a zero-sized CUDA grid for a physical 0-row
            # tensor. vLLM idle execution uses a dummy batch, so mask one row.
            rows = max(1, live_rows)
            x = torch.empty(rows, 4096, dtype=torch.bfloat16, device=device)
            logits = torch.empty(rows, 128, dtype=torch.float32, device=device)
            padding = torch.zeros(rows, dtype=torch.bool, device=device)
            counts_cpu = torch.tensor([max(1, count) for count in counts], dtype=torch.int32)
            routes = {}
            outputs = {}

            def load(replay, x=x, rows=rows):
                x.copy_(basis[(torch.arange(rows, device=device) + replay + rank * 3) % 16])
                logits.copy_((x @ gate.T).float())
                padding.zero_()
                if replay == 1:
                    logits.fill_(-20)
                    logits[:, :8] = torch.arange(8, device=device, dtype=torch.float32)
                elif replay == 2:
                    logits.fill_(-20)
                    logits[:, [0, 31, 32, 63, 64, 95, 96, 127]] = torch.arange(8, device=device, dtype=torch.float32)
                    padding[::2] = True
                elif replay == 3:
                    padding.fill_(True)
                if live_rows == 0:
                    padding.fill_(True)

            def forward(x=x, routes=routes, outputs=outputs):
                with set_forward_context(None, config, num_tokens=rows, num_tokens_across_dp=counts_cpu, is_padding=padding):
                    weights, ids, _ = fused_topk(x, logits, 8, True, indices_type=torch.int32)
                    routes["weights"], routes["ids"] = weights, ids
                    outputs["y"] = runner(x, logits)

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
                graph_output = outputs["y"].clone()
                with set_forward_context(None, config, num_tokens=rows, num_tokens_across_dp=counts_cpu, is_padding=padding):
                    weights, ids, _ = fused_topk(x, logits, 8, True, indices_type=torch.int32)
                    eager_output = runner(x, logits)
                assert torch.equal(routes["ids"], ids) and torch.equal(routes["weights"], weights)
                if rows:
                    errors(graph_output, eager_output, ~padding, context={"comparison": "graph/eager", "counts": counts, "rank": rank, "replay": replay})
                full = {}
                for key, local in (("x", x), ("ids", ids), ("weights", weights)):
                    source = torch.zeros((padded, *local.shape[1:]), dtype=local.dtype, device=device)
                    source[:live_rows].copy_(local[:live_rows])
                    target = torch.empty((4 * padded, *local.shape[1:]), dtype=local.dtype, device=device)
                    dist.all_gather_into_tensor(target, source, group=group.device_group)
                    full[key] = torch.cat([target[i * padded:i * padded + count] for i, count in enumerate(counts)])
                expected = independent_reference(full["x"], full["ids"], full["weights"], raw[0], raw[1], raw[2], raw[3], rank, mega_contract=True)
                start = sum(counts[:rank])
                if live_rows:
                    error = errors(graph_output, expected[start:start + live_rows], ~padding, context={"backend": "actual-adapter", "counts": counts, "rank": rank, "replay": replay})
                else:
                    assert graph_output.shape == (1, 4096) and torch.count_nonzero(graph_output).item() == 0
                    error = {"empty_rank_participated": True}
                gathered = [None] * 4
                dist.all_gather_object(gathered, {"rank": rank, "replay": replay, "pattern": PATTERNS[replay], "eager_compared": True, "rows": live_rows, "physical_rows": rows, "errors": error})
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
        destroy_model_parallel()
        destroy_distributed_environment()


def main(output):
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        save(output, {"schema": "vllm-sm90-actual-adapter-gate/v1", "completed": False, "correctness_passed": False, "graph_passed": False, "phase": "initialization", "cases": [], "official_benchmark_score": None})
    try:
        run(output)
    except BaseException:
        # No collective here: one rank can fail before the others initialize.
        error = {"rank": rank, "traceback": traceback.format_exc()}
        save(output.with_name(f"{output.stem}.rank{rank}.error.json"), error)
        if rank == 0:
            result = json.loads(output.read_text())
            result.update(completed=False, correctness_passed=False, graph_passed=False, error=error)
            save(output, result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
