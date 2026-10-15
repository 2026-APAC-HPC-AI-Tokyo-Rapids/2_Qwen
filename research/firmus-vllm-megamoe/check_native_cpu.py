"""Exercise installed native groups, MoE factory and checkpoint loader on CPU.

Gloo and CpuPlatform deliberately replace CUDA/NCCL for this diagnostic process.
No method/group lookup is mocked; no GPU math or serving success is claimed.
"""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path


def check(model, output):
    import torch
    import torch.distributed as dist
    import vllm.platforms
    from vllm.platforms.cpu import CpuPlatform

    # The installed wheel is a CUDA wheel. Only this standalone CPU process
    # selects CpuPlatform; the serving image and GPU diagnostic are unchanged.
    vllm.platforms._current_platform = CpuPlatform()
    from vllm.config import (
        DeviceConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
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
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
    from vllm.model_executor.layers.quantization.apac_sm90 import Sm90MegaMoEMethod
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 4
    torch.set_num_threads(1)
    config = VllmConfig(
        model_config=ModelConfig(model=str(model), dtype="bfloat16", max_model_len=8192, enforce_eager=True, skip_tokenizer_init=True),
        parallel_config=ParallelConfig(tensor_parallel_size=1, data_parallel_size=4, data_parallel_size_local=4, data_parallel_rank=rank, enable_expert_parallel=True, distributed_executor_backend="external_launcher", all2all_backend="allgather_reducescatter"),
        scheduler_config=SchedulerConfig(max_model_len=8192, max_num_batched_tokens=4096, max_num_seqs=48),
        device_config=DeviceConfig(device="cpu"),
        kernel_config=KernelConfig(moe_backend="apac_sm90_megamoe"),
    )
    quant = Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", weight_block_size=[128, 128])
    try:
        with set_current_vllm_config(config):
            init_distributed_environment(world_size=4, rank=rank, local_rank=rank, backend="gloo", timeout=timedelta(seconds=120))
            initialize_model_parallel(tensor_model_parallel_size=1, backend="gloo")
            assert get_tp_group().ranks == [rank]
            assert get_dp_group().ranks == get_ep_group().ranks == list(range(4))
            runner = FusedMoEFactory(128, 8, 4096, 1536, params_dtype=torch.bfloat16, quant_config=quant, prefix="native_cpu_probe")
            layer = runner.routed_experts
            assert isinstance(layer.quant_method, Sm90MegaMoEMethod)
            assert not runner.do_naive_dispatch_combine and runner._fused_output_is_reduced
            expected_map = [expert - rank * 32 if rank * 32 <= expert < (rank + 1) * 32 else -1 for expert in range(128)]
            assert layer.expert_map.tolist() == expected_map
            assert config.compilation_config.static_forward_context["native_cpu_probe"] is runner
            for expert, local in enumerate(expected_map):
                if local == -1:
                    assert layer.w13_weight.weight_loader(layer.w13_weight, torch.empty(0), "weight", "w1", expert, return_success=True) is False
                    continue
                for shard, param_name, shape, value in (
                    ("w1", "w13_weight", (1536, 4096), 1 + local % 7),
                    ("w3", "w13_weight", (1536, 4096), -1 - local % 7),
                    ("w2", "w2_weight", (4096, 1536), 2 + rank),
                    ("w1", "w13_weight_scale_inv", (12, 32), (expert + 1) / 256),
                    ("w3", "w13_weight_scale_inv", (12, 32), (expert + 129) / 256),
                    ("w2", "w2_weight_scale_inv", (32, 12), (expert + 257) / 256),
                ):
                    param = getattr(layer, param_name)
                    loaded = torch.full(shape, value, dtype=torch.float32).to(param.dtype)
                    assert param.weight_loader(param, loaded, param_name, shard, expert, return_success=True) is True
                    actual = param[local]
                    if shard in ("w1", "w3"):
                        actual = actual.chunk(2, dim=0)[0 if shard == "w1" else 1]
                    assert torch.equal(actual.view(torch.uint8), loaded.view(torch.uint8)), (expert, shard, param_name)
            total = torch.tensor(rank)
            dist.all_reduce(total, group=get_ep_group().device_group)
            assert total.item() == 6 and not torch.cuda.is_initialized()
            result = {"rank": rank, "tp_ranks": get_tp_group().ranks, "dp_ranks": get_dp_group().ranks, "ep_ranks": get_ep_group().ranks, "loaded_experts": 32, "rejected_nonlocal_experts": 96, "registered_parameters": {name: {"shape": list(param.shape), "dtype": str(param.dtype)} for name, param in layer.named_parameters()}, "native_factory_loader_passed": True, "cuda_initialized": False}
            gathered = [None] * 4
            dist.all_gather_object(gathered, result)
            if rank == 0:
                output.write_text(json.dumps({"passed": True, "transport": "gloo", "platform": "cpu", "gpu_execution": False, "ranks": gathered}, indent=2) + "\n")
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.model, args.output)
