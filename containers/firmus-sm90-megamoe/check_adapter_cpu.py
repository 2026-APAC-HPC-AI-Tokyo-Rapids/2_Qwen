"""Actual-image CPU import and metadata/call-order checks for the adapter."""

import ast
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace as NS


def check():
    import torch
    import vllm
    from vllm.config.kernel import KernelConfig
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    root = Path(vllm.__file__).parent
    manifest = json.loads(Path("/opt/apac/vllm-sm90-adapter.json").read_text())
    for name, expected in manifest["patched_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    module = importlib.import_module(
        "vllm.model_executor.layers.quantization.apac_sm90"
    )
    assert (
        hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        == manifest["adapter_sha256"]
    )
    assert (
        KernelConfig(moe_backend="apac_sm90_megamoe").moe_backend == "apac_sm90_megamoe"
    )
    assert KernelConfig(moe_backend="triton").moe_backend == "triton"
    moe = NS(
        tp_size=1,
        dp_size=4,
        ep_size=4,
        pcp_size=1,
        sp_size=1,
        num_experts=128,
        num_local_experts=32,
        experts_per_token=8,
        hidden_dim=4096,
        intermediate_size_per_partition=1536,
        activation=MoEActivation.SILU,
        has_bias=False,
        is_lora_enabled=False,
        skip_final_all_reduce=False,
        in_dtype=torch.bfloat16,
        w13_num_shards=2,
    )
    parallel = NS(
        tensor_parallel_size=1,
        data_parallel_size=4,
        enable_expert_parallel=True,
        enable_eplb=False,
        enable_dbo=False,
        expert_placement_strategy="linear",
    )
    config = NS(
        parallel_config=parallel,
        model_config=NS(hf_config=NS(model_type="qwen3_moe")),
        scheduler_config=NS(max_num_batched_tokens=4096),
    )
    quant = NS(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    layer = torch.nn.Module()
    layer.moe_config = moe
    layer.renormalize = True
    layer.use_grouped_topk = False
    layer.scoring_func = "softmax"
    layer.custom_routing_function = None
    layer.apply_router_weight_on_input = False
    layer.routed_scaling_factor = 1.0
    original = module.get_current_vllm_config
    original_provider = module.provider
    try:
        module.get_current_vllm_config = lambda: config
        method = module.Sm90MegaMoEMethod(quant, layer)
        assert (
            method.output_is_reduced
            and method.supports_internal_mk
            and not method.is_monolithic
        )
        tree = ast.parse(
            (root / "model_executor/layers/fused_moe/runner/moe_runner.py").read_bytes()
        )
        selected = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name
            in (
                "do_naive_dispatch_combine",
                "_maybe_dispatch",
                "_maybe_combine",
                "_maybe_reduce_final_output",
            )
        ]

        def forbidden(*_args, **_kwargs):
            raise AssertionError("Duplicate native dispatch/combine/reduction")

        namespace = {
            "tensor_model_parallel_all_reduce": forbidden,
            "get_ep_group": forbidden,
            "get_pcp_group": forbidden,
        }
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        exec(  # noqa: S102 -- Selected exact native source, after manifest verification.
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[future, *selected], type_ignores=[])
                ),
                "actual-native-runner-contract",
                "exec",
            ),
            namespace,
        )
        state = NS(
            _quant_method=method,
            moe_config=NS(
                dp_size=4,
                tp_size=1,
                ep_size=4,
                pcp_size=1,
                is_sequence_parallel=False,
                skip_final_all_reduce=False,
            ),
            _fused_output_is_reduced=True,
            shared_experts=None,
        )
        state.do_naive_dispatch_combine = namespace["do_naive_dispatch_combine"].fget(
            state
        )
        assert not state.do_naive_dispatch_combine
        assert namespace["_maybe_dispatch"](state, "rows", "logits") == (
            "rows",
            "logits",
        )
        assert namespace["_maybe_combine"](state, None, "rows") == "rows"
        assert namespace["_maybe_reduce_final_output"](state, "rows", None) == "rows"
        with torch.device("meta"):
            method.create_weights(layer, 32, 4096, 1536, torch.bfloat16)
        assert layer.w13_weight.shape == (32, 3072, 4096)
        assert layer.w2_weight.shape == (32, 4096, 1536)
        assert layer.w13_weight.dtype == torch.float8_e4m3fn
        assert layer.w13_weight_scale_inv.shape == (32, 24, 32)
        assert layer.w2_weight_scale_inv.shape == (32, 32, 12)
        assert layer.w13_weight_scale_inv.dtype == torch.float32
        calls = []

        def pre(*args, **kwargs):
            calls.append(("pre", kwargs["num_tokens"]))
            assert args[1].dtype == torch.int32 and args[2].dtype == torch.float32

        def fused(out, *_args, **_kwargs):
            calls.append(("fused", out.shape[0]))
            out.zero_()

        module.provider = lambda: NS(mega_moe_pre_dispatch_sm90=pre, fp8_mega_moe=fused)
        method.packed = (None, None)
        method.buffer = NS(x=None, x_sf=None, topk_idx=None, topk_weights=None)
        for rows in (0, 1, 128):
            x = torch.ones(rows, 4096, dtype=torch.bfloat16)
            y = method.apply(
                layer,
                x,
                torch.ones(rows, 8),
                torch.zeros(rows, 8, dtype=torch.int64),
                None,
                None,
            )
            assert y.shape == x.shape and torch.count_nonzero(y) == 0
        assert calls == [(kind, n) for n in (0, 1, 128) for kind in ("pre", "fused")]
        parallel.tensor_parallel_size = 4
        try:
            module.Sm90MegaMoEMethod(quant, layer)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported TP4 adapter accepted")
    finally:
        module.get_current_vllm_config = original
        module.provider = original_provider
    assert not torch.cuda.is_initialized()
    return {
        "passed": True,
        "metadata_weights": "full Qwen shapes on meta device",
        "cpu_mock_collective_order": [0, 1, 128],
        "cuda_initialized": False,
        "numerics": "not tested here; requires GPU gate/serving",
    }
