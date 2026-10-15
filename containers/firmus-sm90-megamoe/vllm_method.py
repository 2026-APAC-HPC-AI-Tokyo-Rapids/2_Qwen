"""Qwen3-only SM90 MegaMoE method: owns complete EP dispatch and combine."""

import importlib
import sys
from functools import cache
from pathlib import Path

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod
from vllm.model_executor.utils import replace_parameter

logger = init_logger(__name__)
CAPACITY = 4096


@cache
def provider():
    helper = Path("/opt/apac/firmus-sm90-megamoe")
    sys.path.insert(0, str(helper))
    try:
        loader = importlib.import_module("provider")
        if Path(loader.__file__).resolve() != helper / "provider.py":
            raise RuntimeError("Unexpected private SM90 provider loader")
        return loader.load_provider(Path("/opt/apac/sm90-provider"))
    finally:
        sys.path.pop(0)


@cache
def workspace(group):
    # Reuse one symmetric workspace across serialized layers, never one per layer.
    return provider().get_symm_buffer_for_mega_moe(
        group,
        128,
        CAPACITY,
        8,
        4096,
        1536,
        use_fp8_dispatch=True,
        activation="swiglu",
    )


def check_config(quant, layer, config):
    moe = layer.moe_config
    parallel = config.parallel_config
    placement = getattr(
        parallel.expert_placement_strategy, "value", parallel.expert_placement_strategy
    )
    checks = [
        config.model_config.hf_config.model_type == "qwen3_moe",
        parallel.tensor_parallel_size == 1,
        parallel.data_parallel_size == 4,
        parallel.enable_expert_parallel,
        not parallel.enable_eplb,
        not parallel.enable_dbo,
        placement == "linear",
        moe.tp_size == 1 and moe.ep_size == 4 and moe.dp_size == 4,
        moe.pcp_size == 1 and moe.sp_size == 1,
        moe.num_experts == 128 and moe.num_local_experts == 32,
        moe.experts_per_token == 8 and moe.hidden_dim == 4096,
        moe.intermediate_size_per_partition == 1536,
        moe.activation == MoEActivation.SILU,
        not moe.has_bias and not moe.is_lora_enabled and not moe.skip_final_all_reduce,
        moe.in_dtype == torch.bfloat16,
        layer.renormalize and not layer.use_grouped_topk,
        layer.scoring_func == "softmax" and layer.custom_routing_function is None,
        not layer.apply_router_weight_on_input,
        layer.routed_scaling_factor == 1.0,
        quant.is_checkpoint_fp8_serialized,
        quant.activation_scheme == "dynamic" and quant.weight_block_size == [128, 128],
        config.scheduler_config.max_num_batched_tokens <= CAPACITY,
    ]
    if not all(checks):
        raise ValueError(
            "APAC SM90 requires Qwen3 H4096/I1536/E128/top8, TP1/DP4/EP4 linear, BF16/FP8block128, no DBO/EPLB/shared/bias/LoRA, max batch4096"
        )


class Sm90MegaMoEMethod(Fp8MoEMethod):
    supports_pre_processed_weights = False

    def __init__(self, quant_config, layer):
        FusedMoEMethodBase.__init__(self, layer.moe_config)
        check_config(quant_config, layer, get_current_vllm_config())
        self.quant_config = quant_config
        self.weight_block_size = [128, 128]
        self.block_quant = True
        self.weight_scale_name = "weight_scale_inv"
        self.weight_scale_refine = None
        self.moe_block_shape = [128, 128]
        self.packed = None
        self.buffer = None

    @property
    def supports_internal_mk(self):
        # The method owns dispatch/math/combine; runner must not dispatch again.
        return True

    @property
    def output_is_reduced(self):
        return True

    @property
    def supports_eplb(self):
        return False

    @property
    def is_monolithic(self):
        return False

    @property
    def topk_indices_dtype(self):
        return torch.int32

    def maybe_roundup_sizes(self, hidden_size, intermediate_size_per_partition, *_):
        if (hidden_size, intermediate_size_per_partition) != (4096, 1536):
            raise ValueError("Unexpected Qwen3 expert geometry")
        return hidden_size, intermediate_size_per_partition

    def get_fused_moe_quant_config(self, layer):
        return fp8_w8a8_moe_quant_config(
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            block_shape=[128, 128],
        )

    def process_weights_after_loading(self, layer):
        if self.packed is not None:
            raise RuntimeError(
                "Packed MegaMoE weights cannot be reloaded or processed twice"
            )
        if torch.cuda.get_device_capability(layer.w13_weight.device) != (9, 0):
            raise ValueError("APAC MegaMoE requires SM90")
        expected = [
            (layer.w13_weight, (32, 3072, 4096), torch.float8_e4m3fn),
            (layer.w2_weight, (32, 4096, 1536), torch.float8_e4m3fn),
            (layer.w13_weight_scale_inv, (32, 24, 32), torch.float32),
            (layer.w2_weight_scale_inv, (32, 32, 12), torch.float32),
        ]
        for tensor, shape, dtype in expected:
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(
                    "Unexpected original FP8 checkpoint tensor/scale layout"
                )
        mega = provider()
        l1, l2 = mega.transform_weights_for_mega_moe_sm90(
            (layer.w13_weight, layer.w13_weight_scale_inv),
            (layer.w2_weight, layer.w2_weight_scale_inv),
        )
        for name, tensor in [
            ("w13_weight", l1[0]),
            ("w13_weight_scale_inv", l1[1]),
            ("w2_weight", l2[0]),
            ("w2_weight_scale_inv", l2[1]),
        ]:
            replace_parameter(layer, name, tensor)
        self.packed = (l1, l2)
        self.buffer = workspace(get_ep_group().device_group)
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        logger.info(
            "APAC_SM90_MEGAMOE layer=%s ep_rank=%s capacity=%s workspace_bytes=%s output_reduced=true native_dispatch=false",
            layer.layer_name,
            get_ep_group().rank_in_group,
            CAPACITY,
            self.buffer.buffer.numel() * self.buffer.buffer.element_size(),
        )

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if shared_experts is not None or shared_experts_input is not None:
            raise ValueError("Qwen3 MegaMoE does not support shared experts")
        if self.packed is None or self.buffer is None:
            raise RuntimeError(
                "MegaMoE weights/workspace must be initialized before capture"
            )
        rows = x.shape[0]
        if (
            x.ndim != 2
            or x.shape[1] != 4096
            or rows > CAPACITY
            or x.dtype != torch.bfloat16
        ):
            raise ValueError("Unsupported MegaMoE input rows/width/dtype")
        if topk_ids.shape != (rows, 8) or topk_weights.shape != (rows, 8):
            raise ValueError(
                "Native router must supply eight global expert IDs/weights per row"
            )
        x = x.contiguous()
        buf = self.buffer
        mega = provider()
        # Even a zero-live rank must enter both calls in the same layer order.
        mega.mega_moe_pre_dispatch_sm90(
            x,
            topk_ids.to(torch.int32),
            topk_weights.float(),
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            num_tokens=rows,
            group_size=128,
            routed_scaling_factor=1.0,
        )
        out = torch.empty_like(x)
        mega.fp8_mega_moe(
            out,
            self.packed[0],
            self.packed[1],
            buf,
            recipe=(128, 128, 128),
            activation="swiglu",
            fast_math=True,
        )
        return out
