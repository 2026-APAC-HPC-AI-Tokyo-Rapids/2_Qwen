"""One-layer integration check using the installed SGLang classes and entry points.

No model checkpoint or distributed MoE: the latter has a separate EP4 gate.
Inputs stand in for the already-validated QK-norm/RoPE/KV-store prologue.
"""
import json
from unittest.mock import patch

import torch
from sgl_kernel.flash_attn import flash_attn_varlen_func
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.attention import prefill_fp8
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)
from sglang.srt.layers.quantization.decode_dense_gemm import decode_dense_linear
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    default_cuda_graph_config,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.qwen3_moe import Qwen3MoeAttention
from sglang.srt.runtime_context import get_context, get_parallel


def main():
    torch.manual_seed(42024)
    graph_config = default_cuda_graph_config()
    graph_config.prefill.backend = Backend.DISABLED
    with get_context().override_server_args(disaggregation_mode="prefill", attention_backend="fa3", kv_cache_dtype="fp8_e4m3", fp8_gemm_runner_backend="deep_gemm", cuda_graph_config=graph_config, enable_lora=False) as args, get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0), torch.device("cuda"), torch.no_grad():
        initialize_fp8_gemm_config(args)
        model = Qwen3MoeAttention(4096, 64, 4, head_dim=128, quant_config=Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]))
        for layer in (model.qkv_proj, model.o_proj):
            layer.weight.copy_((torch.randn(layer.weight.shape, dtype=torch.bfloat16) * .25).to(layer.weight.dtype))
            layer.weight_scale_inv.fill_(.01)
            assert prefill_fp8.linear_supported(model, "QKV" if layer is model.qkv_proj else "O", layer)

        backend = object.__new__(FlashAttentionBackend)
        backend.fa_impl_ver = 3
        backend.use_mla = backend.kv_cache_is_mxfp8 = backend.has_local_attention = False
        backend.full_cg_prefill_metadata = None
        backend.kv_cache_dtype_str = "fp8_e4m3"
        backend.kv_cache_dtype = torch.float8_e4m3fn
        backend.attn_cp_size = backend.num_splits = backend.topk = 1
        backend.flash_attn_varlen_func = flash_attn_varlen_func
        # forward_core follows a successful prologue that already wrote KV.
        model._stored_fp8_kv_last_call = True
        model._used_fused_qk_norm_rope_last_call = False

        for lengths, padded in (([17, 13], 32), ([3986], 4096), ([2048, 2048], 4096)):
            rows = sum(lengths)
            batch = ForwardBatch(forward_mode=ForwardMode.EXTEND, batch_size=len(lengths), input_ids=torch.zeros(rows, dtype=torch.int64), req_pool_indices=torch.arange(len(lengths)), seq_lens=torch.tensor(lengths, dtype=torch.int32), out_cache_loc=torch.arange(rows), seq_lens_sum=rows)
            batch.extend_seq_lens_cpu = lengths
            batch.extend_prefix_lens_cpu = [0] * len(lengths)
            batch.mha_return_lse = False
            cu = torch.tensor([0, *torch.tensor(lengths, device="cpu").cumsum(0).tolist()], dtype=torch.int32)
            backend.forward_metadata = FlashAttentionMetadata(cu_seqlens_q=cu, max_seq_len_q=max(lengths))
            hidden = torch.randn(padded, 4096, dtype=torch.bfloat16) * .25
            pair = sglang_per_token_group_quant_fp8(hidden, 128, column_major_scales=True, scale_tma_aligned=True)
            for x in (hidden, pair):
                expected, _ = model.qkv_proj(x)
                actual, _ = decode_dense_linear(model, "QKV", model.qkv_proj, x, batch, decode_worker=False)
                assert torch.equal(actual, expected), "QKV production dispatcher"
                decoded, _ = decode_dense_linear(model, "QKV", model.qkv_proj, x, batch, decode_worker=True)
                assert torch.equal(decoded, expected), "Decode worker fallback"

            q, k, v = [(torch.randn(padded, heads, 128, dtype=torch.bfloat16) * .5).to(torch.float8_e4m3fn) for heads in (64, 4, 4)]
            stock = flash_attn_varlen_func(q=q[:rows], k=k[:rows], v=v[:rows], cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max(lengths), max_seqlen_k=max(lengths), causal=True, softmax_scale=128**-.5, num_splits=1)
            if isinstance(stock, tuple):
                stock = stock[0]
            stock_padded = torch.zeros(padded, 8192, dtype=torch.bfloat16)
            stock_padded[:rows].copy_(stock.reshape(rows, 8192))
            expected, _ = model.o_proj(stock_padded)
            state = (hidden, batch, (q, k, v, batch))
            with forward_context(ForwardContext(backend)), patch.object(prefill_fp8, "attention", wraps=prefill_fp8.attention) as fused:
                actual = model.forward_core(state)
                assert fused.call_count == 1, "Fused path did not execute through Qwen/Radix/FA3"
                assert torch.equal(actual, expected), "Integrated Qwen attention + O output"
                # Graph-enabled configurations must keep the original BF16 ABI.
                graph_config.prefill.backend = Backend.BREAKABLE
                fallback = model.forward_core(state)
                assert fused.call_count == 1, "Graph fallback selected FP8 tuple"
                assert torch.equal(fallback, expected), "Graph fallback"
                graph_config.prefill.backend = Backend.DISABLED
            print(json.dumps({"lengths": lengths, "padded_rows": padded, "qwen_radix_fa3_o_exact": True, "qkv_raw_and_tuple_exact": True, "decode_and_graph_fallback_exact": True}), flush=True)
        torch.cuda.synchronize()
    print("PASS production Qwen -> RadixAttention -> FA3 -> O; actual runtime contexts, QKV raw/tuple and fallback paths", flush=True)


if __name__ == "__main__":
    main()
