"""One-layer integration check using the installed SGLang classes and entry points.

No model checkpoint or distributed MoE: the latter has a separate EP4 gate.
Exercise the QKV prologue, cache writes and its FA3/O consumers.
"""
import json
import os
import statistics
from tempfile import TemporaryDirectory
from unittest.mock import patch

# Only compile the exercised shapes for this bounded integration probe.
os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"

import torch
from sgl_kernel.flash_attn import flash_attn_varlen_func
from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args
from sglang.kernels.ops.attention.rope import apply_qknorm_rope_fp8_prefill
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.attention import prefill_fp8
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.layers.quantization.decode_dense_gemm import decode_dense_linear
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    default_cuda_graph_config,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models import qwen3_moe
from sglang.srt.models.qwen3_moe import Qwen3MoeAttention
from sglang.srt.runtime_context import get_context, get_parallel


def equal(actual, expected):
    for a, b in zip(actual, expected):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (a.shape, "not bitwise equal")


def buffers(rows, qh=64, kh=4):
    return [torch.full(shape, 3., device="cuda", dtype=torch.float8_e4m3fn) for shape in ((rows, qh, 128), (rows, kh, 128), (rows, kh, 128), (8256, kh, 128), (8256, kh, 128))]


def invoke(fn, q, k, v, qw, kw, trig, positions, locations, out):
    oq, ck, cv, kc, vc = out
    fn(q, k, qw, kw, trig, positions, 1e-6, oq, v, kc, vc, locations, ck, cv)


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def check_prologue(reference):
    qw, kw = [(torch.randn(128, device="cuda")*.25+1).bfloat16() for _ in range(2)]
    angles = torch.randn(8192, 64, device="cuda")
    trig = torch.cat((angles.cos(), angles.sin()), dim=1)
    for rows, qh, kh, dtype, packed_input in ((0,64,4,torch.int64,True),(1,16,1,torch.int32,False),(31,64,4,torch.int32,True),(33,16,1,torch.int64,False),(129,64,4,torch.int64,True),(3986,64,4,torch.int32,True),(4027,64,4,torch.int64,True),(4096,64,4,torch.int64,True)):
        packed = torch.randn(rows, qh+2*kh, 128, device="cuda", dtype=torch.bfloat16)
        q, k, v = packed.split((qh, kh, kh), dim=1)
        if not packed_input:
            q, k, v = [x.contiguous() for x in (q, k, v)]
        positions = torch.randint(0, 8192, (rows,), device="cuda", dtype=dtype)
        locations = (torch.randperm(8192, device="cuda")[:rows]+1).to(dtype)
        locations[-min(rows,7):] = 0
        expected, actual = buffers(rows,qh,kh), buffers(rows,qh,kh)
        args = (q,k,v,qw,kw,trig,positions,locations)
        invoke(reference, *args, expected)
        invoke(apply_qknorm_rope_fp8_prefill, *args, actual)
        equal(actual,expected)
        if rows in (129,4027):
            graph = capture(lambda args=args, actual=actual: invoke(apply_qknorm_rope_fp8_prefill,*args,actual))
            for _ in range(2):
                packed.neg_()
                positions.copy_(torch.randint_like(positions,0,8192))
                locations.copy_(locations.roll(3))
                for out in (actual,expected):
                    for tensor in out:
                        tensor.fill_(3.)
                invoke(reference,*args,expected)
                graph.replay()
                equal(actual,expected)
        print(f"CHECK production prologue rows={rows} qh={qh} kh={kh} indices={dtype} packed={packed_input} exact=true",flush=True)
    graphs = [capture(lambda fn=fn: invoke(fn,*args,actual)) for fn in (reference,apply_qknorm_rope_fp8_prefill)]
    samples = [[],[]]
    start,end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for repeat in range(25):
        for i in (0,1) if repeat%2==0 else (1,0):
            start.record()
            for _ in range(8):
                graphs[i].replay()
            end.record()
            end.synchronize()
            samples[i].append(start.elapsed_time(end)*1000/8)
    print(f"CHECK production prologue baseline_us={statistics.median(samples[0]):.3f} candidate_us={statistics.median(samples[1]):.3f}",flush=True)


def main():
    torch.manual_seed(42024)
    graph_config = default_cuda_graph_config()
    graph_config.prefill.backend = Backend.DISABLED
    reference = load_jit("qkv_prefill_reference", cuda_files=["elementwise/rope-baseline.cuh"], cuda_wrappers=[("run",f"QKNormRopeFP8Kernel<{make_cpp_args(is_arch_support_pdl())}>::run_prefill")]).run
    check_prologue(reference)
    with get_context().override_server_args(disaggregation_mode="prefill", attention_backend="fa3", kv_cache_dtype="fp8_e4m3", fp8_gemm_runner_backend="deep_gemm", moe_a2a_backend="megamoe", moe_runner_backend="deep_gemm", cuda_graph_config=graph_config, enable_lora=False, enable_fused_qk_norm_rope=False) as args, get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0), torch.device("cuda"), torch.no_grad():
        initialize_moe_config(args)
        initialize_fp8_gemm_config(args)
        model = Qwen3MoeAttention(4096, 64, 4, head_dim=128, quant_config=Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]))
        model.q_norm.to(dtype=torch.bfloat16).weight.fill_(1.)
        model.k_norm.to(dtype=torch.bfloat16).weight.fill_(1.)
        assert model.use_fp8_q_rope and not model.use_fused_qk_norm_rope
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

            packed = expected
            positions = torch.arange(padded, dtype=torch.int64)
            batch.out_cache_loc = torch.arange(1,padded+1,dtype=torch.int64)
            batch.out_cache_loc[rows:] = 0
            reference_out, actual_out = buffers(padded), buffers(padded)
            rq,rk,rv = packed.view(padded,72,128).split((64,4,4),dim=1)
            invoke(reference,rq,rk,rv,model.q_norm.weight,model.k_norm.weight,model.rotary_emb.cos_sin_cache,positions,batch.out_cache_loc,reference_out)
            pool = object.__new__(MHATokenToKVPool)
            batch.token_to_kv_pool = pool
            with patch.object(pool,"get_kv_buffer",return_value=tuple(actual_out[3:])), patch.object(qwen3_moe,"apply_qknorm_rope_fp8_prefill",wraps=apply_qknorm_rope_fp8_prefill) as prologue:
                q,k,v = model.apply_qk_norm_rope(packed,positions,batch)
                assert prologue.call_count==1 and model._stored_fp8_kv_last_call
            q,k,v = [x.view(padded,h,128) for x,h in zip((q,k,v),(64,4,4))]
            equal([q,k,v,*actual_out[3:]],reference_out)
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
            print(json.dumps({"lengths": lengths, "padded_rows": padded, "qwen_qkv_prologue_cache_exact":True, "qwen_radix_fa3_o_exact": True, "qkv_raw_and_tuple_exact": True, "decode_and_graph_fallback_exact": True}), flush=True)
        torch.cuda.synchronize()
    print("PASS production Qwen -> RadixAttention -> FA3 -> O; actual runtime contexts, QKV raw/tuple and fallback paths", flush=True)


if __name__ == "__main__":
    with TemporaryDirectory(prefix="qwen-prefill-probe-") as rendezvous:
        print("Initialize single-rank distributed environment", flush=True)
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method=f"file://{rendezvous}/store", timeout=30)
        print("Initialize model-parallel groups", flush=True)
        initialize_model_parallel()
        try:
            print("Run production entry-point checks", flush=True)
            main()
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
