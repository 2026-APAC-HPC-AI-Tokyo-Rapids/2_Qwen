"""Check the new FC2 activation boundary in the existing numeric-gate step.

No scheduler submission. Compare against the actual unfused CUDA operations,
including exact FP8 bytes/scales and CUDA Graph replay with changed inputs.
"""
import torch

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.decode_silu_quant import decode_silu_quant
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8


def check_decode_silu_quant():
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.manual_seed(381108)
    for rows in (0, 1, 7, 8, 31, 32, 257, 1024, 1536, 4096, 6144):
        x = torch.randn(rows, 768, device="cuda", dtype=torch.bfloat16) * 4
        if rows:
            x[0].zero_()
        expected_activation = torch.empty((rows, 384), device="cuda", dtype=torch.bfloat16)
        if rows:
            silu_and_mul(x, expected_activation)
        expected, scales = sglang_per_token_group_quant_fp8(expected_activation, 128)
        actual, actual_scales = decode_silu_quant(x)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), (rows, "FP8 bytes")
        assert torch.equal(actual_scales, scales), (rows, "FP32 scales")

    x = torch.randn(1024, 768, device="cuda", dtype=torch.bfloat16)
    # Warmup on a side stream before graph capture; every buffer is graph-owned.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode_silu_quant(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, actual_scales = decode_silu_quant(x)
    x.copy_(torch.randn_like(x) * 8)
    graph.replay()
    expected, scales = sglang_per_token_group_quant_fp8(silu_and_mul(x), 128)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), "replay FP8 bytes"
    assert torch.equal(actual_scales, scales), "replay scales"
    print("PASS: Decode SiLU/group128 FP8 exact bytes/scales; empty/tail rows; graph replay", flush=True)


if __name__ == "__main__":
    check_decode_silu_quant()
