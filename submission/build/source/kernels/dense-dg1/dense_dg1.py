"""Bounded companion ABI; graph allocations belong to the captured call."""
import ctypes
from pathlib import Path

import torch

_LIBRARY = None


def prepare(lhs, rhs, out):
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(str(Path(__file__).with_name("dense-dg1.so")))
        _LIBRARY.dense_dg1.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        _LIBRARY.dense_dg1.restype = ctypes.c_int
    a, sa = lhs
    b, sb = rhs
    rows, k = a.shape
    n = b.shape[0]
    assert 0 < rows <= 176 and (n, k) in ((9216, 4096), (4096, 8192))
    assert a.dtype == b.dtype == torch.float8_e4m3fn
    assert sa.dtype == sb.dtype == torch.float32 and out.dtype == torch.bfloat16
    assert a.is_contiguous() and b.is_contiguous() and sb.is_contiguous() and out.is_contiguous()
    assert sa.shape == (rows, k // 128) and sb.shape == (n // 128, k // 128) and out.shape == (rows, n)
    assert all(t.device == a.device for t in (b, sa, sb, out))
    assert torch.cuda.get_device_capability(a.device) == (9, 0)
    assert torch.cuda.get_device_properties(a.device).multi_processor_count == 132
    bn = 128 if rows <= 128 else 160 if rows <= 160 else 176
    scratch = torch.empty((264, bn, 128), dtype=torch.float32, device=a.device)

    def call():
        status = _LIBRARY.dense_dg1(a.data_ptr(), b.data_ptr(), sa.data_ptr(), sb.data_ptr(), scratch.data_ptr(), out.data_ptr(), rows, n, k, sa.stride(0), sa.stride(1), torch.cuda.current_stream(a.device).cuda_stream)
        if status:
            raise RuntimeError(f"dense DG1 CUDA error {status}")

    return call


def run(lhs, rhs, out):
    prepare(lhs, rhs, out)()
