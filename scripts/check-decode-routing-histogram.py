#!/usr/bin/env python3
"""CPU interpreter checks of the actual graph-captured Triton histogram.

Run with uv run --no-project --python 3.12 --with triton==3.6.0 --with numpy
python scripts/check-decode-routing-histogram.py. No torch or GPU is used.
"""
import importlib.util
import os
import tempfile
from pathlib import Path

os.environ["TRITON_INTERPRET"] = "1"
import numpy as np
import triton.language as tl
from triton.runtime.interpreter import TensorHandle

ROOT = Path(__file__).resolve().parents[1]


def kernel_source():
    patch = (ROOT / "patches/sglang/decode-routing-capture.patch").read_text()
    new_file = patch.split("+++ b/python/sglang/srt/layers/moe/decode_routing_capture.py\n", 1)[1]
    source = "\n".join(line[1:] for line in new_file.splitlines() if line.startswith("+"))
    begin = source.index("@triton.jit\n")
    end = source.index("\n\nclass CaptureState:", begin)
    return "import triton\nimport triton.language as tl\n" + source[begin:end] + "\n"


def pointer(array):
    dtype = tl.pointer_type(tl.int64 if array.dtype == np.int64 else tl.int32)
    return tl.tensor(TensorHandle(np.array([array.ctypes.data], dtype=np.uint64), dtype), dtype)


def main():
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "histogram.py"
        path.write_text(kernel_source())
        spec = importlib.util.spec_from_file_location("histogram", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = np.random.default_rng(20261025)
        cases = 0
        for dtype in (np.int32, np.int64):
            for rows in (0, 1, 128, 160, 176, 192):
                for live in sorted({0, rows // 2, rows}):
                    ids = rng.integers(0, 128, size=(rows, 8), dtype=dtype)
                    if rows:
                        ids[0, 0] = -1
                    for device_live in (False, True):
                        live_array = np.array([live], dtype=np.int32)
                        gate = np.array([0], dtype=np.int32)
                        output = np.full(130, -9, dtype=np.int32)
                        args = (pointer(ids), pointer(live_array) if device_live else live, pointer(gate), pointer(output), ids.size, 8, device_live, max(128, 1 << max(0, ids.size - 1).bit_length()))
                        module._histogram[(1,)](*args)
                        assert np.all(output == -9), "Disabled sample wrote output"
                        gate[0] = 1
                        module._histogram[(1,)](*args)
                        valid = ids[:live].reshape(-1)
                        valid = valid[(valid >= 0) & (valid < 128)]
                        np.testing.assert_array_equal(output[:128], np.bincount(valid, minlength=128))
                        assert output[128] == live and output[129] == valid.size
                        # Replay with changed inputs must overwrite, not accumulate.
                        ids.fill(127)
                        module._histogram[(1,)](*args)
                        assert output[127] == live * 8 and output[:127].sum() == 0
                        cases += 1
        print(f"PASS: {cases} actual Triton CPU cases: gating, idle, live padding, invalid IDs, changed-input replay")


if __name__ == "__main__":
    main()
