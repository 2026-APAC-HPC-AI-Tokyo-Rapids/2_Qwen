"""CPU specification and overlay checks; GPU checks compare actual CUDA NOSAT."""
import bisect
import json
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "patches/sglang/decode-qknorm-fast-fp8.patch"


def f32(bits):
    return struct.unpack("!f", struct.pack("!I", bits))[0]


def fp8_value(code):
    exponent, mantissa = divmod(code, 8)
    return mantissa * 2**-9 if exponent == 0 else (1 + mantissa / 8) * 2**(exponent - 7)


VALUES = [fp8_value(code) for code in range(127)]


def nearest_even(value):
    index = bisect.bisect_left(VALUES, value)
    if index == 0:
        return 0
    if index == len(VALUES):
        return 126
    low, high = value - VALUES[index - 1], VALUES[index] - value
    return index - 1 if low < high or (low == high and index % 2) else index


def nosat_spec(bits):
    magnitude, sign = bits & 0x7FFFFFFF, bits >> 24 & 0x80
    if magnitude > 0x7F800000:
        return 0x7F
    value = f32(magnitude)
    # The hypothetical next E4M3 code at 480 is NaN. At the midpoint 464,
    # round-to-even chooses finite 448 (code 126), including negative inputs.
    return sign | (0x7F if value > 464 else nearest_even(value))


def repaired_satfinite(bits):
    magnitude, sign = bits & 0x7FFFFFFF, bits >> 24 & 0x80
    # Hardware saturates overflow; its NaN result is replaced by the repair.
    packed_lane = sign | nearest_even(min(abs(f32(bits)), 448)) if magnitude <= 0x7F800000 else 0x7F
    return (0x7F if magnitude > 0x7F800000 else sign | 0x7F) if magnitude > 0x43E80000 else packed_lane


def test_all_bf16_encodings_both_packed_positions():
    for bits in range(65536):
        other = bits ^ 0xA55A
        expected = nosat_spec(bits << 16) | nosat_spec(other << 16) << 8
        actual = repaired_satfinite(bits << 16) | repaired_satfinite(other << 16) << 8
        assert actual == expected, hex(bits)


def test_overflow_midpoint_nan_sign_and_signed_zero():
    cases = {0: 0, 0x80000000: 0x80, 0x43E7FFFF: 0x7E, 0x43E80000: 0x7E, 0x43E80001: 0x7F, 0xC3E80000: 0xFE, 0xC3E80001: 0xFF, 0x7F800000: 0x7F, 0xFF800000: 0xFF, 0x7F800001: 0x7F, 0xFFC00000: 0x7F}
    for bits, expected in cases.items():
        assert nosat_spec(bits) == repaired_satfinite(bits) == expected


def test_overlay_only_changes_shared_helper_and_adds_separate_gpu_check(tmp_path):
    # Reconstruct the exact preimage from the patch context, no downloaded tree.
    text = PATCH.read_text()
    first = text.split("--- /dev/null", 1)[0]
    preimage = "".join(line[1:] for line in first.splitlines(keepends=True)[3:] if line.startswith((" ", "-")))
    header = tmp_path / "python/sglang/kernels/jit/csrc/elementwise/rope.cuh"
    header.parent.mkdir(parents=True)
    header.write_text(preimage)
    subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=tmp_path, check=True)
    subprocess.run(["git", "apply", str(PATCH)], cwd=tmp_path, check=True)
    candidate = header.read_text()
    assert "__NV_SATFINITE" in candidate and "0x43e80000" in candidate
    assert "qknorm_rope_decode_kernel" not in text and "qknorm_rope_prefill_fast_kernel" not in text
    check = header.with_name("decode_fp8_check.cuh").read_text()
    assert '#include "rope.cuh"' in check and "__NV_NOSAT" in check and "actual[i] = to_fp8x2(value)" in check
    config = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    overlays = config["sglang_overlay_patches"]
    position = overlays.index(str(PATCH.relative_to(ROOT)))
    assert overlays[position:position + 2] == [str(PATCH.relative_to(ROOT)), "patches/sglang/decode-add-rmsnorm-group-quant.patch"]
    gate = (ROOT / "scripts/check-firmus-decode-kernels.py").read_text()
    assert "    check_decode_fp8_conversion()\n" in gate.split('if __name__ == "__main__":')[1]
