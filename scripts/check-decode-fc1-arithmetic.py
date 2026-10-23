"""CPU float32 check of source-extracted FC1 arithmetic; not a CUDA emulator.

Uses host expf/reciprocal as common surrogates, actual CUTLASS min/max/abs, and
CUDA header host FP8 conversion checked against an independent nearest-value
oracle. Does not establish
CUDA intrinsic, compiler, WGMMA, synchronization, or GPU replay equivalence.
"""

import argparse
import array
import bisect
import ctypes
import hashlib
import itertools
import json
import math
import random
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = Path("deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh")


def section(text, start, end):
    return text[text.index(start) : text.index(end, text.index(start))]


def extract(control, candidate):
    def swiglu(text, baseline):
        lambdas = section(
            text, "                    auto silu =", "                    // "
        )
        body = section(
            text,
            "                            float g0 = final_accum[i * 4 + 0];",
            "                        }\n",
        )
        if baseline:
            body = re.sub(r"smem_cd_swap_l1_fp32\[[^\]]+\] =", "v0 =", body)
        return (
            "float final_accum[4] = {g, 0, u, 0}; constexpr unsigned i=0; float v0=0;\n"
            + lambdas
            + body
            + "return v0;"
        )

    # Both token parities must still use the same arithmetic before one scalar
    # lane is lifted into the host harness.
    for text, baseline in ((control, True), (candidate, False)):
        zero = section(
            text,
            "                            float g0 = final_accum[i * 4 + 0];",
            "                        }\n",
        )
        one = section(
            text,
            "                            float g1 = final_accum[i * 4 + 1];",
            "                        }\n",
        )
        for old, new in (
            ("g1", "g0"),
            ("u1", "u0"),
            ("v1", "v0"),
            ("token_1", "token_0"),
            ("+ 1]", "+ 0]"),
            ("+ 3]", "+ 2]"),
        ):
            one = one.replace(old, new)
        assert zero.split() == one.split(), "SwiGLU token parity expressions diverged"
    helper = section(
        control,
        "__forceinline__ __device__ void sm90_fp8_mega_moe_get_e4m3_sf_and_sf_inv(",
        "\ntemplate <uint32_t BLOCK_M",
    )
    assert helper == section(
        candidate,
        "__forceinline__ __device__ void sm90_fp8_mega_moe_get_e4m3_sf_and_sf_inv(",
        "\ntemplate <uint32_t BLOCK_M",
    )
    max_fold = section(
        control,
        "                            amax = cute::max(amax, cute::abs(values[j].x));",
        "                        }\n",
    )
    max_partial = re.search(
        r"float a0 = cute::max\(0.0f, cute::abs\(v0\)\);", candidate
    ).group()
    old_quant = section(
        control,
        "                                const __nv_fp8x2_e4m3 lo(make_float2(values[j].x * sf_inv",
        "                            }\n",
    )
    old_quant = re.sub(
        r"\*reinterpret_cast<uint32_t\*>\([^\n]+\) =", "packed =", old_quant
    )
    product = re.search(
        r"const float v0 = final_accum\[i \* 4 \+ 0\] \* inv0;", candidate
    ).group()
    pair = re.search(
        r"const __nv_fp8x2_e4m3 q0\(make_float2\(v0, next0\)\);", candidate
    ).group()
    packed = re.search(
        r"= (static_cast<uint32_t>\(q0.__x\) \| \(hi0 << 16\));", candidate
    ).group(1)
    # The host wrapper supplies the values selected by these exact source mappings.
    for expression in (
        "warp * BLOCK_M + (token ^ (warp * 4))",
        "i * 8 + col_idx * 2",
        "for (uint32_t mask = 4; mask <= 16; mask *= 2)",
        "a0 = cute::max(a0, __shfl_xor_sync(0xffffffff, a0, mask))",
        "const float v1 = final_accum[i * 4 + 1] * inv1;",
        "const __nv_fp8x2_e4m3 q1(make_float2(v1, next1));",
        "__shfl_xor_sync(0xffffffff, v0, 4)",
        "__shfl_xor_sync(0xffffffff, static_cast<uint32_t>(q0.__x), 8)",
    ):
        assert expression in candidate, expression
    template = r"""
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <initializer_list>
#include <cuda_fp8.h>
#include <cute/numeric/math.hpp>
namespace math { float fast_rcp(float x) { volatile float y=1.0f/x; return y; } }
float host_expf(float x) { return std::exp(x); }
#define __expf host_expf
float __fmul_rn(float a,float b) { volatile float v=a*b; return v; }
// Retain the constructor input for intermediate-bit checks, then invoke the
// exact official CUDA float2 constructor through its host fallback.
struct tracked_fp8x2 {
    uint16_t __x; float2 input;
    explicit tracked_fp8x2(float2 v): __x(__nv_fp8x2_e4m3(v).__x),input(v) {}
};
#define __nv_fp8x2_e4m3 tracked_fp8x2
@HELPER@
template<bool kFastMath,float kActivationClamp> float baseline(float g,float u) { @BASE@ }
template<bool kFastMath,float kActivationClamp> float candidate(float g,float u) { @CAND@ }
extern "C" void swiglu(int version,int fast,int clamp,const float* g,const float* u,float* out,int n) {
    constexpr float inf=std::numeric_limits<float>::infinity();
    for(int i=0;i<n;++i) {
        if(version==0) out[i]=clamp?(fast?baseline<true,8.0f>(g[i],u[i]):baseline<false,8.0f>(g[i],u[i])):(fast?baseline<true,inf>(g[i],u[i]):baseline<false,inf>(g[i],u[i]));
        else out[i]=clamp?(fast?candidate<true,8.0f>(g[i],u[i]):candidate<false,8.0f>(g[i],u[i])):(fast?candidate<true,inf>(g[i],u[i]):candidate<false,inf>(g[i],u[i]));
    }
}
extern "C" void maxima(int version,const float* input,float* output,int rows) {
    for(int token=0;token<rows;++token) {
        const float* row=input+token*64;
        if(version==0) {
            float quad[4]={};
            for(int q=0;q<4;++q) {
                float amax=0;float4 values[4];
                for(unsigned j=0;j<4;++j) {
                    unsigned col=(q+4*((j+(token+1)/2)&3u))*4;
                    values[j]={row[col],row[col+1],row[col+2],row[col+3]};
                    @MAXFOLD@
                }
                quad[q]=amax;
            }
            output[token]=cute::max(cute::max(quad[0],quad[1]),cute::max(quad[2],quad[3]));
        } else {
            float partial[8]={};
            for(int warp=0;warp<8;++warp) {
                float lanes[32];
                for(int lane=0;lane<32;++lane) { float v0=row[warp*8+lane/4];@MAXPARTIAL@ lanes[lane]=a0; }
                for(int mask: {4,8,16}) { float old[32];for(int lane=0;lane<32;++lane)old[lane]=lanes[lane];for(int lane=0;lane<32;++lane)lanes[lane]=cute::max(old[lane],old[lane^mask]); }
                partial[warp]=lanes[0];
            }
            float q[4];for(int i=0;i<4;++i)q[i]=cute::max(partial[2*i],partial[2*i+1]);
            output[token]=cute::max(cute::max(q[0],q[1]),cute::max(q[2],q[3]));
        }
    }
}
extern "C" void scales(const float* values,float* sf_out,float* inv_out,int n) {
    for(int i=0;i<n;++i) { float2 sf,inv; sm90_fp8_mega_moe_get_e4m3_sf_and_sf_inv(make_float2(values[i],values[i]),sf,inv); sf_out[i]=sf.x;inv_out[i]=inv.x; }
}
extern "C" void quantize(int version,const float* input,const float* inverse,float* products,uint32_t* output,int groups) {
    for(int group=0;group<groups;++group) {
        float sf_inv=inverse[group];uint32_t packed=0;
        if(version==0) {
            const float* v=input+4*group;float4 values[1]={{v[0],v[1],v[2],v[3]}};constexpr unsigned j=0;
            @OLDQUANT@
            products[4*group]=lo.input.x;products[4*group+1]=lo.input.y;products[4*group+2]=hi.input.x;products[4*group+3]=hi.input.y;
        } else {
            float result[4];
            for(int channel=0;channel<4;++channel) {
                float final_accum[4]={input[4*group+channel],0,0,0};constexpr unsigned i=0;float inv0=sf_inv;
                @PRODUCT@
                result[channel]=v0;products[4*group+channel]=v0;
            }
            float v0=result[0],next0=result[1];@PAIR@
            float vhi=result[2],nhi=result[3];const __nv_fp8x2_e4m3 high(make_float2(vhi,nhi));uint32_t hi0=high.__x;
            packed=@PACKED@;
        }
        output[group]=packed;
    }
}
"""
    for key, value in {
        "HELPER": helper,
        "BASE": swiglu(control, True),
        "CAND": swiglu(candidate, False),
        "OLDQUANT": old_quant,
        "PRODUCT": product,
        "PAIR": pair,
        "PACKED": packed,
        "MAXFOLD": max_fold,
        "MAXPARTIAL": max_partial,
    }.items():
        template = template.replace("@" + key + "@", value)
    return template


def floats(values):
    return array.array("f", values)


def bits(values):
    return array.array("I", values.tobytes())


def ptr(values):
    return (ctypes.c_float * len(values)).from_buffer(values)


def f32(x):
    return ctypes.c_float(x).value


def next32(x, direction):
    u = struct.unpack("I", struct.pack("f", x))[0]
    return struct.unpack("f", struct.pack("I", u + direction))[0]


FINITE = [
    (code & 7) * 2.0**-9
    if code < 8
    else (1 + (code & 7) / 8) * 2.0 ** ((code >> 3) - 7)
    for code in range(127)
]


def oracle(x):
    sign = 128 if math.copysign(1.0, x) < 0 else 0
    if math.isnan(x):
        return 127
    a = abs(x)
    if a >= 448:
        return sign | 126
    at = bisect.bisect_left(FINITE, a)
    candidates = {max(0, at - 1), min(at, 126)}
    return sign | min(candidates, key=lambda code: (abs(FINITE[code] - a), code & 1))


def ordered_max(a, b):
    return b if a < b else a  # noqa: FURB136 - preserve ordered NaN semantics


def source_amax(row):
    result = 0.0
    for value in row:
        result = ordered_max(result, abs(value))
    return result


def candidate_amax(row):
    partial = []
    for warp in range(8):
        lanes = [ordered_max(0.0, abs(row[warp * 8 + lane // 4])) for lane in range(32)]
        for mask in (4, 8, 16):
            lanes = [
                ordered_max(value, lanes[lane ^ mask])
                for lane, value in enumerate(lanes)
            ]
        partial.append(lanes[0])
    quad = [ordered_max(partial[2 * q], partial[2 * q + 1]) for q in range(4)]
    for mask in (1, 2):
        quad = [ordered_max(value, quad[q ^ mask]) for q, value in enumerate(quad)]
    assert len(set(quad)) == 1
    return quad[0]


def check(lib):
    # Independent scalar arithmetic around the same host expf primitive; this
    # checks FP32 operation ordering, not CUDA's device exponential accuracy.
    host_math = ctypes.CDLL("libm.so.6")
    host_math.expf.argtypes = [ctypes.c_float]
    host_math.expf.restype = ctypes.c_float

    def swiglu_oracle(g, u, clamp):
        if clamp:
            g = g if g < 8.0 else 8.0  # noqa: FURB136 - cute::min selects second operand on NaN
            u = ordered_max(u, -8.0)
            u = u if u < 8.0 else 8.0  # noqa: FURB136 - cute::min selects second operand on NaN
        exponential = host_math.expf(-g)
        sigmoid = f32(1.0 / f32(1.0 + exponential))
        return f32(f32(g * sigmoid) * u)

    special = array.array("f")
    special.frombytes(
        array.array(
            "I",
            [
                0,
                0x80000000,
                1,
                0x80000001,
                0x007FFFFF,
                0x00800000,
                0x7F7FFFFF,
                0xFF7FFFFF,
                0x7F800000,
                0xFF800000,
                0x7FC12345,
                0xFFC12345,
                0x7F800001,
            ],
        ).tobytes()
    )
    ordinary = [-100.0, -20.0, -8.0, -1.0, -(2**-12), 2**-12, 1.0, 8.0, 20.0, 100.0]
    values = list(special) + ordinary
    inputs = [(g, u) for g in values for u in values]
    rng = random.Random(20261023)
    for _ in range(2048):
        # Include arbitrary FP32 normals/subnormals and overflow-producing pairs.
        raw = array.array("I", [rng.getrandbits(32), rng.getrandbits(32)])
        pair = array.array("f")
        pair.frombytes(raw.tobytes())
        inputs.append(tuple(pair))
    g, u = floats(x[0] for x in inputs), floats(x[1] for x in inputs)
    all_outputs = []
    changed_outputs = []
    nan_payload_differences = {}

    def compare(actual, expected, stage):
        differing = 0
        for a, b, ab, bb in zip(actual, expected, bits(actual), bits(expected)):
            if ab != bb:
                assert math.isnan(a) and math.isnan(b), (stage, hex(ab), hex(bb))
                differing += 1
        nan_payload_differences[stage] = (
            nan_payload_differences.get(stage, 0) + differing
        )

    for fast in (0, 1):
        for clamp in (0, 1):
            output = [floats([0] * len(g)), floats([0] * len(g))]
            for version in (0, 1):
                lib.swiglu(
                    version, fast, clamp, ptr(g), ptr(u), ptr(output[version]), len(g)
                )
            compare(output[0], output[1], f"SwiGLU-fast{fast}-clamp{clamp}")
            expected = floats(swiglu_oracle(g0, u0, clamp) for g0, u0 in inputs)
            compare(output[0], expected, f"SwiGLU-oracle-fast{fast}-clamp{clamp}")
            all_outputs.extend(output[0])
            changed_outputs.extend(output[1])
    # Every finite E4M3 rounding midpoint, and its adjacent float32 numbers.
    boundary = [
        0.0,
        -0.0,
        2**-149,
        -(2**-149),
        448.0,
        449.0,
        math.inf,
        -math.inf,
        math.nan,
    ]
    for lo, hi in itertools.pairwise(FINITE):
        mid = f32((lo + hi) / 2)
        for value in (next32(mid, -1), mid, next32(mid, 1)):
            boundary.extend((value, -value))
    direct = boundary + all_outputs
    direct += [0.0] * (-len(direct) % 4)
    altered = boundary + changed_outputs
    altered += [0.0] * (-len(altered) % 4)
    values4 = floats(direct)
    groups = len(values4) // 4
    inverse = floats([1.0] * groups)
    quant_calls = 0

    def quant(values4, inverse, changed=None):
        nonlocal quant_calls
        groups = len(inverse)
        products = [floats([0] * len(values4)), floats([0] * len(values4))]
        encoded = [array.array("I", [0] * groups), array.array("I", [0] * groups)]
        for version in (0, 1):
            out = (ctypes.c_uint32 * groups).from_buffer(encoded[version])
            selected = changed if version == 1 and changed is not None else values4
            lib.quantize(
                version,
                ptr(selected),
                ptr(inverse),
                ptr(products[version]),
                out,
                groups,
            )
        compare(products[0], products[1], "quantization-products")
        expected_products = floats(
            f32(value * inverse[channel // 4]) for channel, value in enumerate(values4)
        )
        compare(products[0], expected_products, "quantization-products-oracle")
        assert encoded[0] == encoded[1], "FP8 packed bytes"
        for group, packed in enumerate(encoded[0]):
            expected = sum(
                oracle(products[0][4 * group + j]) << (8 * j) for j in range(4)
            )
            assert packed == expected, (
                "independent FP8 oracle",
                group,
                packed,
                expected,
            )
        quant_calls += len(values4)

    quant(values4, inverse, floats(altered))
    # Full SF64 rows; both WG maxima, poisoned channels, every valid M tail.
    rows = [
        direct[(token * 71) % len(direct) :] + direct[: (token * 71) % len(direct)]
        for token in range(64)
    ]
    rows = [row[:64] for row in rows]
    rows[0] = [math.nan] * 64
    rows[1] = [-0.0] * 64
    rows[2] = [0.0] * 63 + [math.inf]
    rows[3] = [2**-149] * 64
    rows[4] = [0.0] * 64
    rows[4][1] = math.nan
    rows[4][47] = 123.0
    rows[5] = [math.nan] * 63 + [448.0]
    rows[6] = [-0.0, 0.0] * 32
    for token, value in enumerate(
        (next32(f32(1e-10), -1), f32(1e-10), next32(f32(1e-10), 1)), 7
    ):
        rows[token] = [value] * 64
    rows[10] = [f32(3.4028234663852886e38)] * 64
    for valid in range(1, 65):
        amax = [floats([0] * valid), floats([0] * valid)]
        flat = floats(value for row in rows[:valid] for value in row)
        for version in (0, 1):
            lib.maxima(version, ptr(flat), ptr(amax[version]), valid)
        assert bits(amax[0]) == bits(floats(source_amax(row) for row in rows[:valid]))
        assert bits(amax[1]) == bits(
            floats(candidate_amax(row) for row in rows[:valid])
        )
        assert bits(amax[0]) == bits(amax[1]), ("amax", valid)
        sf = [floats([0] * valid), floats([0] * valid)]
        inv = [floats([0] * valid), floats([0] * valid)]
        for version in (0, 1):
            lib.scales(ptr(amax[version]), ptr(sf[version]), ptr(inv[version]), valid)
        assert bits(sf[0]) == bits(sf[1]) and bits(inv[0]) == bits(inv[1]), (
            "SF/inverse",
            valid,
        )
        # Independent Python scalar reference rounds after each FP32 operation.
        for i, value in enumerate(amax[0]):
            expected_sf = f32(max(value, f32(1e-10)) * f32(1 / 448))
            expected_inv = f32(1 / expected_sf)
            assert bits(floats([expected_sf, expected_inv])) == bits(
                floats([sf[0][i], inv[0][i]])
            )
        quant(
            floats(value for row in rows[:valid] for value in row),
            floats(inv[0][token] for token in range(valid) for _ in range(16)),
        )
    return {
        "swiglu_pairs_per_mode": len(inputs),
        "swiglu_modes": 4,
        "valid_m_cases": 64,
        "quantized_values": quant_calls,
        "intermediate_bit_comparison": "non-NaN SwiGLU/products (including signed zero), amax, SF, inverse-SF, packed FP8; NaN class must match but payload differences recorded",
        "nan_payload_differences": nan_payload_differences,
        "limits": "host expf/reciprocal surrogates; actual CUTLASS min/max/abs and official CUDA host FP8 conversion; CUDA device intrinsic/compiler behavior and GPU ordering/replay not established",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-header", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cuda-include",
        type=Path,
        required=True,
        help="CUDA headers containing cuda_fp8.h and nv/target",
    )
    parser.add_argument(
        "--cutlass-include",
        type=Path,
        required=True,
        help="CUTLASS include directory containing cute/numeric/math.hpp",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fc1-host-source-") as temporary:
        root = Path(temporary)
        header = root / HEADER
        header.parent.mkdir(parents=True)
        shutil.copy2(args.base_header, header)
        assert (
            hashlib.sha256(header.read_bytes()).hexdigest()
            == "ff7a58cca00aa42a90386d12b0437b5d8caf35eb0ad082a8a1a0b4b74001a2a1"
        )
        for patch in json.loads((ROOT / "configs/firmus-framework.json").read_text())[
            "deep_gemm_overlay_patches"
        ]:
            subprocess.run(["git", "apply", str(ROOT / patch)], cwd=root, check=True)
        control = header.read_text()
        patch = (
            ROOT
            / "patches/deep-gemm-bench/decode-fc1-register/sm90-decode-fc1-register.patch"
        )
        subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
        candidate = header.read_text()
    source = args.output_dir / "source-derived.cpp"
    source.write_text(extract(control, candidate))
    binary = args.output_dir / "source-derived.so"
    command = [
        "g++",
        "-std=c++20",
        "-O2",
        "-fPIC",
        "-shared",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-I",
        str(args.cuda_include),
        "-I",
        str(args.cuda_include / "cccl"),
        "-I",
        str(args.cutlass_include),
        str(source),
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True)
    lib = ctypes.CDLL(str(binary.resolve()))
    fptr = ctypes.POINTER(ctypes.c_float)
    lib.swiglu.argtypes = [ctypes.c_int] * 3 + [fptr] * 3 + [ctypes.c_int]
    lib.maxima.argtypes = [ctypes.c_int, fptr, fptr, ctypes.c_int]
    lib.scales.argtypes = [fptr, fptr, fptr, ctypes.c_int]
    lib.quantize.argtypes = [
        ctypes.c_int,
        fptr,
        fptr,
        fptr,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
    ]
    for function in (lib.swiglu, lib.maxima, lib.scales, lib.quantize):
        function.restype = None
    report = check(lib)
    compiler_version = subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]  # fmt: skip
    report.update(
        cuda_fp8_hpp_sha256=hashlib.sha256(
            (args.cuda_include / "cuda_fp8.hpp").read_bytes()
        ).hexdigest(),
        cute_math_hpp_sha256=hashlib.sha256(
            (args.cutlass_include / "cute/numeric/math.hpp").read_bytes()
        ).hexdigest(),
        control_sha256=hashlib.sha256(control.encode()).hexdigest(),
        candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        generated_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        compiler_command=command,
        compiler_version=compiler_version,
    )
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
