"""Offline SM90 code generation for route metadata; no GPU execution."""

import argparse
import hashlib
import json
from pathlib import Path

import triton
from compact_routes_triton import _count_reset, _offsets, _scatter
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for tokens, dtype, base in [
        *[
            (m, dtype, 0)
            for m in [1, 7, 32, 97, 288, 384, 768]
            for dtype in ["*i32", "*i64"]
        ],
        (768, "*i64", 96),
    ]:
        routes = tokens * 8
        constants = {"R": routes, "B": triton.next_power_of_2(routes), "BASE": base}
        for function, warps in [(_count_reset, 8), (_scatter, 8)]:
            signature = {
                "Ids": dtype,
                "Valid": "*i1",
                "Inverse": "*i32",
                "Routes": "*i32",
            }
            specialized = dict(constants)
            if function is _count_reset:
                signature["Counts"] = "*i32"
                specialized["RESET"] = triton.next_power_of_2(triton.cdiv(routes, 32))
            else:
                signature["Offsets"] = "*i32"
            compiled = triton.compile(
                ASTSource(function, signature=signature, constexprs=specialized),
                target=GPUTarget("cuda", 90, 32),
                options={"num_warps": warps},
            )
            assert (
                compiled.metadata.arch == "sm90"
                and compiled.metadata.global_scratch_size == 0
            )
            rows.append(
                {
                    "kernel": function.__name__,
                    "tokens": tokens,
                    "id_dtype": dtype,
                    "expert_base": base,
                    "shared_bytes": compiled.metadata.shared,
                    "ptx_sha256": hashlib.sha256(
                        compiled.asm["ptx"].encode()
                    ).hexdigest(),
                    "cubin_bytes": len(compiled.asm["cubin"]),
                }
            )
    compiled = triton.compile(
        ASTSource(_offsets, signature={"Counts": "*i32", "Offsets": "*i32"}),
        target=GPUTarget("cuda", 90, 32),
        options={"num_warps": 1},
    )
    rows.append(
        {
            "kernel": "_offsets",
            "shared_bytes": compiled.metadata.shared,
            "ptx_sha256": hashlib.sha256(compiled.asm["ptx"].encode()).hexdigest(),
            "cubin_bytes": len(compiled.asm["cubin"]),
        }
    )
    result = {
        "schema": "compact-route-offline-sm90/v1",
        "triton": triton.__version__,
        "compilations": len(rows),
        "kernels": rows,
        "limitations": "Offline code generation only. No GPU launch, latency, CUDA graph replay, engine integration, FP8/BF16 model numerics or official accuracy/performance validation. Local CPU torch companion differs from serving CUDA build; Triton package version matches pinned image metadata.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Offline SM90 compilation: {len(rows)} specializations PASS; no GPU execution"
    )


if __name__ == "__main__":
    main()
