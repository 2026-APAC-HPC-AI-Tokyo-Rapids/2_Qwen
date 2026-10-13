"""Execute selected exact-image decision functions without importing CUDA.

Platform/package availability are simulated. This checks dispatch/shape logic,
not GPU execution, memory capacity, numerical accuracy, or performance.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace


def functions(path: Path, names: set[str], class_name: str | None = None) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = list(tree.body)
    if class_name:
        nodes.extend(
            next(
                n.body
                for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == class_name
            )
        )
    selected = [
        n
        for n in nodes
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    if {n.name for n in selected} != names:
        raise ValueError("requested native decision functions missing")
    for n in selected:
        if isinstance(n, ast.FunctionDef):
            n.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    return {"code": compile(ast.fix_missing_locations(module), str(path), "exec")}


EXPECTED_SOURCE_HASHES = {
    "model_executor__layers__fused_moe__oracle__fp8.py": "245bd36c285df1e6f7c51e52dcda31ad247894577a9f8d765613c2c2feae9aee",
    "model_executor__layers__fused_moe__experts__deep_gemm_moe.py": "16911a64a72eafc26a1bcb544a91f90144d85dc41e3ee57cf20e89925227bbb7",
    "model_executor__layers__fused_moe__experts__triton_deep_gemm_moe.py": "c7209fa5e1ccb184a68b37a0706fc571a76696c2ae79802ddc7f889b1c29684d",
    "model_executor__layers__fused_moe__experts__triton_moe.py": "806fbccfe30f5fbd578acf773add99bdf022b12d617958bd6281aef6de369701",
    "model_executor__layers__fused_moe__fused_moe.py": "7072eb06237be9d33dcb0ef7101410f886a6363c98cbee70a014c68b70f639cb",
    "model_executor__layers__fused_moe__modular_kernel.py": "8dc4e0dbbf35b3aac8add63ca6f2a7bbfdadaabbd60826e24598b22c7d8e5c22",
    "model_executor__layers__fused_moe__prepare_finalize__no_dp_ep.py": "adbbf529cd22059c95b5380952943a70cf11e05e321dcd707f296bebde61a6e7",
    "model_executor__layers__fused_moe__prepare_finalize__naive_dp_ep.py": "db015e5b18750bbbf5b2bbaceca877dd6f3f3e74c2a94e8dc185f7413b387f91",
    "model_executor__layers__fused_moe__config.py": "a3a87cebcc7ac85d73f471dcfafc2d899c6fc08f264777b74a0e919adac9e03f",
    "model_executor__layers__quantization__utils__fp8_utils.py": "03c57e3c11e93818f3bb42ab91fee6386b229e57c7506f65fab3b010425d6cea",
    "utils__deep_gemm.py": "5401bbf2556f072f210817888c3e13918773b6b91fdcb67870a235cf205fb73e",
    "config__parallel.py": "f012ace89e3fb730a467f729f602679bc145c49f0d10721811c9d818570e7b2b",
    "v1__spec_decode__eagle.py": "b2b1f7d15117b43108be368756d9c6e9334d9415a070e46178e18bb7c6476378",
    "v1__spec_decode__draft_model.py": "6b884f49a75f01d1aa6b2d27159b63fa27f7ba7f0518d2810d1a5b565e14f026",
}


def audit(directory: Path) -> dict:
    for name, expected in EXPECTED_SOURCE_HASHES.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"pinned source hash mismatch: {name}")
    prefix = "model_executor__layers__fused_moe__"
    platform = SimpleNamespace(
        is_cuda=lambda: True,
        is_xpu=lambda: False,
        is_cpu=lambda: False,
        is_device_capability=lambda n: n == 90,
        is_device_capability_family=lambda n: False,
    )
    keys = {
        name: name
        for name in (
            "kFp8Static128BlockSym",
            "kFp8Dynamic128Sym",
            "kMxfp8Static",
            "kMxfp8Dynamic",
        )
    }
    scope = {
        "Enum": Enum,
        "dataclass": dataclass,
        "current_platform": platform,
        "get_tensor_model_parallel_rank": lambda: 0,
        **keys,
    }
    oracle = directory / (prefix + "oracle__fp8.py")
    exec(  # noqa: S102 -- Named functions from hash-verified pinned image sources.
        functions(
            oracle, {"Fp8MoeBackend", "_get_priority_backends", "map_fp8_backend"}
        )["code"],
        scope,
    )
    config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(ep_size=1, use_deepep_v2_kernels=False)
    )
    auto = scope["_get_priority_backends"](
        config, keys["kFp8Static128BlockSym"], keys["kFp8Dynamic128Sym"]
    )[0].value
    requested = scope["map_fp8_backend"]("deep_gemm").value
    assert auto == "TRITON" and requested == "DEEPGEMM"
    dg = directory / (prefix + "experts__deep_gemm_moe.py")
    scope.update(get_mk_alignment_for_contiguous_layout=lambda: [128, 128])
    exec(  # noqa: S102 -- Named functions from hash-verified pinned image sources.
        functions(
            dg, {"_valid_deep_gemm_shape", "_supports_quant_scheme"}, "DeepGemmExperts"
        )["code"],
        scope,
    )
    assert scope["_supports_quant_scheme"](
        keys["kFp8Static128BlockSym"], keys["kFp8Dynamic128Sym"]
    )
    shapes = []
    for tokens in (128, 256, 1024, 4096, 8192):
        supported = scope["_valid_deep_gemm_shape"](tokens, 768, 4096)
        assert supported
        shapes.append(
            {
                "tokens": tokens,
                "tp2_expert_intermediate": 768,
                "hidden": 4096,
                "aligned": supported,
            }
        )
    assert not scope["_valid_deep_gemm_shape"](4, 768, 4096)
    source = dg.read_text(encoding="utf-8")
    assert "elif N <= 512:" in source
    assert "silu_mul_per_token_group_quant_fp8_colmajor(" in source
    triton = directory / (prefix + "experts__triton_moe.py")
    assert "and not is_deep_gemm_e8m0_used()" in triton.read_text(encoding="utf-8")
    parallel = directory / (prefix + "config.py")
    exec(functions(parallel, {"FusedMoEParallelConfig"})["code"], scope)  # noqa: S102 -- Named functions from hash-verified pinned image sources.
    native_parallel = scope["FusedMoEParallelConfig"].make(
        4,
        1,
        1,
        1,
        SimpleNamespace(
            enable_expert_parallel=True,
            all2all_backend="allgather_reducescatter",
            enable_eplb=False,
        ),
    )
    assert native_parallel.ep_size == 4 and native_parallel.tp_size == 1
    assert not native_parallel.use_all2all_kernels
    return {
        "native_source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (oracle, dg, triton)
        },
        "simulated_hopper_ep1_auto_priority": auto,
        "explicit_prefill_backend_mapping": requested,
        "native_deepgemm_block_fp8_quant_scheme_supported": True,
        "native_decoder_tp4_ep4_plan": {
            "attention_tp": 4,
            "moe_tp": native_parallel.tp_size,
            "moe_ep": native_parallel.ep_size,
            "uses_all2all_kernels": native_parallel.use_all2all_kernels,
            "expert_intermediate_per_rank": 1536,
            "local_experts": 32,
        },
        "prefill_shape_checks": shapes,
        "small_shape_alignment_fails_without_e8m0_bypass": True,
        "decoder_n384_below_non_e8m0_deepgemm_heuristic_threshold": True,
        "limitations": "Exact image function bodies with simulated Hopper/platform/package dependencies and alignment128. Not actual GPU kernel selection, performance, CUDA capture, package API availability, or accuracy validation.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_directory", type=Path)
    print(json.dumps(audit(parser.parse_args().source_directory), indent=2))
