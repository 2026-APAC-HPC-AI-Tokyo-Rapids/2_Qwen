"""CPU-only compile gate for the opt-in two-queue Decode implementation."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

CANDIDATE = "sm90-decode-independent-wg.patch"
HEADER = Path("deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh")


def compile_one(nvcc, package, overlay, directory, name, geometry):
    m, n, threads, stages, reuse, arrival, sync, swap = geometry
    source = directory / (name + ".cu")
    source.write_text('#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>\nextern "C" void* instantiate() {\nreturn reinterpret_cast<void*>(&deep_gemm::sm90_fp8_mega_moe_impl<4224,4096,1536,128,8,32,' + f'{m},{n},128,141312,2260992,{stages},64,64,{threads},132,4,cute::numeric_limits<float>::infinity(),true,0,{reuse},{arrival},{sync},false,{swap}>);\n}}\n')
    cubin = source.with_suffix(".cubin")
    command = [str(nvcc), "--cubin", "--std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3", "--ptxas-options=-v,-warn-spills,--register-usage-level=10", "-gencode", "arch=compute_90a,code=sm_90a", "-I" + str(overlay / "deep_gemm/include"), "-I" + str(package / "include"), str(source), "-o", str(cubin)]
    compiled = subprocess.run(command, text=True, capture_output=True, check=False)
    log = compiled.stdout + compiled.stderr
    (directory / (name + ".log")).write_text(log)
    assert compiled.returncode == 0, log
    sass = subprocess.check_output([str(nvcc.parent / "cuobjdump"), "-sass", str(cubin)], text=True)
    usage = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", log)
    result = {"geometry": geometry, "spill_bytes": usage, "serialization_warning": bool(re.search(r"C751[04]", log)), "sass_sha256": hashlib.sha256(sass.encode()).hexdigest(), "cubin_bytes": cubin.stat().st_size}
    print(name + " " + json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, default=Path("/opt/apac/deep-gemm-overlay"))
    parser.add_argument("--config", type=Path, default=Path("/opt/apac/firmus-framework.json"))
    parser.add_argument("--nvcc", type=Path, default=Path("/usr/local/cuda/bin/nvcc"))
    parser.add_argument("--output", type=Path, default=Path("/opt/apac/sm90-independent-wg-build.json"))
    args = parser.parse_args()
    patches = [Path(p).name for p in json.loads(args.config.read_text())["deep_gemm_overlay_patches"]]
    assert patches[-1] == CANDIDATE, "The adopted independent-WG header must be the final serving overlay"
    base_patches = patches[:-1]
    decode = (64, 128, 256, 8, "false", "false", "true", "true")
    candidate = (64, 128, 256, 6, "false", "false", "true", "true")
    prefill = (128, 256, 256, 3, "true", "true", "false", "false")
    with tempfile.TemporaryDirectory(prefix="independent-wg-compile-") as temp:
        directory = Path(temp)
        overlay = directory / "overlay"
        (overlay / HEADER).parent.mkdir(parents=True)
        shutil.copy2(args.package / "include/deep_gemm/impls/sm90_fp8_mega_moe.cuh", overlay / HEADER)
        for name in base_patches:
            subprocess.run(["git", "apply", str(args.patch_dir / name)], cwd=overlay, check=True)
        base_decode = compile_one(args.nvcc, args.package, overlay, directory, "base-decode", decode)
        base_prefill = compile_one(args.nvcc, args.package, overlay, directory, "base-prefill", prefill)
        subprocess.run(["git", "apply", str(args.patch_dir / CANDIDATE)], cwd=overlay, check=True)
        control = compile_one(args.nvcc, args.package, overlay, directory, "control-decode", decode)
        control_prefill = compile_one(args.nvcc, args.package, overlay, directory, "control-prefill", prefill)
        experimental = compile_one(args.nvcc, args.package, overlay, directory, "independent-decode", candidate)
        spill_free = bool(experimental["spill_bytes"]) and all(tuple(map(int, pair)) == (0, 0) for pair in experimental["spill_bytes"])
        result = {"spill_zero_target_met": spill_free, "gpu_validated": False, "base_decode": base_decode, "base_prefill": base_prefill, "control_decode": control, "control_prefill": control_prefill, "independent_decode": experimental, "header_sha256": hashlib.sha256((overlay / HEADER).read_bytes()).hexdigest(), "patches": {**{p: hashlib.sha256((args.patch_dir / p).read_bytes()).hexdigest() for p in patches}, CANDIDATE: hashlib.sha256((args.patch_dir / CANDIDATE).read_bytes()).hexdigest()}}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        for log in directory.glob("*.log"):
            shutil.copy2(log, args.output.with_name(args.output.stem + "-" + log.name))
        assert base_decode["sass_sha256"] == control["sass_sha256"], "Opt-out Decode SASS changed"
        assert base_prefill["sass_sha256"] == control_prefill["sass_sha256"], "Prefill SASS changed"
        assert not experimental["serialization_warning"], "Independent WG has serialized WGMMA"
        if not spill_free:
            warnings.warn("Independent WG compiles but misses the zero-spill resource target; GPU correctness and performance remain unvalidated", stacklevel=1)


if __name__ == "__main__":
    main()
