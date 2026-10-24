"""Compile both Prefill queues on CPU, retaining logs and opt-out SASS checks."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HEADER = Path("deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh")
PATCH = "sm90-prefill-independent-wg.patch"


def compile_one(nvcc, package, overlay, output, name, experts, mode, stages, decode=False):
    source = output / (name + ".cu")
    pool, sf_pool = {128: (141312, 2260992), 140: (142080, 2273280)}[experts]
    shape = f"64,128,128,{pool},{sf_pool},{stages},64,64,256,132,4,cute::numeric_limits<float>::infinity(),true,0,false,false,true,false,true" if decode else f"128,256,128,{pool},{sf_pool},{stages},64,64,256,132,4,cute::numeric_limits<float>::infinity(),true,0,true,true,false,false,false"
    prefix = f"#define DG_SM90_PREFILL_INDEPENDENT_WG {mode}\n" if mode else ""
    source.write_text(prefix + '#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>\nextern "C" void* instantiate() { return reinterpret_cast<void*>(&deep_gemm::sm90_fp8_mega_moe_impl<4224,4096,1536,' + f'{experts},8,{experts // 4},' + shape + '>); }\n')
    cubin = source.with_suffix(".cubin")
    command = [str(nvcc), "--cubin", "--std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3", "--ptxas-options=-v,-warn-spills,--register-usage-level=10", "-gencode", "arch=compute_90a,code=sm_90a", "-I" + str(overlay / "deep_gemm/include"), "-I" + str(package / "include"), str(source), "-o", str(cubin)]
    (output / (name + ".command.json")).write_text(json.dumps(command) + "\n")
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    log = result.stdout + result.stderr
    (output / (name + ".log")).write_text(log)
    record = {"name": name, "experts": experts, "mode": mode, "stages": stages, "pool_tokens": pool, "sf_pool_tokens": sf_pool, "compile_exit": result.returncode, "registers": list(map(int, re.findall(r"Used (\d+) registers", log))), "spill_bytes": [list(map(int, pair)) for pair in re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", log)], "serialization_warning": bool(re.search(r"C751[04]", log))}
    if result.returncode == 0:
        sass = subprocess.check_output([str(nvcc.parent / "cuobjdump"), "-sass", str(cubin)], text=True)
        (output / (name + ".sass")).write_text(sass)
        instructions = "\n".join(line for line in sass.splitlines() if "/*" in line)
        record["instruction_sha256"] = hashlib.sha256(instructions.encode()).hexdigest()
    print(json.dumps(record), flush=True)
    return record


def resource_ok(row):
    # A is a measurement candidate: allow each ptxas spill count up to 128 B.
    # B retains the zero-spill requirement. These are not numerical gates.
    limit = 128 if row["mode"] == 1 else 0
    return row["compile_exit"] == 0 and bool(row["spill_bytes"]) and all(max(pair) <= limit for pair in row["spill_bytes"]) and bool(row["registers"]) and max(row["registers"]) <= 168 and not row["serialization_warning"] and row["smem_bytes"] <= 232448


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, default=Path("/opt/apac/deep-gemm-overlay"))
    parser.add_argument("--config", type=Path, default=Path("/opt/apac/firmus-framework.json"))
    parser.add_argument("--nvcc", type=Path, default=Path("/usr/local/cuda/bin/nvcc"))
    parser.add_argument("--output-dir", type=Path, default=Path("/opt/apac/prefill-independent-wg-compile"))
    args = parser.parse_args()
    args.package = args.package.resolve()
    args.patch_dir = args.patch_dir.resolve()
    args.config = args.config.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    baseline = {}
    with tempfile.TemporaryDirectory(prefix="prefill-iwg-compile-") as temp:
        overlay = Path(temp)
        (overlay / HEADER).parent.mkdir(parents=True)
        shutil.copy2(args.package / "include/deep_gemm/impls/sm90_fp8_mega_moe.cuh", overlay / HEADER)
        for patch in json.loads(args.config.read_text())["deep_gemm_overlay_patches"]:
            subprocess.run(["git", "apply", str(args.patch_dir / Path(patch).name)], cwd=overlay, check=True)
        cases = [(f"prefill-{e}", e, 3, False) for e in (128, 140)] + [("decode-8", 128, 8, True), ("decode-6", 128, 6, True)]
        for name, experts, stages, decode in cases:
            row = compile_one(args.nvcc, args.package, overlay, args.output_dir, "before-" + name, experts, 0, stages, decode)
            records.append(row)
            baseline[name] = row
        subprocess.run(["git", "apply", str(args.patch_dir / PATCH)], cwd=overlay, check=True)
        for name, experts, stages, decode in cases:
            row = compile_one(args.nvcc, args.package, overlay, args.output_dir, "after-" + name, experts, 0, stages, decode)
            records.append(row)
            assert row["compile_exit"] == 0 and row.get("instruction_sha256") == baseline[name].get("instruction_sha256"), name + " opt-out SASS changed"
        for mode, stages in ((1, 4), (2, 3)):
            for experts in (128, 140):
                row = compile_one(args.nvcc, args.package, overlay, args.output_dir, f"mode-{mode}-e{experts}", experts, mode, stages)
                row["smem_bytes"] = 208080 if mode == 1 else 199360
                row["spill_limit_bytes"] = 128 if mode == 1 else 0
                row["resource_pass"] = resource_ok(row)
                records.append(row)
                # A above the allowed spill bound stops this mode; B still compiles.
                if not row["resource_pass"]:
                    break
        result = {"gpu_executed": False, "nvcc_version": subprocess.check_output([str(args.nvcc), "--version"], text=True), "header_sha256": hashlib.sha256((overlay / HEADER).read_bytes()).hexdigest(), "records": records}
        (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    candidates = [r for r in records if r["mode"]]
    assert len(candidates) == 4 and all(r["resource_pass"] for r in candidates), "Prefill mode stopped by compile/resource gate; see results.json"


if __name__ == "__main__":
    main()
