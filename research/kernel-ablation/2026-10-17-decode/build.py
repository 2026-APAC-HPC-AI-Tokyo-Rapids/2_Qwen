"""Build one tracked SM90 condition on a CPU host; never load or run CUDA.

Usage: python build.py CONDITION.json WHEEL OUTPUT --nvcc /path/to/nvcc
The resulting cubins are compilation evidence, not a GPU timing measurement.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args):
    condition = json.loads(args.condition.read_text())
    root = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(args.wheel) as wheel:
        for name in wheel.namelist():
            if name.startswith("deep_gemm/include/") and not name.endswith("/"):
                target = args.output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(wheel.read(name))
    header = args.output / "deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh"
    assert sha(header) == "a53d83cad76109588b3d5053be02877a0c2a47c38b27472269dc1319df62676d"
    patches = [root / "patches/deep-gemm" / f"sm90-{name}.patch" for name in
               ("late-topk-fp32", "fc2-pipeline", "prefill-rescale", "decode-exact-tiles", "prefill-phase")]
    for patch in patches:
        subprocess.run(["git", "apply", str(patch)], cwd=args.output, check=True)
    assert sha(header) == condition["baseline_header_sha256"]
    if condition["patch"]:
        patch = args.condition.parent / condition["patch"]
        subprocess.run(["git", "apply", str(patch)], cwd=args.output, check=True)
        patches.append(patch)
    assert sha(header) == condition["header_sha256"]
    evidence = {"condition": condition, "gpu_executed": False,
                "wheel_sha256": sha(args.wheel), "patches": {p.name: sha(p) for p in patches},
                "compiler": subprocess.check_output([args.nvcc, "--version"], universal_newlines=True), "roles": {}}
    for role, config in [("decode", "64,128,128,141312,2260992,8,64,64,256,132,4,cute::numeric_limits<float>::infinity(),true,0,false,false,true,false,true"),
                         ("prefill", "128,256,128,141312,2260992,3,64,64,512,132,4,cute::numeric_limits<float>::infinity(),true,112,true,true,false,false,false")]:
        source = args.output / f"{role}.cu"
        source.write_text('#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>\nextern "C" void* instantiate() {\n'
                          'return reinterpret_cast<void*>(&deep_gemm::sm90_fp8_mega_moe_impl<4224,4096,1536,128,8,32,' + config + '>);\n}\n')
        cubin = source.with_suffix(".cubin")
        command = [args.nvcc, "--cubin", "--std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3", "--ptxas-options=-v,-warn-spills", "-gencode", "arch=compute_90a,code=sm_90a", "-I" + str(args.output / "deep_gemm/include"), str(source), "-o", str(cubin)]
        if args.ccbin:
            command += ["--compiler-bindir", args.ccbin]
        result = subprocess.run(command, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)  # noqa: UP021, UP022 - Genkai login Python is 3.6.
        log = result.stdout + result.stderr
        source.with_suffix(".build.log").write_text(log)
        if result.returncode:
            raise RuntimeError(log)
        sass = subprocess.check_output([str(Path(args.nvcc).with_name("cuobjdump")), "--dump-sass", str(cubin)], universal_newlines=True)
        source.with_suffix(".sass").write_text(sass)
        evidence["roles"][role] = {"cubin_sha256": sha(cubin), "registers": int(re.search(r"Used (\d+) registers", log)[1]),
                                   "stack_spills": re.findall(r"(\d+ bytes stack frame, \d+ bytes spill stores, \d+ bytes spill loads)", log),
                                   "compile_command": command}
        print(condition["name"], role, evidence["roles"][role]["stack_spills"], flush=True)
    (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", type=lambda x: Path(x).resolve())
    parser.add_argument("wheel", type=lambda x: Path(x).resolve())
    parser.add_argument("output", type=lambda x: Path(x).resolve())
    parser.add_argument("--nvcc", default=shutil.which("nvcc"), required=shutil.which("nvcc") is None)
    parser.add_argument("--ccbin")
    build(parser.parse_args())
