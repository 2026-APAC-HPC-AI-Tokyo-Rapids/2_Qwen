"""CPU-only comparison using CUDA and DeepGEMM extracted from image 39110496019 (the 14-patch header).

Offline tool, not a build step: it needs the extracted toolchain and compares the guard + ping-pong
header against that image. The image build itself compiles the same Prefill and Decode geometries in
containers/build-sm90-megamoe.py.

Never imports CUDA Python libraries or launches a kernel. The installed package
is read-only; only the new header patch is applied to a temporary include tree.
"""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

BASE_SHA = "ff7a58cca00aa42a90386d12b0437b5d8caf35eb0ad082a8a1a0b4b74001a2a1"
HEADER = Path("deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="extracted deep_gemm package")
    parser.add_argument("--cuda", type=Path, required=True, help="extracted usr/local/cuda-13.0")
    parser.add_argument("--output", type=Path, required=True, help="JSON containing compiler logs and SASS comparison")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    package, cuda = args.package.resolve(), args.cuda.resolve()
    baseline = (package / "include/deep_gemm/impls/sm90_fp8_mega_moe.cuh").read_bytes()
    assert digest(baseline) == BASE_SHA
    records = {}
    configs = {"prefill": (128, 256, 512, 3, 112, True, True, False, False, 32), "decode32": (64, 128, 256, 8, 0, False, False, True, True, 32), "decode16": (64, 128, 256, 8, 0, False, False, True, True, 16)}
    with tempfile.TemporaryDirectory(prefix="megamoe-pingpong-cpu-") as directory:
        temp = Path(directory)
        header = temp / HEADER
        header.parent.mkdir(parents=True)
        header.write_bytes(baseline)
        # The image header is the 14 main patches; the series continues with the SwapAB guard, then the ping-pong patch.
        for name in ("sm90-swap-ab-geometry-guard.patch", "sm90-prefill-pingpong.patch"):
            subprocess.run(["git", "apply", "--check", str(repo / "patches/deep-gemm" / name)], cwd=temp, check=True)
            subprocess.run(["git", "apply", str(repo / "patches/deep-gemm" / name)], cwd=temp, check=True)
        assert digest(header.read_bytes()) == (repo / "patches/deep-gemm/header.sha256").read_text().strip()
        for role, config in configs.items():
            sass = []
            for variant in ("base", "candidate"):
                bm, bn, math, stages, registers, reuse, arrival, full_sync, swap, wave = config
                if role == "prefill" and variant == "candidate":
                    bn, math, stages, registers, reuse = 128, 256, 5, 168, False
                flags = ",".join(str(value).lower() for value in (reuse, arrival, full_sync, False, swap))
                source = temp / f"{role}-{variant}.cu"
                source.write_text(f'#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>\nextern "C" void* instantiate() {{ return reinterpret_cast<void*>(&deep_gemm::sm90_fp8_mega_moe_impl<4224,4096,1536,128,8,{wave},{bm},{bn},128,141312,2260992,{stages},64,64,{math},132,4,cute::numeric_limits<float>::infinity(),true,{registers},{flags}>); }}\n')
                includes = ["-I" + str(temp / "deep_gemm/include")] if variant == "candidate" else []
                command = [str(cuda / "bin/nvcc"), "--cubin", "--std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3", "--ptxas-options=-v,-warn-spills", "-gencode", "arch=compute_90a,code=sm_90a", *includes, "-I" + str(package / "include"), str(source), "-o", str(source.with_suffix(".cubin"))]
                output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
                dump = subprocess.check_output([str(cuda / "bin/cuobjdump"), "--dump-sass", str(source.with_suffix(".cubin"))], text=True)
                dump = dump[dump.index("code for sm_90a"):]
                sass.append(dump)
                records[f"{role}-{variant}"] = {"compiler_log": output, "registers": int(re.search(r"Used (\d+) registers", output)[1]), "sass_sha256": digest(dump.encode())}
                assert not re.search(r"C751[0-4]", output), output
                if variant == "candidate":
                    assert "0 bytes spill stores, 0 bytes spill loads" in output, output
                    assert records[f"{role}-{variant}"]["registers"] == 168
            if role.startswith("decode"):
                assert sass[0] == sass[1], role
                records[role + "-sass-identical"] = True
        result = {"cuda_version": subprocess.check_output([str(cuda / "bin/nvcc"), "--version"], text=True), "base_header_sha256": BASE_SHA, "header_sha256": digest(header.read_bytes()), "records": records}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: CPU compilation, no candidate spills or WGMMA warnings; both Decode SASS comparisons identical")


if __name__ == "__main__":
    main()
