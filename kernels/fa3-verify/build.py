"""AOT-build the bounded SM90a verify companion without initializing CUDA."""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

PINS = {
    "fa3": ("sgl-project/sgl-attn", "f89bc2306632d1ec5f97b014dded4254f5b4a907", "418b5681584dc3efff496a1cab5ffd58d2728d89dcfe0ea16e6985d6ef35c68c"),
    "cutlass": ("NVIDIA/cutlass", "57e3cfb47a2d9e0d46eb6335c3dc411498efa198", "09237099a70f80bff1dc8bb80c843a674bb4fdcb46e43cc6993e711c5ca89bb5"),
}
IDS = ("control96", "p128s2", "p128split2", "p128split3")


def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def build(root):
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths

    started = time.monotonic()
    work = root / "build"
    work.mkdir()
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    sources = {}
    for name, (repository, commit, digest) in PINS.items():
        archive = work / (name + ".tar.gz")
        subprocess.run(["curl", "-fLsS", f"https://github.com/{repository}/archive/{commit}.tar.gz", "-o", str(archive)], check=True)
        assert sha(archive) == digest, (name, "source archive SHA mismatch")
        target = work / name
        target.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(target, filter="data")
        sources[name] = next(target.iterdir())
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(root / "fixed-specializations.patch")], cwd=sources["fa3"], check=True)
    cuda = Path("/usr/local/cuda")
    includes = [str(root), str(sources["fa3"] / "hopper"), str(sources["cutlass"] / "include"), str(sources["cutlass"] / "tools/util/include"), str(cuda / "include"), str(cuda / "include/cccl"), *include_paths()]
    common = ["-O3", "-std=c++17", "-fPIC", "-fvisibility=hidden", "-fvisibility-inlines-hidden", "-fno-gnu-unique", f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"]
    common += ["-Dflash=apac_fa3_private", "-DFlash_fwd_params=ApacFA3VerifyParams"]
    flags = ["-O3", "-std=c++17", "--use_fast_math", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", "-DCUTLASS_DEBUG_TRACE_LEVEL=0", "-DNDEBUG", "-Xptxas=-v,--warn-on-spills", "-Xcompiler=-fPIC,-fvisibility=hidden,-fvisibility-inlines-hidden,-fno-gnu-unique", *common[6:]]
    tasks = [(root / "wrapper.cpp", "wrapper", False)]
    tasks += [(root / (name + ".cu"), name, True) for name in ("control96", "p128s2", "p128split", "combine")]
    tasks += [(sources["fa3"] / "hopper/flash_prepare_scheduler.cu", "prepare", True)]

    def compile_one(task):
        source, name, is_cuda = task
        obj = work / (name + ".o")
        cmd = [str(cuda / "bin/nvcc") if is_cuda else "g++", *(flags if is_cuda else common), *["-I" + p for p in includes], "-c", str(source), "-o", str(obj)]
        begin = time.monotonic()
        with (logs / (name + ".log")).open("w") as log:
            print(json.dumps({"tu": name, "command": cmd}), flush=True)
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        elapsed = time.monotonic() - begin
        record = {"tu": name, "seconds": elapsed, "exit_code": result.returncode, "command": cmd}
        (logs / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if result.returncode:
            raise RuntimeError((logs / (name + ".log")).read_text())
        return record, obj

    # At most two heavy compilers on both the login node and image builder.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        compiled = list(pool.map(compile_one, tasks))
    binary = root / "libapac_fa3_verify.so"
    libs = library_paths(device_type="cuda")
    cmd = ["g++", "-shared", "-Wl,-Bsymbolic", *[str(obj) for _, obj in compiled], *["-L" + p for p in libs], *["-Wl,-rpath," + p for p in libs], "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda", "-lcudart", "-o", str(binary)]
    subprocess.run(cmd, check=True)
    torch.ops.load_library(str(binary))
    smem = torch.ops.apac_fa3_verify.resources()
    manifest = {"schema": "apac-fa3-verify/v1", "pins": {name: {"repository": repo, "commit": commit, "archive_sha256": digest} for name, (repo, commit, digest) in PINS.items()}, "architecture": "sm_90a", "cuda": subprocess.check_output([str(cuda / "bin/nvcc"), "--version"], text=True), "ptxas": subprocess.check_output([str(cuda / "bin/ptxas"), "--version"], text=True), "torch": torch.__version__, "cuda_flags": flags, "max_compile_jobs": 2, "translation_units": [r for r, _ in compiled], "specializations": [{"id": name, "shared_bytes": size} for name, size in zip(IDS, smem)], "source_sha256": {p.name: sha(p) for p in root.iterdir() if p.suffix in (".cu", ".cpp", ".h", ".py", ".patch")}, "binary_sha256": sha(binary), "binary_bytes": binary.stat().st_size, "elapsed_seconds": time.monotonic() - started, "gpu_executed": False}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run([str(cuda / "bin/cuobjdump"), "--dump-resource-usage", str(binary)], stdout=(logs / "resources.txt").open("w"), check=True)
    subprocess.run(["nm", "-D", "--defined-only", str(binary)], stdout=(logs / "symbols.txt").open("w"), check=True)
    print(json.dumps(manifest), flush=True)
    shutil.rmtree(work)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/opt/apac/fa3-verify"))
    build(parser.parse_args().root.resolve())
