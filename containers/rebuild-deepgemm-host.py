"""Rebuild only the v0.1.5 host extension; retain installed patched JIT headers.

Run with /opt/sglang/venv/bin/python inside the image builder, before the existing
build-sm90-megamoe.py step. The input checkout and its submodules are read-only.

The host patches are the files listed, in application order, in <patch-dir>/series
(patches/deep-gemm-host/series in the repository; each applies to the tree left by
the previous one). At least one is required: without a host patch the rebuild would
only swap the wheel's validated _C.so for an equivalent one built here.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from importlib.metadata import version
from pathlib import Path

SOURCE_COMMIT = "1c8138a59b3df5dac50b76b730834ed57982fdfb"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_series(patch_dir):
    names = (patch_dir / "series").read_text().split()
    assert names, f"{patch_dir}/series is empty: the host rebuild needs at least one host patch (or remove the rebuild step from the Dockerfile)"
    assert len(names) == len(set(names)), names
    return {name: (patch_dir / name).resolve() for name in names}


def build_host(source, temp):
    # Match v0.1.5 build_sgl_deep_gemm.sh: the sgl wheel loads a TVM-FFI
    # module, whereas setup.py builds the different upstream pybind11 ABI.
    import torch
    import tvm_ffi.cpp

    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    torch_dir = Path(torch.__file__).parent
    flags = ["-std=c++17", "-O3", "-fPIC", "-Wno-psabi", "-Wno-deprecated-declarations", f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}"]
    if int(os.environ.get("DG_JIT_USE_RUNTIME_API", "0")):
        flags.append("-DDG_JIT_USE_RUNTIME_API")
    includes = [cuda / "include", Path(sysconfig.get_path("include")), torch_dir / "include", torch_dir / "include/torch/csrc/api/include", source / "deep_gemm/include", source / "third-party/cutlass/include", source / "third-party/fmt/include"]
    if (cuda / "include/cccl").exists():
        includes.append(cuda / "include/cccl")
    links = [f"-L{cuda}/lib64", f"-L{torch_dir}/lib", "-lcudart", "-lnvrtc", "-lcublasLt", "-lcublas", "-ltorch", "-ltorch_cpu", "-lc10", "-lc10_cuda", "-ltorch_cuda"]
    os.environ.setdefault("TVM_FFI_CUDA_ARCH_LIST", "9.0")
    directory = temp / "build"
    directory.mkdir()
    return Path(tvm_ffi.cpp.build(name="_C", cpp_files=[str(source / "csrc/tvm_ffi_api.cpp")], extra_cflags=flags, extra_ldflags=links, extra_include_paths=[str(path) for path in includes], build_directory=str(directory)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="recursive sgl-project/DeepGEMM v0.1.5 checkout")
    parser.add_argument("--patch-dir", type=Path, default=Path("/opt/apac/deep-gemm-host"))
    parser.add_argument("--record", type=Path, default=Path("/opt/apac/deep-gemm-host-build.json"))
    args = parser.parse_args()
    patches = read_series(args.patch_dir)
    assert version("sgl-deep-gemm") == "0.1.5"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.source, text=True).strip()
    assert revision == SOURCE_COMMIT, revision
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=args.source, text=True).strip(), "Source must be clean"
    package = Path(importlib.util.find_spec("deep_gemm").origin).parent
    extensions = list(package.glob("_C*.so"))
    assert len(extensions) == 1 and extensions[0].name == "_C.so", extensions
    target = extensions[0]
    header = package / "include/deep_gemm/impls/sm90_fp8_mega_moe.cuh"
    header_hash = sha256(header)
    old_extension_hash = sha256(target)
    with tempfile.TemporaryDirectory(prefix="deepgemm-host-") as directory:
        temp = Path(directory)
        source = temp / "source"
        shutil.copytree(args.source, source, ignore=shutil.ignore_patterns(".git", "build", "dist", "__pycache__"))
        assert (source / "third-party/cutlass/include/cutlass/cutlass.h").is_file(), "Initialize submodules"
        assert (source / "third-party/fmt/include/fmt/format.h").is_file(), "Initialize submodules"
        for patch in patches.values():
            subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, check=True)
            subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)
        built = build_host(source, temp)
        assert built.is_file(), built
        shutil.copy2(built, target)
        # Load the real ABI in a fresh process without initializing a GPU.
        (temp / "libcuda.so.1").symlink_to("/usr/local/cuda/lib64/stubs/libcuda.so")
        subprocess.run([sys.executable, "-c", "import deep_gemm; assert callable(deep_gemm.fp8_mega_moe); assert callable(deep_gemm.mega_moe_pre_dispatch_sm90); assert callable(deep_gemm._C.fp8_gemm_nt_promotion_diagnostic)"], cwd=temp, env={**os.environ, "LD_LIBRARY_PATH": str(temp) + ":" + os.environ.get("LD_LIBRARY_PATH", "")}, check=True)
        assert sha256(header) == header_hash, "Host build must retain installed JIT header"
        result = {"source_commit": revision, "abi": "tvm-ffi", "entrypoint": "csrc/tvm_ffi_api.cpp", "patches": {name: sha256(patch) for name, patch in patches.items()}, "previous_extension_sha256": old_extension_hash, "extension_sha256": sha256(target), "retained_header_sha256": header_hash, "python": sys.version, "cuda_home": os.environ.get("CUDA_HOME", "/usr/local/cuda")}
        args.record.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
