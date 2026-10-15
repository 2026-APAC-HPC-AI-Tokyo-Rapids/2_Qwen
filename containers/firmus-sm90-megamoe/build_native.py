"""Rebuild only the private host FFI library against the actual image Torch."""

import argparse
import hashlib
import json
import os
import shutil
import sysconfig
import tempfile
from importlib.metadata import version
from pathlib import Path

from contract import (
    FMT_COMMIT,
    FMT_SOURCE_SHA256,
    NATIVE_API_SHA256,
    NATIVE_LIBRARY_SHA256,
    NATIVE_SOURCE_COMMIT,
    NATIVE_SOURCE_SHA256,
    verify,
    verify_package,
)

# Exercise from_blob/view/DLPack in the same native binary without CUDA.
CPU_PROBE = r"""
Tensor apac_cpu_tensor_bridge(TensorView input) {
    TVM_FFI_ICHECK(input.device().device_type == kDLCPU);
    auto dtype = dl_dtype_to_torch(input.dtype());
    auto sizes = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto strides = std::vector<int64_t>(input.strides().begin(), input.strides().end());
    auto ptr = static_cast<char*>(input.data_ptr()) + input.byte_offset();
    auto view = torch::from_blob(ptr, sizes, strides, torch::TensorOptions().dtype(dtype).device(torch::kCPU).requires_grad(false));
    return Tensor::FromDLPack(at::toDLPack(view));
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(apac_cpu_tensor_bridge, apac_cpu_tensor_bridge);
"""


def check_cpu_bridge(module, torch) -> None:
    for dtype in [torch.int32, torch.float32]:
        owner = torch.arange(24, dtype=dtype).reshape(4, 6)
        for source in [owner, owner[1:, ::2], owner.transpose(0, 1)]:
            result = torch.from_dlpack(module.apac_cpu_tensor_bridge(source))
            if result.shape != source.shape or result.stride() != source.stride():
                raise ValueError("Native CPU bridge changed tensor layout")
            if result.dtype != dtype or result.data_ptr() != source.data_ptr() or not torch.equal(result, source):
                raise ValueError("Native CPU bridge broke aliasing or values")
            result[0, 0] = -17
            if source[0, 0].item() != -17:
                raise ValueError("Native CPU bridge copied the input")
    if torch.cuda.is_initialized():
        raise ValueError("Native CPU bridge must not initialize CUDA")


def build(source: Path, package: Path) -> None:
    import torch
    import tvm_ffi
    import tvm_ffi.cpp

    verify_package(package)
    verify(source / "csrc/tvm_ffi_api.cpp", NATIVE_API_SHA256)
    if torch.__version__ != "2.13.0+cu130" or version("apache-tvm-ffi") != "0.1.11":
        raise ValueError("Unexpected native build runtime")
    if torch.cuda.is_initialized():
        raise ValueError("Native build must not initialize CUDA")
    cuda = Path("/usr/local/cuda")
    torch_root = Path(torch.__file__).parent
    nvidia_headers = torch_root.parent / "nvidia/cu13/include"
    for name in ["cusparse.h", "cusolverDn.h", "cublasLt.h"]:
        if not (nvidia_headers / name).is_file():
            raise FileNotFoundError(nvidia_headers / name)
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "9.0a"
    with tempfile.TemporaryDirectory(prefix="firmus-sm90-native-build-") as temp:
        root = Path(temp)
        (root / "build").mkdir()
        api = root / "tvm_ffi_api.cpp"
        api.write_text((source / "csrc/tvm_ffi_api.cpp").read_text() + CPU_PROBE)
        include = [cuda / "include", cuda / "include/cccl", nvidia_headers, Path(sysconfig.get_path("include")), torch_root / "include", torch_root / "include/torch/csrc/api/include", package / "include", source / "csrc", source / "fmt"]
        flags = ["-std=c++17", "-O3", "-fPIC", "-Wno-psabi", "-Wno-deprecated-declarations", "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch.compiled_with_cxx11_abi()))]
        links = ["-L" + str(cuda / "lib64"), "-L" + str(torch_root / "lib"), "-lcudart", "-lnvrtc", "-lcublasLt", "-lcublas", "-ltorch", "-ltorch_cpu", "-lc10", "-lc10_cuda", "-ltorch_cuda"]
        library = tvm_ffi.cpp.build(name="_C", cpp_files=[str(api)], extra_cflags=flags, extra_ldflags=links, extra_include_paths=[str(p) for p in include], build_directory=str(root / "build"))
        module = tvm_ffi.load_module(str(library))
        check_cpu_bridge(module, torch)
        shutil.copyfile(library, package / "_C.so")
    report = {
        "schema": "apac-sm90-native-build/v1",
        "source_commit": NATIVE_SOURCE_COMMIT,
        "source_archive_sha256": NATIVE_SOURCE_SHA256,
        "fmt_commit": FMT_COMMIT,
        "fmt_archive_sha256": FMT_SOURCE_SHA256,
        "original_native_sha256": NATIVE_LIBRARY_SHA256,
        "native_sha256": hashlib.sha256((package / "_C.so").read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "tvm_ffi": version("apache-tvm-ffi"),
        "cxx11_abi": torch.compiled_with_cxx11_abi(),
        "nvidia_headers": str(nvidia_headers),
        "cpu_bridge_probe": True,
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    (package / "native-build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    build(args.source.resolve(), args.package.resolve())
