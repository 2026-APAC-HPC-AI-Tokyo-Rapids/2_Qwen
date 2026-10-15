"""Stage pinned native sources on the CPU builder; no Torch/GPU dependency."""

import argparse
import hashlib
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from contract import (
    FMT_COMMIT,
    FMT_SOURCE_SHA256,
    NATIVE_API_SHA256,
    NATIVE_SOURCE_COMMIT,
    NATIVE_SOURCE_SHA256,
    ORIGINAL_HEADER_SHA256,
    SOURCES,
    verify,
)


def copy_tree(archive: Path, prefix: str, destination: Path) -> None:
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            if not member.name.startswith(prefix):
                continue
            relative = Path(member.name[len(prefix):])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe source archive path")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Unsupported source archive member")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as stream:
                target.write_bytes(stream.read())


def stage(source: Path, fmt: Path, destination: Path) -> None:
    verify(source, NATIVE_SOURCE_SHA256)
    verify(fmt, FMT_SOURCE_SHA256)
    if destination.exists():
        raise FileExistsError(destination)
    source_root = "DeepGEMM-" + NATIVE_SOURCE_COMMIT + "/"
    with tarfile.open(source) as archive:
        for relative, expected in {
            "sgl_deep_gemm/__init__.py": SOURCES["__init__.py"],
            "deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh": ORIGINAL_HEADER_SHA256,
        }.items():
            actual = hashlib.sha256(archive.extractfile(source_root + relative).read()).hexdigest()
            if actual != expected:
                raise ValueError("Upstream source differs from pinned wheel: " + relative)
    copy_tree(source, source_root + "csrc/", destination / "csrc")
    copy_tree(fmt, "fmt-" + FMT_COMMIT + "/include/", destination / "fmt")
    verify(destination / "csrc/tvm_ffi_api.cpp", NATIVE_API_SHA256)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="firmus-sm90-native-stage-") as temp:
        source = Path(temp) / "source.tar.gz"
        fmt = Path(temp) / "fmt.tar.gz"
        urllib.request.urlretrieve("https://codeload.github.com/sgl-project/DeepGEMM/tar.gz/" + NATIVE_SOURCE_COMMIT, source)
        urllib.request.urlretrieve("https://codeload.github.com/fmtlib/fmt/tar.gz/" + FMT_COMMIT, fmt)
        stage(source, fmt, args.destination.resolve())
